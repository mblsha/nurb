"""Disposable POSIX jobs bound snapshot, kernel work and measurement together."""
import errno
import json
import math
import os
from pathlib import Path
import pickle
import select
import signal
import subprocess
import sys
import tempfile
import time

DEFAULT_MEMORY_MB = 2048
POLL_SECONDS = .05
MAX_RESULT_BYTES = 128 * 1024 * 1024
_worker_pid = None
_worker_deadline = None
_phase_fd = None
_workspace = None
_fresh = False


class WorkError(ValueError):
    def __init__(self, message, reason='worker_failure', resources=None):
        super().__init__(message)
        self.reason = reason
        self.resources = resources or {}


class WorkCancelled(WorkError):
    def __init__(self, message='Verification cancelled; no result was produced.', **kwargs):
        super().__init__(message, 'cancelled', **kwargs)


def supported():
    return hasattr(os, 'fork') and sys.platform in ('linux', 'darwin')


def memory_policy(limit_mb):
    if isinstance(limit_mb, bool) or not isinstance(limit_mb, int) or not 32 <= limit_mb <= 32768:
        raise ValueError('child RSS limit must be an integer from 32 to 32768 MiB')
    return {'limit_mb': limit_mb, 'enforcement': 'child RSS monitor' if supported() else 'unavailable',
            'supported': supported(), 'interval_ms': round(POLL_SECONDS * 1000),
            'limitation': 'RSS is sampled, including inherited resident pages; brief allocation overshoot is possible between samples. Triangle limits bound accepted output, not memory.'}


def workspace():
    return Path(_workspace)


def fresh():
    return active() and _fresh


def stage(module, function, **arguments):
    return {'_nurb_next_stage': {'module':module,'function':function,'arguments':arguments}}


def file_stamp(path):
    try:
        value=Path(path).stat()
        return [value.st_mtime_ns,value.st_ctime_ns,value.st_size]
    except FileNotFoundError: return None


def file_identity(paths):
    import hashlib
    result={}
    for path in paths:
        stamp=file_stamp(path)
        result[str(path)]={'stat':stamp,'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest() if stamp is not None else None}
    return result


def verify_files(record):
    phase('Checking final input identities')
    if file_identity([Path(path) for path in record]) != record:
        raise WorkError('Verification inputs changed during the job; run it again.', 'stale')


def active():
    return _worker_pid == os.getpid()


def check():
    if active() and time.monotonic() >= _worker_deadline:
        raise WorkError('Verification unknown: end-to-end time budget exceeded.', 'timeout')


def phase(name):
    check()
    if active() and _phase_fd is not None:
        try: os.write(_phase_fd, (json.dumps({'phase': str(name)[:180]}) + '\n').encode())
        except (BlockingIOError, BrokenPipeError): pass


def rss_bytes(pid):
    if sys.platform == 'linux':
        try: return int(Path(f'/proc/{pid}/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
        except FileNotFoundError: return 0
    if sys.platform == 'darwin':
        # libproc's task info reports physical resident bytes without spawning ps.
        import ctypes
        class TaskInfo(ctypes.Structure):
            _fields_ = [('virtual', ctypes.c_uint64), ('resident', ctypes.c_uint64),
                        ('rest', ctypes.c_byte * 80)]
        data = TaskInfo()
        lib = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True)
        count = lib.proc_pidinfo(pid, 4, 0, ctypes.byref(data), ctypes.sizeof(data))
        if count >= 16: return int(data.resident)
        if ctypes.get_errno() in (errno.ESRCH, errno.ENOENT): return 0
        raise OSError('macOS could not read the verification child RSS')
    raise OSError('child RSS monitoring is unavailable on this platform')


def _reap(pid, exited, process=None):
    if not exited:
        try: os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            try: os.kill(pid, signal.SIGKILL)
            except ProcessLookupError: pass
        except PermissionError:
            os.kill(pid, signal.SIGKILL)
        if process is not None: process.wait()
        else:
            try: os.waitpid(pid, 0)
            except ChildProcessError: pass


def run(function, *, timeout_s=30, memory_limit_mb=DEFAULT_MEMORY_MB, stop=None, progress=None, deadline=None):
    """Run all expensive preparation in the owned child, never before its deadline.

    A short-lived fork inherits the immutable live shape for bounded snapshot copying and serialization. It is reaped before a fresh interpreter performs geometry queries, avoiding inherited OCCT worker-pool locks. One deadline and RSS monitor cover both children; nested verification uses the same fresh worker.
    """
    if not isinstance(timeout_s, (int, float)) or isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or not 0 < timeout_s <= 120:
        raise ValueError('verification time budget must be greater than zero and at most 120 seconds')
    policy = memory_policy(memory_limit_mb)
    if active():
        check()
        return function()
    started = time.monotonic()
    deadline = min(deadline, started + timeout_s) if deadline is not None else started + timeout_s
    if stop and stop(): raise WorkCancelled()
    if not supported():
        raise WorkError('Verification unknown: bounded snapshot isolation and RSS monitoring require macOS or Linux on this build; no unbounded fallback was started.', 'unsupported_platform', {'memory': policy})
    resources = {'memory': policy, 'peak_rss_mb': None, 'timeout_s': timeout_s, 'deadline_scope': 'snapshot preparation through final evidence', 'phase': 'Preparing snapshot'}
    with tempfile.TemporaryDirectory(prefix='nurb-bounded-') as temporary:
        result_file = Path(temporary) / 'result.pickle'
        read_fd, write_fd = os.pipe()
        os.set_blocking(read_fd, False); os.set_blocking(write_fd, False)
        try:
            pid = os.fork()
        except OSError:
            os.close(read_fd); os.close(write_fd)
            raise
        if pid == 0:
            global _worker_pid, _worker_deadline, _phase_fd, _workspace
            _worker_pid, _worker_deadline, _phase_fd, _workspace = os.getpid(), deadline, write_fd, temporary
            os.close(read_fd)
            try:
                os.setsid()
                phase('Preparing snapshot')
                result = function()
                phase('Serializing evidence')
                with result_file.open('wb') as stream: pickle.dump({'result': result}, stream, protocol=5)
                if result_file.stat().st_size > MAX_RESULT_BYTES:
                    raise WorkError('Verification evidence exceeds the 128 MiB transfer limit; reduce mesh or sample budgets.', 'result_limit')
                check()
            except BaseException as exc:
                with result_file.open('wb') as stream:
                    pickle.dump({'error': str(exc)[:1000], 'reason': getattr(exc, 'reason', 'worker_failure')}, stream)
            finally:
                os.close(write_fd)
                os._exit(0)
        os.close(write_fd)
        resources['pid'] = pid; resources['child_pids']=[pid]
        process=None; exited = False; buffer = b''
        try:
            while True:
                if stop and stop(): raise WorkCancelled(resources=resources)
                if time.monotonic() >= deadline:
                    raise WorkError(f"Verification unknown: end-to-end time budget exceeded {timeout_s:g} seconds during {resources['phase']}. Increase the time budget or reduce the selected work.", 'timeout', resources)
                try: memory = rss_bytes(pid)
                except OSError as exc:
                    raise WorkError(f'Verification unknown: memory monitoring failed: {exc}; the child was stopped.', 'memory_unavailable', resources) from exc
                resources['peak_rss_mb'] = max(resources['peak_rss_mb'] or 0, memory / 1024 ** 2)
                if memory > memory_limit_mb * 1024 ** 2:
                    raise WorkError(f'Verification unknown: child RSS exceeded {memory_limit_mb} MiB; reduce the selected work or increase the memory limit.', 'memory_limit', resources)
                try: buffer += os.read(read_fd, 65536)
                except BlockingIOError: pass
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    resources.update(json.loads(line))
                    if progress: progress(dict(resources))
                if process is None:
                    found,status=os.waitpid(pid,os.WNOHANG)
                else:
                    status=process.poll();found=status is not None
                if found:
                    exited=True
                    if status: raise WorkError('Verification child exited unexpectedly; no measurements were published.','worker_failure',resources)
                    if not result_file.is_file() or result_file.stat().st_size>MAX_RESULT_BYTES:
                        raise WorkError('Verification produced no bounded evidence file.','result_limit',resources)
                    with result_file.open('rb') as stream: payload=pickle.load(stream)
                    next_stage=(payload.get('result') or {}).get('_nurb_next_stage') if isinstance(payload.get('result'),dict) else None
                    if next_stage:
                        result_file.replace(Path(temporary)/'job.pickle')
                        os.close(read_fd); read_fd,write_fd=os.pipe()
                        os.set_blocking(read_fd,False);os.set_blocking(write_fd,False)
                        env=dict(os.environ);env['PYTHONPATH']=str(Path(__file__).resolve().parent.parent)+os.pathsep+env.get('PYTHONPATH','')
                        try:
                            process=subprocess.Popen([sys.executable,'-m','nurb.bounded',temporary,str(write_fd),str(deadline)],
                                pass_fds=(write_fd,),start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,env=env)
                        finally:
                            os.close(write_fd)
                        pid=process.pid;exited=False;buffer=b''
                        resources['phase']='Starting isolated geometry worker'
                        resources['pid']=pid;resources['child_pids'].append(pid)
                        continue
                    break
                select.select([read_fd], [], [], min(POLL_SECONDS, max(0, deadline - time.monotonic())))
            if not result_file.is_file() or result_file.stat().st_size > MAX_RESULT_BYTES:
                raise WorkError('Verification produced no bounded evidence file.', 'result_limit', resources)
            with result_file.open('rb') as stream: payload = pickle.load(stream)
            if stop and stop(): raise WorkCancelled(resources=resources)
            if time.monotonic() >= deadline:
                raise WorkError('Verification unknown: end-to-end time budget exceeded while receiving evidence.', 'timeout', resources)
            if 'error' in payload:
                if payload['reason'] == 'cancelled': raise WorkCancelled(payload['error'], resources=resources)
                raise WorkError(payload['error'], payload['reason'], resources)
            result = payload['result']
            resources.update(elapsed_s=time.monotonic() - started, phase='Complete')
            if isinstance(result, dict): result['resources'] = resources
            return result
        finally:
            _reap(pid, exited, process)
            os.close(read_fd)


def capture_command(function, args):
    """Keep CLI stdout deterministic when the entire command runs in a child."""
    import contextlib
    import io
    out, error = io.StringIO(), io.StringIO()
    code = None
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(error):
        try: function(args)
        except SystemExit as exc: code = exc.code
    return {'stdout': out.getvalue(), 'stderr': error.getvalue(), 'exit_code': code}


async def async_run(function, **options):
    """Cancelling the coroutine also waits until its owned worker is reaped."""
    import asyncio
    import threading
    stopped = threading.Event()
    requested = options.pop('stop', None)
    task = asyncio.create_task(asyncio.to_thread(run, function, stop=lambda: stopped.is_set() or bool(requested and requested()), **options))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        stopped.set()
        try: await task
        except WorkError: pass
        raise


def _entrypoint(root, descriptor, deadline):
    import importlib
    global _worker_pid, _worker_deadline, _phase_fd, _workspace, _fresh
    _worker_pid,_worker_deadline,_phase_fd,_workspace,_fresh=os.getpid(),deadline,descriptor,root,True
    destination=Path(root)/'result.pickle'
    try:
        phase('Starting isolated geometry worker')
        with (Path(root)/'job.pickle').open('rb') as stream: job=pickle.load(stream)['result']['_nurb_next_stage']
        result=getattr(importlib.import_module(job['module']),job['function'])(**job['arguments'])
        phase('Serializing evidence')
        payload={'result':result}
    except BaseException as exc:
        payload={'error':str(exc)[:1000],'reason':getattr(exc,'reason','worker_failure')}
    with destination.open('wb') as stream: pickle.dump(payload,stream,protocol=5)


if __name__=='__main__':
    # -m executes this module as __main__, but the analysis modules import nurb.bounded.
    sys.modules['nurb.bounded']=sys.modules[__name__]
    import nurb
    nurb.bounded=sys.modules[__name__]
    _entrypoint(sys.argv[1],int(sys.argv[2]),float(sys.argv[3]))
