"""One deadline owns snapshot and fresh analysis children, including their cleanup."""
import asyncio
import math
import os
from pathlib import Path
import threading
import time

import pytest
from build123d import Box

from nurb import bounded, meshing


def reaped(resources):
    for pid in resources['child_pids']:
        with pytest.raises(ChildProcessError): os.waitpid(pid,os.WNOHANG)
        with pytest.raises(ProcessLookupError): os.kill(pid,0)


def test_slow_brep_snapshot_is_inside_the_deadline_and_reaped(monkeypatch):
    import build123d
    monkeypatch.setattr(build123d,'export_brep',lambda *args:time.sleep(10))
    start=time.monotonic()
    with pytest.raises(bounded.WorkError) as error:
        meshing.verification_mesh(Box(1,1,1),meshing.VerificationPolicy(.01,timeout_s=.2))
    assert error.value.reason=='timeout'
    assert 'Serializing CAD snapshot' in error.value.resources['phase']
    assert time.monotonic()-start<1.5
    reaped(error.value.resources)


def test_snapshot_and_analysis_share_one_deadline(tmp_path,monkeypatch):
    module=tmp_path/'bounded_fixture.py'
    module.write_text('def done():\n    return {"done":True}\n')
    monkeypatch.setenv('PYTHONPATH',str(tmp_path)+os.pathsep+os.environ.get('PYTHONPATH',''))
    def prepare():
        time.sleep(.15)
        return bounded.stage('bounded_fixture','done')
    start=time.monotonic()
    with pytest.raises(bounded.WorkError) as error:bounded.run(prepare,timeout_s=.25)
    assert error.value.reason=='timeout'
    assert len(error.value.resources['child_pids'])==2
    assert time.monotonic()-start<1.5
    reaped(error.value.resources)


def test_actual_distance_query_is_terminated_in_a_fresh_worker(tmp_path,monkeypatch):
    module=tmp_path/'bounded_fixture.py'
    module.write_text('''import time

def slow_distance():
    import trimesh
    from build123d import Box
    from nurb import compare
    def slow(*args,**kwargs):time.sleep(30)
    compare._to_surface=slow
    mesh=trimesh.creation.box(extents=[1,1,1])
    return compare.against(Box(1,1,1),mesh,part_mesh=mesh)
''')
    monkeypatch.setenv('PYTHONPATH',str(tmp_path)+os.pathsep+os.environ.get('PYTHONPATH',''))
    phases=[]
    with pytest.raises(bounded.WorkError) as error:
        bounded.run(lambda:bounded.stage('bounded_fixture','slow_distance'),timeout_s=8,progress=lambda value:phases.append(value['phase']))
    assert error.value.reason=='timeout'
    assert 'CAD to reference distances' in phases
    reaped(error.value.resources)


def test_actual_child_rss_growth_is_stopped_and_reaped():
    baseline=bounded.run(lambda:(time.sleep(.15) or {}),timeout_s=2)
    limit=max(64,math.ceil(baseline['resources']['peak_rss_mb'])+64)
    def allocate():
        bounded.phase('Allocating measurement arrays')
        data=bytearray(192*1024*1024)
        time.sleep(5)
        return {'size':len(data)}
    with pytest.raises(bounded.WorkError) as error:bounded.run(allocate,timeout_s=3,memory_limit_mb=limit)
    assert error.value.reason=='memory_limit'
    assert error.value.resources['peak_rss_mb']>limit
    reaped(error.value.resources)


def test_cancel_during_snapshot_reaps_worker_and_async_task_cancellation_does_too():
    stopped=threading.Event();seen=[]
    def snapshot():bounded.phase('Slow snapshot');time.sleep(30)
    def progress(value):
        seen.append(value)
        if value['phase']=='Slow snapshot':stopped.set()
    with pytest.raises(bounded.WorkCancelled) as error:bounded.run(snapshot,timeout_s=3,stop=stopped.is_set,progress=progress)
    reaped(error.value.resources)
    async def scenario():
        ready=asyncio.Event();loop=asyncio.get_running_loop();children=[]
        def report(value):children.append(value);loop.call_soon_threadsafe(ready.set)
        task=asyncio.create_task(bounded.async_run(snapshot,timeout_s=3,progress=report))
        await asyncio.wait_for(ready.wait(),2);task.cancel()
        with pytest.raises(asyncio.CancelledError):await task
        reaped(children[-1])
    asyncio.run(scenario())


def test_unsupported_platform_does_not_claim_a_memory_budget(monkeypatch):
    monkeypatch.setattr(bounded,'supported',lambda:False)
    assert bounded.memory_policy(128)['enforcement']=='unavailable'
    with pytest.raises(bounded.WorkError) as error:bounded.run(lambda:{},timeout_s=1)
    assert error.value.reason=='unsupported_platform'
