"""Viewer symmetry jobs; evidence shares the feature identity contract."""
import asyncio
import base64
import copy
from pathlib import Path
import secrets
import tempfile
import threading
import time

from . import bounded, checks, compare, scan, symmetry


def _reference(server, message, target):
    uploaded = message.get("cloud")
    if uploaded is not None:
        suffix = scan.reference_suffix(Path(uploaded.get("filename", "")))
        if suffix not in (".ply", ".ply.gz"):
            raise ValueError("choose a PLY or PLY.GZ mesh or point cloud")
        units = uploaded.get("units")
        if units not in scan.UNITS:
            raise ValueError("choose the point cloud's source units")
        body = base64.b64decode(uploaded.get("data", ""), validate=True)
        if not body or len(body) > 48 * 1024 * 1024:
            raise ValueError("point cloud uploads must be nonempty and no larger than 48 MiB")
        with tempfile.TemporaryDirectory(prefix="nurb-symmetry-upload-") as directory:
            temporary = Path(directory) / ("reference" + suffix); temporary.write_bytes(body)
            symmetry.load_reference(temporary, units)
            digest = symmetry.reference_identity(temporary, units)
        source = server.root / "scans" / ("symmetry-" + digest[:16] + suffix)
        source.parent.mkdir(exist_ok=True)
        if not source.exists():
            source.write_bytes(body)
    elif message.get("reference_file"):
        source = (server.root / message["reference_file"]).resolve()
        if not source.is_relative_to(server.root / "scans"):
            raise ValueError("choose a point cloud uploaded to this project's scans directory")
        units = message.get("units")
    else:
        if not target or not target.get("file"):
            raise ValueError("attach a reference mesh or choose a point cloud in the symmetry panel")
        source = Path(target["file"])
        if not source.is_absolute():
            source = server.root / source
        units = target.get("units")
    return source, units


async def handle(server, path, message, client):
    name = path.stem
    jobs = server.symmetry_jobs
    if message["type"] == "target_symmetry_cancel":
        job = jobs.get(name)
        if job and message.get("request_id") == job["request_id"] and message.get("token") == job["token"]:
            job["stop"].set()
        return
    entry = server.state.get(name) or {}
    token = entry.get("token")
    request_id = secrets.token_hex(8)
    base = {"type": "target_symmetry", "name": name, "token": token, "request_id": request_id}
    try:
        if name in jobs or len(jobs) >= 2:
            raise ValueError("symmetry verification is already running; wait or cancel it before starting another")
        if entry.get("shape") is None:
            raise ValueError("build a valid CAD model before checking symmetry")
        options = symmetry.Options(**message.get("options", {}))
        deadline = time.monotonic() + options.timeout_s
        stopped = threading.Event()
        jobs[name] = {"request_id": request_id, "token": token, "stop": stopped}
        previous=entry.get("symmetry") or {}
        if previous.get("status")=="measured": entry["last_symmetry"]=previous
        entry["symmetry"] = {**base, "status": "running", "phase": "Fitting the reference plane and checking trimmed CAD"}
        await server.reply(client, entry["symmetry"])
        jobs[name]["task"] = asyncio.create_task(_job(server, path, message, options, base, entry, stopped, client, deadline))
    except (TypeError, ValueError) as exc:
        await server.reply(client, {**base, "status": "unknown", "error": str(exc)})


async def _job(server, path, message, options, base, entry, stopped, client, deadline=None):
    result={**base,"status":"running"}
    name=path.stem
    target=copy.deepcopy({key:(entry.get("target") or {}).get(key) for key in ("file","units","stamp","transform","content_id","regions")})
    overrides=copy.deepcopy(server.overrides.get(name) or {})
    configuration={"name":entry.get("variant") or "default","parameters":{p["name"]:p["value"] for p in entry.get("params",[])}}
    transform=target.get("transform") or compare.IDENTITY
    loop=asyncio.get_running_loop()
    async def publish_phase(resources):
        if server.state.get(name) is entry and not stopped.is_set() and result.get("status")=="running":
            result.update(phase=resources["phase"],resources=resources)
            entry["symmetry"]=dict(result)
            await server.reply(client,dict(result))
    def progress(resources):
        loop.call_soon_threadsafe(lambda:asyncio.create_task(publish_phase(resources)))
    def analyze():
        bounded.phase("Validating symmetry snapshot inputs")
        snapshot=server._source_snapshot(path)
        built=server._build_inputs(server._build_sources(path,entry["shape"],snapshot),snapshot)
        if server.prints.get(name)!=(built,repr(sorted(overrides.items())),entry.get("shape_id")):
            raise bounded.WorkError("Model inputs changed; wait for the rebuild before checking symmetry.","stale")
        declared = compare.setting(checks.settings(path)) or {}
        if (any(declared.get(key) != target.get(key) for key in ("file", "units"))
                or (declared.get("regions") or []) != (target.get("regions") or [])
                or (declared.get("transform") is not None and declared["transform"] != target.get("transform"))):
            raise bounded.WorkError("Reference or feature settings changed; wait for rebuilding before checking symmetry.", "stale")
        bounded.phase("Loading symmetry reference snapshot")
        source,units=_reference(server,message,target)
        revision=symmetry.source_revision(path)
        if not message.get("cloud") and not message.get("reference_file"):
            if server._target_stamp(target["file"],target.get("units"))!=target.get("stamp"):
                raise bounded.WorkError("The attached reference changed; wait for its refresh and run symmetry again.","stale")
        reference,unit,reference_id=symmetry.reference_snapshot(source,units)
        reference.apply_transform(compare._transform(transform))
        bounded.phase("Copying and identifying finished CAD")
        shape=copy.deepcopy(entry["shape"])
        from .symmetry_categories import landmarks
        locations = landmarks(target.get("regions") or [], transform)
        contract=symmetry.identity(shape,reference_id,transform,configuration,revision,options,locations)
        relative=source.relative_to(server.root).as_posix() if source.is_relative_to(server.root) else source.name
        metadata={"source_revision":revision,"configuration":configuration,"reference_file":relative,"reference_units":unit,
                  "reference_kind":"independent" if message.get("cloud") or message.get("reference_file") else "attached","alignment":list(transform)}
        metadata["feature_records"] = [[r["name"], r["feature"]] for r in target.get("regions") or [] if "feature" in r]
        source_files=bounded.file_identity(set(server._build_sources(path,entry["shape"],snapshot))|{path,path.with_suffix('.md'),source})
        return symmetry.prepare(shape,reference,options,contract,metadata,source_files,landmarks=locations)

    try:
        measured=await bounded.async_run(analyze,timeout_s=options.timeout_s,memory_limit_mb=options.memory_limit_mb,
                                         stop=stopped.is_set,progress=progress,deadline=deadline)
        current=server.state.get(name) or {}
        current_target=current.get("target") or {}
        current_configuration = {"name":current.get("variant") or "default","parameters":{p["name"]:p["value"] for p in current.get("params",[])}}
        if (current is not entry or current.get("token")!=base["token"] or server.overrides.get(name,{})!=overrides
                or current_configuration != configuration
                or any(current_target.get(k)!=target.get(k) for k in ("file","units","stamp","transform","content_id","regions"))):
            result.update(status="stale",reason="stale",error="The model or reference changed during symmetry verification; run it again.")
        else: result.update(measured,phase="Complete")
    except asyncio.CancelledError:
        stopped.set();result.update(status="cancelled",reason="cancelled",error="Symmetry cancelled; child stopped and reaped.");raise
    except Exception as exc:
        reason=getattr(exc,"reason","worker_failure")
        result.update(status=reason if reason in ("stale","cancelled") else "unknown",reason=reason,error=str(exc),resources=getattr(exc,"resources",{}))
    finally:
        if server.state.get(name) is entry: entry["symmetry"]=result
        server.symmetry_jobs.pop(name,None)
    await server.reply(client,result)


def changed(server, paths):
    """The watcher also expires independent clouds and non-Python model inputs."""
    if not any((entry.get("symmetry") or {}).get("status") == "measured" for entry in list(server.state.values())):
        return
    server.loop.call_soon_threadsafe(lambda: asyncio.create_task(invalidate(server, paths)))


async def invalidate(server, paths):
    for name, entry in list(server.state.items()):
        report = entry.get("symmetry") or {}
        if report.get("status") != "measured":
            continue
        source = (report.get("reference_file") if report.get("reference_kind") == "independent"
                  else (entry.get("target") or {}).get("file"))
        source = Path(source) if source else None
        if source is not None and not source.is_absolute():
            source = server.root / source
        inputs_changed = False
        try:
            if source and source.resolve() in paths:
                current = await asyncio.to_thread(symmetry.reference_identity, source, report["reference_units"])
                inputs_changed = current != report["identity"]["reference"]
            relevant = any(path.is_relative_to(server.root) and path.suffix.lower() in (".py", ".md", ".toml", ".json", ".step", ".stp", ".brep")
                           and not any(piece.startswith(".") or piece in ("build", "inspections", "__pycache__") for piece in path.relative_to(server.root).parts) for path in paths)
            if relevant:
                revision = await asyncio.to_thread(symmetry.source_revision, server.root / "parts" / f"{name}.py")
                inputs_changed |= revision != report["source_revision"]
        except OSError:
            inputs_changed = True
        if inputs_changed and server.state.get(name) is entry and entry.get("symmetry") is report:
            stale = {key: report[key] for key in ("type", "name", "token", "request_id", "identity", "options") if key in report}
            stale.update(status="stale", error="The reference or model inputs changed. Run symmetry verification again.")
            entry["symmetry"] = stale
            await server.send(stale)
