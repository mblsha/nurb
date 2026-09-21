"""Viewer symmetry jobs; evidence shares the feature identity contract."""
import asyncio
import base64
import copy
from pathlib import Path
import secrets
import tempfile
import threading

from . import compare, scan, symmetry


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
        stopped = threading.Event()
        jobs[name] = {"request_id": request_id, "token": token, "stop": stopped}
        entry["symmetry"] = {**base, "status": "running", "phase": "Fitting the reference plane and checking trimmed CAD"}
        await server.reply(client, entry["symmetry"])
        jobs[name]["task"] = asyncio.create_task(_job(server, path, message, options, base, entry, stopped, client))
    except (TypeError, ValueError) as exc:
        await server.reply(client, {**base, "status": "unknown", "error": str(exc)})


async def _job(server, path, message, options, base, entry, stopped, client):
    result = {**base, "status": "unknown"}
    name = path.stem
    try:
        async with server.building:
            if server.state.get(name) is not entry:
                raise ValueError("the model changed before symmetry verification started; try again")
            snapshot = server._source_snapshot(path)
            built = server._build_inputs(server._build_sources(path, entry["shape"], snapshot), snapshot)
            overrides = copy.deepcopy(server.overrides.get(name) or {})
            if server.prints.get(name) != (built, repr(sorted(overrides.items())), entry.get("shape_id")):
                raise ValueError("model inputs changed; wait for the rebuild before checking symmetry")
            shape = copy.deepcopy(entry["shape"])
            target = copy.deepcopy(entry.get("target") or {})
            configuration = {"name": entry.get("variant") or "default", "parameters": {p["name"]: p["value"] for p in entry.get("params", [])}}
            transform = target.get("transform") or compare.IDENTITY
            source, units = await asyncio.to_thread(_reference, server, message, target)
            revision = symmetry.source_revision(path)
        def analyze():
            if not message.get("cloud") and not message.get("reference_file"):
                if server._target_mesh(target["file"], target.get("units"))["stamp"] != target.get("stamp"):
                    raise ValueError("the attached reference changed; wait for its refresh and run symmetry again")
            reference, unit, reference_id = symmetry.reference_snapshot(source, units)
            reference.apply_transform(compare._transform(transform))
            contract = symmetry.identity(shape, reference_id, transform, configuration, revision, options)
            measured = symmetry.run(shape, reference, options, contract, stopped.is_set)
            return measured, unit, reference_id
        measured, unit, reference_id = await asyncio.to_thread(analyze)
        current = server.state.get(name) or {}
        current_target = current.get("target") or {}
        current_configuration = {"name": current.get("variant") or "default", "parameters": {p["name"]: p["value"] for p in current.get("params", [])}}
        if (current is not entry or current.get("token") != base["token"] or stopped.is_set()
                or current_configuration != configuration or server.overrides.get(name, {}) != overrides
                or (current_target.get("transform") or compare.IDENTITY) != transform
                or any(current_target.get(k) != target.get(k) for k in ("file", "units", "content_id", "stamp"))
                or symmetry.source_revision(path) != revision or symmetry.reference_identity(source, unit) != reference_id):
            result.update(status="cancelled" if stopped.is_set() else "stale", error="Symmetry evidence was cancelled or its inputs changed; run it again.")
        else:
            relative = source.relative_to(server.root).as_posix() if source.is_relative_to(server.root) else source.name
            result.update(measured, source_revision=revision, reference_kind="independent" if message.get("cloud") or message.get("reference_file") else "attached", configuration=configuration, reference_file=relative, reference_units=unit, alignment=list(transform))
    except asyncio.CancelledError:
        stopped.set(); raise
    except Exception as exc:
        result.update(status="cancelled" if stopped.is_set() else "unknown", error=str(exc))
    finally:
        if server.state.get(name) is entry:
            entry["symmetry"] = result
        server.symmetry_jobs.pop(name, None)
    await server.reply(client, result)


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
