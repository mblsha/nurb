"""Portable, freshness-aware summaries of specialized feature validators."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import pathlib
import re
import sys


KIND = "nurb_feature_validation"
SCHEMA_VERSION = 1
_FEATURE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}")


def _portable_path(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a portable path inside the project")
    path = pathlib.PurePosixPath(value)
    if path.is_absolute() or pathlib.PureWindowsPath(value).drive or ".." in path.parts or "\\" in value:
        raise ValueError(f"{label} must be a portable path inside the project")
    return path.as_posix()


def report_specs(raw):
    """Normalize target validation report declarations without reading them."""
    if not isinstance(raw, list) or len(raw) > 8:
        raise ValueError("target.validation_reports must be a list of at most 8 reports")
    result = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("each target validation report needs a file and label")
        file = _portable_path(item.get("file"), "validation report file")
        label = item.get("label")
        if not isinstance(label, str) or not label.strip() or len(label) > 120:
            raise ValueError("validation report label must be nonempty text of at most 120 characters")
        result.append({"file": file, "label": label.strip()})
    return result


def _feature_ids(raw):
    if not isinstance(raw, list) or not raw or len(raw) > 32:
        raise ValueError("viewer evidence needs 1 to 32 feature_ids")
    result = []
    for value in raw:
        if not isinstance(value, str) or not _FEATURE_ID.fullmatch(value) or value in result:
            raise ValueError("viewer evidence feature_ids must be unique valid feature IDs")
        result.append(value)
    return result


def _sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def engine_source_digest(root=None):
    """Identify the installed engine source that interpreted model evidence."""
    root = pathlib.Path(root) if root is not None else pathlib.Path(__file__).resolve().parent
    content = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in (".py", ".toml") and "__pycache__" not in path.parts:
            content.update(path.relative_to(root).as_posix().encode())
            content.update(b"\0")
            content.update(path.read_bytes())
            content.update(b"\0")
    return content.hexdigest()


def runtime_versions(packages):
    versions = {}
    for package in packages:
        if package == "python":
            versions[package] = sys.version.split()[0]
        elif package == "OCP_module":
            try:
                import OCP

                versions[package] = getattr(OCP, "__version__", "unavailable")
            except ImportError:
                versions[package] = "unavailable"
        else:
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = "unavailable"
    return versions


def viewer_contract(root, feature_ids, files, *, environment=None):
    """Capture the portable input bytes a specialized validator just measured."""
    root = pathlib.Path(root).resolve()
    identities = {}
    for value in files:
        relative = _portable_path(str(value), "viewer evidence input")
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"viewer evidence input does not exist: {relative}")
        identities[relative] = _sha256(path)
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "feature_ids": _feature_ids(feature_ids),
        "inputs": {"files": identities},
    }
    if environment is not None:
        if not isinstance(environment, dict):
            raise ValueError("viewer evidence environment must be an object")
        versions = environment.get("runtime_versions")
        engine = environment.get("nurb_source_sha256")
        if not isinstance(versions, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in versions.items()):
            raise ValueError("viewer evidence runtime_versions must map package names to versions")
        if not isinstance(engine, str) or not re.fullmatch(r"[0-9a-f]{64}", engine):
            raise ValueError("viewer evidence nurb_source_sha256 must be a SHA-256 digest")
        result["inputs"]["runtime_versions"] = dict(sorted(versions.items()))
        result["inputs"]["nurb_source_sha256"] = engine
    return result


def _report_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("validation report leaves the project")
    return path


def _compact_findings(raw):
    if not isinstance(raw, list):
        return []
    result = []
    for item in raw[:8]:
        if not isinstance(item, dict):
            continue
        code, message = item.get("code"), item.get("message")
        if isinstance(code, str) and isinstance(message, str):
            result.append({"code": code[:160], "message": message[:1000]})
    return result


def _evaluate(root, spec):
    changed, watched = [], set()
    report_path = _report_path(root, spec["file"])
    watched.add(report_path)
    try:
        report_before = report_path.read_bytes()
        report = json.loads(report_before)
        envelope = report.get("viewer_evidence")
        if not isinstance(envelope, dict) or envelope.get("schema_version") != SCHEMA_VERSION or envelope.get("kind") != KIND:
            raise ValueError("report has no supported viewer_evidence contract")
        feature_ids = _feature_ids(envelope.get("feature_ids"))
        inputs = envelope.get("inputs")
        files = inputs.get("files") if isinstance(inputs, dict) else None
        if not isinstance(files, dict) or not files:
            raise ValueError("viewer evidence inputs.files must contain recorded project files")
        snapshots = {}
        for relative, recorded in files.items():
            relative = _portable_path(relative, "viewer evidence input")
            if not isinstance(recorded, str) or not re.fullmatch(r"[0-9a-f]{64}", recorded):
                raise ValueError(f"viewer evidence input has no SHA-256 digest: {relative}")
            path = _report_path(root, relative)
            watched.add(path)
            try:
                body = path.read_bytes()
            except OSError:
                changed.append(relative)
                continue
            snapshots[path] = body
            if hashlib.sha256(body).hexdigest() != recorded:
                changed.append(relative)
        versions = inputs.get("runtime_versions")
        if versions is not None:
            if not isinstance(versions, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in versions.items()):
                raise ValueError("viewer evidence runtime_versions must map package names to versions")
            if runtime_versions(versions) != versions:
                changed.append("runtime versions")
        engine = inputs.get("nurb_source_sha256")
        if engine is not None:
            if not isinstance(engine, str) or not re.fullmatch(r"[0-9a-f]{64}", engine):
                raise ValueError("viewer evidence nurb_source_sha256 must be a SHA-256 digest")
            if engine_source_digest() != engine:
                changed.append("nurb engine source")
        if report_path.read_bytes() != report_before:
            changed.append(spec["file"] + " changed during snapshot")
        for path, body in snapshots.items():
            try:
                same = path.read_bytes() == body
            except OSError:
                same = False
            if not same:
                changed.append(path.relative_to(root).as_posix() + " changed during snapshot")
        accepted = report.get("accepted") is True
        reported_status = report.get("status")
        if not isinstance(report.get("accepted"), bool) or not isinstance(reported_status, str):
            raise ValueError("report needs boolean accepted and text status fields")
        status = "stale" if changed else "accepted" if accepted else "failed"
        result = {
            "file": spec["file"],
            "label": spec["label"],
            "feature_ids": feature_ids,
            "status": status,
            "freshness": "stale" if changed else "current",
            "accepted": accepted and not changed,
            "reported_status": reported_status,
            "findings": _compact_findings(report.get("findings")),
            "changed": sorted(set(changed)),
        }
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        result = {
            "file": spec["file"],
            "label": spec["label"],
            "feature_ids": [],
            "status": "stale",
            "freshness": "stale",
            "accepted": False,
            "reported_status": "unavailable",
            "findings": [],
            "changed": [spec["file"]],
            "error": str(exc),
        }
    return result, watched


def load_reports(root, specs):
    """Load compact verdicts and return every path whose replacement expires them."""
    root = pathlib.Path(root).resolve()
    reports, watched = [], set()
    for spec in report_specs(specs):
        report, paths = _evaluate(root, spec)
        reports.append(report)
        watched.update(paths)
    return reports, watched
