"""Specialized validator verdicts stay bound to current portable inputs."""

import json

import pytest
import trimesh

from nurb import checks, compare, validator_evidence
from nurb.server import Server


FEATURES = ["headset-rim", "cushion-opening"]


def report(root, *, accepted=True, status="accepted", files=("parts/model.py",), environment=None):
    value = {
        "status": status,
        "accepted": accepted,
        "findings": [] if accepted else [{"code": "reference.review", "message": "inspect the independent scan evidence"}],
        "viewer_evidence": validator_evidence.viewer_contract(root, FEATURES, files, environment=environment),
    }
    path = root / "references/current.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def project(tmp_path):
    (tmp_path / "parts").mkdir()
    source = tmp_path / "parts/model.py"
    source.write_text("geometry = 1\n")
    return source


def test_current_accepted_and_failed_reports_keep_the_validator_verdict(tmp_path):
    project(tmp_path)
    path = report(tmp_path)
    specs = [{"file": path.relative_to(tmp_path).as_posix(), "label": "Interface validation"}]
    reports, watched = validator_evidence.load_reports(tmp_path, specs)
    assert reports == [{
        "file": "references/current.json", "label": "Interface validation", "feature_ids": FEATURES,
        "status": "accepted", "freshness": "current", "accepted": True, "reported_status": "accepted",
        "findings": [], "changed": [],
    }]
    assert watched == {path.resolve(), (tmp_path / "parts/model.py").resolve()}

    report(tmp_path, accepted=False, status="failed")
    failed = validator_evidence.load_reports(tmp_path, specs)[0][0]
    assert failed["status"] == "failed"
    assert failed["freshness"] == "current"
    assert failed["accepted"] is False
    assert failed["findings"] == [{"code": "reference.review", "message": "inspect the independent scan evidence"}]


def test_changed_and_replaced_inputs_make_an_old_acceptance_stale(tmp_path, monkeypatch):
    source = project(tmp_path)
    path = report(tmp_path)
    specs = [{"file": "references/current.json", "label": "Exact CAD check"}]
    source.write_text("geometry = 2\n")
    stale = validator_evidence.load_reports(tmp_path, specs)[0][0]
    assert stale["status"] == "stale"
    assert stale["accepted"] is False
    assert stale["changed"] == ["parts/model.py"]

    source.write_text("geometry = 1\n")
    original = type(source).read_bytes
    reads = 0

    def replace_during_snapshot(self):
        nonlocal reads
        body = original(self)
        if self.resolve() == source.resolve():
            reads += 1
            if reads == 1:
                source.write_text("geometry = replacement\n")
        return body

    monkeypatch.setattr(type(source), "read_bytes", replace_during_snapshot)
    raced = validator_evidence.load_reports(tmp_path, specs)[0][0]
    assert raced["status"] == "stale"
    assert "parts/model.py changed during snapshot" in raced["changed"]


def test_runtime_and_engine_identity_are_checked_when_the_report_records_them(tmp_path):
    project(tmp_path)
    environment = {
        "runtime_versions": validator_evidence.runtime_versions(["python"]),
        "nurb_source_sha256": validator_evidence.engine_source_digest(),
    }
    report(tmp_path, environment=environment)
    specs = [{"file": "references/current.json", "label": "Bound environment"}]
    assert validator_evidence.load_reports(tmp_path, specs)[0][0]["status"] == "accepted"
    data = json.loads((tmp_path / "references/current.json").read_text())
    data["viewer_evidence"]["inputs"]["runtime_versions"]["python"] = "0.0"
    (tmp_path / "references/current.json").write_text(json.dumps(data))
    stale = validator_evidence.load_reports(tmp_path, specs)[0][0]
    assert stale["status"] == "stale"
    assert stale["changed"] == ["runtime versions"]


def test_card_roundtrip_preserves_validation_report_declarations(tmp_path):
    source = project(tmp_path)
    source.with_suffix(".md").write_text('# model\n\n```toml\ntarget = { file = "scan.stl", units = "mm" }\n```\n')
    declarations = [{"file": "references/current.json", "label": "Interface validation"}]
    compare.update_card(source, validation_reports=declarations)
    compare.update_card(source, regions=[])
    target = compare.setting(checks.settings(source))
    assert target["validation_reports"] == declarations
    with pytest.raises(ValueError, match="portable path"):
        compare.update_card(source, validation_reports=[{"file": "../outside.json", "label": "Bad"}])


def test_malformed_or_uncontracted_reports_fail_closed(tmp_path):
    project(tmp_path)
    path = tmp_path / "references/current.json"
    path.parent.mkdir()
    path.write_text('{"status":"accepted","accepted":true}')
    value = validator_evidence.load_reports(tmp_path, [{"file": "references/current.json", "label": "Missing contract"}])[0][0]
    assert value["status"] == "stale"
    assert value["accepted"] is False
    assert "supported viewer_evidence" in value["error"]


def test_server_attaches_current_validator_status_to_each_covered_feature(tmp_path):
    (tmp_path / "parts").mkdir()
    source = tmp_path / "parts/thing.py"
    source.write_text("from nurb import *\n@part\ndef thing():\n    return Box(10, 10, 10)\n")
    trimesh.creation.box(extents=[10, 10, 10]).export(tmp_path / "scan.stl")
    source.with_suffix(".md").write_text('# thing\n\n```toml\ntarget = { file = "scan.stl", units = "mm" }\n```\n')
    declarations = [{"file": "references/current.json", "label": "Exact interface validation"}]
    compare.update_card(source, regions=[{
        "name": "Headset rim", "bounds_mm": {"min": [-5, -5, -5], "max": [5, 5, 5]},
        "feature": {"id": "headset-rim", "role": "retaining lip"},
    }], validation_reports=declarations)
    report(tmp_path, files=("parts/thing.py",))
    server = Server(tmp_path)
    entry = server.rebuild(source)
    assert entry["target"]["validation_reports"][0]["status"] == "accepted"
    assert entry["target"]["feature_evidence"][0]["validation_reports"][0]["label"] == "Exact interface validation"
    assert server.validation_paths["thing"] == {
        (tmp_path / "references/current.json").resolve(), source.resolve(),
    }
    assert server._validation_dependents({tmp_path / "references/current.json"}) == [source]
