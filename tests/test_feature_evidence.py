"""Evidence identifies the correct interface and expires when its contract changes."""

import asyncio
import json

import numpy as np
import pytest
import trimesh
from build123d import Box, Cylinder, export_step

from nurb import builder, checks, cli, compare, feature_evidence as evidence, scan
from nurb.server import Server


def feature_region():
    return {"name": "Narrow headset rim", "bounds_mm": {"min": [-10, -10, -10], "max": [10, 10, 10]},
            "feature": {"id": "headset-rim", "role": "connects to headset", "orientation": "narrow end",
                        "configuration": "assembled", "reference_point_mm": [10, 0, 0],
                        "required": ["small inverse-T lip"], "excluded": ["nose cloth"],
                        "uncertainty_mm": 0.15, "feature_size_mm": 0.3, "links": ["references/profile.png"],
                        "sections": [{"name": "rim stations", "origin_mm": [0, 0, 0], "normal": [0, 0, 1],
                                      "offsets_mm": [-1, 0, 1], "expected": "two small shoulders"}]}}


def project(tmp_path):
    (tmp_path / "parts").mkdir()
    part = tmp_path / "parts/thing.py"
    part.write_text("from nurb import *\n@part\ndef thing(width=10.0):\n    return Box(width,10,6)\n")
    part.with_suffix(".md").write_text('# thing\n\n```toml\ntarget={file="scan.stl",units="mm",transform=[1,0,0,-10,0,1,0,0,0,0,1,0,0,0,0,1]}\n```\n')
    mesh = trimesh.creation.box(extents=[10, 10, 6]); mesh.apply_translation([10, 0, 0]); mesh.export(tmp_path / "scan.stl")
    compare.update_card(part, regions=[feature_region()])
    server = Server(tmp_path); server.queue = asyncio.Queue()
    server.rebuild(part)
    messages = []

    async def capture(message):
        messages.append(message)

    server.send = capture
    return part, server, messages


def test_local_section_frame_recovers_world_circle_and_rejects_degenerate_axes():
    mesh = trimesh.creation.cylinder(2, 10, sections=128)
    rotation = trimesh.transformations.rotation_matrix(np.pi / 4, [0, 1, 0])
    rotation[:3, 3] = [3, 4, 5]; mesh.apply_transform(rotation)
    normal = rotation[:3, 2]
    raw = {"origin_mm": [3, 4, 5], "normal": normal.tolist(), "x_direction": rotation[:3, 0].tolist()}
    cut = scan._section_structured(scan.section(mesh, raw, tolerance=0))
    circle = cut["loops"][0]["cylinder_candidate"]
    assert cut["axis"] == "local"
    assert circle["axis_point_mm"] == pytest.approx([3, 4, 5], abs=1e-6)
    assert circle["axis_direction"] == pytest.approx(normal)
    assert circle["radius_mm"] == pytest.approx(2, abs=0.001)
    with pytest.raises(ValueError, match="parallel"):
        evidence.section_definition({**raw, "x_direction": normal.tolist()})
    with pytest.raises(ValueError, match="zero"):
        evidence.section_definition({**raw, "normal": [0, 0, 0]})


def test_saved_station_series_preserves_empty_stations_and_exact_step_area(tmp_path, capsys):
    shape = Box(10, 10, 6) - Cylinder(2, 10)
    path = tmp_path / "bored.step"; export_step(shape, path)
    definition = {"origin_mm": [0, 0, 0], "normal": [0, 0, 2], "offsets_mm": [0, 4], "tolerance_mm": 0}
    cli.main(["scan", str(path), "--json", "--local-section", json.dumps(definition)])
    result = json.loads(capsys.readouterr().out)
    assert len(result["sections"]) == 2
    assert result["sections"][1]["origin_mm"] == [0, 0, 4]
    assert result["sections"][1]["loops"] == []
    assert result["inspection"]["analytic"]["sections"][0]["area_mm2"] == pytest.approx(100-4*np.pi)


def test_feature_card_roundtrip_preserves_semantics_and_rejects_duplicate_ids(tmp_path):
    part, server, _ = project(tmp_path)
    regions = compare.setting(checks.settings(part))["regions"]
    assert regions[0]["feature"]["feature_size_mm"] == .3
    assert server.state["thing"]["target"]["feature_evidence"][0]["feature"]["feature_size_mm"] == .3
    assert regions[0]["feature"]["required"] == ["small inverse-T lip"]
    assert regions[0]["feature"]["sections"][0]["normal"] == [0, 0, 1]
    duplicate = {**regions[0], "name": "wrong second name"}
    with pytest.raises(ValueError, match="unique"):
        compare.inspection_regions([regions[0], duplicate])
    context = server.state["thing"]["target"]["feature_evidence"][0]
    assert context["status"] == "unreviewed"
    assert len(context["identity"]["token"]) == 64


def test_exact_geometry_identity_ignores_display_meshing_but_changes_for_holes():
    shape = Box(10, 10, 6)
    first = evidence.shape_identity(shape)
    builder.to_mesh(shape, 0.05)
    assert evidence.shape_identity(shape) == first
    assert evidence.shape_identity(shape - Cylinder(1, 10)) != first


@pytest.mark.parametrize("change", ["geometry", "reference", "alignment", "configuration", "role", "selection", "feature_size", "remove_size", "center", "symmetry_group"])
def test_review_stales_for_every_evidence_identity(change):
    region = compare.inspection_regions([feature_region()])[0]
    args = ["shape1", "scan1", compare.IDENTITY, None, region]
    identity = evidence.region_evidence(*args)["identity"]
    region["feature"]["review"] = {"identity": identity, "note": "shoulders inspected"}
    assert evidence.region_evidence(*args)["status"] == "current"
    if change in ("geometry", "reference"):
        args[["geometry", "reference"].index(change)] += "changed"
    elif change == "alignment":
        args[2] = list(compare.IDENTITY); args[2][3] = 0.1
    elif change == "configuration":
        args[3] = "turned90"
    elif change == "role":
        region["feature"]["role"] = "actually cushion face"
    elif change == "feature_size":
        region["feature"]["feature_size_mm"] = .1
    elif change == "center":
        region["feature"]["center_mm"] = [3, 0, 0]
    elif change == "symmetry_group":
        region["feature"]["symmetry_group"] = "cushion holes"
    elif change == "remove_size":
        del region["feature"]["feature_size_mm"]
    else:
        region["bounds_mm"]["min"][0] -= 1
    assert evidence.region_evidence(*args)["status"] == "stale"


def test_server_sections_align_reference_and_record_only_current_evidence(tmp_path):
    part, server, messages = project(tmp_path)
    token = server.state["thing"]["token"]
    request = {"type": "feature_inspection", "name": "thing", "token": token, "feature_id": "headset-rim"}
    asyncio.run(server.command(json.dumps(request)))
    result = messages[-1]["result"]
    assert len(result["cad"]) == 3
    assert result["reference"][0]["loops"][0]["bounds_mm"] == result["cad"][0]["loops"][0]["bounds_mm"]
    assert result["cad"][0]["origin_mm"] == [0, 0, -1]
    asyncio.run(server.command(json.dumps({**request, "save_review": True, "identity": result["identity"], "note": "lip shoulders checked"})))
    assert messages[-1]["result"]["saved"]
    server.rebuild(part)
    assert server.state["thing"]["target"]["feature_evidence"][0]["status"] == "current"
    server.overrides["thing"] = {"width": 12.0}; server.rebuild(part)
    assert server.state["thing"]["target"]["feature_evidence"][0]["status"] == "stale"
    before = part.with_suffix(".md").read_bytes()
    asyncio.run(server.command(json.dumps({**request, "save_review": True, "identity": result["identity"]})))
    assert "rebuilt" in messages[-1]["error"]
    assert part.with_suffix(".md").read_bytes() == before


def test_cli_regions_file_carries_saved_local_evidence(tmp_path, monkeypatch, capsys):
    part, _, _ = project(tmp_path)
    regions = tmp_path / "regions.json"; regions.write_text(json.dumps([feature_region()]))
    monkeypatch.chdir(tmp_path)
    cli.main(["compare", "thing", "--regions-file", str(regions), "--json"])
    result = json.loads(capsys.readouterr().out)["comparisons"][0]
    feature = result["feature_evidence"][0]
    assert feature["id"] == "headset-rim"
    assert len(feature["cad"]) == len(feature["reference"]) == 3
    assert feature["frame"] == "part_mm"
    assert feature["feature"]["feature_size_mm"] == .3
    assert feature["provenance"]["feature_size_mm"] == .3


def test_local_plane_normalization_is_idempotent_for_saved_evidence():
    raw = {"origin_mm": [1, 2, 3], "normal": [0, 1, 1], "x_direction": [3, 1, 2]}
    once = evidence.section_definition(raw)
    assert evidence.section_definition(once) == once
    assert evidence.section_definition(evidence.section_definition(once)) == once


def test_feature_scale_migrates_without_inventing_unknown_values_or_losing_conflicts():
    legacy = {"id": "rim", "feature_scale_mm": .25}
    assert evidence.feature_record(legacy) == {"id": "rim", "feature_size_mm": .25}
    assert evidence.feature_record({"id": "rim"}) == {"id": "rim"}
    assert evidence.feature_record(evidence.feature_record(legacy)) == evidence.feature_record(legacy)
    with pytest.raises(ValueError, match="disagree"):
        evidence.feature_record({**legacy, "feature_size_mm": .1})
    for value in (0, -1, float("nan"), float("inf"), True, "unknown"):
        with pytest.raises(ValueError, match="positive finite"):
            evidence.feature_record({"id": "rim", "feature_size_mm": value})
    old = {"name": "rim", "feature": {"id": "rim"}}
    identity = evidence.region_evidence("cad", "scan", compare.IDENTITY, None, old)["identity"]
    old["feature"]["review"] = {"identity": identity}
    normalized = {**old, "feature": evidence.feature_record(old["feature"])}
    assert evidence.region_evidence("cad", "scan", compare.IDENTITY, None, normalized)["status"] == "current"


def test_cli_exports_json_and_svg_from_requested_verification_mesh(tmp_path, monkeypatch, capsys):
    from xml.etree import ElementTree
    from nurb import meshing

    part, _, _ = project(tmp_path)
    monkeypatch.chdir(tmp_path)
    policies = []
    def verified(shape, policy, stop=None):
        policies.append(policy)
        return trimesh.creation.box(extents=[10, 10, 6])
    monkeypatch.setattr(meshing, "verification_mesh", verified)
    destination = tmp_path / "evidence"
    cli.main(["compare", "thing", "--json", "--sections-output", str(destination), "--mesh-accuracy", ".004",
              "--mesh-timeout", "2", "--mesh-triangles", "1200", "--feature-size", ".5"])
    result = json.loads(capsys.readouterr().out)["comparisons"][0]
    assert len(policies) == 1
    assert policies[0].accuracy_mm == .004
    assert policies[0].max_triangles == 1200 and 0 < policies[0].timeout_s <= 2
    assert policies[0].feature_size_mm == .3  # A coarser request cannot erase the saved small lip.
    assert len(result["section_exports"]) == 4
    payload = json.loads(next(destination.glob("*.json")).read_text())
    assert payload["kind"] == "local-section-evidence"
    assert payload["feature"]["feature_size_mm"] == .3
    assert len(payload["cad"]) == 3
    assert payload["provenance"]["absolute_deflection_mm"] == .004
    assert payload["provenance"]["measured_error_bound_mm"] is None
    assert "not exact B-rep" in payload["limitation"]
    svg = ElementTree.fromstring(next(destination.glob("*.svg")).read_text())
    ns = {"s": "http://www.w3.org/2000/svg"}
    metadata = json.loads(svg.find("s:metadata", ns).text)
    assert metadata["identity"] == payload["identity"]
    assert len(metadata["cad"]) == 1
    assert svg.findall("s:polyline", ns)
    assert "Requested deflection: 0.004 mm" in " ".join(svg.itertext())


def test_cli_section_export_never_falls_back_after_verification_timeout(tmp_path, monkeypatch, capsys):
    from nurb import meshing
    project(tmp_path)
    monkeypatch.chdir(tmp_path)
    def fail(*args, **kwargs):
        raise meshing.MeshingError("Verification unknown: timeout")
    monkeypatch.setattr(meshing, "verification_mesh", fail)
    destination = tmp_path / "failed"
    cli.main(["compare", "thing", "--json", "--sections-output", str(destination)])
    result = json.loads(capsys.readouterr().out)
    assert result["comparisons"] == []
    assert result["skipped"][0]["status"] == "unknown"
    assert not destination.exists()


def test_server_exports_only_current_sections_with_honest_preview_provenance(tmp_path):
    part, server, messages = project(tmp_path)
    token = server.state["thing"]["token"]
    request = {"type": "feature_inspection", "name": "thing", "token": token, "feature_id": "headset-rim"}
    asyncio.run(server.command(json.dumps(request)))
    result = messages[-1]["result"]
    export = {**request, "type": "feature_export", "identity": result["identity"], "format": "json"}
    asyncio.run(server.command(json.dumps(export)))
    payload = json.loads(messages[-1]["artifact"]["data"])
    assert payload["provenance"]["method"] == "Display mesh estimate"
    assert payload["provenance"]["absolute_deflection_mm"] is None
    assert payload["feature"]["feature_size_mm"] == .3
    asyncio.run(server.command(json.dumps({**export, "format": "svg", "station": 0})))
    assert messages[-1]["artifact"]["mime"] == "image/svg+xml"
    assert "Display mesh accuracy in mm is unknown" in messages[-1]["artifact"]["data"]
    assert "not exact B-rep" in messages[-1]["artifact"]["data"]
    changed = feature_region(); changed["feature"]["feature_size_mm"] = .1
    compare.update_card(part, regions=[changed])
    asyncio.run(server.command(json.dumps(export)))
    assert "changed" in messages[-1]["error"]  # Reject even before the watcher rebuilds.
    server.rebuild(part)
    asyncio.run(server.command(json.dumps({**export, "token": server.state["thing"]["token"]})))
    assert "expired" in messages[-1]["error"]


def test_feature_export_rejects_source_edits_before_watcher_rebuild(tmp_path):
    part, server, messages = project(tmp_path)
    request = {"type": "feature_inspection", "name": "thing", "token": server.state["thing"]["token"], "feature_id": "headset-rim"}
    asyncio.run(server.command(json.dumps(request)))
    identity = messages[-1]["result"]["identity"]
    part.write_text(part.read_text().replace("width=10.0", "width=11.0"))
    asyncio.run(server.command(json.dumps({**request, "type": "feature_export", "identity": identity, "format": "json"})))
    assert "model inputs changed" in messages[-1]["error"]


def test_cli_rejects_reference_changes_during_section_measurement(tmp_path, monkeypatch, capsys):
    from nurb import meshing
    project(tmp_path)
    monkeypatch.chdir(tmp_path)
    def changed(shape, policy, stop=None):
        reference = tmp_path / "scan.stl"
        reference.write_bytes(reference.read_bytes()+b"changed")
        return trimesh.creation.box(extents=[10, 10, 6])
    monkeypatch.setattr(meshing, "verification_mesh", changed)
    cli.main(["compare", "thing", "--json", "--sections-output", str(tmp_path / "output")])
    result = json.loads(capsys.readouterr().out)
    assert result["comparisons"] == []
    assert "reference changed" in result["skipped"][0]["reason"]
    assert not (tmp_path / "output").exists()


def test_section_export_filename_normalization_cannot_collide():
    assert evidence.section_stem("part", "rim:left") != evidence.section_stem("part", "rim_left")
    assert ":" not in evidence.section_stem("part", "rim:left")
    assert len(evidence.section_stem("a"*120, "b"*96)) == 129


def test_server_precise_verification_keeps_smallest_saved_feature_scale(tmp_path, monkeypatch):
    _, server, _ = project(tmp_path)
    policies = []
    async def capture(path, request, policy, client, stopped):
        policies.append(policy)
    monkeypatch.setattr(server, "_verify_target", capture)
    async def run():
        await server.command(json.dumps({"type": "target_verify", "name": "thing", "feature_size_mm": .8}))
        await server.verifications["thing"]
    asyncio.run(run())
    assert policies[0].feature_size_mm == .3
