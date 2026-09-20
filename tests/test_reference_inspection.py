"""Inspection evidence stays local, aligned, and distinct from the global comparison."""

import asyncio
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh
from build123d import Box, Cylinder, Pos, export_brep, export_step

from nurb import checks, cli, compare, scan
from nurb.server import Server


def project(tmp_path):
    (tmp_path / "parts").mkdir()
    part = tmp_path / "parts" / "thing.py"
    part.write_text("from nurb import *\n@part\ndef thing():\n    return Box(40,30,10)\n")
    part.with_suffix(".md").write_text('# thing\n\n```toml\ntarget = {file="source.stl", units="mm"}\n```\n')
    trimesh.creation.box(extents=[40, 30, 10]).export(tmp_path / "source.stl")
    return part, Server(tmp_path)


def test_region_clips_sampling_without_artificial_cut_surfaces():
    shape = Box(40, 30, 10) + Pos(10, 0, 6) * Box(2, 2, 2)
    mesh = trimesh.creation.box(extents=[40, 30, 10])
    regions = [
        {"name": "body", "bounds_mm": {"min": [-10, -5, -6], "max": [0, 5, 6]}},
        {"name": "boss", "bounds_mm": {"min": [8, -2, 4], "max": [12, 2, 8]}},
        {"name": "air", "bounds_mm": {"min": [100, 100, 100], "max": [110, 110, 110]}},
    ]
    result = compare.against(shape, mesh, transform=compare.IDENTITY, regions=regions)
    body, boss, air = result["inspection_regions"]
    assert result["part"]["sampled_max"] > 1.9
    assert body["status"] == "measured"
    assert body["part"]["sampled_max"] < 0.001
    assert body["target"]["sampled_max"] < 0.001
    assert boss["part"]["sampled_max"] > 1.9
    assert boss["worst_regions"]
    assert air["status"] == "empty"
    assert air["part"] is None


def test_component_region_is_resolved_or_explicitly_unresolved():
    shape = Box(10, 10, 10)
    shape._nurb_scene = SimpleNamespace(components=[SimpleNamespace(id="base/1", label="Base", solid=shape)])
    regions = [{"name": "base", "component": "base/1"}, {"name": "missing", "component": "other"}]
    result = compare.against(shape, trimesh.creation.box(extents=[10, 10, 10]), regions=regions)
    base, missing = result["inspection_regions"]
    assert base["status"] == "measured"
    assert base["capture"]["frame"] == "part_mm"
    assert missing["status"] == "unresolved"
    assert "part" not in missing


@pytest.mark.parametrize("target", ['target = "source.stl"', '[target]\nfile="source.stl"'])
def test_named_regions_survive_card_edits(tmp_path, target):
    part, _ = project(tmp_path)
    part.with_suffix(".md").write_text(f'# thing\n\n```toml\n{target}\n\n[variants.small.params]\nwidth=10\n```\n')
    regions = [{"name": 'top "plate"', "bounds_mm": {"min": [-5, -5, 0], "max": [5, 5, 10]}}]
    compare.update_card(part, regions=regions)
    compare.update_card(part, tolerance_mm=0.25)
    settings = checks.settings(part)
    assert compare.setting(settings)["regions"] == regions
    assert settings["variants"]["small"]["params"]["width"] == 10


def test_datum_alignment_composes_in_part_frame_and_rejects_degenerate_landmarks():
    current = np.eye(4)
    current[:3, 3] = [10, 0, 0]
    preview = compare.datum_alignment(current.ravel(), {"kind": "plane", "origin_mm": [10, 0, 2], "normal": [0, 1, 0]})
    delta = np.asarray(preview["delta_transform"]).reshape(4, 4)
    assert delta @ [10, 0, 2, 1] == pytest.approx([0, 0, 0, 1])
    assert delta[:3, :3] @ [0, 1, 0] == pytest.approx([0, 0, 1])
    assert np.asarray(preview["transform"]).reshape(4, 4) == pytest.approx(delta @ current)
    landmarks = {"kind": "landmarks", "source_mm": [[0, 0, 0], [1, 0, 0], [0, 1, 0]], "target_mm": [[3, 4, 5], [3, 5, 5], [2, 4, 5]]}
    result = compare.datum_alignment(compare.IDENTITY, landmarks)
    assert result["max_residual_mm"] < 1e-10
    assert np.linalg.det(np.asarray(result["transform"]).reshape(4, 4)[:3, :3]) == pytest.approx(1)
    with pytest.raises(ValueError, match="one line"):
        compare.datum_alignment(compare.IDENTITY, {**landmarks, "source_mm": [[0, 0, 0], [1, 0, 0], [2, 0, 0]]})


def test_preview_is_readonly_and_save_invalidates_metrics(tmp_path):
    part, server = project(tmp_path)
    entry = server.rebuild(part)
    entry["target"]["metrics"] = {"inspection_regions": [{"name": "old"}]}
    server.queue = asyncio.Queue()
    messages = []

    async def capture(payload):
        messages.append(payload)

    server.send = capture
    before = part.with_suffix(".md").read_text()
    command = {"type": "target_alignment", "name": "thing", "token": entry["token"], "operation": {"kind": "axis", "origin_mm": [0, 0, 0], "direction": [1, 0, 0]}}
    asyncio.run(server.command(json.dumps(command)))
    assert messages[-1]["written"] == []
    assert part.with_suffix(".md").read_text() == before
    assert server.queue.empty()
    asyncio.run(server.command(json.dumps({**command, "save": True})))
    assert messages[-1]["written"] == ["transform"]
    assert "metrics" not in entry["target"]
    assert entry["target"]["stale"] is True
    assert not server.queue.empty()
    asyncio.run(server.command(json.dumps({**command, "token": "stale"})))
    assert "rebuilt" in messages[-1]["error"]


def test_server_inspection_returns_compact_source_frame_with_alignment(tmp_path):
    part, server = project(tmp_path)
    entry = server.rebuild(part)
    server.queue = asyncio.Queue()
    messages = []

    async def capture(payload):
        messages.append(payload)

    server.send = capture
    asyncio.run(server.command(json.dumps({"type": "target_inspection", "name": "thing"})))
    response = messages[-1]
    assert response["stamp"] == entry["target"]["stamp"]
    assert response["inspection"]["frame"] == "source_mm"
    assert len(response["inspection"]["sections"]) == 3
    assert response["transform"] == entry["target"]["transform"]
    asyncio.run(server.command(json.dumps({"type": "target_inspection", "name": "thing", "sections": [False]})))
    assert "error" in messages[-1]


class PendingBuild:
    """Hold the kernel gate until the test has completed a queued build."""

    def __init__(self):
        self.waiting = asyncio.Event()
        self.finished = asyncio.Event()

    async def __aenter__(self):
        self.waiting.set()
        await self.finished.wait()

    async def __aexit__(self, *args):
        pass


def test_inspection_reads_reference_and_token_after_pending_build(tmp_path):
    part, server = project(tmp_path)
    old = server.rebuild(part)
    server.queue = asyncio.Queue()
    responses = []

    async def capture(payload):
        responses.append(payload)

    server.send = capture

    async def run():
        server.building = PendingBuild()
        request = asyncio.create_task(server.command(json.dumps({"type": "target_inspection", "name": "thing", "sections": []})))
        await asyncio.wait_for(server.building.waiting.wait(), 2)
        transform = list(compare.IDENTITY)
        transform[3:12:4] = [3, 4, 5]
        compare.update_card(part, transform=transform)
        trimesh.creation.box(extents=[12, 8, 6]).export(tmp_path / "source.stl")
        current = server.rebuild(part)
        server.building.finished.set()
        await asyncio.wait_for(request, 2)
        return current

    current = asyncio.run(run())
    response = responses[-1]
    assert response["token"] == current["token"] != old["token"]
    assert response["stamp"] == current["target"]["stamp"] != old["target"]["stamp"]
    assert response["transform"] == current["target"]["transform"]
    assert response["inspection"]["extents_mm"] == pytest.approx([12, 8, 6])


@pytest.mark.parametrize("change", ["entry", "settings", "transform", "reference_file"])
def test_inspection_rejects_changes_while_measurement_is_running(tmp_path, monkeypatch, change):
    part, server = project(tmp_path)
    entry = server.rebuild(part)
    server.queue = asyncio.Queue()
    responses = []

    async def capture(payload):
        responses.append(payload)

    server.send = capture

    async def run():
        loop = asyncio.get_running_loop()
        measuring, finish = asyncio.Event(), threading.Event()

        def inspection(*args):
            loop.call_soon_threadsafe(measuring.set)
            if not finish.wait(2):
                raise AssertionError("the test did not release inspection")
            return {"frame": "source_mm"}

        monkeypatch.setattr(scan, "inspection", inspection)
        request = asyncio.create_task(server.command(json.dumps({"type": "target_inspection", "name": "thing", "sections": []})))
        await asyncio.wait_for(measuring.wait(), 2)
        try:
            if change == "entry":
                server.state["thing"] = {**entry, "token": "new-build"}
            elif change == "settings":
                await server.command(json.dumps({"type": "target_settings", "name": "thing", "tolerance_mm": 0.25}))
            elif change == "reference_file":
                trimesh.creation.box(extents=[12, 8, 6]).export(tmp_path / "source.stl")
            else:
                entry["target"]["transform"][3] += 1
        finally:
            finish.set()
        await asyncio.wait_for(request, 2)

    asyncio.run(run())
    response = responses[-1]
    assert response["type"] == "target_inspection"
    assert "changed during inspection" in response["error"]
    assert "inspection" not in response
    assert "token" not in response


def test_alignment_checks_build_token_after_waiting_and_does_not_save_old_datums(tmp_path):
    part, server = project(tmp_path)
    old = server.rebuild(part)
    server.queue = asyncio.Queue()
    responses = []

    async def capture(payload):
        responses.append(payload)

    server.send = capture
    before = part.with_suffix(".md").read_bytes()

    async def run():
        server.building = PendingBuild()
        request = asyncio.create_task(server.command(json.dumps({
            "type": "target_alignment", "name": "thing", "token": old["token"], "save": True,
            "operation": {"kind": "axis", "origin_mm": [0, 0, 0], "direction": [1, 0, 0]},
        })))
        await asyncio.wait_for(server.building.waiting.wait(), 2)
        server.rebuild(part)
        server.building.finished.set()
        await asyncio.wait_for(request, 2)

    asyncio.run(run())
    assert "rebuilt" in responses[-1]["error"]
    assert part.with_suffix(".md").read_bytes() == before
    assert server.queue.empty()


@pytest.mark.parametrize("suffix,export", [(".step", export_step), (".brep", export_brep)])
def test_analytic_summary_keeps_exact_cylinder_and_hole_section(tmp_path, monkeypatch, capsys, suffix, export):
    path = tmp_path / f"bored{suffix}"
    export(Box(20, 20, 10) - Cylinder(3, 20), path)
    monkeypatch.chdir(tmp_path)
    cli.main(["scan", str(path), "--summary", "--section", "z:0mm", "--json"])
    result = json.loads(capsys.readouterr().out)
    analytic = result["analytic"]
    bore = next(face for face in analytic["faces"] if face.get("surface") == "bore")
    assert bore["radius_mm"] == pytest.approx(3)
    section = analytic["sections"][0]
    assert section["area_mm2"] == pytest.approx(400 - 9 * np.pi)
    assert section["faces"][0]["hole_count"] == 1
    assert all("points_mm" not in feature for feature in result["sections"][0]["features"])


def test_cli_regions_and_datum_use_same_persisted_settings(tmp_path, monkeypatch, capsys):
    part, _ = project(tmp_path)
    monkeypatch.chdir(tmp_path)
    cli.main(["compare", "thing", "--region", "top=-10,-10,0:10,10,8", "--save-regions", "--datum", '{"kind":"landmarks","source_mm":[[0,0,0]],"target_mm":[[0,0,1]]}', "--save-alignment", "--json"])
    result = json.loads(capsys.readouterr().out)["comparisons"][0]
    assert result["inspection_regions"][0]["name"] == "top"
    settings = compare.setting(checks.settings(part))
    assert settings["transform"][11] == pytest.approx(1)
    assert settings["regions"][0]["name"] == "top"
