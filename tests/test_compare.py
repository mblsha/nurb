"""The compare loop's contract: the mesh is the ground truth, and the two directions
never blur. A part that grew a boss the original lacks must show up as "part off the
target" even while every target sample sits happily on the part."""

import asyncio
import gzip
import json
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh
from build123d import Box, Cylinder, Pos

from nurb import builder, checks, cli, compare
from nurb.server import Server

PART = """from nurb import *

@part
def thing(width=40.0, depth=30.0, height=10.0):
    return Box(width, depth, height)
"""

CARD = """# thing

```toml
target = "scans/original.stl"
```
"""


def test_setting_reads_a_path_or_a_table():
    assert compare.setting({}) is None
    assert compare.setting({"target": "scans/a.stl"}) == {
        "file": "scans/a.stl",
        "units": None,
        "tolerance_mm": 0.1,
        "transform": None,
    }
    target = compare.setting(
        {
            "target": {
                "file": "a.stl",
                "units": "in",
                "tolerance_mm": 0.2,
                "transform": compare.IDENTITY,
            }
        }
    )
    assert target["units"] == "in"
    assert target["tolerance_mm"] == 0.2
    assert target["transform"] == compare.IDENTITY
    with pytest.raises(ValueError):
        compare.setting({"target": 3})
    with pytest.raises(ValueError, match="target.units"):
        compare.setting({"target": {"file": "a.stl", "units": ["mm"]}})
    with pytest.raises(ValueError, match="rigid rotation"):
        compare.setting(
            {
                "target": {
                    "file": "a.stl",
                    "transform": [2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                }
            }
        )
    with pytest.raises(ValueError, match="rigid rotation"):
        compare.setting(
            {
                "target": {
                    "file": "a.stl",
                    "transform": [1, 0.25, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                }
            }
        )
    with pytest.raises(ValueError, match="end with"):
        compare.setting(
            {
                "target": {
                    "file": "a.stl",
                    "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1],
                }
            }
        )


def test_identical_geometry_measures_zero_both_ways():
    metrics = compare.against(Box(40, 30, 10), trimesh.creation.box(extents=[40, 30, 10]))
    assert metrics["part"]["max"] < 0.05
    assert metrics["target"]["max"] < 0.05


def test_a_size_difference_is_visible_and_bounded():
    # The part is 10mm taller: its extra skin sits up to 5mm off the target after
    # centering, and the target's top face lies 5mm inside the part, which is surface
    # the part equally fails to reproduce. Both directions must say so.
    metrics = compare.against(Box(40, 30, 20), trimesh.creation.box(extents=[40, 30, 10]))
    assert metrics["part"]["max"] == pytest.approx(5.0, abs=0.3)
    assert metrics["target"]["max"] == pytest.approx(5.0, abs=0.3)


def test_a_translated_target_is_centered_before_measuring():
    mesh = trimesh.creation.box(extents=[40, 30, 10])
    mesh.apply_translation([100.0, -50.0, 25.0])
    metrics = compare.against(Box(40, 30, 10), mesh)
    assert metrics["part"]["max"] < 0.05
    # Box() sits centered at the origin, so the offset is the translation undone.
    assert metrics["offset"] == [-100.0, 50.0, -25.0]


def test_a_stored_transform_is_used_without_recentering():
    mesh = trimesh.creation.box(extents=[40, 30, 10])
    mesh.apply_translation([100.0, -50.0, 25.0])
    identity = compare.against(Box(40, 30, 10), mesh, transform=compare.IDENTITY)
    transform = np.eye(4)
    transform[:3, 3] = [-100.0, 50.0, -25.0]
    aligned = compare.against(Box(40, 30, 10), mesh, transform=transform.reshape(-1))
    assert identity["part"]["sampled_max"] > 50
    assert aligned["part"]["sampled_max"] < 0.03
    assert aligned["transform"] == transform.reshape(-1).tolist()


def test_a_stored_rigid_rotation_maps_the_reference_into_the_cad_frame():
    mesh = trimesh.creation.box(extents=[40, 30, 10])
    outward = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])
    outward[:3, 3] = [12, -8, 3]
    mesh.apply_transform(outward)
    transform = np.linalg.inv(outward)
    metrics = compare.against(
        Box(40, 30, 10),
        mesh,
        transform=transform.reshape(-1),
    )
    assert metrics["part"]["sampled_max"] < 0.03
    assert metrics["target"]["sampled_max"] < 0.03


def test_tolerance_ignores_coarse_faceting_but_keeps_raw_deviation():
    reference = trimesh.creation.cylinder(radius=20, height=10, sections=32)
    metrics = compare.against(Cylinder(20, 10), reference, tolerance_mm=0.11)
    assert 0.05 < metrics["part"]["sampled_max"] < 0.11
    assert metrics["part"]["within_tolerance"] == 1.0
    assert metrics["part"]["excess_max"] == 0.0


def test_feature_samples_keep_a_small_pocket_visible():
    reference = trimesh.creation.cylinder(radius=20, height=10, sections=32)
    pocketed = Cylinder(20, 10) - Pos(0, 0, 4.7) * Box(2, 2, 0.6)
    metrics = compare.against(pocketed, reference, tolerance_mm=0.11)
    assert metrics["part"]["sampled_max"] > 0.5
    assert metrics["target"]["sampled_max"] > 0.5
    assert metrics["part"]["excess_max"] > 0.4
    assert len(metrics["samples"]["part"]["points"]) == metrics["sample_count"]["part"]
    assert metrics["detected_above_tolerance"] is True
    assert metrics["worst_regions"]
    assert metrics["worst_regions"][0]["peak_deviation_mm"] > 0.5
    assert metrics["worst_regions"][0]["direction"] in {"part_to_target", "target_to_part"}


def test_detection_does_not_disappear_when_estimated_coverage_rounds_to_100_percent():
    reference = trimesh.creation.cylinder(radius=20, height=10, sections=32)
    pocketed = Cylinder(20, 10) - Pos(0, 0, 4.7) * Box(0.4, 0.4, 0.6)
    metrics = compare.against(pocketed, reference, tolerance_mm=0.11)
    assert metrics["part"]["within_tolerance"] == 1.0
    assert metrics["target"]["within_tolerance"] == 1.0
    assert metrics["detected_above_tolerance"] is True
    text = "\n".join(compare.report("thing", "reference.stl", {**metrics, "alignment": "identity"}, "mm", "argument"))
    assert "estimated sampled coverage" in text
    assert "DEVIATIONS DETECTED ABOVE TOLERANCE" in text


def project(tmp_path):
    (tmp_path / "parts").mkdir()
    (tmp_path / "parts" / "thing.py").write_text(PART)
    (tmp_path / "parts" / "thing.md").write_text(CARD)
    (tmp_path / "scans").mkdir()
    trimesh.creation.box(extents=[40, 30, 10]).export(tmp_path / "scans" / "original.stl")
    return Server(tmp_path)


def test_rebuild_attaches_the_cards_target(tmp_path):
    server = project(tmp_path)
    entry = server.rebuild(tmp_path / "parts" / "thing.py")
    assert entry["target"]["file"] == "scans/original.stl"
    assert entry["target"]["offset"] == [0.0, 0.0, 0.0]
    assert entry["target"]["dimensions"] == [40.0, 30.0, 10.0]
    assert entry["target"]["unit_source"] == "guess"
    assert entry["target"]["alignment"] == "auto"
    assert entry["target_glb"][:4] == b"glTF"
    # The GLB is served, never wired: a scan is megabytes and the socket is JSON.
    assert "target_glb" not in server._meta(entry)


def test_check_adds_the_deviation_and_reuses_the_loaded_mesh(tmp_path, monkeypatch):
    server = project(tmp_path)
    server.rebuild(tmp_path / "parts" / "thing.py")
    held = server.targets[("scans/original.stl", None)]
    monkeypatch.setattr(
        compare,
        "_part_mesh",
        lambda *_: pytest.fail("the server comparison remeshed the CAD shape"),
    )
    entry = server.check(tmp_path / "parts" / "thing.py")
    assert entry["target"]["metrics"]["part"]["max"] < 0.05
    assert entry["target"]["metrics"]["target"]["max"] < 0.05
    assert entry["target"]["metrics"]["provenance"]["method"] == "Display mesh estimate"
    assert server.targets[("scans/original.stl", None)] is held


def test_display_mesh_comparison_applies_node_transforms_and_reuses_component_meshes(
    tmp_path, monkeypatch
):
    scene = trimesh.Scene()
    transform = trimesh.transformations.translation_matrix([25.0, -10.0, 7.0])
    scene.add_geometry(
        trimesh.creation.box(extents=[2.0, 4.0, 6.0]),
        node_name="shifted-node",
        geom_name="shifted-node",
        transform=transform,
    )
    entry = {
        "glb": scene.export(file_type="glb"),
        "bbox": [2.0, 4.0, 6.0],
        "components": [
            {
                "id": "shifted_1",
                "label": "Shifted",
                "role": "part",
                "node": "shifted-node",
            }
        ],
    }
    whole, components = Server(tmp_path)._comparison_meshes(entry)
    assert whole.bounds.mean(axis=0) == pytest.approx([25.0, -10.0, 7.0])
    assert components["shifted_1"].bounds == pytest.approx(whole.bounds)

    shape = Box(2, 4, 6)
    shape._nurb_scene = SimpleNamespace(
        components=[
            SimpleNamespace(id="shifted_1", label="Shifted", solid=shape)
        ]
    )
    target = trimesh.creation.box(extents=[2.0, 4.0, 6.0])
    target.apply_transform(transform)
    monkeypatch.setattr(
        compare,
        "_part_mesh",
        lambda *_: pytest.fail("the display comparison remeshed a component"),
    )
    metrics = compare.against(
        shape,
        target,
        transform=compare.IDENTITY,
        regions=[{"name": "shifted", "component": "shifted_1"}],
        part_mesh=whole,
        component_meshes=components,
        provenance={"method": "Display mesh estimate"},
    )
    assert metrics["part"]["max"] < 1e-6
    assert metrics["target"]["max"] < 1e-6
    assert metrics["inspection_regions"][0]["status"] == "measured"
    assert metrics["inspection_regions"][0]["provenance"]["method"] == "Display mesh estimate"


def test_display_mesh_comparison_rejects_a_unit_or_transform_mismatch(tmp_path):
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.creation.box(extents=[2.0, 4.0, 6.0]))
    body = scene.export(file_type="glb")
    for expected in ([2000.0, 4000.0, 6000.0], [6.0, 4.0, 2.0]):
        with pytest.raises(ValueError, match="does not match the CAD bounds in millimetres"):
            Server(tmp_path)._comparison_meshes({"glb": body, "bbox": expected})


def test_display_mesh_comparison_allows_relative_faceting_gap_at_smooth_extrema(tmp_path):
    # Relative-deflection tessellation can stop inside a smooth B-rep extremum. These
    # are the measured extents from the spline-loft reconstruction that exposed it.
    displayed = [164.2959, 95.6038, 85.5368]
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.creation.box(extents=displayed))
    whole, _ = Server(tmp_path, tolerance=0.1)._comparison_meshes(
        {"glb": scene.export(file_type="glb"), "bbox": [164.3, 95.72, 85.54]}
    )
    assert whole.extents == pytest.approx(displayed, abs=1e-4)


def test_legacy_auto_alignment_does_not_move_after_a_part_edit(tmp_path):
    server = project(tmp_path)
    part = tmp_path / "parts" / "thing.py"
    first = server.rebuild(part)["target"]["transform"]
    part.write_text(PART.replace("width=40.0", "width=50.0"))
    second = server.rebuild(part)["target"]["transform"]
    assert second == first


def test_legacy_auto_alignment_does_not_jump_when_the_next_build_fails(tmp_path):
    server = project(tmp_path)
    part = tmp_path / "parts" / "thing.py"
    mesh = tmp_path / "scans" / "original.stl"
    moved = trimesh.load(mesh, force="mesh")
    moved.apply_translation([20, 0, 0])
    moved.export(mesh)
    first = server.rebuild(part)["target"]["transform"]
    assert first[3] == pytest.approx(-20)
    part.write_text("from nurb import *\n\nraise RuntimeError('editing')\n")
    failed = server.rebuild(part)
    assert failed["error"]
    assert failed["target"]["transform"] == first


def test_target_settings_write_the_card_and_acknowledge_before_rebuild(tmp_path):
    server = project(tmp_path)
    server.queue = asyncio.Queue()
    sent = []

    async def capture(payload):
        sent.append(payload)

    server.send = capture
    asyncio.run(
        server.command(
            json.dumps(
                {
                    "type": "target_settings",
                    "name": "thing",
                    "tolerance_mm": 0.1234567,
                    "transform": compare.IDENTITY,
                }
            )
        )
    )
    assert sent == [
        {
            "type": "target_settings",
            "name": "thing",
            "written": ["tolerance_mm", "transform"],
        }
    ]
    assert server.queue.get_nowait() == str(tmp_path / "parts" / "thing.py")
    target = compare.setting(checks.settings(tmp_path / "parts" / "thing.py"))
    assert target["tolerance_mm"] == 0.1234567
    assert target["transform"] == compare.IDENTITY
    card_text = (tmp_path / "parts" / "thing.md").read_text()
    assert "```toml\ntarget = {" in card_text
    assert "```tomltarget" not in card_text


def test_updating_a_target_preserves_backslashes_in_its_path(tmp_path):
    part = tmp_path / "thing.py"
    part.write_text(PART)
    part.with_suffix(".md").write_text(
        '# thing\n\n```toml\ntarget = { file = "scans/folder\\\\file.stl", units = "mm" }\n```\n'
    )
    compare.update_card(part, tolerance_mm=0.1234567)
    target = compare.setting(checks.settings(part))
    assert target["file"] == "scans/folder\\file.stl"
    assert target["tolerance_mm"] == 0.1234567


def test_updating_a_multiline_target_table_preserves_following_settings(tmp_path):
    part = tmp_path / "thing.py"
    part.write_text(PART)
    part.with_suffix(".md").write_text(
        "# thing\n\n```toml\n[target]\nfile = \"scans/original.stl\"\nunits = \"mm\"\ntolerance_mm = 0.2\n\n[variants.small.params]\nwidth = 20.0\n```\n"
    )
    compare.update_card(part, tolerance_mm=0.125, transform=compare.IDENTITY)
    target = compare.setting(checks.settings(part))
    assert target["file"] == "scans/original.stl"
    assert target["tolerance_mm"] == 0.125
    assert target["transform"] == compare.IDENTITY
    assert checks.settings(part)["variants"]["small"]["params"]["width"] == 20.0


def test_reference_card_helpers_add_replace_and_remove_multiline_targets(tmp_path):
    part = tmp_path / "thing.py"
    part.write_text(PART)
    part.with_suffix(".md").write_text("# thing\n\n## What it is\n")
    compare.attach_reference(part, "scans/thing.stl", units="mm", tolerance_mm=0.25)
    assert compare.setting(checks.settings(part))["file"] == "scans/thing.stl"
    compare.attach_reference(part, "scans/replacement.stl", units="cm", tolerance_mm=0.1)
    assert compare.setting(checks.settings(part))["units"] == "cm"
    assert compare.remove_reference(part) == ["target"]
    assert compare.setting(checks.settings(part)) is None
    assert "## What it is" in part.with_suffix(".md").read_text()


@pytest.mark.parametrize("reference", [r"C:\outside\mesh.stl", r"\\server\share\mesh.stl"])
def test_reference_card_helpers_reject_windows_absolute_paths(tmp_path, reference):
    part = tmp_path / "thing.py"
    part.write_text(PART)
    part.with_suffix(".md").write_text("# thing\n")
    with pytest.raises(ValueError, match="portable path inside the project"):
        compare.attach_reference(part, reference, units="mm")


def test_attaching_a_reference_inserts_the_target_before_existing_tables(tmp_path):
    part = tmp_path / "thing.py"
    part.write_text(PART)
    part.with_suffix(".md").write_text(
        "# thing\n\n```toml\n[part]\nmin_wall = 2.0\n\n[variants.small.params]\nwidth = 20.0\n```\n"
    )
    compare.attach_reference(part, "scans/thing.stl", units="mm", tolerance_mm=0.25)
    settings = checks.settings(part)
    assert compare.setting(settings)["file"] == "scans/thing.stl"
    assert settings["part"]["min_wall"] == 2.0
    assert "target" not in settings["part"]
    assert settings["variants"]["small"]["params"]["width"] == 20.0


def test_a_nested_target_change_invalidates_it_and_queues_its_part(tmp_path, monkeypatch):
    from nurb import server as server_mod

    server = project(tmp_path)
    part = tmp_path / "parts" / "thing.py"
    original = tmp_path / "scans" / "original.stl"
    target = tmp_path / "scans" / "_original.stl"
    original.rename(target)
    card = part.with_suffix(".md")
    card.write_text(CARD.replace("scans/original.stl", "scans/_original.stl"))
    server.rebuild(part)
    server.queue = asyncio.Queue()
    server.loop = SimpleNamespace(call_soon_threadsafe=lambda fn, arg: fn(arg))

    class FakeObserver:
        def __init__(self):
            self.scheduled = []

        def schedule(self, handler, path, recursive):
            self.scheduled.append((handler, pathlib.Path(path), recursive))

        def start(self):
            pass

    monkeypatch.setattr(server_mod, "Observer", FakeObserver)
    server.watch()
    handler, watched, recursive = next(
        item for item in server.observer.scheduled if item[1] == tmp_path
    )
    assert recursive is True
    target.write_bytes(target.read_bytes() + b"\n")
    handler.on_any_event(
        SimpleNamespace(is_directory=False, src_path=str(target), dest_path="")
    )
    assert server.queue.get_nowait() == str(part)
    assert server.targets == {}


def test_renaming_a_target_away_invalidates_it_and_queues_its_part(tmp_path, monkeypatch):
    from nurb import server as server_mod

    server = project(tmp_path)
    part = tmp_path / "parts" / "thing.py"
    server.rebuild(part)
    server.queue = asyncio.Queue()
    server.loop = SimpleNamespace(call_soon_threadsafe=lambda fn, arg: fn(arg))

    class FakeObserver:
        def __init__(self):
            self.scheduled = []

        def schedule(self, handler, path, recursive):
            self.scheduled.append((handler, pathlib.Path(path), recursive))

        def start(self):
            pass

    monkeypatch.setattr(server_mod, "Observer", FakeObserver)
    server.watch()
    handler = next(
        item[0] for item in server.observer.scheduled if item[1] == tmp_path
    )
    source = tmp_path / "scans" / "original.stl"
    destination = tmp_path / "scans" / "_renamed.stl"
    source.rename(destination)
    handler.on_any_event(
        SimpleNamespace(
            is_directory=False,
            src_path=str(source),
            dest_path=str(destination),
        )
    )
    assert server.queue.get_nowait() == str(part)
    assert server.targets == {}


def test_new_from_mesh_copies_the_reference_and_persists_confirmed_units(tmp_path):
    source = tmp_path / "source.stl"
    trimesh.creation.box(extents=[40, 30, 10]).export(source)
    root = tmp_path / "project"
    cli.main(
        [
            "new",
            "thing",
            "--root",
            str(root),
            "--from",
            str(source),
            "--units",
            "mm",
            "--tolerance",
            "0.2",
        ]
    )
    target = compare.setting(checks.settings(root / "parts" / "thing.py"))
    assert checks.settings(root / "parts" / "thing.py")["reconstruction"] == "bounding_box_draft"
    assert (root / "scans" / "source.stl").is_file()
    assert target["units"] == "mm"
    assert target["tolerance_mm"] == 0.2
    assert target["transform"] == compare.IDENTITY
    source_text = (root / "parts" / "thing.py").read_text()
    assert "width=40.0, depth=30.0, height=10.0" in source_text


def test_compressed_ply_new_project_compares_with_its_saved_reference(tmp_path, monkeypatch, capsys):
    source = tmp_path / "source.PLY.GZ"
    body = gzip.compress(trimesh.creation.box(extents=[40, 30, 10]).export(file_type="ply"))
    source.write_bytes(body)
    root = tmp_path / "project"
    cli.main(["new", "thing", "--root", str(root), "--from", str(source), "--units", "mm"])
    capsys.readouterr()
    part = root / "parts" / "thing.py"
    target = compare.setting(checks.settings(part))
    assert target["file"] == "scans/source.PLY.GZ"
    assert (root / target["file"]).read_bytes() == body
    assert target["transform"] == compare.IDENTITY
    monkeypatch.chdir(root)
    for arguments in ([], ["--against", target["file"]]):
        cli.main(["compare", "thing", *arguments, "--json"])
        result = json.loads(capsys.readouterr().out)["comparisons"][0]
        assert result["reference"]["matches_declared_reference"] is True
        assert result["directions"]["part_to_target"]["sampled_max"] < 1e-6
        assert result["directions"]["target_to_part"]["sampled_max"] < 1e-6
    server = Server(root)
    entry = server.rebuild(part)
    assert entry["target"]["file"] == target["file"]
    assert entry["target_glb"][:4] == b"glTF"


def test_new_from_an_open_planar_reference_starts_with_a_valid_thin_solid(tmp_path):
    source = tmp_path / "sheet.stl"
    trimesh.Trimesh(
        vertices=[[-5, -4, 0], [5, -4, 0], [5, 4, 0], [-5, 4, 0]],
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
    ).export(source)
    root = tmp_path / "sheet-project"
    cli.main(
        [
            "new",
            "sheet",
            "--root",
            str(root),
            "--from",
            str(source),
            "--units",
            "mm",
        ]
    )
    part = root / "parts" / "sheet.py"
    text = part.read_text()
    assert "width=10.0, depth=8.0, height=0.1" in text
    shape, _, _ = builder.build(part)
    assert shape.volume == pytest.approx(8.0)


def test_target_units_version_the_viewers_cached_geometry(tmp_path):
    server = project(tmp_path)
    millimetres = server._target_mesh("scans/original.stl", "mm")
    metres = server._target_mesh("scans/original.stl", "m")
    assert millimetres["stamp"] != metres["stamp"]
    assert metres["mesh"].extents.max() == pytest.approx(
        millimetres["mesh"].extents.max() * 1000
    )


def test_embedded_glb_is_served_exactly_while_comparison_uses_millimetres(tmp_path):
    server = project(tmp_path)
    source = trimesh.creation.box(extents=[0.04, 0.03, 0.01]).export(file_type="glb")
    (tmp_path / "scans" / "textured.glb").write_bytes(source)
    (tmp_path / "parts" / "thing.md").write_text(
        '# thing\n\n```toml\ntarget = { file = "scans/textured.glb", units = "m" }\n```\n'
    )

    entry = server.rebuild(tmp_path / "parts" / "thing.py")
    response = asyncio.run(
        server.http(None, SimpleNamespace(path="/glb/thing.target.glb?cache=ignored"))
    )

    assert response.body == source
    assert entry["target_glb"] == source
    assert entry["target"]["display_scale"] == 1000.0
    assert entry["target"]["dimensions"] == pytest.approx([40.0, 30.0, 10.0])
    assert server.targets[("scans/textured.glb", "m")]["mesh"].extents == pytest.approx(
        [40.0, 30.0, 10.0]
    )


def test_compare_command_walks_the_cards_variants(tmp_path, monkeypatch, capsys):
    project(tmp_path)
    card = CARD.replace(
        "```\n",
        "\n[variants.narrow.params]\nwidth = 20.0\n```\n",
    )
    (tmp_path / "parts" / "thing.md").write_text(card)
    monkeypatch.chdir(tmp_path)

    cli.main(["compare", "thing"])

    output = capsys.readouterr().out
    assert "thing against scans/original.stl" in output
    assert "narrow against scans/original.stl" in output


def test_against_the_declared_file_preserves_its_transform_units_and_tolerance(tmp_path, monkeypatch, capsys):
    project(tmp_path)
    reference = trimesh.creation.box(extents=[40, 30, 10])
    reference.apply_translation([2.0, 0.0, 0.0])
    reference.export(tmp_path / "scans" / "original.stl")
    card = tmp_path / "parts" / "thing.md"
    card.write_text(
        CARD.replace(
            'target = "scans/original.stl"',
            f'target = {{ file = "scans/original.stl", units = "mm", tolerance_mm = 0.25, transform = {compare.IDENTITY} }}',
        )
    )
    monkeypatch.chdir(tmp_path)

    cli.main(["compare", "thing", "--against", "scans/original.stl", "--json"])

    result = json.loads(capsys.readouterr().out)["comparisons"][0]
    assert result["reference"]["matches_declared_reference"] is True
    assert result["reference"]["units"] == "mm"
    assert result["tolerance_mm"] == 0.25
    assert result["alignment"]["mode"] == "stored"
    assert result["alignment"]["persisted"] is True
    assert result["alignment"]["saved_by_this_run"] is False
    assert result["alignment"]["transform"] == compare.IDENTITY
    assert result["directions"]["part_to_target"]["sampled_max"] > 1.9


def test_compare_json_exposes_provenance_counts_and_regions(tmp_path, monkeypatch, capsys):
    project(tmp_path)
    monkeypatch.chdir(tmp_path)
    cli.main(["compare", "thing", "--json"])
    result = json.loads(capsys.readouterr().out)["comparisons"][0]
    assert result["reference"]["identity"]["resolved_path"].endswith("scans/original.stl")
    assert result["sample_counts"]["part_coverage"] == compare.SAMPLES
    assert set(result["directions"]) == {"part_to_target", "target_to_part"}
    assert set(result["directions"]["part_to_target"]) == {
        "estimated_sampled_coverage",
        "sampled_max",
        "sampled_median",
        "sampled_p95",
        "sampled_excess_max",
        "sampled_excess_p95",
    }
    assert result["detected_above_tolerance"] is False


def test_project_wide_compare_json_records_parts_without_targets(tmp_path, monkeypatch, capsys):
    project(tmp_path)
    (tmp_path / "parts" / "unreferenced.py").write_text(PART)
    (tmp_path / "parts" / "unreferenced.md").write_text("# unreferenced\n")
    monkeypatch.chdir(tmp_path)

    cli.main(["compare", "--json"])

    result = json.loads(capsys.readouterr().out)
    assert [item["part"] for item in result["skipped"]] == ["unreferenced"]
    assert result["skipped"][0]["reason"] == "no target in its card"


def test_named_compare_json_records_a_missing_target_in_its_envelope(tmp_path, monkeypatch, capsys):
    project(tmp_path)
    (tmp_path / "parts" / "thing.md").write_text("# thing\n")
    monkeypatch.chdir(tmp_path)

    cli.main(["compare", "thing", "--json"])

    result = json.loads(capsys.readouterr().out)
    assert result["comparisons"] == []
    assert result["skipped"] == [{"part": "thing", "reason": "no target in its card"}]


def test_compare_json_records_malformed_variants_without_printing_prose(tmp_path, monkeypatch, capsys):
    project(tmp_path)
    card = tmp_path / "parts" / "thing.md"
    card.write_text(CARD.replace("```\n", "\n[variants.bad]\nwidth = 20.0\n```\n"))
    monkeypatch.chdir(tmp_path)

    cli.main(["compare", "--json"])

    result = json.loads(capsys.readouterr().out)
    assert result["comparisons"] == []
    assert result["skipped"][0]["part"] == "thing"
    assert "variants.bad.params" in result["skipped"][0]["reason"]


def test_compare_json_records_an_unreadable_reference_without_printing_prose(tmp_path, monkeypatch, capsys):
    project(tmp_path)
    (tmp_path / "scans" / "original.stl").unlink()
    monkeypatch.chdir(tmp_path)

    cli.main(["compare", "thing", "--json"])

    result = json.loads(capsys.readouterr().out)
    assert result["comparisons"] == []
    assert result["skipped"][0]["part"] == "thing"
    assert "no file" in result["skipped"][0]["reason"]


def test_compare_json_keeps_user_part_prints_off_stdout(tmp_path, monkeypatch, capsys):
    project(tmp_path)
    (tmp_path / "parts" / "thing.py").write_text(
        "from nurb import *\n"
        "print('USER MODULE NOISE')\n\n"
        "@part\n"
        "def thing(width=40.0, depth=30.0, height=10.0):\n"
        "    print('USER BUILD NOISE')\n"
        "    return Box(width, depth, height)\n"
    )
    monkeypatch.chdir(tmp_path)

    cli.main(["compare", "thing", "--json"])

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["comparisons"][0]["part"] == "thing"
    assert "USER MODULE NOISE" in captured.err
    assert "USER BUILD NOISE" in captured.err


def test_compare_can_deliberately_persist_an_explicit_alignment(tmp_path, monkeypatch, capsys):
    project(tmp_path)
    monkeypatch.chdir(tmp_path)
    cli.main(["compare", "thing", "--alignment", "identity", "--save-alignment", "--json"])
    result = json.loads(capsys.readouterr().out)["comparisons"][0]
    assert result["alignment"]["mode"] == "identity"
    assert result["alignment"]["persisted"] is True
    assert compare.setting(checks.settings(tmp_path / "parts" / "thing.py"))["transform"] == compare.IDENTITY


def test_saving_alignment_also_persists_an_explicit_unit_override(tmp_path, monkeypatch, capsys):
    project(tmp_path)
    monkeypatch.chdir(tmp_path)
    cli.main(["compare", "thing", "--units", "cm", "--alignment", "center", "--save-alignment", "--json"])
    capsys.readouterr()
    target = compare.setting(checks.settings(tmp_path / "parts" / "thing.py"))
    assert target["units"] == "cm"
    assert target["transform"] is not None


def test_json_does_not_save_a_variant_alignment_when_the_base_build_fails(tmp_path, monkeypatch, capsys):
    project(tmp_path)
    (tmp_path / "parts" / "thing.py").write_text(
        "from nurb import *\n\n"
        "@part\n"
        "def thing(width=40.0):\n"
        "    if width == 40.0:\n"
        "        raise ValueError('base failed')\n"
        "    return Box(width, 30, 10)\n"
    )
    (tmp_path / "parts" / "thing.md").write_text(
        CARD.replace("```\n", "\n[variants.good.params]\nwidth = 20.0\n```\n")
    )
    monkeypatch.chdir(tmp_path)

    cli.main(["compare", "thing", "--alignment", "center", "--save-alignment", "--json"])

    result = json.loads(capsys.readouterr().out)
    assert result["comparisons"] == []
    assert result["skipped"][0]["configuration"] == "thing"
    assert compare.setting(checks.settings(tmp_path / "parts" / "thing.py"))["transform"] is None


def test_compare_text_names_actionable_worst_regions(tmp_path, monkeypatch, capsys):
    project(tmp_path)
    trimesh.creation.box(extents=(36, 30, 10)).export(tmp_path / "scans" / "original.stl")
    monkeypatch.chdir(tmp_path)

    cli.main(["compare", "thing"])

    output = capsys.readouterr().out
    assert "DEVIATIONS DETECTED ABOVE TOLERANCE" in output
    assert "worst " in output
    assert " near (" in output
    assert "excess " in output


def test_viewer_discards_a_ghost_loaded_for_a_replaced_mesh_group():
    from nurb import server as server_mod

    viewer = server_mod.VIEWER.read_text(encoding="utf-8")
    ghost = viewer.split("async function ghostAttach", 1)[1].split(
        "// ---- orientation cube ----", 1
    )[0]
    assert "const group = mesh;" in ghost
    assert "mesh !== group" in ghost
    assert "group.add(g);" in ghost


def test_a_missing_target_file_reports_instead_of_breaking_the_build(tmp_path):
    server = project(tmp_path)
    (tmp_path / "scans" / "original.stl").unlink()
    entry = server.rebuild(tmp_path / "parts" / "thing.py")
    assert entry["error"] is None
    assert "no file at" in entry["target"]["error"]


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ('{ file = ["scans/original.stl"] }', "target is a path"),
        ('{ file = "scans/original.stl", units = ["mm"] }', "target.units"),
        ('{ file = "scans/original.stl", tolerance_mm = "close" }', "target.tolerance_mm"),
        ('{ file = "scans/original.stl", transform = [1, 0] }', "target.transform"),
    ],
)
def test_a_malformed_target_setting_does_not_hide_valid_cad(tmp_path, target, message):
    server = project(tmp_path)
    card = tmp_path / "parts" / "thing.md"
    card.write_text(CARD.replace('target = "scans/original.stl"', f"target = {target}"))
    entry = server.rebuild(tmp_path / "parts" / "thing.py")
    assert entry["error"] is None
    assert entry["glb"][:4] == b"glTF"
    assert message in entry["target"]["error"]


def test_a_reference_remains_available_while_the_cad_source_is_broken(tmp_path):
    server = project(tmp_path)
    part = tmp_path / "parts" / "thing.py"
    card = tmp_path / "parts" / "thing.md"
    card.write_text(
        CARD.replace(
            'target = "scans/original.stl"',
            f'target = {{ file = "scans/original.stl", units = "mm", tolerance_mm = 0.1, transform = {compare.IDENTITY} }}',
        )
    )
    part.write_text("from nurb import *\n\nthis is broken\n")
    entry = server.rebuild(part)
    assert entry["glb"] is None
    assert entry["error"]
    assert entry["target"]["alignment"] == "stored"
    assert entry["target"]["transform"] == compare.IDENTITY
    assert entry["target_glb"][:4] == b"glTF"
    assert "target_glb" not in server._meta(entry)


def test_precise_verification_returns_without_blocking_commands_and_rejects_a_rebuild(tmp_path, monkeypatch):
    import threading
    server = project(tmp_path)
    server.queue = asyncio.Queue()
    path = tmp_path / "parts" / "thing.py"
    original = server.rebuild(path)
    started = threading.Event()
    sent = []

    async def capture(payload):
        sent.append(dict(payload))
        if payload.get("resources", {}).get("phase") == "Validating snapshot inputs": started.set()

    import time
    original_snapshot = server._source_snapshot
    def held(*args, **kwargs):
        time.sleep(1.5)
        return original_snapshot(*args, **kwargs)

    server.send = capture
    monkeypatch.setattr(server, "_source_snapshot", held)

    async def exercise():
        await server.command(json.dumps({"type": "target_verify", "name": "thing", "feature_size_mm": 0.3}))
        task = server.verifications["thing"]
        assert await asyncio.to_thread(started.wait, 5)
        assert not server.building.locked()
        assert sent[0]["status"] == "queued"
        assert sent[1]["status"] == "running"
        assert sent[1]["token"] == original["token"]
        server.rebuild(path)
        await task

    asyncio.run(exercise())
    assert sent[-1]["status"] == "stale"
    assert "metrics" not in sent[-1]
    assert "verification" not in server.state["thing"]["target"]


def test_precise_verification_failure_replaces_old_evidence_without_replacing_live_metrics(tmp_path, monkeypatch):
    from nurb.meshing import MeshingError
    server = project(tmp_path)
    server.queue = asyncio.Queue()
    entry = server.rebuild(tmp_path / "parts" / "thing.py")
    entry["target"]["metrics"] = {"display": "kept"}
    entry["target"]["verification"] = {"status": "measured", "metrics": {"stale": True}}
    sent = []

    async def capture(payload):
        sent.append(dict(payload))

    def failed(*args, **kwargs):
        raise MeshingError("Verification unknown: meshing exceeded its budget")

    server.send = capture
    monkeypatch.setattr(compare, "prepare_precise", failed)

    async def exercise():
        await server.command(json.dumps({"type": "target_verify", "name": "thing"}))
        assert "metrics" not in entry["target"]["verification"]
        await server.verifications["thing"]

    asyncio.run(exercise())
    assert sent[-1]["status"] == "unknown"
    assert "metrics" not in entry["target"]["verification"]
    assert entry["target"]["metrics"] == {"display": "kept"}
    assert entry["target"]["last_verification"]["metrics"] == {"stale": True}
    assert not server.verifications


def test_absolute_cli_evidence_names_accuracy_and_feature_budget(tmp_path, monkeypatch, capsys):
    project(tmp_path)
    monkeypatch.chdir(tmp_path)
    cli.main(["compare", "thing", "--json", "--mesh-accuracy", "0.02", "--feature-size", "0.3"])
    result = json.loads(capsys.readouterr().out)["comparisons"][0]
    assert result["status"] == "measured"
    assert len(result["source_revision"]) == 64
    assert result["provenance"]["absolute_deflection_mm"] == 0.02
    assert result["provenance"]["feature_size_mm"] == 0.3
    assert result["provenance"]["cad_triangles"] == 12
    assert result["provenance"]["measured_error_bound_mm"] is None


def test_failed_live_comparison_drops_previous_metrics(tmp_path, monkeypatch):
    server = project(tmp_path)
    path = tmp_path / "parts" / "thing.py"
    entry = server.rebuild(path)
    entry["target"]["metrics"] = {"outdated": True}
    def fail(*args, **kwargs):
        raise ValueError("invalid display geometry")
    monkeypatch.setattr(compare, "against", fail)
    server.check(path)
    assert "metrics" not in entry["target"]
    assert "invalid display geometry" in entry["target"]["error"]


def test_precise_verification_real_worker_finishes_and_keeps_preview_estimate(tmp_path):
    server = project(tmp_path)
    server.queue = asyncio.Queue()
    entry = server.rebuild(tmp_path / "parts" / "thing.py")
    entry["target"]["metrics"] = {"display": "estimate"}
    sent = []

    async def capture(payload):
        sent.append(dict(payload))

    server.send = capture

    async def exercise():
        await server.command(json.dumps({"type": "target_verify", "name": "thing", "accuracy_mm": 0.02, "feature_size_mm": 0.3}))
        await server.verifications["thing"]

    asyncio.run(exercise())
    assert sent[0]["status"] == "queued" and sent[-1]["status"] == "measured"
    assert all(item["status"] == "running" for item in sent[1:-1])
    assert sent[-1]["resources"]["memory"]["enforcement"] == "child RSS monitor"
    assert sent[-1]["metrics"]["provenance"]["cad_triangles"] == 12
    assert sent[-1]["identity"]["shape_id"] == entry["shape_id"]
    assert sent[-1]["identity"]["build_inputs"]
    assert entry["target"]["metrics"] == {"display": "estimate"}


def test_cli_timeout_is_unknown_and_has_no_comparison_numbers(tmp_path, monkeypatch, capsys):
    project(tmp_path)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as error:
        cli.main(["compare", "thing", "--json", "--mesh-timeout", "0.001"])
    assert error.value.code == 2
    result = json.loads(capsys.readouterr().out)
    assert result["comparisons"] == []
    assert result["skipped"][0]["status"] == "unknown"
    assert result["skipped"][0]["failure"] == "timeout"


@pytest.mark.parametrize("changed", ["source", "reference", "sliders", "card"])
def test_verification_refuses_inputs_that_changed_before_the_rebuild(tmp_path, monkeypatch, changed):
    server = project(tmp_path)
    server.queue = asyncio.Queue()
    path = tmp_path / "parts" / "thing.py"
    server.rebuild(path)
    if changed == "source":
        path.write_text(PART.replace("width=40.0", "width=45.0"))
    elif changed == "reference":
        trimesh.creation.box(extents=[45, 30, 10]).export(tmp_path / "scans" / "original.stl")
    elif changed == "card":
        compare.update_card(path, tolerance_mm=.02)
    else:
        server.overrides["thing"] = {"width": 45.0}
    sent = []

    async def capture(payload):
        sent.append(dict(payload))

    server.send = capture
    monkeypatch.setattr(compare, "against", lambda *a, **kw: pytest.fail("outdated geometry reached verification"))

    async def exercise():
        await server.command(json.dumps({"type": "target_verify", "name": "thing"}))
        await server.verifications["thing"]

    asyncio.run(exercise())
    assert sent[-1]["status"] == "stale"
    assert sent[-1]["reason"] == "stale"
    assert "metrics" not in sent[-1]


def test_cancel_verification_requires_matching_request_and_produces_no_metrics(tmp_path, monkeypatch):
    import threading
    import time
    server = project(tmp_path)
    server.queue = asyncio.Queue()
    entry = server.rebuild(tmp_path / "parts" / "thing.py")
    started = threading.Event()
    sent = []

    async def capture(payload):
        sent.append(dict(payload))
        if payload.get("resources", {}).get("phase") == "Validating snapshot inputs": started.set()

    original_snapshot = server._source_snapshot
    def held(*args, **kwargs):
        time.sleep(2)
        return original_snapshot(*args, **kwargs)

    server.send = capture
    monkeypatch.setattr(server, "_source_snapshot", held)

    async def exercise():
        await server.command(json.dumps({"type": "target_verify", "name": "thing"}))
        task = server.verifications["thing"]
        assert await asyncio.to_thread(started.wait, 5)
        request, stopped = server.verification_controls["thing"]
        cancel = {"type": "target_verify_cancel", "name": "thing", "token": entry["token"], "request_id": "old"}
        await server.command(json.dumps(cancel))
        assert not stopped.is_set()
        cancel["request_id"] = request["request_id"]
        await server.command(json.dumps(cancel))
        await task

    asyncio.run(exercise())
    assert sent[0]["status"] == "queued"
    assert sent[-1]["status"] == "cancelled"
    assert "cancelling" in [response["status"] for response in sent]
    assert "metrics" not in sent[-1]
    assert entry["target"]["verification"]["status"] == "cancelled"
    assert not server.verifications
    assert not server.verification_controls
