"""Assemblies: the sweep must find the jam the printability rules cannot see.

The fixture geometry is chosen so the jam angle is analytic, not observed: a flap
hinged along the edge of a slab it lies flat against can rise and swing over its own
hinge, and the first material interference is at exactly 180 degrees, when its
trailing face crosses into the slab's quadrant. Downward it interferes immediately.
No tolerance in these tests was tuned to make them pass.
"""

import pathlib

import pytest

from nurb import builder, checks

FLAP_ASM = '''from nurb import *


@assembly
def rig(open_deg=0.0, wall=True):
    slab = Pos(0, 15, -1) * Box(20, 30, 2)
    flap = Pos(0, 15, 1) * Box(20, 30, 2)
    flap = hinge(flap, Axis((0, 0, 0), (1, 0, 0)), through=(-90, 270), at=open_deg,
                 name="flap")
    if not wall:
        return (flap,)
    return slab, flap
'''

PLATE = '''from nurb import *


@part
def plate(width=10.0, draft=False):
    return Box(width, 10, 2)
'''

USE_ASM = '''from nurb import *


@assembly
def carrier(open_deg=0.0):
    base = Pos(0, 0, -5) * Box(40, 40, 2)
    arm = use("plate")
    arm = hinge(Pos(0, 10, 0) * arm, Axis((0, 5, 0), (1, 0, 0)), through=(0, 90),
                at=open_deg, name="arm")
    return base, arm
'''


@pytest.fixture
def project(tmp_path):
    (tmp_path / "parts").mkdir()
    return tmp_path


def write(project, name, text):
    path = project / "parts" / f"{name}.py"
    path.write_text(text)
    return path


def motion(shape):
    return [f for f in checks.run(shape) if f.rule == "motion"]


def test_static_fit_findings_have_a_plain_viewer_label():
    assert checks.label("clearance") == "fit clearance"


def test_an_assembly_builds_to_one_compound_the_existing_pipeline_can_carry(project):
    path = write(project, "rig", FLAP_ASM)
    shape, params, _ = builder.build(path)
    assert shape.volume > 0
    assert [p["name"] for p in params] == ["open_deg", "wall"]
    # The whole point of returning a Compound: to_glb and stats need no new code.
    assert builder.to_glb(shape)
    assert builder.stats(shape)["bbox"][0] == pytest.approx(20, abs=0.1)


def test_the_sweep_finds_the_analytic_jam_in_both_directions(project):
    path = write(project, "rig", FLAP_ASM)
    shape, _, _ = builder.build(path)
    found = motion(shape)
    assert len(found) == 2
    up = max(f.value for f in found)
    down = min(f.value for f in found)
    # Rising, the flap swings over the hinge and its trailing face crosses into the
    # slab's quadrant at exactly 180. Falling, it presses into the slab at once.
    assert up == pytest.approx(180.0, abs=0.2)
    assert down == pytest.approx(0.0, abs=0.2)
    for f in found:
        assert f.severity == checks.FAIL
        assert f.where is not None
        assert "flap" in f.message


def test_lying_flat_against_the_slab_is_tangent_and_tangent_is_clear(project):
    """Zero-volume contact is not a collision, the same way a fit is not a weld."""
    path = write(project, "rig", FLAP_ASM)
    shape, _, _ = builder.build(path)
    found = motion(shape)
    assert all("before it moves" not in f.message for f in found)


def test_without_the_wall_the_whole_range_is_clear(project):
    path = write(project, "rig", FLAP_ASM)
    shape, _, _ = builder.build(path, overrides={"wall": False})
    assert motion(shape) == []


def test_a_pose_already_inside_the_wall_is_its_own_finding(project):
    path = write(project, "rig", FLAP_ASM)
    shape, _, _ = builder.build(path, overrides={"open_deg": -30.0})
    found = motion(shape)
    assert len(found) == 1
    assert "before it moves" in found[0].message


def test_use_builds_a_sibling_part_and_the_scene_carries_it(project):
    write(project, "plate", PLATE)
    path = write(project, "carrier", USE_ASM)
    shape, _, _ = builder.build(path)
    scene = shape._nurb_scene
    assert [h.name for h in scene.hinges] == ["arm"]
    assert len(scene.statics) == 1
    # 0..90 with the base 5mm below the hinge: nothing in the way going up.
    assert motion(shape) == []


def test_a_hinged_solid_that_is_not_returned_is_an_error_not_a_silence(project):
    path = write(
        project,
        "broken",
        FLAP_ASM.replace("return slab, flap", "return (slab,)"),
    )
    with pytest.raises(Exception, match="not.*returned|returned.*not"):
        builder.build(path)


def test_printability_rules_stay_off_assemblies(project):
    """An assembled scene would fail overhang and min_wall for reasons that mean
    nothing: each part already answered those alone."""
    path = write(project, "rig", FLAP_ASM)
    shape, _, _ = builder.build(path)
    assert all(f.rule == "motion" for f in checks.run(shape))


def test_a_hinge_knows_which_parameter_posed_it(project):
    """`hinge(at=open_deg)` ties the joint to the slider without anyone saying so.
    Arithmetic strips the link, correctly: `at=open_deg / 2` is not the slider's own
    value, and a rebuild is then the only honest way to pose it."""
    path = write(project, "rig", FLAP_ASM)
    shape, _, _ = builder.build(path)
    assert shape._nurb_scene.hinges[0].param == "open_deg"

    halved = write(project, "rig2", FLAP_ASM.replace("at=open_deg,", "at=open_deg / 2,"))
    shape, _, _ = builder.build(halved)
    assert shape._nurb_scene.hinges[0].param is None


def test_a_shared_file_edit_reaches_through_the_use_cache(project):
    """measurements.toml and system.py feed geometry without touching the part
    file's mtime, so the cache stamp has to cover them or an assembly keeps
    serving the old solid -- silently, which is the failure measured() exists
    to prevent."""
    import os

    # The module, not the decorator `from nurb import assembly` would fetch.
    from nurb import assembly as _decorator  # noqa: F401  -- documents the trap
    from nurb.assembly import _built

    shared = project / "system.py"
    shared.write_text("# shared\n")
    write(project, "plate", PLATE)
    path = write(project, "carrier", USE_ASM)

    _built.clear()
    builder.build(path)
    before = set(_built)
    assert len(before) == 1

    stat = shared.stat()
    os.utime(shared, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    builder.build(path)
    after = set(_built)
    assert len(after) == 1
    assert after != before  # the old entry was evicted, not joined


def test_the_watcher_learns_dependents_from_the_scenes_already_built(project):
    """Editing a part while watching its assembly is the whole editing loop, so a
    changed part has to drag its assemblies into the rebuild."""
    from nurb.server import Server

    write(project, "plate", PLATE)
    path = write(project, "carrier", USE_ASM)
    shape, _, _ = builder.build(path)

    srv = Server.__new__(Server)
    srv.root = project
    srv.state = {"carrier": {"name": "carrier", "shape": shape}}
    plate = str(project / "parts" / "plate.py")
    assert srv._dependents({plate}) == {str(path)}
    assert srv._dependents({str(path)}) == set()  # nothing uses the assembly


def test_export_of_an_assembly_writes_its_placed_parts_and_never_the_weld(project, monkeypatch, capsys):
    """A merged scene is a weld, and its obstacles were never going to be printed.
    Named explicitly, an assembly exports the parts it places; a project sweep
    skips it because those parts already export as themselves."""
    import argparse

    from nurb import cli

    write(project, "plate", PLATE)
    write(project, "carrier", USE_ASM)
    monkeypatch.chdir(project)

    cli.cmd_export(argparse.Namespace(part="carrier", formats=["stl"]))
    assert (project / "build" / "plate.stl").exists()
    assert not (project / "build" / "carrier.stl").exists()
    assert "exporting the 1 part(s) it places" in capsys.readouterr().out

    (project / "build" / "plate.stl").unlink()
    cli.cmd_export(argparse.Namespace(part=None, formats=["stl"]))
    assert (project / "build" / "plate.stl").exists()
    assert not (project / "build" / "carrier.stl").exists()
    assert "skipped" in capsys.readouterr().out


def test_the_viewer_export_route_zips_an_assemblys_placed_parts(project):
    """One download: separate files die silently on the browser's
    multiple-download permission, and the merged scene is a weld."""
    import io
    import zipfile

    from nurb.server import Server

    write(project, "plate", PLATE)
    write(project, "carrier", USE_ASM)
    srv = Server(project)

    body, attach, mime, _ = srv._export("carrier", "stl")
    assert (attach, mime) == ("carrier-stl.zip", "application/zip")
    assert zipfile.ZipFile(io.BytesIO(body)).namelist() == ["plate.stl"]

    body, attach, mime, _ = srv._export("plate", "stl")
    assert (attach, mime) == ("plate.stl", "model/stl")
    assert body[:80]  # a real file, not a wrapper


def test_an_assembly_glb_keeps_its_movers_as_named_nodes(project):
    """The node names are the viewer's contract for posing a joint client-side."""
    import io

    import trimesh

    path = write(project, "rig", FLAP_ASM)
    shape, _, _ = builder.build(path)
    glb = builder.to_glb(shape)
    scene = trimesh.load(io.BytesIO(glb), file_type="glb")
    names = set(scene.geometry)
    assert "joint0" in names
    assert "component_part_1" in names
    # A plain part stays one anonymous blob; nothing downstream should start
    # depending on node names existing for everything.
    write(project, "plate", PLATE)
    solid, _, _ = builder.build(project / "parts" / "plate.py")
    plain = trimesh.load(io.BytesIO(builder.to_glb(solid)), file_type="glb")
    assert len(plain.geometry) == 1


def test_solids_whose_boxes_are_apart_never_reach_a_kernel_boolean():
    """The AABB reject is what keeps a clean full-circle sweep affordable: most
    poses of a real rotor are nowhere near most of the scene (#127)."""
    from build123d import Box, Pos

    from nurb.assembly import _hits

    booleans = []

    class Spy:
        def __init__(self, solid):
            self._solid = solid

        def bounding_box(self):
            return self._solid.bounding_box()

        def __and__(self, other):
            booleans.append(other)
            return self._solid & other

    near, far = Pos(1, 0, 0) * Box(2, 2, 2), Pos(100, 0, 0) * Box(2, 2, 2)
    vol, where = _hits(Spy(Box(2, 2, 2)), [far, near])
    assert booleans == [near]  # far was rejected on boxes alone
    assert vol == pytest.approx(1 * 2 * 2, rel=1e-6)


def test_a_stop_that_turns_true_interrupts_the_sweep_between_poses(project):
    """The dev server aborts a sweep when a rebuild is queued, then drain retries
    it after the queued geometry lands (#127)."""
    from nurb.assembly import Interrupted

    path = write(project, "rig", FLAP_ASM)
    shape, _, _ = builder.build(path)
    with pytest.raises(Interrupted):
        checks.run(shape, stop=lambda: True)
    # A stop that stays false changes nothing.
    assert [f.rule for f in checks.run(shape, stop=lambda: False)] == ["motion"] * 2


def glb_nodes(shape):
    import json
    import struct

    glb = builder.to_glb(shape)
    length = struct.unpack_from("<I", glb, 12)[0]
    return json.loads(glb[20:20 + length])["nodes"]


def test_component_nodes_keep_repeated_uses_and_context_identity(project):
    write(project, "plate", PLATE)
    path = write(project, "pair", '''from nurb import *

@assembly
def pair(offset=20.0):
    left = use("plate")
    right = Pos(offset, 0, 0) * use("plate", width=8.0)
    machine = Pos(0, 0, -5) * obstacle(Box(40, 40, 2), "camera body")
    return left, right, machine
''')
    first, _, _ = builder.build(path)
    second, _, _ = builder.build(path, overrides={"offset": 30.0})
    nodes = [n for n in glb_nodes(first) if "mesh" in n]
    assert [n["name"] for n in nodes] == [
        "component_plate_1", "component_plate_2", "component_camera_body_1",
    ]
    metadata = [n["extras"]["nurb"] for n in nodes]
    assert metadata == [
        {"id": "plate_1", "label": "plate", "role": "part"},
        {"id": "plate_2", "label": "plate", "role": "part"},
        {"id": "camera_body_1", "label": "camera body", "role": "context"},
    ]
    assert [c.wire() for c in second._nurb_scene.components] == metadata
    assert len(first._nurb_scene.obstacles) == 1
    assert len(first._nurb_scene.instances) == 2
    assert first._nurb_scene.instances[1].overrides == (("width", 8.0),)


@pytest.mark.parametrize("gap, minimum, failure", [
    (0.0, 0.0, False), (0.5, 0.5, False), (1.0, 0.5, False),
    (0.25, 0.5, True), (-0.5, 0.0, True), (-0.5, 0.5, True),
])
def test_declared_static_clearance_measures_gaps_and_overlap(project, gap, minimum, failure):
    path = write(project, "fit", '''from nurb import *

@assembly
def fit(gap=0.0, minimum=0.0):
    mount = component(Box(2, 2, 2), "mount")
    camera = obstacle(Pos(2 + gap, 0, 0) * Box(2, 2, 2), "camera")
    clearance(mount, "camera", minimum=minimum)
    return mount, camera
''')
    shape, _, _ = builder.build(path, overrides={"gap": gap, "minimum": minimum})
    findings = checks.run(shape)
    assert bool(findings) is failure
    if failure:
        finding, = findings
        assert finding.rule == "clearance"
        assert finding.components == ("mount_1", "camera_1")
        assert finding.value == pytest.approx(max(0, gap))
        assert finding.measurements == pytest.approx({
            "clearance_mm": max(0, gap), "minimum_mm": minimum,
            "overlap_mm3": max(0, -gap) * 4,
        })
        assert finding.where[0] == pytest.approx(1 + gap / 2)
        assert "mount and camera" in finding.message
    assert checks.run(shape, only=["motion"]) == []
    assert checks.run(shape, only=["min_wall"]) == []


def test_empty_intersections_are_clear_even_when_bounding_boxes_overlap():
    from build123d import Pos, Sphere

    from nurb.assembly import _hits

    assert _hits(Sphere(1), [Pos(1.5, 1.5, 0) * Sphere(1)]) == (0.0, None)


def test_static_clearance_is_declarative_not_a_global_collision_policy(project):
    path = write(project, "fit", '''from nurb import *

@assembly
def fit():
    return Box(2, 2, 2), Pos(1, 0, 0) * Box(2, 2, 2)
''')
    shape, _, _ = builder.build(path)
    assert checks.run(shape) == []


@pytest.mark.parametrize("declaration, match", [
    ('clearance("missing", second)', "was not returned"),
    ('clearance("plate", second)', "ambiguous"),
    ('clearance(first, first)', "two different"),
    ('clearance(first, second, minimum=-1)', "finite, nonnegative"),
    ('clearance(first, second, minimum=float("nan"))', "finite, nonnegative"),
])
def test_clearance_declaration_errors_name_the_repair(project, declaration, match):
    path = write(project, "fit", f'''from nurb import *

@assembly
def fit():
    first = component(Box(2, 2, 2), "plate")
    second = component(Pos(4, 0, 0) * Box(2, 2, 2), "plate")
    {declaration}
    return first, second
''')
    with pytest.raises(Exception, match=match):
        builder.build(path)


def test_clearance_can_address_repeated_components_by_stable_id(project):
    path = write(project, "fit", '''from nurb import *

@assembly
def fit():
    first = component(Box(2, 2, 2), "plate")
    second = component(Pos(4, 0, 0) * Box(2, 2, 2), "plate")
    clearance("plate_1", "plate_2", minimum=3.0)
    return first, second
''')
    shape, _, _ = builder.build(path)
    assert checks.run(shape)[0].components == ("plate_1", "plate_2")


def test_repeated_nested_assemblies_preserve_child_roles_and_placed_geometry(project):
    import io
    import zipfile

    import trimesh

    from nurb.server import Server

    write(project, "plate", PLATE)
    write(project, "nested", '''from nurb import *

@assembly
def nested():
    plate = Pos(2, 0, 0) * use("plate", width=4.0)
    camera = obstacle(Pos(0, 8, 1) * Box(6, 4, 2), "camera")
    clearance(plate, camera, minimum=2.0)
    return plate, camera
''')
    pair = write(project, "pair", '''from nurb import *

@assembly
def pair():
    left = Pos(10, 20, 30) * Rot(0, 0, 90) * use("nested")
    right = Pos(-20, 0, 0) * use("nested")
    clearance(left, right)
    return left, right
''')
    shape, _, _ = builder.build(pair)
    scene = shape._nurb_scene
    components = {c.id: c for c in scene.components}
    assert list(components) == [
        "nested_1", "nested_1/plate_1", "nested_1/camera_1",
        "nested_2", "nested_2/plate_1", "nested_2/camera_1",
    ]
    assert components["nested_1"].group is True
    assert components["nested_2/camera_1"].role == "context"
    assert components["nested_1/plate_1"].parent == "nested_1"
    centers = {
        "nested_1/plate_1": (10, 22, 30), "nested_1/camera_1": (2, 20, 31),
        "nested_2/plate_1": (-18, 0, 0), "nested_2/camera_1": (-20, 8, 1),
    }
    for identity, expected in centers.items():
        assert tuple(components[identity].solid.bounding_box().center()) == pytest.approx(expected)
    glb = builder.to_glb(shape)
    loaded = trimesh.load(io.BytesIO(glb), file_type="glb")
    nodes = {node["extras"]["nurb"]["id"]: node for node in glb_nodes(shape) if "extras" in node}
    assert "mesh" not in nodes["nested_1"]
    assert len(loaded.geometry) == 4  # groups must not duplicate their children's meshes
    for identity, expected in centers.items():
        record = components[identity]
        assert nodes[identity]["extras"]["nurb"] == record.wire()
        mesh = loaded.geometry[record.node]
        assert mesh.bounds.mean(axis=0) == pytest.approx(expected)
    assert len(scene.instances) == 2
    assert all(instance.path.endswith("/plate.py") for instance in scene.instances)
    assert all(instance.overrides == (("width", 4.0),) for instance in scene.instances)
    body, filename, _, _ = Server(project)._export("pair", "stl")
    assert filename == "pair-stl.zip"
    assert zipfile.ZipFile(io.BytesIO(body)).namelist() == ["plate.stl"]
    findings = checks.run(shape)
    assert [f.components for f in findings] == [
        ("nested_1/plate_1", "nested_1/camera_1"),
        ("nested_2/plate_1", "nested_2/camera_1"),
    ]
    assert all(f.value == pytest.approx(1.0) for f in findings)

    outer = write(project, "outer", '''from nurb import *

@assembly
def outer():
    return Pos(100, 0, 0) * use("pair")
''')
    wrapped, _, _ = builder.build(outer)
    deepest = next(c for c in wrapped._nurb_scene.components if c.id == "pair_1/nested_1/plate_1")
    assert tuple(deepest.solid.bounding_box().center()) == pytest.approx((110, 22, 30))
    assert len(trimesh.load(io.BytesIO(builder.to_glb(wrapped)), file_type="glb").geometry) == 4


def test_a_hinged_nested_assembly_keeps_its_group_node_and_context_children(project):
    write(project, "nested", '''from nurb import *

@assembly
def nested():
    return component(Box(2, 2, 2), "plate"), obstacle(Pos(4, 0, 0) * Box(2, 2, 2), "camera")
''')
    path = write(project, "outer", '''from nurb import *

@assembly
def outer(angle=30.0):
    return hinge(use("nested"), Axis.Z, through=(0,90), at=angle, name="carrier")
''')
    shape, _, _ = builder.build(path)
    nodes = {node["name"]: node for node in glb_nodes(shape)}
    assert "mesh" not in nodes["joint0"]
    assert len(nodes["joint0"]["children"]) == 2
    camera = next(c for c in shape._nurb_scene.components if c.label == "camera")
    assert camera.role == "context"
    assert tuple(camera.solid.bounding_box().center()) == pytest.approx((2 * 3 ** 0.5, 2, 0))


def test_failed_boolean_is_an_unknown_clearance_not_zero_overlap():
    from build123d import Box
    from types import SimpleNamespace
    from nurb.assembly import _hits, check_clearances

    class BrokenBoolean:
        label = "broken mount"

        def bounding_box(self):
            return Box(2, 2, 2).bounding_box()

        def __and__(self, other):
            raise RuntimeError("kernel refused the intersection")

    first = SimpleNamespace(id="mount_1", label="mount", solid=BrokenBoolean())
    second = SimpleNamespace(id="camera_1", label="camera", solid=Box(2, 2, 2))
    scene = SimpleNamespace(clearances=[SimpleNamespace(first=first, second=second, minimum=0.2)])
    with pytest.raises(ValueError, match="could not be verified"):
        _hits(first.solid, [second.solid])
    finding, = check_clearances(scene)
    assert finding.value is None
    assert finding.where is None
    assert finding.components == ("mount_1", "camera_1")
    assert finding.measurements == {"status": "unknown", "minimum_mm": 0.2}
    assert "Clearance unknown" in finding.message
