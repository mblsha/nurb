"""The shipped reconstruction accepts coarse facets and catches a real size error."""

import importlib.util
import pathlib

import pytest

from nurb import builder, card, checks, compare


ROOT = pathlib.Path(__file__).resolve().parents[1] / "examples" / "reverse_engineering"
PART = ROOT / "parts" / "disk.py"
BRACKET = ROOT / "parts" / "offset_bracket.py"


def test_reference_fixtures_match_their_generator_geometry():
    generator_path = ROOT / "generate_fixtures.py"
    spec = importlib.util.spec_from_file_location("reverse_engineering_fixtures", generator_path)
    assert spec and spec.loader
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    for name, contents in generator.generated().items():
        assert generator.stl_geometry_matches((ROOT / "scans" / name).read_bytes(), contents)
        mesh = generator.trimesh.load_mesh(generator.io.BytesIO(contents), file_type="stl", process=False)
        reordered = generator.trimesh.Trimesh(
            vertices=mesh.vertices, faces=mesh.faces[::-1, ::-1], process=False
        ).export(file_type="stl")
        assert generator.stl_geometry_matches(reordered, contents)

        moved = mesh.copy()
        moved.apply_translation([0.05, 0.0, 0.0])
        assert not generator.stl_geometry_matches(moved.export(file_type="stl"), contents)


@pytest.fixture(scope="module")
def reference():
    target = compare.setting(checks.settings(PART))
    mesh, unit, source = compare.load(ROOT, target["file"], units=target["units"])
    assert (unit, source) == ("mm", "argument")
    assert mesh.is_watertight
    assert mesh.extents == pytest.approx([40, 40, 8])
    assert target["transform"] == compare.IDENTITY
    assert card.thin(PART.with_suffix(".md").read_text()) == []
    return target, mesh


@pytest.mark.parametrize("variant", ["disk", "oversized"])
def test_reference_workflow_accepts_faceting_but_rejects_the_oversized_variant(reference, variant):
    target, mesh = reference
    configurations = {name: (params, ctx) for name, params, ctx in checks.configurations(PART)}
    params, ctx = configurations[variant]
    shape, _, _ = builder.build(PART, overrides=params or None)
    assert shape.is_valid
    assert len(shape.solids()) == 1
    assert checks.run(shape, ctx) == []
    metrics = compare.against(
        shape,
        mesh,
        tolerance_mm=target["tolerance_mm"],
        transform=target["transform"],
    )
    for direction in ("part", "target"):
        result = metrics[direction]
        if variant == "disk":
            assert 0.03 < result["sampled_max"] < target["tolerance_mm"]
            assert result["within_tolerance"] == 1.0
            assert result["excess_max"] == 0.0
        else:
            assert result["sampled_max"] > 0.9
            assert result["within_tolerance"] < 0.95
            assert result["excess_max"] > 0.7


def test_offset_bracket_preserves_reference_frame_and_finds_small_locator():
    target = compare.setting(checks.settings(BRACKET))
    assert target["transform"] == compare.IDENTITY
    assert card.thin(BRACKET.with_suffix(".md").read_text(encoding="utf-8")) == []
    reference, _, _ = compare.load(ROOT, target["file"], units=target["units"])
    assert reference.bounds[0, 0] == pytest.approx(12.0, abs=0.05)
    assert reference.bounds[0, 1] == pytest.approx(-7.0, abs=0.05)

    exact, _, _ = builder.build(BRACKET)
    exact_metrics = compare.against(
        exact,
        reference,
        tolerance_mm=target["tolerance_mm"],
        transform=target["transform"],
    )
    assert exact_metrics["part"]["within_tolerance"] > 0.995
    assert exact_metrics["target"]["within_tolerance"] > 0.995

    missing, _, _ = builder.build(BRACKET, overrides={"locator_notch_depth": 0.0})
    missing_metrics = compare.against(
        missing,
        reference,
        tolerance_mm=target["tolerance_mm"],
        transform=target["transform"],
    )
    assert missing_metrics["part"]["sampled_max"] > target["tolerance_mm"]
    assert missing_metrics["target"]["sampled_max"] > target["tolerance_mm"]
    assert missing_metrics["detected_above_tolerance"] is True
    assert missing_metrics["worst_regions"]
    assert missing_metrics["worst_regions"][0]["peak_deviation_mm"] > 1.5
    assert missing_metrics["worst_regions"][0]["position_mm"][0] > 55.0
    assert missing_metrics["worst_regions"][0]["sample_count"] > 1
