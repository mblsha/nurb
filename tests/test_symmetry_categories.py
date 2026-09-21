"""Authored landmarks and periodic seams are distinct from material boundary evidence."""
import pytest
from build123d import Compound, Cylinder, Pos, Rot

from nurb import compare, feature_evidence, symmetry, symmetry_categories as categories

PLANE = {"normal": [1, 0, 0], "offset_mm": 0}


def records(right=(3, 0, 0)):
    return [{"name": name, "component": "body", "feature": {"id": name, "role": "cushion hole", "center_mm": list(point)}}
            for name, point in (("left", (-3, 0, 0)), ("right", right))]


@pytest.mark.parametrize("right,remove,status,error,unmatched", [((3,0,0),False,"within_sampled_threshold",0,0),
    ((3,.4,0),False,"deviations",.4,0), ((3,0,0),True,"deviations",None,1)])
def test_explicit_center_pairing_detects_shifted_and_unmatched_feature(right, remove, status, error, unmatched):
    locations = categories.landmarks(records(right)[:1] if remove else records(right), compare.IDENTITY)
    result = categories.feature_centers(locations["cad_centers"], PLANE, .01)
    assert result["status"] == status
    assert result["unmatched_count"] == unmatched
    assert result["sides"]["negative"]["count"] == 1
    assert result["sides"]["negative"]["unmatched_count"] == unmatched
    if error is None:
        assert result["records"][0]["error_mm"] is None
        assert "max_mm" not in result["statistics"]
    else:
        assert result["sides"]["negative"]["max_mm"] == pytest.approx(error)
        assert result["sides"]["positive"]["max_mm"] == pytest.approx(error)
        assert result["records"][0]["paired_id"] == "right"


def test_annotations_use_saved_alignment_but_never_invent_centers_or_mix_groups():
    regions = records()
    regions[0]["feature"].update(reference_point_mm=[7,0,0], symmetry_group="left-only")
    regions[1]["feature"]["reference_point_mm"] = [13,0,0]
    regions.append({"name":"unknown", "bounds_mm":{"min":[0,0,0],"max":[2,2,2]},"feature":{"id":"unknown","role":"hole"}})
    matrix = list(compare.IDENTITY); matrix[3] = -10
    saved = categories.landmarks(regions, matrix)
    assert saved["cad_centers"][0]["point_mm"] == [-3,0,0]
    assert saved["reference_points"][0]["point_mm"] == [-3,0,0]
    assert len(saved["unlocated_features"]) == 1
    assert saved["unlocated_features"][0]["id"] == "unknown"
    measured = categories.feature_centers(saved["cad_centers"], PLANE, .01)
    assert measured["unmatched_count"] == 2
    regions[0]["feature"].pop("symmetry_group")
    aligned = categories.landmarks(regions, matrix)
    assert categories.feature_centers(aligned["reference_points"], PLANE, .2)["status"] == "within_sampled_threshold"
    empty = categories.feature_centers([], PLANE, .01)
    assert empty["status"] == "not_assessed"
    assert empty["count"] == 0


def test_center_normalization_and_semantic_identity_preserve_unspecified_values():
    assert feature_evidence.feature_record({"id":"old"}) == {"id":"old"}
    center = feature_evidence.feature_record({"id":"center","point_mm":[1,2,3],"symmetry_group":"holes"})
    assert center == {"id":"center","center_mm":[1,2,3],"symmetry_group":"holes"}
    with pytest.raises(ValueError, match="disagree"):
        feature_evidence.feature_record({**center,"point_mm":[4,5,6]})
    with pytest.raises(ValueError, match="finite"):
        feature_evidence.feature_record({"id":"bad","center_mm":[float("nan"),0,0]})
    first = categories.landmarks(records(), compare.IDENTITY)
    moved = categories.landmarks(records((3,.4,0)), compare.IDENTITY)
    shape = Cylinder(2,4)
    args = [shape,"reference",compare.IDENTITY,"default","revision",symmetry.Options()]
    identity = symmetry.identity(*args, first)
    assert symmetry.identity(*args, moved)["token"] != identity["token"]
    assert categories.category_identity(identity,"cad_centers") != categories.category_identity(identity,"periodic_seams")


@pytest.mark.parametrize("left_rotation,status", [(180,"within_sampled_threshold"),(150,"deviations")])
def test_periodic_seam_perturbation_is_visible_even_when_material_is_symmetric(left_rotation, status):
    shape = Compound(children=[Pos(3,0,0)*Cylinder(1,4), Pos(-3,0,0)*Rot(0,0,left_rotation)*Cylinder(1,4)])
    result = symmetry.cad_symmetry(shape, PLANE, symmetry.Options(edge_step_mm=2))
    assert result["status"] == "within_sampled_threshold"
    seams = result["periodic_seams"]
    assert seams["status"] == status
    assert seams["periodic_face_count"] == seams["count"] == 2
    assert seams["sample_count"] == 10
    assert seams["tolerance_mm"] == .01
    for side in ("negative", "positive"):
        assert seams["sides"][side]["count"] == 1
        assert seams["sample_sides"][side]["count"] == 5
        if left_rotation == 150:
            assert seams["sides"][side]["max_mm"] > .5
        else:
            assert seams["sides"][side]["max_mm"] < 1e-6
    assert "parameterization" in seams["limitation"]


def test_seam_on_reflection_plane_is_self_paired_and_still_budgeted():
    shape = Rot(0,0,90)*Cylinder(1,4)
    options = symmetry.Options(edge_step_mm=2)
    result = categories.periodic_seams(shape, PLANE, options, 5)
    assert result["status"] == "within_sampled_threshold"
    assert result["sides"]["on_plane"]["count"] == 1
    assert result["records"][0]["paired_id"] == result["records"][0]["id"]
    with pytest.raises(ValueError, match="budget"):
        categories.periodic_seams(shape, PLANE, options, 4)
