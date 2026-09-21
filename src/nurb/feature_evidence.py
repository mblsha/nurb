"""Named functional features and repeatable local sections, in the aligned part frame."""

import hashlib
import json
import re

import numpy as np


def _text(value, label, limit=2000):
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError(f"{label} must be text of at most {limit} characters")
    return value


def vector(value, label):
    try:
        result = np.asarray(value, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label} needs three finite numbers") from exc
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{label} needs three finite numbers")
    return result


def section_definition(raw):
    """Normalize one local plane and up to 24 parallel stations, without rounding."""
    if not isinstance(raw, dict):
        raise ValueError("a local section needs origin_mm and normal vectors")
    origin = vector(raw.get("origin_mm"), "section origin_mm")
    normal = vector(raw.get("normal"), "section normal")
    if np.linalg.norm(normal) < 1e-12:
        raise ValueError("section normal must not be zero")
    if abs(np.linalg.norm(normal) - 1.0) > 1e-12:
        normal /= np.linalg.norm(normal)
    if "x_direction" in raw:
        u = vector(raw["x_direction"], "section x_direction")
    else:
        u = np.eye(3)[np.argmin(np.abs(normal))]
    if abs(np.dot(u, normal)) > 1e-12:
        u = u - normal * np.dot(u, normal)
    if np.linalg.norm(u) < 1e-12:
        raise ValueError("section x_direction must not be parallel to its normal")
    if abs(np.linalg.norm(u) - 1.0) > 1e-12:
        u /= np.linalg.norm(u)
    offsets = raw.get("offsets_mm", [0.0])
    try:
        offsets = np.asarray(offsets, dtype=float)
        tolerance = float(raw.get("tolerance_mm", 0.0))
    except (ValueError, TypeError) as exc:
        raise ValueError("section offsets and tolerance must be finite numbers") from exc
    if offsets.ndim != 1 or not 1 <= len(offsets) <= 24 or not np.isfinite(offsets).all():
        raise ValueError("section offsets_mm needs 1 to 24 finite station offsets")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("section tolerance_mm must be finite and nonnegative; zero preserves the contour")
    return {"name": _text(raw.get("name", "Local section"), "section name", 120),
            "origin_mm": origin.tolist(), "normal": normal.tolist(), "x_direction": u.tolist(),
            "offsets_mm": offsets.tolist(), "tolerance_mm": tolerance,
            "expected": _text(raw.get("expected", ""), "expected section")}


def feature_record(raw):
    if not isinstance(raw, dict):
        raise ValueError("region feature must be an object")
    identity = raw.get("id")
    if not isinstance(identity, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}", identity):
        raise ValueError("feature id needs 1 to 96 letters, digits, dots, colons, underscores or hyphens")
    result = {"id": identity}
    for key in ("role", "configuration", "orientation", "notes"):
        if key in raw:
            result[key] = _text(raw[key], f"feature {key}")
    if "reference_point_mm" in raw:
        result["reference_point_mm"] = vector(raw["reference_point_mm"], "reference point in source mm").tolist()
    for key in ("required", "excluded", "links"):
        if key in raw:
            if not isinstance(raw[key], list) or len(raw[key]) > 32:
                raise ValueError(f"feature {key} needs at most 32 text entries")
            result[key] = [_text(item, f"feature {key} entry") for item in raw[key]]
    if "uncertainty_mm" in raw:
        try:
            uncertainty = float(raw["uncertainty_mm"])
        except (TypeError, ValueError) as exc:
            raise ValueError("feature uncertainty_mm must be finite and nonnegative") from exc
        if not np.isfinite(uncertainty) or uncertainty < 0:
            raise ValueError("feature uncertainty_mm must be finite and nonnegative")
        result["uncertainty_mm"] = uncertainty
    if "sections" in raw:
        if not isinstance(raw["sections"], list) or len(raw["sections"]) > 8:
            raise ValueError("feature sections needs at most 8 named section series")
        result["sections"] = [section_definition(section) for section in raw["sections"]]
        if sum(len(s["offsets_mm"]) for s in result["sections"]) > 24:
            raise ValueError("a feature supports at most 24 section stations in total")
    if "review" in raw:
        review = raw["review"]
        if not isinstance(review, dict) or not isinstance(review.get("identity"), dict):
            raise ValueError("feature review needs the captured evidence identity")
        fields = ("geometry", "reference", "alignment", "configuration", "feature", "token")
        result["review"] = {"identity": {key: _text(review["identity"].get(key), f"review {key}", 128) for key in fields},
                            "note": _text(review.get("note", ""), "review note")}
    return result


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def evidence_identity(geometry, reference, transform, configuration, feature):
    """Hash the evidence contract too: relabeling a rim invalidates its old review."""
    contract = {key: value for key, value in feature.items() if key != "review"}
    identity = {"geometry": str(geometry), "reference": str(reference),
                "alignment": _digest(list(transform)), "configuration": str(configuration or "default"),
                "feature": _digest(contract)}
    return {**identity, "token": _digest(identity)}


def freshness(feature, identity):
    previous = feature.get("review", {}).get("identity")
    return {"status": "unreviewed" if previous is None else "current" if previous == identity else "stale",
            "identity": identity, "review": feature.get("review")}


def sections(mesh, definitions):
    from . import scan

    cuts = []
    for raw in definitions:
        definition = section_definition(raw)
        for offset in definition["offsets_mm"]:
            spec = {**definition, "origin_mm": (np.asarray(definition["origin_mm"]) + np.asarray(definition["normal"]) * offset).tolist()}
            cut = scan.section(mesh, spec, tolerance=definition["tolerance_mm"])
            cuts.append({"name": definition["name"], "offset_mm": offset, "expected": definition["expected"],
                         **scan._section_structured(cut)})
    return cuts


def shape_identity(shape):
    """Fingerprint exact geometry without cached display triangles or source paths."""
    import io
    from OCP.BRepTools import BRepTools
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
    from OCP.TopoDS import TopoDS_Iterator
    from OCP.TopTools import TopTools_FormatVersion

    copied = BRepBuilderAPI_Copy(shape.wrapped, False, False).Shape()

    def unchecked(topology):
        # Meshing changes Checked flags without changing geometry. Normalize only
        # the copied topology so inspecting never changes the live CAD object.
        topology.Checked(False)
        children = TopoDS_Iterator(topology)
        while children.More():
            unchecked(children.Value())
            children.Next()

    unchecked(copied)
    body = io.BytesIO()
    BRepTools.Write_s(copied, body, False, False, TopTools_FormatVersion.TopTools_FormatVersion_VERSION_3)
    return hashlib.sha256(body.getvalue()).hexdigest()


def region_evidence(shape_id, reference_id, transform, configuration, region):
    feature = region["feature"]
    contract = {**feature, "selection": {key: region[key] for key in ("name", "bounds_mm", "component") if key in region}}
    identity = evidence_identity(shape_id, reference_id, transform, configuration, contract)
    return {"id": feature["id"], "name": region["name"], **freshness(feature, identity)}


def selected_meshes(shape, cad, reference, region, component_meshes=None):
    """Clip a station's scope without inventing material on the selection boundary."""
    from . import compare

    bounds = region.get("bounds_mm")
    if "component" in region:
        components = getattr(getattr(shape, "_nurb_scene", None), "components", ())
        matches = [c for c in components if region["component"] in (c.id, c.label)]
        if len(matches) != 1:
            raise ValueError("the feature needs a unique assembly component")
        component = matches[0]
        cad = component_meshes[component.id] if component_meshes is not None else compare._part_mesh(component.solid, 0.1)
        bounds = {"min": cad.bounds[0].tolist(), "max": cad.bounds[1].tolist()}
    return compare._clip_region(cad, bounds), compare._clip_region(reference, bounds)
