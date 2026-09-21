"""Semantic landmarks and parameterization seams, separate from material symmetry."""
import hashlib
import json

import numpy as np


def landmarks(regions, transform):
    from .feature_evidence import feature_record
    matrix = np.asarray(transform).reshape(4, 4)
    result = {"cad_centers": [], "reference_points": [], "unlocated_features": []}
    for region in regions:
        if "feature" not in region:
            continue
        feature = feature_record(region["feature"])
        item = {"id": feature["id"], "name": region["name"], "role": feature.get("role", ""),
                "group": feature.get("symmetry_group") or feature.get("role") or "unclassified"}
        if "center_mm" in feature:
            result["cad_centers"].append({**item, "point_mm": feature["center_mm"], "location_source": "explicit CAD center in part mm"})
        if "reference_point_mm" in feature:
            point = (matrix @ np.r_[feature["reference_point_mm"], 1])[:3]
            result["reference_points"].append({**item, "point_mm": point.tolist(), "location_source": "explicit reference annotation transformed from source mm"})
        if "center_mm" not in feature and "reference_point_mm" not in feature:
            result["unlocated_features"].append({**item, "reason": "no explicit center or reference point"})
    return result


def _side(point, plane):
    signed = float(np.dot(point, plane["normal"]) - plane["offset_mm"])
    return "negative" if signed < -1e-7 else "positive" if signed > 1e-7 else "on_plane"


def _summary(records, tolerance, count_key="count"):
    from .symmetry import statistics
    sides = {}
    for side in ("negative", "positive", "on_plane"):
        selected = [r for r in records if r["side"] == side]
        values = [r["error_mm"] for r in selected if r["error_mm"] is not None]
        sides[side] = {**statistics(values, tolerance), count_key: len(selected), "measured_count": len(values),
                       "unmatched_count": sum(r["error_mm"] is None for r in selected)}
    errors = [r["error_mm"] for r in records if r["error_mm"] is not None]
    unmatched = sum(r["error_mm"] is None for r in records)
    return {"status": "not_assessed" if not records else "deviations" if unmatched or any(error > tolerance for error in errors) else "within_sampled_threshold",
            "count": len(records), "matched_count": len(errors), "unmatched_count": unmatched,
            "tolerance_mm": tolerance, "threshold_statistic": "maximum", "statistics": statistics(errors, tolerance), "sides": sides, "records": records}


def feature_centers(items, plane, tolerance):
    """One-to-one counterparts within a declared semantic group; never infer a center."""
    from scipy.optimize import linear_sum_assignment
    from .symmetry import reflect
    records = [{**item, "side": _side(item["point_mm"], plane), "paired_id": None, "error_mm": None} for item in items]
    for group in sorted({record["group"] for record in records}):
        members = [record for record in records if record["group"] == group]
        for record in members:
            if record["side"] == "on_plane":
                record.update(paired_id=record["id"], error_mm=2*abs(float(np.dot(record["point_mm"], plane["normal"])-plane["offset_mm"])))
        left = [record for record in members if record["side"] == "negative"]
        right = [record for record in members if record["side"] == "positive"]
        if not left or not right:
            continue
        reflected = reflect([r["point_mm"] for r in left], plane["normal"], plane["offset_mm"])
        distances = np.linalg.norm(reflected[:, None, :] - np.asarray([r["point_mm"] for r in right])[None, :, :], axis=2)
        a, b = linear_sum_assignment(distances)
        for i, j in zip(a, b):
            error = float(distances[i, j])
            left[i].update(paired_id=right[j]["id"], error_mm=error)
            right[j].update(paired_id=left[i]["id"], error_mm=error)
    return {**_summary(records, tolerance), "method": "one-to-one reflected explicit locations within symmetry_group, falling back to role",
            "limitation": "These locations are authored annotations, not centers extracted or proven from the B-rep. No location is inferred from a region box; unlocated features remain unassessed."}


def periodic_seams(shape, plane, options, budget):
    from build123d import Vertex
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from scipy.optimize import linear_sum_assignment
    from .symmetry import reflect, per_side

    faces, seams, points = [], [], []
    for index, face in enumerate(shape.faces()):
        surface = BRepAdaptor_Surface(face.wrapped)
        periodic = (surface.IsUPeriodic(), surface.IsVPeriodic())
        if not any(periodic):
            continue
        kind = str(surface.GetType()).split(".")[-1]
        face_info = {"face_index": index, "surface_type": kind, "u_periodic": periodic[0], "v_periodic": periodic[1], "seam_count": 0}
        for edge in face.edges():
            if not BRep_Tool.IsClosed_s(edge.wrapped, face.wrapped):
                continue
            count = max(5, int(np.ceil(edge.length/options.edge_step_mm))+1)
            if sum(len(p) for p in points)+count > budget:
                raise ValueError("CAD periodic-seam samples exceed the symmetry budget; increase it or use a larger edge step")
            sample = np.asarray([tuple(edge.position_at(float(t))) for t in np.linspace(0, 1, count)])
            points.append(sample)
            seams.append({"edge": edge, "group": (kind, *periodic), "face_index": index,
                          "id": f"face-{index}-seam-{face_info['seam_count']}", "side": _side(sample.mean(axis=0), plane)})
            face_info["seam_count"] += 1
        faces.append(face_info)
    records = [{key: value for key, value in seam.items() if key not in ("edge", "group")} | {"paired_id": None, "error_mm": None} for seam in seams]
    sample_errors = [None]*len(seams)
    for group in sorted({seam["group"] for seam in seams}):
        selected = [i for i, seam in enumerate(seams) if seam["group"] == group]
        # The assigned counterpart must itself be a seam on a compatible periodic face,
        # rather than any material boundary that happens to contain the reflected point.
        distances = {}
        for i in selected:
            reflected = reflect(points[i], plane["normal"], plane["offset_mm"])
            for j in selected:
                distances[i, j] = np.asarray([seams[j]["edge"].distance_to(Vertex(*p)) for p in reflected])
        costs = np.asarray([[max(distances[i, j].max(), distances[j, i].max()) for j in selected] for i in selected])
        rows, columns = linear_sum_assignment(costs)
        for row, column in zip(rows, columns):
            i, j = selected[row], selected[column]
            error = float(costs[row, column])
            if not np.isfinite(error):
                raise ValueError("the CAD kernel returned a non-finite periodic seam distance")
            records[i].update(paired_id=records[j]["id"], error_mm=error)
            sample_errors[i] = distances[i, j]
    summary = _summary(records, options.cad_tolerance_mm)
    return {**summary, "periodic_face_count": len(faces), "periodic_faces": faces,
            "sample_count": sum(len(p) for p in points),
            "sample_sides": per_side(np.vstack(points), np.concatenate(sample_errors), np.asarray(plane["normal"]), plane["offset_mm"], options.cad_tolerance_mm) if points else {},
            "method": "one-to-one periodic seam curves reflected to compatible finished B-rep seam curves",
            "limitation": "Seams are parameterization topology, not necessarily material boundaries. A seam deviation can occur on physically symmetric material. Samples do not prove periodic continuity or equality of control points."}


def category_identity(identity, category):
    token = hashlib.sha256(json.dumps([identity, category], sort_keys=True).encode()).hexdigest()
    return {**identity, "category": category, "token": token}
