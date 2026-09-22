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
    for key in ("role", "configuration", "orientation", "notes", "symmetry_group"):
        if key in raw:
            result[key] = _text(raw[key], f"feature {key}")
    centers = [vector(raw[key], "CAD center in part mm").tolist() for key in ("center_mm", "point_mm") if key in raw]
    if centers:
        if any(center != centers[0] for center in centers):
            raise ValueError("center_mm and point_mm disagree; keep one explicit CAD center")
        result["center_mm"] = centers[0]
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
    # Older producers called this feature_scale_mm. Normalize that spelling on read;
    # absence stays absent so existing reviews do not expire just from upgrading.
    sizes = [raw[key] for key in ("feature_size_mm", "feature_scale_mm") if key in raw]
    if sizes:
        try:
            values = [float(value) for value in sizes]
        except (TypeError, ValueError) as exc:
            raise ValueError("feature_size_mm must be a positive finite size in mm") from exc
        if any(isinstance(value, bool) for value in sizes) or any(not np.isfinite(value) or value <= 0 for value in values):
            raise ValueError("feature_size_mm must be a positive finite size in mm")
        if len(set(values)) != 1:
            raise ValueError("feature_size_mm and legacy feature_scale_mm disagree; keep one size")
        result["feature_size_mm"] = values[0]
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
        if "source" in review:
            if review["source"] not in ("preview","verified"):
                raise ValueError("feature review source must be preview or verified")
            result["review"]["source"]=review["source"]
        if "verification_request_id" in review:
            result["review"]["verification_request_id"]=_text(review["verification_request_id"],"verification request ID",128)
        if review.get("source")=="verified" and not result["review"].get("verification_request_id"):
            raise ValueError("verified feature review needs its verification request ID")
        if "provenance" in review:
            if not isinstance(review["provenance"],dict):
                raise ValueError("feature review provenance must be an object")
            try: encoded=json.dumps(review["provenance"],allow_nan=False)
            except (TypeError,ValueError) as exc: raise ValueError("feature review provenance needs finite JSON values") from exc
            if len(encoded)>32768: raise ValueError("feature review provenance exceeds 32 KiB")
            # Cards are TOML, which has no null. An absent optional bound still means unknown.
            def without_null(value):
                if isinstance(value,dict): return {key:without_null(item) for key,item in value.items() if item is not None}
                if isinstance(value,list): return [without_null(item) for item in value if item is not None]
                return value
            result["review"]["provenance"]=without_null(json.loads(encoded))
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


def portable_shape_identity(shape):
    """Fingerprint built geometry without OCCT serialization or tessellation order."""
    # OCCT's derived B-spline measures can differ by a few last-place digits across
    # platforms. A portable identity should ignore that numerical noise while still
    # expiring for changes far below normal manufacturing tolerances.
    resolution = {"linear_mm": 0.001, "area_mm2": 0.01, "volume_mm3": 0.1}

    def number(value, unit="linear_mm"):
        return round(float(value) / resolution[unit])

    def point(value):
        return [number(value.X), number(value.Y), number(value.Z)]

    def bounds(value):
        box = value.bounding_box()
        return [point(box.min), point(box.max)]

    def record(value, measure, unit):
        return {
            "type": str(value.geom_type),
            measure: number(getattr(value, measure), unit),
            "center": point(value.center()),
            "bounds": bounds(value),
        }

    vertices = sorted(point(vertex.center()) for vertex in shape.vertices())
    edges = []
    for edge in shape.edges():
        item = record(edge, "length", "linear_mm")
        item.pop("bounds")
        item["vertices"] = sorted(point(vertex.center()) for vertex in edge.vertices())
        item["samples"] = sorted(point(edge.position_at(fraction)) for fraction in (0.0, 0.25, 0.5, 0.75, 1.0))
        edges.append(item)
    faces = []
    for face in shape.faces():
        item = record(face, "area", "area_mm2")
        item["edges"] = sorted(number(edge.length, "linear_mm") for edge in face.edges())
        faces.append(item)
    solids = []
    for solid in shape.solids():
        item = record(solid, "volume", "volume_mm3")
        item["area"] = number(solid.area, "area_mm2")
        item["faces"] = len(solid.faces())
        solids.append(item)
    document = {
        "resolution": resolution,
        "bounds": bounds(shape),
        "vertices": vertices,
        "edges": sorted(edges, key=lambda value: json.dumps(value, sort_keys=True)),
        "faces": sorted(faces, key=lambda value: json.dumps(value, sort_keys=True)),
        "solids": sorted(solids, key=lambda value: json.dumps(value, sort_keys=True)),
    }
    return _digest(document)


def region_evidence(shape_id, reference_id, transform, configuration, region):
    feature = region["feature"]
    contract = {**feature, "selection": {key: region[key] for key in ("name", "bounds_mm", "component") if key in region}}
    identity = evidence_identity(shape_id, reference_id, transform, configuration, contract)
    return {"id": feature["id"], "name": region["name"], "feature": feature, **freshness(feature, identity)}


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


def smallest_feature_size(regions, requested=None):
    sizes = [region["feature"]["feature_size_mm"] for region in regions
             if region.get("feature", {}).get("feature_size_mm") is not None]
    if requested is not None:
        sizes.append(feature_record({"id": "requested", "feature_size_mm": requested})["feature_size_mm"])
    return min(sizes) if sizes else None


def section_payload(result):
    """A self-contained snapshot; freshness is tied to the included input identity."""
    return {"schema_version": 1, "kind": "local-section-evidence", **result,
            "limitation": "Contours intersect tessellated surfaces, not exact B-rep curves. Open contours remain open; neither mesh deflection nor these sections certifies physical fit."}


def section_svg(result, station):
    """Render one station without filling open loops or hiding its mesh provenance."""
    from xml.etree.ElementTree import Element, SubElement, tostring

    if isinstance(station, bool) or not isinstance(station, int) or not 0 <= station < len(result.get("cad", [])):
        raise ValueError("choose a saved local section station before exporting SVG")
    cuts = [result["cad"][station], result["reference"][station]]
    payload = section_payload({**result, "cad": [cuts[0]], "reference": [cuts[1]]})
    svg = Element("svg", xmlns="http://www.w3.org/2000/svg", width="900", height="640", viewBox="0 0 900 640")
    SubElement(svg, "title").text = f'{result["name"]}: {cuts[0]["name"]} at {cuts[0]["offset_mm"]:g} mm'
    SubElement(svg, "metadata").text = json.dumps(payload, sort_keys=True, allow_nan=False)
    SubElement(svg, "rect", width="900", height="640", fill="#16181d")
    def text(y, value, size="15"):
        SubElement(svg, "text", x="24", y=str(y), fill="#ddd", **{"font-family": "sans-serif", "font-size": size}).text = value
    text(30, f'{result["name"]} / {cuts[0]["name"]} / offset {cuts[0]["offset_mm"]:g} mm', "18")
    provenance = result.get("provenance", {})
    text(56, provenance.get("method", result.get("method", "Mesh contours")))
    size = result.get("feature", {}).get("feature_size_mm")
    text(79, f'Feature size: {size:g} mm' if size is not None else "Feature size: unspecified")
    deflection = provenance.get("absolute_deflection_mm")
    text(102, f'Requested deflection: {deflection:g} mm; achieved error unknown' if deflection is not None else "Display mesh accuracy in mm is unknown")
    points = [point for cut in cuts for loop in cut.get("loops", []) for point in loop["points_mm"]]
    if points:
        data = np.asarray(points, dtype=float)
        low, high = data.min(axis=0), data.max(axis=0)
        scale = min(840 / max(.01, high[0]-low[0]), 360 / max(.01, high[1]-low[1]))
        center = (low+high)/2
        for cut, color in zip(cuts, ("#f0c274", "#62c7ef")):
            for loop in cut.get("loops", []):
                path = loop["points_mm"]
                if loop["closed"] and path:
                    path = [*path, path[0]]
                coordinates = " ".join(f'{450+(point[0]-center[0])*scale:.6f},{320-(point[1]-center[1])*scale:.6f}' for point in path)
                SubElement(svg, "polyline", points=coordinates, fill="none", stroke=color, **{"stroke-width": "1.5"})
        text(538, f'u right; v up; {100/scale:.6g} mm per 100 SVG units. CAD amber; reference blue.')
    else:
        text(320, "This station misses both surfaces.")
    text(568, "Tessellated contours, not exact B-rep curves. Open contours remain open.")
    text(593, "No physical-fit certification. The JSON metadata retains frame, identity and full precision.", "13")
    text(617, f'Evidence: {result["identity"]["token"]}', "11")
    return tostring(svg, encoding="unicode", xml_declaration=True)


def section_stem(configuration, feature_id):
    label = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{configuration}.{feature_id}")
    return f"{label[:120]}-{_digest([configuration, feature_id])[:8]}"


def write_section_exports(directory, configuration, results):
    from pathlib import Path

    output = Path(directory)
    if not any(result.get("cad") for result in results):
        raise ValueError("section export needs a saved feature with local section stations")
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for result in results:
        if not result.get("cad"):
            continue
        stem = section_stem(configuration, result["id"])
        path = output / f"{stem}.sections.json"
        path.write_text(json.dumps(section_payload(result), indent=2, allow_nan=False)+"\n", encoding="utf-8")
        paths.append(str(path))
        for station in range(len(result["cad"])):
            path = output / f"{stem}.station-{station+1:02}.svg"
            path.write_text(section_svg(result, station), encoding="utf-8")
            paths.append(str(path))
    return paths
