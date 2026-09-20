"""Measure an editable part against the mesh it is reconstructing.

The comparison is bidirectional because each surface can contain a feature the other does not. Distances are unsigned so open references work, and the result never pretends to know whether a difference is extra or missing material.

The target is declared once in the part card. New projects use an explicit unit, acceptance tolerance, and rigid transform while old string declarations continue to load:

    ```toml
    target = { file = "scans/bracket.stl", units = "mm", tolerance_mm = 0.1, transform = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] }
    ```

The transform is row-major and maps the unit-normalized target into the part's coordinate frame. A missing transform keeps the historical center alignment for compatibility; the server holds that initial alignment stable during the session and exposes it so the viewer can persist it.
"""

import json
import pathlib
import re
import shutil

import numpy as np

DEFAULT_TOLERANCE_MM = 0.1
MIN_TOLERANCE_MM = 0.001
SAMPLES = 1500
FEATURE_SAMPLES = 1000
REFINEMENT_SAMPLES = 500
MAX_EDGE = 3.0
MAX_REGIONS = 8
IDENTITY = [
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
]


def setting(settings):
    """Return a normalized target declaration, or None when the card names none."""
    raw = settings.get("target")
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = {"file": raw}
    if not isinstance(raw, dict) or not isinstance(raw.get("file"), str):
        raise ValueError(
            'target is a path in quotes, or a table: target = { file = "scans/x.stl", units = "mm", tolerance_mm = 0.1 }'
        )
    file = raw["file"].strip()
    if not file:
        raise ValueError("target.file cannot be empty")
    units = raw.get("units")
    if units is not None:
        from .scan import UNITS

        if not isinstance(units, str) or units not in UNITS:
            raise ValueError(f"target.units must be one of {', '.join(UNITS)}")
    tolerance = raw.get("tolerance_mm", DEFAULT_TOLERANCE_MM)
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise ValueError("target.tolerance_mm must be a number in millimetres")
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance < MIN_TOLERANCE_MM:
        raise ValueError(f"target.tolerance_mm must be at least {MIN_TOLERANCE_MM:g} mm")
    transform = raw.get("transform")
    if transform is not None:
        transform = _transform(transform).reshape(-1).tolist()
    return {
        "file": file,
        "units": units,
        "tolerance_mm": tolerance,
        "transform": transform,
    }


def load(root, file, units=None):
    """Load the target in millimetres, resolving relative paths against the project."""
    from . import scan

    path = pathlib.Path(file)
    if not path.is_absolute():
        path = pathlib.Path(root) / path
    return scan.load(path, units=units)


def centered_transform(part, mesh):
    """A translation that reproduces the legacy bounding-box center alignment."""
    matrix = np.eye(4)
    matrix[:3, 3] = _center(part) - _center(mesh)
    return matrix.reshape(-1).tolist()


def against(shape, mesh, tolerance_mm=DEFAULT_TOLERANCE_MM, transform=None):
    """Return bidirectional deviation, tolerance coverage, and spatial samples.

    The random area samples provide unbiased coverage and percentile estimates. Vertices and the centroids of small faces are added to the sampled maximum, then nearest points found from the opposite direction are folded back into each side. That refinement makes a small pocket visible even when its footprint is too small for a random sample to land inside.
    """
    tolerance_mm = float(tolerance_mm)
    if not np.isfinite(tolerance_mm) or tolerance_mm < MIN_TOLERANCE_MM:
        raise ValueError(f"comparison tolerance must be at least {MIN_TOLERANCE_MM:g} mm")
    part = _part_mesh(shape, tolerance_mm)
    if not len(part.faces):
        raise ValueError("the part has no surface to compare")
    matrix = _transform(transform) if transform is not None else np.asarray(centered_transform(part, mesh)).reshape(4, 4)
    moved = mesh.copy()
    moved.apply_transform(matrix)

    part_points, part_faces, part_stat_count = _sample(part)
    target_points, target_faces, target_stat_count = _sample(moved)
    part_surface = _surface(part)
    target_surface = _surface(moved)
    part_d, on_target = _to_surface(part_points, target_surface, closest=True)
    target_d, on_part = _to_surface(target_points, part_surface, closest=True)

    # A point already known to lie on one surface is a useful directed sample of that surface. Folding nearest points back catches a local recess or boss from both directions without pretending the additional points are area-weighted statistics.
    on_part = _refinement(on_part, target_d, tolerance_mm)
    on_target = _refinement(on_target, part_d, tolerance_mm)
    part_refined_d = _to_surface(on_part, target_surface)
    target_refined_d = _to_surface(on_target, part_surface)
    part_all_points = np.concatenate((part_points, on_part))
    target_all_points = np.concatenate((target_points, on_target))
    part_all_d = np.concatenate((part_d, part_refined_d))
    target_all_d = np.concatenate((target_d, target_refined_d))
    part_all_faces = np.concatenate((part_faces, np.full(len(on_part), -1, dtype=int)))
    target_all_faces = np.concatenate((target_faces, np.full(len(on_target), -1, dtype=int)))

    part_regions = _regions(part_all_points, part_all_d, tolerance_mm, "part_to_target")
    target_regions = _regions(target_all_points, target_all_d, tolerance_mm, "target_to_part")
    detected = bool(np.any(part_all_d > tolerance_mm) or np.any(target_all_d > tolerance_mm))
    return {
        "transform": [float(v) for v in matrix.reshape(-1)],
        "offset": [round(float(v), 6) for v in matrix[:3, 3]],
        "transform_frame": "unit-normalized target coordinates to part coordinates, both in millimetres",
        "tolerance_mm": tolerance_mm,
        "part": _stats(part_d[:part_stat_count], part_all_d, tolerance_mm),
        "target": _stats(target_d[:target_stat_count], target_all_d, tolerance_mm),
        "detected_above_tolerance": detected,
        "worst_regions": sorted(
            [*part_regions, *target_regions],
            key=lambda region: region["peak_deviation_mm"],
            reverse=True,
        )[:MAX_REGIONS],
        "samples": {
            "part": _spatial(part_all_points, part_all_d, part_all_faces),
            "target": _spatial(target_all_points, target_all_d, target_all_faces),
        },
        "sample_count": {
            "part": len(part_all_points),
            "target": len(target_all_points),
            "coverage_each": SAMPLES,
            "part_coverage": part_stat_count,
            "target_coverage": target_stat_count,
            "part_feature_and_refinement": len(part_all_points) - part_stat_count,
            "target_feature_and_refinement": len(target_all_points) - target_stat_count,
        },
    }


def report(name, file, metrics, unit, source):
    """Format comparison facts for the CLI; the doctrine decides what they mean."""
    from .scan import LONG

    said = {
        "file": "declared by the file format",
        "argument": "said by the card or --units",
        "guess": "guessed from the file's span; wrong means set units",
    }
    alignment = {
        "stored": "stored alignment",
        "identity": "identity alignment",
        "center": "center alignment",
        "auto": "center alignment",
    }.get(metrics.get("alignment"), f"{metrics.get('alignment')} alignment")
    lines = [
        f"  {name} against {file}  (read as {LONG[unit]}, {said[source]})",
        f"      {alignment}; target-to-part offset {metrics['offset']} mm; tolerance {metrics['tolerance_mm']:g} mm",
        f"      applied row-major transform {metrics['transform']} maps unit-normalized target mm into part mm",
        f"      part to target: {_line(metrics['part'])}",
        f"      target to part: {_line(metrics['target'])}",
    ]
    if metrics.get("detected_above_tolerance"):
        peak = max(metrics["part"]["sampled_max"], metrics["target"]["sampled_max"])
        lines.insert(3, f"      DEVIATIONS DETECTED ABOVE TOLERANCE; sampled peak {peak:.3f} mm")
        for region in metrics.get("worst_regions", [])[:3]:
            position = ", ".join(f"{value:.2f}" for value in region["position_mm"])
            extent = " x ".join(f"{value:.2f}" for value in region["extent_mm"])
            direction = region["direction"].replace("_", " ")
            lines.append(
                f"      worst {direction} near ({position}) mm; extent {extent} mm; "
                f"sampled peak {region['peak_deviation_mm']:.3f} mm, "
                f"excess {region['excess_mm']:.3f} mm, {region['sample_count']} samples"
            )
    else:
        lines.insert(3, "      no sampled deviation above tolerance detected")
    return lines


def update_card(part_path, **changes):
    """Persist target units, tolerance, or transform in the card's TOML block."""
    from . import checks

    path = pathlib.Path(part_path)
    card = path.with_suffix(".md")
    if not card.is_file():
        raise ValueError(f"{path.stem} has no card to store target settings")
    current = setting(checks.settings(path))
    if current is None:
        raise ValueError(f"{path.stem} has no target in its card")
    allowed = {"units", "tolerance_mm", "transform"}
    unknown = set(changes) - allowed
    if unknown:
        raise ValueError(f"unknown target setting: {', '.join(sorted(unknown))}")
    normalized = setting({"target": {**current, **changes}})
    text = card.read_text(encoding="utf-8")
    opening = f"```{checks.CARD_SETTINGS}"
    if opening not in text:
        raise ValueError(f"{card.name} has no ```toml settings block")
    before, rest = text.split(opening, 1)
    block, after = rest.split("```", 1)
    block = _replace_target(block, normalized, require=True)
    card.write_text(before + opening + block + "```" + after, encoding="utf-8")
    return [name for name in ("units", "tolerance_mm", "transform") if name in changes]


def attach_reference(part_path, relative_file, units=None, tolerance_mm=DEFAULT_TOLERANCE_MM):
    """Add or replace a card target after its portable reference file is written."""
    from . import checks

    part = pathlib.Path(part_path)
    raw_reference = str(relative_file)
    reference = pathlib.PurePosixPath(raw_reference.replace("\\", "/"))
    windows_drive = pathlib.PureWindowsPath(raw_reference).drive
    if (
        reference.is_absolute()
        or windows_drive
        or ".." in reference.parts
        or not reference.name
    ):
        raise ValueError("target file must be a portable path inside the project")
    target = setting(
        {
            "target": {
                "file": str(reference),
                "units": units,
                "tolerance_mm": tolerance_mm,
                "transform": None,
            }
        }
    )
    card = part.with_suffix(".md")
    text = card.read_text(encoding="utf-8") if card.is_file() else f"# {part.stem}\n"
    opening = f"```{checks.CARD_SETTINGS}"
    if opening in text:
        before, rest = text.split(opening, 1)
        if "```" not in rest:
            raise ValueError(f"{card.name} has an unclosed ```toml settings block")
        block, after = rest.split("```", 1)
        block = _replace_target(block, target, require=False)
        text = before + opening + block + "```" + after
    else:
        text = text.rstrip() + "\n\n```toml\n" + _format_setting(target) + "\n```\n"
    card.write_text(text, encoding="utf-8")
    return ["file", "units", "tolerance_mm", "transform"]


def copy_reference(part_path, source_path, units=None, tolerance_mm=DEFAULT_TOLERANCE_MM):
    """Copy a local mesh into the project and attach the portable card reference."""
    part = pathlib.Path(part_path)
    source = pathlib.Path(source_path)
    if not source.is_file():
        raise ValueError(f"no file at {source}")
    root = part.parent.parent
    scans = root / "scans"
    scans.mkdir(exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", source.name).strip(".-") or "reference.stl"
    destination = scans / f"{part.stem}-{safe}"
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    relative = destination.relative_to(root).as_posix()
    attach_reference(part, relative, units=units, tolerance_mm=tolerance_mm)
    return relative


def remove_reference(part_path):
    """Remove a card target declaration while preserving unrelated settings."""
    from . import checks

    part = pathlib.Path(part_path)
    card = part.with_suffix(".md")
    if not card.is_file():
        raise ValueError(f"{part.stem} has no card to remove a target from")
    text = card.read_text(encoding="utf-8")
    opening = f"```{checks.CARD_SETTINGS}"
    if opening not in text:
        return []
    before, rest = text.split(opening, 1)
    if "```" not in rest:
        raise ValueError(f"{card.name} has an unclosed ```toml settings block")
    block, after = rest.split("```", 1)
    updated = _remove_target(block)
    if updated == block:
        return []
    if updated.strip():
        text = before + opening + updated + "```" + after
    else:
        text = before.rstrip() + "\n" + after.lstrip("\n")
    card.write_text(text, encoding="utf-8")
    return ["target"]


def _replace_target(block, target, require):
    inline = re.compile(r"(?m)^[ \t]*target[ \t]*=.*$")
    table = re.compile(r"(?m)^[ \t]*\[target\][ \t]*(?:#.*)?$")
    first_table = re.search(r"(?m)^[ \t]*\[[^\n]+\][ \t]*(?:#.*)?$", block)
    root_end = first_table.start() if first_table else len(block)
    inline_matches = list(inline.finditer(block, 0, root_end))
    table_matches = list(table.finditer(block))
    count = len(inline_matches) + len(table_matches)
    if count > 1 or (require and count != 1):
        raise ValueError("the settings block must declare exactly one target")
    if inline_matches:
        return inline.sub(lambda _: _format_setting(target), block, count=1)
    if table_matches:
        start = table_matches[0].start()
        following = re.search(r"(?m)^[ \t]*\[[^\n]+\][ \t]*(?:#.*)?$", block[table_matches[0].end():])
        end = table_matches[0].end() + following.start() if following else len(block)
        replacement = _format_table(target)
        return block[:start] + replacement + block[end:]
    formatted = _format_setting(target)
    if first_table:
        prefix = block[:root_end]
        if prefix and not prefix.endswith("\n"):
            prefix += "\n"
        return prefix + formatted + "\n\n" + block[root_end:].lstrip("\n")
    prefix = "" if block.endswith("\n") else "\n"
    return block + prefix + formatted + "\n"


def _remove_target(block):
    inline = re.compile(r"(?m)^[ \t]*target[ \t]*=.*\n?")
    table = re.compile(r"(?m)^[ \t]*\[target\][ \t]*(?:#.*)?$")
    first_table = re.search(r"(?m)^[ \t]*\[[^\n]+\][ \t]*(?:#.*)?$", block)
    root_end = first_table.start() if first_table else len(block)
    inline_matches = list(inline.finditer(block, 0, root_end))
    table_matches = list(table.finditer(block))
    if len(inline_matches) + len(table_matches) > 1:
        raise ValueError("the settings block must declare at most one target")
    if inline_matches:
        return inline.sub("", block, count=1)
    if table_matches:
        start = table_matches[0].start()
        following = re.search(r"(?m)^[ \t]*\[[^\n]+\][ \t]*(?:#.*)?$", block[table_matches[0].end():])
        end = table_matches[0].end() + following.start() if following else len(block)
        return block[:start] + block[end:]
    return block


def _format_setting(target):
    fields = [f"file = {_toml_string(target['file'])}"]
    if target.get("units") is not None:
        fields.append(f"units = {_toml_string(target['units'])}")
    fields.append(f"tolerance_mm = {float(target['tolerance_mm'])!r}")
    if target.get("transform") is not None:
        matrix = ", ".join(repr(float(v)) for v in target["transform"])
        fields.append(f"transform = [{matrix}]")
    return "target = { " + ", ".join(fields) + " }"


def _format_table(target):
    fields = ["[target]", f"file = {_toml_string(target['file'])}"]
    if target.get("units") is not None:
        fields.append(f"units = {_toml_string(target['units'])}")
    fields.append(f"tolerance_mm = {float(target['tolerance_mm'])!r}")
    if target.get("transform") is not None:
        matrix = ", ".join(repr(float(v)) for v in target["transform"])
        fields.append(f"transform = [{matrix}]")
    return "\n".join(fields) + "\n"


def _toml_string(value):
    return json.dumps(value, ensure_ascii=False)


def _line(stats):
    return f"estimated sampled coverage {stats['within_tolerance'] * 100:.1f}% within tolerance, sampled max {stats['sampled_max']:.3f} mm, p95 {stats['p95']:.3f} mm, excess max {stats['excess_max']:.3f} mm"


def _stats(area_d, all_d, tolerance):
    excess = np.maximum(all_d - tolerance, 0.0)
    area_excess = np.maximum(area_d - tolerance, 0.0)
    sampled_max = float(all_d.max())
    return {
        "sampled_max": round(sampled_max, 5),
        "max": round(sampled_max, 5),
        "median": round(float(np.median(area_d)), 5),
        "p95": round(float(np.percentile(area_d, 95)), 5),
        "within_tolerance": round(float(np.mean(area_d <= tolerance)), 6),
        "excess_max": round(float(excess.max()), 5),
        "excess_p95": round(float(np.percentile(area_excess, 95)), 5),
    }


def _spatial(points, distances, faces):
    return {
        "points": np.round(points, 4).tolist(),
        "distances": np.round(distances, 5).tolist(),
        "face_indices": faces.tolist(),
    }


def _regions(points, distances, tolerance, direction):
    """Cluster above-tolerance evidence into bounded regions an agent can inspect."""
    from scipy.spatial import cKDTree

    indices = np.flatnonzero(distances > tolerance)
    if not len(indices):
        return []
    selected = np.asarray(points)[indices]
    selected_d = np.asarray(distances)[indices]
    span = float(np.linalg.norm(np.ptp(selected, axis=0))) if len(selected) > 1 else 0.0
    # Comparison sampling is capped to MAX_EDGE spacing. Use that same physical
    # scale when grouping evidence so one small notch becomes one useful region
    # instead of a page of isolated one-sample "regions".
    radius = max(MAX_EDGE, tolerance * 4.0, span * 0.04)
    parent = np.arange(len(selected))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    for a, b in cKDTree(selected).query_pairs(radius):
        union(a, b)
    groups = {}
    for i in range(len(selected)):
        groups.setdefault(find(i), []).append(i)
    regions = []
    for members in groups.values():
        member = np.asarray(members, dtype=int)
        cloud = selected[member]
        values = selected_d[member]
        peak = int(np.argmax(values))
        low, high = cloud.min(axis=0), cloud.max(axis=0)
        regions.append(
            {
                "direction": direction,
                "position_mm": [float(v) for v in cloud[peak]],
                "bounds_mm": {
                    "min": [float(v) for v in low],
                    "max": [float(v) for v in high],
                },
                "extent_mm": [float(v) for v in high - low],
                "peak_deviation_mm": float(values[peak]),
                "excess_mm": float(values[peak] - tolerance),
                "sample_count": len(member),
            }
        )
    return sorted(regions, key=lambda region: region["peak_deviation_mm"], reverse=True)[:MAX_REGIONS]


def _transform(raw):
    try:
        matrix = np.asarray(raw, dtype=float).reshape(4, 4)
    except (TypeError, ValueError) as exc:
        raise ValueError("target.transform must contain 16 numbers in row-major order") from exc
    if not np.isfinite(matrix).all():
        raise ValueError("target.transform must contain only finite numbers")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError("target.transform must be rigid and end with [0, 0, 0, 1]")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError("target.transform must contain a rigid rotation without scale or reflection")
    return matrix


def _part_mesh(shape, tolerance_mm):
    """Tessellate with an absolute error budget below the acceptance tolerance."""
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRepTools import BRepTools

    from . import builder

    deflection = min(0.025, tolerance_mm / 4.0)
    BRepTools.Clean_s(shape.wrapped)
    BRepMesh_IncrementalMesh(shape.wrapped, deflection, False, 0.05, True)
    return builder.to_mesh(shape, deflection)


def _center(mesh):
    return mesh.bounds.mean(axis=0)


def _sample(mesh):
    """Area samples first, then bounded feature samples for local extrema."""
    import trimesh

    points, faces = trimesh.sample.sample_surface(mesh, SAMPLES, seed=0)
    point_sets = [np.asarray(points)]
    face_sets = [np.asarray(faces, dtype=int)]
    vertices = np.unique(np.round(np.asarray(mesh.vertices), 10), axis=0)
    if len(vertices) > FEATURE_SAMPLES // 2:
        vertices = vertices[np.linspace(0, len(vertices) - 1, FEATURE_SAMPLES // 2, dtype=int)]
    point_sets.append(vertices)
    face_sets.append(np.full(len(vertices), -1, dtype=int))
    centroids = np.asarray(mesh.triangles_center)
    count = min(len(centroids), FEATURE_SAMPLES - len(vertices))
    if count:
        chosen = np.argsort(np.asarray(mesh.area_faces), kind="stable")[:count]
        point_sets.append(centroids[chosen])
        face_sets.append(chosen.astype(int))
    return np.concatenate(point_sets), np.concatenate(face_sets), SAMPLES


def _refinement(points, distances, tolerance):
    """Keep the strongest cross-direction seeds, which target local disagreements."""
    indices = np.flatnonzero(distances > tolerance)
    if len(indices) > REFINEMENT_SAMPLES:
        order = np.argsort(distances[indices], kind="stable")[-REFINEMENT_SAMPLES:]
        indices = indices[order]
    return points[indices]


def _surface(mesh):
    """Prepare the triangle surface once for several directed queries."""
    import trimesh
    from scipy.spatial import cKDTree

    vertices, faces = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge=MAX_EDGE)
    triangles = vertices[faces]
    corners = triangles.reshape(-1, 3)
    return triangles, cKDTree(corners)


def _to_surface(points, surface, closest=False):
    """Return exact unsigned point-to-triangle distances without an rtree dependency."""
    import trimesh

    triangles, tree = surface
    nearest, _ = tree.query(points)
    out = np.empty(len(points))
    nearest_points = np.empty_like(points, dtype=float) if closest else None
    for i, (point, ball) in enumerate(zip(points, tree.query_ball_point(points, nearest + MAX_EDGE + 1e-6))):
        near = triangles[np.unique(np.asarray(ball) // 3)]
        candidates = trimesh.triangles.closest_point(near, np.broadcast_to(point, near.shape[:1] + (3,)))
        distances = np.linalg.norm(candidates - point, axis=1)
        best = int(np.argmin(distances))
        out[i] = distances[best]
        if nearest_points is not None:
            nearest_points[i] = candidates[best]
    if closest:
        return out, nearest_points
    return out
