"""What a mesh measured, so a part can be modelled against something that exists.

A photo names a shape but carries no millimetres. A mesh carries them, and two kinds
arrive: a ten-second phone scan (Scaniverse, Polycam, and their kind), and a file
downloaded off a model site. This module reads either the same way, because they are
the same question at this stage: overall size in mm with the file's units made
explicit, and a cross-section sliced into a polyline short enough to sketch against.

Where they differ is what the numbers are worth, and that is not visible in the geometry. A phone scan is reference geometry and its fits stay provisional until a coupon proves them. A downloaded design can carry precise triangle coordinates while still approximating its original curves with flat polygons. Nothing in the mesh says which one it is, so this module never guesses. The 3DBenchy, a designed model, scores like a capture on every statistic worth computing. Provenance is something the user said and the agent knows, so the judgement lives in the skill, and this reports facts.

The units question is the dangerous one. Scan apps export metres and slicers export
millimetres, and a mesh 0.3 units across does not say which it is. Guessing wrong is a
part a thousand times off that still builds, so the guess is stated in the report and
overridable rather than silent.
"""

import pathlib
import re
from collections import defaultdict

import numpy as np

UNITS = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4}
LONG = {"mm": "millimetres", "cm": "centimetres", "m": "metres", "in": "inches"}
DECLARED = {
    "mm": "mm",
    "millimeter": "mm",
    "millimeters": "mm",
    "cm": "cm",
    "centimeter": "cm",
    "centimeters": "cm",
    "m": "m",
    "meter": "m",
    "meters": "m",
    "in": "in",
    "inch": "in",
    "inches": "in",
}

# Under this many file units across, a mesh is read as metres: nothing a phone can
# scan is under a centimetre, and metres are what the scan apps export.
METRES_BELOW = 10.0

# The render command's grammar for a cut, reused so an agent learns it once:
# an axis, optionally with a fraction of the span or an absolute millimetre.
CUT = re.compile(r"^[xyz](:-?\d*\.?\d+(mm)?)?$")

# Plane intersection on a scan yields confetti along with the profile: pinholes and
# fold-overs each shed a chain a few segments long. Anything shorter than this
# fraction of the longest chain is reported as a count instead of listed.
FRAGMENT = 0.02


def load(path, units=None):
    """The mesh scaled to millimetres, its input unit, and that unit's source."""
    import trimesh

    path = pathlib.Path(path)
    if not path.is_file():
        raise ValueError(f"no file at {path}")
    if path.suffix.lower() == ".3mf":
        # As common as STL on the model sites, and trimesh reads it only with networkx
        # installed. "No module named 'networkx'" is not something a user can act on,
        # and a dependency is not worth one format the slicer already converts.
        raise ValueError(
            f"{path.name} is a 3MF, which nurb does not read. Open it in your slicer "
            f"and export the plate as STL, then run this on that file"
        )
    try:
        mesh = trimesh.load(str(path), force="mesh")
    except Exception as exc:
        raise ValueError(f"{path.name}: {exc}") from exc
    if not hasattr(mesh, "faces") or len(mesh.faces) == 0:
        # The likeliest way here is a gaussian-splat export, which is what some scan
        # apps offer first and which carries points with no surfaces between them.
        raise ValueError(
            f"{path.name} has no triangles to measure, only points. If this came "
            f"from a scan app, export a mesh format (OBJ, STL or GLB) instead. If "
            f"the app only offers splat or point-cloud formats, the scan itself "
            f"was captured as a splat, and the object needs a quick rescan in the "
            f"app's mesh mode"
        )
    # GLB/glTF defines metres and trimesh carries that declaration in `units`.
    # Prefer any declaration over a size guess: an area scan can legitimately be
    # more than ten metres across, where the heuristic would be wrong by 1,000x.
    declared = DECLARED.get(str(mesh.units).lower()) if mesh.units else None
    if units:
        unit, source = units, "argument"
    elif declared:
        unit, source = declared, "file"
    else:
        unit = "m" if float(mesh.extents.max()) < METRES_BELOW else "mm"
        source = "guess"
    if UNITS[unit] != 1.0:
        mesh.apply_scale(UNITS[unit])
    # Textured GLBs commonly split a physical vertex at normal and UV seams. The
    # geometry is still closed, but the split topology makes `is_watertight` lie.
    # Welding by position changes no measurement and gives topology its true shape.
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    return mesh, unit, source


def report(path, mesh, unit, source):
    path = pathlib.Path(path)
    # .4g, not .1f: a mis-unit mesh read as mm can be 0.04mm across, and a size
    # line that rounds that to 0.0 states a falsehood right where units go wrong.
    size = " x ".join(f"{v:.4g}" for v in mesh.extents)
    surface = "watertight" if mesh.is_watertight else "open surface"
    low, high = mesh.bounds
    lines = [f"  {path.name}  {len(mesh.faces):,} triangles, {surface}, {size} mm"]
    if source == "guess":
        span = float(mesh.extents.max()) / UNITS[unit]
        # No longer says "which is what phone scan apps export": that is why the
        # threshold exists, but stating it as the file's origin reads as a verdict on
        # provenance, and a downloaded part is exact however this line is worded.
        lines.append(
            f"      read as {LONG[unit]}: the file spans {span:.3g} units, and only a "
            f"mesh under {METRES_BELOW:.0f} units across is read as metres. "
            f"Wrong? --units says so"
        )
    elif source == "file":
        lines.append(f"      read as {LONG[unit]} (declared by the file format)")
    else:
        lines.append(f"      read as {LONG[unit]} (--units)")
    lines.append(f"      reconstruction bounds: x {low[0]:.6g} to {high[0]:.6g}, y {low[1]:.6g} to {high[1]:.6g}, z {low[2]:.6g} to {high[2]:.6g} mm; center {tuple(float(v) for v in mesh.bounds.mean(axis=0))}")
    lines.append("      comparison reference: usable with nurb compare even when solid conversion is unavailable")
    lines.append(f"      optional faceted solid: {_solid_line(path, mesh, unit, source)}")
    return lines


def structured(path, mesh, unit, source, sections=()):
    """Complete machine-readable measurements in the mesh's millimetre frame."""
    path = pathlib.Path(path)
    low, high = mesh.bounds
    return {
        "schema_version": 1,
        "file": str(path.resolve()),
        "units": {
            "input": unit,
            "source": source,
            "scale_to_mm": UNITS[unit],
        },
        "mesh": {
            "triangles": int(len(mesh.faces)),
            "watertight": bool(mesh.is_watertight),
            "surface_area_mm2": float(mesh.area),
            "volume_mm3": float(abs(mesh.volume)) if mesh.is_watertight else None,
            "coordinate_frame": "source mesh coordinates scaled to millimetres",
            "origin_mm": [0.0, 0.0, 0.0],
            "bounds_mm": {
                "min": [float(v) for v in low],
                "max": [float(v) for v in high],
            },
            "center_mm": [float(v) for v in mesh.bounds.mean(axis=0)],
            "extents_mm": [float(v) for v in mesh.extents],
            "largest_planar_region_fit": _largest_planar_region_fit(mesh),
        },
        "comparison_reference": {
            "usable": True,
            "note": "The mesh remains a comparison reference whether or not it converts to a solid.",
        },
        "solid_conversion": _solid_facts(path, mesh, unit, source),
        "sections": [_section_structured(cut) for cut in sections],
    }


def _solid_line(path, mesh, unit, source):
    """Whether this mesh can be a part's solid, answered before the agent tries it.

    The one question a mesh report has to settle, because the alternative is an agent
    guessing at `import_stl` and finding out through a refusal. Flat faces are counted
    only in the case that can import, where they are what the part would be made of;
    counting them on a 226,000-triangle download would take a third of a second to
    report a number nobody can act on.

    The call it prints carries `units` whenever this report only got the size right
    because someone passed it, because the same file imported without that argument is
    a part off by a factor of ten or a thousand that still builds.
    """
    from . import mesh as mesh_module

    problem = mesh_module.refusal(path.suffix, mesh)
    if problem:
        return (
            f"no solid from this one: it is {problem}. "
            f"Rebuild it from these measurements"
        )
    solid, problem = mesh_module.conversion(path)
    if problem:
        return (
            f"no solid from this one: it is {problem}. "
            f"Rebuild it from these measurements"
        )
    argument = f", units={unit!r}" if source == "argument" else ""
    call = f"import_stl({str(path)!r}{argument})"
    flats = len(solid.faces())
    return (
        f"{call} returns this as a solid: {flats} flat "
        f"{'face' if flats == 1 else 'faces'}. Any curve in it comes back as facets"
    )


def _solid_facts(path, mesh, unit, source):
    from . import mesh as mesh_module

    problem = mesh_module.refusal(path.suffix, mesh)
    if problem:
        return {"available": False, "reason": problem, "representation": None}
    solid, problem = mesh_module.conversion(path)
    if problem:
        return {"available": False, "reason": problem, "representation": None}
    return {
        "available": True,
        "reason": None,
        "representation": "faceted",
        "flat_face_count": len(solid.faces()),
        "warning": "Curves remain triangle-derived facets; reconstruct analytic geometry for editable CAD.",
    }


def section(mesh, spec, tolerance=0.2):
    """A cross-section as polylines an agent can sketch against, longest first.

    Returns axis and position of the cut, the names of the two in-plane axes the
    points are reported in, every profile as a measured dict, and how many short
    chains are marked as probable scan noise for the concise text report.
    """
    import trimesh

    try:
        tolerance = float(tolerance)
    except (TypeError, ValueError) as exc:
        raise ValueError("section tolerance must be a finite number at least 0 mm") from exc
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("section tolerance must be a finite number at least 0 mm")

    if not CUT.match(spec or ""):
        raise ValueError(
            f"section {spec!r} is not AXIS[:POS]. z cuts mid-mesh, z:0.7 at a "
            f"fraction of the span, z:40mm at that coordinate in the scan's own frame"
        )
    axis = "xyz".index(spec[0])
    lo, hi = float(mesh.bounds[0][axis]), float(mesh.bounds[1][axis])
    pos = (lo + hi) / 2
    if ":" in spec:
        raw = spec.split(":", 1)[1]
        pos = float(raw[:-2]) if raw.endswith("mm") else lo + float(raw) * (hi - lo)
    normal, origin = np.zeros(3), np.zeros(3)
    normal[axis], origin[axis] = 1.0, pos
    segments = trimesh.intersections.mesh_plane(
        mesh, plane_normal=normal, plane_origin=origin
    )
    keep = [i for i in range(3) if i != axis]
    chains = _chains(np.asarray(segments)[:, :, keep]) if len(segments) else []
    chains.sort(key=_length, reverse=True)
    floor = _length(chains[0]) * FRAGMENT if chains else 0.0
    profiles, skipped = [], 0
    for chain in chains:
        noise_candidate = _length(chain) < floor
        if noise_candidate:
            skipped += 1
        closed = len(chain) > 3 and _key(chain[0]) == _key(chain[-1])
        profiles.append(
            {
                "points": _simplify(chain, tolerance),
                "raw": len(chain),
                "closed": closed,
                "length": _length(chain),
                "fits": _profile_fits(chain, closed=closed),
                "noise_candidate": noise_candidate,
            }
        )
    return {
        "axis": spec[0],
        "pos": pos,
        "plane": tuple("xyz"[i] for i in keep),
        "profiles": profiles,
        "skipped": skipped,
        "floor": floor,
        "tolerance": tolerance,
        "origin": [float(v) for v in origin],
        "normal": [float(v) for v in normal],
    }


def _section_structured(cut):
    loops = []
    for profile in cut["profiles"]:
        points = np.asarray(profile["points"])
        loop = {
            "closed": bool(profile["closed"]),
            "noise_candidate": bool(profile["noise_candidate"]),
            "length_mm": float(profile["length"]),
            "raw_point_count": int(profile["raw"]),
            "simplified_point_count": int(len(points)),
            "bounds_mm": {
                "min": [float(v) for v in points.min(axis=0)],
                "max": [float(v) for v in points.max(axis=0)],
            },
            "points_mm": [[float(value) for value in point] for point in points],
            "fits": profile["fits"],
        }
        circle = profile["fits"].get("circle")
        if (
            profile["closed"]
            and circle
            and circle["angular_coverage_degrees"] >= 300.0
            and circle["relative_rms_residual"] is not None
            and circle["relative_rms_residual"] <= 0.05
        ):
            axis_point = [0.0, 0.0, 0.0]
            axis_point["xyz".index(cut["axis"])] = float(cut["pos"])
            for coordinate, axis_name in zip(circle["center_mm"], cut["plane"]):
                axis_point["xyz".index(axis_name)] = float(coordinate)
            loop["cylinder_candidate"] = {
                "axis_point_mm": axis_point,
                "axis_direction": list(cut["normal"]),
                "radius_mm": circle["radius_mm"],
                "rms_residual_mm": circle["rms_residual_mm"],
                "max_residual_mm": circle["max_residual_mm"],
                "sample_count": circle["sample_count"],
                "evidence": "single cross-section circle fit",
            }
        loops.append(loop)
    return {
        "axis": cut["axis"],
        "position_mm": float(cut["pos"]),
        "plane_axes": list(cut["plane"]),
        "origin_mm": list(cut["origin"]),
        "normal": list(cut["normal"]),
        "simplification": {
            "algorithm": "Douglas-Peucker",
            "tolerance_mm": float(cut["tolerance"]),
            "fragment_floor_mm": float(cut["floor"]),
            "noise_candidate_count": int(cut["skipped"]),
        },
        "loops": loops,
    }


def _largest_planar_region_fit(mesh):
    """Fit the largest connected coplanar face group, not an arbitrary whole solid."""
    facets = list(mesh.facets)
    if not facets:
        return None
    areas = np.asarray(mesh.facets_area, dtype=float)
    faces = np.asarray(facets[int(np.argmax(areas))], dtype=int)
    vertices = np.unique(np.asarray(mesh.faces)[faces].reshape(-1))
    fit = _plane_fit(np.asarray(mesh.vertices)[vertices])
    if fit is not None:
        fit["area_mm2"] = float(areas.max())
        fit["face_count"] = int(len(faces))
    return fit


def _plane_fit(points):
    points = np.unique(np.asarray(points, dtype=float), axis=0)
    if len(points) < 3:
        return None
    center = points.mean(axis=0)
    _, _, axes = np.linalg.svd(points - center, full_matrices=False)
    normal = axes[-1]
    residuals = np.abs((points - center) @ normal)
    return {
        "origin_mm": [float(v) for v in center],
        "normal": [float(v) for v in normal],
        "rms_residual_mm": float(np.sqrt(np.mean(residuals ** 2))),
        "max_residual_mm": float(residuals.max()),
        "sample_count": int(len(points)),
    }


def _profile_fits(points, closed=False):
    """Return measured line and circle candidates without claiming design intent."""
    points = np.unique(np.asarray(points, dtype=float), axis=0)
    if len(points) < 2:
        return {}
    center = points.mean(axis=0)
    _, _, axes = np.linalg.svd(points - center, full_matrices=False)
    direction = axes[0]
    normal = np.array([-direction[1], direction[0]])
    line_residuals = np.abs((points - center) @ normal)
    fits = {
        "line": {
            "point_mm": [float(v) for v in center],
            "direction": [float(v) for v in direction],
            "rms_residual_mm": float(np.sqrt(np.mean(line_residuals ** 2))),
            "max_residual_mm": float(line_residuals.max()),
            "sample_count": int(len(points)),
        }
    }
    if closed and len(points) >= 3 and np.linalg.matrix_rank(points - center) == 2:
        a = np.column_stack((2 * points[:, 0], 2 * points[:, 1], np.ones(len(points))))
        solution, _, _, _ = np.linalg.lstsq(a, np.sum(points ** 2, axis=1), rcond=None)
        circle_center = solution[:2]
        radius = float(np.sqrt(max(solution[2] + circle_center @ circle_center, 0.0)))
        residuals = np.abs(np.linalg.norm(points - circle_center, axis=1) - radius)
        angles = np.sort(
            np.mod(
                np.arctan2(
                    points[:, 1] - circle_center[1],
                    points[:, 0] - circle_center[0],
                ),
                2 * np.pi,
            )
        )
        gaps = np.diff(np.concatenate((angles, [angles[0] + 2 * np.pi])))
        fits["circle"] = {
            "center_mm": [float(v) for v in circle_center],
            "radius_mm": radius,
            "rms_residual_mm": float(np.sqrt(np.mean(residuals ** 2))),
            "max_residual_mm": float(residuals.max()),
            "sample_count": int(len(points)),
            "relative_rms_residual": float(np.sqrt(np.mean(residuals ** 2)) / radius) if radius else None,
            "angular_coverage_degrees": float(np.degrees(2 * np.pi - gaps.max())),
        }
    return fits


# How many points of one profile get printed. A slice of a noisy scan can survive
# simplification hundreds of points long, and a wall of coordinates stops being
# something to sketch against.
LISTED = 120


def section_report(cut):
    u, v = cut["plane"]
    lines = [
        f"  section {cut['axis']} = {cut['pos']:.2f}mm  points are ({u}, {v}) in mm"
    ]
    listed = [profile for profile in cut["profiles"] if not profile["noise_candidate"]]
    if not cut["profiles"]:
        lines.append("      the plane misses the mesh")
        return lines
    for prof in listed:
        shape = "closed loop" if prof["closed"] else "open"
        points = prof["points"]
        lines.append(
            f"      {shape}, {prof['length']:.1f}mm, {prof['raw']} points -> "
            f"{len(points)} at {cut['tolerance']}mm tolerance"
        )
        for p in points[:LISTED]:
            lines.append(f"          ({p[0]:8.2f}, {p[1]:8.2f})")
        if len(points) > LISTED:
            lines.append(
                f"          and {len(points) - LISTED} more. Raise --tolerance to thin it"
            )
    if cut["skipped"]:
        lines.append(
            f"      {cut['skipped']} noise candidate(s) under {cut['floor']:.1f}mm omitted from this text report; JSON includes them"
        )
    return lines


def _key(p):
    """An endpoint on a 0.001mm grid, so segments that meet actually match."""
    return (round(float(p[0]), 3), round(float(p[1]), 3))


def _length(points):
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _chains(segments):
    """Raw plane-intersection segments joined end to end into polylines.

    trimesh chains sections itself, but only through networkx, and a dependency is
    not worth what forty lines cover. A junction where three segments meet takes
    whichever continuation comes first, which is fine at scan fidelity.
    """
    at = defaultdict(list)
    for i, seg in enumerate(segments):
        at[_key(seg[0])].append(i)
        at[_key(seg[1])].append(i)
    used, chains = set(), []
    for start in range(len(segments)):
        if start in used:
            continue
        used.add(start)
        chain = [segments[start][0], segments[start][1]]
        for forward in (True, False):
            while True:
                tip = chain[-1] if forward else chain[0]
                free = [i for i in at[_key(tip)] if i not in used]
                if not free:
                    break
                i = free[0]
                used.add(i)
                a, b = segments[i]
                grown = b if _key(a) == _key(tip) else a
                chain.append(grown) if forward else chain.insert(0, grown)
        chains.append(np.asarray(chain))
    return chains


def _simplify(points, tolerance):
    """Douglas-Peucker, keeping every point that moves the line more than the tolerance."""
    if len(points) < 3:
        return points
    keep = np.zeros(len(points), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        seg = points[b] - points[a]
        rel = points[a + 1 : b] - points[a]
        span = float(np.hypot(*seg))
        if span == 0:  # a closed loop's ends coincide; distance from the point instead
            d = np.linalg.norm(rel, axis=1)
        else:
            d = np.abs(rel[:, 0] * seg[1] - rel[:, 1] * seg[0]) / span
        worst = int(d.argmax())
        if d[worst] > tolerance:
            i = a + 1 + worst
            keep[i] = True
            stack += [(a, i), (i, b)]
    return points[keep]
