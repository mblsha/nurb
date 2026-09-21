"""Symmetry evidence from a reference and the finished, trimmed CAD boundary."""

from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile

import numpy as np


@dataclass(frozen=True)
class Options:
    axis: str = "x"
    reference_tolerance_mm: float = 0.5
    cad_tolerance_mm: float = 0.01
    edge_step_mm: float = 0.5
    face_samples: int = 25
    sample_budget: int = 20000
    timeout_s: float = 30.0
    max_angle_deg: float = 15.0
    memory_limit_mb: int = 2048
    reference_bounds_mm: dict | None = None

    def __post_init__(self):
        from .bounded import memory_policy
        memory_policy(self.memory_limit_mb)
        if self.axis not in ("x", "y", "z"):
            raise ValueError("choose an approximate symmetry normal: x, y, or z")
        for name in ("reference_tolerance_mm", "cad_tolerance_mm", "edge_step_mm", "timeout_s", "max_angle_deg"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        if self.timeout_s > 120 or self.max_angle_deg > 45:
            raise ValueError("use at most 120 seconds and a 45 degree fit search")
        for name, low, high in (("face_samples", 9, 400), ("sample_budget", 100, 100000)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f"{name} must be an integer from {low} to {high}")
        if self.reference_bounds_mm is not None:
            from .compare import inspection_regions
            inspection_regions([{"name": "symmetry fit", "bounds_mm": self.reference_bounds_mm}])


def reflect(points, normal, offset):
    points, normal = np.asarray(points), np.asarray(normal)
    return points - 2 * (points @ normal - offset)[:, None] * normal


def statistics(distances, tolerance):
    d = np.asarray(distances)
    if not len(d):
        return {"status": "unknown", "count": 0}
    return {"status": "measured", "count": len(d), "median_mm": float(np.median(d)),
            "p95_mm": float(np.percentile(d, 95)), "max_mm": float(d.max()),
            "fraction_within_tolerance": float(np.mean(d <= tolerance))}


def per_side(points, distances, normal, offset, tolerance):
    signed = np.asarray(points) @ normal - offset
    return {"negative": statistics(np.asarray(distances)[signed < -1e-7], tolerance),
            "positive": statistics(np.asarray(distances)[signed > 1e-7], tolerance),
            "on_plane": statistics(np.asarray(distances)[np.abs(signed) <= 1e-7], tolerance)}


def load_reference(path, units=None):
    """Mesh loading follows scan units; point clouds require explicit units."""
    import trimesh
    from . import scan

    path = Path(path)
    try:
        reference, unit, _ = scan.load(path, units)
        return reference, unit
    except ValueError as exc:
        if "only points" not in str(exc):
            raise
    if units not in scan.UNITS:
        raise ValueError("point-cloud symmetry requires explicit units: choose mm, cm, m, or in")
    if scan.reference_suffix(path) == ".ply.gz":
        with path.open("rb") as stream:
            reference = trimesh.load(io.BytesIO(scan.decompress_ply(stream, path.name)), file_type="ply", resolver={})
    else:
        reference = trimesh.load(path, resolver={})
    if not isinstance(reference, trimesh.points.PointCloud) or len(reference.vertices) < 32:
        raise ValueError("provide a PLY point cloud with at least 32 points or a triangle mesh")
    reference.apply_scale(scan.UNITS[units])
    return reference, units


def reference_identity(path, units):
    digest = hashlib.sha256(Path(path).read_bytes())
    digest.update(str(units).encode())
    return digest.hexdigest()


def reference_snapshot(path, units=None):
    """Tie geometry and content identity to the same immutable bytes."""
    path = Path(path)
    body = path.read_bytes()
    with tempfile.TemporaryDirectory(prefix="nurb-symmetry-reference-") as temporary:
        snapshot = Path(temporary) / path.name
        snapshot.write_bytes(body)
        reference, unit = load_reference(snapshot, units)
    digest = hashlib.sha256(body); digest.update(unit.encode())
    return reference, unit, digest.hexdigest()


def identity(shape, reference_id, transform, configuration, revision, options, landmarks=None):
    from .feature_evidence import evidence_identity, shape_identity
    return evidence_identity(shape_identity(shape), reference_id, transform,
                             json.dumps(configuration, sort_keys=True),
                             {"method": "symmetry-v2", "revision": revision, "options": asdict(options), "landmarks": landmarks or {}})


def fit_plane(reference, options):
    """A bounded local fit around the chosen axis, evaluated on independent samples."""
    import trimesh
    from scipy.optimize import least_squares
    from scipy.spatial import cKDTree
    from . import compare

    is_mesh = hasattr(reference, "faces") and len(reference.faces) > 0
    if options.reference_bounds_mm is not None:
        if is_mesh:
            reference = compare._clip_region(reference, options.reference_bounds_mm)
        else:
            p = np.asarray(reference.vertices)
            keep = np.all((p >= options.reference_bounds_mm["min"]) & (p <= options.reference_bounds_mm["max"]), axis=1)
            reference = trimesh.points.PointCloud(p[keep])
    vertices = np.asarray(reference.vertices)
    if len(vertices) < (4 if is_mesh else 32) or not np.isfinite(vertices).all():
        raise ValueError("the fit selection needs at least 32 finite points spanning both sides")
    if np.count_nonzero(np.ptp(vertices, axis=0) > 1e-5) < 2:
        raise ValueError("the fit selection is nearly a line; include a wider portion of both sides")
    if is_mesh:
        target, face_ids = trimesh.sample.sample_surface(reference, 50000, seed=91)
        normals = np.asarray(reference.face_normals)[face_ids]
        fitting, _ = trimesh.sample.sample_surface(reference, 4000, seed=92)
        validation, _ = trimesh.sample.sample_surface(reference, 5000, seed=93)
    else:
        # Preserve paired cloud points in the search tree; subsampling only one side
        # would add a sampling error that can look like a misaligned plane.
        if len(vertices) > 500000:
            raise ValueError("point cloud exceeds 500000 points; crop the fit region or downsample it first")
        target, normals = vertices, None
        rng = np.random.default_rng(92)
        fitting = vertices[rng.choice(len(vertices), min(4000, len(vertices)), replace=False)]
        validation = vertices[np.random.default_rng(93).choice(len(vertices), min(5000, len(vertices)), replace=False)]
    tree = cKDTree(target)
    axis = np.eye(3)["xyz".index(options.axis)]
    basis = np.eye(3)[[i for i in range(3) if i != "xyz".index(options.axis)]]
    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    span = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    if span <= 1e-5:
        raise ValueError("the reference has no measurable span")

    def unpack(parameters):
        normal = axis + parameters[:2] @ basis
        normal /= np.linalg.norm(normal)
        return normal, float(center @ normal + parameters[2])

    def residual(parameters):
        normal, offset = unpack(parameters)
        mirrored = reflect(fitting, normal, offset)
        _, indices = tree.query(mirrored)
        delta = mirrored - target[indices]
        return np.einsum("ij,ij->i", delta, normals[indices]) if normals is not None else delta.reshape(-1)

    slope = math.tan(math.radians(options.max_angle_deg)) / math.sqrt(2)
    solved = least_squares(residual, np.zeros(3), bounds=([-slope, -slope, -span / 4], [slope, slope, span / 4]),
                           loss="soft_l1", f_scale=max(options.reference_tolerance_mm / 2, 0.01), max_nfev=80,
                           diff_step=1e-4, ftol=1e-9, xtol=1e-9, gtol=1e-9)
    surface = compare._surface(reference) if is_mesh else None
    baseline_normal, baseline_offset = unpack(np.zeros(3))
    baseline_mirrored = reflect(validation, baseline_normal, baseline_offset)
    baseline_distances = compare._to_surface(baseline_mirrored, surface) if is_mesh else tree.query(baseline_mirrored)[0]
    baseline_sides = per_side(validation, baseline_distances, baseline_normal, baseline_offset, options.reference_tolerance_mm)
    normal, offset = unpack(solved.x)
    mirrored = reflect(validation, normal, offset)
    distances = compare._to_surface(mirrored, surface) if is_mesh else tree.query(mirrored)[0]
    baseline_stats = statistics(baseline_distances, options.reference_tolerance_mm)
    fitted_stats = statistics(distances, options.reference_tolerance_mm)
    sides = per_side(validation, distances, normal, offset, options.reference_tolerance_mm)
    if min(sides[side]["count"] for side in ("negative", "positive")) < 16:
        raise ValueError("the fitted plane lacks evidence on both sides; expand the selected region")
    warnings = []
    if not solved.success or np.any(np.abs(solved.active_mask)):
        warnings.append("The local fit reached its search limit; choose a closer approximate axis or expand the search angle.")
    if not is_mesh:
        spacing = tree.query(target[::max(1, len(target) // 2000)], k=2)[0][:, 1]
        warnings.append(f"Point-cloud distances include sampling gaps; median nearest-point spacing is {np.median(spacing):.4g} mm.")
    # Minimal rotation takes the fitted normal to the selected coordinate axis.
    cross = np.cross(normal, axis)
    skew = np.array([[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]])
    rotation = np.eye(3) + skew + skew @ skew / (1 + float(normal @ axis))
    alignment = np.eye(4); alignment[:3, :3] = rotation; alignment[:3, 3] = -offset * axis
    return {"normal": normal.tolist(), "offset_mm": offset, "equation": "normal dot point = offset_mm",
            "frame": "part_mm", "approximate_axis": options.axis, "to_axis_transform": alignment.reshape(-1).tolist(),
            "fit_evaluations": solved.nfev, "reference_sides": sides,
            "baseline": {"normal": baseline_normal.tolist(), "offset_mm": baseline_offset, "reference_sides": baseline_sides,
                         "statistics": baseline_stats, "definition": "chosen axis through reference bounding-box center, before fitting"},
            "after_fit": {"statistics": fitted_stats, "reference_sides": sides},
            "p95_improvement_mm": baseline_stats["p95_mm"] - fitted_stats["p95_mm"],
            "comparison_method": "same independent validation samples and same distance query before and after; negative improvement is retained",
            "reference_tolerance_mm": options.reference_tolerance_mm,
            "reference_status": "within_sampled_threshold" if solved.success and not np.any(solved.active_mask) and all(sides[s]["p95_mm"] <= options.reference_tolerance_mm for s in ("negative", "positive")) else "review",
            "reference_method": "reflected holdout samples to triangle surface" if is_mesh else "reflected cloud points to cloud points",
            "warnings": warnings}


def cad_symmetry(shape, plane, options):
    """Sample trimmed faces and every edge, then query the boundary-only B-rep."""
    from build123d import Compound, Vertex
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepClass import BRepClass_FaceClassifier
    from OCP.BRepTools import BRepTools
    from OCP.TopAbs import TopAbs_IN, TopAbs_ON
    from OCP.gp import gp_Pnt2d

    if not shape.is_valid:
        raise ValueError("CAD symmetry is unknown because the finished B-rep is invalid; repair the model first")
    faces, edges = list(shape.faces()), list(shape.edges())
    if not faces:
        raise ValueError("the model has no finished CAD faces to verify")
    samples, kinds = [], []

    def add(point, kind):
        if len(samples) >= options.sample_budget:
            raise ValueError("CAD symmetry sample budget exceeded; increase the budget or select a smaller part")
        samples.append(tuple(point)); kinds.append(kind)

    for edge in edges:
        count = max(5, math.ceil(edge.length / options.edge_step_mm) + 1)
        if len(samples) + count > options.sample_budget:
            raise ValueError("CAD trim-edge samples exceed the budget; increase it or use a larger edge step")
        for parameter in np.linspace(0, 1, count):
            add(edge.position_at(float(parameter)), "trim_edges")
    grid = math.ceil(math.sqrt(options.face_samples))
    for face in faces:
        u0, u1, v0, v1 = BRepTools.UVBounds_s(face.wrapped)
        if not np.isfinite([u0, u1, v0, v1]).all():
            raise ValueError("a CAD face has unbounded parameters; close the modeled part before verifying")
        adaptor = BRepAdaptor_Surface(face.wrapped)
        accepted = 0
        for u in np.linspace(u0, u1, grid + 2)[1:-1]:
            for v in np.linspace(v0, v1, grid + 2)[1:-1]:
                classifier = BRepClass_FaceClassifier(face.wrapped, gp_Pnt2d(float(u), float(v)), 1e-7)
                if classifier.State() in (TopAbs_IN, TopAbs_ON):
                    point = adaptor.Value(float(u), float(v))
                    add((point.X(), point.Y(), point.Z()), "trimmed_faces"); accepted += 1
        if not accepted:
            raise ValueError("a narrow trimmed CAD face received no interior samples; increase face samples before claiming symmetry")
    points = np.asarray(samples)
    normal, offset = np.asarray(plane["normal"]), plane["offset_mm"]
    reflected = reflect(points, normal, offset)
    # A solid distance can classify points in material as distance zero. A compound
    # of faces measures the actual trimmed boundary and detects an unpaired hole.
    boundary = Compound(children=faces)
    from . import bounded
    bounded.phase('Trimmed CAD boundary distances')
    distances = np.asarray([boundary.distance_to(Vertex(*point)) for point in reflected])
    if not np.isfinite(distances).all():
        raise ValueError("the CAD kernel returned a non-finite symmetry distance")
    bounded.phase('Symmetry statistics')
    kinds = np.asarray(kinds)
    from .symmetry_categories import periodic_seams
    bounded.phase("Periodic seam distances")
    seam_result = periodic_seams(shape, plane, options, options.sample_budget-len(points))
    return {"status": "within_sampled_threshold" if np.max(distances) <= options.cad_tolerance_mm else "deviations",
            "method": "reflected trimmed-face and trim-edge samples to finished boundary-only B-rep",
            "tolerance_mm": options.cad_tolerance_mm, "edge_step_mm": options.edge_step_mm,
            "faces": len(faces), "edges": len(edges), "sample_count": len(points),
            "sides": per_side(points, distances, normal, offset, options.cad_tolerance_mm),
            "periodic_seams": seam_result,
            "trim_edges": statistics(distances[kinds == "trim_edges"], options.cad_tolerance_mm),
            "trimmed_faces": statistics(distances[kinds == "trimmed_faces"], options.cad_tolerance_mm),
            "worst_point_mm": points[int(np.argmax(distances))].tolist(),
            "worst_reflected_point_mm": reflected[int(np.argmax(distances))].tolist(),
            "limitation": "Finite samples include every trim edge and every trimmed face, but do not prove exact symmetry at every point. Reference fit and alignment error also contribute to these CAD distances."}


def run(shape, reference, options, evidence_identity, stop=None, landmarks=None):
    """Bound snapshot, fit, all kernel distances and statistics as one job."""
    from . import bounded
    if bounded.fresh(): return _measure(shape,reference,options,evidence_identity,landmarks)
    return bounded.run(lambda: prepare(shape,reference,options,evidence_identity,landmarks=landmarks),timeout_s=options.timeout_s,memory_limit_mb=options.memory_limit_mb,stop=stop)


def _measure(shape,reference,options,evidence_identity,landmarks=None):
    from . import bounded
    bounded.phase('Fitting reference symmetry plane')
    plane=fit_plane(reference,options)
    bounded.phase('Trimmed CAD sampling and distances')
    result={'plane':plane,'cad':cad_symmetry(shape,plane,options)}
    from .symmetry_categories import category_identity, feature_centers
    bounded.phase('Authored feature center distances')
    saved = landmarks or {}
    result['feature_centers'] = {
        'cad_centers': feature_centers(saved.get('cad_centers', []), plane, options.cad_tolerance_mm),
        'reference_points': feature_centers(saved.get('reference_points', []), plane, options.reference_tolerance_mm),
        'unlocated_features': saved.get('unlocated_features', []),
    }
    for key, category in (('periodic_seams', result['cad']['periodic_seams']),
                          ('cad_centers', result['feature_centers']['cad_centers']),
                          ('reference_points', result['feature_centers']['reference_points'])):
        category['identity'] = category_identity(evidence_identity, key)
    return {'kind':'symmetry_evidence','schema_version':2,'status':'measured','identity':evidence_identity,
            'options':asdict(options),'memory':bounded.memory_policy(options.memory_limit_mb),**result}


def prepare(shape,reference,options,evidence_identity,metadata=None,source_files=None,landmarks=None):
    from . import bounded
    from build123d import export_brep
    bounded.phase('Serializing symmetry snapshots')
    root=bounded.workspace()
    if not export_brep(shape,root/'shape.brep'): raise ValueError('Could not snapshot the CAD for symmetry.')
    np.savez(root/'reference.npz',vertices=np.asarray(reference.vertices),faces=np.asarray(getattr(reference,'faces',[]),dtype=int).reshape(-1,3))
    return bounded.stage('nurb.symmetry','_snapshot_job',root=str(root),options=asdict(options),evidence_identity=evidence_identity,metadata=metadata,source_files=source_files,landmarks=landmarks)


def _snapshot_job(root,options,evidence_identity,metadata=None,source_files=None,landmarks=None):
    import trimesh
    from build123d import import_brep
    from . import bounded
    root=Path(root)
    bounded.phase('Loading isolated symmetry snapshots')
    with np.load(root/'reference.npz',allow_pickle=False) as data:
        reference=trimesh.Trimesh(vertices=data['vertices'],faces=data['faces'],process=False) if len(data['faces']) else trimesh.points.PointCloud(data['vertices'])
    result=_measure(import_brep(root/'shape.brep'),reference,Options(**options),evidence_identity,landmarks)
    if source_files: bounded.verify_files(source_files)
    return {**result,**(metadata or {})}


def source_revision(part):
    """Portable content identity for model source and local analytic inputs."""
    part = Path(part); root = part.parent.parent
    sources = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(piece.startswith(".") or piece in ("build", "inspections", "__pycache__") for piece in relative.parts):
            continue
        if path.is_file() and path.suffix.lower() in (".py", ".md", ".toml", ".json", ".step", ".stp", ".brep"):
            if path.suffix.lower() == ".json":
                try:
                    if json.loads(path.read_text()).get("kind") in ("symmetry_evidence", "local-section-evidence"):
                        continue
                except (ValueError, AttributeError, UnicodeError):
                    pass
            sources.append(path)
    digest = hashlib.sha256()
    for path in sorted(sources):
        digest.update(path.relative_to(root).as_posix().encode()); digest.update(b"\0"); digest.update(path.read_bytes())
    return digest.hexdigest()


def command(args):
    from . import bounded
    if bounded.fresh(): return _command(args)
    try:
        result=bounded.run(lambda:bounded.stage('nurb.symmetry','_bounded_command',arguments={k:v for k,v in vars(args).items() if k!='fn'}),
            timeout_s=args.timeout,memory_limit_mb=getattr(args,'memory_limit',2048),
            progress=None if args.json else lambda state:print('  '+state['phase'],file=sys.stderr,flush=True))
        if args.json and result['stdout'].strip():
            evidence=json.loads(result['stdout'])
            evidence['resources']=result['resources']
            result['stdout']=json.dumps(evidence,indent=2,allow_nan=False)+'\n'
        print(result['stdout'],end='');print(result['stderr'],end='',file=sys.stderr)
        if result['exit_code'] is not None: raise SystemExit(result['exit_code'])
    except (bounded.WorkError, ValueError) as exc:
        result={'status':'unknown','reason':getattr(exc,'reason','invalid_options'),'error':str(exc),'resources':getattr(exc,'resources',{})}
        if args.json: print(json.dumps(result))
        else: print(str(exc),file=sys.stderr)
        raise SystemExit(2) from exc


def _bounded_command(arguments):
    from argparse import Namespace
    from . import bounded
    arguments.pop('fn',None)
    return bounded.capture_command(_command,Namespace(**arguments))


def _command(args):
    from . import builder, checks, compare
    from .cli import _resolve, project_root
    root = project_root()
    paths = _resolve(root, args.part)
    if len(paths) != 1:
        raise SystemExit("choose one part for symmetry evidence")
    part = paths[0]
    try:
        target = compare.setting(checks.settings(part)) or {}
        source = args.against or target.get("file")
        if not source:
            raise ValueError("attach a reference or choose --against <mesh-or-point-cloud.ply>")
        source = Path(source)
        if not source.is_absolute():
            source = root / source
        options = Options(axis=args.axis, reference_tolerance_mm=args.reference_tolerance,
                          cad_tolerance_mm=args.cad_tolerance, edge_step_mm=args.edge_step,
                          face_samples=args.face_samples, sample_budget=args.sample_budget,
                          timeout_s=args.timeout, max_angle_deg=args.max_angle, memory_limit_mb=getattr(args,"memory_limit",2048),
                          reference_bounds_mm=json.loads(args.bounds) if args.bounds else None)
        configurations = {name: params for name, params, _ in checks.configurations(part)}
        name = args.variant or part.stem
        if name not in configurations:
            raise ValueError(f"unknown variant {name}; choose one of {', '.join(configurations)}")
        from contextlib import redirect_stdout
        with redirect_stdout(sys.stderr):
            shape, parameters, _ = builder.build(part, overrides=configurations[name])
        configuration = {"name": name if name != part.stem else "default", "parameters": {p["name"]: p["value"] for p in parameters}}
        transform = target.get("transform") or compare.IDENTITY
        if args.identity_alignment:
            transform = compare.IDENTITY
        reference, unit, reference_id = reference_snapshot(source, args.units or target.get("units"))
        reference.apply_transform(compare._transform(transform))
        revision = source_revision(part)
        from .symmetry_categories import landmarks
        locations = landmarks(target.get("regions", []), transform)
        contract = identity(shape, reference_id, transform, configuration, revision, options, locations)
        if args.check_report:
            previous = json.loads(Path(args.check_report).read_text())
            result = {"status": "current" if previous.get("identity") == contract else "stale", "identity": contract}
        else:
            result = run(shape, reference, options, contract, landmarks=locations)
            if source_revision(part) != revision or reference_identity(source, unit) != reference_id:
                result = {"status": "stale", "identity": contract, "error": "Model source or reference changed during verification; run it again."}
            result["configuration"] = configuration
            result["reference_units"] = unit
            result["alignment"] = list(transform)
        result.update(kind="symmetry_evidence", source_revision=revision)
        if args.output:
            destination = Path(args.output); destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        if args.json:
            print(json.dumps(result, indent=2, allow_nan=False))
        else:
            print(f"  symmetry: {result['status']}")
            if "plane" in result:
                print(f"  reference plane: normal {result['plane']['normal']}, offset {result['plane']['offset_mm']:.6g} mm")
                plane=result["plane"]
                print(f"  reference p95 before fit {plane['baseline']['statistics']['p95_mm']:.6g} mm; after fit {plane['after_fit']['statistics']['p95_mm']:.6g} mm; improvement {plane['p95_improvement_mm']:.6g} mm")
                print(f"  finished CAD: {result['cad']['status']}; sampled tolerance {options.cad_tolerance_mm:g} mm")
                for side in ("negative", "positive"):
                    measured = result["cad"]["sides"][side]
                    print(f"    {side}: {measured['count']} samples, p95 {measured.get('p95_mm', float('nan')):.6g} mm, max {measured.get('max_mm', float('nan')):.6g} mm")
                for key, category in (("periodic seams", result["cad"]["periodic_seams"]),
                                      ("CAD centers", result["feature_centers"]["cad_centers"]),
                                      ("reference annotations", result["feature_centers"]["reference_points"])):
                    print(f"  {key}: {category['status']}; {category['count']} locations, {category['unmatched_count']} unmatched; tolerance {category['tolerance_mm']:g} mm")
                print(f"  unlocated features: {len(result['feature_centers']['unlocated_features'])}")
                print(f"  {result['cad']['limitation']}")
    except (ValueError, OSError) as exc:
        if args.json:
            print(json.dumps({"status": "unknown", "error": str(exc), "reason": getattr(exc, "reason", "worker_failure")}))
            raise SystemExit(2) from exc
        else:
            raise SystemExit(f"  symmetry unknown: {exc}") from exc
