"""Regenerate the synthetic reference meshes used by this example."""

import argparse
import io
import tempfile
from pathlib import Path

import numpy as np
import trimesh
from build123d import Align, Box, Cylinder, Pos, Rot, export_stl


HERE = Path(__file__).parent
SCANS = HERE / "scans"


def _stl_mesh(contents: bytes) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(io.BytesIO(contents), file_type="stl", process=False)
    assert isinstance(mesh, trimesh.Trimesh)
    return mesh


def _surface_samples(mesh: trimesh.Trimesh) -> np.ndarray:
    """Sample vertices, edge midpoints, and centroids without depending on face diagonals."""
    triangles = mesh.triangles
    return np.concatenate(
        (
            mesh.vertices,
            triangles.mean(axis=1),
            (triangles[:, 0] + triangles[:, 1]) / 2.0,
            (triangles[:, 1] + triangles[:, 2]) / 2.0,
            (triangles[:, 2] + triangles[:, 0]) / 2.0,
        )
    )


def _max_surface_distance(source: trimesh.Trimesh, target: trimesh.Trimesh) -> float:
    maximum = 0.0
    points = _surface_samples(source)
    for start in range(0, len(points), 128):
        _, distances, _ = trimesh.proximity.closest_point_naive(target, points[start : start + 128])
        maximum = max(maximum, float(distances.max(initial=0.0)))
    return maximum


def stl_geometry_matches(first: bytes, second: bytes, *, tolerance_mm: float = 0.025) -> bool:
    """Compare STL surfaces while allowing equivalent OCCT triangulations."""
    first_mesh = _stl_mesh(first)
    second_mesh = _stl_mesh(second)
    if first_mesh.is_watertight != second_mesh.is_watertight:
        return False
    if not np.allclose(first_mesh.bounds, second_mesh.bounds, atol=tolerance_mm, rtol=0.0):
        return False
    return max(
        _max_surface_distance(first_mesh, second_mesh),
        _max_surface_distance(second_mesh, first_mesh),
    ) <= tolerance_mm


def disk_stl() -> bytes:
    """Return a readable 32-sided cylinder with stable numeric formatting."""
    mesh = trimesh.creation.cylinder(radius=20.0, height=8.0, sections=32)
    lines = ["solid disk"]
    for normal, triangle in zip(mesh.face_normals, mesh.triangles, strict=True):
        lines.append(f"  facet normal {' '.join(f'{value:.9g}' for value in normal)}")
        lines.append("    outer loop")
        for vertex in triangle:
            lines.append(f"      vertex {' '.join(f'{value:.9g}' for value in vertex)}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append("endsolid disk")
    return ("\n".join(lines) + "\n").encode()


def offset_bracket_stl() -> bytes:
    """Return the asymmetric source geometry in its deliberately offset frame."""
    origin_x = 12.0
    origin_y = -7.0
    plate_width = 46.0
    plate_depth = 28.0
    base_thickness = 5.0
    wall_height = 15.0

    base = Pos(origin_x, origin_y, 0.0) * Box(
        plate_width,
        plate_depth,
        base_thickness,
        align=(Align.MIN, Align.MIN, Align.MIN),
    )
    wall = Pos(origin_x, origin_y, 0.0) * Box(
        6.0,
        plate_depth,
        wall_height,
        align=(Align.MIN, Align.MIN, Align.MIN),
    )
    first_hole = Pos(origin_x + 17.0, origin_y + 8.0, -1.0) * Cylinder(3.0, 7.0)
    second_hole = Pos(origin_x + 35.0, origin_y + 20.0, -1.0) * Cylinder(3.0, 7.0)
    side_hole = Pos(origin_x - 1.0, origin_y + 19.0, 10.0) * Rot(0.0, 90.0, 0.0) * Cylinder(2.0, 8.0)
    locator_notch = Pos(origin_x + plate_width - 2.0, origin_y + 3.0, -1.0) * Box(
        3.0,
        4.0,
        7.0,
        align=(Align.MIN, Align.MIN, Align.MIN),
    )
    shape = base + wall - first_hole - second_hole - side_hole - locator_notch

    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "offset_bracket.stl"
        export_stl(shape, target, tolerance=0.01, angular_tolerance=0.15)
        return target.read_bytes()


def generated() -> dict[str, bytes]:
    return {
        "disk.stl": disk_stl(),
        "offset_bracket.stl": offset_bracket_stl(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if a checked-in mesh differs from generated geometry")
    args = parser.parse_args()
    meshes = generated()
    if args.check:
        changed = [name for name, data in meshes.items() if not stl_geometry_matches((SCANS / name).read_bytes(), data)]
        if changed:
            parser.error(f"regenerate changed fixture(s): {', '.join(changed)}")
        print(f"checked {len(meshes)} fixtures")
        return 0
    for name, data in meshes.items():
        target = SCANS / name
        target.write_bytes(data)
        print(f"wrote {target.relative_to(HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
