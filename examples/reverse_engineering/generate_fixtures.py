"""Regenerate the synthetic reference meshes used by this example."""

import argparse
import io
import tempfile
from pathlib import Path

import trimesh
from build123d import Align, Box, Cylinder, Pos, Rot, export_stl


HERE = Path(__file__).parent
SCANS = HERE / "scans"


def stl_geometry(contents: bytes) -> tuple[tuple[tuple[float, float, float], ...], ...]:
    """Return triangles without serialization order, winding, or signed zero."""
    mesh = trimesh.load_mesh(io.BytesIO(contents), file_type="stl", process=False)
    triangles = []
    for triangle in mesh.triangles:
        vertices = [
            tuple(0.0 if abs(float(value)) < 5e-6 else round(float(value), 5) for value in vertex)
            for vertex in triangle
        ]
        triangles.append(tuple(sorted(vertices)))
    return tuple(sorted(triangles))


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
        changed = [name for name, data in meshes.items() if stl_geometry((SCANS / name).read_bytes()) != stl_geometry(data)]
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
