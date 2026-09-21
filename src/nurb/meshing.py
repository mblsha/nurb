"""Meshing policies for display estimates and bounded absolute verification.

Display meshes are reused for interactive feedback. Verification runs OCCT in a disposable process: a pathological tessellation cannot monopolize the viewer's Python process, and a timeout is an unknown result rather than a coarser silent fallback.
"""

from dataclasses import dataclass
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MAX_TRIANGLES = 1_000_000


class MeshingError(ValueError):
    """Verification could not produce evidence within its declared policy."""


class VerificationCancelled(MeshingError):
    """The requested verification was stopped and has no current result."""


def check_cancelled(stop):
    if stop is not None and stop():
        raise VerificationCancelled("Verification cancelled; no result was produced.")


@dataclass(frozen=True)
class VerificationPolicy:
    accuracy_mm: float
    timeout_s: float = DEFAULT_TIMEOUT_S
    max_triangles: int = DEFAULT_MAX_TRIANGLES
    feature_size_mm: float | None = None

    def __post_init__(self):
        if not math.isfinite(self.accuracy_mm) or not 0.00001 <= self.accuracy_mm <= 1.0:
            raise ValueError("mesh accuracy must be between 0.00001 and 1 mm")
        if not math.isfinite(self.timeout_s) or not 0 < self.timeout_s <= 120:
            raise ValueError("mesh time budget must be greater than 0 and at most 120 seconds")
        if isinstance(self.max_triangles, bool) or not isinstance(self.max_triangles, int) or not 12 <= self.max_triangles <= 5_000_000:
            raise ValueError("mesh triangle budget must be an integer from 12 to 5000000")
        if self.feature_size_mm is not None and (not math.isfinite(self.feature_size_mm) or self.feature_size_mm <= 0):
            raise ValueError("the smallest feature must be a positive size in mm")

    @classmethod
    def for_tolerance(cls, tolerance_mm, **options):
        accuracy = options.pop("accuracy_mm", None)
        return cls(min(0.025, tolerance_mm / 4.0) if accuracy is None else accuracy, **options)

    def provenance(self, tolerance_mm=None):
        warnings = []
        if tolerance_mm is not None and self.accuracy_mm > tolerance_mm / 4.0:
            warnings.append("Mesh accuracy exceeds one quarter of the acceptance tolerance; refine it before judging fit.")
        if self.feature_size_mm is None:
            warnings.append("Smallest feature size is unspecified; this does not establish that every small lip or recess was resolved.")
        elif self.accuracy_mm > self.feature_size_mm / 5.0:
            warnings.append("Mesh accuracy exceeds one fifth of the smallest feature; refine it before judging that feature.")
        return {
            "method": "Absolute mesh verification", "absolute_deflection_mm": self.accuracy_mm,
            "angular_deflection_rad": 0.05, "timeout_s": self.timeout_s,
            "max_triangles": self.max_triangles, "feature_size_mm": self.feature_size_mm,
            "warnings": warnings,
            "feature_to_deflection_ratio": None if self.feature_size_mm is None else self.feature_size_mm / self.accuracy_mm,
            "detail": "Requested OCCT absolute deflection, not a measured error bound. Distances and coverage are sampled; reference scan accuracy is not certified.",
        }


def display_provenance(relative_deflection=None):
    return {
        "method": "Display mesh estimate", "absolute_deflection_mm": None,
        "relative_deflection": relative_deflection,
        "warnings": ["Display mesh accuracy in mm is unknown; use absolute verification for small mating features."],
        "detail": "Distances use the cached viewer mesh. A relative display setting is not an absolute geometric tolerance.",
    }


def verification_mesh(shape, policy, stop=None):
    """Return an independently tessellated mesh, or raise without altering the shape."""
    import numpy as np
    import trimesh
    from build123d import export_brep

    check_cancelled(stop)
    with tempfile.TemporaryDirectory(prefix="nurb-verify-") as directory:
        root = Path(directory)
        source, output = root / "source.brep", root / "mesh.npz"
        if not export_brep(shape, source):
            raise MeshingError("Verification unknown: could not snapshot the CAD shape.")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent) + os.pathsep + env.get("PYTHONPATH", "")
        deadline = time.monotonic() + policy.timeout_s
        process = subprocess.Popen(
            [sys.executable, "-m", "nurb.meshing", str(source), str(output),
             str(policy.accuracy_mm), str(policy.max_triangles)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
        )
        try:
            while True:
                check_cancelled(stop)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise MeshingError(f"Verification unknown: meshing exceeded {policy.timeout_s:g} seconds. Increase the time budget, relax accuracy, or verify a smaller component.")
                try:
                    _, stderr = process.communicate(timeout=min(0.1, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
            check_cancelled(stop)
            if process.returncode or not output.is_file():
                message = stderr.strip().splitlines()[-1:] or [f"meshing process exited with code {process.returncode}"]
                raise MeshingError(f"Verification unknown: {message[0][:500]}")
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate()
        with np.load(output, allow_pickle=False) as data:
            vertices, faces = data["vertices"], data["faces"]
        if not len(faces) or len(faces) > policy.max_triangles or not np.isfinite(vertices).all():
            raise MeshingError("Verification unknown: mesher returned an empty, invalid, or over-budget surface.")
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _worker(source, output, accuracy, max_triangles):
    import numpy as np
    from build123d import import_brep
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRepTools import BRepTools
    from OCP.TopLoc import TopLoc_Location
    from . import builder

    shape = import_brep(source)
    BRepTools.Clean_s(shape.wrapped)
    mesher = BRepMesh_IncrementalMesh(shape.wrapped, accuracy, False, 0.05, True)
    if not mesher.IsDone():
        raise MeshingError("OCCT did not complete the requested tessellation")
    triangles = 0
    for face in shape.faces():
        poly = BRep_Tool.Triangulation_s(face.wrapped, TopLoc_Location())
        if poly is None or not poly.NbTriangles():
            raise MeshingError("OCCT left a CAD face without triangles")
        triangles += poly.NbTriangles()
        if triangles > max_triangles:
            raise MeshingError(f"mesh exceeds the {max_triangles} triangle budget; relax accuracy or verify a smaller component")
    # Read exactly the absolute mesh above. A second relative meshing pass would
    # invalidate the accuracy metadata and duplicate the expensive work.
    vertices, faces, _ = builder._triangulate(shape, accuracy, remesh=False)
    np.savez(output, vertices=np.asarray(vertices), faces=np.asarray(faces))


if __name__ == "__main__":
    try:
        _worker(sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]))
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
