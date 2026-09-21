"""Portable PLY references: immutable source assets, texture seams and explicit filtering."""

import hashlib
import io
import json
import pathlib
import re
import stat
import zipfile
from dataclasses import dataclass

import numpy as np

from . import scan

MAX_BYTES = 256 * 1024 * 1024
MAX_FILES = 128
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_name(name):
    # ZIP and PLY paths use POSIX separators; reject Windows paths on every host.
    p = pathlib.PurePosixPath(name)
    if (not name or "\\" in name or ":" in name or p.is_absolute()
            or any(x in ("", ".", "..") for x in name.split("/"))
            or any(ord(c) < 32 for c in name)):
        raise ValueError(f"Unsafe reference asset path: {name!r}; use relative paths inside the bundle")
    return p.as_posix()


def _ply_body(name, data):
    return scan.decompress_ply(io.BytesIO(data), name) if scan.reference_suffix(name) == ".ply.gz" else data


def texture_name(name, data):
    body = _ply_body(name, data)
    end = body.find(b"end_header")
    if end < 0 or end > 1024 * 1024 or not body.startswith(b"ply"):
        raise ValueError("PLY header is missing or too large; export a valid triangle mesh")
    header = body[:end].decode("ascii", errors="strict")
    names = re.findall(r"^comment\s+TextureFile\s+(.+?)\s*$", header, re.MULTILINE)
    if len(names) > 1:
        raise ValueError("Multiple PLY TextureFile declarations are ambiguous; export one texture atlas")
    return safe_name(names[0]) if names else None


def is_bundle(path):
    path = pathlib.Path(path)
    if path.suffix.lower() == ".zip":
        return True
    if scan.reference_suffix(path) in {".ply", ".ply.gz"}:
        return texture_name(path.name, path.read_bytes()) is not None
    return False


def _assets(filename, body, sidecars=None):
    if len(body) > MAX_BYTES:
        raise ValueError("Reference exceeds the 256 MiB import limit")
    safe_name(filename)
    if pathlib.Path(filename).suffix.lower() != ".zip":
        if filename in (sidecars or {}):
            raise ValueError("A sidecar cannot replace the selected PLY source")
        assets = {filename: body, **(sidecars or {})}
    else:
        if sidecars:
            raise ValueError("Choose a ZIP bundle alone, or a PLY with its image sidecar")
        assets = {}
        try:
            with zipfile.ZipFile(io.BytesIO(body)) as archive:
                entries = archive.infolist()
                if len(entries) > MAX_FILES:
                    raise ValueError("Reference ZIP has too many entries (maximum 128)")
                total = 0
                seen = set()
                for entry in entries:
                    name = safe_name(entry.filename.rstrip("/"))
                    folded = name.casefold()
                    if folded in seen:
                        raise ValueError("Reference ZIP contains duplicate or case-ambiguous asset names")
                    seen.add(folded)
                    mode = entry.external_attr >> 16
                    if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
                        raise ValueError("Reference ZIP cannot contain symlinks or special files")
                    if entry.flag_bits & 1:
                        raise ValueError("Encrypted reference ZIP files are not supported")
                    if entry.is_dir():
                        continue
                    total += entry.file_size
                    if total > MAX_BYTES:
                        raise ValueError("Expanded reference ZIP exceeds the 256 MiB import limit")
                    with archive.open(entry) as stream:
                        data = stream.read(MAX_BYTES + 1)
                    if len(data) != entry.file_size or len(data) > MAX_BYTES:
                        raise ValueError("Reference ZIP asset exceeds its declared size")
                    assets[name] = data
        except (zipfile.BadZipFile, RuntimeError, EOFError) as exc:
            raise ValueError(f"Could not read reference ZIP: {exc}") from exc
    if len(assets) > MAX_FILES or sum(len(v) for v in assets.values()) > MAX_BYTES:
        raise ValueError("Reference assets exceed the import size or file count limit")
    for name in assets:
        safe_name(name)
    if len({n.casefold() for n in assets}) != len(assets):
        raise ValueError("Reference assets have case-ambiguous names")
    names = {name.casefold() for name in assets}
    if any(parent.as_posix().casefold() in names for name in assets for parent in pathlib.PurePosixPath(name).parents if parent.as_posix() != "."):
        raise ValueError("Reference assets contain a conflicting file and directory path")
    return assets


@dataclass
class ReferenceImport:
    filename: str
    original: bytes
    assets: dict
    mesh: object
    provenance: dict
    face_groups: list

    def summary(self):
        return {**self.provenance, "bounds": self.mesh.bounds.tolist(), "triangles": len(self.mesh.faces)}

    def scene(self, excluded=()):
        import trimesh
        excluded = set(excluded)
        known = {c["id"] for c in self.provenance["components"]}
        if not excluded <= known:
            raise ValueError("Unknown component ID; inspect this exact reference before excluding components")
        if excluded == known:
            raise ValueError("Keep at least one reference component")
        scene = trimesh.Scene()
        for component, indices in zip(self.provenance["components"], self.face_groups):
            if component["id"] in excluded:
                continue
            # Submeshing the seam-preserving mesh copies its UVs. Never export the welded analysis copy.
            part = self.mesh.submesh([indices], append=True, repair=False)
            part.metadata.clear()
            scene.add_geometry(part, node_name=component["id"], geom_name=component["id"])
        scene.metadata["units"] = "mm"
        return scene


def read_bytes(filename, body, units=None, sidecars=None):
    import trimesh
    assets = _assets(filename, body, sidecars)
    candidates = [name for name in assets if scan.reference_suffix(name) in {".ply", ".ply.gz"}]
    if len(candidates) != 1:
        raise ValueError("Choose a bundle with exactly one PLY or PLY.GZ mesh; multiple meshes are ambiguous")
    ply = candidates[0]
    texture = texture_name(ply, assets[ply])
    image_path = safe_name(str(pathlib.PurePosixPath(ply).parent / texture)) if texture else None
    images = [name for name in assets if pathlib.Path(name).suffix.lower() in IMAGE_SUFFIXES]
    if not texture and images:
        raise ValueError("PLY has no TextureFile declaration; add the exact image name to its header before importing")
    if texture:
        if pathlib.Path(texture).suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError("PLY textures must be PNG or JPEG images")
        if image_path not in assets:
            raise ValueError(f"Missing PLY texture {image_path!r}; select the matching image or include it in the ZIP")
        from PIL import Image
        try:
            with Image.open(io.BytesIO(assets[image_path])) as image:
                if image.format not in {"PNG", "JPEG"} or image.width * image.height > 32_000_000:
                    raise ValueError("Texture must be PNG/JPEG with at most 32 million pixels")
                image.verify()
        except (OSError, Image.DecompressionBombError) as exc:
            raise ValueError(f"Could not read PLY texture: {exc}") from exc
    ply_body = _ply_body(ply, assets[ply])
    if len(ply_body) + sum(len(data) for name, data in assets.items() if name != ply) > MAX_BYTES:
        raise ValueError("Expanded reference assets exceed the 256 MiB import limit")
    try:
        resolver = {texture: assets[image_path]} if texture else {}
        mesh = trimesh.load(io.BytesIO(ply_body), file_type="ply", force="mesh",
                            process=False, resolver=resolver, fix_texture=True)
    except Exception as exc:
        raise ValueError(f"Could not read PLY mesh: {exc}") from exc
    if not hasattr(mesh, "faces") or not len(mesh.faces):
        raise ValueError("PLY contains only points; export a triangle mesh")
    declared_faces = mesh.metadata.get("_ply_raw", {}).get("face", {}).get("length")
    if declared_faces != len(mesh.faces):
        raise ValueError("PLY faces must be triangles; triangulate the mesh before importing")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("PLY contains non-finite vertex coordinates")
    if texture and (mesh.visual.kind != "texture" or mesh.visual.uv is None
                    or getattr(mesh.visual.material, "image", None) is None):
        raise ValueError("Textured PLY needs valid per-face texcoord or per-vertex UV coordinates")
    if texture and not np.isfinite(mesh.visual.uv).all():
        raise ValueError("PLY contains non-finite texture coordinates")
    unit = units or ("m" if float(mesh.extents.max()) < scan.METRES_BELOW else "mm")
    if unit not in scan.UNITS:
        raise ValueError("Reference units must be mm, cm, m, or in")
    mesh.apply_scale(scan.UNITS[unit])
    # Raw parser arrays are preserved in source assets, not copied into every component.
    mesh.metadata.clear()
    welded = mesh.copy()
    welded.merge_vertices(merge_tex=True, merge_norm=True)
    groups = list(trimesh.graph.connected_components(welded.face_adjacency, nodes=np.arange(len(mesh.faces)), engine="scipy"))
    groups.sort(key=lambda faces: (-len(faces), int(np.min(faces))))
    grouped = len(groups) > 256
    if grouped:
        groups = groups[:255] + [np.concatenate(groups[255:])]
    components = []
    for index, faces in enumerate(groups):
        faces = np.sort(faces)
        groups[index] = faces
        points = mesh.vertices[mesh.faces[faces].reshape(-1)]
        components.append({"id": f"component-{index + 1}", "triangles": len(faces),
                           "bounds_mm": [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
                           "grouped_fragments": grouped and index == 255})
    asset_records = [{"path": name, "sha256": digest(data), "bytes": len(data)} for name, data in sorted(assets.items())]
    identity = digest(json.dumps(asset_records, sort_keys=True).encode())
    provenance = {"schema_version": 1, "source": {"name": filename, "sha256": digest(body)},
                  "assets": asset_records, "asset_identity": identity,
                  "units": {"input": unit, "source": "argument" if units else "guess", "scale_to_mm": scan.UNITS[unit]},
                  "coordinate_frame": "original reference coordinates scaled to millimetres",
                  "texture": {"path": image_path, "sha256": digest(assets[image_path])} if texture else None,
                  "mesh": ply, "components": components,
                  "operations": ["Scale coordinates to mm", "Split vertices at UV seams; preserve original triangles"]}
    return ReferenceImport(filename, body, assets, mesh, provenance, groups)


def read(path, units=None):
    path = pathlib.Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("Reference exceeds the 256 MiB import limit")
    body = path.read_bytes()
    sidecars = {}
    if len(body) > MAX_BYTES:
        raise ValueError("Reference exceeds the 256 MiB import limit")
    if scan.reference_suffix(path) in {".ply", ".ply.gz"}:
        texture = texture_name(path.name, body)
        if texture:
            image = (path.parent / texture).resolve()
            if not image.is_relative_to(path.parent.resolve()):
                raise ValueError("PLY texture must stay inside its source directory")
            if not image.is_file():
                raise ValueError(f"Missing PLY texture {texture!r}; keep the image beside the PLY or use a ZIP bundle")
            if image.stat().st_size > MAX_BYTES:
                raise ValueError("Texture exceeds the import size limit")
            sidecars[texture] = image.read_bytes()
    return read_bytes(path.name, body, units, sidecars)


def save(imported, root, excluded=()):
    root = pathlib.Path(root).resolve()
    excluded = sorted(set(excluded))
    scene = imported.scene(excluded)
    policy = {"asset_identity": imported.provenance["asset_identity"], "units": imported.provenance["units"], "excluded": excluded}
    key = digest(json.dumps(policy, sort_keys=True).encode())[:20]
    source_identity = digest(json.dumps([imported.provenance["asset_identity"], imported.provenance["source"]], sort_keys=True).encode())
    folder = root / "scans" / ("import-" + source_identity[:20])
    if not folder.resolve().is_relative_to(root):
        raise ValueError("Unsafe reference destination; scans must be inside the project")
    assets_folder = folder / "assets"
    items = {folder / "original" / imported.filename: imported.original}
    items.update({assets_folder / name: body for name, body in imported.assets.items()})
    glb = scene.export(file_type="glb")
    output = folder / f"reference-{key}.glb"
    record = {**imported.provenance, "excluded_components": excluded, "derived_sha256": digest(glb),
              "original": f"original/{imported.filename}", "assets_directory": "assets", "output_units": "mm"}
    items[output] = glb
    items[output.with_suffix(".import.json")] = (json.dumps(record, indent=2) + "\n").encode()
    for path, body in items.items():
        if not path.resolve().is_relative_to(folder.resolve()):
            raise ValueError("Unsafe reference asset destination")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != body:
                raise ValueError("Stored reference assets changed; import into a fresh project or restore the original bytes")
        else:
            with path.open("xb") as stream:
                stream.write(body)
    return output.relative_to(root).as_posix(), record


def manifest(path):
    path = pathlib.Path(path)
    record_path = path.with_suffix(".import.json")
    if not record_path.is_file():
        return None
    record = json.loads(record_path.read_text())
    if record.get("derived_sha256") != digest(path.read_bytes()):
        raise ValueError("Imported reference differs from its provenance manifest; reimport the original assets")
    return record


def reopen(path):
    path = pathlib.Path(path)
    record = manifest(path)
    if not record:
        raise ValueError("This reference has no preserved PLY source bundle; import a PLY or ZIP first")
    base = path.parent.resolve()
    assets = {}
    for asset in record["assets"]:
        source = (base / "assets" / safe_name(asset["path"])).resolve()
        if not source.is_relative_to(base):
            raise ValueError("Unsafe stored reference asset path")
        data = source.read_bytes()
        if digest(data) != asset["sha256"]:
            raise ValueError("Original reference asset changed; restore the recorded source before filtering components")
        assets[asset["path"]] = data
    original = (base / safe_name(record["original"])).resolve()
    if not original.is_relative_to(base):
        raise ValueError("Unsafe stored reference source path")
    body = original.read_bytes()
    if digest(body) != record["source"]["sha256"]:
        raise ValueError("Original reference source changed; restore it before filtering components")
    return read_bytes(record["source"]["name"], body, record["units"]["input"],
                      None if record["source"]["name"].lower().endswith(".zip") else {k: v for k, v in assets.items() if k != record["mesh"]})
