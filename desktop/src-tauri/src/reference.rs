use std::path::Path;

use crate::env::Launcher;

const MIN_TOLERANCE_MM: f64 = 0.001;

#[derive(serde::Deserialize, serde::Serialize)]
pub(crate) struct ReferenceInfo {
    pub bounds: [[f64; 3]; 2],
    pub triangles: usize,
    pub texture: Option<String>,
    pub components: Option<usize>,
}

fn source_suffix(source: &Path) -> Result<&'static str, String> {
    let name = source
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_ascii_lowercase();
    [".ply.gz", ".ply", ".stl", ".obj", ".glb", ".zip", ".step", ".stp", ".brep"]
        .into_iter()
        .find(|suffix| name.ends_with(suffix))
        .ok_or_else(|| "Choose STL, OBJ, GLB, PLY, .ply.gz, STEP, BREP, or a textured PLY ZIP bundle.".into())
}

pub(crate) fn inspect(launcher: &Launcher, source: &Path) -> Result<ReferenceInfo, String> {
    if !source.is_file() {
        return Err("The selected mesh is missing. Choose the file again.".into());
    }
    source_suffix(source)?;
    // Explicit millimetres here preserve the file coordinates for the unit picker.
    // The scan reader's size heuristic is appropriate for scans, not small parts.
    let output = launcher.python().args(["-c", r#"
import json, sys
from nurb import scan
try:
    mesh, _, _ = scan.load(sys.argv[1], units="mm")
    imported = mesh.metadata.get("reference_import", {})
    print(json.dumps({"bounds": mesh.bounds.tolist(), "triangles": len(mesh.faces), "texture": (imported.get("texture") or {}).get("path"), "components": len(imported.get("components", []))}, allow_nan=False))
except Exception as exc:
    print(str(exc), file=sys.stderr)
    sys.exit(1)
"#]).arg(source).output().map_err(|e| format!("Could not read the mesh with the CAD engine: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "Could not read this mesh: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let info: ReferenceInfo = serde_json::from_slice(&output.stdout)
        .map_err(|e| format!("Could not read the mesh dimensions: {e}"))?;
    validate_bounds(&info)?;
    Ok(info)
}

fn validate_bounds(info: &ReferenceInfo) -> Result<(), String> {
    if info.triangles == 0
        || !info.bounds.iter().flatten().all(|v| v.is_finite())
        || (0..3).any(|axis| info.bounds[1][axis] < info.bounds[0][axis])
        || (0..3).all(|axis| info.bounds[1][axis] == info.bounds[0][axis])
    {
        return Err(
            "This mesh has no usable surface. Export a mesh with triangles and try again.".into(),
        );
    }
    Ok(())
}

pub(crate) fn unit_factor(units: &str) -> Result<f64, String> {
    match units {
        "mm" => Ok(1.0),
        "cm" => Ok(10.0),
        "m" => Ok(1000.0),
        "in" => Ok(25.4),
        _ => Err("Choose millimetres, centimetres, metres, or inches for the mesh units.".into()),
    }
}

pub(crate) fn validate_tolerance(tolerance_mm: f64) -> Result<(), String> {
    if !tolerance_mm.is_finite() || tolerance_mm < MIN_TOLERANCE_MM {
        return Err("Comparison tolerance must be at least 0.001 mm.".into());
    }
    Ok(())
}

pub(crate) fn create(launcher: &Launcher, dir: &Path, module: &str, source: &Path, units: &str, tolerance_mm: f64) -> Result<(), String> {
    source_suffix(source)?;
    unit_factor(units)?;
    validate_tolerance(tolerance_mm)?;
    // Use the engine's one import path so sidecars, UV seams and provenance cannot drift between app and CLI.
    let output = launcher.nurb().args(["new", module, "--embed", "--root"]).arg(dir)
        .arg("--from").arg(source).args(["--units", units, "--tolerance", &tolerance_mm.to_string()])
        .current_dir(dir).output().map_err(|e| format!("Could not import the reference: {e}"))?;
    if !output.status.success() {
        return Err(format!("Could not create the reference project: {}", String::from_utf8_lossy(&output.stderr).trim()));
    }
    if !dir.join("parts").join(format!("{module}.py")).is_file() {
        return Err("The CAD engine did not create the editable reference part.".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_formats_match_the_engine_and_compound_suffixes_stay_intact() {
        assert_eq!(source_suffix(Path::new("scan.PLY.GZ")).unwrap(), ".ply.gz");
        for name in ["scan.ply", "scan.stl", "scan.zip", "scan.glb", "scan.obj", "scan.step", "scan.stp", "scan.brep"] {
            assert!(source_suffix(Path::new(name)).is_ok());
        }
        for name in ["scan.gz", "scan.stl.gz", "scan.ply.gz.backup"] {
            assert!(source_suffix(Path::new(name)).is_err());
        }
    }

    #[test]
    fn invalid_settings_fail_before_spawning_the_engine() {
        for value in [0.0, -0.1, 0.000999999, f64::NAN, f64::INFINITY] {
            assert!(validate_tolerance(value).is_err());
        }
        assert!(validate_tolerance(0.001).is_ok());
        assert_eq!(unit_factor("in").unwrap(), 25.4);
        assert!(unit_factor("guess").is_err());
        let info = ReferenceInfo {bounds:[[0.0;3],[1.0;3]],triangles:12,texture:None,components:None};
        assert!(validate_bounds(&info).is_ok());
    }
}
