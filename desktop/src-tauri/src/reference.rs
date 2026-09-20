use std::fs::{self, OpenOptions};
use std::path::Path;

use crate::env::Launcher;

const MIN_TOLERANCE_MM: f64 = 0.001;

#[derive(serde::Deserialize, serde::Serialize)]
pub(crate) struct ReferenceInfo {
    pub bounds: [[f64; 3]; 2],
    pub triangles: usize,
}

fn source_suffix(source: &Path) -> Result<&'static str, String> {
    let name = source
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_ascii_lowercase();
    [".ply.gz", ".ply", ".stl"]
        .into_iter()
        .find(|suffix| name.ends_with(suffix))
        .ok_or_else(|| "Choose an STL, PLY, or .ply.gz mesh reference.".into())
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
    print(json.dumps({"bounds": mesh.bounds.tolist(), "triangles": len(mesh.faces)}, allow_nan=False))
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

pub(crate) fn copy_source(dir: &Path, source: &Path) -> Result<std::path::PathBuf, String> {
    let suffix = source_suffix(source)?;
    let scans = dir.join("scans");
    fs::create_dir_all(&scans)
        .map_err(|e| format!("Could not create the reference folder: {e}"))?;
    let copied = scans.join(format!("original{suffix}"));
    let mut input =
        fs::File::open(source).map_err(|e| format!("Could not open the selected mesh: {e}"))?;
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&copied)
        .map_err(|e| format!("Could not copy the mesh into the new project: {e}"))?;
    std::io::copy(&mut input, &mut output).map_err(|e| format!("Could not copy the mesh: {e}"))?;
    Ok(copied)
}

pub(crate) fn write_part(
    dir: &Path,
    module: &str,
    units: &str,
    tolerance_mm: f64,
    info: &ReferenceInfo,
    source: &Path,
) -> Result<(), String> {
    let suffix = source_suffix(source)?;
    validate_bounds(info)?;
    validate_tolerance(tolerance_mm)?;
    let factor = unit_factor(units)?;
    let size: Vec<f64> = (0..3)
        .map(|axis| ((info.bounds[1][axis] - info.bounds[0][axis]) * factor).max(0.1))
        .collect();
    let center: Vec<f64> = (0..3)
        .map(|axis| (info.bounds[0][axis] + info.bounds[1][axis]) * factor / 2.0)
        .collect();
    if !size.iter().chain(center.iter()).all(|v| v.is_finite()) {
        return Err(
            "These units make the mesh dimensions too large. Choose the file's actual units."
                .into(),
        );
    }
    let part = format!("from nurb import *\n\n\n@part\ndef {module}(width={:?}, depth={:?}, height={:?}, draft=False):\n    return Pos({:?}, {:?}, {:?}) * Box(width, depth, height)\n", size[0], size[1], size[2], center[0], center[1], center[2]);
    let card = format!("# {module}\n\n## What it is\n\nA bounding box placeholder for rebuilding the reference mesh as an editable CAD part. Replace the box with the measured features while keeping the reference coordinate frame.\n\n```toml\nreconstruction = \"bounding_box_draft\"\ntarget = {{ file = \"scans/original{suffix}\", units = \"{units}\", tolerance_mm = {tolerance_mm:?}, transform = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] }}\n```\n\n## Design notes\n\nThe reference is copied into this project. Differences within the comparison tolerance are acceptable; inspect remaining regions before declaring the reconstruction complete.\n\n## Don't\n\n## Changelog\n");
    fs::write(dir.join("parts").join(format!("{module}.py")), part)
        .map_err(|e| format!("Could not write the editable part: {e}"))?;
    fs::write(dir.join("parts").join(format!("{module}.md")), card)
        .map_err(|e| format!("Could not write the reference settings: {e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temporary() -> std::path::PathBuf {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "nurb-reference-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        ));
        fs::create_dir_all(dir.join("parts")).unwrap();
        dir
    }

    #[test]
    fn reference_project_keeps_units_frame_and_a_portable_copy() {
        let dir = temporary();
        let source = dir.join("selected.STL");
        fs::write(&source, b"original mesh bytes").unwrap();
        let copied = copy_source(&dir, &source).unwrap();
        assert_eq!(fs::read(&copied).unwrap(), b"original mesh bytes");
        let info = ReferenceInfo {
            bounds: [[1.0, -2.0, 0.0], [3.0, 1.0, 4.0]],
            triangles: 12,
        };
        write_part(&dir, "bracket", "in", 0.15, &info, &copied).unwrap();
        let part = fs::read_to_string(dir.join("parts/bracket.py")).unwrap();
        assert!(part.contains("width=50.8, depth=76.19999999999999, height=101.6"));
        assert!(part.contains("draft=False"));
        assert!(part.contains("Pos(50.8, -12.7, 50.8)"));
        let card = fs::read_to_string(dir.join("parts/bracket.md")).unwrap();
        assert!(card.contains("reconstruction = \"bounding_box_draft\""));
        assert!(card.contains("file = \"scans/original.stl\", units = \"in\", tolerance_mm = 0.15"));
        assert!(card.contains("transform = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]"));
        assert!(!card.contains(&source.to_string_lossy().to_string()));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn copying_a_reference_never_overwrites_a_previous_copy() {
        let dir = temporary();
        let source = dir.join("selected.stl");
        fs::write(&source, b"first").unwrap();
        let copied = copy_source(&dir, &source).unwrap();
        fs::write(&source, b"second").unwrap();
        assert!(copy_source(&dir, &source).is_err());
        assert_eq!(fs::read(copied).unwrap(), b"first");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn compressed_ply_keeps_its_suffix_bytes_and_card_reference() {
        let dir = temporary();
        let source = dir.join("scan.PLY.GZ");
        let original = [0x1f, 0x8b, 0x08, 0x00, 0x01, 0x02];
        fs::write(&source, original).unwrap();
        let copied = copy_source(&dir, &source).unwrap();
        assert_eq!(copied, dir.join("scans/original.ply.gz"));
        assert_eq!(fs::read(&copied).unwrap(), original);
        let info = ReferenceInfo {
            bounds: [[1.0, 2.0, 3.0], [3.0, 5.0, 7.0]],
            triangles: 12,
        };
        write_part(&dir, "scan", "cm", 0.1, &info, &copied).unwrap();
        let card = fs::read_to_string(dir.join("parts/scan.md")).unwrap();
        assert!(card.contains("file = \"scans/original.ply.gz\", units = \"cm\""));
        let part = fs::read_to_string(dir.join("parts/scan.py")).unwrap();
        assert!(part.contains("width=20.0, depth=30.0, height=40.0"));
        assert!(!dir.join("scans/original.stl").exists());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn source_suffix_rejects_unrelated_gzip_files_before_copying() {
        assert_eq!(source_suffix(Path::new("scan.PLY.GZ")).unwrap(), ".ply.gz");
        assert_eq!(source_suffix(Path::new("scan.ply")).unwrap(), ".ply");
        for name in ["scan.gz", "scan.stl.gz", "scan.ply.gz.backup", "scan.zip"] {
            assert!(source_suffix(Path::new(name)).is_err());
        }
        let dir = temporary();
        assert!(copy_source(&dir, Path::new("scan.gz")).is_err());
        assert!(!dir.join("scans").exists());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn invalid_reference_settings_fail_before_writing_a_part() {
        let dir = temporary();
        let info = ReferenceInfo {
            bounds: [[0.0; 3], [1.0; 3]],
            triangles: 12,
        };
        for tolerance in [0.0, -0.1, 0.000999999, f64::NAN, f64::INFINITY] {
            assert!(write_part(
                &dir,
                "part",
                "mm",
                tolerance,
                &info,
                Path::new("original.stl")
            )
            .is_err());
        }
        assert!(write_part(&dir, "part", "guess", 0.1, &info, Path::new("original.stl")).is_err());
        assert!(!dir.join("parts/part.py").exists());
        write_part(&dir, "part", "mm", 0.001, &info, Path::new("original.stl")).unwrap();
        assert!(fs::read_to_string(dir.join("parts/part.md"))
            .unwrap()
            .contains("tolerance_mm = 0.001"));
        fs::remove_dir_all(dir).unwrap();
    }
}
