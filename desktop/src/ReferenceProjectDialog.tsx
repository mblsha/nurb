import { FormEvent, useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { MIN_REFERENCE_TOLERANCE_MM, referenceDimensions, referenceFileName, referenceProjectName, validReferenceTolerance, type ReferenceInfo, type ReferenceProject, type ReferenceUnit } from "./referenceProject";

type Props = {
  source: string;
  creating: boolean;
  error: string | null;
  onCreate: (project: ReferenceProject) => void;
  onClose: () => void;
};

export default function ReferenceProjectDialog({ source, creating, error, onCreate, onClose }: Props) {
  const [name, setName] = useState(() => referenceProjectName(source));
  const fixedUnits = /\.(step|stp|brep)$/i.test(source);
  const [units, setUnits] = useState<ReferenceUnit>("mm");
  const [tolerance, setTolerance] = useState("0.1");
  const [info, setInfo] = useState<ReferenceInfo | null>(null);
  const [readError, setReadError] = useState<string | null>(null);
  useEffect(() => {
    let current = true;
    invoke<ReferenceInfo>("inspect_reference", { source }).then(
      (result) => { if (current) setInfo(result); },
      (failure) => { if (current) setReadError(String(failure)); },
    );
    return () => { current = false; };
  }, [source]);

  const toleranceMm = Number(tolerance);
  const valid = name.trim().length > 0 && validReferenceTolerance(toleranceMm) && info !== null;
  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (valid && !creating) onCreate({ name: name.trim(), source, units, toleranceMm });
  };
  const close = () => { if (!creating) onClose(); };

  return (
    <div className="about" onClick={(event) => event.target === event.currentTarget && close()}>
      <form className="about-card reference-project-dialog" role="dialog" aria-modal="true" aria-labelledby="reference-project-title" onSubmit={submit} onKeyDown={(event) => { if (event.key === "Escape") close(); }}>
        <button type="button" className="about-close" title="close" disabled={creating} onClick={close}>×</button>
        <div className="about-title" id="reference-project-title">New from mesh</div>
        <div className="about-body">
          <p>Keep the original as a reference and rebuild an editable CAD model. Start with a box matching its bounds.</p>
          <p>For a textured PLY, keep the declared PNG/JPEG beside it or choose their ZIP bundle. Source files and image hashes are preserved.</p>
          <div className="reference-file">{referenceFileName(source)}</div>
          <label className="api-key-label" htmlFor="reference-project-name">Project name</label>
          <input className="api-key-input" id="reference-project-name" value={name} onChange={(event) => setName(event.target.value)} disabled={creating} autoFocus required autoComplete="off" />
          <label className="api-key-label" htmlFor="reference-units">Original file units</label>
          <select className="api-key-input" id="reference-units" value={units} onChange={(event) => setUnits(event.target.value as ReferenceUnit)} disabled={creating || fixedUnits}>
            <option value="mm">Millimetres</option>
            <option value="cm">Centimetres</option>
            <option value="m">Metres</option>
            <option value="in">Inches</option>
          </select>
          <p className="reference-size" aria-live="polite">
            {info ? `${referenceDimensions(info, units).map((size) => Number(size.toPrecision(6)).toLocaleString(undefined, { maximumSignificantDigits: 6 })).join(" × ")} mm` : readError ? "Dimensions unavailable" : "Reading mesh dimensions…"}
          </p>
          {info?.texture && <p className="reference-hint">Texture: {info.texture}. {info.components} connected component groups. Review and exclude detached noise in Compare → Source assets and detached components after opening the project.</p>}
          <p className="reference-hint">Confirm that these dimensions match the original. Choose the units used when exporting the mesh.</p>
          <label className="api-key-label" htmlFor="reference-tolerance">Ignore differences within (mm)</label>
          <input className="api-key-input" id="reference-tolerance" type="number" min={MIN_REFERENCE_TOLERANCE_MM} step="any" value={tolerance} onChange={(event) => setTolerance(event.target.value)} disabled={creating} required />
          <p className="reference-hint">Allow small differences from the mesh's flat triangles. Minimum 0.001 mm. You can adjust this in the comparison panel.</p>
          <p className="reference-hint">Creating the project saves the reference and opens its bounding-box draft. Choose Rebuild as editable CAD beside the part when you are ready to prepare the agent instructions.</p>
          {(readError || error) && <p className="reference-error" role="alert">{readError || error}</p>}
          <div className="settings-actions">
            <button className="settings-action" type="submit" disabled={!valid || creating}>{creating ? "Creating…" : "Create project"}</button>
            <button className="settings-action secondary" type="button" disabled={creating} onClick={close}>Cancel</button>
          </div>
        </div>
      </form>
    </div>
  );
}
