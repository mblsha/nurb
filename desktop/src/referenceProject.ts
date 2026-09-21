export type ReferenceUnit = "mm" | "cm" | "m" | "in";

export const MIN_REFERENCE_TOLERANCE_MM = 0.001;

export function validReferenceTolerance(value: number): boolean {
  return Number.isFinite(value) && value >= MIN_REFERENCE_TOLERANCE_MM;
}

export type ReferenceInfo = {
  bounds: [[number, number, number], [number, number, number]];
  triangles: number;
  texture?: string | null;
  components?: number;
};

export type ReferenceProject = {
  name: string;
  source: string;
  units: ReferenceUnit;
  toleranceMm: number;
};

// The target payload comes from the viewer server's /api/parts response. Keep
// optional fields optional so the desktop can still show a useful reference state
// while a build or comparison is in flight.
export type PartReference = {
  file?: string;
  units?: ReferenceUnit | null;
  tolerance_mm?: number;
  transform?: number[] | null;
  alignment?: string;
  error?: string;
};

const UNIT_MM: Record<ReferenceUnit, number> = { mm: 1, cm: 10, m: 1000, in: 25.4 };

export function referenceFileName(source: string): string {
  return source.split(/[\\/]/).pop() ?? source;
}

export function referenceMeshSuffix(source: string): string | null {
  return referenceFileName(source).toLowerCase().match(/\.(stl|obj|glb|ply(?:\.gz)?|zip|step|stp|brep)$/)?.[0] ?? null;
}

export function referenceProjectName(source: string): string {
  const filename = referenceFileName(source);
  const suffix = referenceMeshSuffix(source);
  return (suffix ? filename.slice(0, -suffix.length) : filename).replace(/^\.+/, "").trim() || "Mesh reconstruction";
}

export function referenceDimensions(info: ReferenceInfo, units: ReferenceUnit): number[] {
  return info.bounds[1].map((maximum, axis) => (maximum - info.bounds[0][axis]) * UNIT_MM[units]);
}

function referenceTransform(reference: PartReference): string {
  const transform = reference.transform;
  if (reference.alignment !== "stored" || !Array.isArray(transform) || transform.length !== 16 || !transform.every(Number.isFinite)) {
    return "The current target alignment is not saved in the card. Inspect the overlay and save a deliberate rigid alignment before changing geometry, then keep that alignment fixed.";
  }
  return `Keep this saved row-major target transform fixed while the CAD changes: [${transform.join(", ")}].`;
}

export function reconstructionPrompt(part: string, reference: PartReference): string {
  const file = reference.file || "the copied project reference";
  const units = reference.units || "the units recorded in the part card";
  const tolerance = Number.isFinite(reference.tolerance_mm) ? `${reference.tolerance_mm} mm` : "the tolerance recorded in the part card";
  return [
    `Rebuild ${part} from the copied reference ${file} as editable analytic CAD.`,
    `The shape on screen is only a bounding-box draft. The reference uses ${units}, and differences above ${tolerance} need inspection.`,
    referenceTransform(reference),
    "Measure the reference and model its planes, holes, cylinders, walls, and repeated features with meaningful keyword defaults. Do not use import_stl as the finished part or trace triangle facets.",
    "Preserve sharp edges, functional clearances, and small functional features from the reference unless there is evidence they are mesh noise. Keep the reference coordinate frame.",
    "Once analytic reconstruction has started, remove the reconstruction = \"bounding_box_draft\" marker from the part card so the app no longer describes the box as untouched.",
    "Compare the finished CAD against the reference in both directions. Inspect and correct the worst discrepancy regions, then report sampled coverage, both directional sampled maxima, tolerance excess, and any deliberate residual differences without claiming an exact match from rounded coverage.",
  ].join("\n\n");
}
