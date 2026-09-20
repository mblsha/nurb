import assert from "node:assert/strict";
import test from "node:test";
import { reconstructionPrompt, referenceDimensions, referenceProjectName, validReferenceTolerance, type ReferenceInfo } from "../src/referenceProject.ts";

test("comparison tolerance accepts the engine minimum and rejects smaller or non-finite values", () => {
  assert.equal(validReferenceTolerance(0.001), true);
  assert.equal(validReferenceTolerance(0.1), true);
  for (const value of [0.000999999, 0, -0.1, Number.NaN, Number.POSITIVE_INFINITY]) {
    assert.equal(validReferenceTolerance(value), false);
  }
});

test("small STL dimensions stay millimetres unless the user chooses another unit", () => {
  const info: ReferenceInfo = { bounds: [[10, -5, 100], [12, -2, 104]], triangles: 12 };
  assert.deepEqual(referenceDimensions(info, "mm"), [2, 3, 4]);
  assert.deepEqual(referenceDimensions(info, "m"), [2000, 3000, 4000]);
  assert.equal(referenceDimensions(info, "in")[0], 50.8);
});

test("the proposed project name uses only the STL filename", () => {
  assert.equal(referenceProjectName("/downloads/Bracket v2.STL"), "Bracket v2");
  assert.equal(referenceProjectName("C:\\Downloads\\clip.stl"), "clip");
  assert.equal(referenceProjectName("/downloads/.stl"), "STL reconstruction");
});

test("the reconstruction handoff preserves the comparison contract", () => {
  const prompt = reconstructionPrompt("bracket", {
    file: "scans/original.stl",
    units: "in",
    tolerance_mm: 0.15,
    alignment: "stored",
    transform: [1, 0, 0, 25.4, 0, 1, 0, -12.7, 0, 0, 1, 0, 0, 0, 0, 1],
  });
  for (const expected of [
    "editable analytic CAD",
    "bounding-box draft",
    "scans/original.stl",
    "uses in",
    "0.15 mm",
    "[1, 0, 0, 25.4, 0, 1, 0, -12.7, 0, 0, 1, 0, 0, 0, 0, 1]",
    "Do not use import_stl",
    "Preserve sharp edges",
    "both directions",
    "worst discrepancy regions",
    "sampled maxima",
  ]) {
    assert.match(prompt, new RegExp(expected.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));
  }
});

test("the handoff does not invent an alignment when the server has none", () => {
  const prompt = reconstructionPrompt("clip", { file: "scans/clip.stl" });
  assert.match(prompt, /save a deliberate rigid alignment/);
  assert.doesNotMatch(prompt, /\[1, 0, 0/);
});

test("the handoff does not describe a session auto-center as saved", () => {
  const prompt = reconstructionPrompt("clip", {
    file: "scans/clip.stl",
    alignment: "auto",
    transform: [1, 0, 0, 12, 0, 1, 0, 3, 0, 0, 1, -4, 0, 0, 0, 1],
  });
  assert.match(prompt, /save a deliberate rigid alignment/);
  assert.doesNotMatch(prompt, /\[1, 0, 0, 12/);
});
