# STL reconstruction example

This project reconstructs a coarse 32-sided STL disk as an editable analytic cylinder. The original synthetic reference is 40mm in diameter and 8mm tall. Its chord error is about 0.0963mm, so the card's 0.15mm acceptance band ignores the faceting. The `oversized` variant adds a deliberate 1mm radius error that must still be detected.

The `offset_bracket` exercise is deliberately less forgiving. Its reference frame starts at x = 12mm and y = -7mm, it contains base and side holes, and the `missing_locator` variant removes a small edge notch. It exercises fixed alignment and localized defect reporting even when aggregate sampled coverage remains high.

The reference meshes are deterministic synthetic fixtures, generated from explicit geometry in `generate_fixtures.py`. From this directory, run `uv run --project ../.. python generate_fixtures.py` to regenerate them or add `--check` to detect drift without writing files.

From this directory, run:

```bash
nurb dev
nurb compare disk --json
nurb compare offset_bracket --json
```

Open **compare** in the viewer. The default disk has near-zero excess beyond tolerance even though raw surface distances are nonzero. Select `oversized` to see lower coverage and colored samples on the cylinder's sides. The CAD and reference map buttons inspect the two unsigned directions independently. `nurb compare disk` reports both the default and the variant.

The same panel can save a named inspection box or assembly component, so a functional interface keeps its own bidirectional result even when an intentionally simplified exterior dominates the global score. Section mode applies one part-frame plane to both bodies and shows shared, CAD-only, and reference-only material in a matched orthographic view. Model, reference, overlay, deviation, side-by-side, and section views can also be captured with `nurb render --mode`; use `--region NAME` to frame a saved region and repeat `--hide COMPONENT` to omit assembly context without changing geometry.

For `offset_bracket`, keep the stored identity frame and inspect the base holes, side hole and locator notch. The default must remain inside its 0.12mm band. The `missing_locator` variant must report detected deviation above tolerance and locate it at the reference notch rather than allowing high overall coverage to imply success.

Measure it without writing a mesh-specific probe:

```bash
nurb scan scans/offset_bracket.stl --units mm --section z:2.5mm --section x:15mm --json
```

The result contains full-precision bounds in the offset source frame, complete section loops and fit residuals. `nurb compare offset_bracket --alignment stored --json` must retain that frame. A deliberate frame change can be previewed in the viewer or requested with `--alignment identity|center`; add `--save-alignment` only when the replacement transform is intentional.

To start another reconstruction from the same reference in a new directory:

```bash
nurb new rebuilt_disk --from /path/to/disk.stl --units mm --tolerance 0.15
nurb dev
```

That command creates a bounding-box draft to replace with CAD; it does not recover design parameters automatically. The source copy, explicit units, acceptance tolerance and identity transform live in the part card. Keep the reconstruction in that frame. A deliberate alignment correction belongs in the stored transform, not in a tolerance large enough to conceal the mismatch.

The acceptance band is separate from section-profile simplification (`nurb scan --tolerance`). It leaves the CAD and reference geometry intact; the comparison engine chooses a finer tessellation budget when needed for that band. Coverage is sampled surface coverage, and the sampled maximum is not a certified bound. The tests in `tests/test_reverse_engineering_example.py` check the real part, card, STL and both configurations together.
