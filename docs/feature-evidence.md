# Functional feature evidence and local sections

A region can identify a functional interface as well as a volume to compare. Give it a stable feature ID and a real-world role, such as “narrow rim connecting to the headset”, before interpreting its fit. Orientation, configuration, required and excluded details, reference annotations, uncertainty and evidence links travel with the region. A corrected role invalidates an old inspection even when its geometry did not change.

In the browser or desktop, open **Compare → Named inspection regions → Functional feature and local sections**. Select or create a region, enable feature tracking and save it. Enter a plane origin, normal and horizontal direction in the aligned part frame. Station offsets move parallel planes along the normalized normal. Zero contour simplification preserves small shoulders and lips. **Inspect saved feature sections** overlays CAD in amber and the reference in blue in one local millimetre frame. Cuts are restricted to the named region without artificial caps. Open contours stay open; this view does not invent material or certify a mating fit. The editor changes the first saved series and preserves any additional series supplied through the CLI.

**Record this inspection** stores the exact geometry fingerprint, reference content and units, alignment, configuration and feature contract. Changing any of them makes the record visibly stale. A rebuild immediately clears displayed section plots, including a rebuild that ultimately produces identical geometry; an unchanged geometry can retain its recorded inspection. Links are recorded as text and are never fetched automatically. A reference annotation is an original-reference point in millimetres, not a transformed CAD point.

The live section view uses the cached display mesh. It can omit small details that are present in the CAD. The separate **Verify with a finer CAD mesh** action requests a bounded precise comparison; requested deflection is not an achieved error bound. Review each important local feature and retain uncertainty or fit-coupon evidence alongside sampled distances.

The existing X/Y/Z APIs still work. An arbitrary plane and a parallel station series can be measured without a project:

```sh
nurb scan original.ply.gz --units mm --json --local-section '{"name":"rim","origin_mm":[10,0,3],"normal":[1,1,0],"x_direction":[0,0,1],"offsets_mm":[-1,0,1],"tolerance_mm":0}'
```

Local section output carries `origin_mm`, `normal` and a two-vector `basis`. Recover a 3D point as `origin_mm + u * basis[0] + v * basis[1]`. The old axis sections retain their original axis-coordinate convention. STEP/B-rep inspection also reports exact local section material area and hole count. `scan.section(mesh, definition)` accepts one local plane; `feature_evidence.sections(mesh, definitions)` expands station series. These Python helpers are internal APIs.

A JSON region file can persist the full feature contract:

```json
[
  {
    "name": "Narrow headset rim",
    "bounds_mm": {"min": [-80,-50,10], "max": [80,50,60]},
    "feature": {
      "id": "headset-rim",
      "role": "connects the narrow rim to the headset",
      "orientation": "narrow end",
      "configuration": "assembled",
      "reference_point_mm": [70,0,25],
      "required": ["small inverted-T lip with two shoulders"],
      "excluded": ["nose cloth"],
      "uncertainty_mm": 0.15,
      "links": ["references/rim-profile.png"],
      "sections": [
        {"name":"temple", "origin_mm":[70,0,25], "normal":[0,1,0], "x_direction":[1,0,0], "offsets_mm":[-1,0,1], "tolerance_mm":0, "expected":"small inverted T"}
      ]
    }
  }
]
```

```sh
nurb compare light_seal --regions-file regions.json --save-regions --json
```

The output includes `feature_evidence`, identity tokens, freshness, and aligned CAD/reference local contours. An ordinary `--region` remains supported. Feature IDs must be unique and remain stable when the user-facing name changes. Each feature supports up to eight series and 24 total stations. The server's `feature_inspection` command selects a saved feature by ID and current build token; recording additionally requires the identity from the completed inspection. A changed build, reference, alignment or card rejects stale recording requests.
