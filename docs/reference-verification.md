# Reference verification

The live comparison reuses the displayed mesh so editing remains responsive. Its policy is **Display mesh estimate**: absolute accuracy in millimetres is unknown, even when a relative display setting has a small numeric value. Use **Absolute mesh verification** before assessing a small mating lip, recess, or dovetail. The comparison panel's verification action uses the same policy as the CLI.

```bash
nurb compare bracket --mesh-accuracy 0.02 --feature-size 0.3 --mesh-timeout 30 --json
```

`--tolerance` is the acceptable surface deviation. `--mesh-accuracy` is the requested absolute OCCT deflection in millimetres, defaulting to the smaller of 0.025 mm and one quarter of the acceptance tolerance. `--feature-size` states the smallest feature being assessed. The result warns when deflection exceeds one fifth of that feature or one quarter of the acceptance tolerance. It also warns when feature size is unspecified. These ratios are screening rules, not evidence that a feature exists or was reconstructed correctly.

Verification meshes a B-rep snapshot in a separate process. The default meshing budget is 30 seconds and one million triangles per mesh. `--mesh-timeout` accepts up to 120 seconds; `--mesh-triangles` accepts up to five million triangles. Component meshes share the remaining time budget with the whole model. A timeout, failed CAD face, or exceeded triangle budget yields **unknown**, with no automatic coarse fallback and no reused numerical result. The triangle budget limits the accepted mesh and extraction work; it is not a peak-memory guarantee for OCCT. Initial preview builds and other CAD operations still run in the main modeling process.

Results report requested absolute deflection, triangle count, feature-size ratio, policy warnings, and `measured_error_bound_mm: null`. OCCT's requested deflection is not an independently measured error bound. Distances and coverage are sampled, and the source scan's own uncertainty remains separate. A result called `measured` means the requested computation completed, not that the interface has passed physical fit testing.

The viewer's verification job returns immediately and leaves the build lock free while meshing. Progress identifies preparation and measurement phases without inventing a percentage. Results carry the build token, displayed shape identity, source and card identities, and reference stamp. A changed model, source dependency, reference, transform, tolerance, or region invalidates completion. New requests clear previous verification numbers; failed live comparisons also discard old display metrics.

Static assembly clearance failures remain distinct from measured collisions. If OCCT cannot intersect a declared pair or measure its distance, its finding says **Clearance unknown**, identifies both components, and carries `measurements.status = "unknown"`. It provides no fabricated zero distance or overlap. The check fails because verification is incomplete. Valid empty intersections still mean no overlap.

## Viewer protocol

Send `target_verify` with `name` and optional `accuracy_mm`, `feature_size_mm`, `timeout_s`, and `max_triangles`. Responses use `target_verification`, a build `token`, and a per-request `request_id`. States are `queued`, `running`, `cancelling`, `cancelled`, `measured`, `unknown`, or `stale`; `phase` describes progress. Only `measured` includes `metrics`. Failed or superseded results contain `error`. `target.verification` persists the latest result separately from the cached live `target.metrics` estimate. The viewer displays this panel in embedded desktop mode too; the desktop rail does not duplicate numerical comparison results.

Send `target_verify_cancel` with the same `name`, `token`, and `request_id` to stop an active request. A matching request first reports `cancelling`; the child is killed and reaped, then the job reports `cancelled` with no metrics. Tokens from earlier requests cannot cancel a later job. Cancellation is checked between distance-query phases as well as while meshing.

A portable peak-memory guard for OCCT is a follow-up: address-space limits are not equivalent to resident-memory limits and can reject normal library mappings on supported platforms. Until that guard is implemented, verification reports the output triangle budget without claiming a memory ceiling.
