"""Execute the comparison presentation against legacy and spatial payloads."""

import json
import pathlib
import shutil
import subprocess

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
VIEWER = (ROOT / "src/nurb/viewer.html").read_text(encoding="utf-8")
THREE = (ROOT / "src/nurb/vendor/three/build/three.module.min.js").as_uri()


def run_js(source):
    node = shutil.which("node")
    if not node:
        pytest.skip("viewer JavaScript checks need Node.js")
    result = subprocess.run(
        [node, "--input-type=module", "-"], input=source, encoding="utf-8", capture_output=True
    )
    assert result.returncode == 0, result.stderr


def function(start, end):
    return start + VIEWER.split(start, 1)[1].split(end, 1)[0]


def test_unsigned_metrics_keep_raw_deviation_and_do_not_invent_legacy_coverage():
    rows = function("function comparisonRows(", "function compareClear(")
    run_js(
        "import assert from 'node:assert/strict';\n"
        + rows
        + """
const old = comparisonRows({part: {max: 1.25, p95: .8}, target: {max: 2, p95: .9}});
assert.deepEqual(old[0], ['CAD → reference', '1.250', '0.800', 'n/a']);
assert.deepEqual(old[1], ['reference → CAD', '2.000', '0.900', 'n/a']);
const current = comparisonRows({part: {sampled_max: .15, max: .15, p95: .1, within_tolerance: .997}});
assert.deepEqual(current[0], ['CAD → reference', '0.150', '0.100', '99.7%']);
assert.equal(comparisonRows(null)[0][1], 'n/a');
"""
    )


def test_heatmap_ignores_tolerated_samples_and_clears_for_stale_geometry():
    clear = function("function compareClear(", "function compareInvalidate(")
    samples = function("function compareSamples(", "function comparePanel(")
    run_js(
        f"import * as THREE from {json.dumps(THREE)};\n"
        "import assert from 'node:assert/strict';\n"
        "const renderer = true, plane = new THREE.Plane(), compareMarks = new THREE.Group();\n"
        "const mesh = new THREE.Group(); mesh.userData.token = 'new'; mesh.position.set(2,3,4);\n"
        "let compareMap = 'both', comparePainted = null, compareAuto = false, compareScale = 1, compareThrough = true;\n"
        + clear
        + samples
        + """
const entry = {name: 'part', token: 'new', target: {stamp: 'ref'}};
const metrics = {part: {p95: .4}, target: {p95: .5}, samples: {
  part: {points: [[0,0,0], [1,0,0], [2,0,0]], distances: [.1,.2,.4]},
  target: {points: [[0,1,0], [0,2,0]], distances: [.5,NaN]},
}};
compareSamples(entry, metrics, .2);
assert.equal(compareMarks.children.length, 2);
assert.deepEqual(Array.from(compareMarks.children[0].geometry.attributes.position.array), [2,0,0]);
assert.deepEqual(Array.from(compareMarks.children[1].geometry.attributes.position.array), [0,1,0]);
assert.deepEqual(compareMarks.position.toArray(), [2,3,4]);
compareMap = 'target'; compareSamples(entry, metrics, .2);
assert.equal(compareMarks.children.length, 1);
assert.equal(compareMarks.children[0].name, 'comparison-target');
compareSamples({...entry, token: 'old'}, metrics, .2);
assert.equal(compareMarks.children.length, 0);
compareMap = 'both'; compareSamples(entry, metrics, 1);
assert.equal(compareMarks.children.length, 0);
"""
    )


def test_reference_transform_is_row_major_and_old_offsets_still_work():
    attach = function("async function ghostAttach(", "// ---- comparison panel ----")
    run_js(
        f"import * as THREE from {json.dumps(THREE)};\n"
        "import assert from 'node:assert/strict';\n"
        "const renderer = true, plane = new THREE.Plane(), mesh = new THREE.Group();\n"
        "const ghostGeo = new Map(), ghostMaterial = new THREE.MeshBasicMaterial();\n"
        "const current = 'part', ghostWanted = true, parts = new Map(), comparePreview = new Map();\n"
        "function comparePanel() {}\n"
        "const geometry = new THREE.BoxGeometry();\n"
        "const loader = {loadAsync: async () => ({scene: new THREE.Mesh(geometry)})};\n"
        + attach
        + """
const target = {stamp: 'ref', transform: [0,-1,0,10, 1,0,0,20, 0,0,1,30, 0,0,0,1]};
const entry = {target}; parts.set(current, entry);
const ghost = await ghostAttach(current, entry);
assert.deepEqual(new THREE.Vector3(2,3,4).applyMatrix4(ghost.matrix).toArray(), [7,22,34]);
delete target.transform; target.offset = [5,6,7];
const legacy = await ghostAttach(current, entry);
assert.deepEqual(legacy.position.toArray(), [5,6,7]);
parts.set(current, {target: {stamp: 'changed'}});
assert.equal(await ghostAttach(current, entry), undefined);
"""
    )


def test_tolerance_input_rejects_zero_and_values_below_the_backend_minimum():
    change = function(
        "document.getElementById('comparetolerance').onchange =",
        "\n\nconst compareFile =",
    )

    run_js(
        "import assert from 'node:assert/strict';\n"
        "const input = {}, sent = [], current = 'disk', WebSocket = {OPEN: 1};\n"
        "const document = {getElementById: () => input};\n"
        "const sock = {readyState: 1, send: text => sent.push(JSON.parse(text))};\n"
        "const comparePending = new Map([[current, {}]]);\n"
        "let invalidated = 0; function compareInvalidate() { invalidated++; }\n"
        + change
        + """
let error = '', reports = 0;
const target = {value: '', setCustomValidity: value => error = value,
  reportValidity: () => reports++};
for (const value of ['0', '0.0009', '-1', '', 'NaN']) {
  target.value = value; input.onchange({target});
  assert.match(error, /at least 0.001 mm/);
}
assert.equal(reports, 5); assert.equal(sent.length, 0); assert.equal(invalidated, 0);
target.value = '0.001'; input.onchange({target});
assert.equal(error, ''); assert.equal(invalidated, 1);
assert.deepEqual(sent, [{type: 'target_settings', name: 'disk', tolerance_mm: .001}]);
"""
    )

def test_pending_tolerance_accepts_decimal_serialization_without_accepting_other_settings():
    matching = function("function comparisonToleranceMatches(", "function comparePanel(")
    run_js(
        "import assert from 'node:assert/strict';\n"
        + matching
        + """
assert.equal(comparisonToleranceMatches(.1 + .2, .3), true);
assert.equal(comparisonToleranceMatches(.0010000000000000002, .001), true);
assert.equal(comparisonToleranceMatches(.3, .300001), false);
assert.equal(comparisonToleranceMatches(.3, undefined), false);
assert.equal(comparisonToleranceMatches(.3, NaN), false);
"""
    )


def test_alignment_controls_preview_before_persisting_rigid_transforms():
    align = function("function comparisonCenteredTransform(", "function comparePanel(")
    run_js(
        f"import * as THREE from {json.dumps(THREE)};\n"
        "import assert from 'node:assert/strict';\n"
        "const current = 'disk', sent = [], comparePending = new Map(), comparePreview = new Map();\n"
        "const WebSocket = {OPEN: 1}, sock = {readyState: 1, send: text => sent.push(JSON.parse(text))};\n"
        "const transform = [0,-1,0,10, 1,0,0,20, 0,0,1,30, 0,0,0,1];\n"
        "const entry = {token: 'built', target: {transform, alignment: 'auto'}};\n"
        "const parts = new Map([[current, entry]]), mesh = new THREE.Group();\n"
        "mesh.userData.token = 'built'; mesh.position.set(-2,7,4);\n"
        "const cad = new THREE.Mesh(new THREE.BoxGeometry(2,2,2)); cad.position.set(4,5,6);\n"
        "const reference = new THREE.Mesh(new THREE.BoxGeometry(4,4,4)); reference.name = 'target';\n"
        "reference.matrixAutoUpdate = false; reference.matrix.set(...transform); mesh.add(cad, reference);\n"
        "function compareInvalidate(reason) { comparePending.set(current, {reason, token: entry.token}); }\n"
        "function compareClear() {} function comparePanel() {}\n"
        + align
        + """
compareAlignment('lock');
assert.deepEqual(sent[0], {type: 'target_settings', name: current, transform});
assert.deepEqual(comparePending.get(current).transform, transform);
assert.match(comparePending.get(current).reason, /Saving alignment/);
compareAlignment('center');
const centered = [0,-1,0,4, 1,0,0,5, 0,0,1,6, 0,0,0,1];
assert.equal(sent.length, 1);
assert.deepEqual(comparePreview.get(current).transform, centered);
assert.match(comparePreview.get(current).reason, /preview/);
compareAlignment('apply');
assert.deepEqual(sent[1], {type: 'target_settings', name: current, transform: centered});
assert.deepEqual(comparePreview.get(current).transform, centered);
assert.match(comparePreview.get(current).reason, /Saving alignment/);
assert.deepEqual(comparePending.get(current).transform, centered);
assert.equal(comparePending.get(current).token, 'built');
assert.match(comparePending.get(current).reason, /Saving alignment/);
assert.deepEqual(transform, [0,-1,0,10, 1,0,0,20, 0,0,1,30, 0,0,0,1]);
mesh.userData.token = 'old'; compareAlignment('center');
assert.equal(sent.length, 2);
assert.equal(comparisonCenteredTransform(transform, [NaN,0,0], [0,0,0]), null);
"""
    )


def test_comparison_helpers_report_detected_regions_and_preserve_rigid_transform():
    helpers = function("function comparisonWorst(", "function comparePanel(")
    matrix = function("function comparisonMatrix(", "function compareAlignment(")
    run_js(
        f"import * as THREE from {json.dumps(THREE)};\n"
        "import assert from 'node:assert/strict';\n"
        + matrix
        + helpers
        + """
const transform = comparisonRigidTransform([1, 2, 3, 10, 20, 30]);
const rigid = comparisonRigid(transform);
assert.deepEqual(rigid.translation.map(x => Math.round(x)), [1, 2, 3]);
assert.deepEqual(rigid.rotation.map(x => Math.round(x)), [10, 20, 30]);
const metrics = {part: {sampled_max: .3}, target: {sampled_max: .05},
  worst_regions: [{direction: 'part_to_target', position_mm: [1,2,3], peak_deviation_mm: .3}]};
assert.equal(comparisonAbove(metrics, .1), true);
assert.deepEqual(comparisonWorst(metrics), metrics.worst_regions);
"""
    )


def test_compare_is_discoverable_without_a_reference_and_labels_coverage_as_estimated():
    assert 'id="ghostbtn" style="display:none"' not in VIEWER
    assert 'id="compareadd"' in VIEWER
    assert 'id="comparechange"' in VIEWER
    assert 'id="compareremove"' in VIEWER
    assert "estimated within" in VIEWER
    assert "Apply alignment" in VIEWER
    assert 'accept=".stl,.obj,.glb,.ply"' in VIEWER
    assert ".3mf" not in VIEWER.split('id="compareref"', 1)[1].split(">", 1)[0]


def test_narrow_embed_uses_a_full_width_bottom_sheet_and_failed_apply_keeps_preview():
    assert "@media (max-width: 820px)" in VIEWER
    assert "@media (max-width: 600px)" not in VIEWER
    assert "body.embed.comparing #comparepanel { position: fixed; inset: auto 0 0 0;" in VIEWER
    assert "const dirty = waiting && !waiting.error" in VIEWER
    apply = function("function compareAlignment(", "function comparisonWorst(")
    assert "comparePreview.delete(current);\n  compareInvalidate('Saving alignment" not in apply


def test_changed_reference_disposes_only_the_retired_cached_geometry():
    attach = function("async function ghostAttach(", "// ---- comparison panel ----")
    run_js(
        f"import * as THREE from {json.dumps(THREE)};\n"
        "import assert from 'node:assert/strict';\n"
        "const renderer = true, plane = new THREE.Plane(), mesh = new THREE.Group();\n"
        "const ghostGeo = new Map(), ghostMaterial = new THREE.MeshBasicMaterial();\n"
        "const current = 'part', ghostWanted = true, parts = new Map(), comparePreview = new Map();\n"
        "function comparePanel() {}\n"
        "const oldGeometry = new THREE.BoxGeometry(), newGeometry = new THREE.SphereGeometry();\n"
        "let oldDisposed = 0, newDisposed = 0;\n"
        "oldGeometry.dispose = () => oldDisposed++; newGeometry.dispose = () => newDisposed++;\n"
        "const loads = [oldGeometry, newGeometry];\n"
        "const loader = {loadAsync: async () => ({scene: new THREE.Mesh(loads.shift())})};\n"
        + attach
        + """
const first = {target: {stamp: 'first'}}; parts.set(current, first);
await ghostAttach(current, first);
const second = {target: {stamp: 'second'}}; parts.set(current, second);
const ghost = await ghostAttach(current, second);
assert.equal(ghost.geometry, newGeometry);
assert.equal(oldDisposed, 1);
assert.equal(newDisposed, 0);
"""
    )
