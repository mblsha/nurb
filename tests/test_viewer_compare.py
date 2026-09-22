"""Execute the comparison presentation against legacy and spatial payloads."""

import json
import pathlib
import shutil
import subprocess

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
VIEWER = (ROOT / "src/nurb/viewer.html").read_text(encoding="utf-8")
THREE = (ROOT / "src/nurb/vendor/three/build/three.module.min.js").as_uri()
INSPECTION_VIEWER = (ROOT / "src/nurb/inspection-viewer.js").as_uri()


def run_js(source):
    source=f"import * as InspectionViewer from {json.dumps(INSPECTION_VIEWER)};\n"+source
    node = shutil.which("node")
    if not node:
        pytest.skip("viewer JavaScript checks need Node.js")
    result = subprocess.run(
        [node, "--input-type=module", "-"], input=source, encoding="utf-8", capture_output=True
    )
    assert result.returncode == 0, result.stderr


def functions(names, bindings=()):
    accessors="\n".join(f"get {name}() {{ return {name}; }}, set {name}(value) {{ {name}=value; }}," for name in bindings)
    return "const {"+",".join(names)+"} = {...InspectionViewer,...InspectionViewer.createInspectionController({"+accessors+"})};\n"


def test_unsigned_metrics_keep_raw_deviation_and_do_not_invent_legacy_coverage():
    rows = functions(['comparisonRows'])
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
    clear = functions(['compareClear'], bindings=['compareMarks', 'comparePainted'])
    samples = functions(['compareSamples', 'comparisonToleranceMatches', 'comparisonCenteredTransform', 'comparisonMatrix', 'comparisonTransform', 'comparisonRigid', 'comparisonRigidTransform', 'comparisonPreview', 'compareAlignment', 'comparisonWorst', 'comparisonAbove', 'comparisonFocus', 'comparisonTransformFields'], bindings=['WebSocket', 'camera', 'compareAuto', 'compareClear', 'compareInvalidate', 'compareMap', 'compareMarks', 'comparePainted', 'comparePanel', 'comparePending', 'comparePreview', 'compareScale', 'compareThrough', 'comparisonPreview', 'controls', 'current', 'datumPreview', 'document', 'mesh', 'modelNodes', 'parts', 'plane', 'renderer', 'sock'])
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
    helpers = functions(['referenceUsesSourceMaterial', 'referenceMaterialList', 'referenceMaterialShape', 'referenceMeshes', 'referenceResources', 'referenceDispose', 'referenceSanitizeScene', 'referenceCloneScene', 'referenceInstance', 'referenceSetMode', 'modelGeometryDispose', 'displayedMeshDispose'], bindings=['ghostMaterial', 'plane'])
    attach = functions(['ghostAttach'], bindings=['comparePanel', 'comparePreview', 'current', 'ghostGeo', 'ghostWanted', 'inspectionMode', 'loader', 'mesh', 'parts', 'referenceInstance', 'renderer'])
    run_js(
        f"import * as THREE from {json.dumps(THREE)};\n"
        "import assert from 'node:assert/strict';\n"
        "const renderer = true, plane = new THREE.Plane(), mesh = new THREE.Group();\n"
        "const ghostGeo = new Map(), ghostMaterial = new THREE.MeshBasicMaterial();\n"
        "const current = 'part', ghostWanted = true, parts = new Map(), comparePreview = new Map();\n"
        "let inspectionMode = 'overlay';\n"
        "function comparePanel() {}\n"
        "const geometry = new THREE.BoxGeometry();\n"
        "let loads = 0; const loader = {loadAsync: async () => { loads++; return {scene: new THREE.Mesh(geometry)}; }};\n"
        + helpers
        + attach
        + """
const target = {stamp: 'ref', transform: [0,-1,0,10, 1,0,0,20, 0,0,1,30, 0,0,0,1]};
const entry = {target}; parts.set(current, entry);
const ghost = await ghostAttach(current, entry);
assert.deepEqual(new THREE.Vector3(2,3,4).applyMatrix4(ghost.matrix).toArray(), [7,22,34]);
delete target.transform; target.offset = [5,6,7];
const legacy = await ghostAttach(current, entry);
assert.deepEqual(legacy.position.toArray(), [5,6,7]);
assert.equal(loads, 1);
parts.set(current, {target: {stamp: 'changed'}});
assert.equal(await ghostAttach(current, entry), undefined);
"""
    )


def test_reference_and_side_by_side_use_source_texture_while_analysis_modes_use_amber():
    helpers = functions(['referenceUsesSourceMaterial', 'referenceMaterialList', 'referenceMaterialShape', 'referenceMeshes', 'referenceResources', 'referenceDispose', 'referenceSanitizeScene', 'referenceCloneScene', 'referenceInstance', 'referenceSetMode', 'modelGeometryDispose', 'displayedMeshDispose'], bindings=['ghostMaterial', 'plane'])
    run_js(
        f"import * as THREE from {json.dumps(THREE)};\n"
        "import assert from 'node:assert/strict';\n"
        "const plane = new THREE.Plane();\n"
        "const ghostMaterial = new THREE.MeshBasicMaterial({color: 0xd9a066, transparent: true, opacity: .35, depthWrite: false});\n"
        + helpers
        + """
const texture = new THREE.Texture();
const sourceMaterial = new THREE.MeshBasicMaterial({map: texture, transparent: true, opacity: .72, depthWrite: true});
const geometry = new THREE.BoxGeometry(1,2,3);
const source = new THREE.Group(); source.add(new THREE.Mesh(geometry, sourceMaterial));
const reference = referenceInstance(source, 10);
const shown = referenceMeshes(reference)[0];
referenceSetMode(reference, 'reference');
assert.equal(shown.material.map, texture);
assert.equal(shown.material.opacity, .72);
assert.equal(shown.material.depthWrite, true);
assert.equal(shown.material.side, THREE.DoubleSide);
assert.deepEqual(shown.material.clippingPlanes, [plane]);
referenceSetMode(reference, 'overlay');
assert.equal(shown.material.map, null);
assert.equal(shown.material.opacity, .35);
assert.equal(shown.material.depthWrite, false);
assert.equal(shown.material.color.getHex(), 0xd9a066);
referenceSetMode(reference, 'deviation');
assert.equal(shown.material.map, null);
referenceSetMode(reference, 'section');
assert.equal(shown.material.map, null);
referenceSetMode(reference, 'side-by-side');
assert.equal(shown.material.map, texture);
assert.equal(reference.children[0].scale.x, 10);
"""
    )


def test_reference_scene_strips_non_mesh_content_and_remaps_skinned_bones():
    helpers = functions(['referenceUsesSourceMaterial', 'referenceMaterialList', 'referenceMaterialShape', 'referenceMeshes', 'referenceResources', 'referenceDispose', 'referenceSanitizeScene', 'referenceCloneScene', 'referenceInstance', 'referenceSetMode', 'modelGeometryDispose', 'displayedMeshDispose'], bindings=['ghostMaterial', 'plane'])
    run_js(
        f"import * as THREE from {json.dumps(THREE)};\n"
        "import assert from 'node:assert/strict';\n"
        "const plane = new THREE.Plane();\n"
        "const ghostMaterial = new THREE.MeshBasicMaterial();\n"
        + helpers
        + """
const source = new THREE.Group();
const carrier = new THREE.Group(); carrier.position.set(3,4,5);
carrier.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial())); source.add(carrier);
const bone = new THREE.Bone(); bone.name = 'kept-bone'; source.add(bone);
const skinned = new THREE.SkinnedMesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial());
skinned.bind(new THREE.Skeleton([bone])); source.add(skinned);
const camera = new THREE.PerspectiveCamera(); camera.position.set(7,8,9);
camera.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial())); source.add(camera);
source.add(new THREE.DirectionalLight(),
  new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial()),
  new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial()));
referenceSanitizeScene(source);
const nodes = []; source.traverse(node => nodes.push(node));
assert.equal(nodes.some(node => node.isCamera || node.isLight || node.isLine || node.isPoints), false);
assert.equal(referenceMeshes(source).length, 3);
assert.equal(source.getObjectByName('kept-bone'), bone);
assert.ok(referenceMeshes(source).some(node => node.parent?.isGroup && node.parent.position.equals(new THREE.Vector3(7,8,9))));
const instance = referenceInstance(source, 1);
const clonedSkin = referenceMeshes(instance).find(node => node.isSkinnedMesh);
assert.ok(clonedSkin);
assert.notEqual(clonedSkin.skeleton.bones[0], bone);
assert.equal(clonedSkin.skeleton.bones[0].name, 'kept-bone');
"""
    )


def test_reference_section_assembles_open_primitive_chains_into_one_closed_contour():
    contours = functions(['sectionSegments', 'sectionAssemble', 'sectionContours', 'sectionContoursMany'])
    run_js(
        f"import * as THREE from {json.dumps(THREE)};\n"
        "import assert from 'node:assert/strict';\n"
        + contours
        + """
function sides(edges) {
  const positions = [];
  for (const [[ax,ay],[bx,by]] of edges) positions.push(
    ax,ay,-1, ax,ay,1, bx,by,1,
    ax,ay,-1, bx,by,1, bx,by,-1);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions,3));
  return new THREE.Mesh(geometry);
}
const first = sides([[[-1,-1],[1,-1]], [[1,-1],[1,1]]]);
const second = sides([[[1,1],[-1,1]], [[-1,1],[-1,-1]]]);
const identity = new THREE.Matrix4();
assert.equal(sectionContours(first.geometry, identity, 2, 0).valid, false);
assert.equal(sectionContours(second.geometry, identity, 2, 0).valid, false);
const combined = sectionContoursMany([first,second], () => identity, 2, 0);
assert.equal(combined.valid, true);
assert.equal(combined.loops.length, 1);
assert.equal(combined.segments.length, 8);
"""
    )


def test_tolerance_input_rejects_zero_and_values_below_the_backend_minimum():
    change = functions(['comparisonToleranceChanged'], bindings=['WebSocket', 'compareInvalidate', 'comparePending', 'current', 'inspectionToleranceOverride', 'sock'])

    run_js(
        "import assert from 'node:assert/strict';\n"
        "const input = {}, sent = [], current = 'disk', WebSocket = {OPEN: 1};\n"
        "const document = {getElementById: () => input};\n"
        "const sock = {readyState: 1, send: text => sent.push(JSON.parse(text))};\n"
        "const comparePending = new Map([[current, {}]]);\n"
        "let inspectionToleranceOverride = .3;\n"
        "let invalidated = 0; function compareInvalidate() { invalidated++; }\n"
        + change
        + """
let error = '', reports = 0;
const target = {value: '', setCustomValidity: value => error = value,
  reportValidity: () => reports++};
for (const value of ['0', '0.0009', '-1', '', 'NaN']) {
  target.value = value; comparisonToleranceChanged({target});
  assert.match(error, /at least 0.001 mm/);
}
assert.equal(reports, 5); assert.equal(sent.length, 0); assert.equal(invalidated, 0);
target.value = '0.001'; comparisonToleranceChanged({target});
assert.equal(error, ''); assert.equal(invalidated, 1);
assert.deepEqual(sent, [{type: 'target_settings', name: 'disk', tolerance_mm: .001}]);
"""
    )

def test_pending_tolerance_accepts_decimal_serialization_without_accepting_other_settings():
    matching = functions(['comparisonToleranceMatches', 'comparisonCenteredTransform', 'comparisonMatrix', 'comparisonTransform', 'comparisonRigid', 'comparisonRigidTransform', 'comparisonPreview', 'compareAlignment', 'comparisonWorst', 'comparisonAbove', 'comparisonFocus', 'comparisonTransformFields'], bindings=['WebSocket', 'camera', 'compareClear', 'compareInvalidate', 'compareMap', 'comparePanel', 'comparePending', 'comparePreview', 'comparisonPreview', 'controls', 'current', 'datumPreview', 'document', 'mesh', 'modelNodes', 'parts', 'sock'])
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
    align = functions(['comparisonCenteredTransform', 'comparisonMatrix', 'comparisonTransform', 'comparisonRigid', 'comparisonRigidTransform', 'comparisonPreview', 'compareAlignment', 'comparisonWorst', 'comparisonAbove', 'comparisonFocus', 'comparisonTransformFields'], bindings=['WebSocket', 'camera', 'compareClear', 'compareInvalidate', 'compareMap', 'comparePanel', 'comparePending', 'comparePreview', 'comparisonPreview', 'controls', 'current', 'datumPreview', 'document', 'mesh', 'modelNodes', 'parts', 'sock'])
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
        + functions(['componentInfo', 'modelNodes'], bindings=['mesh'])
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
    helpers = functions(['comparisonWorst', 'comparisonAbove', 'comparisonFocus', 'comparisonTransformFields'], bindings=['camera', 'compareMap', 'comparePanel', 'controls', 'current', 'document', 'mesh', 'parts'])
    matrix = functions(['comparisonMatrix', 'comparisonTransform', 'comparisonRigid', 'comparisonRigidTransform', 'comparisonPreview'], bindings=['compareClear', 'comparePanel', 'comparePending', 'comparePreview', 'current', 'mesh', 'parts'])
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
    assert 'accept=".stl,.obj,.glb,.ply,.ply.gz,.zip,.png,.jpg,.jpeg,.step,.stp,.brep" multiple' in VIEWER
    assert ".3mf" not in VIEWER.split('id="compareref"', 1)[1].split(">", 1)[0]


def test_narrow_embed_uses_a_full_width_bottom_sheet():
    assert "@media (max-width: 820px)" in VIEWER
    assert "@media (max-width: 600px)" not in VIEWER
    assert "body.embed.comparing #comparepanel { position: fixed; inset: auto 0 0 0;" in VIEWER
    assert "const dirty = waiting && !waiting.error" in VIEWER


def test_changed_reference_disposes_only_the_retired_cached_geometry():
    helpers = functions(['referenceUsesSourceMaterial', 'referenceMaterialList', 'referenceMaterialShape', 'referenceMeshes', 'referenceResources', 'referenceDispose', 'referenceSanitizeScene', 'referenceCloneScene', 'referenceInstance', 'referenceSetMode', 'modelGeometryDispose', 'displayedMeshDispose'], bindings=['ghostMaterial', 'plane'])
    attach = functions(['ghostAttach'], bindings=['comparePanel', 'comparePreview', 'current', 'ghostGeo', 'ghostWanted', 'inspectionMode', 'loader', 'mesh', 'parts', 'referenceInstance', 'renderer'])
    run_js(
        f"import * as THREE from {json.dumps(THREE)};\n"
        "import assert from 'node:assert/strict';\n"
        "const renderer = true, plane = new THREE.Plane(), mesh = new THREE.Group();\n"
        "const ghostGeo = new Map(), ghostMaterial = new THREE.MeshBasicMaterial();\n"
        "const current = 'part', ghostWanted = true, parts = new Map(), comparePreview = new Map();\n"
        "let inspectionMode = 'overlay';\n"
        "function comparePanel() {}\n"
        "const oldGeometry = new THREE.BoxGeometry(), newGeometry = new THREE.SphereGeometry();\n"
        "let oldImageClosed = 0, newImageClosed = 0;\n"
        "const oldImage = {close: () => oldImageClosed++}, newImage = {close: () => newImageClosed++};\n"
        "const oldTexture = new THREE.Texture(oldImage), oldNormal = new THREE.Texture(oldImage), newTexture = new THREE.Texture(newImage);\n"
        "const oldMaterial = new THREE.MeshBasicMaterial({map: oldTexture}); oldMaterial.normalMap = oldNormal;\n"
        "const newMaterial = new THREE.MeshBasicMaterial({map: newTexture});\n"
        "let oldDisposed = 0, newDisposed = 0, oldTextureDisposed = 0, oldNormalDisposed = 0, newTextureDisposed = 0;\n"
        "oldGeometry.dispose = () => oldDisposed++; newGeometry.dispose = () => newDisposed++;\n"
        "oldTexture.dispose = () => oldTextureDisposed++; oldNormal.dispose = () => oldNormalDisposed++; newTexture.dispose = () => newTextureDisposed++;\n"
        "const loads = [new THREE.Mesh(oldGeometry, oldMaterial), new THREE.Mesh(newGeometry, newMaterial)];\n"
        "const loader = {loadAsync: async () => ({scene: loads.shift()})};\n"
        + helpers
        + attach
        + """
const first = {target: {stamp: 'first'}}; parts.set(current, first);
await ghostAttach(current, first);
const second = {target: {stamp: 'second'}}; parts.set(current, second);
const ghost = await ghostAttach(current, second);
assert.equal(referenceMeshes(ghost)[0].geometry, newGeometry);
assert.equal(oldDisposed, 1);
assert.equal(newDisposed, 0);
assert.equal(oldTextureDisposed, 1);
assert.equal(oldNormalDisposed, 1);
assert.equal(newTextureDisposed, 0);
assert.equal(oldImageClosed, 1);
assert.equal(newImageClosed, 0);
"""
    )


def test_lost_reference_load_race_disposes_texture_and_closes_decoded_image():
    helpers = functions(['referenceUsesSourceMaterial', 'referenceMaterialList', 'referenceMaterialShape', 'referenceMeshes', 'referenceResources', 'referenceDispose', 'referenceSanitizeScene', 'referenceCloneScene', 'referenceInstance', 'referenceSetMode', 'modelGeometryDispose', 'displayedMeshDispose'], bindings=['ghostMaterial', 'plane'])
    attach = functions(['ghostAttach'], bindings=['comparePanel', 'comparePreview', 'current', 'ghostGeo', 'ghostWanted', 'inspectionMode', 'loader', 'mesh', 'parts', 'referenceInstance', 'renderer'])
    run_js(
        f"import * as THREE from {json.dumps(THREE)};\n"
        "import assert from 'node:assert/strict';\n"
        "const renderer = true, plane = new THREE.Plane(), mesh = new THREE.Group();\n"
        "const ghostGeo = new Map(), ghostMaterial = new THREE.MeshBasicMaterial();\n"
        "const current = 'part', ghostWanted = true, parts = new Map(), comparePreview = new Map();\n"
        "let inspectionMode = 'overlay', closed = 0, textureDisposed = 0;\n"
        "function comparePanel() {}\n"
        "const image = {close: () => closed++}, texture = new THREE.Texture(image);\n"
        "texture.dispose = () => textureDisposed++;\n"
        "const source = new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial({map: texture}));\n"
        "const loader = {loadAsync: async () => { parts.set(current, {target: {stamp: 'new'}}); return {scene: source}; }};\n"
        + helpers
        + attach
        + """
const entry = {target: {stamp: 'old'}}; parts.set(current, entry);
assert.equal(await ghostAttach(current, entry), undefined);
assert.equal(textureDisposed, 1);
assert.equal(closed, 1);
assert.equal(ghostGeo.size, 0);
"""
    )


def test_reference_mode_hides_cad_finding_highlights():
    sync = functions(['syncPins'], bindings=['inspectionMode', 'pins', 'pinsPlaced', 'pinsWanted'])
    run_js("import assert from 'node:assert/strict'; let inspectionMode='reference'; const pins={}; let pinsWanted=true,pinsPlaced=true;\n" + sync + "\nsyncPins();assert.equal(pins.visible,false);inspectionMode='overlay';syncPins();assert.equal(pins.visible,true);")


def test_reference_load_discards_geometry_when_the_displayed_mesh_group_changes():
    run_js(f"import * as THREE from {json.dumps(THREE)};\n"+"import assert from 'node:assert/strict';\n"+"""
const original=new THREE.Group(),parts=new Map(),ghostGeo=new Map();let mesh=original,disposed=0;
const geometry=new THREE.BoxGeometry();geometry.dispose=()=>disposed++;
const entry={target:{stamp:'unchanged'}};parts.set('part',entry);
const runtime={renderer:true,get mesh(){return mesh;},ghostGeo,parts,
 loader:{loadAsync:async()=>{mesh=new THREE.Group();return {scene:new THREE.Mesh(geometry)};}}};
const controller=InspectionViewer.createInspectionController(runtime);
assert.equal(await controller.ghostAttach('part',entry),undefined);
assert.equal(original.children.length,0);assert.equal(mesh.children.length,0);
assert.equal(disposed,1);assert.equal(ghostGeo.size,0);
""")


def test_import_component_preview_hides_only_the_unchecked_source_groups():
    visibility = functions(['importVisibility'], bindings=['document', 'importPreview'])
    run_js(f"import * as THREE from {json.dumps(THREE)};\n" + "import assert from 'node:assert/strict';\n" + """
const root=new THREE.Group(), first=new THREE.Mesh(), second=new THREE.Mesh();
first.name='component-1';second.name='component-2';root.add(first,second);
let importPreview={root,record:{components:[{},{}]}};
const apply={},status={};
const document={querySelectorAll:()=>[{value:'component-1'}],getElementById:id=>id==='importapply'?apply:status};
""" + visibility + """
importVisibility();assert.equal(first.visible,true);assert.equal(second.visible,false);
assert.equal(apply.disabled,false);assert.match(status.textContent,/saved reference until you save/);
""")


@pytest.mark.parametrize("alignment_first", [False, True])
@pytest.mark.parametrize("cancel_alignment", [False, True])
def test_component_preview_cancel_preserves_visible_alignment_before_apply(alignment_first, cancel_alignment):
    alignment = functions(['comparisonCenteredTransform', 'comparisonMatrix', 'comparisonTransform', 'comparisonRigid', 'comparisonRigidTransform', 'comparisonPreview', 'compareAlignment'], bindings=['WebSocket', 'compareClear', 'compareInvalidate', 'comparePanel', 'comparePending', 'comparePreview', 'comparisonPreview', 'current', 'datumPreview', 'document', 'mesh', 'modelNodes', 'parts', 'sock'])
    components = functions(['importCancel', 'importVisibility', 'importLanded'], bindings=['atob', 'current', 'document', 'importCancel', 'importPreview', 'importVisibility', 'inspectionSetMode', 'loader', 'mesh', 'parts', 'referenceInstance'])
    run_js(f"import * as THREE from {json.dumps(THREE)};\n" + "import assert from 'node:assert/strict';\n" + """
const current = 'fixture', sent = [], comparePending = new Map(), comparePreview = new Map();
const WebSocket = {OPEN:1}, sock = {readyState:1,send:text=>sent.push(JSON.parse(text))};
const saved = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];
const moved = [0,-1,0,5, 1,0,0,7, 0,0,1,9, 0,0,0,1];
const entry = {token:'built',target:{stamp:'source',transform:saved,import:{excluded_components:[]}}};
const parts = new Map([[current,entry]]), mesh = new THREE.Group(); mesh.userData.token='built';
const original = new THREE.Group(); original.name='target'; original.matrixAutoUpdate=false;
original.matrix.set(...saved); mesh.add(original);
let importPreview = null, datumPreview = null;
const elements = new Map();
const document = {
  getElementById(id) { if (!elements.has(id)) elements.set(id,{replaceChildren(){},append(){},textContent:''}); return elements.get(id); },
  querySelectorAll:()=>[{value:'component-1'}],
  createElement:()=>({append(){}}), createTextNode:text=>text,
};
const loader = {parseAsync:async()=>({scene:new THREE.Group()})};
function referenceSanitizeScene(scene) {return scene;}
function referenceInstance() {const root=new THREE.Group();root.name='target';return root;}
function referenceDispose() {}
function inspectionSetMode() {}
function compareClear() {} function comparePanel() {}
function compareInvalidate(reason) {comparePending.set(current,{reason,token:entry.token});}
""" + alignment + components + f"""
const alignmentFirst = {json.dumps(alignment_first)}, cancelAlignment = {json.dumps(cancel_alignment)};
if (alignmentFirst) comparisonPreview(moved,'Alignment preview');
await importLanded({{name:current,stamp:'source',glb:'AA==',import:{{components:[{{id:'component-1',triangles:1}}]}}}});
assert.ok(importPreview); assert.notEqual(mesh.getObjectByName('target'),original);
if (!alignmentFirst) comparisonPreview(moved,'Symmetry plane alignment preview');
if (cancelAlignment) compareAlignment('cancel');
const visibleBeforeCancel = comparisonTransform(mesh.getObjectByName('target').matrix);
importCancel();
assert.equal(mesh.getObjectByName('target'),original);
assert.deepEqual(comparisonTransform(original.matrix),visibleBeforeCancel);
assert.deepEqual(visibleBeforeCancel,cancelAlignment ? saved : moved);
assert.equal(comparePreview.has(current),!cancelAlignment);
compareAlignment('apply');
assert.equal(sent.length,1);
assert.deepEqual(sent[0],{{type:'target_settings',name:current,transform:comparisonTransform(original.matrix)}});
assert.deepEqual(comparePending.get(current).transform,comparisonTransform(original.matrix));
assert.deepEqual(entry.target.transform,saved);
""")
