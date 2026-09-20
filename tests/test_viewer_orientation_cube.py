"""Exercise the orientation cube's geometry and camera choices in Three.js."""

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


def test_cube_has_every_face_edge_and_corner_target_once():
    specs = function("function cubeTargetSpecs(", "\n\nconst cubeTargets")
    run_js(
        "import assert from 'node:assert/strict';\n"
        + specs
        + """
const specs = cubeTargetSpecs();
assert.equal(specs.length, 26);
assert.equal(new Set(specs.map(({direction}) => direction.join(','))).size, 26);
assert.equal(specs.filter(({direction}) => direction.filter(Boolean).length === 1).length, 6);
assert.equal(specs.filter(({direction}) => direction.filter(Boolean).length === 2).length, 12);
assert.equal(specs.filter(({direction}) => direction.filter(Boolean).length === 3).length, 8);
"""
    )


def test_canonical_views_preserve_distance_and_flip_an_aligned_face():
    canonical = function("function cubeCanonicalView(", "\n\nlet cameraTransition")
    specs = function("function cubeTargetSpecs(", "\n\nconst cubeTargets")
    run_js(
        f"import * as THREE from {json.dumps(THREE)};\n"
        "import assert from 'node:assert/strict';\n"
        + specs
        + canonical
        + """
const target = new THREE.Vector3(4, 5, 6);
const camera = new THREE.PerspectiveCamera(45, 1, .1, 1000);
camera.up.set(0, 0, 1); camera.position.set(20, -15, 14); camera.lookAt(target);
const distance = camera.position.distanceTo(target);
for (const {direction} of cubeTargetSpecs()) {
  const view = cubeCanonicalView(camera, target, direction);
  assert.ok(view.position.toArray().every(Number.isFinite));
  assert.ok(view.up.toArray().every(Number.isFinite));
  assert.ok(view.orientation.toArray().every(Number.isFinite));
  assert.ok(Math.abs(view.position.distanceTo(target) - distance) < 1e-9);
}
const first = cubeCanonicalView(camera, target, [1, 0, 0]);
camera.position.copy(first.position); camera.up.copy(first.up); camera.quaternion.copy(first.orientation);
const flipped = cubeCanonicalView(camera, target, [1, 0, 0]);
assert.ok(first.up.dot(flipped.up) < -.999999);
camera.position.copy(target).addScalar(.01); camera.lookAt(target);
const clamped = cubeCanonicalView(camera, target, [0, -1, 0], 20);
assert.ok(Math.abs(clamped.position.distanceTo(target) - 20) < 1e-9);
"""
    )


def test_spacemouse_motion_refuses_new_cube_transitions():
    canonical = function("function cubeCanonicalView(", "\n\nlet cameraTransition")
    start = function("function startCubeView(", "\n\nfunction updateCameraTransition")
    run_js(
        f"import * as THREE from {json.dumps(THREE)};\n"
        "import assert from 'node:assert/strict';\n"
        + canonical
        + "\nlet cameraTransition = null;\n"
        + start
        + """
const camera = new THREE.PerspectiveCamera(45, 1, .1, 1000);
camera.position.set(10, -10, 10); camera.lookAt(0, 0, 0);
const controls = { target: new THREE.Vector3() };
const currentFrameBox = null;
const spaceMouse = { moving: true };
assert.equal(startCubeView([1, 0, 0]), false);
assert.equal(cameraTransition, null);
spaceMouse.moving = false;
assert.equal(startCubeView([1, 0, 0]), true);
assert.ok(cameraTransition);
"""
    )


def test_direction_names_and_shortcuts_describe_the_keyboard_surface():
    helpers = function("function cubeDirectionName(", "\n\nfunction cubeCanonicalView")
    run_js(
        "import assert from 'node:assert/strict';\n"
        + helpers
        + """
assert.equal(cubeDirectionName([0, -1, 0]), 'front view');
assert.equal(cubeDirectionName([1, -1, 0]), 'front right edge');
assert.equal(cubeDirectionName([-1, 1, 1]), 'top back left corner');
assert.deepEqual(cubeKeyDirection('5'), [0, 0, 1]);
assert.deepEqual(cubeKeyDirection('7'), [1, -1, 1]);
assert.equal(cubeKeyDirection('8'), null);
"""
    )


def test_cube_is_accessible_and_keeps_one_webgl_context():
    assert VIEWER.count("new THREE.WebGLRenderer") == 1
    assert 'id="viewcubehit" tabindex="0" role="application"' in VIEWER
    assert 'aria-keyshortcuts="1 2 3 4 5 6 7 F"' in VIEWER
    assert "Select a face, edge, or corner" in VIEWER
    assert 'id="viewcubeactions" role="group"' in VIEWER
    assert "cubeActions.appendChild(action)" in VIEWER
    assert 'id="viewcubefit"' in VIEWER
    assert "body.bare #viewcube, body.nogl #viewcube, body.noparts #viewcube" in VIEWER
    assert "controls.addEventListener('start', () => { cameraTransition = null; })" in VIEWER
    assert "renderTriad" not in VIEWER


def test_camera_roll_zoom_and_legacy_positions_are_persisted():
    assert "`nurb.cam.v5.${n}`" in VIEWER
    assert "u: camera.up.toArray()" in VIEWER
    assert "z: camera.zoom" in VIEWER
    assert "setTimeout(() => saveCamera(name, state), 250)" in VIEWER
    assert "localStorage.getItem(oldCamKey(name))" in VIEWER
    assert "localStorage.getItem(legacyCamKey(name))" in VIEWER
    assert "localStorage.getItem(oldestCamKey(name))" in VIEWER
    assert "camera.up.fromArray(s.u).normalize()" in VIEWER
    assert "Math.abs(camera.top - camera.bottom) / verticalSpan" in VIEWER
    assert "orthographicFitZoom(camera, box, 2.1)" in VIEWER


def test_orthographic_zoom_round_trips_and_perspective_state_migrates(tmp_path):
    persistence = function("const camKey", "\n\n// A view is")
    fit = function("function orthographicFitZoom(", "\n\nfunction frame")
    run_js(
        f"import * as THREE from {json.dumps(THREE)};\n"
        "import assert from 'node:assert/strict';\n"
        + """
const storage = new Map();
const localStorage = { getItem: key => storage.get(key) ?? null,
  setItem: (key, value) => storage.set(key, value) };
const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, .1, 1000);
const controls = { target: new THREE.Vector3(), addEventListener() {},
  update() { camera.lookAt(this.target); } };
let current = 'part', cameraTransition = null;
"""
        + persistence
        + fit
        + """
const box = new THREE.Box3(new THREE.Vector3(-5, -5, -5), new THREE.Vector3(5, 5, 5));
camera.position.set(0, -20, 10); camera.up.set(0, 0, 1); camera.zoom = .063;
controls.target.set(0, 0, 0); camera.lookAt(controls.target);
queueCameraSave();
await new Promise(resolve => setTimeout(resolve, 275));
const saved = JSON.parse(storage.get(camKey('part')));
assert.equal(saved.z, .063);

camera.position.set(30, 30, 30); controls.target.set(1, 1, 1); camera.zoom = 1;
assert.equal(restoreCamera('part', box), true);
assert.equal(camera.zoom, .063);

storage.set(camKey('part'), JSON.stringify({ ...saved, z: -1 }));
camera.zoom = 1;
assert.equal(restoreCamera('part', box), false);
assert.equal(camera.zoom, 1);

storage.delete(camKey('part'));
storage.set(oldCamKey('part'), JSON.stringify({ p: saved.p, t: saved.t, u: saved.u, f: 63 }));
camera.zoom = 1;
assert.equal(restoreCamera('part', box), true);
const distance = new THREE.Vector3().fromArray(saved.p).distanceTo(new THREE.Vector3().fromArray(saved.t));
const expectedMigrated = 2 / (2 * distance * Math.tan(63 * Math.PI / 360));
assert.ok(Math.abs(camera.zoom - expectedMigrated) < 1e-12);

storage.set(oldCamKey('part'), JSON.stringify({ p: saved.p, t: saved.t, u: saved.u }));
camera.zoom = 1;
assert.equal(restoreCamera('part', box), true);
assert.ok(Math.abs(camera.zoom - orthographicFitZoom(camera, box, 2.1)) < 1e-12);
"""
    )


def test_main_camera_is_orthographic_and_resize_preserves_vertical_scale():
    resize = function("function resizeOrthographicCamera(", "\nfunction resize()")
    fit = function("function orthographicFitZoom(", "\n\nfunction frame")
    run_js(
        f"import * as THREE from {json.dumps(THREE)};\n"
        "import assert from 'node:assert/strict';\n"
        + resize
        + fit
        + """
const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, .1, 1000);
resizeOrthographicCamera(camera, 600, 300);
assert.deepEqual([camera.left, camera.right, camera.top, camera.bottom], [-2, 2, 1, -1]);
camera.position.set(40, -40, 40); camera.lookAt(0, 0, 0);
const box = new THREE.Box3(new THREE.Vector3(-10, -5, -2), new THREE.Vector3(10, 5, 2));
const zoom = orthographicFitZoom(camera, box, 2);
assert.ok(Number.isFinite(zoom) && zoom > 0);
camera.zoom = zoom; camera.updateProjectionMatrix(); camera.updateMatrixWorld(true);
for (const x of [box.min.x, box.max.x]) for (const y of [box.min.y, box.max.y])
  for (const z of [box.min.z, box.max.z]) {
    const projected = new THREE.Vector3(x, y, z).project(camera);
    assert.ok(Math.abs(projected.x) <= .5 + 1e-9);
    assert.ok(Math.abs(projected.y) <= .5 + 1e-9);
  }
"""
    )
    assert "const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10000);" in VIEWER


def test_frame_and_restore_persist_the_final_zoom(tmp_path):
    persistence = function("const camKey", "\n\n// A view is")
    framing = function("function viewDir(", "\n\n// ---- loading")
    run_js(
        f"import * as THREE from {json.dumps(THREE)};\n"
        "import assert from 'node:assert/strict';\n"
        + """
const storage = new Map();
const localStorage = { getItem: key => storage.get(key) ?? null,
  setItem: (key, value) => storage.set(key, value) };
const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, .1, 1000);
const changeListeners = [];
const controls = { target: new THREE.Vector3(),
  addEventListener(type, listener) { if (type === 'change') changeListeners.push(listener); },
  update() { camera.lookAt(this.target); changeListeners.forEach(listener => listener()); } };
const VIEWS = { iso: [1, -1, 1] };
let current = 'part', cameraTransition = null;
"""
        + persistence
        + framing
        + """
const box = new THREE.Box3(new THREE.Vector3(-10, -5, -2), new THREE.Vector3(10, 5, 2));
frame(box);
const framedZoom = camera.zoom;
assert.notEqual(framedZoom, 1);
await new Promise(resolve => setTimeout(resolve, 275));
assert.ok(Math.abs(JSON.parse(storage.get(camKey('part'))).z - framedZoom) < 1e-12);

const restored = { p: [0, -40, 20], t: [0, 0, 0], u: [0, 0, 1], z: .04 };
storage.set(camKey('part'), JSON.stringify(restored));
camera.zoom = 1;
assert.equal(restoreCamera('part', box), true);
await new Promise(resolve => setTimeout(resolve, 275));
assert.ok(Math.abs(JSON.parse(storage.get(camKey('part'))).z - restored.z) < 1e-12);
"""
    )
