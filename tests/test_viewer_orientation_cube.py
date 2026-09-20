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


def test_camera_roll_is_saved_and_legacy_positions_still_restore():
    assert "`nurb.cam.v3.${n}`" in VIEWER
    assert "u: camera.up.toArray()" in VIEWER
    assert "setTimeout(() => saveCamera(name, state), 250)" in VIEWER
    assert "localStorage.getItem(oldCamKey(name))" in VIEWER
    assert "camera.up.fromArray(s.u).normalize()" in VIEWER
