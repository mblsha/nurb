"""Exercise the SpaceMouse camera bridge without requiring physical hardware."""

import asyncio
import hashlib
import json
import pathlib
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from nurb.server import Server


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/nurb/spacemouse.js"
VIEWER = (ROOT / "src/nurb/viewer.html").read_text(encoding="utf-8")
THREE = (ROOT / "src/nurb/vendor/three/build/three.module.min.js").as_uri()
VENDOR = ROOT / "src/nurb/vendor/3dconnexion/3dconnexion.min.js"


def run_module(tmp_path, source):
    node = shutil.which("node")
    if not node:
        pytest.skip("viewer JavaScript checks need Node.js")
    module = tmp_path / "spacemouse.mjs"
    module.write_text(
        SOURCE.read_text(encoding="utf-8").replace("from 'three'", f"from {json.dumps(THREE)}")
    )
    result = subprocess.run(
        [node, "--input-type=module", "-"],
        input=f"import * as THREE from {json.dumps(THREE)};\n"
        f"import * as nav from {json.dumps(module.as_uri())};\n{source}",
        encoding="utf-8",
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_perspective_math_z_up_frame_and_orbit_target(tmp_path):
    run_module(
        tmp_path,
        """
import assert from 'node:assert/strict';
const diagonal = nav.diagonalFovRadians(48, 16 / 9);
assert.ok(Math.abs(nav.verticalFovDegrees(diagonal, 16 / 9) - 48) < 1e-10);

const camera = new THREE.PerspectiveCamera(48, 16 / 9, .1, 1000);
camera.position.set(0, 0, 10); camera.updateMatrixWorld(true);
const controls = { enabled: true, target: new THREE.Vector3(),
  _sphericalDelta: new THREE.Spherical(3, 2, 1), _panOffset: new THREE.Vector3(1, 2, 3),
  _scale: 2, _performCursorZoom: true };
const bridge = new nav.SpaceMouseControls({ camera, controls, viewport: {},
  getModelBounds: box => box.set(new THREE.Vector3(-2, -3, -4), new THREE.Vector3(5, 6, 7)) });
assert.equal(bridge.getPerspective(), true);
assert.equal(bridge.getUnitsToMeters(), .001);
assert.deepEqual(bridge.getFloorPlane(), [0, 0, 1, 0]);
assert.deepEqual(bridge.getConstructionPlane(), [0, 0, 1, 0]);
assert.deepEqual(bridge.getCoordinateSystem(), new THREE.Matrix4().makeRotationX(-Math.PI / 2).toArray());
assert.deepEqual(bridge.getFrontView(), new THREE.Matrix4().makeRotationX(Math.PI / 2).toArray());
assert.deepEqual(bridge.getModelExtents(), [-2, -3, -4, 5, 6, 7]);
assert.ok(Math.abs(bridge.getFov() - diagonal) < 1e-10);

const panned = bridge.getViewMatrix(); panned[12] += 2;
bridge.setViewMatrix(panned);
assert.ok(camera.position.distanceTo(new THREE.Vector3(2, 0, 10)) < 1e-9);
assert.ok(controls.target.distanceTo(new THREE.Vector3(2, 0, 0)) < 1e-9);
const rejected = new THREE.Matrix4().makeScale(2, 2, 2).toArray();
bridge.setViewMatrix(rejected);
assert.ok(camera.position.distanceTo(new THREE.Vector3(2, 0, 10)) < 1e-9);

bridge.onStartMotion();
assert.equal(controls._sphericalDelta.radius, 0);
assert.deepEqual(controls._panOffset.toArray(), [0, 0, 0]);
assert.equal(controls._scale, 1);
assert.equal(controls._performCursorZoom, false);
""",
    )


def test_fake_sdk_connects_moves_focuses_and_ignores_stale_callbacks(tmp_path):
    run_module(
        tmp_path,
        """
import assert from 'node:assert/strict';
const browser = new EventTarget();
Object.assign(browser, { setTimeout, clearTimeout });
globalThis.window = browser;
globalThis.document = { body: { contains: () => true } };
const clients = [], instances = [];
class FakeSdk {
  constructor(client) { this.client = client; this.deleted = 0; this.frames = 0; this.focused = 0; this.blurred = 0; clients.push(client); instances.push(this); }
  connect() { queueMicrotask(() => this.client.onConnect()); return 1; }
  create3dmouse(focus) {
    this.focus = focus;
    focus.addEventListener('focus', () => this.focused++);
    focus.addEventListener('blur', () => this.blurred++);
    this.client.on3dmouseCreated();
  }
  delete3dmouse() { this.deleted++; }
  update3dcontroller(data) { if (data.frame?.time !== undefined) this.frames++; }
}
window._3Dconnexion = FakeSdk;
const camera = new THREE.PerspectiveCamera(45, 1, .1, 1000);
camera.position.set(0, 0, 10);
const controls = { enabled: true, target: new THREE.Vector3(), dispatchEvent() {} };
const states = [];
const bridge = new nav.SpaceMouseControls({ camera, controls, viewport: document.body,
  followWindowFocus: false, getModelBounds: box => box.makeEmpty(),
  onStateChange: state => states.push(state), connectionTimeoutMilliseconds: 500 });
bridge.setApplicationFocus(true);
await bridge.connect();
await new Promise(resolve => setTimeout(resolve, 0));
assert.equal(bridge.state, 'ready');
assert.equal(instances[0].focused, 1);
assert.deepEqual(states.slice(0, 3), ['loading', 'connecting', 'ready']);
bridge.onStartMotion(); bridge.updateFrame(42);
assert.equal(instances[0].frames, 1);
bridge.setApplicationFocus(false);
assert.equal(instances[0].blurred, 1);

const stale = clients[0];
await bridge.connect();
await new Promise(resolve => setTimeout(resolve, 0));
assert.equal(bridge.state, 'ready');
assert.equal(instances[1].blurred, 1);
stale.onDisconnect('late callback');
assert.equal(bridge.state, 'ready');
assert.ok(instances[0].deleted >= 1);
bridge.dispose();
assert.ok(instances[1].deleted >= 1);
""",
    )


def test_viewer_and_desktop_join_the_navigation_lifecycle():
    desktop = (ROOT / "desktop/src/App.tsx").read_text(encoding="utf-8")
    assert "import { SpaceMouseControls } from '/spacemouse.js';" in VIEWER
    assert 'id="spacemouse"' in VIEWER
    assert "if (controls && !bare)" in VIEWER
    assert "getModelBounds: target => currentFrameBox" in VIEWER
    assert "onMotionStart: () => { cameraTransition = null; }" in VIEWER
    assert "else { updateCameraTransition(now); controls.update(); }" in VIEWER
    assert "controls.dispatchEvent({ type: 'change' }); queueCameraSave();" in VIEWER
    assert "e.data.type === 'nurb:focus'" in VIEWER
    assert "allow ${location.origin} in your local SpaceMouse bridge" in VIEWER
    assert '{ type: "nurb:focus", focused }' in desktop
    assert "postFocus(document.hasFocus())" in desktop


def test_spacemouse_motion_exclusively_owns_camera_navigation():
    assert "if (!controls || spaceMouse?.moving) return false;" in VIEWER
    assert "if (spaceMouse?.moving || !currentFrameBox" in VIEWER
    assert "if (spaceMouse?.moving) cameraTransition = null;" in VIEWER
    assert "cubeHit.setAttribute('aria-disabled', String(navigating))" in VIEWER
    assert "document.getElementById('viewcubefit').disabled = navigating" in VIEWER


def test_offline_sdk_is_exactly_the_documented_vendor_file():
    assert hashlib.sha256(VENDOR.read_bytes()).hexdigest() == (
        "a11b78252aebc0ebd1cd5e4b14f9f2a87a1bf7ee552dd9e0507342548a7551d0"
    )
    assert VENDOR.read_text(encoding="utf-8").startswith("/**\n* @license 3DconnexionJS")
    assert '"src/nurb/spacemouse.js"' in (ROOT / "pyproject.toml").read_text(encoding="utf-8")


def test_server_serves_adapter_and_sdk_but_rejects_vendor_traversal(tmp_path):
    server = Server(tmp_path)
    module = asyncio.run(server.http(None, SimpleNamespace(path="/spacemouse.js")))
    sdk = asyncio.run(
        server.http(None, SimpleNamespace(path="/vendor/3dconnexion/3dconnexion.min.js"))
    )
    traversal = asyncio.run(
        server.http(None, SimpleNamespace(path="/vendor/../spacemouse.js"))
    )
    assert module.status_code == 200
    assert module.headers["Content-Type"] == "text/javascript; charset=utf-8"
    assert b"class SpaceMouseControls" in module.body
    assert sdk.status_code == 200
    assert sdk.body == VENDOR.read_bytes()
    assert traversal.status_code == 404
