import * as THREE from 'three';

export const SPACEMOUSE_SDK_URL = '/vendor/3dconnexion/3dconnexion.min.js';

let sdkLoad = null;

export function finiteArray(data, length) {
  return Array.isArray(data) && data.length === length
    && data.every(value => typeof value === 'number' && Number.isFinite(value));
}

export function diagonalFovRadians(verticalFovDegrees, aspect) {
  const vertical = THREE.MathUtils.degToRad(verticalFovDegrees);
  return 2 * Math.atan(Math.tan(vertical / 2) * Math.sqrt(1 + aspect * aspect));
}

export function verticalFovDegrees(diagonalFov, aspect) {
  return THREE.MathUtils.radToDeg(
    2 * Math.atan(Math.tan(diagonalFov / 2) / Math.sqrt(1 + aspect * aspect)),
  );
}

export function synchronizedOrbitTarget(previousPosition, previousTarget, position, orientation) {
  const distance = Math.max(previousPosition.distanceTo(previousTarget), 1e-6);
  const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(orientation).normalize();
  const offset = previousTarget.clone().sub(position);
  const depth = offset.dot(forward);
  const lateral = offset.addScaledVector(forward, -depth).length();
  if (depth > 0 && lateral <= Math.max(depth, distance) * 1e-5) return previousTarget.clone();
  return position.clone().addScaledVector(forward, distance);
}

/** Clear OrbitControls r169's queued mouse motion before another controller takes over. */
export function stopOrbitInertia(controls) {
  controls?._sphericalDelta?.set(0, 0, 0);
  controls?._panOffset?.set(0, 0, 0);
  if (controls) {
    controls._scale = 1;
    controls._performCursorZoom = false;
  }
}

export class PageFocusTarget extends EventTarget {
  constructor(scope, windowTarget = window) {
    super();
    this.id = 'nurb-viewer';
    this.scope = scope;
    this.windowTarget = windowTarget;
    windowTarget?.addEventListener('focus', this.onFocus);
    windowTarget?.addEventListener('blur', this.onBlur);
  }
  focus = () => this.dispatchEvent(new Event('focus'));
  blur = () => this.dispatchEvent(new Event('blur'));
  onFocus = () => this.focus();
  onBlur = () => this.blur();
  contains(node) { return node !== null && this.scope.contains(node); }
  dispose() {
    this.windowTarget?.removeEventListener('focus', this.onFocus);
    this.windowTarget?.removeEventListener('blur', this.onBlur);
  }
}

export function loadSpaceMouseSdk(timeoutMilliseconds = 8000) {
  if (window._3Dconnexion) return Promise.resolve(window._3Dconnexion);
  if (sdkLoad) return sdkLoad;
  sdkLoad = new Promise((resolve, reject) => {
    const existing = document.querySelector(`script[src="${SPACEMOUSE_SDK_URL}"]`);
    const script = existing || document.createElement('script');
    const timeout = window.setTimeout(() => fail('3Dconnexion adapter load timed out.'), timeoutMilliseconds);
    const fail = message => {
      window.clearTimeout(timeout);
      script.remove();
      reject(new Error(message));
    };
    const loaded = () => {
      window.clearTimeout(timeout);
      if (window._3Dconnexion) resolve(window._3Dconnexion);
      else fail('3Dconnexion adapter did not expose its client.');
    };
    script.addEventListener('load', loaded, { once: true });
    script.addEventListener('error', () => fail('3Dconnexion adapter is unavailable.'), { once: true });
    if (!existing) {
      script.src = SPACEMOUSE_SDK_URL;
      script.async = true;
      script.referrerPolicy = 'no-referrer';
      document.head.appendChild(script);
    }
  }).catch(error => {
    sdkLoad = null;
    throw error;
  });
  return sdkLoad;
}

export class SpaceMouseControls {
  constructor(options) {
    this.options = options;
    this.state = 'loading';
    this.sdk = null;
    this.focus = null;
    this.timer = null;
    this.generation = 0;
    this.frameTime = 0;
    this.applicationFocused = null;
    this.pivotVisible = false;
    this.lookOrigin = new THREE.Vector3();
    this.lookDirection = new THREE.Vector3(0, 0, -1);
  }

  get moving() { return this.state === 'moving'; }

  async connect() {
    this.disconnect();
    const generation = this.generation;
    this.present('loading');
    try {
      const Constructor = await loadSpaceMouseSdk(this.options.connectionTimeoutMilliseconds);
      if (generation !== this.generation) return;
      this.focus = new PageFocusTarget(
        this.options.focusScope || this.options.viewport,
        this.options.followWindowFocus === false ? null : window,
      );
      let sdk;
      const callbacks = new Proxy(this, {
        get: (target, key) => {
          const value = Reflect.get(target, key);
          if (typeof value !== 'function') return value;
          return (...args) => {
            if (generation === this.generation) return value.apply(target, args);
            if (key === 'onConnect' || key === 'on3dmouseCreated') this.closeSdk(sdk);
          };
        },
      });
      sdk = new Constructor(callbacks);
      this.sdk = sdk;
      this.present('connecting');
      this.timer = window.setTimeout(() => {
        if (generation !== this.generation) return;
        this.disconnect();
        this.present('unavailable');
      }, this.options.connectionTimeoutMilliseconds ?? 8000);
      if (sdk.connect() === 0) throw new Error('The local 3DxWare bridge rejected the connection.');
    } catch (error) {
      if (generation !== this.generation) return;
      this.disconnect();
      this.present('unavailable', error instanceof Error ? error.message : String(error));
    }
  }

  disconnect() {
    this.generation += 1;
    window.clearTimeout(this.timer);
    this.timer = null;
    const sdk = this.sdk;
    this.sdk = null;
    this.closeSdk(sdk);
    this.focus?.dispose();
    this.focus = null;
  }

  dispose() { this.disconnect(); }

  closeSdk(sdk) {
    try { sdk?.delete3dmouse(); } catch { /* The bridge may already be gone. */ }
    try { sdk?.session?.close(); } catch { /* Best-effort transport cleanup. */ }
  }

  setApplicationFocus(focused) {
    this.applicationFocused = !!focused;
    if (focused) this.focus?.focus();
    else this.focus?.blur();
  }

  updateFrame(time) {
    this.frameTime = time;
    if (this.moving) this.sdk?.update3dcontroller({ frame: { time } });
  }

  onConnect() {
    if (this.focus) this.sdk?.create3dmouse(this.focus, this.options.applicationName || 'nurb');
  }

  on3dmouseCreated() {
    window.clearTimeout(this.timer);
    this.timer = null;
    this.sdk?.update3dcontroller({ frame: { timingSource: 1 } });
    if (this.applicationFocused !== null) {
      if (this.applicationFocused) this.focus?.focus();
      else this.focus?.blur();
    }
    this.present('ready');
  }

  onDisconnect(reason) {
    this.disconnect();
    this.present('disconnected', reason === undefined ? undefined : String(reason));
  }

  onStartMotion() {
    stopOrbitInertia(this.options.controls);
    this.options.onMotionStart?.();
    this.present('moving');
  }

  onStopMotion() { this.present('ready'); }

  getCoordinateSystem() {
    return new THREE.Matrix4().makeRotationX(-Math.PI / 2).toArray();
  }
  getFrontView() { return new THREE.Matrix4().makeRotationX(Math.PI / 2).toArray(); }
  getConstructionPlane() { return [0, 0, 1, 0]; }
  getFloorPlane() { return [0, 0, 1, 0]; }
  getUnitsToMeters() { return 0.001; }
  getPerspective() { return true; }
  getViewRotatable() { return this.options.controls.enabled; }
  getViewTarget() { return this.options.controls.target.toArray(); }
  getPivotPosition() { return this.getViewTarget(); }
  getModelExtents() {
    const bounds = this.options.getModelBounds(new THREE.Box3());
    if (bounds.isEmpty() || ![...bounds.min.toArray(), ...bounds.max.toArray()].every(Number.isFinite)) {
      bounds.setFromCenterAndSize(this.options.controls.target, new THREE.Vector3(2, 2, 2));
    }
    return [...bounds.min.toArray(), ...bounds.max.toArray()];
  }
  getFov() { return diagonalFovRadians(this.options.camera.fov, this.options.camera.aspect); }
  getViewMatrix() {
    this.options.camera.updateMatrixWorld(true);
    return this.options.camera.matrixWorld.toArray();
  }
  getViewExtents() {
    const camera = this.options.camera;
    const halfHeight = camera.near * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
    const halfWidth = halfHeight * camera.aspect;
    return [-halfWidth, -halfHeight, -camera.far, halfWidth, halfHeight, -camera.near];
  }
  getViewFrustum() {
    const [left, bottom, far, right, top, near] = this.getViewExtents();
    return [left, right, bottom, top, -near, -far];
  }
  getLookAt() { return null; }
  getPointerPosition() { return null; }
  getSelectionAffine() { return null; }
  getSelectionEmpty() { return true; }
  getSelectionExtents() { return null; }
  getFrameTimingSource() { return 1; }
  getFrameTime() { return this.frameTime; }

  setViewMatrix(data) {
    if (!this.options.controls.enabled || !finiteArray(data, 16)) return;
    const matrix = new THREE.Matrix4().fromArray(data);
    const position = new THREE.Vector3();
    const orientation = new THREE.Quaternion();
    const scale = new THREE.Vector3();
    matrix.decompose(position, orientation, scale);
    if (!finiteArray(position.toArray(), 3) || !finiteArray(orientation.toArray(), 4)
        || scale.distanceTo(new THREE.Vector3(1, 1, 1)) > 1e-4) return;
    const rigid = new THREE.Matrix4().compose(position, orientation, scale);
    if (rigid.elements.some((value, index) => Math.abs(value - data[index]) > 1e-4)) return;
    const { camera, controls } = this.options;
    controls.target.copy(synchronizedOrbitTarget(camera.position, controls.target, position, orientation));
    camera.position.copy(position);
    camera.quaternion.copy(orientation).normalize();
    camera.scale.set(1, 1, 1);
    camera.up.set(0, 1, 0).applyQuaternion(camera.quaternion).normalize();
    camera.updateMatrixWorld(true);
  }

  setViewExtents(_data) {}
  setFov(data) {
    if (!this.options.controls.enabled || typeof data !== 'number'
        || !Number.isFinite(data) || data <= 0 || data >= Math.PI) return;
    this.options.camera.fov = verticalFovDegrees(data, this.options.camera.aspect);
    this.options.camera.updateProjectionMatrix();
  }
  setTarget(data) {
    if (this.options.controls.enabled && finiteArray(data, 3)) this.options.controls.target.fromArray(data);
  }
  setPivotPosition(data) { this.setTarget(data); }
  setPivotVisible(data) { this.pivotVisible = data === true; }
  setLookFrom(data) { if (finiteArray(data, 3)) this.lookOrigin.fromArray(data); }
  setLookDirection(data) { if (finiteArray(data, 3)) this.lookDirection.fromArray(data); }
  setLookAperture(_data) {}
  setSelectionOnly(_data) {}
  setSelectionAffine(_data) {}
  setActiveCommand(_data) {}
  setKeyPress(_data) {}
  setKeyRelease(_data) {}
  setSettingsChanged(_data) {}
  setMoving(data) {
    if (data === true) this.onStartMotion();
    else if (data === false) this.onStopMotion();
  }
  setTransaction(transaction) {
    if (transaction !== 0 || !this.options.controls.enabled) return;
    stopOrbitInertia(this.options.controls);
    this.options.onCameraChange?.();
  }

  present(state, detail) {
    this.state = state;
    this.options.onStateChange?.(state, detail);
  }
}
