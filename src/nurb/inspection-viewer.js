import * as THREE from './vendor/three/build/three.module.min.js';
import { inspectionActions, inspectionGuidance, inspectionHydrateVerification, inspectionTransition, sectionSeriesInitial, sectionSeriesSelect, sectionSeriesSelected, sectionSeriesUpdate } from './inspection-state.js';

export function referenceUsesSourceMaterial(mode) {
  return mode === 'reference' || mode === 'side-by-side';
}

export function referenceMaterialList(material) {
  return Array.isArray(material) ? material : [material];
}

export function referenceMaterialShape(source, make) {
  const materials = referenceMaterialList(source).map(make);
  return Array.isArray(source) ? materials : materials[0];
}

export function referenceMeshes(root) {
  const found = [];
  root?.traverse(node => { if (node.isMesh) found.push(node); });
  return found;
}

export function referenceResources(root, cache) {
  const geometries = new Set(), materials = new Set(), textures = new Set(), images = new Set();
  root?.traverse(node => {
    if (cache && node.geometry) geometries.add(node.geometry);
    const owned = cache
      ? referenceMaterialList(node.material)
      : [...referenceMaterialList(node.userData.referenceSourceMaterial),
        ...referenceMaterialList(node.userData.referenceAnalysisMaterial)];
    for (const material of owned.filter(Boolean)) {
      materials.add(material);
      if (cache) for (const value of Object.values(material)) if (value?.isTexture) {
        textures.add(value);
        const image = value.source?.data;
        if (image && typeof image.close === 'function') images.add(image);
      }
    }
  });
  return {geometries, materials, textures, images};
}

export function referenceDispose(root, cache, preserve = null) {
  const resources = referenceResources(root, cache);
  const kept = preserve ? referenceResources(preserve, true) : null;
  for (const kind of ['geometries','materials','textures','images']) if (kept)
    for (const item of kept[kind]) resources[kind].delete(item);
  const {geometries, materials, textures, images} = resources;
  for (const geometry of geometries) geometry.dispose();
  for (const material of materials) material.dispose();
  for (const texture of textures) texture.dispose();
  for (const image of images) image.close();
}

export function referenceSanitizeScene(scene) {
  const keep = new Set([scene]);
  const include = node => { for (let here = node; here; here = here.parent) keep.add(here); };
  for (const node of referenceMeshes(scene)) {
    include(node);
    if (node.isSkinnedMesh) for (const bone of node.skeleton?.bones || []) include(bone);
  }
  const discarded = new THREE.Group();
  const nodes = [];
  scene.traverse(node => nodes.push(node));
  for (const node of nodes.reverse()) if (node !== scene && !keep.has(node) && node.parent && keep.has(node.parent)) {
    node.parent.remove(node); discarded.add(node);
  }
  // A camera or light is occasionally used as a transform parent. Preserve that
  // transform without importing its rendering behavior into the comparison scene.
  for (const node of nodes) if (node.parent && (node.isCamera || node.isLight || node.isLine || node.isPoints)) {
    const parent = node.parent, replacement = new THREE.Group().copy(node, false);
    for (const child of [...node.children]) replacement.add(child);
    parent.add(replacement); parent.remove(node); discarded.add(node);
  }
  referenceDispose(discarded, true, scene);
  return scene;
}

export function referenceCloneScene(scene) {
  const cloned = scene.clone(true), copies = new Map();
  const pair = (source, target) => {
    copies.set(source, target);
    for (let i = 0; i < source.children.length; i++) pair(source.children[i], target.children[i]);
  };
  pair(scene, cloned);
  scene.traverse(source => {
    if (!source.isSkinnedMesh || !source.skeleton) return;
    const target = copies.get(source);
    const bones = source.skeleton.bones.map(bone => copies.get(bone)).filter(Boolean);
    if (bones.length !== source.skeleton.bones.length) return;
    target.skeleton = new THREE.Skeleton(bones, source.skeleton.boneInverses.map(matrix => matrix.clone()));
    target.bindMode = source.bindMode;
    target.bindMatrix.copy(source.bindMatrix);
    target.bindMatrixInverse.copy(source.bindMatrixInverse);
  });
  return cloned;
}

export function referenceSetMode(root, mode) {
  const source = referenceUsesSourceMaterial(mode);
  for (const node of referenceMeshes(root)) node.material = source
    ? node.userData.referenceSourceMaterial : node.userData.referenceAnalysisMaterial;
}

export function modelGeometryDispose(root) {
  root?.traverse(node => {
    if ((node.isMesh || node.isLineSegments) && !node.userData.nurbReference) node.geometry.dispose();
  });
}

export function displayedMeshDispose(root) {
  const reference = root?.getObjectByName('target');
  if (reference) referenceDispose(reference, false);
  modelGeometryDispose(root);
}

export function componentInfo(node) {
  const metadata = node.userData?.nurb || {};
  return { id: metadata.id || node.name, label: metadata.label || node.name || 'Part',
    role: metadata.role || (node.name === 'context' ? 'context' : 'part') };
}

// A degree-two contour graph establishes inside versus outside before filling.
// Even-odd filling preserves holes; open and coplanar boundaries remain outlines.
export function sectionSegments(geometry, matrix, axis, at, epsilon = 1e-5) {
  const positions = geometry.attributes.position, indices = geometry.index;
  const axes = [0,1,2].filter(value => value !== axis), segments = [];
  let ambiguous = false;
  const vertex = index => new THREE.Vector3().fromBufferAttribute(positions, index).applyMatrix4(matrix).toArray();
  const count = indices ? indices.count : positions.count;
  const key = point => point.map(value => Math.round(value / epsilon)).join(',');
  for (let offset = 0; offset < count; offset += 3) {
    const triangle = [0,1,2].map(i => vertex(indices ? indices.getX(offset + i) : offset + i));
    const distances = triangle.map(point => point[axis] - at);
    if (distances.every(value => value > epsilon) || distances.every(value => value < -epsilon)) continue;
    if (distances.filter(value => Math.abs(value) <= epsilon).length > 1) { ambiguous = true; continue; }
    const hits = [];
    for (let i = 0; i < 3; i++) {
      const j = (i + 1) % 3, a = triangle[i], b = triangle[j], da = distances[i], db = distances[j];
      if (Math.abs(da) <= epsilon) hits.push(axes.map(index => a[index]));
      else if ((da < 0) !== (db < 0) && Math.abs(db) > epsilon) {
        const t = da / (da - db); hits.push(axes.map(index => a[index] + t * (b[index] - a[index])));
      }
    }
    const unique = [...new Map(hits.map(point => [key(point), point])).values()];
    if (unique.length !== 2) continue;
    segments.push(unique);
  }
  return {segments, ambiguous};
}

export function sectionAssemble(segments, ambiguous = false, epsilon = 1e-5) {
  const points = new Map(), edges = new Set();
  const key = point => point.map(value => Math.round(value / epsilon)).join(',');
  for (const segment of segments) {
    const ids = segment.map(key), edge = [...ids].sort().join('|');
    if (edges.has(edge)) { ambiguous = true; continue; }
    edges.add(edge);
    for (let i = 0; i < 2; i++) {
      if (!points.has(ids[i])) points.set(ids[i], {point: segment[i], adjacent: []});
      points.get(ids[i]).adjacent.push(ids[1-i]);
    }
  }
  const open = [...points.values()].some(point => point.adjacent.length !== 2);
  const loops = [], visited = new Set();
  if (!open) for (const start of points.keys()) {
    if (visited.has(start)) continue;
    const loop = []; let previous = null, here = start;
    while (!visited.has(here)) {
      visited.add(here); const node = points.get(here); loop.push(node.point);
      const next = node.adjacent.find(other => other !== previous); previous = here; here = next;
    }
    if (here !== start || loop.length < 3) ambiguous = true;
    else loops.push(loop);
  }
  // Reject crossing boundaries, including overlapping shells in a single mesh.
  const cross = (a,b,c) => (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
  for (let i = 0; i < segments.length && !ambiguous; i++) for (let j = i + 1; j < segments.length; j++) {
    const [a,b] = segments[i], [c,d] = segments[j];
    if (cross(a,b,c)*cross(a,b,d) < -(epsilon ** 2) && cross(c,d,a)*cross(c,d,b) < -(epsilon ** 2)) { ambiguous = true; break; }
  }
  return {loops, segments, valid: !open && !ambiguous, reason: ambiguous ? 'ambiguous or coplanar contours' : open ? 'open contours' : null};
}

export function sectionContours(geometry, matrix, axis, at, epsilon = 1e-5) {
  const cut = sectionSegments(geometry, matrix, axis, at, epsilon);
  return sectionAssemble(cut.segments, cut.ambiguous, epsilon);
}

export function sectionContoursMany(nodes, matrixFor, axis, at, epsilon = 1e-5) {
  // GLTF splits one closed surface into primitives at material boundaries. Join
  // their cut segments before judging degree two, or every such seam looks open.
  const cuts = nodes.map(node => sectionSegments(node.geometry, matrixFor(node), axis, at, epsilon));
  return sectionAssemble(cuts.flatMap(cut => cut.segments), cuts.some(cut => cut.ambiguous), epsilon);
}

// Separable squared Euclidean distance keeps work linear even at large tolerances.
export function sectionMaskDistance(mask, width, height) {
  if (!mask.some(Boolean)) return new Float64Array(width*height).fill(Infinity);
  const far = (width + height) ** 2, first = new Float64Array(width * height), result = new Float64Array(width * height);
  const line = (values, count) => {
    const sites = new Int32Array(count), limits = new Float64Array(count + 1), out = new Float64Array(count);
    let k = 0; sites[0] = 0; limits[0] = -Infinity; limits[1] = Infinity;
    for (let q = 1; q < count; q++) {
      let crossing;
      do {
        const p = sites[k]; crossing = ((values[q] + q*q) - (values[p] + p*p)) / (2*(q-p));
        if (crossing <= limits[k]) k--; else break;
      } while (k >= 0);
      k++; sites[k] = q; limits[k] = crossing; limits[k+1] = Infinity;
    }
    k = 0;
    for (let q = 0; q < count; q++) { while (limits[k+1] < q) k++; out[q] = (q-sites[k])**2 + values[sites[k]]; }
    return out;
  };
  for (let y = 0; y < height; y++) {
    const values = new Float64Array(width);
    for (let x = 0; x < width; x++) values[x] = mask[y*width+x] ? 0 : far;
    first.set(line(values,width),y*width);
  }
  for (let x = 0; x < width; x++) {
    const values = new Float64Array(height);
    for (let y = 0; y < height; y++) values[y] = first[y*width+x];
    const out = line(values,height);
    for (let y = 0; y < height; y++) result[y*width+x] = out[y];
  }
  return result;
}

export function sectionMaskClassify(cad, reference, width, height, tolerancePixels) {
  const limit = Math.max(0,tolerancePixels) ** 2;
  const toCad = sectionMaskDistance(cad,width,height), toReference = sectionMaskDistance(reference,width,height);
  const classes = new Uint8Array(width*height);
  for (let i = 0; i < classes.length; i++) {
    if (cad[i] && reference[i]) classes[i] = 1;
    else if (cad[i]) classes[i] = toReference[i] <= limit ? 1 : 2;
    else if (reference[i]) classes[i] = toCad[i] <= limit ? 1 : 3;
  }
  return classes;
}

export function sectionScaleLabelY(height, embed) {
  return height - (embed ? 44 : 18);
}

export function inspectionSelectedMetrics(metrics, name) {
  if (!name) return metrics;
  return metrics?.inspection_regions?.find(region => region.name === name) || null;
}

export function inspectionRegionStatus(metrics, tolerance) {
  if (!metrics) return 'Waiting for measurements of this region.';
  if (metrics.status === 'unresolved') return metrics.error || 'The region selector could not be resolved.';
  if (metrics.status === 'empty') return 'No sampled surfaces in this region.';
  const verdict = comparisonAbove(metrics,tolerance) ? 'Deviation above tolerance detected.' : 'No sampled deviation above tolerance detected.';
  if (metrics.status === 'partial') {
    const missing = [!metrics.part && 'CAD', !metrics.target && 'reference'].filter(Boolean).join(' and ');
    return `Partial region: no ${missing} samples. ${verdict}`;
  }
  return verdict;
}

export function comparisonRows(metrics) {
  return [['part', 'CAD → reference'], ['target', 'reference → CAD']].map(([key, label]) => {
    const row = metrics?.[key];
    const mm = value => Number.isFinite(value) ? value.toFixed(3) : 'n/a';
    return [label, mm(row?.sampled_max ?? row?.max), mm(row?.p95),
      Number.isFinite(row?.within_tolerance) ? `${(100 * row.within_tolerance).toFixed(1)}%` : 'n/a'];
  });
}

export function comparisonToleranceMatches(requested, actual) {
  // A persisted decimal can round off a binary float's last digit. That must not
  // leave an otherwise current comparison waiting for a value that cannot arrive.
  return Number.isFinite(requested) && Number.isFinite(actual)
    && Math.abs(requested - actual) <= 1e-12 * Math.max(1, Math.abs(requested), Math.abs(actual));
}

export function comparisonCenteredTransform(transform, cadCenter, referenceCenter) {
  if (!Array.isArray(transform) || transform.length !== 16 || !transform.every(Number.isFinite)
      || ![cadCenter, referenceCenter].every(center => Array.isArray(center) && center.length === 3 && center.every(Number.isFinite))) return null;
  const moved = [...transform];
  for (let axis = 0; axis < 3; axis++) moved[axis * 4 + 3] += cadCenter[axis] - referenceCenter[axis];
  return moved;
}

export function comparisonMatrix(transform) {
  if (!Array.isArray(transform) || transform.length !== 16 || !transform.every(Number.isFinite)) return null;
  return new THREE.Matrix4().set(...transform);
}

export function comparisonTransform(matrix) {
  const e = matrix.elements;
  return [e[0], e[4], e[8], e[12], e[1], e[5], e[9], e[13], e[2], e[6], e[10], e[14], e[3], e[7], e[11], e[15]];
}

export function comparisonRigid(transform) {
  const matrix = comparisonMatrix(transform);
  if (!matrix) return null;
  const position = new THREE.Vector3(), rotation = new THREE.Quaternion(), scale = new THREE.Vector3();
  matrix.decompose(position, rotation, scale);
  const euler = new THREE.Euler().setFromQuaternion(rotation, 'XYZ');
  return {
    translation: position.toArray(),
    rotation: [euler.x, euler.y, euler.z].map(THREE.MathUtils.radToDeg),
  };
}

export function comparisonRigidTransform(values) {
  if (!values || !values.every(Number.isFinite)) return null;
  const [tx, ty, tz, rx, ry, rz] = values;
  const rotation = new THREE.Quaternion().setFromEuler(new THREE.Euler(
    THREE.MathUtils.degToRad(rx), THREE.MathUtils.degToRad(ry), THREE.MathUtils.degToRad(rz), 'XYZ'));
  const matrix = new THREE.Matrix4().compose(new THREE.Vector3(tx, ty, tz), rotation, new THREE.Vector3(1, 1, 1));
  return comparisonTransform(matrix);
}

export function comparisonWorst(metrics) {
  const raw = metrics?.worst_regions || metrics?.regions;
  if (Array.isArray(raw)) return raw;
  if (!raw || typeof raw !== 'object') return [];
  return Object.entries(raw).flatMap(([side, regions]) => Array.isArray(regions)
    ? regions.map(region => ({ side, ...region })) : []);
}

export function comparisonAbove(metrics, tolerance) {
  if (typeof metrics?.detected_above_tolerance === 'boolean') return metrics.detected_above_tolerance;
  return ['part', 'target'].some(side => {
    const row = metrics?.[side];
    return Number.isFinite(row?.sampled_max ?? row?.max) && (row.sampled_max ?? row.max) > tolerance;
  });
}

export function symmetryFeatureSignature(entry) {
  return JSON.stringify((entry?.target?.regions || []).filter(r => r.feature).map(r => [r.name,r.feature]));
}

export function symmetryCategoryRows(report) {
  const fixed = value => Number.isFinite(value) ? value.toFixed(4) : 'unknown';
  return [['Category','Count / unmatched','Negative max mm','Positive max mm','Threshold mm','Result','On-plane max mm'],
    ...[['Authored CAD centers',report.feature_centers?.cad_centers],
        ['Reference annotations',report.feature_centers?.reference_points],
        ['Periodic seams',report.cad?.periodic_seams]].map(([label,category]) =>
      [label,`${category?.count || 0} total, ${category?.unmatched_count || 0} unmatched`,fixed(category?.sides?.negative?.max_mm),
        fixed(category?.sides?.positive?.max_mm),fixed(category?.tolerance_mm),(category?.status || 'not_assessed').replaceAll('_',' '),fixed(category?.sides?.on_plane?.max_mm)])];
}

// The viewer owns geometry and evidence. Live accessors keep this controller on the
// same state across rebuilds, selection changes, captures, and embedded views.
export function createInspectionController(viewer) {
  function syncPins() { viewer.pins.visible = viewer.pinsWanted && viewer.pinsPlaced && viewer.inspectionMode !== 'reference'; }

  function referenceInstance(scene, displayScale) {
    const root = new THREE.Group(), content = referenceCloneScene(scene);
    root.name = 'target';
    content.scale.multiplyScalar(Number.isFinite(displayScale) && displayScale > 0 ? displayScale : 1);
    root.add(content);
    for (const node of referenceMeshes(root)) {
      if (!node.geometry.attributes.normal) node.geometry.computeVertexNormals();
      node.userData.nurbReference = true;
      node.userData.referenceSourceMaterial = referenceMaterialShape(node.material, source => {
        const cloned = source.clone();
        cloned.side = THREE.DoubleSide;
        cloned.clippingPlanes = [viewer.plane];
        cloned.needsUpdate = true;
        return cloned;
      });
      node.userData.referenceAnalysisMaterial = referenceMaterialShape(node.material, () => {
        const cloned = viewer.ghostMaterial.clone();
        cloned.clippingPlanes = [viewer.plane];
        return cloned;
      });
    }
    return root;
  }

  async function ghostAttach(name, entry) {
    const t = entry.target;
    const group = viewer.mesh;
    if (!t || !t.stamp || !viewer.renderer || !group) return;
    let rec = viewer.ghostGeo.get(name), retired = null;
    if (!rec || rec.stamp !== t.stamp) {
      let gltf;
      // Not awaited by any caller, so a rejection here would surface as an unhandled
      // error for what is only a race: a server restart or a rebuild replacing the
      // entry mid-fetch. The next paint fetches again.
      try { gltf = await viewer.loader.loadAsync(`/glb/${name}.target.glb?v=${t.stamp}`); }
      catch (e) { return; }
      const source = referenceSanitizeScene(gltf.scene);
      if (!referenceMeshes(source).length) return;
      if (viewer.current !== name || viewer.mesh !== group || viewer.parts.get(name)?.target?.stamp !== t.stamp) {
        referenceDispose(source, true);
        return;
      }
      const latest = viewer.ghostGeo.get(name);
      if (latest?.stamp === t.stamp) {
        if (latest.scene !== source) referenceDispose(source, true);
        rec = latest;
      } else {
        rec = { stamp: t.stamp, scene: source };
        retired = latest?.scene;
        viewer.ghostGeo.set(name, rec);
      }
    }
    // The fetch was awaited, so the part on screen may have rebuilt or moved on;
    // attach only to the exact group that asked for this ghost, and never twice.
    if (viewer.current !== name || viewer.mesh !== group || viewer.parts.get(name)?.target?.stamp !== t.stamp) return;
    const stale = group.getObjectByName('target');
    if (stale) { group.remove(stale); referenceDispose(stale, false); }
    if (retired && retired !== rec.scene) referenceDispose(retired, true);
    const g = viewer.referenceInstance(rec.scene, t.display_scale);
    referenceSetMode(g, viewer.inspectionMode);
    const shownTransform = viewer.comparePreview.get(name)?.transform || t.transform;
    if (Array.isArray(shownTransform) && shownTransform.length === 16) {
      g.matrixAutoUpdate = false;
      g.matrix.set(...shownTransform);
    } else g.position.set(...(t.offset || [0, 0, 0]));
    g.visible = viewer.ghostWanted;
    group.add(g);
    viewer.comparePanel(viewer.parts.get(name) || entry);
    return g;
  }

  function modelNodes() {
    const nodes = [];
    viewer.mesh?.traverse(node => { if (node.isMesh && !node.userData.nurbReference) nodes.push(node); });
    return nodes;
  }

  function inspectionRestore(name) {
    if (viewer.inspectionFor === name) return;
    viewer.inspectionFor = name; viewer.inspectionToleranceOverride = null;
    viewer.datumReset();
    viewer.inspectionRegionName = null; viewer.inspectionRegionError = null;
    viewer.inspectionFrameBox = null;
    viewer.hiddenComponents = new Set();
    if (!viewer.view && !viewer.q.has('mode')) {
      try {
        const state = JSON.parse(viewer.localStorage.getItem(viewer.inspectionKey(name)) || '{}');
        viewer.inspectionMode = viewer.INSPECTION_MODES.includes(state.mode) ? state.mode : 'model';
        viewer.hiddenComponents = new Set(Array.isArray(state.hidden) ? state.hidden : []);
        if (!viewer.q.has('cut')) {
          viewer.cutAxis = ['x','y','z'].includes(state.axis) ? state.axis : 'z';
          viewer.cutAt = Number.isFinite(state.fraction) ? state.fraction : .5;
          viewer.cutMm = Number.isFinite(state.mm) ? state.mm : null;
        }
      } catch { viewer.inspectionMode = 'model'; }
    }
    if (viewer.q.has('hide')) viewer.hiddenComponents = new Set(viewer.q.get('hide').split(','));
    viewer.sectionDrawingKey = null;
  }

  function componentAncestors(node) {
    const result = []; let parent = node.userData?.nurb?.parent;
    while (parent && !result.includes(parent)) { result.push(parent); parent = viewer.inspectionGroups.get(parent)?.parent; }
    return result;
  }

  function componentMembers(selector) {
    const ids = new Set([...viewer.inspectionGroups.values(),...viewer.modelNodes().map(componentInfo)]
      .filter(info => info.id === selector || info.label === selector).map(info => info.id));
    return viewer.modelNodes().filter(node => [componentInfo(node).id,...viewer.componentAncestors(node)].some(id => ids.has(id)));
  }

  function componentResolveHidden() {
    const selectors = viewer.hiddenComponents;
    viewer.hiddenComponents = new Set([...viewer.inspectionGroups.values(),...viewer.modelNodes().map(componentInfo)]
      .filter(info => selectors.has(info.id) || selectors.has(info.label)).map(info => info.id));
  }

  function componentHidden(node) {
    return [componentInfo(node).id, ...viewer.componentAncestors(node)].some(id => viewer.hiddenComponents.has(id));
  }

  function sectionDifference() {
    if (viewer.inspectionMode !== 'section' || !viewer.mesh) return;
    const reference = viewer.mesh.getObjectByName('target');
    if (!reference) { viewer.sectionStatus.textContent = 'Reference geometry is not ready. No filled difference.'; viewer.window.__nurb.section = null; return; }
    const axis = {x:0,y:1,z:2}[viewer.cutAxis], at = viewer.plane.constant / viewer.cutSign - viewer.mesh.position[viewer.cutAxis];
    const tolerance = Math.max(0,Number(viewer.inspectionToleranceOverride ?? viewer.parts.get(viewer.current)?.target?.tolerance_mm) || 0);
    const key = [viewer.current, viewer.mesh.userData.token, viewer.parts.get(viewer.current)?.target?.stamp, tolerance, viewer.cutAxis, at,
      reference.matrix.elements.join(','), [...viewer.hiddenComponents].join(','), ...(viewer.inspectionCaptureViewport || [viewer.main.clientWidth,viewer.main.clientHeight]), viewer.camera.zoom, viewer.controls.target.toArray().join(',')].join(':');
    if (key === viewer.sectionDrawingKey) return;
    viewer.sectionDrawingKey = key;
    viewer.mesh.updateMatrixWorld(true);
    const inverse = viewer.mesh.matrixWorld.clone().invert();
    const cad = viewer.modelNodes().filter(node => node.visible && componentInfo(node).role !== 'context');
    const gather = node => sectionContours(node.geometry, new THREE.Matrix4().multiplyMatrices(inverse, node.matrixWorld), axis, at);
    const referenceNodes = referenceMeshes(reference);
    const referenceContours = sectionContoursMany(referenceNodes,
      node => new THREE.Matrix4().multiplyMatrices(inverse, node.matrixWorld), axis, at);
    const sides = [cad.map(gather), [referenceContours]];
    const all = sides.flat(), segments = all.flatMap(result => result.segments);
    const [canvasWidth,canvasHeight]=viewer.inspectionCaptureViewport || [viewer.main.clientWidth,viewer.main.clientHeight];
    const width = Math.max(1, Math.round(canvasWidth / 2)), height = Math.max(1,canvasHeight);
    viewer.sectionCanvas.width = width; viewer.sectionCanvas.height = height;
    const context = viewer.sectionCanvas.getContext('2d'); context.fillStyle = '#16181d'; context.fillRect(0,0,width,height);
    const prefix = `${viewer.cutAxis.toUpperCase()} = ${at.toFixed(3)} mm (part frame). `;
    if (!segments.length) { viewer.sectionStatus.textContent = prefix + 'No contours intersect this plane.'; viewer.window.__nurb.section = {axis:viewer.cutAxis,position_mm:at,frame:'part_mm',valid:true,empty:true}; return; }
    const valid = all.every(result => result.valid);
    viewer.cap.visible = viewer.cutting && valid;
    const planeAxes = [0,1,2].filter(index => index !== axis);
    const center = viewer.controls.target.clone().sub(viewer.mesh.position).toArray();
    const scale = height / Math.abs(viewer.camera.top - viewer.camera.bottom) * viewer.camera.zoom;
    const project = point => [width/2+(point[0]-center[planeAxes[0]])*scale, height/2-(point[1]-center[planeAxes[1]])*scale];
    if (valid) {
      const masks = sides.map(results => {
        const canvas = viewer.document.createElement('canvas'); canvas.width = width; canvas.height = height;
        const mask = canvas.getContext('2d'); mask.fillStyle = '#fff';
        for (const result of results) {
          mask.beginPath();
          for (const loop of result.loops) { loop.forEach((point, i) => mask[i ? 'lineTo' : 'moveTo'](...project(point))); mask.closePath(); }
          mask.fill('evenodd');
        }
        const rgba = mask.getImageData(0,0,width,height).data;
        return Uint8Array.from({length:width*height}, (_,i) => rgba[i*4+3] > 127 ? 1 : 0);
      });
      const image = context.getImageData(0,0,width,height);
      const classes = sectionMaskClassify(masks[0],masks[1],width,height,tolerance*scale);
      const colors = [null,[139,160,153,255],[240,194,116,255],[98,199,239,255]];
      for (let i = 0; i < classes.length; i++) if (classes[i]) image.data.set(colors[classes[i]],i*4);
      context.putImageData(image,0,0);
    }
    sides.forEach((results, side) => {
      context.strokeStyle = side ? '#62c7ef' : '#f0c274'; context.lineWidth = 1;
      context.beginPath();
      for (const result of results) for (const [a,b] of result.segments) { context.moveTo(...project(a)); context.lineTo(...project(b)); }
      context.stroke();
    });
    const axes = ['X','Y','Z'].filter((_, index) => index !== axis);
    context.fillStyle = '#e6e8ec'; context.font = '12px monospace';
    context.fillText(`${axes[0]} →   ${axes[1]} ↑   ${(100/scale).toFixed(2)} mm / 100 px`, 16, sectionScaleLabelY(height,viewer.embed));
    viewer.sectionStatus.textContent = prefix + (valid ? `Gray: shared or within ${tolerance.toFixed(3)} mm · amber: excess CAD · blue: excess reference. Unmuted outlines; triangle sections, raster fill.`
      : `No filled difference: ${[...new Set(all.map(result => result.reason).filter(Boolean))].join('; ')}. Outlines only; move the plane or repair the mesh.`);
    viewer.window.__nurb.section = {axis: viewer.cutAxis, position_mm: at, frame: 'part_mm', valid, tolerance_mm:tolerance, method: 'triangle-plane contours, even-odd raster masks with Euclidean tolerance'};
  }

  function datumReset() {
    if (viewer.datumPreview) viewer.comparePreview.delete(viewer.datumPreview.name);
    viewer.datumPreview = null;
    viewer.document.getElementById('datumapply').disabled = true;
    viewer.document.getElementById('datumstatus').textContent = '';
  }

  function datumSend(save) {
    const status = viewer.document.getElementById('datumstatus');
    try {
      if (!viewer.sock || viewer.sock.readyState !== viewer.WebSocket.OPEN) throw new Error('Reconnect to preview alignment.');
      if (save) {
        if (!viewer.datumPreview || viewer.datumPreview.name !== viewer.current || viewer.datumPreview.token !== viewer.parts.get(viewer.current)?.token) {
          viewer.datumReset(); throw new Error('The selected part or build changed. Preview alignment again before applying.');
        }
        if (!viewer.datumPreview.transform) throw new Error('Wait for the alignment preview before applying.');
      } else {
        const token = viewer.parts.get(viewer.current)?.token;
        if (!viewer.current || !token) throw new Error('Build the selected part before previewing alignment.');
        viewer.datumPreview = {name:viewer.current,token,operation:viewer.datumOperation(),transform:null};
        viewer.document.getElementById('datumapply').disabled = true;
      }
      const preview = viewer.datumPreview;
      viewer.sock.send(JSON.stringify({type:'target_alignment',name:preview.name,operation:preview.operation,token:preview.token,save}));
      status.textContent = save ? 'Saving alignment…' : 'Measuring alignment preview…';
    } catch (error) { status.textContent = error.message; }
  }

  function datumLanded(message) {
    if (message.name !== viewer.current || !viewer.datumPreview || viewer.datumPreview.name !== viewer.current || viewer.datumPreview.token !== viewer.parts.get(viewer.current)?.token) return;
    if (message.token && message.token !== viewer.datumPreview.token) return;
    const status = viewer.document.getElementById('datumstatus');
    if (message.error) { status.textContent = message.error; return; }
    if (message.token !== viewer.datumPreview.token) return;
    if (Array.isArray(message.written) ? message.written.length > 0 : !!message.written) { viewer.datumPreview = null; viewer.document.getElementById('datumapply').disabled = true; status.textContent = 'Alignment saved.'; return; }
    if (!Array.isArray(message.transform) || message.transform.length !== 16 || !message.transform.every(Number.isFinite)) {
      viewer.datumReset(); status.textContent = 'The alignment response had no valid transform. Preview again.'; return;
    }
    viewer.datumPreview.transform = [...message.transform];
    viewer.comparisonPreview(viewer.datumPreview.transform, 'Datum alignment preview. Apply to save it.');
    viewer.inspectionSetMode('overlay');
    viewer.document.getElementById('datumapply').disabled = false;
    const residual = Number.isFinite(message.max_residual_mm) ? ` Maximum residual ${message.max_residual_mm.toFixed(3)} mm.` : '';
    status.textContent = `Preview ready.${residual} Plane and axis alignment leave unconstrained motion unchanged.`;
  }

  function regionSave(remove = false) {
    const status = viewer.document.getElementById('regionstatus');
    try {
      if (!viewer.sock || viewer.sock.readyState !== viewer.WebSocket.OPEN) throw new Error('Reconnect to save regions.');
      if (viewer.regionWrites.has(viewer.current)) throw new Error('Wait for the previous region change to finish.');
      const name = viewer.document.getElementById('regionname').value.trim();
      const previous = viewer.document.getElementById('regionexisting').value;
      const regions = [...(viewer.parts.get(viewer.current)?.target?.regions || [])].filter(region => region.name !== (previous || name));
      if (!remove) regions.push(viewer.regionValues());
      viewer.sock.send(JSON.stringify({type:'target_settings',name:viewer.current,regions}));
      viewer.regionWrites.set(viewer.current,regions);
      viewer.document.getElementById('regionsave').disabled = true;
      viewer.document.getElementById('regionremove').disabled = true;
      status.textContent = remove ? 'Removing region…' : 'Saving region…';
    } catch (error) { status.textContent = error.message; }
  }

  function evidenceRender() {
    const guidance=viewer.document.getElementById('evidenceguidance'); if (!guidance) return;
    const actions=inspectionActions(viewer.evidenceWorkflow);
    guidance.textContent=inspectionGuidance(viewer.evidenceWorkflow);
    viewer.featureField('inspect').disabled=!actions.inspect;
    viewer.document.getElementById('verifyrun').disabled=!actions.verify;
    viewer.document.getElementById('verifycancel').disabled=!actions.cancel;
    viewer.inspectionField('save').disabled=!actions.save || viewer.inspectionBusy;
    viewer.inspectionField('capture').disabled=!actions.capture || viewer.inspectionBusy;
    viewer.document.getElementById('inspectionworkflow').dataset.phase=viewer.evidenceWorkflow.capture.status==='capturing'?'capture':viewer.evidenceWorkflow.verification.status;
  }

  function featureSectionValue() {
    const offsets = viewer.featureField('offsets').value.trim().split(/[\s,]+/).filter(Boolean).map(Number);
    return {name:viewer.featureField('sectionname').value.trim(),origin_mm:viewer.inspectionVector(viewer.featureField('origin').value,'Local origin'),
      normal:viewer.inspectionVector(viewer.featureField('normal').value,'Local normal'),x_direction:viewer.inspectionVector(viewer.featureField('x').value,'Local horizontal direction'),
      offsets_mm:offsets,tolerance_mm:Number(viewer.featureField('tolerance').value),expected:viewer.featureField('expected').value.trim()};
  }

  function featureSectionStore() {
    if (viewer.featureField('sectionenabled').checked) viewer.featureSeries=sectionSeriesUpdate(viewer.featureSeries,viewer.featureSectionValue());
  }

  function featureSectionLoad(index) {
    viewer.featureSeries=sectionSeriesSelect(viewer.featureSeries,index);
    const section=sectionSeriesSelected(viewer.featureSeries);
    viewer.featureField('sectionchoice').replaceChildren(...viewer.featureSeries.items.map((item,i)=>new viewer.Option(item.name || `Series ${i+1}`,String(i))));
    viewer.featureField('sectionchoice').value=String(viewer.featureSeries.index); viewer.featureField('sectionchoice').disabled=!section;
    viewer.featureField('sectionremove').disabled=!section;
    for (const [id,key,fallback] of [['origin','origin_mm',[0,0,0]],['normal','normal',[0,0,1]],['x','x_direction',[1,0,0]],['offsets','offsets_mm',[0]]]) viewer.featureField(id).value=(section?.[key] || fallback).join(' ');
    viewer.featureField('sectionname').value=section?.name || 'Mating profile'; viewer.featureField('tolerance').value=section?.tolerance_mm ?? 0; viewer.featureField('expected').value=section?.expected || '';
    viewer.featureField('extras').textContent=viewer.featureSeries.items.length ? `${viewer.featureSeries.index+1} of ${viewer.featureSeries.items.length} independently oriented section series.` : 'Add a section series to inspect this interface.';
  }

  function featureRegionValues(region) {
    if (!viewer.featureField('enabled').checked) return region;
    const previous = viewer.selectedFeatureRegion()?.feature;
    const feature = {...previous, id: previous?.id || viewer.crypto.randomUUID()};
    for (const key of ['role','configuration','orientation','notes','symmetry_group']) feature[key] = viewer.featureField(key.replace('_','')).value.trim();
    for (const key of ['required','excluded','links']) feature[key] = viewer.featureField(key).value.split('\n').map(line => line.trim()).filter(Boolean);
    const point = viewer.featureField('point').value.trim();
    if (point) feature.reference_point_mm = viewer.inspectionVector(point,'Reference annotation'); else delete feature.reference_point_mm;
    const center = viewer.featureField('center').value.trim();
    if (center) feature.center_mm = viewer.inspectionVector(center,'CAD center'); else delete feature.center_mm;
    delete feature.point_mm;
    const size = viewer.featureField('size').value.trim();
    if (size) feature.feature_size_mm = Number(size); else delete feature.feature_size_mm;
    delete feature.feature_scale_mm;
    const uncertainty = viewer.featureField('uncertainty').value.trim();
    if (uncertainty) feature.uncertainty_mm = Number(uncertainty); else delete feature.uncertainty_mm;
    if (viewer.featureField('sectionenabled').checked) {
      viewer.featureSectionStore(); feature.sections=structuredClone(viewer.featureSeries.items);
    } else feature.sections = [];
    return {...region,feature};
  }

  function featureEditorLoad(region) {
    viewer.featureDirty = false;
    const feature = region?.feature;
    const interfaceSelect=viewer.document.getElementById('evidenceinterface');
    if (interfaceSelect && [...interfaceSelect.options].some(option=>option.value===region?.name)) interfaceSelect.value=region.name;
    else if (interfaceSelect) interfaceSelect.value='';
    viewer.featureField('enabled').checked = !!feature; viewer.featureField('id').value = feature?.id || '';
    for (const key of ['role','configuration','orientation','notes','symmetry_group']) viewer.featureField(key.replace('_','')).value = feature?.[key] || '';
    for (const key of ['required','excluded','links']) viewer.featureField(key).value = (feature?.[key] || []).join('\n');
    viewer.featureField('point').value = feature?.reference_point_mm?.join(' ') || '';
    viewer.featureField('center').value = (feature?.center_mm || feature?.point_mm)?.join(' ') || '';
    viewer.featureField('size').value = feature?.feature_size_mm ?? feature?.feature_scale_mm ?? '';
    viewer.featureField('uncertainty').value = feature?.uncertainty_mm ?? '';
    viewer.featureSeries=sectionSeriesInitial(feature?.sections || []);
    viewer.featureField('sectionenabled').checked = !!viewer.featureSeries.items.length; viewer.featureSectionLoad(0);
    viewer.evidenceWorkflow=inspectionTransition(viewer.evidenceWorkflow,{type:'select',part:viewer.current,token:viewer.parts.get(viewer.current)?.token,
      interfaceId:feature?.id || null,source:viewer.inspectionField('sectionquality')?.value || viewer.evidenceWorkflow.source,
      freshness:viewer.parts.get(viewer.current)?.target?.stale?'stale':'current'});
    viewer.featureInspection = null; viewer.featureField('review').disabled = true; viewer.featureField('plot').setAttribute('hidden',''); viewer.featureField('station').hidden = true;
    viewer.featureField('status').textContent = ''; viewer.featureEditorState(viewer.parts.get(viewer.current));
  }

  function featureVerifiedResult(entry, featureId=viewer.selectedFeatureRegion()?.feature?.id) {
    const verification=entry?.target?.verification;
    if (verification?.status!=='measured' || verification.token!==entry?.token || entry?.target?.stale) return null;
    return (verification.metrics?.feature_evidence || []).find(result=>result.id===featureId) || null;
  }

  function featureEditorState(entry) {
    const selected = viewer.selectedFeatureRegion(), record = entry?.target?.feature_evidence?.find(item => item.id === selected?.feature?.id);
    viewer.featureField('freshness').textContent = viewer.featureDirty ? 'Unsaved feature edits: save the region before inspecting.' : !selected?.feature ? 'Save a feature to keep its identity and evidence.'
      : entry?.target?.stale ? 'Evidence stale: model or reference settings changed.'
      : record?.status === 'current' ? 'Recorded inspection matches the current geometry, reference, alignment and feature contract. This is evidence, not a fit certificate.'
      : record?.status === 'stale' ? 'Evidence stale: geometry, reference, alignment or feature contract changed. Inspect again.'
      : 'No inspection recorded for this feature.';
    if (viewer.evidenceWorkflow.token && entry?.token && viewer.evidenceWorkflow.token!==entry.token)
      viewer.evidenceWorkflow=inspectionTransition(viewer.evidenceWorkflow,{type:'rebuild',token:entry.token,freshness:entry.target?.stale?'stale':'current'});
    viewer.evidenceWorkflow={...viewer.evidenceWorkflow,part:viewer.current,token:entry?.token,interfaceId:selected?.feature?.id || null,
      freshness:entry?.target?.stale?'stale':viewer.evidenceWorkflow.freshness,dirty:viewer.featureDirty};
    viewer.evidenceWorkflow=inspectionHydrateVerification(viewer.evidenceWorkflow,entry);
    viewer.evidenceRender();
    viewer.featureField('verified').disabled = viewer.featureDirty || !viewer.featureVerifiedResult(entry,selected?.feature?.id);
    if (viewer.featureInspection && (viewer.featureInspection.name !== viewer.current || viewer.featureInspection.token !== entry?.token || entry?.target?.stale
        || (record?.identity && record.identity.token !== viewer.featureInspection.result?.identity?.token)
        || (viewer.featureInspection.result?.source==='verified' && (entry?.target?.verification?.status!=='measured'
            || viewer.featureInspection.result.verification_request_id!==entry?.target?.verification?.request_id)))) {
      viewer.featureInspection = null; viewer.featureField('review').disabled = true; viewer.featureField('plot').setAttribute('hidden',''); viewer.featureField('station').hidden = true;
      viewer.featureField('status').textContent = 'Displayed sections expired after a rebuild, reference or feature change. Inspect again.';
    }
    viewer.featureExportState();
  }

  function featureExportState() {
    const entry=viewer.parts.get(viewer.current);
    const ready = !viewer.featureDirty && viewer.featureInspection?.token === viewer.parts.get(viewer.current)?.token
      && !entry?.target?.stale && viewer.featureInspection?.result?.id
      && viewer.featureInspection?.result?.identity?.token && viewer.featureInspection?.result?.cad?.length;
    const currentVerification=viewer.featureInspection?.result?.source!=='verified'
      || entry?.target?.verification?.status==='measured' && viewer.featureInspection.result.verification_request_id===entry?.target?.verification?.request_id;
    viewer.featureField('json').disabled = !ready || !currentVerification;
    viewer.featureField('svg').disabled = !ready || !currentVerification;
  }

  function featureExport(format) {
    viewer.featureEditorState(viewer.parts.get(viewer.current));
    if (viewer.featureDirty || !viewer.featureInspection || viewer.featureField(format).disabled || !viewer.sock || viewer.sock.readyState !== viewer.WebSocket.OPEN) return;
    viewer.featureField('status').textContent = 'Preparing current section evidence…';
    viewer.sock.send(JSON.stringify({type:'feature_export',name:viewer.current,token:viewer.featureInspection.token,
      feature_id:viewer.featureInspection.result.id,identity:viewer.featureInspection.result.identity,
      verification_request_id:viewer.featureInspection.result.source==='verified' ? viewer.featureInspection.result.verification_request_id : undefined,
      format,station:Number(viewer.featureField('station').value || 0)}));
  }

  async function featureExportLanded(message) {
    if (message.name !== viewer.current) return;
    if (message.error) { viewer.featureField('status').textContent = message.error; return; }
    if (viewer.featureDirty || message.token !== viewer.parts.get(viewer.current)?.token || viewer.parts.get(viewer.current)?.target?.stale
        || message.identity?.token !== viewer.featureInspection?.result?.identity?.token) return;
    try {
      const saved = await viewer.window.downloadArtifact(message.artifact);
      viewer.featureField('status').textContent = saved?.path ? 'Section evidence saved.' : 'Section evidence download started.';
    } catch (error) { viewer.featureField('status').textContent = `Could not export section evidence: ${error.message}`; }
  }

  function featurePlot() {
    if (!viewer.featureInspection) return;
    const result = viewer.featureInspection.result, index = Number(viewer.featureField('station').value || 0), svg = viewer.featureField('plot');
    svg.replaceChildren();
    const cuts = [result.cad[index],result.reference[index]];
    const all = cuts.flatMap(cut => cut?.loops.flatMap(loop => loop.points_mm) || []);
    const make = (name, attrs, text) => { const el = viewer.document.createElementNS('http://www.w3.org/2000/svg',name); for (const [key,value] of Object.entries(attrs)) el.setAttribute(key,value); if (text) el.textContent = text; svg.append(el); return el; };
    if (!all.length) { make('text',{x:15,y:35,fill:'#ddd'},'This station misses both surfaces.'); return; }
    const low=[0,1].map(axis => all.reduce((n,p)=>Math.min(n,p[axis]),Infinity)), high=[0,1].map(axis => all.reduce((n,p)=>Math.max(n,p[axis]),-Infinity));
    const scale=Math.min(320/Math.max(.01,high[0]-low[0]),180/Math.max(.01,high[1]-low[1]));
    const project=p=>[180+(p[0]-(low[0]+high[0])/2)*scale,112-(p[1]-(low[1]+high[1])/2)*scale];
    cuts.forEach((cut,side) => (cut?.loops || []).forEach(loop => {
      const points=loop.closed ? [...loop.points_mm,loop.points_mm[0]] : loop.points_mm;
      make('polyline',{points:points.map(p=>project(p).join(',')).join(' '),fill:'none',stroke:side?'#62c7ef':'#f0c274','stroke-width':1.4});
    }));
    make('text',{x:12,y:229,fill:'#ddd','font-size':11},`u →  v ↑   ${(100/scale).toPrecision(4)} mm / 100 px`);
    svg.setAttribute('aria-label',`${cuts[0]?.name || 'Local section'}, CAD amber, reference blue; ${cuts[0]?.expected || 'no expected profile recorded'}`);
  }

  function featureEditorInput(event) {
    if (['regionexisting','featurestation','featuresectionchoice','featurereviewnote'].includes(event.target.id)) return;
    viewer.featureDirty = true; viewer.featureInspection = null; viewer.featureField('review').disabled = true;
    viewer.featureField('inspect').disabled = true; viewer.featureField('plot').setAttribute('hidden',''); viewer.featureField('station').hidden = true;
    viewer.featureEditorState(viewer.parts.get(viewer.current));
    viewer.featureField('status').textContent='Save the edited region before inspecting or recording its evidence.';
    viewer.symmetryPanel(viewer.parts.get(viewer.current));
  }

  function verificationShow(entry) {
    if (entry?.target?.verification) viewer.verificationHistory.set(entry.name,entry.target.verification);
    const state = entry?.target?.verification || viewer.verificationHistory.get(entry?.name), output = viewer.document.getElementById('verifyresult');
    const currentState = state && state.token === entry?.token && !entry?.target?.stale;
    viewer.evidenceWorkflow=inspectionHydrateVerification(viewer.evidenceWorkflow,entry);
    viewer.evidenceRender();
    output.replaceChildren();
    const status = viewer.document.getElementById('verifystatus');
    if (!state) { status.textContent = 'No precise verification for this build.'; return; }
    if (!currentState) { status.textContent = 'Precise evidence is stale after a model or reference change. Run verification again.'; return; }
    status.textContent = `${state.phase || state.status}${state.error ? ': '+state.error : ''}`;
    const add = text => { const p=viewer.document.createElement('p'); p.textContent=text; output.append(p); };
    const policy=state.provenance;
    if (policy) {
      add(`Requested deflection ${policy.absolute_deflection_mm} mm; budget ${policy.timeout_s} s / ${Number(policy.max_triangles).toLocaleString()} triangles.`);
      if (policy.memory) add(`RSS limit ${policy.memory.limit_mb} MiB: ${policy.memory.enforcement}. ${policy.memory.limitation}`);
      for (const warning of policy.warnings || []) add(warning);
    }
    if (state.resources) add(`Phase: ${state.resources.phase || state.phase}; peak sampled child RSS ${state.resources.peak_rss_mb == null ? 'not recorded' : Number(state.resources.peak_rss_mb).toFixed(1)+' MiB'}.`);
    if (entry.target.last_verification && state.status!=='measured') add('The last completed result is retained separately. It is not the result of this request. The live display estimate is unchanged.');
    if (state.status === 'measured' && state.metrics) {
      for (const [key,label] of [['part','CAD → reference'],['target','reference → CAD']]) {
        const result=state.metrics[key];
        if (result) add(`${label}: sampled max ${Number(result.sampled_max).toFixed(4)} mm; p95 ${Number(result.p95).toFixed(4)} mm.`);
      }
      add(state.metrics.detected_above_tolerance ? 'Sampled deviations exceed the accepted tolerance.' : 'No sampled deviation exceeds the accepted tolerance. Check every functional feature.');
      add('Meshing deflection and sampled distances do not certify physical fit or an exact error bound.');
    } else if (['unknown','stale','cancelled'].includes(state.status)) add('No precise measurement was produced for this request.');
  }

  function verificationLanded(message) {
    if (message.current_verification) message=message.current_verification;
    const entry=viewer.parts.get(message.name);
    if (!entry?.target || message.token !== entry.token) return;
    const previous=entry.target.verification;
    if (previous?.request_id && message.request_id !== previous.request_id && message.status !== 'queued'
        && message.replaces_request_id !== previous.request_id) return;
    entry.target.verification=message;
    if (message.name === viewer.current) {
      viewer.evidenceWorkflow=inspectionHydrateVerification(viewer.evidenceWorkflow,entry);
      if (message.status==='measured') {
        const result=viewer.featureVerifiedResult(entry);
        if (result) viewer.featureInspectionLanded({name:viewer.current,token:entry.token,result});
      }
      viewer.verificationShow(entry); viewer.featureEditorState(entry);
    }
  }

  function inspectionRegionBounds(entry, name) {
    const measured = inspectionSelectedMetrics(entry?.target?.metrics,name);
    const region = entry?.target?.regions?.find(region => region.name === name) || measured?.selector || measured;
    if (!region) return {error:`No inspection region named ${name}. Choose a saved region.`};
    if (measured?.status === 'unresolved') return {error:`Region ${name}: ${measured.error || 'the selector could not be resolved'}.`};
    const bounds = measured?.bounds_mm || region.bounds_mm || measured?.capture?.bounds_mm;
    if (bounds && viewer.mesh) {
      const lo = bounds.min, hi = bounds.max;
      if ([lo,hi].every(values => Array.isArray(values) && values.length === 3 && values.every(Number.isFinite))
          && lo.every((value,index) => value <= hi[index])) {
        return {box:new THREE.Box3(new THREE.Vector3(...lo),new THREE.Vector3(...hi)).translate(viewer.mesh.position)};
      }
    }
    const selector = region.component || region.selector?.component;
    if (selector && viewer.mesh) {
      const matches = [...viewer.inspectionGroups.values(),...viewer.modelNodes().map(componentInfo)].filter(info => [info.id,info.label].includes(selector));
      const members = matches.length === 1 ? viewer.componentMembers(matches[0].id) : [];
      if (matches.length === 1 && members.length) {
        const box = new THREE.Box3(); members.forEach(node => box.expandByObject(node));
        if (!box.isEmpty()) return {box};
      }
    }
    return {error:`Region ${name} has no resolvable bounds. Choose a unique component ID or label, or edit its box.`};
  }

  function inspectionSelectRegion(name, capture = false) {
    viewer.inspectionRegionName = name || null;
    viewer.inspectionRegionError = null;
    if (!name) {
      viewer.inspectionFrameBox = null;
      if (viewer.currentFrameBox) viewer.frame(viewer.currentFrameBox,null,true);
    } else {
      const resolved = viewer.inspectionRegionBounds(viewer.parts.get(viewer.current),name);
      if (resolved.error) {
        viewer.inspectionRegionError = resolved.error;
        if (capture) viewer.window.__nurb.error = resolved.error;
      } else {
        viewer.inspectionFrameBox = resolved.box.clone(); viewer.frame(resolved.box,null,true);
      }
    }
    viewer.comparePanel(viewer.parts.get(viewer.current));
    return !viewer.inspectionRegionError;
  }

  function compareClear() {
    for (const child of [...viewer.compareMarks.children]) {
      viewer.compareMarks.remove(child);
      child.geometry.dispose(); child.material.dispose();
    }
    viewer.comparePainted = null;
  }

  function compareSamples(entry, metrics, tolerance) {
    if (!viewer.renderer || !viewer.mesh || !metrics || viewer.mesh.userData.token !== entry.token || viewer.compareMap === 'off') { viewer.compareClear(); return; }
    const allDistances = ['part', 'target'].flatMap(side => metrics.samples?.[side]?.distances || []).filter(Number.isFinite);
    const automatic = Math.max(tolerance + 0.001, ...allDistances, tolerance * 2);
    const high = viewer.compareAuto ? automatic : Math.max(tolerance + 0.001, viewer.compareScale);
    const key = [entry.name, entry.token, entry.target.stamp, tolerance, viewer.compareMap, high, viewer.compareThrough].join(':');
    if (viewer.comparePainted === key) { viewer.compareMarks.position.copy(viewer.mesh.position); return; }
    viewer.compareClear();
    const sides = viewer.compareMap === 'both' ? ['part', 'target'] : [viewer.compareMap];
    for (const side of sides) {
      const samples = metrics.samples?.[side];
      if (!samples?.points || !samples?.distances) continue;
      const positions = [], colors = [];
      const low = new THREE.Color(side === 'part' ? 0xf0c274 : 0x62c7ef);
      const severe = new THREE.Color(side === 'part' ? 0xf87171 : 0xc084fc);
      samples.points.forEach((point, index) => {
        const distance = samples.distances[index];
        if (!Array.isArray(point) || point.length !== 3 || !point.every(Number.isFinite)
            || !Number.isFinite(distance) || distance <= tolerance) return;
        positions.push(...point);
        const color = low.clone().lerp(severe, Math.min(1, (distance - tolerance) / (high - tolerance)));
        colors.push(color.r, color.g, color.b);
      });
      if (!positions.length) continue;
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
      geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
      // Small translucent samples show the locations of disagreement without assuming
      // the reference and the CAD viewer share triangle numbering or tessellation.
      const material = new THREE.PointsMaterial({ size: 5, sizeAttenuation: false,
        vertexColors: true, transparent: true, opacity: .8, depthWrite: false,
        depthTest: !viewer.compareThrough, clippingPlanes: [viewer.plane] });
      const points = new THREE.Points(geometry, material);
      points.name = `comparison-${side}`;
      points.renderOrder = 4;
      viewer.compareMarks.add(points);
    }
    viewer.compareMarks.position.copy(viewer.mesh.position);
    viewer.comparePainted = key;
  }

  function comparisonPreview(transform, reason) {
    const entry = viewer.parts.get(viewer.current), ghost = viewer.mesh?.getObjectByName('target');
    if (!entry?.target || !comparisonMatrix(transform) || !ghost || viewer.mesh.userData.token !== entry.token) return;
    if (viewer.comparePending.get(viewer.current)?.error) viewer.comparePending.delete(viewer.current);
    viewer.comparePreview.set(viewer.current, { transform: [...transform], reason });
    ghost.matrixAutoUpdate = false;
    ghost.matrix.set(...transform);
    ghost.matrixWorldNeedsUpdate = true;
    viewer.compareClear();
    viewer.comparePanel(entry);
  }

  function compareAlignment(action) {
    const entry = viewer.parts.get(viewer.current), target = entry?.target;
    if (!Array.isArray(target?.transform)) return;
    const preview = viewer.comparePreview.get(viewer.current);
    let transform = [...(preview?.transform || target.transform)];
    if (action === 'original') {
      viewer.comparisonPreview([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], 'Original coordinates preview. Apply to save it.');
      return;
    }
    if (action === 'center') {
      const reference = viewer.mesh?.getObjectByName('target');
      if (!reference || entry.error || viewer.mesh.userData.token !== entry.token) return;
      viewer.mesh.updateMatrixWorld(true);
      const cad = new THREE.Box3();
      for (const child of [...viewer.modelNodes(), viewer.mesh.getObjectByName('target')].filter(Boolean)) {
        if (child.isMesh && child.name !== 'target' && componentInfo(child).role !== 'context') cad.expandByObject(child, true);
      }
      const referenceBox = new THREE.Box3().setFromObject(reference, true);
      if (cad.isEmpty() || referenceBox.isEmpty()) return;
      const cadCenter = viewer.mesh.worldToLocal(cad.getCenter(new THREE.Vector3())).toArray();
      const referenceCenter = viewer.mesh.worldToLocal(referenceBox.getCenter(new THREE.Vector3())).toArray();
      transform = comparisonCenteredTransform(transform, cadCenter, referenceCenter);
      if (!transform) return;
      viewer.comparisonPreview(transform, 'Centered preview. Apply to save it.');
      return;
    }
    if (action === 'cancel') {
      if (typeof viewer.datumPreview !== 'undefined') { viewer.datumPreview = null; viewer.document.getElementById('datumapply').disabled = true; }
      viewer.comparePreview.delete(viewer.current);
      if (viewer.comparePending.get(viewer.current)?.error) viewer.comparePending.delete(viewer.current);
      const reference = viewer.mesh?.getObjectByName('target');
      if (reference) { reference.matrixAutoUpdate = false; reference.matrix.set(...target.transform); reference.matrixWorldNeedsUpdate = true; }
      viewer.comparePanel(entry);
      return;
    }
    if (action !== 'apply' && action !== 'lock') return;
    if (!viewer.sock || viewer.sock.readyState !== viewer.WebSocket.OPEN) return;
    if (preview) preview.reason = 'Saving alignment. Measuring again…';
    viewer.compareInvalidate('Saving alignment. Measuring again…');
    viewer.comparePending.get(viewer.current).transform = transform;
    viewer.sock.send(JSON.stringify({ type: 'target_settings', name: viewer.current, transform }));
  }

  function comparisonFocus(region) {
    const point = region?.position_mm || region?.point || region?.center || region?.where;
    if (!viewer.mesh || !Array.isArray(point) || point.length !== 3 || !point.every(Number.isFinite)) return;
    if (region.direction === 'part_to_target') viewer.compareMap = 'part';
    if (region.direction === 'target_to_part') viewer.compareMap = 'target';
    const at = new THREE.Vector3(...point).add(viewer.mesh.position);
    const motion = at.clone().sub(viewer.controls.target);
    viewer.controls.target.copy(at);
    viewer.camera.position.add(motion);
    viewer.controls.update();
    viewer.comparePanel(viewer.parts.get(viewer.current));
  }

  function comparisonTransformFields(transform) {
    const rigid = comparisonRigid(transform);
    if (!rigid) return;
    const values = [...rigid.translation, ...rigid.rotation];
    viewer.document.querySelectorAll('#comparetransform input').forEach((input, index) => {
      if (viewer.document.activeElement !== input) input.value = values[index].toFixed(index < 3 ? 3 : 1);
    });
    viewer.document.getElementById('comparesummary').textContent = `Translation ${rigid.translation.map(v => v.toFixed(3)).join(', ')} mm · rotation ${rigid.rotation.map(v => v.toFixed(1)).join(', ')}°`;
  }

  function comparisonToleranceChanged(event) {
    viewer.inspectionToleranceOverride = null;
    const tolerance = Number(event.target.value);
    if (!Number.isFinite(tolerance) || tolerance < 0.001 || !event.target.value.trim()) {
      event.target.setCustomValidity('Enter a tolerance of at least 0.001 mm.');
      event.target.reportValidity(); return;
    }
    event.target.setCustomValidity('');
    if (!viewer.sock || viewer.sock.readyState !== viewer.WebSocket.OPEN || !viewer.current) return;
    viewer.compareInvalidate('Tolerance changed. Measuring again…');
    viewer.comparePending.get(viewer.current).tolerance_mm = tolerance;
    viewer.sock.send(JSON.stringify({ type: 'target_settings', name: viewer.current, tolerance_mm: tolerance }));
  }

  function importCancel() {
    if (viewer.importPreview) {
      const {root, original, owner, source} = viewer.importPreview;
      root.removeFromParent(); referenceDispose(root, false); referenceDispose(source, true);
      if (owner === viewer.mesh) {
        // Alignment previews can move the source root while the saved reference is detached.
        original.matrix.copy(root.matrix); original.matrixAutoUpdate = root.matrixAutoUpdate;
        original.matrixWorldNeedsUpdate = true;
        owner.add(original);
      } else referenceDispose(original, false);
    }
    viewer.importPreview = null;
    viewer.document.getElementById('importselection').hidden = true;
    viewer.document.getElementById('importcomponentlist').replaceChildren();
    viewer.document.getElementById('importstatus').textContent = '';
  }

  function importVisibility() {
    if (!viewer.importPreview) return;
    const keep = new Set([...viewer.document.querySelectorAll('#importcomponentlist input:checked')].map(input => input.value));
    viewer.importPreview.root.traverse(node => { if (node.name.startsWith('component-')) node.visible = keep.has(node.name); });
    viewer.document.getElementById('importapply').disabled = !keep.size;
    viewer.document.getElementById('importstatus').textContent = `Preview: ${keep.size} of ${viewer.importPreview.record.components.length} component groups included. Measurements still use the saved reference until you save.`;
  }

  async function importLanded(msg) {
    if (msg.name !== viewer.current) return;
    if (msg.error) { viewer.document.getElementById('importstatus').textContent = msg.error; return; }
    if (msg.saved) { viewer.importCancel(); viewer.document.getElementById('importstatus').textContent = 'Selection saved. Rebuilding comparison…'; return; }
    if (!msg.glb || msg.stamp !== viewer.parts.get(viewer.current)?.target?.stamp) return;
    const entry = viewer.parts.get(viewer.current), owner = viewer.mesh;
    try {
      const bytes = Uint8Array.from(viewer.atob(msg.glb), char => char.charCodeAt(0));
      const gltf = await viewer.loader.parseAsync(bytes.buffer, '');
      const source = referenceSanitizeScene(gltf.scene);
      if (owner !== viewer.mesh || msg.stamp !== viewer.parts.get(viewer.current)?.target?.stamp) { referenceDispose(source, true); return; }
      viewer.importCancel();
      const original = owner.getObjectByName('target');
      if (!original) { referenceDispose(source, true); throw new Error('Wait for the reference to finish loading.'); }
      const root = viewer.referenceInstance(source, 1);
      root.matrix.copy(original.matrix); root.matrixAutoUpdate = false;
      original.removeFromParent(); owner.add(root);
      viewer.importPreview = {name:viewer.current, token:entry.token, root, original, owner, source, record:msg.import};
      const list = viewer.document.getElementById('importcomponentlist'), excluded = new Set(entry.target.import.excluded_components);
      for (const component of msg.import.components) {
        const label = viewer.document.createElement('label'), input = viewer.document.createElement('input');
        input.type = 'checkbox'; input.value = component.id; input.checked = !excluded.has(component.id);
        input.onchange = viewer.importVisibility;
        label.append(input, viewer.document.createTextNode(`${component.id}: ${component.triangles.toLocaleString()} triangles${component.grouped_fragments ? ' (small fragments grouped)' : ''}`)); list.append(label);
      }
      viewer.document.getElementById('importselection').hidden = false;
      viewer.inspectionSetMode('reference'); viewer.importVisibility();
    } catch (error) { viewer.document.getElementById('importstatus').textContent = `Could not preview source: ${error.message}`; }
  }

  function sectionUpdate() {
    const on = viewer.cutting && !!viewer.mesh;
    viewer.cap.visible = on;
    for (const w of viewer.writers) w.visible = on && viewer.writerSourceVisible(w.userData.source);
    if (!on) { viewer.plane.constant = viewer.PARKED; return; }
    // A freshly attached reference must inherit the plated part transform before bounds are read.
    viewer.mesh.updateMatrixWorld(true);
    // The group has no geometry of its own, and 'context' scenery must not stretch the
    // cut range: the slider spans the printed parts, in world space, joints included.
    const box = new THREE.Box3();
    const reference = viewer.mesh.getObjectByName('target');
    for (const c of [...viewer.modelNodes(), ...referenceMeshes(reference)]) if (componentInfo(c).role !== 'context') box.expandByObject(c);
    if (box.isEmpty()) { viewer.plane.constant = viewer.PARKED; return; }
    const lo = box.min[viewer.cutAxis], hi = box.max[viewer.cutAxis];
    // The cut faces the camera as it stood when the section opened or the axis was
    // picked, so the first look always shows the cross-section instead of the part's
    // own outside. Chosen once and then left alone: a side that follows the camera
    // swaps the surviving half mid-orbit, and with the slider near an end that
    // collapses the whole part to a sliver the moment the view crosses the plane.
    if (!viewer.cutSign) viewer.cutSign = viewer.camera.position[viewer.cutAxis] >= (lo + hi) / 2 ? 1 : -1;
    // Normalized so dragging right always keeps more material, whichever side survives.
    const t = viewer.inspectionMode === 'section' || viewer.cutSign === 1 ? viewer.cutAt : 1 - viewer.cutAt;
    // A hair inside the bounds at each end, so neither extreme is a cut through nothing.
    // An absolute ?cut position gets the same treatment: clamped just inside, so z:0mm
    // cuts the bottom skin instead of grazing the surface and drawing nothing. It sits
    // where it says regardless of cutSign, which only picks the half that survives.
    // x and y positions are quoted in the part's own coordinates, z from the bed, so
    // the centering shift is added back for the first two and already folded into z.
    const mm = viewer.cutMm === null ? null : viewer.inspectionMode === 'section' || viewer.cutAxis !== 'z' ? viewer.cutMm + viewer.mesh.position[viewer.cutAxis] : viewer.cutMm;
    const at = mm !== null && viewer.inspectionMode === 'section' ? mm : mm !== null
      ? Math.min(hi - (hi - lo) * 0.002, Math.max(lo + (hi - lo) * 0.002, mm))
      : lo + (hi - lo) * (0.02 + 0.96 * t);
    const dir = viewer.AXES[viewer.cutAxis];
    viewer.plane.normal.set(-viewer.cutSign * dir[0], -viewer.cutSign * dir[1], -viewer.cutSign * dir[2]);  // keeps the side the normal points at
    viewer.plane.constant = viewer.cutSign * at;
    const span = box.getSize(new THREE.Vector3()).length();
    viewer.cap.scale.set(span, span, 1);
    // Centred on the part, not on coplanarPoint, which is where the plane passes the
    // world origin. A part's own origin is a modeling datum: the shelf lives entirely at
    // x <= 0, so an origin-centred cap left the far 49mm of its cut face bare and spent
    // half its area on empty space.
    viewer.plane.projectPoint(box.getCenter(new THREE.Vector3()), viewer.cap.position);
    viewer.cap.lookAt(viewer.cap.position.x - viewer.plane.normal.x, viewer.cap.position.y - viewer.plane.normal.y,
               viewer.cap.position.z - viewer.plane.normal.z);
    viewer.sectionDrawingKey = null;
    viewer.inspectionSave();
  }

  function symmetryClearPlane() {
    if (viewer.symmetryPlane) { viewer.scene.remove(viewer.symmetryPlane); viewer.symmetryPlane.geometry.dispose(); viewer.symmetryPlane.material.dispose(); viewer.symmetryPlane = null; }
  }

  function symmetryDrawPlane(report, force = false) {
    viewer.symmetryClearPlane();
    if (!viewer.mesh || !report?.plane || (!force && (!viewer.symmetryElement('symmetryshow').checked || !viewer.compareOpen))) return;
    const bounds = new THREE.Box3().setFromObject(viewer.mesh), size = bounds.getSize(new THREE.Vector3()).length();
    const normal = new THREE.Vector3(...report.plane.normal);
    const center = bounds.getCenter(new THREE.Vector3()).sub(viewer.mesh.position);
    center.addScaledVector(normal, report.plane.offset_mm - center.dot(normal)).add(viewer.mesh.position);
    viewer.symmetryPlane = new THREE.Mesh(new THREE.PlaneGeometry(size, size), new THREE.MeshBasicMaterial({color:0xffcb6b,transparent:true,opacity:.17,side:THREE.DoubleSide,depthTest:false,depthWrite:false}));
    viewer.symmetryPlane.name = 'symmetry-evidence-plane';
    viewer.symmetryPlane.quaternion.setFromUnitVectors(new THREE.Vector3(0,0,1),normal);
    viewer.symmetryPlane.position.copy(center); viewer.scene.add(viewer.symmetryPlane);
  }

  function symmetryFresh(entry, report) {
    return report?.status === 'measured' && report.token === entry?.token && !entry.error && !entry.target?.stale
      && !viewer.comparePreview.has(entry.name) && !viewer.comparePending.has(entry.name)
      && (!report.feature_records || JSON.stringify(report.feature_records) === symmetryFeatureSignature(entry))
      && !(entry.name === viewer.current && viewer.featureDirty)
      && JSON.stringify(report.options) === JSON.stringify(viewer.symmetryOptions(entry));
  }

  return {
    syncPins,
    referenceInstance,
    ghostAttach,
    modelNodes,
    inspectionRestore,
    componentAncestors,
    componentMembers,
    componentResolveHidden,
    componentHidden,
    sectionDifference,
    datumReset,
    datumSend,
    datumLanded,
    regionSave,
    evidenceRender,
    featureSectionValue,
    featureSectionStore,
    featureSectionLoad,
    featureRegionValues,
    featureEditorLoad,
    featureVerifiedResult,
    featureEditorState,
    featureExportState,
    featureExport,
    featureExportLanded,
    featurePlot,
    featureEditorInput,
    verificationShow,
    verificationLanded,
    inspectionRegionBounds,
    inspectionSelectRegion,
    compareClear,
    compareSamples,
    comparisonPreview,
    compareAlignment,
    comparisonFocus,
    comparisonTransformFields,
    comparisonToleranceChanged,
    importCancel,
    importVisibility,
    importLanded,
    sectionUpdate,
    symmetryClearPlane,
    symmetryDrawPlane,
    symmetryFresh,
  };
}
