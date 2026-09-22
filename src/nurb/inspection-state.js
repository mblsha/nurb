const VERIFY_ACTIVE = new Set(['queued', 'running', 'cancelling']);

export function policyFromProvenance(provenance) {
  if (!provenance) return null;
  return {
    accuracy_mm: provenance.absolute_deflection_mm ?? null,
    timeout_s: provenance.timeout_s ?? 30,
    max_triangles: provenance.max_triangles ?? 1000000,
    feature_size_mm: provenance.feature_size_mm ?? null,
    memory_limit_mb: provenance.memory?.limit_mb ?? 2048,
  };
}

export function inspectionInitial(fields = {}) {
  return {
    part: fields.part ?? null,
    token: fields.token ?? null,
    interfaceId: fields.interfaceId ?? null,
    setupId: fields.setupId ?? null,
    source: fields.source === 'verified' ? 'verified' : 'preview',
    freshness: fields.freshness ?? 'unknown',
    dirty: !!fields.dirty,
    sections: {status: 'idle', source: null, requestId: null, error: null, ...(fields.sections || {})},
    verification: {status: 'idle', requestId: null, policy: null, error: null, ...(fields.verification || {})},
    capture: {status: 'idle', error: null, ...(fields.capture || {})},
  };
}

function sameRequest(state, event) {
  return !state.verification.requestId || !event.requestId || state.verification.requestId === event.requestId;
}

export function inspectionTransition(state, event) {
  const next = structuredClone(state);
  switch (event.type) {
    case 'select':
      return inspectionInitial({part:event.part ?? state.part,token:event.token ?? state.token,interfaceId:event.interfaceId ?? null,
        setupId:event.setupId ?? null,source:event.source ?? state.source,freshness:event.freshness ?? 'unknown'});
    case 'source':
      next.source = event.source === 'verified' ? 'verified' : 'preview';
      next.sections = {status:'idle',source:null,requestId:null,error:null};
      return next;
    case 'dirty':
      next.dirty = true; next.sections = {status:'stale',source:next.sections.source,requestId:next.sections.requestId,error:null};
      return next;
    case 'saved':
      next.dirty = false; next.setupId = event.setupId ?? next.setupId; next.freshness = event.freshness ?? 'current';
      return next;
    case 'preview-ready':
      if (event.token !== undefined && next.token !== event.token) return state;
      next.sections = {status:'ready',source:'preview',requestId:null,error:null};
      return next;
    case 'verify-requested':
      next.verification = {status:'queued',requestId:event.requestId ?? null,policy:event.policy ?? null,error:null};
      if (next.sections.source === 'verified') next.sections = {status:'stale',source:'verified',requestId:next.sections.requestId,error:null};
      return next;
    case 'verify-progress':
      if (!sameRequest(next,event)) return state;
      next.verification.status = event.status ?? 'running';
      next.verification.requestId = event.requestId ?? next.verification.requestId;
      next.verification.policy = event.policy ?? next.verification.policy;
      next.verification.error = null;
      return next;
    case 'verify-measured':
      if (!sameRequest(next,event) || event.token !== next.token) return state;
      next.verification = {status:'measured',requestId:event.requestId ?? next.verification.requestId,policy:event.policy ?? next.verification.policy,error:null};
      next.sections = {status:'ready',source:'verified',requestId:next.verification.requestId,error:null};
      return next;
    case 'verify-failed':
      if (!sameRequest(next,event)) return state;
      next.verification = {...next.verification,status:event.status ?? 'unknown',error:event.error || 'Verification produced no result.'};
      next.sections = {status:next.sections.source === 'verified' ? 'unavailable' : next.sections.status,source:next.sections.source,requestId:next.sections.requestId,error:next.verification.error};
      return next;
    case 'verify-cancel':
      if (!sameRequest(next,event)) return state;
      next.verification.status = 'cancelling';
      return next;
    case 'rebuild':
      next.token = event.token ?? null; next.freshness = event.freshness ?? 'unknown'; next.dirty = false;
      next.sections = {status:'stale',source:next.sections.source,requestId:next.sections.requestId,error:null};
      next.verification = {status:'stale',requestId:null,policy:next.verification.policy,error:null};
      next.capture = {status:'idle',error:null};
      return next;
    case 'capture-requested':
      next.capture = {status:'capturing',error:null};
      return next;
    case 'capture-complete':
      next.capture = {status:'captured',error:null}; next.setupId=event.setupId ?? next.setupId;
      return next;
    case 'capture-failed':
      next.capture = {status:'failed',error:event.error || 'Capture produced no bundle.'};
      return next;
    default:
      return state;
  }
}

export function inspectionActions(state) {
  const selected = !!state.interfaceId;
  const active = VERIFY_ACTIVE.has(state.verification.status);
  const previewReady = state.sections.status === 'ready' && state.sections.source === 'preview';
  const verifiedReady = state.sections.status === 'ready' && state.sections.source === 'verified'
    && state.sections.requestId === state.verification.requestId && state.verification.status === 'measured';
  const sectionsReady = state.source === 'verified' ? verifiedReady : previewReady;
  const current = state.freshness !== 'stale' && !state.dirty;
  return {
    inspect: selected && current && !active,
    verify: selected && current && !active,
    cancel: active && !!state.verification.requestId,
    save: current && !active,
    capture: current && (!selected || sectionsReady) && !active && state.capture.status !== 'capturing',
    sectionsReady,
  };
}

export function inspectionGuidance(state) {
  const actions = inspectionActions(state);
  if (!state.interfaceId) return '1. Select a named interface or restore a saved setup.';
  if (state.dirty) return 'Save the interface edits before inspecting evidence.';
  if (state.freshness === 'stale') return 'This setup is stale. Restore or save it against the current model.';
  if (VERIFY_ACTIVE.has(state.verification.status)) return state.verification.status === 'cancelling' ? 'Stopping verification…' : 'Verification is running. You can cancel it.';
  if (state.verification.error && state.source === 'verified') return `${state.verification.error} No preview was substituted.`;
  if (!actions.sectionsReady) return state.source === 'verified' ? '2. Inspect a preview if useful, then run Verify to produce bounded section evidence.' : '2. Inspect sections to review the selected interface.';
  if (state.source === 'preview') return '3. Preview sections are ready. Verify for bounded evidence, or save this preview setup.';
  if (state.capture.status === 'captured') return '4. Evidence bundle captured. The verified request and policy are recorded.';
  return '4. Verified sections are ready. Save the setup or capture an evidence bundle.';
}

export function sectionSeriesInitial(sections = []) {
  return {index:0,items:structuredClone(sections)};
}

export function sectionSeriesSelected(state) {
  return state.items[state.index] || null;
}

export function sectionSeriesUpdate(state, value) {
  const next=structuredClone(state);
  if (next.items.length) next.items[next.index]=structuredClone(value);
  else { next.items=[structuredClone(value)]; next.index=0; }
  return next;
}

export function sectionSeriesSelect(state, index) {
  const next=structuredClone(state);
  next.index=Math.max(0,Math.min(Number(index) || 0,Math.max(0,next.items.length-1)));
  return next;
}

export function sectionSeriesAdd(state, value) {
  const next=structuredClone(state); next.items.push(structuredClone(value)); next.index=next.items.length-1; return next;
}

export function sectionSeriesRemove(state) {
  const next=structuredClone(state);
  if (next.items.length) next.items.splice(next.index,1);
  next.index=Math.min(next.index,Math.max(0,next.items.length-1));
  return next;
}
