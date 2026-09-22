"""Pure inspection workflow state shared by the real viewer and direct tests."""

import asyncio
import json
import pathlib
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from nurb.server import Server


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE = ROOT / "src/nurb/inspection-state.js"


def run_js(checks):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is needed for inspection workflow checks")
    source = f"import * as state from {json.dumps(MODULE.as_uri())};\nimport assert from 'node:assert/strict';\n{checks}"
    result = subprocess.run([node, "--input-type=module", "-"], input=source, encoding="utf-8", capture_output=True)
    assert result.returncode == 0, result.stderr


def test_multiple_section_series_keep_independent_frames_and_selection():
    run_js("""
let series=state.sectionSeriesInitial([
  {name:'front',origin_mm:[0,0,0],normal:[1,0,0]},
  {name:'side',origin_mm:[1,2,3],normal:[0,1,0]},
]);
series=state.sectionSeriesUpdate(series,{...state.sectionSeriesSelected(series),name:'front edited'});
series=state.sectionSeriesSelect(series,1);
assert.equal(state.sectionSeriesSelected(series).name,'side');
assert.deepEqual(state.sectionSeriesSelected(series).origin_mm,[1,2,3]);
assert.equal(series.items[0].name,'front edited');
series=state.sectionSeriesAdd(series,{name:'curved station',origin_mm:[4,5,6],normal:[0,0,1]});
assert.equal(series.index,2);assert.equal(state.sectionSeriesSelected(series).name,'curved station');
series=state.sectionSeriesRemove(series);assert.equal(series.index,1);assert.equal(series.items.length,2);
series=state.sectionSeriesRemove(series);series=state.sectionSeriesRemove(series);
assert.equal(series.index,0);assert.equal(state.sectionSeriesSelected(series),null);
""")


def test_preview_state_expires_on_edits_and_rebuilds():
    run_js("""
let flow=state.inspectionInitial();
flow=state.inspectionTransition(flow,{type:'select',part:'seal',token:'build-1',interfaceId:'rim',freshness:'current'});
assert.equal(state.inspectionActions(flow).inspect,true);assert.equal(state.inspectionActions(flow).capture,false);
flow=state.inspectionTransition(flow,{type:'preview-ready',token:'other'});
assert.equal(state.inspectionActions(flow).capture,false);
flow=state.inspectionTransition(flow,{type:'preview-ready',token:'build-1'});
assert.equal(state.inspectionActions(flow).capture,true);
flow=state.inspectionTransition(flow,{type:'dirty'});
assert.equal(state.inspectionActions(flow).capture,false);assert.match(state.inspectionGuidance(flow),/Save the interface edits/);
flow=state.inspectionTransition(flow,{type:'rebuild',token:'build-2',freshness:'current'});
assert.equal(flow.sections.status,'stale');assert.equal(state.inspectionActions(flow).capture,false);
""")


def test_verified_state_is_request_scoped_cancellable_and_fails_closed():
    run_js("""
let flow=state.inspectionInitial({part:'seal',token:'build',interfaceId:'rim',source:'verified',freshness:'current'});
flow=state.inspectionTransition(flow,{type:'verify-requested',policy:{accuracy_mm:.02}});
assert.equal(state.inspectionActions(flow).capture,false);assert.equal(state.inspectionActions(flow).cancel,false);
flow=state.inspectionTransition(flow,{type:'verify-progress',requestId:'request-a',status:'running'});
assert.equal(state.inspectionActions(flow).cancel,true);
const before=flow;
flow=state.inspectionTransition(flow,{type:'verify-measured',requestId:'request-b',token:'build'});
assert.equal(flow,before);assert.equal(state.inspectionActions(flow).capture,false);
flow=state.inspectionTransition(flow,{type:'verify-measured',requestId:'request-a',token:'build'});
assert.equal(state.inspectionActions(flow).capture,true);
flow=state.inspectionTransition(flow,{type:'verify-requested',requestId:'request-c'});
assert.equal(state.inspectionActions(flow).capture,false);
flow=state.inspectionTransition(flow,{type:'verify-cancel',requestId:'request-c'});
assert.equal(flow.verification.status,'cancelling');
flow=state.inspectionTransition(flow,{type:'verify-failed',requestId:'request-c',status:'cancelled',error:'Verification cancelled.'});
assert.equal(state.inspectionActions(flow).capture,false);assert.match(state.inspectionGuidance(flow),/No preview was substituted/);
""")


def test_capture_state_records_success_and_failure_without_unlocking_stale_evidence():
    run_js("""
let flow=state.inspectionInitial({part:'rail',token:'build',freshness:'current'});
assert.equal(state.inspectionActions(flow).capture,true);
flow=state.inspectionTransition(flow,{type:'capture-requested'});assert.equal(state.inspectionActions(flow).capture,false);
flow=state.inspectionTransition(flow,{type:'capture-complete',setupId:'setup-1'});
assert.equal(flow.setupId,'setup-1');assert.equal(flow.capture.status,'captured');
flow=state.inspectionTransition(flow,{type:'capture-failed',error:'Reference changed.'});
assert.equal(flow.capture.status,'failed');assert.equal(flow.capture.error,'Reference changed.');
flow=state.inspectionTransition(flow,{type:'rebuild',token:'new',freshness:'stale'});
assert.equal(state.inspectionActions(flow).capture,false);
""")


def test_module_is_shipped_and_served_offline(tmp_path):
    package = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"src/nurb/inspection-state.js"' in package
    server = Server(tmp_path, port=7373, draft=False)
    response = asyncio.run(server.http(None, SimpleNamespace(path="/inspection-state.js")))
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "text/javascript; charset=utf-8"
    assert b"export function inspectionTransition" in response.body
