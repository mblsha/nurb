"""Geometry and display contracts for assembly inspection and reference sections."""

import json
import pathlib
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
VIEWER = (ROOT / "src/nurb/viewer.html").read_text()
THREE = (ROOT / "src/nurb/vendor/three/build/three.module.min.js").as_uri()
INSPECTION_STATE = (ROOT / "src/nurb/inspection-state.js").as_uri()


def js(functions, checks):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is needed for viewer geometry checks")
    source = f"""import * as THREE from {json.dumps(THREE)};
import assert from 'node:assert/strict';
import * as InspectionState from {json.dumps(INSPECTION_STATE)};
const {{inspectionActions,inspectionGuidance,inspectionInitial,inspectionTransition,policyFromProvenance,
  sectionSeriesAdd,sectionSeriesInitial,sectionSeriesRemove,sectionSeriesSelect,sectionSeriesSelected,sectionSeriesUpdate}}=InspectionState;
"""
    for start, end in functions:
        source += start + VIEWER.split(start, 1)[1].split(end, 1)[0] + "\n"
    result = subprocess.run([node, "--input-type=module", "-"], input=source + checks,
                            encoding="utf-8", capture_output=True)
    assert result.returncode == 0, result.stderr


def test_section_closes_solids_in_part_coordinates_and_preserves_nested_holes():
    js([("function sectionSegments(", "function sectionDifference(")], """
const identity = new THREE.Matrix4();
const square = new THREE.Shape(); square.moveTo(-5,-5); square.lineTo(5,-5);
square.lineTo(5,5); square.lineTo(-5,5); square.closePath();
const hole = new THREE.Path(); hole.moveTo(-2,-2); hole.lineTo(-2,2);
hole.lineTo(2,2); hole.lineTo(2,-2); hole.closePath(); square.holes.push(hole);
const geometry = new THREE.ExtrudeGeometry(square, {depth:4, bevelEnabled:false});
const section = sectionContours(geometry, identity, 2, 1.37);
assert.equal(section.valid, true); assert.equal(section.loops.length, 2);
const shifted = sectionContours(geometry, new THREE.Matrix4().makeTranslation(10,20,30), 2, 31.37);
assert.equal(shifted.valid, true); assert.equal(shifted.loops.length, 2);
const positions = shifted.loops.flat();
assert.equal(Math.min(...positions.map(p => p[0])), 5);
assert.equal(Math.max(...positions.map(p => p[1])), 25);
const islands = new THREE.BoxGeometry(2,2,2);
assert.equal(sectionContours(islands, identity, 0, .317).valid, true);
assert.equal(sectionContours(islands, identity, 1, .317).valid, true);
""")


def test_section_refuses_open_meshes_and_coplanar_boundaries_instead_of_filling_them():
    js([("function sectionSegments(", "function sectionDifference(")], """
const geometry = new THREE.BufferGeometry();
geometry.setAttribute('position', new THREE.Float32BufferAttribute([-1,0,-1, 1,0,1, 0,2,-1],3));
const open = sectionContours(geometry, new THREE.Matrix4(), 2, 0);
assert.equal(open.valid,false); assert.match(open.reason,/open/);
assert.equal(open.segments.length,1);
const grazing = sectionContours(new THREE.BoxGeometry(2,2,2), new THREE.Matrix4(), 2, 1);
assert.equal(grazing.valid,false); assert.match(grazing.reason,/coplanar/);
const empty = sectionContours(new THREE.BoxGeometry(2,2,2), new THREE.Matrix4(), 2, 4);
assert.equal(empty.valid,true); assert.equal(empty.segments.length,0);
""")


def test_component_identity_is_preserved_and_visibility_does_not_change_geometry():
    js([("function componentInfo(", "function inspectionSave(")], """
const mesh = new THREE.Group();
const a = new THREE.Mesh(new THREE.BoxGeometry()), b = new THREE.Mesh(new THREE.BoxGeometry());
a.name = 'fixed-a'; a.userData.nurb = {id:'socket-1',label:'Left socket',role:'part'};
b.name = 'context'; mesh.add(a,b);
assert.deepEqual(componentInfo(a), {id:'socket-1',label:'Left socket',role:'part'});
assert.equal(componentInfo(b).role,'context');
const before = a.geometry.attributes.position.array.slice(); a.visible = false;
assert.equal(modelNodes().length,2); assert.deepEqual(a.geometry.attributes.position.array,before);
""")


def test_datum_preview_with_no_written_files_stays_a_preview_until_apply():
    js([("function datumLanded(", "function regionValues(")], """
const current = 'part', parts = new Map([['part',{token:'a'}]]);
let datumPreview = {name:'part',token:'a',operation:{kind:'axis'}}, previewed = null, mode = null;
const fields = {datumstatus:{textContent:''},datumapply:{disabled:true}};
const document = {getElementById: id => fields[id]};
function comparisonPreview(transform) { previewed=transform; }
function inspectionSetMode(value) { mode=value; }
const transform = [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
datumLanded({name:'part',token:'a',written:[],transform,max_residual_mm:.01});
assert.deepEqual(previewed,transform); assert.deepEqual(datumPreview.transform,transform); assert.equal(mode,'overlay');
assert.ok(datumPreview); assert.equal(fields.datumapply.disabled,false);
assert.match(fields.datumstatus.textContent,/Preview ready/);
datumLanded({name:'part',token:'a',written:['parts/part.md']});
assert.equal(datumPreview,null); assert.equal(fields.datumapply.disabled,true);
assert.equal(fields.datumstatus.textContent,'Alignment saved.');
""")


def test_section_tolerance_mutes_small_offsets_but_preserves_larger_material_defects():
    js([("function sectionMaskDistance(", "function sectionDifference(")], """
const width=12,height=5,cad=new Uint8Array(width*height),ref=new Uint8Array(width*height);
for(let y=1;y<4;y++) for(let x=2;x<7;x++) {cad[y*width+x]=1;ref[y*width+x+1]=1;}
cad[2*width+10]=1;
const exact=sectionMaskClassify(cad,ref,width,height,0);
assert.equal(exact[2*width+2],2); assert.equal(exact[2*width+7],3);
const accepted=sectionMaskClassify(cad,ref,width,height,1);
assert.equal(accepted[2*width+2],1); assert.equal(accepted[2*width+7],1);
assert.equal(accepted[2*width+10],2);
assert.equal(accepted[0],0);
const mask=new Uint8Array(25);mask[12]=1;
const distance=sectionMaskDistance(mask,5,5);
assert.equal(distance[0],8);assert.equal(distance[12],0);assert.equal(distance[13],1);
""")


def test_nested_component_groups_select_and_hide_descendant_leaves():
    js([("function componentInfo(", "function inspectionSave("),
        ("function componentAncestors(", "function componentPanel(")], """
const mesh = new THREE.Group(), inspectionGroups = new Map([
  ['mount',{id:'mount',parent:null}],['mount/insert',{id:'mount/insert',parent:'mount'}]]);
const a = new THREE.Mesh(), b = new THREE.Mesh();
a.userData.nurb={id:'mount/insert/body',label:'body',role:'part',parent:'mount/insert'};
b.userData.nurb={id:'mount/wall',label:'wall',role:'context',parent:'mount'};mesh.add(a,b);
const hiddenComponents = new Set(['mount']);
assert.deepEqual(componentAncestors(a),['mount/insert','mount']);
assert.deepEqual(componentMembers('mount'),[a,b]);
assert.deepEqual(componentMembers('mount/insert'),[a]);
assert.equal(componentHidden(a),true);assert.equal(componentHidden(b),true);
hiddenComponents.clear();hiddenComponents.add('mount/insert');
assert.equal(componentHidden(a),true);assert.equal(componentHidden(b),false);
""")


def test_embedded_section_scale_reserves_space_above_the_viewer_footer():
    section = VIEWER.split("function sectionDifference()", 1)[1].split("function referenceInspectionLanded", 1)[0]
    assert "mm / 100 px`, 16, height - (embed ? 44 : 18)" in section


def test_datum_apply_rejects_other_parts_and_new_builds_and_uses_the_captured_token():
    js([("function datumReset(", "function inspectionVector("),
        ("function datumSend(", "document.getElementById('datumkind').onchange")], """
let current='A',datumPreview=null;
const parts=new Map([['A',{token:'a1'}],['B',{token:'b1'}]]),comparePreview=new Map(),sent=[];
const fields={datumstatus:{textContent:''},datumapply:{disabled:false}};
const document={getElementById:id=>fields[id]},WebSocket={OPEN:1};
const sock={readyState:1,send:value=>sent.push(JSON.parse(value))};
function datumOperation(){return {kind:'axis',origin_mm:[0,0,0]};}
datumSend(false);
assert.equal(sent[0].name,'A');assert.equal(sent[0].token,'a1');
assert.equal(datumPreview.transform,null);
datumSend(true);assert.equal(sent.length,1);
datumPreview.transform=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
current='B';datumSend(true);
assert.equal(sent.length,1);assert.equal(datumPreview,null);assert.equal(fields.datumapply.disabled,true);
assert.match(fields.datumstatus.textContent,/selected part or build changed/);
current='A';datumSend(false);datumPreview.transform=[1];parts.set('A',{token:'a2'});datumSend(true);
assert.equal(sent.length,2);assert.equal(datumPreview,null);
datumSend(false);datumPreview.transform=[1];datumSend(true);
assert.equal(sent.length,4);assert.equal(sent[3].token,'a2');assert.equal(sent[3].name,'A');assert.equal(sent[3].save,true);
""")


def test_switching_parts_resets_the_pending_datum_preview():
    js([("function inspectionRestore(", "function componentAncestors("),
        ("function datumReset(", "function inspectionVector(")], """
let inspectionToleranceOverride=.3;
let inspectionFor='A',inspectionFrameBox={},hiddenComponents=new Set(),inspectionRegionName='old',inspectionRegionError='old',sectionDrawingKey='old';
let datumPreview={name:'A',token:'a1',operation:{},transform:[1]};
const comparePreview=new Map([['A',{transform:[1]}]]),q=new URLSearchParams(),view='top';
const fields={datumapply:{disabled:false},datumstatus:{textContent:'Preview ready'}};
const document={getElementById:id=>fields[id]};
inspectionRestore('B');
assert.equal(datumPreview,null);assert.equal(fields.datumapply.disabled,true);assert.equal(fields.datumstatus.textContent,'');
assert.equal(comparePreview.has('A'),false);assert.equal(inspectionFor,'B');assert.equal(inspectionToleranceOverride,null);
""")


def test_subpixel_section_tolerance_does_not_accept_a_full_pixel_gap():
    js([("function sectionMaskDistance(", "function sectionDifference(")], """
const cad=Uint8Array.from([1,0,0]),reference=Uint8Array.from([0,1,0]);
for(const tolerance of [.00001,.2,.999]) assert.deepEqual(Array.from(sectionMaskClassify(cad,reference,3,1,tolerance)),[2,3,0]);
assert.deepEqual(Array.from(sectionMaskClassify(cad,reference,3,1,1)),[1,1,0]);
""")


def test_initial_hide_resolves_ids_and_repeated_labels_for_leaves_and_groups():
    js([("function componentInfo(", "function inspectionSave("),
        ("function componentAncestors(", "function componentPanel(")], """
const mesh=new THREE.Group(),inspectionGroups=new Map([
 ['a',{id:'a',label:'Assembly'}],['b',{id:'b',label:'Assembly'}]]);
const leaf=(id,label,parent)=>{const n=new THREE.Mesh(new THREE.BoxGeometry());n.userData.nurb={id,label,parent,role:'part'};mesh.add(n);return n;};
const one=leaf('a/socket','Socket','a'),two=leaf('b/socket','Socket','b'),other=leaf('body','Body',null);
let hiddenComponents=new Set(['Socket']);componentResolveHidden();
assert.deepEqual([...hiddenComponents],['a/socket','b/socket']);
assert.equal(componentHidden(one),true);assert.equal(componentHidden(two),true);assert.equal(componentHidden(other),false);
hiddenComponents=new Set(['Assembly']);componentResolveHidden();
assert.deepEqual([...hiddenComponents],['a','b']);assert.equal(componentHidden(one),true);assert.equal(componentHidden(two),true);
hiddenComponents=new Set(['body']);componentResolveHidden();assert.equal(componentHidden(other),true);
assert.deepEqual(componentMembers('Socket'),[one,two]);
""")


def test_region_labels_resolve_and_unresolved_capture_fails_instead_of_using_whole_model():
    js([("function componentInfo(", "function inspectionSave("),
        ("function componentAncestors(", "function componentPanel("),
        ("function inspectionSelectedMetrics(", "function inspectionRegions(")], """
const current='assembly',mesh=new THREE.Group(),inspectionGroups=new Map();
const board=new THREE.Mesh(new THREE.BoxGeometry(4,6,2));board.userData.nurb={id:'board_1',label:'Board',role:'part'};mesh.add(board);
const entry={target:{regions:[{name:'board fit',component:'Board'},{name:'missing',component:'Missing'}]}};
const parts=new Map([[current,entry]]),window={__nurb:{error:null}};
let inspectionRegionName=null,inspectionRegionError=null,inspectionFrameBox=null,currentFrameBox=null,framed=[];
function frame(box){framed.push(box);}function comparePanel(){}function comparisonAbove(){return false;}
assert.equal(inspectionSelectRegion('board fit',true),true);assert.equal(framed.length,1);
assert.deepEqual(framed[0].getSize(new THREE.Vector3()).toArray(),[4,6,2]);
assert.equal(inspectionSelectRegion('missing',true),false);assert.equal(framed.length,1);
assert.match(window.__nurb.error,/no resolvable bounds/);
entry.target.metrics={inspection_regions:[{name:'board fit',status:'partial',bounds_mm:{min:[1,2,3],max:[5,8,9]}}]};
assert.deepEqual(inspectionRegionBounds(entry,'board fit').box.min.toArray(),[1,2,3]);
""")


def test_selected_regions_show_local_stats_status_and_worst_regions_until_whole_model():
    js([("function inspectionSelectedMetrics(", "function inspectionRegionBounds("),
        ("function comparisonRows(", "function compareClear("),
        ("function comparisonWorst(", "function comparisonFocus(")], """
const regional={name:'seat',status:'partial',part:{sampled_max:.6,p95:.4,within_tolerance:.5},target:null,
 detected_above_tolerance:true,worst_regions:[{position_mm:[1,2,3],peak_deviation_mm:.6}]};
const global={part:{sampled_max:8},target:{sampled_max:9},inspection_regions:[regional]};
const selected=inspectionSelectedMetrics(global,'seat');
assert.equal(selected,regional);assert.equal(comparisonRows(selected)[0][1],'0.600');assert.equal(comparisonRows(selected)[1][1],'n/a');
assert.equal(comparisonWorst(selected),regional.worst_regions);
assert.match(inspectionRegionStatus(selected,.2),/Partial region: no reference samples/);
assert.match(inspectionRegionStatus(selected,.2),/Deviation above tolerance/);
assert.equal(inspectionSelectedMetrics(global,null),global);
assert.equal(inspectionSelectedMetrics(global,'missing'),null);
""")


def test_region_crud_waits_for_previous_write_instead_of_replacing_unseen_regions():
    js([("function regionSave(", "document.getElementById('regionsave').onclick")], """
const current='part',regionWrites=new Map(),sent=[],WebSocket={OPEN:1};
const parts=new Map([[current,{target:{regions:[]}}]]);
const fields={regionstatus:{textContent:''},regionname:{value:'A'},regionexisting:{value:''},regionsave:{disabled:false},regionremove:{disabled:false}};
const document={getElementById:id=>fields[id]},sock={readyState:1,send:message=>sent.push(JSON.parse(message))};
function regionValues(){return {name:fields.regionname.value,component:'body'};}
regionSave();assert.deepEqual(sent[0].regions,[{name:'A',component:'body'}]);assert.equal(fields.regionsave.disabled,true);
fields.regionname.value='B';regionSave();assert.equal(sent.length,1);assert.match(fields.regionstatus.textContent,/previous region change/);
fields.regionexisting.value='A';regionSave(true);assert.equal(sent.length,1);
parts.set(current,{target:{regions:sent[0].regions}});regionWrites.delete(current);
fields.regionexisting.value='';regionSave();assert.deepEqual(sent[1].regions,[{name:'A',component:'body'},{name:'B',component:'body'}]);
fields.regionexisting.value='B';fields.regionname.value='edited';regionSave();assert.equal(sent.length,2);
""")


def test_feature_sections_expire_on_rebuild_even_before_new_metrics_arrive():
    js([("function featureEditorState(", "function featureInspect(")], """
function featureExportState() {} function featureVerifiedResult(){return null;}
const fields={freshness:{},inspect:{},verified:{},review:{},plot:{setAttribute(){this.hidden=true}},station:{},status:{}};
const featureField=id=>fields[id], current='part';
const selectedFeatureRegion=()=>({feature:{id:'rim'}});
let evidenceWorkflow=inspectionInitial({part:'part',token:'old-build',interfaceId:'rim'});function evidenceRender(){}
let featureInspection={name:'part',token:'old-build'}, featureDirty=false;
featureEditorState({token:'new-build',target:{feature_evidence:[{id:'rim',status:'stale'}]}});
assert.equal(featureInspection,null);
assert.equal(evidenceWorkflow.sections.status,'stale');
assert.equal(fields.plot.hidden,true);
assert.equal(fields.review.disabled,true);
assert.match(fields.freshness.textContent,/stale/);
assert.match(fields.status.textContent,/expired/);
""")


def test_feature_rename_preserves_identity_and_other_saved_series():
    js([("function featureSectionValue(", "function featureEditorLoad(")], """
const previous={id:'stable-rim',feature_size_mm:.3,sections:[{name:'first'},{name:'second'}],review:{identity:{token:'old'}}};
const selectedFeatureRegion=()=>({feature:previous});
let featureSeries=sectionSeriesInitial(previous.sections);
const fields={enabled:{checked:true},point:{value:''},center:{value:'-3 0 0'},symmetrygroup:{value:'cushion holes'},size:{value:'0.3'},uncertainty:{value:''},sectionenabled:{checked:true},
 offsets:{value:'-1 0 1'},sectionname:{value:'Updated station'},origin:{value:'0 0 0'},normal:{value:'0 0 1'},
 x:{value:'1 0 0'},tolerance:{value:'0'},expected:{value:'small T'}};
for (const key of ['role','configuration','orientation','notes','required','excluded','links']) fields[key]={value:''};
const featureField=id=>fields[id], inspectionVector=text=>text.split(' ').map(Number);
const result=featureRegionValues({name:'Corrected headset rim',component:'body'});
assert.equal(result.feature.id,'stable-rim');
assert.deepEqual(result.feature.center_mm,[-3,0,0]);
assert.equal(result.feature.symmetry_group,'cushion holes');
assert.equal(result.feature.feature_size_mm,.3);
assert.equal(result.feature.sections[1].name,'second');
assert.deepEqual(result.feature.sections[0].offsets_mm,[-1,0,1]);
assert.equal(result.feature.review.identity.token,'old');
""")


def test_precise_verification_rejects_late_build_results():
    js([("function verificationLanded(", "document.getElementById('verifycancel').onclick")], """
const entry={token:'new',target:{verification:{status:'running'}}};
const parts=new Map([['part',entry]]),current='part'; let renders=0,editorUpdates=0;
const verificationShow=()=>renders++,featureEditorState=()=>editorUpdates++;
const evidenceMove=()=>{},featureVerifiedResult=()=>null;
verificationLanded({name:'part',token:'old',status:'measured',metrics:{part:{max:0}}});
assert.equal(entry.target.verification.status,'running'); assert.equal(renders,0);
verificationLanded({name:'part',token:'new',status:'unknown',error:'deadline'});
assert.equal(entry.target.verification.status,'unknown'); assert.equal(renders,1);assert.equal(editorUpdates,1);
assert.equal(entry.target.verification.metrics,undefined);
""")


def test_saved_symmetry_plane_ignores_live_panel_and_checkbox_visibility():
    js([('function symmetryClearPlane(', 'function symmetryFeatureSignature(')], """
const scene=new THREE.Scene(),mesh=new THREE.Mesh(new THREE.BoxGeometry(5,5,5));
let symmetryPlane=null,compareOpen=false;
const symmetryElement=()=>({checked:false});
const report={plane:{normal:[1,0,0],offset_mm:0}};
symmetryDrawPlane(report);assert.equal(symmetryPlane,null);
symmetryDrawPlane(report,true);assert.ok(symmetryPlane);assert.ok(scene.children.includes(symmetryPlane));
""")


def test_precise_verification_shows_stale_without_reusing_old_metrics():
    js([("function verificationShow(", "function verificationLanded(")], """
const current='part', verificationHistory=new Map([['part',{token:'old',status:'measured',metrics:{part:{sampled_max:0}}}]]);
const fields={verifyresult:{replaceChildren(){this.cleared=true}},verifyrun:{},verifycancel:{},verifystatus:{}};
const document={getElementById:id=>fields[id]};
function evidenceRender(){fields.verifycancel.disabled=true;}
verificationShow({name:'part',token:'new',target:{}});
assert.equal(fields.verifyresult.cleared,true);
assert.match(fields.verifystatus.textContent,/stale/);
assert.equal(fields.verifycancel.disabled,true);
""")


def test_replaced_reference_is_in_part_frame_before_a_section_is_restored():
    js([('function sectionUpdate()', '// ---- download ----')], """
const mesh=new THREE.Group();mesh.position.z=4;
const cad=new THREE.Mesh(new THREE.BoxGeometry(40,24,8));mesh.add(cad);mesh.updateMatrixWorld(true);
const reference=new THREE.Group();reference.name='target';
const ref=new THREE.Mesh(new THREE.BoxGeometry(40,24,8));reference.add(ref);mesh.add(reference);
const cutting=true,writers=[],cap=new THREE.Object3D(),plane=new THREE.Plane(),PARKED=1e10;
let cutSign=1,cutAxis='z',cutAt=.5,cutMm=null,sectionDrawingKey=null;
const inspectionMode='section',camera={position:new THREE.Vector3(20,-20,30)},AXES={z:[0,0,1]};
function modelNodes(){return [cad];}function referenceMeshes(){return [ref];}
function componentInfo(){return {role:'part'};}function inspectionSave(){}
sectionUpdate();
assert.equal(plane.constant/cutSign-mesh.position.z,0);
assert.equal(ref.matrixWorld.elements[14],4);
cutMm=1.25;sectionUpdate();assert.equal(plane.constant/cutSign-mesh.position.z,1.25);
""")


def test_feature_scale_survives_editor_load_save_and_can_be_cleared_explicitly():
    js([("function featureSectionValue(", "function featureEditorState(")], """
const previous={id:'stable-rim',feature_size_mm:.3,role:'small headset lip'};
const selectedFeatureRegion=()=>({feature:previous}), current='part', parts=new Map();
const Option=(text,value)=>({text,value}),fields=new Map(), featureField=id=>{if(!fields.has(id))fields.set(id,{value:'',checked:false,setAttribute(){},replaceChildren(){}});return fields.get(id);};
const document={getElementById:id=>id==='evidenceinterface'?{options:[],value:''}:null};
const inspectionField=()=>({value:'preview'});let evidenceWorkflow=inspectionInitial();function evidenceRender(){}
let featureDirty=false,featureInspection=null,featureSeries=sectionSeriesInitial();
function featureEditorState(){} function inspectionVector(text){return text.split(' ').map(Number);}
featureEditorLoad({feature:previous});
assert.equal(featureField('size').value,.3);
featureField('size').value=String(featureField('size').value);
assert.equal(featureRegionValues({name:'rim'}).feature.feature_size_mm,.3);
featureField('size').value='.1';
assert.equal(featureRegionValues({name:'rim'}).feature.feature_size_mm,.1);
featureField('size').value='';
assert.equal('feature_size_mm' in featureRegionValues({name:'rim'}).feature,false);
featureEditorLoad({feature:{id:'old',feature_scale_mm:.4}});
assert.equal(featureField('size').value,.4);
""")


def test_feature_exports_use_shared_download_and_discard_scale_edits_or_stale_results():
    js([("function featureExportState(", "function featurePlot(")], """
const entry={token:'build',target:{}},parts=new Map([['part',entry]]),current='part',sent=[],downloads=[];
const WebSocket={OPEN:1},sock={readyState:1,send:value=>sent.push(JSON.parse(value))};
const fields={json:{},svg:{},status:{},station:{value:'0'}},featureField=id=>fields[id];
let featureDirty=false,featureInspection={token:'build',result:{id:'rim',identity:{token:'feature-.3'},cad:[{}]}};
function featureEditorState(){featureExportState();}
const window={downloadArtifact:async artifact=>{downloads.push(artifact);return {path:'/output/test.svg'};}};
featureExport('svg'); assert.equal(sent[0].identity.token,'feature-.3');assert.equal(sent[0].station,0);
const result={name:'part',token:'build',identity:{token:'feature-.3'},artifact:{filename:'rim.svg',mime:'image/svg+xml',data:'<svg/>'}};
await featureExportLanded({...result,identity:{token:'feature-.1'}}); assert.equal(downloads.length,0);
featureDirty=true; await featureExportLanded(result); featureExportState();
assert.equal(downloads.length,0);assert.equal(fields.svg.disabled,true);
featureDirty=false;entry.target.stale=true;await featureExportLanded(result);assert.equal(downloads.length,0);
entry.target.stale=false;await featureExportLanded(result);assert.equal(downloads.length,1);
assert.match(fields.status.textContent,/saved/);
entry.token='rebuilt';await featureExportLanded(result);assert.equal(downloads.length,1);
""")


def test_feature_scale_identity_change_expires_contours_even_with_same_build_token():
    js([("function featureEditorState(", "function featureInspect(")], """
const fields={freshness:{},inspect:{},verified:{},review:{},plot:{setAttribute(){this.hidden=true}},station:{},status:{}};
const featureField=id=>fields[id], current='part';
function featureExportState(){} function featureVerifiedResult(){return null;}
const selectedFeatureRegion=()=>({feature:{id:'rim',feature_size_mm:.1}});
let evidenceWorkflow=inspectionInitial({part:'part',token:'same-build',interfaceId:'rim'});function evidenceRender(){}
let featureInspection={name:'part',token:'same-build',result:{identity:{token:'old-scale'}}},featureDirty=false;
featureEditorState({token:'same-build',target:{feature_evidence:[{id:'rim',status:'stale',identity:{token:'new-scale'}}]}});
assert.equal(featureInspection,null);assert.equal(fields.review.disabled,true);
assert.equal(fields.plot.hidden,true);assert.match(fields.status.textContent,/expired/);
""")


def test_editing_feature_scale_immediately_replaces_the_current_review_label():
    start = "document.getElementById('regioneditor').addEventListener('input'"
    handler = start + VIEWER.split(start, 1)[1].split("featureField('inspect').onclick", 1)[0]
    js([("function featureEditorState(", "function featureInspect("),
        ("function featureExportState(", "function featureExport(")], """
const current='part',entry={token:'build',target:{feature_evidence:[{id:'rim',status:'current'}]}},parts=new Map([[current,entry]]);
const fields={freshness:{},inspect:{},verified:{},review:{},plot:{setAttribute(){this.hidden=true}},station:{},status:{},json:{},svg:{}};
const featureField=id=>fields[id],selectedFeatureRegion=()=>({feature:{id:'rim',feature_size_mm:.3}});
function featureVerifiedResult(){return null;}
let evidenceWorkflow=inspectionInitial({part:'part',token:'build',interfaceId:'rim'});function evidenceRender(){}
let featureDirty=false,featureInspection={name:'part',token:'build',result:{id:'rim',identity:{token:'known'},cad:[{}]}},listener;
const document={getElementById:()=>({addEventListener:(event,callback)=>{listener=callback;}})};
let symmetryRefreshes=0;function symmetryPanel(){symmetryRefreshes++;}
featureEditorState(entry);assert.match(fields.freshness.textContent,/matches/);
""" + handler + """
listener({target:{id:'featuresize'}});
assert.match(fields.freshness.textContent,/Unsaved feature edits/);assert.equal(symmetryRefreshes,1);
assert.equal(featureInspection,null);assert.equal(fields.json.disabled,true);assert.equal(fields.svg.disabled,true);
""")


def test_symmetry_categories_keep_unknowns_counts_thresholds_and_stale_feature_records():
    js([("function symmetryFeatureSignature(", "function symmetryPanel(")], """
const current='part',comparePreview=new Map(),comparePending=new Map();let featureDirty=false;
const options={axis:'x'},symmetryOptions=()=>options;
const category={count:2,unmatched_count:1,tolerance_mm:.01,status:'deviations',sides:{negative:{count:1},positive:{count:1,max_mm:.4}}};
const entry={name:'part',token:'build',target:{regions:[{name:'hole',feature:{id:'hole',center_mm:[3,0,0]}}]}};
const report={status:'measured',token:'build',options,feature_records:[['hole',entry.target.regions[0].feature]],feature_centers:{cad_centers:category},cad:{}};
const rows=symmetryCategoryRows(report);
assert.deepEqual(rows[1],['Authored CAD centers','2 total, 1 unmatched','unknown','0.4000','0.0100','deviations','unknown']);
assert.equal(rows[2][5],'not assessed');assert.equal(rows[3][5],'not assessed');
assert.equal(symmetryFresh(entry,report),true);
featureDirty=true;assert.equal(symmetryFresh(entry,report),false);featureDirty=false;
report.feature_records=JSON.parse(JSON.stringify(report.feature_records));
entry.target.regions[0].feature.center_mm[0]=4;
assert.equal(symmetryFresh(entry,report),false);
""")


def test_explicit_center_editor_roundtrip_and_clear_do_not_restore_a_hidden_alias():
    js([("function featureSectionValue(", "function featureEditorState(")], """
const previous={id:'center',point_mm:[-3,0,0],symmetry_group:'cushion holes'};
const selectedFeatureRegion=()=>({feature:previous}),current='part',parts=new Map();
const Option=(text,value)=>({text,value}),fields=new Map(),featureField=id=>{if(!fields.has(id))fields.set(id,{value:'',checked:false,setAttribute(){},replaceChildren(){}});return fields.get(id);};
const document={getElementById:id=>id==='evidenceinterface'?{options:[],value:''}:null};
const inspectionField=()=>({value:'preview'});let evidenceWorkflow=inspectionInitial();function evidenceRender(){}
let featureDirty=false,featureInspection=null,featureSeries=sectionSeriesInitial();
function featureEditorState(){}function inspectionVector(text){return text.split(' ').map(Number);}
featureEditorLoad({feature:previous});
assert.equal(featureField('center').value,'-3 0 0');assert.equal(featureField('symmetrygroup').value,'cushion holes');
let saved=featureRegionValues({name:'left hole'}).feature;
assert.deepEqual(saved.center_mm,[-3,0,0]);assert.equal(saved.symmetry_group,'cushion holes');assert.equal('point_mm' in saved,false);
featureField('center').value='';saved=featureRegionValues({name:'left hole'}).feature;
assert.equal('center_mm' in saved,false);assert.equal('point_mm' in saved,false);
""")
