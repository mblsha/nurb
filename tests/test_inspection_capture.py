"""Saved inspection state identifies inputs and capture races cannot publish evidence."""
import asyncio
import base64
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest
import trimesh

from nurb import cli, compare, inspection, symmetry
from nurb.server import Server


def state(mode='overlay'):
    return {'camera':{'position_mm':[25,-35,30],'target_mm':[0,0,0],'up':[0,0,1],'zoom':2,'height_mm':80},
            'viewport':[480,360],'mode':mode,'hidden_components':[],
            'section':{'enabled':False,'axis':'z','position_mm':0,'fraction':.5,'sign':1},
            'alignment':compare.IDENTITY,'tolerance_mm':.1,'region':'Rim','station':1,
            'deviation_scale_mm':1,'deviation_map':'both','deviation_auto':False,'deviation_through':True,'pins':False,'symmetry_plane':False}


def project(tmp_path):
    (tmp_path/'parts').mkdir()
    part=tmp_path/'parts/thing.py';part.write_text('from nurb import *\n@part\ndef thing(width=10.0):\n    return Box(width,8,6)-Cylinder(1,10)\n')
    trimesh.creation.box(extents=[10,8,6]).export(tmp_path/'scan.ply')
    part.with_suffix('.md').write_text('# Thing\n\n```toml\ntarget={file="scan.ply",units="mm",transform=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]}\n```\n')
    compare.update_card(part,regions=[{'name':'Rim','bounds_mm':{'min':[-6,-5,-4],'max':[6,5,4]},'feature':{'id':'rim','role':'mating rim','sections':[{'name':'Rim stations','origin_mm':[0,0,0],'normal':[0,0,1],'offsets_mm':[-1,1]}]}}])
    server=Server(tmp_path);server.queue=asyncio.Queue();server.rebuild(part);server.check(part)
    messages=[]
    async def send(message):messages.append(message)
    server.send=send
    return part,server,messages


def png():
    from PIL import Image
    stream=io.BytesIO();Image.new('RGB',(100,100),'#304050').save(stream,format='PNG');return stream.getvalue()


def test_saved_setup_roundtrip_keeps_semantic_sections_and_does_not_expire_itself(tmp_path):
    part,server,_=project(tmp_path);entry=server.state['thing']
    before=symmetry.source_revision(part)
    saved=inspection.save(server,part,entry,state(),'Narrow rim / comparison')
    assert symmetry.source_revision(part)==before
    loaded=inspection.load(tmp_path,saved['id'])
    assert loaded==saved
    assert loaded['region']['feature']['sections'][0]['offsets_mm']==[-1,1]
    assert loaded['configuration']['parameters']=={'width':10.0}
    assert loaded['verification']['comparison']['status']=='measured'
    assert inspection.freshness(loaded['identity'],inspection.identity(server,part,entry,loaded['view']))['status']=='current'
    assert inspection.list_setups(tmp_path,'thing')==[saved]
    assert inspection.sections(server,entry,saved)['cad'][0]['offset_mm']==-1


@pytest.mark.parametrize('change',['geometry','source','reference','configuration','alignment','tolerance','feature'])
def test_input_changes_are_explicitly_stale(tmp_path,change):
    part,server,_=project(tmp_path);entry=server.state['thing']
    saved=inspection.save(server,part,entry,state(),'Rim')
    current=copy.deepcopy(saved['identity'])
    key={'source':'source_revision','tolerance':'tolerance_mm','feature':'region'}.get(change,change)
    current[key]=None
    result=inspection.freshness(saved['identity'],current)
    assert result=={'status':'stale','changed':[key]}


def test_capture_bundle_has_human_report_assets_frames_and_current_evidence(tmp_path):
    part,server,_=project(tmp_path);entry=server.state['thing']
    saved=inspection.save(server,part,entry,state(),'Rim')
    sections=inspection.sections(server,entry,saved)
    cached=entry['target']['metrics']
    destination,evidence=inspection.bundle(server,entry,saved,saved['identity'],png(),sections,
        ['<svg xmlns="http://www.w3.org/2000/svg"><text x="1">Rim</text></svg>'],cached)
    with zipfile.ZipFile(destination) as z:
        assert {'view.png','model.glb','reference.glb','setup.json','evidence.json','report.md','sections/001.svg'}<=set(z.namelist())
        report=z.read('report.md').decode()
        assert 'Sampled max mm' in report and '**current**' in report
        payload=json.loads(z.read('evidence.json'))
        assert payload['asset_frames']['reference.glb']['transform_to_part_mm']==compare.IDENTITY
    assert entry['target']['metrics'] is cached
    assert evidence['verification']['comparison']['status']=='measured'
    moved=copy.deepcopy(saved['view']);moved['alignment'][3]=1
    assert inspection.verification(entry,moved)['comparison']['status']=='unknown'


def test_capture_filename_traversal_symlinks_and_active_svg_are_rejected(tmp_path):
    part,server,_=project(tmp_path);entry=server.state['thing'];saved=inspection.save(server,part,entry,state(),'Rim')
    for bad in ('../outside','a/b','x'*32):
        with pytest.raises(ValueError):inspection.setup_path(tmp_path,bad)
    path=inspection.setup_path(tmp_path,'f'*32);path.symlink_to(part)
    with pytest.raises(ValueError):inspection.load(tmp_path,'f'*32)
    for svg in ('<svg onload="alert(1)"/>','<svg><script>bad()</script></svg>','<svg><text fill="url(https://bad)">bad</text></svg>'):
        with pytest.raises(ValueError):inspection.bundle(server,entry,saved,saved['identity'],png(),section_images=[svg])
    with pytest.raises(ValueError,match='PNG'):inspection.png_bytes(base64.b64encode(b'not an image').decode())
    bad=state();bad['camera']['zoom']=float('nan')
    with pytest.raises(ValueError,match='positive'):inspection.view_state(bad)


@pytest.mark.parametrize('change',['source','reference','rebuild'])
def test_two_phase_capture_refuses_changed_inputs(tmp_path,change):
    part,server,messages=project(tmp_path)
    async def scenario():
        await server.command(json.dumps({'type':'inspection_prepare','name':'thing','token':server.state['thing']['token'],'view':state(),'label':'Rim'}))
        prepared=messages[-1];assert 'error' not in prepared,prepared
        if change=='source':part.write_text(part.read_text()+'# edited\n')
        elif change=='reference':(tmp_path/'scan.ply').write_bytes((tmp_path/'scan.ply').read_bytes()+b'\n')
        else:server.rebuild(part)
        await server.command(json.dumps({'type':'inspection_capture','name':'thing','token':prepared['token'],'ticket':prepared['ticket'],'png':base64.b64encode(png()).decode()}))
    asyncio.run(scenario())
    assert 'error' in messages[-1]
    assert not list((tmp_path/'build/inspection-evidence').glob('*.zip'))


def test_websocket_save_restore_and_capture_and_cli_list(tmp_path,monkeypatch,capsys):
    part,server,messages=project(tmp_path)
    async def scenario():
        await server.command(json.dumps({'type':'inspection_save','name':'thing','token':server.state['thing']['token'],'view':state(),'label':'Rim'}))
        saved=messages[-1]['setup']
        await server.command(json.dumps({'type':'inspection_prepare','name':'thing','token':server.state['thing']['token'],'view':state(),'id':saved['id']}))
        prepared=messages[-1];assert 'error' not in prepared,prepared
        await server.command(json.dumps({'type':'inspection_capture','name':'thing','token':prepared['token'],'ticket':prepared['ticket'],'png':base64.b64encode(png()).decode()}))
        assert Path(messages[-1]['path']).is_file()
        await server.command(json.dumps({'type':'inspection_restore','name':'thing','id':saved['id']}))
        assert server.overrides['thing']=={'width':10.0}
        assert messages[-1]['restoring']
        return saved
    saved=asyncio.run(scenario());capsys.readouterr();monkeypatch.chdir(tmp_path)
    cli.main(['inspection','thing','--list','--json'])
    result=json.loads(capsys.readouterr().out)
    assert result['setups'][0]['id']==saved['id']
    assert result['setups'][0]['status']=='current'


def test_capture_download_rejects_paths_outside_fixed_bundle_directory(tmp_path):
    _,server,_=project(tmp_path)
    response=asyncio.run(server.http(None,SimpleNamespace(path='/inspection-evidence/../../secret',headers={})))
    assert response.status_code==404


def test_desktop_evidence_save_is_safe_unique_and_reports_written_path(tmp_path):
    part,server,messages=project(tmp_path)
    async def save():
        await server.command(json.dumps({'type':'artifact_save','name':'thing','request_id':'download',
            'filename':'thing-symmetry.json','mime':'application/json','data':'{"status":"measured"}'}))
    asyncio.run(save());first=Path(messages[-1]['path'])
    assert first.read_text()=='{"status":"measured"}'
    assert messages[-1]['request_id']=='download'
    asyncio.run(save());assert Path(messages[-1]['path'])!=first
    for name in ('../secret.json','/tmp/secret.json','bad.html'):
        with pytest.raises(ValueError): inspection.save_artifact(tmp_path,{'filename':name,'mime':'application/json','data':'{}'})
    with pytest.raises(ValueError,match='inert'):
        inspection.save_artifact(tmp_path,{'filename':'section.svg','mime':'image/svg+xml','data':'<svg><use href="file:///secret"/></svg>'})
    assert inspection.save_artifact(tmp_path,{'filename':'section.svg','mime':'image/svg+xml','data':'<svg><metadata>{"frame":"local_mm"}</metadata><polyline points="0,0 1,1"/></svg>'}).is_file()


def test_cli_saved_inspection_render_adapter_and_export(tmp_path,monkeypatch,capsys):
    import contextlib
    import sys
    import threading
    from nurb import render
    part,server,_=project(tmp_path)
    saved=inspection.save(server,part,server.state['thing'],state(),'Rim')
    calls=[]
    class Page:
        def goto(self,url): calls.append(('goto',url))
        def wait_for_function(self,value,**kwargs): calls.append(('ready',value))
        def locator(self,value): return self
        def evaluate(self,script,data=None):
            if data is None:return None
            calls.append(('capture',data))
            assert data['sections']['cad'][1]['offset_mm']==1
            assert data['freshness']['status']=='current'
            return {'png':base64.b64encode(png()).decode(),'section_images':['<svg/>']}
    class Browser:
        def new_page(self,**kwargs): calls.append(('viewport',kwargs));return Page()
        def close(self):pass
    @contextlib.contextmanager
    def playwright():yield SimpleNamespace(chromium=SimpleNamespace(launch=lambda:Browser()))
    monkeypatch.setitem(sys.modules,'playwright',SimpleNamespace())
    monkeypatch.setitem(sys.modules,'playwright.sync_api',SimpleNamespace(sync_playwright=playwright))
    monkeypatch.setattr(render,'_host',lambda server:threading.Event())
    monkeypatch.chdir(tmp_path);capsys.readouterr()
    destination=tmp_path/'chosen-evidence.zip'
    cli.main(['inspection','thing','--export',saved['id'],'--output',str(destination),'--json'])
    result=json.loads(capsys.readouterr().out)
    assert result['status']=='current'
    assert destination.is_file() and destination.with_suffix('.png').read_bytes()==png()
    assert any(call[0]=='ready' for call in calls)
    assert next(call[1] for call in calls if call[0]=='capture')['setup']['view']['camera']==saved['view']['camera']
    assert next(call[1] for call in calls if call[0]=='viewport')['viewport']=={'width':480,'height':360}


def test_saved_inspection_render_missing_optional_extra_is_actionable(tmp_path,monkeypatch,capsys):
    import sys
    part,server,_=project(tmp_path)
    saved=inspection.save(server,part,server.state['thing'],state(),'Rim')
    monkeypatch.setitem(sys.modules,'playwright',None)
    monkeypatch.setitem(sys.modules,'playwright.sync_api',None)
    monkeypatch.chdir(tmp_path);capsys.readouterr()
    with pytest.raises(SystemExit) as exc:cli.main(['inspection','thing','--render',saved['id'],'--json'])
    assert exc.value.code==2
    result=json.loads(capsys.readouterr().out)
    assert result['status']=='unknown'
    assert 'nurb[render]' in result['error'] and 'playwright install chromium' in result['error']
    assert not list((tmp_path/'build/inspection-evidence').glob('*.zip'))
