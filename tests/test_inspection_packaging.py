"""The installed wheel serves the same offline inspection controller as development."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest


def test_wheel_contains_importable_inspection_controller_and_dependencies(tmp_path):
    uv=shutil.which('uv')
    node=shutil.which('node')
    if not uv or not node:
        pytest.skip('uv and Node.js are needed for the installed inspection asset check')
    root=Path(__file__).resolve().parents[1]
    built=subprocess.run([uv,'build','--offline','--wheel','--out-dir',str(tmp_path/'dist')],cwd=root,capture_output=True,text=True)
    assert built.returncode==0,built.stderr
    installed=tmp_path/'installed'
    with zipfile.ZipFile(next((tmp_path/'dist').glob('*.whl'))) as archive:
        for asset in ('inspection-state.js','inspection-viewer.js','vendor/three/build/three.module.min.js'):
            assert 'nurb/'+asset in archive.namelist()
        archive.extractall(installed)
    source=f"""
import {{createInspectionController,sectionContours}} from {json.dumps((installed/'nurb/inspection-viewer.js').as_uri())};
import * as THREE from {json.dumps((installed/'nurb/vendor/three/build/three.module.min.js').as_uri())};
import assert from 'node:assert/strict';
assert.equal(sectionContours(new THREE.BoxGeometry(2,2,2),new THREE.Matrix4(),2,.25).valid,true);
let mesh=new THREE.Group();mesh.add(new THREE.Mesh(new THREE.BoxGeometry()));
const viewer=createInspectionController({{get mesh(){{return mesh;}}}});
assert.equal(viewer.modelNodes().length,1);
mesh=new THREE.Group();assert.equal(viewer.modelNodes().length,0);
"""
    imported=subprocess.run([node,'--input-type=module','-'],input=source,capture_output=True,text=True,cwd=tmp_path)
    assert imported.returncode==0,imported.stderr
    served=subprocess.run([sys.executable,'-c',f"""
import asyncio,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,{str(installed)!r})
import nurb.server
assert Path(nurb.server.__file__).is_relative_to({str(installed)!r})
server=nurb.server.Server(Path({str(tmp_path)!r}))
for asset in ('inspection-viewer.js','inspection-state.js','vendor/three/build/three.module.min.js'):
    response=asyncio.run(server.http(None,SimpleNamespace(path='/'+asset)))
    assert response.status_code==200
    assert response.headers['Content-Type'].startswith('text/javascript')
"""],cwd=tmp_path,capture_output=True,text=True)
    assert served.returncode==0,served.stderr
