"""Portable textured references retain source bytes and UV seams across every surface."""
import asyncio
import base64
import gzip
import io
import json
import pathlib
import stat
import zipfile

import numpy as np
import pytest
import trimesh
from PIL import Image

from nurb import checks, cli, compare, reference_import as imports, scan
from nurb.server import Server


def assets(compressed=False):
    body = b'''ply
format ascii 1.0
comment TextureFile texture.png
element vertex 7
property float x
property float y
property float z
element face 3
property list uchar int vertex_indices
property list uchar float texcoord
end_header
0 0 0
2 0 0
2 2 0
0 2 0
5 0 0
6 0 0
5 1 0
3 0 1 2 6 0 0 1 0 1 1
3 0 2 3 6 0.5 0.5 0.25 0.75 0 1
3 4 5 6 6 0 0 1 0 0 1
'''
    image = Image.new('RGB', (4, 4), 'red')
    image.putpixel((0, 0), (0, 255, 0))
    image.putpixel((3, 3), (0, 0, 255))
    stream = io.BytesIO(); image.save(stream, 'PNG')
    return {'scan.ply.gz' if compressed else 'scan.ply': gzip.compress(body) if compressed else body, 'texture.png': stream.getvalue()}


def zipped(items):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        for name, body in items.items():
            archive.writestr(name, body)
    return stream.getvalue()


@pytest.mark.parametrize('compressed', [False, True])
def test_bundle_preserves_uv_seams_units_sources_and_exclusion(tmp_path, compressed):
    source = assets(compressed)
    body = zipped(source)
    imported = imports.read_bytes('scan.zip', body, 'cm')
    assert len(imported.mesh.vertices) > 7  # UV seams split the original shared vertices.
    assert len(imported.mesh.faces) == 3
    assert imported.mesh.visual.kind == 'texture'
    assert np.allclose(imported.mesh.extents, [60, 20, 0])
    assert imported.provenance['units'] == {'input':'cm','source':'argument','scale_to_mm':10.0}
    assert len(imported.provenance['components']) == 2  # UV seams are not detached noise.
    file, record = imports.save(imported, tmp_path, ['component-2'])
    output = tmp_path / file
    assert record['texture']['sha256'] == imports.digest(source['texture.png'])
    assert (output.parent / 'original/scan.zip').read_bytes() == body
    for name, value in source.items():
        assert (output.parent / 'assets' / name).read_bytes() == value
    reloaded = trimesh.load(output, force='mesh', process=False)
    assert len(reloaded.faces) == 2
    assert reloaded.visual.kind == 'texture'
    assert np.allclose(reloaded.visual.uv[reloaded.faces], imported.mesh.visual.uv[imported.mesh.faces[:2]], atol=1e-7)
    assert np.allclose(reloaded.extents, [20, 20, 0])
    assert imports.manifest(output)['excluded_components'] == ['component-2']
    restored, _ = imports.save(imports.reopen(output), tmp_path)
    assert len(trimesh.load(tmp_path / restored, force='mesh').faces) == 3
    assert imports.save(imported, tmp_path, ['component-2'])[0] == file
    with pytest.raises(ValueError, match='at least one'):
        imported.scene(['component-1', 'component-2'])


@pytest.mark.parametrize('name', ['../outside.png', '/absolute.png', 'C:/drive.png', 'folder\\outside.png', 'a/../b.png', 'a//b.png'])
def test_zip_rejects_unsafe_paths(name):
    with pytest.raises(ValueError, match='Unsafe'):
        imports.read_bytes('scan.zip', zipped({**assets(), name:b'x'}), 'mm')


def test_zip_rejects_symlinks_duplicates_and_bombs(monkeypatch):
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w') as archive:
        entry=zipfile.ZipInfo('scan.ply'); entry.create_system=3; entry.external_attr=(stat.S_IFLNK | 0o777)<<16
        archive.writestr(entry,'../secret')
    with pytest.raises(ValueError,match='symlinks'):
        imports.read_bytes('scan.zip',buf.getvalue(),'mm')
    with pytest.raises(ValueError,match='ambiguous'):
        imports.read_bytes('scan.zip',zipped({**assets(),'SCAN.PLY':b'x'}),'mm')
    monkeypatch.setattr(imports,'MAX_BYTES',10)
    with pytest.raises(ValueError,match='limit'):
        imports.read_bytes('scan.zip',zipped(assets()),'mm')


@pytest.mark.parametrize('case, message', [('missing','Missing'), ('ambiguous','exactly one'), ('undeclared','no TextureFile'), ('unsafe','Unsafe'), ('badimage','texture'), ('no_uv','coordinates')])
def test_bad_texture_bundles_fail_explicitly(case, message):
    data=assets()
    if case=='missing': data.pop('texture.png')
    if case=='ambiguous': data['second.ply']=data['scan.ply']
    if case=='undeclared': data['scan.ply']=data['scan.ply'].replace(b'comment TextureFile texture.png\n',b'')
    if case=='unsafe': data['scan.ply']=data['scan.ply'].replace(b'texture.png',b'../texture.png')
    if case=='badimage': data['texture.png']=b'not an image'
    if case=='no_uv':
        data['scan.ply']=data['scan.ply'].replace(b'property list uchar float texcoord\n',b'')
        data['scan.ply']=data['scan.ply'].replace(b' 6 0 0 1 0 1 1',b'').replace(b' 6 0.5 0.5 0.25 0.75 0 1',b'').replace(b' 6 0 0 1 0 0 1',b'')
    with pytest.raises(ValueError,match=message):
        imports.read_bytes('scan.zip',zipped(data),'mm')


def test_sidecars_compressed_ply_and_source_tampering(tmp_path):
    data=assets(True)
    for name,body in data.items(): (tmp_path/name).write_bytes(body)
    source=tmp_path/'scan.ply.gz'
    imported=imports.read(source,'mm')
    assert imported.provenance['texture']['path']=='texture.png'
    mesh,unit,origin=scan.load(source,'mm')
    assert (unit,origin)==('mm','argument')
    assert len(mesh.vertices)==7  # Measurements weld UV seams; renderer never uses this copy.
    file,_=imports.save(imported,tmp_path/'project')
    output=tmp_path/'project'/file
    (output.parent/'assets/texture.png').write_bytes(b'changed')
    with pytest.raises(ValueError,match='asset changed'):
        imports.reopen(output)


def test_cli_preview_import_and_new_use_one_provenance_format(tmp_path,capsys):
    source=tmp_path/'scan.zip'; source.write_bytes(zipped(assets()))
    cli.main(['import',str(source),'--preview','--units','cm'])
    summary=json.loads(capsys.readouterr().out)
    assert len(summary['components'])==2
    root=tmp_path/'project'
    cli.main(['new','thing','--root',str(root),'--from',str(source),'--units','cm','--exclude-component','component-2'])
    target=compare.setting(checks.settings(root/'parts/thing.py'))
    assert target['units']=='mm'
    assert target['file'].endswith('.glb')
    assert imports.manifest(root/target['file'])['units']['input']=='cm'
    assert 'width=20.0' in (root/'parts/thing.py').read_text()
    capsys.readouterr()
    cli.main(['import',str(source),'--root',str(root),'--units','cm'])
    assert json.loads(capsys.readouterr().out)['excluded_components']==[]


class Client:
    def __init__(self): self.messages=[]
    async def send(self,raw): self.messages.append(json.loads(raw))


def test_server_bundle_upload_preview_filter_preserves_alignment_and_sources(tmp_path):
    (tmp_path/'parts').mkdir()
    part=tmp_path/'parts/thing.py'; part.write_text('from nurb import *\n@part\ndef thing(): return Box(2,2,1)\n')
    server=Server(tmp_path); server.queue=asyncio.Queue(); server.rebuild(part)
    client=Client()
    def command(**message):
        asyncio.run(server.command(json.dumps({'name':'thing',**message}),client))
        assert 'error' not in client.messages[-1], client.messages[-1]
        return client.messages[-1]
    command(type='target_reference',filename='scan.zip',units='cm',data=base64.b64encode(zipped(assets())).decode())
    compare.update_card(part, transform=compare.IDENTITY)
    server.rebuild(part)
    target=server.state['thing']['target']; original=tmp_path/target['file']
    asyncio.run(server.command(json.dumps({'type':'target_settings','name':'thing','units':'cm'}),client))
    assert 'normalized to mm' in client.messages[-1]['error']
    assert target['import']['texture']
    reply=command(type='target_components',stamp=target['stamp'],preview=True)
    assert len(trimesh.load(io.BytesIO(base64.b64decode(reply['glb'])),file_type='glb',force='mesh').faces)==3
    command(type='target_components',stamp=target['stamp'],excluded=['component-2'])
    updated=compare.setting(checks.settings(part))
    assert updated['transform']==compare.IDENTITY
    assert original.is_file()  # Previous derived reference and all source files remain available.
    assert imports.manifest(tmp_path/updated['file'])['excluded_components']==['component-2']
    server.rebuild(part)
    asyncio.run(server.command(json.dumps({'type':'target_components','name':'thing','stamp':target['stamp'],'excluded':[]}),client))
    assert 'changed' in client.messages[-1]['error']


def test_repackaged_sources_do_not_overwrite_each_other(tmp_path):
    data=assets()
    first=imports.read_bytes('scan.zip',zipped(data),'mm')
    second=imports.read_bytes('scan.zip',zipped(dict(reversed(list(data.items())))),'mm')
    assert first.provenance['asset_identity']==second.provenance['asset_identity']
    a,_=imports.save(first,tmp_path); b,_=imports.save(second,tmp_path)
    assert a!=b
    assert (tmp_path/a).parent.joinpath('original/scan.zip').read_bytes()==first.original
    assert (tmp_path/b).parent.joinpath('original/scan.zip').read_bytes()==second.original


def test_local_texture_symlink_cannot_escape_the_source_folder(tmp_path):
    folder=tmp_path/'source'; folder.mkdir()
    (folder/'scan.ply').write_bytes(assets()['scan.ply'])
    image=tmp_path/'outside.png'; image.write_bytes(assets()['texture.png'])
    (folder/'texture.png').symlink_to(image)
    with pytest.raises(ValueError,match='inside'):
        imports.read(folder/'scan.ply','mm')


def test_bundle_limits_and_conflicting_asset_paths(monkeypatch):
    data=assets()
    with pytest.raises(ValueError,match='conflicting'):
        imports.read_bytes('scan.zip',zipped({**data,'sub':b'a','sub/file.txt':b'b'}),'mm')
    monkeypatch.setattr(imports,'MAX_FILES',1)
    with pytest.raises(ValueError,match='too many'):
        imports.read_bytes('scan.zip',zipped(data),'mm')
    monkeypatch.setattr(imports,'MAX_FILES',128)
    data['scan.ply']=data['scan.ply'].replace(b'comment TextureFile texture.png',b'comment TextureFile texture.png\ncomment TextureFile other.png')
    with pytest.raises(ValueError,match='Multiple'):
        imports.read_bytes('scan.zip',zipped(data),'mm')


def test_server_compressed_sidecar_and_failed_upload_keep_existing_target(tmp_path):
    (tmp_path/'parts').mkdir()
    part=tmp_path/'parts/thing.py';part.write_text('from nurb import *\n@part\ndef thing(): return Box(2,2,1)\n')
    server=Server(tmp_path);server.queue=asyncio.Queue();server.rebuild(part);client=Client()
    data=assets(True)
    message={'type':'target_reference','name':'thing','filename':'scan.ply.gz','units':'cm',
             'data':base64.b64encode(data['scan.ply.gz']).decode(),
             'sidecars':[{'filename':'texture.png','data':base64.b64encode(data['texture.png']).decode()}]}
    asyncio.run(server.command(json.dumps(message),client))
    assert 'error' not in client.messages[-1]
    target=compare.setting(checks.settings(part))
    record=imports.manifest(tmp_path/target['file'])
    assert record['source']['name']=='scan.ply.gz'
    assert record['units']['input']=='cm'
    original_card=part.with_suffix('.md').read_bytes()
    for sidecars in [[], None, [{'filename':'texture.png','data':'!bad!'}], [{'filename':'../texture.png','data':'abcd'}]]:
        asyncio.run(server.command(json.dumps({**message,'sidecars':sidecars}),client))
        assert 'error' in client.messages[-1]
        assert part.with_suffix('.md').read_bytes()==original_card
