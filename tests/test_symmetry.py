"""Symmetry evidence covers the finished trim geometry, not only parent surfaces."""
import asyncio
from dataclasses import replace
import gzip
import json
import threading

import numpy as np
import pytest
import trimesh
from build123d import Box, Cylinder, Pos

from nurb import cli, compare, symmetry
from nurb.server import Server


def reference_mesh():
    return trimesh.creation.box(extents=[10, 8, 6])


def project(tmp_path):
    (tmp_path / "parts").mkdir()
    path = tmp_path / "parts/thing.py"
    path.write_text("from nurb import *\n@part\ndef thing(width=10.0):\n    return Box(width,8,6)\n")
    path.with_suffix('.md').write_text('# Thing\n\n```toml\ntarget={file="scan.ply",units="mm",transform=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]}\n```\n')
    reference_mesh().export(tmp_path / "scan.ply")
    return path


def test_mesh_fit_recovers_rotated_translated_plane_with_holdout_evidence():
    mesh = reference_mesh()
    matrix = trimesh.transformations.rotation_matrix(np.deg2rad(6), [0, 0, 1])
    matrix[:3, 3] = [2, 3, 4]; mesh.apply_transform(matrix)
    result = symmetry.fit_plane(mesh, symmetry.Options())
    assert result['normal'] == pytest.approx(matrix[:3, 0], abs=0.001)
    assert result['offset_mm'] == pytest.approx(matrix[:3, 0] @ matrix[:3, 3], abs=0.005)
    assert result['reference_status'] == 'within_sampled_threshold'
    for side in ('negative', 'positive'):
        assert result['reference_sides'][side]['p95_mm'] < 0.01
    aligned = symmetry.reflect(np.asarray(mesh.vertices), result['normal'], result['offset_mm'])
    assert aligned.shape == (8, 3)
    transform = np.asarray(result['to_axis_transform']).reshape(4, 4)
    assert transform[:3, :3] @ result['normal'] == pytest.approx([1, 0, 0], abs=1e-6)


def test_compressed_point_cloud_requires_units_and_keeps_original_bytes(tmp_path):
    points = np.random.default_rng(12).uniform([0.5, -4, -3], [5, 4, 3], (500, 3))
    points = np.vstack([points, points * [-1, 1, 1]]) + [2, 3, 4]
    body = trimesh.points.PointCloud(points).export(file_type='ply')
    path = tmp_path / 'cloud.ply.gz'; path.write_bytes(gzip.compress(body, mtime=0))
    original = path.read_bytes()
    with pytest.raises(ValueError, match='explicit units'):
        symmetry.load_reference(path)
    reference, unit = symmetry.load_reference(path, 'mm')
    result = symmetry.fit_plane(reference, symmetry.Options())
    assert unit == 'mm'
    assert result['offset_mm'] == pytest.approx(2, abs=1e-5)
    assert result['reference_sides']['positive']['max_mm'] < 1e-5
    assert 'sampling gaps' in result['warnings'][0]
    assert path.read_bytes() == original


def test_trimmed_holes_detect_asymmetry_that_distance_to_material_hides():
    plane = {'normal': [1, 0, 0], 'offset_mm': 0}
    options = symmetry.Options(edge_step_mm=2)
    one_hole = Box(12, 8, 6) - Pos(3, 0, 0) * Cylinder(1, 10)
    result = symmetry.cad_symmetry(one_hole, plane, options)
    assert result['status'] == 'deviations'
    assert result['trim_edges']['max_mm'] >= 0.99
    assert result['trimmed_faces']['max_mm'] >= 2.9
    paired = one_hole - Pos(-3, 0, 0) * Cylinder(1, 10)
    measured = symmetry.cad_symmetry(paired, plane, options)
    assert measured['status'] == 'within_sampled_threshold'
    assert measured['trim_edges']['max_mm'] < 1e-6
    assert measured['trimmed_faces']['max_mm'] < 1e-6
    assert measured['faces'] == len(paired.faces())
    assert measured['edges'] == len(paired.edges())


def test_sampling_never_truncates_to_claim_a_pass():
    with pytest.raises(ValueError, match='budget'):
        symmetry.cad_symmetry(Box(10, 8, 6), {'normal': [1, 0, 0], 'offset_mm': 0}, symmetry.Options(sample_budget=100))
    cloud = trimesh.points.PointCloud([[x, 0, 0] for x in range(40)])
    with pytest.raises(ValueError, match='nearly a line'):
        symmetry.fit_plane(cloud, symmetry.Options())
    for options in ({'cad_tolerance_mm': float('nan')}, {'axis': 'q'}, {'sample_budget': True}, {'timeout_s': 121}):
        with pytest.raises(ValueError):
            symmetry.Options(**options)


@pytest.mark.parametrize('change', ['geometry', 'reference', 'alignment', 'configuration', 'revision', 'threshold'])
def test_identity_changes_for_each_evidence_input(change):
    args = [Box(10, 8, 6), 'ref1', compare.IDENTITY, {'name': 'default', 'parameters': {'width': 10}}, 'rev1', symmetry.Options()]
    first = symmetry.identity(*args)
    if change == 'geometry': args[0] -= Pos(3, 0, 0) * Cylinder(1, 10)
    elif change == 'reference': args[1] = 'ref2'
    elif change == 'alignment': args[2] = list(compare.IDENTITY); args[2][3] = 1
    elif change == 'configuration': args[3]['name'] = 'folded'
    elif change == 'revision': args[4] = 'rev2'
    else: args[5] = replace(args[5], cad_tolerance_mm=0.2)
    assert symmetry.identity(*args)['token'] != first['token']


def test_cli_real_worker_report_stays_current_then_expires(tmp_path, monkeypatch, capsys):
    part = project(tmp_path); monkeypatch.chdir(tmp_path)
    report = tmp_path / 'symmetry.json'
    cli.main(['symmetry', 'thing', '--json', '--output', str(report), '--edge-step', '2'])
    measured = json.loads(capsys.readouterr().out)
    assert measured['status'] == 'measured'
    assert measured['cad']['status'] == 'within_sampled_threshold'
    assert measured['kind'] == 'symmetry_evidence'
    assert len(measured['source_revision']) == 64
    cli.main(['symmetry', 'thing', '--json', '--check-report', str(report), '--edge-step', '2'])
    assert json.loads(capsys.readouterr().out)['status'] == 'current'
    part.write_text(part.read_text() + '# a source revision\n')
    cli.main(['symmetry', 'thing', '--json', '--check-report', str(report), '--edge-step', '2'])
    assert json.loads(capsys.readouterr().out)['status'] == 'stale'


def test_child_worker_cancellation_and_deadline_are_unknown():
    shape = Box(10, 8, 6)
    with pytest.raises(ValueError, match='cancelled'):
        symmetry.run(shape, reference_mesh(), symmetry.Options(), {}, stop=lambda: True)
    with pytest.raises(ValueError, match='exceeded'):
        symmetry.run(shape, reference_mesh(), symmetry.Options(timeout_s=0.001), {})


@pytest.mark.parametrize('action', ['cancel', 'rebuild', 'reference'])
def test_viewer_jobs_are_async_cancellable_and_do_not_publish_stale_distances(tmp_path, monkeypatch, action):
    part = project(tmp_path)
    server = Server(tmp_path); server.queue = asyncio.Queue(); server.rebuild(part)
    messages = []
    async def capture(message): messages.append(message)
    server.send = capture
    started, finish = threading.Event(), threading.Event()
    def wait(shape, reference, options, identity, stop):
        started.set(); finish.wait(5)
        return {'status': 'measured', 'identity': identity, 'cad': {'sample_count': 1}}
    monkeypatch.setattr(symmetry, 'run', wait)
    async def scenario():
        await server.command(json.dumps({'type': 'target_symmetry', 'name': 'thing'}))
        job = server.symmetry_jobs['thing']
        for _ in range(500):
            if started.is_set(): break
            await asyncio.sleep(.01)
        assert started.is_set()
        assert messages[0]['status'] == 'running'
        if action == 'cancel':
            await server.command(json.dumps({'type':'target_symmetry_cancel','name':'thing','token':job['token'],'request_id':job['request_id']}))
        elif action == 'rebuild':
            server.overrides['thing'] = {'width': 12}; server.rebuild(part)
        else:
            (tmp_path / 'scan.ply').write_bytes(reference_mesh().export(file_type='ply') + b'\n')
        finish.set(); await job['task']
    asyncio.run(scenario())
    assert messages[-1]['status'] == ('cancelled' if action == 'cancel' else 'stale')
    assert 'cad' not in messages[-1]
    assert not server.symmetry_jobs


def test_cached_independent_cloud_and_nested_inputs_expire_without_cad_rebuild(tmp_path):
    from nurb import symmetry_service
    part = project(tmp_path)
    server = Server(tmp_path); server.queue = asyncio.Queue(); server.rebuild(part)
    (tmp_path / 'scans').mkdir(); cloud = tmp_path / 'scans/cloud.ply'
    cloud.write_bytes((tmp_path / 'scan.ply').read_bytes())
    entry = server.state['thing']
    report = {'type':'target_symmetry','name':'thing','token':entry['token'],'request_id':'run1','status':'measured',
              'reference_kind':'independent','reference_file':'scans/cloud.ply','reference_units':'mm',
              'source_revision':symmetry.source_revision(part),
              'identity':{'reference':symmetry.reference_identity(cloud, 'mm')},'options':{}, 'cad':{'sample_count':1}}
    messages = []
    async def send(message): messages.append(message)
    server.send = send
    entry['symmetry'] = report
    asyncio.run(symmetry_service.invalidate(server, {cloud}))
    assert not messages
    cloud.write_bytes(cloud.read_bytes()+b'\n')
    asyncio.run(symmetry_service.invalidate(server, {cloud}))
    assert messages[-1]['status'] == 'stale'
    assert 'cad' not in entry['symmetry']
    entry['symmetry'] = report
    (tmp_path / 'measurements').mkdir(); dimensions=tmp_path/'measurements/dimensions.json'
    dimensions.write_text('{"spacing":22}')
    asyncio.run(symmetry_service.invalidate(server, {dimensions}))
    assert len(messages) == 2
    assert messages[-1]['status'] == 'stale'


def test_reference_snapshot_geometry_and_hash_share_the_same_bytes(tmp_path, monkeypatch):
    path=tmp_path/'scan.ply'; reference_mesh().export(path)
    digest=symmetry.reference_identity(path,'mm'); original=symmetry.load_reference
    def changed_during_load(snapshot, units):
        trimesh.creation.box(extents=[20, 8, 6]).export(path)
        return original(snapshot,units)
    monkeypatch.setattr(symmetry,'load_reference',changed_during_load)
    mesh,unit,identity=symmetry.reference_snapshot(path,'mm')
    assert mesh.extents == pytest.approx([10,8,6])
    assert identity == digest
    assert symmetry.reference_identity(path,unit) != identity


def test_server_point_cloud_upload_runs_without_replacing_attached_reference(tmp_path):
    import base64
    part=project(tmp_path)
    server=Server(tmp_path); server.queue=asyncio.Queue(); server.rebuild(part)
    server.check(part)
    original_target=server.state['thing']['target']['file']
    cached_metrics=server.state['thing']['target']['metrics']
    points=np.random.default_rng(5).uniform([.5,-4,-3],[5,4,3],(200,3))
    body=trimesh.points.PointCloud(np.vstack([points,points*[-1,1,1]])).export(file_type='ply')
    compressed=gzip.compress(body,mtime=0)
    messages=[]
    async def capture(message): messages.append(message)
    server.send=capture
    async def scenario():
        await server.command(json.dumps({'type':'target_symmetry','name':'thing','options':{'edge_step_mm':2},
            'cloud':{'filename':'reference.ply.gz','units':'mm','data':base64.b64encode(compressed).decode()}}))
        await server.symmetry_jobs['thing']['task']
    asyncio.run(scenario())
    result=messages[-1]
    assert result['status'] == 'measured'
    assert result['reference_kind'] == 'independent'
    assert result['cad']['status'] == 'within_sampled_threshold'
    assert (tmp_path / result['reference_file']).read_bytes() == compressed
    assert server.state['thing']['target']['file'] == original_target
    assert server.state['thing']['target']['metrics'] is cached_metrics


def test_cli_unknown_is_machine_readable_and_unsuccessful(tmp_path, monkeypatch, capsys):
    project(tmp_path); monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as error:
        cli.main(['symmetry','thing','--timeout','0.001','--json'])
    assert error.value.code == 2
    result=json.loads(capsys.readouterr().out)
    assert result['status'] == 'unknown'
    assert 'exceeded' in result['error']
    assert 'cad' not in result
