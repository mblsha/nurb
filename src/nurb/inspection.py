"""Named inspection views and reproducible evidence bundles."""
from datetime import datetime, timezone
import base64
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import re
import struct
import uuid
import zipfile

from . import compare, feature_evidence, symmetry, validator_evidence

MODES = ('model', 'reference', 'overlay', 'deviation', 'side-by-side', 'section')
ID = re.compile(r'^[a-f0-9]{32}$')
ENGINE_RUNTIME_PACKAGES = ('python','build123d','numpy','OCP_module')


def vector(value, length, label):
    if not isinstance(value, (list, tuple)) or len(value) != length or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
        raise ValueError(f'{label} needs {length} finite numbers')
    return list(value)


def view_state(value):
    if not isinstance(value, dict):
        raise ValueError('save a view from a built model first')
    camera = value.get('camera', {})
    camera = {**{k: vector(camera.get(k), 3, f'camera {k}') for k in ('position_mm', 'target_mm', 'up')},
              **{k: float(camera.get(k, 0)) for k in ('zoom', 'height_mm')}}
    if any(not math.isfinite(camera[k]) or camera[k] <= 0 for k in ('zoom', 'height_mm')):
        raise ValueError('camera zoom and height must be positive')
    if sum(x*x for x in camera['up']) < 1e-12 or sum((a-b)**2 for a,b in zip(camera['position_mm'],camera['target_mm'])) < 1e-12:
        raise ValueError('choose a camera with a direction and a nonzero up vector')
    viewport = vector(value.get('viewport', [1200,900]), 2, 'viewport')
    if any(int(v) != v or not 64 <= v <= 8192 for v in viewport):
        raise ValueError('capture dimensions must be integers from 64 to 8192 pixels')
    mode = value.get('mode', 'model')
    if mode not in MODES:
        raise ValueError('choose a supported comparison mode')
    cut = value.get('section') or {}
    axis = cut.get('axis', 'z')
    if axis not in ('x','y','z'):
        raise ValueError('section axis must be x, y, or z')
    at = cut.get('position_mm')
    fraction = float(cut.get('fraction', .5))
    if at is not None:
        at = vector([at],1,'section position')[0]
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('section fraction must be between zero and one')
    hidden = value.get('hidden_components', [])
    if not isinstance(hidden,list) or len(hidden)>2000 or any(not isinstance(v,str) or len(v)>200 for v in hidden):
        raise ValueError('component visibility must contain stable component identifiers')
    alignment = compare._transform(value.get('alignment') or compare.IDENTITY).reshape(-1).tolist()
    tolerance = float(value.get('tolerance_mm', .1))
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('inspection tolerance must be a positive millimetre value')
    region = value.get('region')
    if region is not None and (not isinstance(region,str) or len(region)>200):
        raise ValueError('choose a named inspection region')
    station = value.get('station',0)
    if not isinstance(station,int) or isinstance(station,bool) or not 0 <= station <= 1000:
        raise ValueError('section station must be a nonnegative index')
    scale = float(value.get('deviation_scale_mm',1))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('deviation color scale must be positive')
    section_source=value.get('section_source','preview')
    if section_source not in ('preview','verified'):
        raise ValueError('choose Preview or Verified for local section evidence')
    policy=None
    if section_source=='verified':
        from dataclasses import asdict
        from .meshing import VerificationPolicy
        raw=value.get('verification_policy')
        if not isinstance(raw,dict):
            raise ValueError('verified sections need the saved verification policy; run precise verification first')
        try: policy=asdict(VerificationPolicy(**raw))
        except (TypeError,ValueError) as exc:
            raise ValueError(f'the saved section verification policy is invalid: {exc}') from exc
    return {'camera':camera,'viewport':[int(v) for v in viewport],'mode':mode,'hidden_components':sorted(set(hidden)),
            'section':{'enabled':bool(cut.get('enabled')) or mode=='section','axis':axis,'position_mm':at,'fraction':fraction,'sign':-1 if cut.get('sign')==-1 else 1},
            'alignment':alignment,'tolerance_mm':tolerance,'region':region,'station':station,
            'deviation_scale_mm':scale,'deviation_map':value.get('deviation_map') if value.get('deviation_map') in ('part','target','both','off') else 'both',
            'deviation_auto':bool(value.get('deviation_auto')),'deviation_through':bool(value.get('deviation_through',True)),
            'pins':bool(value.get('pins',False)),'symmetry_plane':bool(value.get('symmetry_plane',False)),
            'section_source':section_source,'verification_policy':policy}


def verification_policy(provenance):
    """The reproducible request, separate from measured output and resource usage."""
    return {'accuracy_mm':provenance.get('absolute_deflection_mm'),
            'timeout_s':provenance.get('timeout_s'), 'max_triangles':provenance.get('max_triangles'),
            'feature_size_mm':provenance.get('feature_size_mm'),
            'memory_limit_mb':(provenance.get('memory') or {}).get('limit_mb')}


def same_view(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(same_view(left[k],right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left)==len(right) and all(same_view(a,b) for a,b in zip(left,right))
    if type(left) in (int,float) and type(right) in (int,float):
        return math.isclose(left,right,rel_tol=1e-10,abs_tol=1e-8)
    return left == right


def directory(root, *pieces):
    root = Path(root).resolve()
    path = root.joinpath(*pieces).resolve()
    if not path.is_relative_to(root):
        raise ValueError('inspection storage must stay inside the project; remove its external symlink')
    path.mkdir(parents=True,exist_ok=True)
    return path


def setup_path(root, identifier):
    if not isinstance(identifier,str) or not ID.fullmatch(identifier):
        raise ValueError('choose a saved inspection ID from the list')
    return directory(root,'inspections') / (identifier+'.json')


def load(root, identifier):
    path = setup_path(root,identifier)
    if path.is_symlink() or not path.is_file():
        raise ValueError('that saved inspection is unavailable; refresh the list')
    item = json.loads(path.read_text())
    if not isinstance(item,dict) or item.get('kind') != 'nurb_inspection_setup' or item.get('id') != identifier:
        raise ValueError('the saved inspection is not a supported setup')
    if not isinstance(item.get('part'),str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',item['part']):
        raise ValueError('the saved inspection has an unsafe part name')
    config=item.get('configuration') or {}
    parameters=config.get('parameters')
    if not isinstance(parameters,dict) or any(not isinstance(k,str) or not k.isidentifier() or not isinstance(v,(str,int,float,bool)) or isinstance(v,float) and not math.isfinite(v) for k,v in parameters.items()):
        raise ValueError('the saved inspection needs finite named parameter values')
    if not isinstance(item.get('identity'),dict) or not isinstance(item.get('name'),str):
        raise ValueError('the saved inspection is missing its name or source identity')
    item['view'] = view_state(item['view'])
    return item


def list_setups(root, part=None):
    output = []
    for path in sorted(directory(root,'inspections').glob('*.json')):
        if not ID.fullmatch(path.stem):
            continue
        try:
            item=load(root,path.stem)
            if part is None or item['part']==part:
                output.append(item)
        except (ValueError, OSError, KeyError):
            continue
    return output


def configuration(entry):
    return {'name':entry.get('variant') or 'default', 'parameters':{p['name']:p['value'] for p in entry.get('params',[])}}


def engine_revision(source_root=None, runtime=None):
    """Identify portable engine source and runtime versions without native CAD output."""
    versions=runtime if runtime is not None else validator_evidence.runtime_versions(ENGINE_RUNTIME_PACKAGES)
    document={'schema':'nurb-engine-recipe-v1','source_revision':validator_evidence.engine_source_digest(source_root),
              'runtime_versions':{name:str(version) for name,version in sorted(versions.items())}}
    return hashlib.sha256(json.dumps(document,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def build_recipe_identity(source_revision, current_configuration, draft, current_engine_revision=None):
    """Identify deterministic build inputs after the model has built successfully."""
    document={'schema':'nurb-build-recipe-v1','source_revision':str(source_revision),
              'configuration':current_configuration,'draft':bool(draft),
              'engine_revision':current_engine_revision or engine_revision()}
    return hashlib.sha256(json.dumps(document,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def identity(server, path, entry, state):
    if entry.get('shape') is None or entry.get('error'):
        raise ValueError('the model must build successfully before saving or checking an inspection')
    target=entry.get('target') or {}
    reference=None
    if target.get('file'):
        hit=server._target_mesh(target['file'],target.get('units'))
        if hit['stamp'] != target.get('stamp'):
            raise ValueError('the reference changed; wait for its refresh before capturing')
        reference={'file':Path(target['file']).name if Path(target['file']).is_absolute() else target['file'],
                   'content':hit['content_id'],'units':hit['unit'],'display_scale':hit['display_scale']}
    if state['mode']!='model' and not reference:
        raise ValueError('this capture mode needs a reference; add one in Compare or choose Model')
    region=next((r for r in target.get('regions',[]) if r['name']==state['region']),None)
    if state['region'] and not region:
        raise ValueError('the selected inspection region is missing; choose a current region')
    source_revision=symmetry.source_revision(path)
    current_configuration=configuration(entry)
    draft=bool(server.draft)
    current_engine_revision=engine_revision()
    fields={'geometry':build_recipe_identity(source_revision,current_configuration,draft,current_engine_revision),
            'source_revision':source_revision,'reference':reference,'configuration':current_configuration,
            'alignment':state['alignment'],'tolerance_mm':state['tolerance_mm'],'region':region,'draft':draft,
            'engine_revision':current_engine_revision}
    fields['token']=hashlib.sha256(json.dumps(fields,sort_keys=True,allow_nan=False).encode()).hexdigest()
    return fields


def freshness(saved, current):
    fields=('geometry','source_revision','engine_revision','reference','configuration','alignment','tolerance_mm','region','draft')
    changed=[key for key in fields if saved.get(key)!=current.get(key)]
    return {'status':'stale' if changed else 'current','changed':changed}


def save(server,path,entry,state,name):
    if not isinstance(name,str) or not name.strip() or len(name)>120 or any(ord(c)<32 for c in name):
        raise ValueError('give the inspection a one-line name of at most 120 characters')
    state=view_state(state)
    item={'kind':'nurb_inspection_setup','schema_version':1,'id':uuid.uuid4().hex,'name':name.strip(),'part':path.stem,
          'created_at':datetime.now(timezone.utc).isoformat(),'view':state,'identity':identity(server,path,entry,state)}
    item['configuration']=item['identity']['configuration']
    item['region']=item['identity']['region']
    item['verification']=verification(entry,state)
    if state['section_source']=='verified':
        selected=verified_sections(entry,item)
        if item['verification']['precise'].get('request_id')!=selected.get('verification_request_id'):
            raise ValueError('verification changed while saving the setup; save it again')
    destination=setup_path(server.root,item['id'])
    temporary=destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(item,indent=2,allow_nan=False)+'\n')
    temporary.replace(destination)
    return item


def verification(entry,state):
    target=entry.get('target') or {}
    matching=not target.get('stale') and state['alignment']==(target.get('transform') or compare.IDENTITY) and state['tolerance_mm']==target.get('tolerance_mm',.1)
    output={'comparison':{'status':'unknown','reason':'No measurements for this view alignment and tolerance.'},
            'precise':{'status':'unknown'},'symmetry':{'status':'unknown'},'features':[]}
    if matching:
        if target.get('metrics'): output['comparison']={'status':'measured','result':copy.deepcopy(target['metrics'])}
        precise=target.get('verification') or {}
        if precise.get('token')==entry.get('token'): output['precise']=copy.deepcopy(precise)
        output['features']=copy.deepcopy(target.get('feature_evidence',[]))
        symmetric=entry.get('symmetry') or {}
        if symmetric.get('token')==entry.get('token'): output['symmetry']=copy.deepcopy(symmetric)
    output['printability']={'findings':copy.deepcopy(entry.get('findings')),'separate_from_reconstruction':True}
    return output


def verified_sections(entry,item):
    """Never substitute preview contours for an explicitly verified setup."""
    region=item.get('region')
    target=entry.get('target') or {}; precise=target.get('verification') or {}
    current_region=next((value for value in target.get('regions',[]) if value.get('name')==(region or {}).get('name')),None)
    if (not region or not region.get('feature',{}).get('sections') or target.get('stale')
            or precise.get('status')!='measured' or precise.get('token')!=entry.get('token')
            or current_region!=region or item['view']['alignment']!=target.get('transform')
            or item['view']['tolerance_mm']!=target.get('tolerance_mm')
            or item['view'].get('verification_policy')!=verification_policy(precise.get('provenance') or {})):
        raise ValueError('Verified section evidence is unavailable for this setup and policy. Run precise verification with its saved policy in the viewer, then capture again. Preview contours were not substituted.')
    matches=[result for result in (precise.get('metrics') or {}).get('feature_evidence',[]) if result.get('id')==region['feature']['id']]
    current_record=next((result for result in target.get('feature_evidence',[]) if result.get('id')==region['feature']['id']),{})
    if (len(matches)!=1 or matches[0].get('verification_request_id')!=precise.get('request_id')
            or matches[0].get('identity')!=current_record.get('identity')):
        raise ValueError('Verified section identity changed; inspect the current feature again')
    return copy.deepcopy(matches[0])


def sections(server,entry,item):
    if item['view'].get('section_source','preview')=='verified':
        return verified_sections(entry,item)
    region=item.get('region')
    if not region or not region.get('feature',{}).get('sections'):
        return None
    cad, components=server._comparison_meshes(entry)
    target=entry['target']; reference=server._target_mesh(target['file'],target.get('units'))['mesh'].copy()
    reference.apply_transform(compare._transform(item['view']['alignment']))
    cad,reference=feature_evidence.selected_meshes(entry['shape'],cad,reference,region,components)
    definitions=region['feature']['sections']
    return {'id':region['feature']['id'],'cad':feature_evidence.sections(cad,definitions),
            'reference':feature_evidence.sections(reference,definitions), 'source':'preview',
            'provenance':{'method':'Display mesh estimate; saved local frame and station definitions.','absolute_deflection_mm':None}}


def comparison(server, entry, item):
    target=entry.get('target') or {}
    if not target.get('file'):
        return None
    state=item['view']
    if state['alignment']==target.get('transform') and state['tolerance_mm']==target.get('tolerance_mm') and target.get('metrics') and not target.get('stale'):
        return target['metrics']
    from .meshing import display_provenance
    cad,components=server._comparison_meshes(entry)
    reference=server._target_mesh(target['file'],target.get('units'))['mesh']
    return compare.against(entry['shape'],reference,tolerance_mm=state['tolerance_mm'],transform=state['alignment'],
                           regions=[item['region']] if item.get('region') else [],part_mesh=cad,component_meshes=components,
                           provenance=display_provenance(server.tolerance))


def png_bytes(encoded):
    try:
        body=base64.b64decode(encoded,validate=True)
    except (TypeError,ValueError) as exc:
        raise ValueError('capture image data is invalid; capture the view again') from exc
    if len(body)<33 or len(body)>48*1024*1024 or body[:8]!=b'\x89PNG\r\n\x1a\n' or body[12:16]!=b'IHDR':
        raise ValueError('capture must contain a PNG image no larger than 48 MiB')
    width,height=struct.unpack('>II',body[16:24])
    if not 1<=width<=8192 or not 1<=height<=8192:
        raise ValueError('capture image dimensions exceed the 8192 pixel limit')
    return body


def report_text(evidence):
    item=evidence['setup']; current=evidence['current_identity']; state=item['view']
    name=re.sub(r'[\\`*\[\]<>#]','',item['name'])
    lines=[f'# Inspection: {name}', '', f"Part: `{item['part']}`. Capture: {evidence['captured_at']}. Saved setup: **{evidence['freshness']['status']}**.", '',
           f"Mode: {state['mode']}. Tolerance: {state['tolerance_mm']:g} mm. Configuration: {json.dumps(current['configuration'],ensure_ascii=False)}.", '',
           f"Changed since the setup was saved: {', '.join(evidence['freshness']['changed']) or 'none'}.", '',
           f"Build recipe: `{current['geometry']}`. Model source revision: `{current['source_revision']}`.", '',
           f"Reference: `{(current['reference'] or {}).get('file','none')}`. Content identity: `{(current['reference'] or {}).get('content','none')}`.", '',
           '![Captured inspection](view.png)', '',
           'The image shows the captured view. Evidence includes the exact camera, alignment, configuration, visibility and section settings. `model.glb` and `reference.glb`, when present, are the display meshes; they are not precision certificates.', '',
           'Verification results below belong to the current capture inputs. Results retained in the saved setup are historical and must not be treated as current when its status is stale.', '']
    verification=evidence['verification']
    for key in ('comparison','precise','symmetry'):
        result=verification[key]; lines.append(f"- {key.capitalize()}: {result.get('status','unknown')}.")
    metrics=verification['comparison'].get('result') or {}
    if item['view'].get('region'):
        name=item['view']['region']; lines+=['', f'Comparison region: {name}.']
        metrics=next((region for region in metrics.get('inspection_regions',[]) if region['name']==name),{})
    if metrics:
        lines+=['', '| Surface | p95 mm | Sampled max mm | Within tolerance |', '| --- | ---: | ---: | ---: |']
        for key,label in (('part','CAD to reference'),('target','Reference to CAD')):
            row=metrics.get(key) or {}
            fmt=lambda value: f'{value:.4g}' if isinstance(value,(int,float)) else 'unknown'
            within=row.get('within_tolerance')
            coverage=f'{100*within:.1f}%' if isinstance(within,(int,float)) else 'unknown'
            lines.append(f"| {label} | {fmt(row.get('p95'))} | {fmt(row.get('sampled_max',row.get('max')))} | {coverage} |")
    for index,cut in enumerate((evidence.get('local_sections') or {}).get('cad',[])):
        if f'sections/{index+1:03d}.svg' not in evidence['images']: continue
        lines+=['', f"[Local section {index+1}, offset {cut.get('offset_mm',0):g} mm](sections/{index+1:03d}.svg)"]
    if evidence.get('local_sections'):
        local=evidence['local_sections']; provenance=local.get('provenance') or {}
        lines+=['',f"Local section source: **{local.get('source','preview')}**. Verification request: `{local.get('verification_request_id','none')}`. Requested absolute deflection: {provenance.get('absolute_deflection_mm','unknown')} mm. This is not a measured geometric error bound."]
    lines+=['', 'Printability and reconstruction accuracy are separate findings. Surface sampling cannot certify physical fit.', '',
            f"Reproduce in this project: `nurb inspection {item['part']} --render {item['id']}`. If inputs changed, add `--allow-stale` to render the new inputs with the saved view; the new report will remain marked stale."]
    return '\n'.join(lines)+'\n'


def bundle(server,entry,item,current,png,local_sections=None,section_images=None,display_metrics=None,verification_snapshot=None):
    packet=copy.deepcopy(verification_snapshot) if verification_snapshot is not None else verification(entry,item['view'])
    if (local_sections or {}).get('source')=='verified' and (packet['precise'].get('status')!='measured'
            or packet['precise'].get('request_id')!=local_sections.get('verification_request_id')):
        raise ValueError('capture sections and verification packet do not match; capture again')
    evidence={'kind':'nurb_inspection_evidence','schema_version':1,'captured_at':datetime.now(timezone.utc).isoformat(),
              'setup':item,'current_identity':current,'freshness':freshness(item['identity'],current),
              'verification':packet,'local_sections':local_sections,
              'images':['view.png'], 'asset_frames':{'model.glb':{'units':'mm'},'reference.glb':{'scale_to_mm':(current.get('reference') or {}).get('display_scale',1),'transform_to_part_mm':item['view']['alignment']}}}
    if display_metrics is not None:
        evidence['verification']['comparison']={'status':'measured','result':display_metrics}
    images=section_images or []
    if len(images)>192:
        raise ValueError('capture at most 192 local section images')
    # Only the viewer's inert path/text subset is accepted as an SVG attachment.
    import xml.etree.ElementTree as ET
    safe=[]
    for value in images:
        if not isinstance(value,str) or len(value)>4*1024*1024:
            raise ValueError('a local section image is too large')
        try: tree=ET.fromstring(value)
        except ET.ParseError as exc: raise ValueError('the local section image is invalid') from exc
        for node in tree.iter():
            if node.tag.rsplit('}',1)[-1] not in ('svg','polyline','text','rect') or any(k not in ('id','viewBox','role','aria-label','width','height','points','fill','stroke','stroke-width','x','y','font-size') or 'url(' in str(v).lower() for k,v in node.attrib.items()):
                raise ValueError('local section images must contain only inert contours and text')
        safe.append(value.encode())
    destination=directory(server.root,'build','inspection-evidence') / (item['id']+'-'+uuid.uuid4().hex+'.zip')
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('view.png',png)
        for index,image in enumerate(safe):
            filename=f'sections/{index+1:03d}.svg'; evidence['images'].append(filename); archive.writestr(filename,image)
        for filename,body in [('model.glb',entry.get('glb')),('reference.glb',entry.get('target_glb'))]:
            if body: archive.writestr(filename,body)
        archive.writestr('setup.json',json.dumps(item,indent=2,allow_nan=False))
        archive.writestr('evidence.json',json.dumps(evidence,indent=2,allow_nan=False))
        archive.writestr('report.md',report_text(evidence))
    temporary=destination.with_suffix('.tmp'); temporary.write_bytes(stream.getvalue()); temporary.replace(destination)
    return destination,evidence


def command(args):
    """List saved setups or render/export their view against the current project."""
    from contextlib import redirect_stdout
    import shutil
    import sys
    from . import builder, render
    from .cli import project_root
    from .server import Server
    root=project_root()
    try:
        if args.render or args.export:
            item=load(root,args.render or args.export)
            if args.part and item['part']!=args.part: raise ValueError('this setup belongs to a different part')
            path=(root/'parts'/f"{item['part']}.py").resolve()
            if path.parent!=(root/'parts').resolve() or not path.is_file(): raise ValueError('the inspection part is missing from this project')
            output=Path(args.output) if args.output else root/'build'/'inspection-evidence'/f"{item['id']}.png"
            if args.export and args.output: output=output.with_suffix('.png')
            shot={'part':path,'file':output,'view':'iso','size':item['view']['viewport'],'mode':item['view']['mode'],
                  'overrides':item['configuration']['parameters'],'setup':item,'allow_stale':args.allow_stale,'check':True,'marks':item['view']['pins']}
            with redirect_stdout(sys.stderr): render.snapshots(root,[shot],timeout=args.timeout*1000)
            context=shot['_inspection_context']
            destination,evidence=bundle(context['server'],context['entry'],item,context['identity'],output.read_bytes(),
                                        context['sections'],context['images'],context['metrics'])
            if args.export and args.output:
                requested=Path(args.output); requested.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(destination,requested); destination=requested
            result={'status':evidence['freshness']['status'],'image':str(output),'bundle':str(destination),'changed':evidence['freshness']['changed']}
        else:
            rows=[]
            for item in list_setups(root,args.part):
                path=(root/'parts'/f"{item['part']}.py").resolve()
                if path.parent!=(root/'parts').resolve(): raise ValueError('the saved part name is not safe')
                server=Server(root,draft=item['identity'].get('draft',False)); server.overrides[item['part']]=item['configuration']['parameters']
                with redirect_stdout(sys.stderr): entry=server.rebuild(path)
                try: status=freshness(item['identity'],identity(server,path,entry,item['view']))
                except (ValueError,KeyError,OSError) as exc: status={'status':'unknown','error':str(exc),'changed':[]}
                rows.append({'id':item['id'],'name':item['name'],'part':item['part'],**status})
            result={'setups':rows}
        if args.json: print(json.dumps(result,indent=2))
        elif 'setups' in result:
            for row in result['setups']: print(f"  {row['id']}  {row['part']}: {row['name']}  [{row['status']}]")
            if not result['setups']: print('  No saved inspections. Use Compare > Saved inspections and capture in the viewer.')
        else: print(f"  {result['status']}: {result['image']}\n  evidence: {result['bundle']}")
    except (ValueError,OSError,builder.BuildError) as exc:
        if args.json: print(json.dumps({'status':'unknown','error':str(exc)}))
        else: print(f'  inspection: {exc}',file=sys.stderr)
        raise SystemExit(2) from exc


def save_artifact(root, message):
    """Save a bounded evidence export for desktop webviews without anchor downloads."""
    filename=message.get('filename')
    types={'application/json':'.json','image/svg+xml':'.svg','image/png':'.png','application/zip':'.zip'}
    suffix=types.get(message.get('mime'))
    if not isinstance(filename,str) or len(filename)>180 or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._ -]*',filename) or not suffix or not filename.endswith(suffix):
        raise ValueError('choose a simple JSON, SVG, PNG, or ZIP evidence filename without directories')
    data=message.get('data'); encoding=message.get('encoding','utf8')
    if not isinstance(data,str) or len(data)>48*1024*1024:
        raise ValueError('evidence exports must be smaller than 48 MiB')
    try:
        if encoding=='base64': body=base64.b64decode(data,validate=True)
        elif encoding=='utf8': body=data.encode('utf8')
        else: raise ValueError('choose UTF-8 text or base64 for evidence data')
    except (ValueError,UnicodeError) as exc:
        raise ValueError('the evidence data could not be decoded; export it again') from exc
    if suffix=='.json':
        try: json.loads(body)
        except (ValueError,UnicodeError) as exc: raise ValueError('the evidence JSON is invalid; export it again') from exc
    elif suffix=='.svg':
        import xml.etree.ElementTree as ET
        try: tree=ET.fromstring(body)
        except ET.ParseError as exc: raise ValueError('the section image is invalid; export it again') from exc
        for node in tree.iter():
            if node.tag.rsplit('}',1)[-1] not in ('svg','g','polyline','path','text','rect','metadata','title','desc') or any(key.lower().startswith('on') or key.rsplit('}',1)[-1] in ('href','style') or 'url(' in value.lower() for key,value in node.attrib.items()):
                raise ValueError('section evidence must contain inert geometry and text only')
    elif suffix=='.png': png_bytes(base64.b64encode(body).decode())
    elif suffix=='.zip' and not zipfile.is_zipfile(io.BytesIO(body)):
        raise ValueError('the evidence bundle is not a ZIP file; capture it again')
    destination=directory(root,'build','evidence-exports')/(Path(filename).stem+'-'+uuid.uuid4().hex+suffix)
    temporary=destination.with_suffix('.tmp');temporary.write_bytes(body);temporary.replace(destination)
    return destination
