"""Small websocket surface for saved inspection views and two-phase captures."""
import asyncio
import copy
import time
import uuid

from . import builder, inspection


def require_current(server,path,message):
    entry=server.state.get(path.stem) or {}
    if entry.get('shape') is None or entry.get('error') or message.get('token')!=entry.get('token'):
        raise ValueError('the displayed model changed or is not ready; wait for it and try again')
    snapshot=server._source_snapshot(path)
    inputs=server._build_inputs(server._build_sources(path,entry['shape'],snapshot),snapshot)
    overrides=server.overrides.get(path.stem) or {}
    if server.prints.get(path.stem)!=(inputs,repr(sorted(overrides.items())),entry.get('shape_id')):
        raise ValueError('model inputs changed; wait for the rebuild before saving or capturing')
    return entry


async def handle(server,path,message,client):
    response={'type':'inspection_result','name':path.stem,'action':message['type'],'request_id':message.get('request_id')}
    try:
        async with server.building:
            action=message['type']
            entry=server.state.get(path.stem) or {}
            if action=='artifact_save':
                destination=await asyncio.to_thread(inspection.save_artifact,server.root,message)
                response.update(path=str(destination),filename=destination.name)
            elif action=='inspection_list':
                items=await asyncio.to_thread(inspection.list_setups,server.root,path.stem)
                result=[]
                for item in items:
                    try:
                        current=await asyncio.to_thread(inspection.identity,server,path,entry,item['view'])
                        status=inspection.freshness(item['identity'],current)
                    except (ValueError,KeyError,OSError) as exc:
                        status={'status':'unknown','changed':[],'error':str(exc)}
                    result.append({**item,'freshness':status})
                response['setups']=result
            elif action=='inspection_sections':
                entry=require_current(server,path,message)
                item=inspection.load(server.root,message.get('id'))
                if item['part']!=path.stem: raise ValueError('select an inspection saved for this part')
                response['sections']=await asyncio.to_thread(inspection.sections,server,entry,item)
            elif action=='inspection_restore':
                item=inspection.load(server.root,message.get('id'))
                if item['part']!=path.stem: raise ValueError('select an inspection saved for this part')
                values=item['configuration']['parameters']
                draft=bool(item['identity'].get('draft')); draft_changed=server.draft!=draft
                server.draft=draft
                server.overrides[path.stem]=copy.deepcopy(values)
                for rebuild_path in builder.find_parts(server.root) if draft_changed else [path]:
                    server.queue.put_nowait(str(rebuild_path))
                response.update(setup=item,restoring=True,previous_token=entry.get('token'),draft=server.draft)
            elif action in ('inspection_save','inspection_prepare'):
                entry=require_current(server,path,message)
                state=inspection.view_state(message.get('view'))
                if action=='inspection_save' or not message.get('id'):
                    item=await asyncio.to_thread(inspection.save,server,path,entry,state,message.get('label'))
                else:
                    item=inspection.load(server.root,message['id'])
                    if item['part']!=path.stem: raise ValueError('select an inspection saved for this part')
                    if not inspection.same_view(item['view'],state): raise ValueError('the view changed after loading this setup; save it as a new setup before capturing')
                response.update(setup=item,token=entry['token'])
                if action=='inspection_prepare':
                    current=await asyncio.to_thread(inspection.identity,server,path,entry,state)
                    cuts=await asyncio.to_thread(inspection.sections,server,entry,item)
                    metrics=await asyncio.to_thread(inspection.comparison,server,entry,item)
                    tickets=getattr(server,'inspection_tickets',{})
                    tickets={key:value for key,value in tickets.items() if time.monotonic()-value['at']<120}
                    if len(tickets)>=4: raise ValueError('too many captures are pending; finish one or wait two minutes')
                    ticket=uuid.uuid4().hex
                    tickets[ticket]={'at':time.monotonic(),'entry':entry,'setup':item,'identity':current,'sections':cuts,'metrics':metrics}
                    server.inspection_tickets=tickets
                    response.update(ticket=ticket,sections=cuts,display_metrics=metrics,freshness=inspection.freshness(item['identity'],current))
            elif action=='inspection_capture':
                ticket=getattr(server,'inspection_tickets',{}).pop(message.get('ticket'),None)
                if not ticket or time.monotonic()-ticket['at']>120:
                    raise ValueError('the capture expired; capture the view again')
                entry=require_current(server,path,message)
                if entry is not ticket['entry']: raise ValueError('the model rebuilt during capture; try again')
                item=ticket['setup']
                current=await asyncio.to_thread(inspection.identity,server,path,entry,item['view'])
                if current!=ticket['identity']: raise ValueError('capture inputs changed; no evidence bundle was written')
                png=inspection.png_bytes(message.get('png'))
                destination,evidence=await asyncio.to_thread(inspection.bundle,server,entry,item,current,png,ticket['sections'],message.get('section_images'),ticket['metrics'])
                response.update(path=str(destination),file=destination.name,freshness=evidence['freshness'],setup_id=item['id'])
            else:
                raise ValueError('choose save, restore, list, or capture for an inspection')
    except (ValueError,TypeError,KeyError,OSError) as exc:
        response['error']=str(exc)
    await server.reply(client,response)
