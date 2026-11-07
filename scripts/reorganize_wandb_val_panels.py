#!/usr/bin/env python3
"""Batch reorganize saved W&B run/project layouts without rewriting run history.

Default is a read-only remote preview with local before/after backups. --apply
updates discovered existing view IDs only; it never invents a run-page view ID.
The graph API for run views varies by server: discovery/coverage are reported,
and absent independent run layouts are not claimed to have been edited.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
from numbers import Real
import re
import sys
from urllib.parse import parse_qs, unquote, urlparse

DEFAULT_CONFIG = Path('/home/czh23/analysis/2026-11/2026-11-03/wandb_val_sections/config.json')
DETECTION_PREFIXES = ('val/detection/', 'detection/val/', 'detection/')
EPOCH_PREFIXES = ('val/epoch/', 'val/epcoh/')


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def metric_group(key):
    for prefix in DETECTION_PREFIXES:
        if key.startswith(prefix):
            return 'detection', key[len(prefix):]
    for prefix in EPOCH_PREFIXES:
        if key.startswith(prefix):
            return 'val/epoch', key[len(prefix):]
    return None, key


def metric_aliases(key):
    group, suffix = metric_group(key)
    if group == 'detection':
        return ['val/detection/' + suffix, 'detection/val/' + suffix]
    return [key]


def project_address(address):
    parsed = urlparse(address if '://' in address else 'https://wandb.ai/' + address)
    parts = [unquote(x) for x in parsed.path.split('/') if x]
    if len(parts) < 2:
        raise ValueError('Project address must be ENTITY/PROJECT or a full W&B URL')
    return parts[0], parts[1]


def workspace_names(addresses):
    names = set()
    for address in addresses:
        raw = parse_qs(urlparse(address).query).get('nw', [])
        if not raw:
            raise ValueError('An explicit workspace URL must include ?nw=...')
        value = raw[0]
        names.update((value, f'nw-{value}-v'))
        if value.startswith('nw-') and value.endswith('-v'):
            names.add(value[3:-2])
    return names


def panel_metrics(panel):
    config = panel.get('config', {})
    metrics = config.get('metrics', [])
    if isinstance(metrics, str):
        metrics = [metrics]
    return [m for m in metrics if isinstance(m, str)]


def new_panel(metric, section_name, x_axis, template=None):
    panel = copy.deepcopy(template) if template else dict(
        viewType='Run History Line Plot', layout=dict(x=0, y=0, w=8, h=6), config={})
    panel['__id__'] = 'valsplit-' + sha([section_name, metric_group(metric)[1]])[:16]
    panel['isAuto'] = False
    panel['config']['metrics'] = metric_aliases(metric)
    if section_name in ('detection', 'val/epoch'):
        panel['config']['xAxis'] = x_axis
    elif not template:
        panel['config']['xAxis'] = 'iteration' if '/iteration/' in metric or metric.startswith('adam/') else x_axis
    panel['config'].setdefault('chartTitle', metric_group(metric)[1])
    return panel


def new_section(name):
    return dict(__id__='section-' + sha(name)[:16], name=name, type='flow',
        isOpen=name != 'detection', isPanelsAuto=False,
        flowConfig=dict(snapToColumns=True, columnsPerPage=3, rowsPerPage=2),
        panels=[], localPanelSettings=dict(xAxis='iteration' if name=='adam' else 'epoch'))


def repair_adam_spec(spec, available_metrics):
    """Restore missing Adam charts only; retain every other layout field."""
    result=copy.deepcopy(spec);host=result.get('section',result)
    bank=host.get('panelBankConfig')
    if not isinstance(bank,dict) or not isinstance(bank.get('sections',[]),list):
        raise ValueError('Unsupported layout: no section.panelBankConfig.sections')
    sections=bank.setdefault('sections',[])
    represented={m for section in sections for panel in section.get('panels',[]) for m in panel_metrics(panel)}
    keys={m for m in available_metrics if m.startswith('adam/')}
    missing=keys-represented
    if missing:
        section=next((s for s in sections if s.get('name')=='adam'),None)
        if section is None:section=new_section('adam');sections.append(section)
        section.setdefault('panels',[]).extend(new_panel(m,'adam','iteration') for m in sorted(missing))
        section['isPanelsAuto']=False;section.pop('defaultName',None)
        section['isOpen']=True
        section.setdefault('localPanelSettings',{}).setdefault('xAxis','iteration')
    return result,dict(changed=result!=spec,repair_adam_only=True,
        available_adam_metrics=len(keys),added_adam_charts=len(missing),
        section_counts={s.get('name',''):len(s.get('panels',[])) for s in sections},
        reason='only missing Adam charts restored; all non-Adam layout state retained')


def transform_spec(spec, available_metrics, x_axis='epoch'):
    """Materialize automatic panels, preserve other custom panels, split Val.

Freezing automatic layout is necessary: otherwise the app regenerates the old
val/detection panels on its next save. All known scalar metrics are materialized
before that switch, rather than dropping other automatic chart families.
    """
    result = copy.deepcopy(spec)
    host = result.get('section', result)
    bank = host.get('panelBankConfig')
    if not isinstance(bank, dict) or not isinstance(bank.get('sections', []), list):
        raise ValueError('Unsupported layout: no section.panelBankConfig.sections')
    sections = bank.get('sections', [])
    target_panels = {}; represented = set(); kept = []; moved = Counter()
    for section in sections:
        section = copy.deepcopy(section)
        remaining = []
        for panel in section.get('panels', []):
            metrics = panel_metrics(panel)
            target = [m for m in metrics if metric_group(m)[0]]
            other = [m for m in metrics if not metric_group(m)[0]]
            if not target:
                remaining.append(panel); represented.update(metrics)
                continue
            if other:
                residual = copy.deepcopy(panel)
                residual['config']['metrics'] = other
                remaining.append(residual); represented.update(other)
            if panel.get('viewType') != 'Run History Line Plot':
                raise ValueError('A non-line custom validation panel needs manual review; this view is left unchanged')
            for metric in target:
                group, suffix = metric_group(metric)
                target_panels.setdefault((group, suffix), (metric, panel))
                moved[group] += 1
        section['panels'] = remaining
        # Empty generic Val banks disappear; unrelated sections/settings survive.
        if remaining or section.get('name') not in ('val', 'validation', 'Validation', 'val/epcoh'):
            kept.append(section)
    all_metrics = set(available_metrics)
    for metric in all_metrics:
        group, suffix = metric_group(metric)
        if group:
            target_panels.setdefault((group, suffix), (metric, None))
    if not target_panels:
        return result, dict(changed=False, reason='no validation/detection scalar metrics')
    by_name = {}
    for section in kept:
        by_name.setdefault(section.get('name', ''), section)
    # Preserve custom section properties when they already exist.
    for name in ('val/epoch', 'detection'):
        if name not in by_name:
            section = new_section(name); kept.append(section); by_name[name] = section
    for (group, suffix), (metric, template) in sorted(target_panels.items()):
        section = by_name[group]
        section['panels'].append(new_panel(metric, group, x_axis, template))
        represented.update(metric_aliases(metric))
    # Automatic panels may be absent from saved JSON. Rebuild the remaining
    # available numeric metric families too, so disabling auto-generation does
    # not discard train/lr/optimizer charts.
    added_other = 0
    for metric in sorted(all_metrics - represented):
        if metric.startswith('_') or metric in ('epoch', 'iteration', 'global_step') or metric_group(metric)[0]:
            continue
        name = metric.split('/', 1)[0]
        if name == 'val':name = 'val/other'
        if name not in by_name:
            section = new_section(name); kept.append(section); by_name[name] = section
        by_name[name]['panels'].append(new_panel(metric, name, x_axis)); added_other += 1
    overrides = bank.get('panelConfigOverrides', {})
    placements = bank.get('panelPlacementOverrides', {})
    for section in kept:
        if section.get('name') in ('val/epoch', 'detection'):
            section['isOpen'] = section['name'] == 'val/epoch'
            section.setdefault('localPanelSettings', {})['xAxis'] = x_axis
        section['isPanelsAuto'] = False
        section.pop('defaultName', None)
        for panel in section.get('panels', []):
            metrics = panel_metrics(panel)
            identifier = panel.get('__id__')
            # Preserve opaque UI customizations on untouched panels. Overrides
            # for moved panels must not undo the new metrics/placement.
            if identifier and identifier.startswith('valsplit-') and any(metric_group(m)[0] for m in metrics):
                keys = [identifier, '__custom_panel__:' + identifier, *metrics]
                for key in keys:
                    if isinstance(overrides, dict):overrides.pop(key, None)
                    if isinstance(placements, dict):placements.pop(key, None)
            panel['isAuto'] = False
    # Old target-panel override keys can refer to removed IDs; prune only those
    # identifiable target metric overrides, retaining all unrelated UI state.
    for mapping in (overrides, placements):
        if isinstance(mapping, dict):
            for key in list(mapping):
                if metric_group(key)[0]:mapping.pop(key)
    bank['sections'] = kept
    if isinstance(overrides, dict):bank['panelConfigOverrides'] = overrides
    if isinstance(placements, dict):bank['panelPlacementOverrides'] = placements
    host.setdefault('settings', {})['shouldAutoGeneratePanels'] = False
    if isinstance(host.get('workspaceSettings'), dict):
        host['workspaceSettings']['shouldAutoGeneratePanels'] = False
    return result, dict(changed=result != spec,
        section_counts={s['name']:len(s.get('panels', [])) for s in kept},
        detection_charts=sum(g == 'detection' for g, _ in target_panels),
        epoch_charts=sum(g == 'val/epoch' for g, _ in target_panels),
        materialized_other_charts=added_other, moved_existing=dict(moved),
        auto_generation=False)


def graphql(api, query, variables):
    service = getattr(api, '__dict__', {}).get('_service_api')
    if service is not None and hasattr(service, 'execute_graphql'):
        return service.execute_graphql(query, variables=variables)
    from wandb_gql import gql
    return api.client.execute(gql(query), variable_values=variables)


VIEWS_QUERY = '''query Layouts($entity:String!,$project:String!,$kind:String,$after:String,$name:String) {
  project(name:$project,entityName:$entity) {
    allViews(viewType:$kind,viewName:$name,first:50,after:$after) {
      edges { node { id name displayName description spec updatedAt user { username } } }
      pageInfo { endCursor hasNextPage }
    }
  }
}'''


def views(api, entity, project, kind, name=None):
    cursor = None
    while True:
        data = graphql(api, VIEWS_QUERY, dict(entity=entity, project=project, kind=kind, after=cursor, name=name))
        if data.get('project') is None:
            raise ValueError(f'Project not accessible: {entity}/{project}')
        connection = data['project']['allViews']
        for edge in connection.get('edges', []):yield edge['node']
        page = connection.get('pageInfo', {})
        if not page.get('hasNextPage'):break
        next_cursor = page.get('endCursor')
        if not next_cursor or next_cursor == cursor:raise RuntimeError('View pagination did not advance')
        cursor = next_cursor


def decode_spec(view):
    raw = view['spec']
    return json.loads(raw) if isinstance(raw, str) else copy.deepcopy(raw)


def read_current(api, record):
    matches = [v for v in views(api, record['entity'], record['project'], record['view_type'], record['view']['name'])
               if v['id'] == record['view']['id']]
    if len(matches) != 1:
        raise RuntimeError('Cannot uniquely re-read existing view ID; refusing to write')
    return matches[0]


def write_view(api, record, spec):
    view = record['view']
    query = '''mutation SetLayout($id:ID!,$entity:String!,$project:String!,$kind:String!,
      $name:String!,$display:String,$description:String,$spec:String!) {
      upsertView(input:{id:$id,entityName:$entity,projectName:$project,type:$kind,name:$name,
        displayName:$display,description:$description,spec:$spec,createdUsing:WANDB_SDK}) {
        view { id name }
      }
    }'''
    result = graphql(api, query, dict(id=view['id'], entity=record['entity'], project=record['project'],
        kind=record['view_type'], name=view['name'], display=view.get('displayName'),
        description=view.get('description', ''), spec=json.dumps(spec)))
    if result['upsertView']['view']['id'] != view['id']:
        raise RuntimeError('Server returned a different view ID; inspect the backup before proceeding')


def run_references(view, run_ids):
    """Recognize actual run identifiers in metadata; never derive a view name."""
    found = set()
    def scan(value):
        if isinstance(value, dict):
            for v in value.values():scan(v)
        elif isinstance(value, list):
            for v in value:scan(v)
        elif isinstance(value, str):
            if value in run_ids:found.add(value)
            else:
                for part in re.split(r'[^a-zA-Z0-9]+', value):
                    if part in run_ids:found.add(part)
                if len(value) < 512:
                    try:
                        decoded = base64.b64decode(value, validate=True).decode()
                        if decoded.startswith(('Run:v1:', 'BucketType:v1:')):scan(decoded)
                    except (ValueError, UnicodeDecodeError):pass
    scan(view['name']);scan(decode_spec(view))
    return found


def metric_keys(run):
    values = getattr(run, 'summary_metrics', None)
    if values is None:values = run.summary._json_dict
    keys={k for k,value in values.items() if isinstance(value,Real) and not isinstance(value,bool)}
    # A metric can exist in history without a final summary value. Run listing
    # metadata normally includes this catalog; never fetch full history here.
    history=getattr(run,'__dict__',{}).get('_attrs',{}).get('historyKeys',{})
    if isinstance(history,str):
        try:history=json.loads(history)
        except ValueError:history={}
    catalog=history.get('keys',{}) if isinstance(history,dict) else {}
    if isinstance(catalog,dict):
        for key,description in catalog.items():
            if not isinstance(description,dict):continue
            types={description.get('type')} if isinstance(description.get('type'),str) else set()
            types.update(v.get('type') for v in (description.get('typeCounts') or []) if isinstance(v,dict) and isinstance(v.get('type'),str))
            if types & {'number','numeric','int','float'}:keys.add(key)
    return keys


def backup_record(folder, record, before, after):
    stem = sha([record['entity'], record['project'], record['view']['id']])[:20]
    directory = folder/stem;directory.mkdir(parents=True, exist_ok=True)
    record.update(before_sha256=sha(before), after_sha256=sha(after), directory=str(directory), status='preview')
    save(directory/'record.json', record)
    save(directory/'before.json', before);save(directory/'after.json', after)
    return directory


def runs_with_metrics(api, path, **kwargs):
    """Batch summary/history-key metadata instead of fetching each run lazily."""
    if 'lazy' in inspect.signature(api.runs).parameters:
        kwargs['lazy'] = False
    return api.runs(path, **kwargs)


def process(api, config, output, apply=False):
    owner = config.get('username') or api.viewer.username
    print(f'[layout-owner] username={owner!r} (layout creator, not project entity)', flush=True)
    since = config.get('since', '2026-09-01')
    until = config.get('until')
    date_filter = {'$gte': since}
    if config.get('until'):date_filter['$lt'] = config['until']
    scope = config.get('scope', 'both')
    names = workspace_names(config.get('workspace_urls', []))
    regex = re.compile(config.get('view_name_regex', '.*'))
    results = [];projects = [];seen = set()
    for address in config['project_urls']:
        entity, project = project_address(address)
        print(f'[runs] {entity}/{project} createdAt >= {since}' + (f' and < {until}' if until else ' (no end date)'), flush=True)
        selected = []
        for run in runs_with_metrics(api, f'{entity}/{project}', filters={'createdAt':date_filter}, per_page=50):
            if config.get('run_name_regex') and not re.search(config['run_name_regex'], run.name):continue
            selected.append(dict(id=run.id, name=run.name, metrics=sorted(metric_keys(run))))
        save(output/(sha([entity,project])[:12]+'_selected_runs.json'), selected)
        by_run = {r['id']:set(r['metrics']) for r in selected}
        if not by_run:
            projects.append(dict(entity=entity, project=project, selected_runs=0,
                reason='no runs in date/name selection'))
            print('[skip] no runs in date/name selection', flush=True);continue
        print(f'[runs] selected={len(by_run)}; discovering saved layouts', flush=True)
        union = set().union(*by_run.values())
        project_other_metrics=set()
        # A project workspace is shared even if its run table currently selects
        # a handful of old runs. Freezing it using only those runs hides Adam
        # and other metric families first logged outside the date/name filter.
        if 'project-view' in config.get('view_types',['run-view','project-view']):
            project_run_count=0
            for run in runs_with_metrics(api, f'{entity}/{project}',per_page=50):
                project_run_count+=1
                project_other_metrics.update(k for k in metric_keys(run) if not metric_group(k)[0])
                if project_run_count%25==0:
                    print(f'[workspace-metrics] scanned={project_run_count}; '
                          f'adam={sum(k.startswith("adam/") for k in project_other_metrics)}',flush=True)
            save(output/(sha([entity,project])[:12]+'_project_nonval_metrics.json'),sorted(project_other_metrics))
            print(f'[workspace-metrics] all project runs={project_run_count}; nonval={len(project_other_metrics)}; '
                  f'adam={sum(k.startswith("adam/") for k in project_other_metrics)}',flush=True)
        matched_runs = set();counts = Counter();shared = 0;discovery_errors = {}
        inventory = []; skipped = Counter()
        for kind in config.get('view_types', ['run-view', 'project-view']):
            iterator = iter(views(api, entity, project, kind))
            while True:
                try:view = next(iterator)
                except StopIteration:break
                except Exception as exc:
                    discovery_errors[kind]=str(exc)
                    print(f'[discovery-error] {kind}: {exc}',file=sys.stderr,flush=True)
                    break
                counts[kind] += 1
                identity = (entity, project, view['id'])
                view_owner = (view.get('user') or {}).get('username')
                item = dict(id=view['id'], name=view['name'], display_name=view.get('displayName'),
                    view_type=kind, username=view_owner)
                inventory.append(item)
                def skip(reason):
                    item['status'] = 'skipped';item['reason'] = reason;skipped[reason] += 1
                    print(f'[skip-layout] {kind} {view["name"]}: {reason}; '
                          f'owner={view_owner!r}, requested_owner={owner!r}', flush=True)
                if identity in seen:skip('duplicate_view_id');continue
                if owner != '*' and view_owner not in (owner, None):skip('username_mismatch');continue
                if names and view['name'] not in names:skip('workspace_name_mismatch');continue
                if not regex.search(view.get('displayName') or view['name']):skip('view_name_regex_mismatch');continue
                references = run_references(view, set(by_run))
                is_shared = kind=='project-view' or not references
                if scope == 'runs' and is_shared:skip('scope_requires_run_reference');continue
                if scope == 'shared' and not is_shared:skip('scope_requires_shared_layout');continue
                seen.add(identity)
                metrics = union if is_shared else set().union(*(by_run[r] for r in references))
                if kind=='project-view':metrics=metrics|project_other_metrics
                before = decode_spec(view)
                record = dict(entity=entity, project=project, view_type=kind, view=view,
                    run_ids=sorted(references), shared_layout=is_shared,
                    note='Shared layouts may affect runs outside the date filter too' if is_shared else '')
                try:
                    if config.get('repair_adam_only'):
                        after,change=repair_adam_spec(before,metrics)
                    else:
                        after, change = transform_spec(before, metrics, config.get('x_axis', 'epoch'))
                    record['change'] = change
                    directory = backup_record(output, record, before, after)
                    if apply and change['changed']:
                        if sha(decode_spec(read_current(api, record))) != sha(before):
                            raise RuntimeError('Layout changed since it was read; refusing to overwrite concurrent edits')
                        write_view(api, record, after)
                        record['status'] = 'written_unverified';save(directory/'record.json',record)
                        if sha(decode_spec(read_current(api, record))) != sha(after):
                            raise RuntimeError('Read-back layout differs from proposed spec; inspect backup')
                        record['status'] = 'applied_verified'
                    elif apply:record['status'] = 'unchanged'
                    save(directory/'record.json', record)
                    if not is_shared:matched_runs.update(references)
                    shared += int(is_shared)
                    print(f'[{record["status"]}] {kind} {view.get("displayName") or view["name"]}: '
                          f'adam added={change.get("added_adam_charts",0)} '
                          f'detection={change.get("detection_charts",0)} val/epoch={change.get("epoch_charts",0)}',flush=True)
                except Exception as exc:
                    record['status']='error';record['error']=str(exc)
                    print(f'[error] {view["name"]}: {exc}',file=sys.stderr,flush=True)
                item['status'] = record['status']
                if record.get('error'):item['error'] = record['error']
                results.append(record)
        missing = sorted(set(by_run)-matched_runs)
        projects.append(dict(entity=entity,project=project,selected_runs=len(by_run),views_discovered=dict(counts),
            explicit_run_layouts_matched=len(matched_runs),shared_layouts_processed=shared,
            discovery_errors=discovery_errors,
            requested_username=owner, layout_inventory=inventory, skipped_layouts=dict(skipped),
            runs_without_discovered_explicit_layout=missing,
            note='Runs may inherit a shared layout; inheritance is not verified by this script. No per-run success is claimed for these IDs.'))
        print(f'[coverage] {entity}/{project}: selected runs={len(by_run)}, '
              f'explicit layouts matched={len(matched_runs)}, shared layouts={shared}, '
              f'no discovered independent layout={len(missing)}. '
              'Shared-layout inheritance is not verified.',flush=True)
        if skipped['username_mismatch']:
            print('[hint] username filters the person who saved the layout, not the project entity/team. '
                  'Set username=null to use the logged-in account, or select an owner from layout_inventory '
                  'in report.json. No automatic expansion to other users is performed.', flush=True)
    report=dict(mode='apply' if apply else 'preview',projects=projects,
        views=[{k:v for k,v in r.items() if k!='view'} for r in results],
        errors=sum(r['status']=='error' for r in results),history_modified=False)
    save(output/'report.json',report)
    if not results:
        raise RuntimeError(f'No accessible matching saved layouts found; no changes were made. '
                           f'See {output}/report.json for layout owners and exact skip reasons.')
    if any(r['status']=='error' for r in results):raise RuntimeError(f'Some layouts failed; see {output}/report.json')
    print(f'Report/backups: {output}',flush=True)


def restore(api, folder, apply=False):
    count=0
    for path in sorted(folder.glob('*/record.json')):
        record=json.loads(path.read_text())
        if record.get('status') not in ('applied_verified','written_unverified'):continue
        before=json.loads((path.parent/'before.json').read_text())
        current=decode_spec(read_current(api,record))
        if sha(current)==record['before_sha256']:continue
        if sha(current)!=record['after_sha256']:
            raise RuntimeError(f'Concurrent layout changes prevent safe restore: {path}')
        if apply:
            write_view(api,record,before)
            if sha(decode_spec(read_current(api,record)))!=sha(before):raise RuntimeError('Restore read-back mismatch')
            record['status']='restored_verified';save(path,record)
        count+=1
        print(f'[{"restore" if apply else "restore-preview"}] {record["project"]}/{record["view"]["name"]}')
    print(f'{count} layout(s); no run history changes')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--project-url',action='append',help='Override configured projects; repeat for multiple projects')
    parser.add_argument('--since',help='Override createdAt start, e.g. 2026-09-01')
    parser.add_argument('--until',help='Override exclusive end date; use none to include later runs')
    parser.add_argument('--username',help='Layout creator username, not entity/team; use @me for logged-in account')
    parser.add_argument('--apply',action='store_true',help='Save the planned layout changes remotely')
    parser.add_argument('--repair-adam-only',action='store_true',help='Only add missing Adam charts; preserve val/detection and every other layout setting')
    parser.add_argument('--restore',type=Path,help='Preview/restore previously applied backups')
    parser.add_argument('--output-dir',type=Path,help='Local backup directory; otherwise a timestamped directory')
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    if args.project_url:config['project_urls']=args.project_url
    if args.since:config['since']=args.since
    if args.until:config['until']=None if args.until.lower()=='none' else args.until
    if args.repair_adam_only:config['repair_adam_only']=True
    if args.username is not None:config['username']=None if args.username=='@me' else args.username
    if config.get('scope','both') not in ('both','runs','shared'):parser.error('scope must be both/runs/shared')
    if not args.restore and any('YOUR_' in p for p in config.get('project_urls',[])):
        parser.error('Set your project_urls in config.json or pass --project-url ENTITY/PROJECT')
    import wandb
    kwargs=dict(timeout=int(config.get('timeout_seconds',60)))
    if config.get('api_base_url'):kwargs['overrides']={'base_url':config['api_base_url']}
    try:api=wandb.Api(**kwargs)
    except Exception as exc:
        parser.exit(2,f'W&B authentication failed: {exc}\nRun `python -m wandb login` in this environment; do not put an API key in config.json.\n')
    if args.restore:
        restore(api,args.restore,args.apply);return
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output=args.output_dir or args.config.parent/'backups'/stamp
    output.mkdir(parents=True,exist_ok=True)
    save(output/'config.json',config)
    process(api,config,output,args.apply)


if __name__=='__main__':main()
