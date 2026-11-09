"""Offline coverage of panel migration, round trips, and guarded remote writes."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

PATH=Path(__file__).resolve().parents[1]/'scripts/reorganize_wandb_val_panels.py'
spec=importlib.util.spec_from_file_location('wb_val_sections',PATH)
wb=importlib.util.module_from_spec(spec);spec.loader.exec_module(wb)


def panel(key, name='original'):
    return dict(__id__=name,viewType='Run History Line Plot',isAuto=True,
        config=dict(metrics=[key],xAxis='_step',smoothingWeight=.7),
        layout=dict(x=0,y=0,w=8,h=6))


def layout(panels=None):
    return dict(unknown_top={'preserve':True},section=dict(
        runSets=[dict(runFeed=dict(columnPinned={'Name':True}),filters={'run':'abc12345'})],
        settings={'shouldAutoGeneratePanels':True},workspaceSettings={'ref':'settings:1','shouldAutoGeneratePanels':True},
        panelBankConfig=dict(sections=[dict(name='val',panels=panels or [],isPanelsAuto=True)],
            panelConfigOverrides={'train/loss':{'keep':1}},panelPlacementOverrides={'train/loss':{'keep':2}})))


def sections(spec):return {s['name']:s for s in spec['section']['panelBankConfig']['sections']}


class PanelMigrationTests(unittest.TestCase):
    def test_offset_only_repair_preserves_history_layout_and_has_correct_axes(self):
        source=layout([panel('val/detection/F444W/recall')])
        source['section']['panelBankConfig']['sections'].append(dict(name='adam',panels=[panel('adam/test')]))
        metrics={'train/epoch/offset','val/epoch/offset','train/iteration/loss/offset','val/epoch/loss'}
        out,report=wb.repair_offset_spec(source,metrics)
        self.assertEqual(out['section']['panelBankConfig']['sections'][:2],source['section']['panelBankConfig']['sections'])
        for key in ('runSets','settings','workspaceSettings'):
            self.assertEqual(out['section'][key],source['section'][key])
        charts=sections(out)['center_offset']['panels']
        self.assertEqual(report['added_offset_charts'],3)
        for chart in charts:
            metric=chart['config']['metrics'][0]
            self.assertEqual(chart['config']['xAxis'],'iteration' if '/iteration/' in metric else 'epoch')
        again,report=wb.repair_offset_spec(out,metrics)
        self.assertEqual(again,out);self.assertFalse(report['changed'])

    def test_offset_repair_does_not_duplicate_existing_custom_charts(self):
        source=layout([panel('val/epoch/offset','my-offset')])
        out,report=wb.repair_offset_spec(source,{'val/epoch/offset'})
        self.assertEqual(out,source);self.assertFalse(report['changed'])

    def test_run_metadata_is_batched_when_sdk_supports_it(self):
        calls=[]
        def runs(path, *, lazy=True, **kwargs):
            calls.append((path,lazy,kwargs));return []
        wb.runs_with_metrics(types.SimpleNamespace(runs=runs),'me/proj',per_page=50)
        self.assertEqual(calls,[('me/proj',False,{'per_page':50})])

    def test_adam_only_repair_keeps_every_other_layout_field(self):
        source=layout([panel('val/detection/F444W/recall')])
        source['section']['panelBankConfig']['sections'].append(dict(
            name='adam',__id__='original-adam',isPanelsAuto=True,defaultName='adam',isOpen=False,panels=[]))
        out,report=wb.repair_adam_spec(source,{'adam/encoder/gradient_norm','val/epoch/loss'})
        self.assertEqual(sections(out)['val'],sections(source)['val'])
        for key in ('runSets','settings','workspaceSettings'):
            self.assertEqual(out['section'][key],source['section'][key])
        self.assertEqual(sections(out)['adam']['__id__'],'original-adam')
        self.assertEqual(sections(out)['adam']['panels'][0]['config']['xAxis'],'iteration')
        self.assertEqual(sections(out)['adam']['localPanelSettings']['xAxis'],'iteration')
        self.assertEqual(report['added_adam_charts'],1)
        again,report=wb.repair_adam_spec(out,{'adam/encoder/gradient_norm'})
        self.assertEqual(again,out);self.assertFalse(report['changed'])

    def test_adam_only_repair_does_not_duplicate_custom_or_hidden_charts(self):
        source=layout([panel('adam/encoder/gradient_norm','my-adam')])
        out,report=wb.repair_adam_spec(source,{'adam/encoder/gradient_norm'})
        self.assertEqual(out,source);self.assertFalse(report['changed'])

    def test_materialized_adam_axis_and_unrelated_overrides_survive_normal_migration(self):
        out,_=wb.transform_spec(layout(),{'val/epoch/loss','adam/encoder/parameter_norm'})
        p=sections(out)['adam']['panels'][0]
        self.assertEqual(p['config']['xAxis'],'iteration')
        bank=out['section']['panelBankConfig'];bank['panelConfigOverrides'][p['__id__']]={'logY':True}
        again,_=wb.transform_spec(out,{'val/epoch/loss','adam/encoder/parameter_norm'})
        self.assertEqual(again['section']['panelBankConfig']['panelConfigOverrides'][p['__id__']],{'logY':True})

    def test_history_numeric_keys_are_used_even_without_final_summary(self):
        run=types.SimpleNamespace(summary_metrics={'loss':2.,'bool':True})
        run._attrs={'historyKeys':json.dumps({'keys':{
            'adam/encoder/gradient_norm':{'type':'number'},
            'adam/mask_decoder/parameter_norm':{'typeCounts':[{'type':'number','count':2}]},
            'image':{'type':'image'},'null':{'typeCounts':None}}})}
        self.assertEqual(wb.metric_keys(run),{'loss','adam/encoder/gradient_norm','adam/mask_decoder/parameter_norm'})

    def test_project_view_visible_old_runs_do_not_hide_new_adam_metrics(self):
        source=layout([panel('val/epoch/loss')])
        source['section']['runSets'][0]['filters']={'run':'old12345'}
        source['section']['panelBankConfig']['sections'].append(dict(name='adam',isPanelsAuto=True,panels=[]))
        view=dict(id='workspace',name='saved',spec=json.dumps(source),user={'username':'me'})
        old=types.SimpleNamespace(id='old12345',name='old',summary_metrics={'val/epoch/loss':1.})
        new=types.SimpleNamespace(id='new12345',name='new',summary_metrics={'adam/encoder/gradient_norm':2.,'val/epoch/new_metric':3.})
        calls=[]
        def runs(*a,**kw):
            calls.append(kw)
            return [old] if 'filters' in kw else [old,new]
        api=types.SimpleNamespace(viewer=types.SimpleNamespace(username='me'),runs=runs)
        cfg=dict(project_urls=['me/proj'],view_types=['project-view'],until='2026-11-03')
        with tempfile.TemporaryDirectory() as tmp,patch.object(wb,'views',return_value=iter([view])):
            wb.process(api,cfg,Path(tmp))
            after=json.loads(next(Path(tmp).glob('*/after.json')).read_text())
            self.assertEqual(panel_metrics:=wb.panel_metrics(sections(after)['adam']['panels'][0]),['adam/encoder/gradient_norm'])
            targets={m for s in sections(after).values() for p in s['panels'] for m in wb.panel_metrics(p)}
            self.assertNotIn('val/epoch/new_metric',targets)
            report=json.loads((Path(tmp)/'report.json').read_text())['projects'][0]
            self.assertEqual(report['shared_layouts_processed'],1)
            self.assertEqual(report['explicit_run_layouts_matched'],0)
            self.assertEqual(calls[0]['filters']['createdAt']['$lt'],'2026-11-03')
            self.assertNotIn('filters',calls[1])

    def test_adam_repair_process_does_not_reorganize_validation_again(self):
        source=layout([panel('val/detection/F444W/recall')])
        view=dict(id='workspace',name='saved',spec=json.dumps(source),user={'username':'me'})
        run=types.SimpleNamespace(id='abc12345',name='new',summary_metrics={'adam/encoder/parameter_norm':1.})
        api=types.SimpleNamespace(viewer=types.SimpleNamespace(username='me'),runs=lambda *a,**kw:[run])
        with tempfile.TemporaryDirectory() as tmp,patch.object(wb,'views',return_value=iter([view])):
            wb.process(api,dict(project_urls=['me/proj'],view_types=['project-view'],repair_adam_only=True),Path(tmp))
            after=json.loads(next(Path(tmp).glob('*/after.json')).read_text())
            self.assertEqual(sections(after)['val'],sections(source)['val'])
            self.assertNotIn('detection',sections(after))
            self.assertEqual(len(sections(after)['adam']['panels']),1)

    def test_auto_metrics_are_materialized_and_old_new_detection_share_chart(self):
        source=layout()
        out,report=wb.transform_spec(source,{
            'val/detection/F115W/recall','detection/val/F115W/recall',
            'val/epoch/loss','train/iteration/loss','lr/epoch/default','epoch','_step'})
        s=sections(out)
        self.assertNotIn('val',s)
        self.assertEqual(len(s['detection']['panels']),1)
        self.assertEqual(s['detection']['panels'][0]['config']['metrics'],
            ['val/detection/F115W/recall','detection/val/F115W/recall'])
        self.assertEqual(len(s['val/epoch']['panels']),1)
        self.assertIn('train',s);self.assertIn('lr',s)
        self.assertEqual(s['train']['panels'][0]['config']['xAxis'],'iteration')
        self.assertFalse(out['section']['workspaceSettings']['shouldAutoGeneratePanels'])
        self.assertFalse(out['section']['settings']['shouldAutoGeneratePanels'])
        self.assertFalse(s['detection']['isOpen']);self.assertTrue(s['val/epoch']['isOpen'])
        self.assertEqual(out['section']['runSets'],source['section']['runSets'])
        self.assertEqual(out['unknown_top'],source['unknown_top'])
        self.assertTrue(source['section']['settings']['shouldAutoGeneratePanels'])

    def test_existing_custom_nonval_panels_and_ui_overrides_preserved(self):
        p=dict(__id__='image',viewType='Image',config={'arbitrary':{'value':4}})
        source=layout([panel('val/detection/F444W/precision'),p])
        out,_=wb.transform_spec(source,{'val/epoch/loss'})
        s=sections(out)
        self.assertEqual(s['val']['panels'][0]['config'],p['config'])
        bank=out['section']['panelBankConfig']
        self.assertEqual(bank['panelConfigOverrides']['train/loss'],{'keep':1})
        det=s['detection']['panels'][0]
        self.assertEqual(det['config']['smoothingWeight'],.7)
        self.assertEqual(det['config']['xAxis'],'epoch')

    def test_aliases_and_second_application_are_idempotent(self):
        for metric in ('val/detection/F115W/recall','detection/val/F115W/recall'):
            out,_=wb.transform_spec(layout([panel(metric)]),{metric,'val/epoch/loss'})
            again,report=wb.transform_spec(out,{metric,'val/epoch/loss'})
            self.assertEqual(again,out)
            self.assertFalse(report['changed'])

    def test_mixed_line_chart_keeps_unrelated_metrics(self):
        p=panel('val/epoch/loss');p['config']['metrics'].append('train/epoch/loss')
        out,_=wb.transform_spec(layout([p]),set())
        self.assertEqual(sections(out)['val']['panels'][0]['config']['metrics'],['train/epoch/loss'])
        self.assertEqual(sections(out)['val/epoch']['panels'][0]['config']['metrics'],['val/epoch/loss'])

    def test_non_line_custom_validation_panel_fails_without_mutating_input(self):
        p=panel('val/detection/F115W/recall');p['viewType']='Custom Chart'
        source=layout([p]);original=copy.deepcopy(source)
        with self.assertRaisesRegex(ValueError,'manual review'):wb.transform_spec(source,set())
        self.assertEqual(source,original)

    def test_target_section_settings_survive_and_typo_is_only_section_normalization(self):
        source=layout();source['section']['panelBankConfig']['sections']=[
            dict(name='detection',__id__='existing',localPanelSettings={'smoothingWeight':.6},panels=[panel('detection/val/F444W/recall')])]
        out,_=wb.transform_spec(source,{'val/epcoh/loss'})
        self.assertEqual(sections(out)['detection']['__id__'],'existing')
        self.assertEqual(sections(out)['detection']['localPanelSettings']['smoothingWeight'],.6)
        self.assertEqual(sections(out)['val/epoch']['panels'][0]['config']['metrics'],['val/epcoh/loss'])

    def test_existing_run_references_are_recognized_without_guessing_view_ids(self):
        import base64
        source=layout();source['section']['runSets'][0]['filters']={
            'id':base64.b64encode(b'Run:v1:abc12345:proj:entity').decode()}
        view=dict(name='a-private-view',spec=json.dumps(source))
        self.assertEqual(wb.run_references(view,{'abc12345','unmatched'}),{'abc12345'})
        self.assertEqual(wb.project_address('https://wandb.ai/myteam/myproject/runs/abc12345'),('myteam','myproject'))

    def test_dry_run_has_no_remote_writes_and_apply_verifies_readback(self):
        source=layout([panel('val/detection/F444W/recall')])
        view=dict(id='real-view-id',name='run-abc12345',displayName='My Run',spec=json.dumps(source),user={'username':'me'})
        run=types.SimpleNamespace(id='abc12345',name='run',summary_metrics={'val/epoch/loss':1.,'val/detection/F444W/recall':.5})
        api=types.SimpleNamespace(viewer=types.SimpleNamespace(username='me'),runs=lambda *a,**k:[run])
        cfg=dict(project_urls=['me/proj'],view_types=['run-view'],scope='both')
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(wb,'views',return_value=iter([view])),patch.object(wb,'write_view') as write:
                wb.process(api,cfg,Path(tmp)/'preview')
                write.assert_not_called()
            planned,_=wb.transform_spec(source,set(run.summary_metrics))
            with patch.object(wb,'views',return_value=iter([view])),patch.object(wb,'write_view') as write, \
                 patch.object(wb,'read_current',side_effect=[view,{**view,'spec':json.dumps(planned)}]):
                wb.process(api,cfg,Path(tmp)/'apply',True)
                write.assert_called_once()
                records=list((Path(tmp)/'apply').glob('*/record.json'))
                self.assertEqual(json.loads(records[0].read_text())['status'],'applied_verified')

    def test_concurrent_edit_is_not_overwritten(self):
        source=layout([panel('val/epoch/loss')])
        view=dict(id='v1',name='abc12345',displayName='run',spec=json.dumps(source),user={'username':'me'})
        run=types.SimpleNamespace(id='abc12345',name='run',summary_metrics={'val/epoch/loss':1.})
        api=types.SimpleNamespace(viewer=types.SimpleNamespace(username='me'),runs=lambda *a,**k:[run])
        cfg=dict(project_urls=['me/proj'],view_types=['run-view'])
        with tempfile.TemporaryDirectory() as tmp,patch.object(wb,'views',return_value=iter([view])), \
             patch.object(wb,'read_current',return_value={**view,'spec':json.dumps(layout())}),patch.object(wb,'write_view') as write:
            with self.assertRaisesRegex(RuntimeError,'failed'):wb.process(api,cfg,Path(tmp),True)
            write.assert_not_called()

    def test_missing_run_view_api_is_reported_not_claimed_as_per_run_success(self):
        source=layout();source['section']['runSets']=[]
        view=dict(id='shared',name='shared-layout',displayName='Project',spec=json.dumps(source),user={'username':'me'})
        run=types.SimpleNamespace(id='abc12345',name='run',summary_metrics={'val/epoch/loss':1.})
        api=types.SimpleNamespace(viewer=types.SimpleNamespace(username='me'),runs=lambda *a,**k:[run])
        def get_views(api,entity,project,kind):
            if kind=='run-view':raise RuntimeError('run layouts unavailable')
            yield view
        with tempfile.TemporaryDirectory() as tmp,patch.object(wb,'views',side_effect=get_views):
            wb.process(api,dict(project_urls=['me/proj']),Path(tmp))
            p=json.loads((Path(tmp)/'report.json').read_text())['projects'][0]
            self.assertEqual(p['explicit_run_layouts_matched'],0)
            self.assertEqual(p['runs_without_discovered_explicit_layout'],['abc12345'])
            self.assertIn('run-view',p['discovery_errors'])

    def test_entity_is_not_layout_owner_and_skip_reason_is_recorded(self):
        source=layout();source['section']['runSets']=[]
        view=dict(id='workspace-id',name='workspace',spec=json.dumps(source),user={'username':'person'})
        run=types.SimpleNamespace(id='abc12345',name='run',summary_metrics={'val/epoch/loss':1.})
        api=types.SimpleNamespace(viewer=types.SimpleNamespace(username='person'),runs=lambda *a,**k:[run])
        cfg=dict(project_urls=['team/proj'],view_types=['project-view'],username='team')
        with tempfile.TemporaryDirectory() as tmp,patch.object(wb,'views',return_value=iter([view])), \
             patch.object(wb,'write_view') as write:
            with self.assertRaisesRegex(RuntimeError,'exact skip reasons'):
                wb.process(api,cfg,Path(tmp))
            p=json.loads((Path(tmp)/'report.json').read_text())['projects'][0]
            self.assertEqual(p['selected_runs'],1)
            self.assertEqual(p['skipped_layouts'],{'username_mismatch':1})
            self.assertEqual(p['layout_inventory'][0]['username'],'person')
            self.assertEqual(p['layout_inventory'][0]['reason'],'username_mismatch')
            write.assert_not_called()

    def test_null_username_uses_personal_account_and_never_other_owners(self):
        source=layout();source['section']['runSets']=[]
        own=dict(id='own',name='workspace',spec=json.dumps(source),user={'username':'person'})
        other={**own,'id':'other','name':'other-workspace','user':{'username':'colleague'}}
        run=types.SimpleNamespace(id='abc12345',name='run',summary_metrics={'val/epoch/loss':1.})
        api=types.SimpleNamespace(viewer=types.SimpleNamespace(username='person'),runs=lambda *a,**k:[run])
        cfg=dict(project_urls=['team/proj'],view_types=['project-view'],username=None)
        with tempfile.TemporaryDirectory() as tmp,patch.object(wb,'views',return_value=iter([own,other])), \
             patch.object(wb,'write_view') as write:
            wb.process(api,cfg,Path(tmp))
            report=json.loads((Path(tmp)/'report.json').read_text());p=report['projects'][0]
            self.assertEqual(p['requested_username'],'person')
            self.assertEqual(p['shared_layouts_processed'],1)
            self.assertEqual(p['skipped_layouts'],{'username_mismatch':1})
            self.assertEqual(report['views'][0]['status'],'preview')
            write.assert_not_called()

    def test_restore_uses_recorded_existing_id_and_verifies_original_spec(self):
        source=layout([panel('val/epoch/loss')]);after,_=wb.transform_spec(source,{'val/epoch/loss'})
        view=dict(id='real-id',name='saved',spec=json.dumps(source))
        record=dict(entity='me',project='proj',view_type='project-view',view=view,status='applied_verified',
            before_sha256=wb.sha(source),after_sha256=wb.sha(after))
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);wb.save(folder/'one/record.json',record);wb.save(folder/'one/before.json',source)
            with patch.object(wb,'read_current',side_effect=[{**view,'spec':json.dumps(after)},view]), \
                 patch.object(wb,'write_view') as write:
                wb.restore(object(),folder,True)
                self.assertEqual(write.call_args.args[1]['view']['id'],'real-id')
                self.assertEqual(json.loads((folder/'one/record.json').read_text())['status'],'restored_verified')


if __name__=='__main__':unittest.main()
