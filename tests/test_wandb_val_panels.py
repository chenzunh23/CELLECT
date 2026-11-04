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
