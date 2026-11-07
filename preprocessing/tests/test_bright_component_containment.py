"""Bright isophotes must not suppress otherwise usable isolated sources."""
import unittest
from unittest.mock import patch
import numpy as np
from astropy.table import Table
from preprocessing.bright_label import (
    BrightLabelConfig, classify_component_bright, final_label_to_source_class,
    source_region_fully_inside_component,
)
from preprocessing.labels import SourceClass as C


def source(sid, x=80., y=80., a=25., b=15., blend=0.):
    return dict(source_id=sid, table_index=sid-1, x=x, y=y, output_x=x, output_y=y,
                major=a, minor=b, theta_deg=0., area=np.pi*a*b, axis_ratio=a/b,
                mag=16., blendedness_abs=blend, **{'class': 'galaxy'})


class BrightComponentContainmentTests(unittest.TestCase):
    def classify(self, rows, *, small=False, gaia=None):
        components=np.zeros((200,200),np.int32)
        components[70:91,70:91] = 1 if small else 0
        if not small: components[10:190,10:190]=1
        for row in rows:
            self.assertTrue(source_region_fully_inside_component(row,components,1))
        out,_,_,_=classify_component_bright(sources=rows,gaia_rows=gaia or [],
            component_labels=components,quality_mask=None,config=BrightLabelConfig())
        return out

    def test_contained_single_large_source_retains_weak_shape(self):
        row=self.classify([source(1)])[0]
        self.assertEqual(final_label_to_source_class(row['final_label']),C.WEAK_SHAPE)
        self.assertEqual(row['reason'],'single_cluster_large_bright_component_weak_shape')

    def test_contained_small_source_retains_clean(self):
        row=self.classify([source(1,a=5.,b=3.)],small=True)[0]
        self.assertEqual(final_label_to_source_class(row['final_label']),C.CLEAN)

    def test_disjoint_sources_share_bright_component(self):
        rows=self.classify([source(1,x=50.),source(2,x=145.)])
        self.assertEqual([final_label_to_source_class(r['final_label']) for r in rows],
                         [C.WEAK_SHAPE,C.WEAK_SHAPE])

    def test_no_gaia_large_component_adds_only_one_fallback_center(self):
        stars=[source(1,x=78.),source(2,x=82.)]
        for row in stars:row['class']='star'
        rows=self.classify(stars)
        self.assertEqual([final_label_to_source_class(r['final_label']) for r in rows[:2]],
                         [C.ORDINARY_IGNORE,C.ORDINARY_IGNORE])
        self.assertEqual(final_label_to_source_class(rows[2]['final_label']),C.STRICT_CENTER_ONLY)
        self.assertNotIn('gaia',rows[2]['reason'])

    def test_unmatched_blended_galaxies_do_not_get_stellar_geometry(self):
        rows=self.classify([source(1,x=78.),source(2,x=82.)])
        self.assertEqual(len(rows),2)
        self.assertTrue(all(final_label_to_source_class(r['final_label'])==C.ORDINARY_IGNORE for r in rows))

    def test_gaia_matched_fragments_keep_external_center(self):
        rows=self.classify([source(1,x=78.),source(2,x=82.)],
            gaia=[dict(source_id=100,x=80.,y=80.,phot_g_mean_mag=15.,astrometric_evidence=True)])
        self.assertEqual([final_label_to_source_class(r['final_label']) for r in rows[:2]],
                         [C.RESTRICTED_BRIGHT_REGION]*2)
        self.assertEqual(final_label_to_source_class(rows[2]['final_label']),C.STRICT_CENTER_ONLY)

    def test_high_blendedness_without_gaia_never_gets_geometric_center(self):
        rows=self.classify([source(1,blend=.9)])
        self.assertEqual(final_label_to_source_class(rows[0]['final_label']),C.ORDINARY_IGNORE)
        self.assertEqual(len(rows),1)

    def test_high_blendedness_uses_reliable_gaia_instead_of_geometry(self):
        rows=self.classify([source(1,blend=.9)],gaia=[dict(source_id=100,x=84.,y=83.,
            phot_g_mean_mag=21.,astrometric_evidence=True)])
        self.assertEqual(final_label_to_source_class(rows[0]['final_label']),C.RESTRICTED_BRIGHT_REGION)
        self.assertEqual(rows[1]['class'],'gaia_star')
        self.assertEqual((rows[1]['x'],rows[1]['y']),(84.,83.))

    def test_uncertain_gaia_preserves_isolated_galaxy(self):
        rows=self.classify([source(1)],gaia=[dict(source_id=100,x=100.,y=80.,phot_g_mean_mag=17.,astrometric_evidence=False)])
        self.assertEqual(len(rows),1)
        self.assertEqual(final_label_to_source_class(rows[0]['final_label']),C.WEAK_SHAPE)
        self.assertEqual(rows[0]['reason'],'single_cluster_large_bright_component_weak_shape')

    def test_reliable_gaia_overrides_weak_galaxy_and_wins_over_uncertain(self):
        rows=self.classify([source(1)],gaia=[
            dict(source_id=100,x=100.,y=80.,phot_g_mean_mag=21.9,astrometric_evidence=True),
            dict(source_id=101,x=80.,y=80.,phot_g_mean_mag=15.,astrometric_evidence=False)])
        self.assertEqual(final_label_to_source_class(rows[0]['final_label']),C.RESTRICTED_BRIGHT_REGION)
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[1]['gaia_source_id'],100)

    def test_g22_is_excluded_and_external_center_mapping_is_strict(self):
        rows=self.classify([source(1)],gaia=[dict(source_id=100,x=80.,y=80.,phot_g_mean_mag=22.,astrometric_evidence=True)])
        self.assertEqual(final_label_to_source_class(rows[0]['final_label']),C.WEAK_SHAPE)
        self.assertEqual(len(rows),1)
        self.assertEqual(final_label_to_source_class('center_only_external'),C.STRICT_CENTER_ONLY)

    def test_large_isolated_star_is_strict_but_galaxy_stays_weak(self):
        row=source(1);row['class']='star'
        self.assertEqual(final_label_to_source_class(self.classify([row])[0]['final_label']),C.STRICT_CENTER_ONLY)

    def test_unknown_follows_galaxy_not_stellar_center_only_policy(self):
        for small in (False,True):
            row=source(1,a=5. if small else 25.,b=3. if small else 15.);row['class']='unknown'
            classified=self.classify([row],small=small)
            self.assertEqual(final_label_to_source_class(classified[0]['final_label']),C.CLEAN if small else C.WEAK_SHAPE)
            self.assertEqual(len(classified),1)

    def test_unknown_blended_group_does_not_get_stellar_geometric_fallback(self):
        rows=[source(1,x=78.),source(2,x=82.)]
        for row in rows:row['class']='unknown'
        classified=self.classify(rows)
        self.assertEqual(len(classified),2)
        self.assertTrue(all(final_label_to_source_class(row['final_label'])==C.ORDINARY_IGNORE for row in classified))

    def test_reliable_gaia_still_overrides_unknown_shape(self):
        row=source(1);row['class']='unknown'
        classified=self.classify([row],gaia=[dict(source_id=100,x=80.,y=80.,phot_g_mean_mag=17.,astrometric_evidence=True)])
        self.assertEqual(final_label_to_source_class(classified[0]['final_label']),C.RESTRICTED_BRIGHT_REGION)
        self.assertEqual(classified[1]['class'],'gaia_star')

    def test_outside_bright_mask_still_matches_isolated_source(self):
        comp=np.zeros((100,100),np.int32);comp[30:70,30:70]=1
        row=source(1,x=32,y=50,a=3,b=2,blend=.9)
        result,_,_,_=classify_component_bright(sources=[row],component_labels=comp,quality_mask=None,
            gaia_rows=[dict(source_id=100,x=28.,y=50.,phot_g_mean_mag=17.,astrometric_evidence=True)],config=BrightLabelConfig())
        self.assertEqual(result[-1]['class'],'gaia_star')
        self.assertEqual(result[-1]['x'],28.)

    def test_empty_geometry_requires_area_strictly_greater_than_1000(self):
        for area,expected in [(1000,False),(1001,True)]:
            comp=np.zeros((5,area+2),np.int32);comp[2,1:area+1]=1
            result,_,_,_=classify_component_bright(sources=[],gaia_rows=[],component_labels=comp,
                quality_mask=None,config=BrightLabelConfig())
            self.assertEqual(bool(result),expected)

    def test_uncertain_gaia_preserves_bright_ordinary_and_dropped_rows(self):
        from preprocessing.bright_label import label_bright_sources
        from preprocessing.labels import SourceLabels
        from preprocessing.utils.geometry import EllipseGeometry
        geom=EllipseGeometry(np.array([40.,50.,60.]),np.array([50.,50.,50.]),
            np.full(3,3.),np.full(3,2.),np.zeros(3),np.full(3,6*np.pi))
        tab=Table(dict(id=[1,2,3],base_ClassificationExtendedness_value=[1.,1.,1.]))
        labels=SourceLabels(np.array([C.CLEAN,C.CLEAN,C.DROPPED]),np.array(['start']*3,object))
        comp=np.zeros((100,100),np.int32);comp[10:90,10:90]=1
        with patch('preprocessing.bright_label.compute_kron_ellipse',return_value=geom):
            result=label_bright_sources(tab,np.array([True,False,False]),labels,bright_region=comp>0,
                component_labels=comp,gaia_rows=[dict(source_id=100,x=50.,y=50.,phot_g_mean_mag=17.,astrometric_evidence=False)],
                mag=np.array([16.,23.,23.]))
        np.testing.assert_array_equal(result.labels.source_class,[C.WEAK_SHAPE,C.CLEAN,C.DROPPED])
        self.assertEqual(len(result.strict_center_x),0)

    def test_uncertain_outside_gaia_does_not_change_source_or_block_reliable_match(self):
        import copy
        comp=np.zeros((100,100),np.int32);comp[30:70,30:70]=1
        row=source(1,x=32,y=50,a=3,b=2)
        uncertain=dict(source_id=101,x=29.,y=50.,phot_g_mean_mag=17.,astrometric_evidence=False)
        reliable=dict(source_id=100,x=28.,y=50.,phot_g_mean_mag=17.,astrometric_evidence=True)
        def run(gaia):
            return classify_component_bright(sources=[copy.deepcopy(row)],component_labels=comp,
                quality_mask=None,gaia_rows=gaia,config=BrightLabelConfig())[0]
        self.assertEqual(run([uncertain]),run([]))
        self.assertEqual(run([uncertain,reliable]),run([reliable]))

    def test_uncertain_gaia_is_absent_for_empty_and_ambiguous_component_fallback(self):
        import copy
        comp=np.zeros((100,100),np.int32);comp[10:90,10:90]=1
        uncertain=dict(source_id=100,x=50.,y=50.,phot_g_mean_mag=17.,astrometric_evidence=False)
        for sources in ([],[source(1,x=50,y=50,blend=.9)]):
            def run(gaia):
                out,meta,_,ignored=classify_component_bright(sources=copy.deepcopy(sources),
                    component_labels=comp,quality_mask=None,gaia_rows=gaia,config=BrightLabelConfig())
                return out,meta,ignored.tolist()
            self.assertEqual(run([uncertain]),run([]))

    def test_unusable_shape_not_promoted(self):
        row=self.classify([source(1,a=40.,b=3.)])[0]
        self.assertEqual(final_label_to_source_class(row['final_label']),C.ORDINARY_IGNORE)


if __name__=='__main__': unittest.main()
