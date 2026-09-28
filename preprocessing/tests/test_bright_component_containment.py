"""Bright isophotes must not suppress otherwise usable isolated sources."""
import unittest
import numpy as np
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

    def test_overlapping_unmatched_cluster_still_ignored(self):
        rows=self.classify([source(1,x=78.),source(2,x=82.)])
        self.assertEqual([final_label_to_source_class(r['final_label']) for r in rows],
                         [C.ORDINARY_IGNORE,C.ORDINARY_IGNORE])

    def test_gaia_matched_fragments_keep_external_center(self):
        rows=self.classify([source(1,x=78.),source(2,x=82.)],
            gaia=[dict(source_id=100,x=80.,y=80.,phot_g_mean_mag=15.)])
        self.assertEqual([final_label_to_source_class(r['final_label']) for r in rows[:2]],
                         [C.RESTRICTED_BRIGHT_REGION]*2)
        self.assertEqual(final_label_to_source_class(rows[2]['final_label']),C.STRICT_CENTER_ONLY)

    def test_high_blendedness_still_uses_component_center(self):
        rows=self.classify([source(1,blend=.9)])
        self.assertEqual(final_label_to_source_class(rows[0]['final_label']),C.ORDINARY_IGNORE)
        self.assertEqual(final_label_to_source_class(rows[1]['final_label']),C.STRICT_CENTER_ONLY)

    def test_unusable_shape_not_promoted(self):
        row=self.classify([source(1,a=40.,b=3.)])[0]
        self.assertEqual(final_label_to_source_class(row['final_label']),C.ORDINARY_IGNORE)


if __name__=='__main__': unittest.main()
