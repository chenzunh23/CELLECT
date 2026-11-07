import unittest

import numpy as np
from astropy.table import Table

from preprocessing.labels import DenseLabel as D, SourceClass as C, SourceLabels
from preprocessing.region_filling import fill_dense_regions, RegionFillingConfig
from preprocessing.utils.geometry import EllipseGeometry


class DenseTests(unittest.TestCase):
    def fill(self, classes, axes, **kwargs):
        a = np.asarray(axes, float)
        n = len(a)
        geom = EllipseGeometry(np.full(n, 150.), np.full(n, 150.), a, a / 2,
                               np.zeros(n), np.pi * a * a / 2)
        labels = SourceLabels(np.array(classes), np.full(n, '', object))
        return fill_dense_regions(Table(), labels, (301, 301), geometry=geom,
                                  background_mask=np.ones((301, 301), bool), **kwargs)

    def test_dropped_giant_does_not_exclude_background(self):
        dense = self.fill([C.DROPPED], [100000])
        self.assertTrue(np.all(dense == D.BACKGROUND))

    def test_cap_preserves_ratio_and_excludes_full_outskirts(self):
        dense = self.fill([C.CLEAN], [140])
        self.assertEqual(dense[150, 249], D.CLEAN)
        self.assertEqual(dense[199, 150], D.CLEAN)
        self.assertEqual(dense[201, 150], D.ORDINARY_IGNORE)
        self.assertEqual(dense[150, 270], D.ORDINARY_IGNORE)
        self.assertEqual(dense[150, 295], D.BACKGROUND)

    def test_ignore_not_truncated_and_bright_wins(self):
        bright = np.zeros((301, 301), bool); bright[145:155, 260:275] = True
        dense = self.fill([C.ORDINARY_IGNORE], [140], restricted_fallback_mask=bright)
        self.assertEqual(dense[150, 270], D.RESTRICTED_BRIGHT_REGION)
        self.assertEqual(dense[150, 280], D.ORDINARY_IGNORE)

    def test_clean_weak_priority(self):
        dense = self.fill([C.CLEAN, C.WEAK_SHAPE, C.ORDINARY_IGNORE], [20, 60, 140])
        self.assertEqual(dense[150, 150], D.CLEAN)
        self.assertEqual(dense[150, 195], D.WEAK_SHAPE)
        self.assertEqual(dense[150, 270], D.ORDINARY_IGNORE)

    def test_small_bright_excludes_background_without_becoming_restricted(self):
        bright = np.zeros((301, 301), bool); bright[0:3, 0:3] = True
        dense = self.fill([C.DROPPED], [100000], bright_region_mask=bright)
        self.assertTrue(np.all(dense[:3, :3] == D.ORDINARY_IGNORE))
        self.assertEqual(dense[4, 4], D.BACKGROUND)

    def test_all_source_classes_erase_background_without_any_bright(self):
        for cls in (C.CLEAN, C.WEAK_SHAPE, C.ORDINARY_IGNORE, C.STRICT_CENTER_ONLY):
            with self.subTest(cls=cls):
                dense = self.fill([cls], [140])
                self.assertEqual(dense[150, 280], D.ORDINARY_IGNORE)
                self.assertEqual(dense[150, 295], D.BACKGROUND)

    def test_center_only_does_not_overwrite_bright(self):
        bright = np.zeros((301, 301), bool); bright[145:155, 145:155] = True
        dense = self.fill([C.STRICT_CENTER_ONLY], [140], restricted_fallback_mask=bright)
        self.assertEqual(dense[150, 150], D.RESTRICTED_BRIGHT_REGION)
        self.assertEqual(dense[150, 200], D.ORDINARY_IGNORE)

    def test_six_step_overlap_order(self):
        bright = np.zeros((301, 301), bool); bright[145:155, 150:260] = True
        dense = self.fill([C.CLEAN, C.WEAK_SHAPE, C.ORDINARY_IGNORE], [20, 60, 140],
                          restricted_fallback_mask=bright, ordinary_ignore_mask=bright.copy())
        self.assertEqual(dense[150, 150], D.CLEAN)
        self.assertEqual(dense[150, 195], D.WEAK_SHAPE)
        self.assertEqual(dense[150, 240], D.RESTRICTED_BRIGHT_REGION)
        self.assertEqual(dense[150, 280], D.ORDINARY_IGNORE)
        self.assertEqual(dense[150, 295], D.BACKGROUND)

    def test_no_input_mutation(self):
        mask = np.ones((301, 301), bool)
        self.fill([C.ORDINARY_IGNORE], [140], ordinary_ignore_source_mask=np.array([True]),
                  ordinary_ignore_mask=mask)
        self.assertTrue(mask.all())

    def test_restricted_hsc_geometry_does_not_paint_or_erase_background(self):
        bright=np.zeros((301,301),bool);bright[145:155,145:155]=True
        config=RegionFillingConfig(restricted_source_geometry=False)
        for radius in (10,140,100000):
            dense=self.fill([C.RESTRICTED_BRIGHT_REGION],[radius],restricted_fallback_mask=bright,config=config)
            np.testing.assert_array_equal(dense,np.where(bright,D.RESTRICTED_BRIGHT_REGION,D.BACKGROUND))

    def test_restricted_hsc_rows_without_component_have_no_dense_geometry(self):
        dense=self.fill([C.RESTRICTED_BRIGHT_REGION],[140],
                        config=RegionFillingConfig(restricted_source_geometry=False))
        self.assertTrue(np.all(dense==D.BACKGROUND))

    def test_default_retains_other_dataset_restricted_geometry(self):
        dense=self.fill([C.RESTRICTED_BRIGHT_REGION],[140])
        self.assertEqual(dense[150,270],D.RESTRICTED_BRIGHT_REGION)

    def test_restricted_hsc_component_still_yields_to_valid_shapes(self):
        bright=np.zeros((301,301),bool);bright[140:161,140:221]=True
        dense=self.fill([C.RESTRICTED_BRIGHT_REGION,C.WEAK_SHAPE],[140,20],
            restricted_fallback_mask=bright,config=RegionFillingConfig(restricted_source_geometry=False))
        self.assertEqual(dense[150,150],D.WEAK_SHAPE)
        self.assertEqual(dense[150,210],D.RESTRICTED_BRIGHT_REGION)
        self.assertEqual(dense[150,270],D.BACKGROUND)

    def test_invalid_cap(self):
        with self.assertRaises(ValueError):
            self.fill([C.CLEAN], [10], config=RegionFillingConfig(max_major_pixels=0))


if __name__ == '__main__':
    unittest.main()
