import unittest
from eval.strict_center_metrics import point_metrics,finalize
class StrictMetricsTests(unittest.TestCase):
    def test_prediction_cannot_match_both_base_and_strict(self):
        c,m,_=point_metrics([[0,0]],[[0,0]],[[1,0]],[-8000000000000001],3)
        self.assertEqual(c['combined_tp'],1);self.assertEqual(c['strict_tp'],0)
        self.assertEqual(finalize(c)['combined_completeness'],.5)
    def test_inserted_and_gaia_and_native_strict_are_reported_separately(self):
        c,_,_=point_metrics([[10,10],[20,20],[30,30]],[],[[10,10],[20,20],[30,30]],[-8000000000000001,-2320211461525382272,23])
        self.assertEqual(c['strict_tp'],3);self.assertEqual(c['inserted_tp'],1);self.assertEqual(c['inserted_gt'],1)
        self.assertEqual(finalize(c)['combined_purity'],1)
    def test_exact_duplicate_gt_not_counted_twice(self):
        c,_,_=point_metrics([[2,2]],[[2,2]],[[2,2],[2,2]],[-8000000000000001,-8000000000000002])
        self.assertEqual(c['combined_gt'],1);self.assertEqual(c['strict_exact_duplicates_removed'],2)
    def test_empty_ground_truth_and_empty_predictions(self):
        c,_,_=point_metrics([],[],[],[])
        self.assertIsNone(finalize(c)['strict_completeness']);self.assertEqual(finalize(c)['combined_purity'],0)
    def test_global_matching_and_base_only_are_distinct_explicit_metrics(self):
        c,_,_=point_metrics([[1,0]],[[0,0]],[[1,0]],[-8000000000000001])
        self.assertEqual(c['base_tp'],1);self.assertEqual(c['combined_base_tp'],0);self.assertEqual(c['strict_tp'],1)
if __name__=='__main__':unittest.main()
