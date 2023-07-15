from unittest import TestCase

from results_analysis import generate_indices_list, get_feature_count


class Test(TestCase):
    def test_generate_indices_list(self):
        test_data = {
            "PTC_MR": ["n_impurity", "narumi", "label_entropy", "estrada", "balaban"],
            "DHFR_MD": ["mod_zagreb", "label_entropy", "nodes", "randic", "balaban"]
        }
        test_lists = generate_indices_list(test_data)
        correct_lists = [[14, 2, 15, 1, 0],
                         [12, 15, 9, 5, 0]]
        print(test_lists)
        self.assertTrue(test_lists, correct_lists)

    def test_get_best_features(self):
        test_data = {'PTC_MR': ['n_impurity', 'narumi', 'label_entropy', 'estrada', 'balaban'],
                     'DHFR_MD': ['mod_zagreb', 'label_entropy', 'nodes', 'randic', 'balaban']}

        test_result = get_feature_count(test_data)
        correct_result = {'n_impurity': 1, 'narumi': 1, 'label_entropy': 2, 'estrada': 1, 'balaban': 2, 'mod_zagreb': 1,
                          'nodes': 1, 'randic': 1}

        self.assertTrue(test_result, correct_result)
