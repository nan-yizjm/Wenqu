from pathlib import Path
import tempfile
import unittest

from src.sft_data_v2 import CATEGORIES, load, prepare


class SFTDataV2Tests(unittest.TestCase):
    def test_all_behaviors_exist_and_fact_groups_do_not_cross_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "v2"
            diagnostics = prepare(path, group_count=30)
            dataset, manifest, _ = load(path)
            self.assertEqual(set(diagnostics["category_counts"]["train"]), set(CATEGORIES))
            self.assertTrue(all(count > 0 for count in diagnostics["category_counts"]["validation"].values()))
            train_groups = {item["group_id"] for item in dataset["train"]}
            validation_groups = {item["group_id"] for item in dataset["validation"]}
            self.assertFalse(train_groups & validation_groups)
            self.assertEqual(manifest["external_test"], "data/rag_sft_challenge_v1.json")


if __name__ == "__main__":
    unittest.main()
