"""Native-file backup must be proved independently of complete metric/ROI counts."""

from copy import deepcopy
import unittest

import native_file_delivery as native


class NativeFileDeliveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = native.read(native.ROOT / "native_test_file_delivery.json")

    def test_actual_complete_mapping_and_nonoverlapping_totals(self):
        result = native.validate(self.payload)
        self.assertEqual(result["native_test_keys"], 912)
        self.assertEqual(result["post_pr21_member_checks"], 1374)
        self.assertEqual(result["all_delivery_provider_files"], 928)
        self.assertEqual(result["all_delivery_bytes"], 340723262725)

    def test_complete_roi_counts_cannot_hide_missing_native_keys(self):
        changed = deepcopy(self.payload)
        rows = changed["post_pr21_229"]["records"]
        rows[-1] = deepcopy(rows[0])
        with self.assertRaisesRegex(ValueError, "Selected native-key coverage"):
            native.validate(changed)

    def test_missing_frozen_prediction_is_rejected(self):
        changed = deepcopy(self.payload)
        first = changed["frozen671"]["records"][0]
        first["files"] = [
            entry for entry in first["files"]
            if not entry["source_path"].endswith("/predictions.pkl")
        ]
        with self.assertRaisesRegex(ValueError, "Missing frozen selected file"):
            native.validate(changed)

    def test_member_presence_size_and_regular_content_are_required(self):
        for mutation in ("wrong_path", "wrong_size", "external_link"):
            with self.subTest(mutation=mutation):
                changed = deepcopy(self.payload)
                member = changed["post_pr21_229"]["records"][0]["required_members"][0]
                if mutation == "wrong_path":
                    member["source_path"] += ".different-selection"
                elif mutation == "wrong_size":
                    member["stored_bytes"] += 1
                else:
                    member["symlink"] = True
                    member["regular_file"] = False
                with self.assertRaisesRegex(ValueError, "Native TAR member"):
                    native.validate(changed)

    def test_upload_return_or_provider_value_alone_is_not_proof(self):
        changed = deepcopy(self.payload)
        changed["post_pr21_229"]["provider_receipts"][0]["provider_md5"] = "0" * 32
        with self.assertRaisesRegex(ValueError, "Native archive provider mismatch"):
            native.validate(changed)

    def test_overlapping_native_containers_are_not_added_twice(self):
        changed = deepcopy(self.payload)
        changed["combined_delivery_totals"]["provider_files"] = 699 + 252
        with self.assertRaisesRegex(ValueError, "double-counted"):
            native.validate(changed)


if __name__ == "__main__":
    unittest.main()
