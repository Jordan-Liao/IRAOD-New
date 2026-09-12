"""Focused checks for the private ByPy inventory integrity contract."""
from copy import deepcopy
import unittest

from validate_artifacts import validate_bypy_assets


class ByPyAssetTests(unittest.TestCase):
    def setUp(self):
        root = "/apps/bypy/IRAOD-New/results/completed_snapshot_20260912T014556Z"
        self.asset = {"name": "results.tar", "bytes": 42, "sha256": "a" * 64,
                      "source_md5": "b" * 32, "provider_md5": "b" * 32,
                      "provider_size": 42, "netdisk_path": root + "/results.tar"}
        self.manifest = {"storage": {"netdisk_path": root},
                         "snapshot_cutoff_utc": "2026-09-12T01:45:56Z",
                         "assets": [self.asset], "asset_count": 1, "uploaded_asset_bytes": 42}
        self.listing = [{"path": self.asset["netdisk_path"], "size": 42,
                         "md5": "opaque-provider-listing-value", "block_list": ["b" * 32]}]

    def test_matching_source_and_provider(self):
        self.assertEqual(validate_bypy_assets(self.manifest, self.listing), 1)

    def test_rejects_size_only_hash_and_coverage_failures(self):
        cases = [
            ("provider_md5", lambda m, rows: rows[0].update(block_list=["c" * 32])),
            ("missing_provider_hash", lambda m, rows: rows[0].update(block_list=[])),
            ("first_block_is_not_full_file", lambda m, rows: rows[0].update(block_list=["b" * 32, "c" * 32])),
            ("source_md5", lambda m, rows: m["assets"][0].update(source_md5="c" * 32)),
            ("size", lambda m, rows: rows[0].update(size=41)),
            ("receipt_size", lambda m, rows: m["assets"][0].update(provider_size=41)),
            ("source_sha256", lambda m, rows: m["assets"][0].update(sha256="b" * 32)),
            ("cutoff", lambda m, rows: m.update(snapshot_cutoff_utc="2026-09-13T00:00:00Z")),
            ("missing", lambda m, rows: rows.clear()),
            ("duplicate_listing", lambda m, rows: rows.append(deepcopy(rows[0]))),
            ("unexpected_file", lambda m, rows: rows.append({**rows[0], "path": rows[0]["path"] + ".other"})),
            ("duplicate_asset", lambda m, rows: (m["assets"].append(deepcopy(m["assets"][0])), m.update(asset_count=2, uploaded_asset_bytes=84))),
            ("byte_total", lambda m, rows: m.update(uploaded_asset_bytes=41)),
            ("destination", lambda m, rows: m["assets"][0].update(netdisk_path="/wrong/results.tar")),
        ]
        for name, mutate in cases:
            with self.subTest(name=name):
                manifest, listing = deepcopy(self.manifest), deepcopy(self.listing)
                mutate(manifest, listing)
                with self.assertRaises(ValueError):
                    validate_bypy_assets(manifest, listing)


if __name__ == "__main__":
    unittest.main()
