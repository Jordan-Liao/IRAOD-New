from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from tools.cga_research.build_data_manifest import (
    DataManifestError,
    build_manifest,
    canonical_json_bytes,
    main,
    sidecar_path,
    verify_manifest,
)


class BuildDataManifestTests(unittest.TestCase):
    CLASSES = ("ship", "aircraft", "car", "tank", "bridge", "harbor")

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.ann_root = self.root / "annotations"
        self.image_root = self.root / "images"
        self.output = self.root / "manifests" / "corrupted_val.json"

        self._write(self.ann_root / "nested" / "alpha.txt", b"alpha annotation\n")
        self._write(self.ann_root / "zeta.txt", b"zeta annotation\n")
        self._write(self.image_root / "tiles" / "zeta.png", b"synthetic png bytes")
        self._write(self.image_root / "alpha.JPG", b"synthetic jpg bytes")

    @staticmethod
    def _write(path: Path, content: bytes) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def _build(self, output: Path | None = None, *, overwrite: bool = False) -> str:
        return build_manifest(
            ann_root=self.ann_root,
            image_root=self.image_root,
            split="val",
            corruption="synthetic_chaff",
            class_order=self.CLASSES,
            output=self.output if output is None else output,
            overwrite=overwrite,
        )

    @staticmethod
    def _rewrite_sidecar(manifest: Path) -> str:
        content = manifest.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        sidecar_path(manifest).write_text(
            f"{digest}  {manifest.name}\n", encoding="ascii"
        )
        return digest

    def test_build_is_canonical_deterministic_and_verifiable(self) -> None:
        digest = self._build()
        payload = json.loads(self.output.read_text(encoding="utf-8"))

        self.assertEqual(self.output.read_bytes(), canonical_json_bytes(payload))
        self.assertEqual(digest, hashlib.sha256(self.output.read_bytes()).hexdigest())
        self.assertEqual(
            sidecar_path(self.output).read_text(encoding="ascii"),
            f"{digest}  {self.output.name}\n",
        )
        self.assertEqual(verify_manifest(self.output), digest)

        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["split"], "val")
        self.assertEqual(payload["corruption"], "synthetic_chaff")
        self.assertEqual(payload["class_order"], list(self.CLASSES))
        self.assertEqual(
            [entry["relative_path"] for entry in payload["annotations"]["files"]],
            ["nested/alpha.txt", "zeta.txt"],
        )
        self.assertEqual(
            [entry["relative_path"] for entry in payload["images"]["files"]],
            ["alpha.JPG", "tiles/zeta.png"],
        )
        self.assertEqual(
            payload["annotations"]["summary"]["extension_counts"], {".txt": 2}
        )
        self.assertEqual(
            payload["images"]["summary"]["extension_counts"],
            {".jpg": 1, ".png": 1},
        )
        self.assertEqual(payload["alignment"]["stem_count"], 2)
        self.assertTrue(payload["alignment"]["stems_equal"])
        for dataset in ("annotations", "images"):
            for entry in payload[dataset]["files"]:
                self.assertEqual(set(entry), {"relative_path", "sha256", "size_bytes"})

        second = self.root / "other" / "same_input.json"
        second_digest = self._build(second)
        self.assertEqual(second.read_bytes(), self.output.read_bytes())
        self.assertEqual(second_digest, digest)

    def test_existing_output_or_sidecar_requires_overwrite(self) -> None:
        original_digest = self._build()
        with self.assertRaisesRegex(DataManifestError, "--overwrite"):
            self._build()

        self.output.write_bytes(b"not a manifest")
        replacement_digest = self._build(overwrite=True)
        self.assertEqual(replacement_digest, original_digest)
        self.assertEqual(verify_manifest(self.output), original_digest)

        other = self.root / "other_manifest.json"
        sidecar_path(other).write_text("occupied\n", encoding="ascii")
        with self.assertRaisesRegex(DataManifestError, "sidecar.*already exists"):
            self._build(other)

    def test_no_overwrite_is_atomic_against_late_competing_writer(self) -> None:
        real_link = os.link

        def competing_link(source, destination):
            destination_path = Path(destination)
            destination_path.write_bytes(b"competing writer")
            return real_link(source, destination)

        with patch(
            "tools.cga_research.build_data_manifest.os.link",
            side_effect=competing_link,
        ), self.assertRaisesRegex(DataManifestError, "refusing to overwrite"):
            self._build()
        self.assertFalse(self.output.exists())
        self.assertEqual(sidecar_path(self.output).read_bytes(), b"competing writer")

    def test_no_overwrite_half_publish_rolls_back_owned_sidecar(self) -> None:
        real_link = os.link
        calls = 0

        def fail_manifest_link(source, destination):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("synthetic manifest-link failure")
            return real_link(source, destination)

        with patch(
            "tools.cga_research.build_data_manifest.os.link",
            side_effect=fail_manifest_link,
        ), self.assertRaisesRegex(DataManifestError, "manifest-link failure"):
            self._build()
        self.assertFalse(self.output.exists())
        self.assertFalse(sidecar_path(self.output).exists())

    def test_no_overwrite_verification_failure_rolls_back_bundle(self) -> None:
        with patch(
            "tools.cga_research.build_data_manifest.verify_manifest",
            side_effect=DataManifestError("synthetic verification failure"),
        ), self.assertRaisesRegex(DataManifestError, "verification failure"):
            self._build()
        self.assertFalse(self.output.exists())
        self.assertFalse(sidecar_path(self.output).exists())

    def test_stems_must_be_unique_and_sets_must_match(self) -> None:
        self._write(self.ann_root / "duplicate" / "alpha.json", b"duplicate")
        with self.assertRaisesRegex(DataManifestError, "duplicate stem 'alpha'"):
            self._build()

        (self.ann_root / "duplicate" / "alpha.json").unlink()
        self._write(self.ann_root / "orphan.txt", b"orphan")
        with self.assertRaisesRegex(DataManifestError, "stem sets differ"):
            self._build()

    def test_class_order_is_explicit_nonempty_and_unique(self) -> None:
        with self.assertRaisesRegex(DataManifestError, "duplicate classes"):
            build_manifest(
                self.ann_root,
                self.image_root,
                "val",
                "synthetic",
                ("ship", "ship"),
                self.output,
            )
        with self.assertRaisesRegex(DataManifestError, "at least one class"):
            build_manifest(
                self.ann_root,
                self.image_root,
                "val",
                "synthetic",
                (),
                self.output,
            )

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are unavailable")
    def test_symlink_inside_tree_and_symlink_root_are_rejected(self) -> None:
        external = self._write(self.root / "outside.txt", b"outside")
        link = self.ann_root / "escape.txt"
        link.symlink_to(external)
        with self.assertRaisesRegex(DataManifestError, "contains a symlink"):
            self._build()

        link.unlink()
        alias = self.root / "ann_alias"
        alias.symlink_to(self.ann_root, target_is_directory=True)
        with self.assertRaisesRegex(DataManifestError, "symlink path component"):
            build_manifest(
                alias,
                self.image_root,
                "val",
                "synthetic",
                self.CLASSES,
                self.output,
            )

    def test_output_must_not_be_inside_either_dataset_root(self) -> None:
        with self.assertRaisesRegex(DataManifestError, "inside a dataset root"):
            self._build(self.ann_root / "manifest.json")
        with self.assertRaisesRegex(DataManifestError, "inside a dataset root"):
            self._build(self.image_root / "manifest.json")

    def test_verify_detects_content_addition_deletion_and_sidecar_tampering(
        self,
    ) -> None:
        self._build()
        (self.image_root / "alpha.JPG").write_bytes(b"changed bytes")
        with self.assertRaisesRegex(DataManifestError, "no longer matches"):
            verify_manifest(self.output)

        self._build(overwrite=True)
        self._write(self.image_root / "extra.jpg", b"extra")
        with self.assertRaisesRegex(DataManifestError, "stem sets differ"):
            verify_manifest(self.output)

        (self.image_root / "extra.jpg").unlink()
        (self.image_root / "tiles" / "zeta.png").unlink()
        with self.assertRaisesRegex(
            DataManifestError, "no longer matches|stem sets differ"
        ):
            verify_manifest(self.output)

        sidecar_path(self.output).write_text(
            "0" * 64 + "  wrong.json\n", encoding="ascii"
        )
        with self.assertRaisesRegex(DataManifestError, "detached SHA-256 mismatch"):
            verify_manifest(self.output)

    def test_verify_rejects_noncanonical_or_unknown_schema_even_with_new_sidecar(
        self,
    ) -> None:
        self._build()
        payload = json.loads(self.output.read_text(encoding="utf-8"))
        self.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._rewrite_sidecar(self.output)
        with self.assertRaisesRegex(DataManifestError, "not canonical JSON"):
            verify_manifest(self.output)

        payload["unexpected"] = True
        self.output.write_bytes(canonical_json_bytes(payload))
        self._rewrite_sidecar(self.output)
        with self.assertRaisesRegex(DataManifestError, "unknown fields"):
            verify_manifest(self.output)

    def test_verify_rejects_boolean_counts_even_with_valid_detached_hash(self) -> None:
        self._build()
        payload = json.loads(self.output.read_text(encoding="utf-8"))
        payload["annotations"]["summary"]["file_count"] = True
        self.output.write_bytes(canonical_json_bytes(payload))
        self._rewrite_sidecar(self.output)
        with self.assertRaisesRegex(DataManifestError, "positive integer"):
            verify_manifest(self.output)

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are unavailable")
    def test_verify_rejects_manifest_symlink_before_following_it(self) -> None:
        self._build()
        outside = self._write(self.root / "outside_manifest.json", b"not JSON")
        self.output.unlink()
        self.output.symlink_to(outside)
        with self.assertRaisesRegex(DataManifestError, "regular non-symlink"):
            verify_manifest(self.output)

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are unavailable")
    def test_verify_rejects_symlink_introduced_after_build(self) -> None:
        self._build()
        external = self._write(self.root / "outside_after.txt", b"outside")
        (self.image_root / "escape.jpg").symlink_to(external)
        with self.assertRaisesRegex(DataManifestError, "contains a symlink"):
            verify_manifest(self.output)

    def test_cli_build_and_verify_use_only_declared_synthetic_roots(self) -> None:
        output = self.root / "cli" / "manifest.json"
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            status = main(
                [
                    "build",
                    "--ann-root",
                    str(self.ann_root),
                    "--image-root",
                    str(self.image_root),
                    "--split",
                    "val",
                    "--corruption",
                    "synthetic",
                    "--class-order",
                    ",".join(self.CLASSES),
                    "--output",
                    str(output),
                ]
            )
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "built")

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            status = main(["verify", "--manifest", str(output)])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "verified")

        (self.image_root / "alpha.JPG").write_bytes(b"changed")
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            status = main(["verify", "--manifest", str(output)])
        self.assertEqual(status, 2)
        self.assertIn("data manifest error", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
