#!/usr/bin/env python3
"""Build and verify a deterministic, content-addressed dataset manifest.

The tool is deliberately independent from MMDetection/MMRotate.  It hashes
files as opaque bytes and never parses annotation semantics or evaluates a
model.  ``build`` records both dataset trees and writes canonical JSON plus a
detached ``.sha256`` sidecar.  ``verify`` validates the sidecar and schema,
then rescans and rehashes both roots.  Every inconsistency is fatal.

Stem identity is the final path component without its last suffix.  Stems
must be unique within each root and the annotation/image stem sets must match
exactly.  Extension counts are case-normalized and include the leading dot;
files without a suffix use ``<none>``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA_VERSION = 1
MANIFEST_TYPE = "iraod_dataset_file_manifest"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
NO_EXTENSION = "<none>"


class DataManifestError(ValueError):
    """Raised when a data manifest cannot be built or trusted."""


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DataManifestError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise DataManifestError(f"non-finite JSON value is forbidden: {value}")


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize one manifest in its unique UTF-8 representation."""

    try:
        content = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        encoded = (content + "\n").encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise DataManifestError(
            f"manifest is not canonical-JSON serializable: {error}"
        ) from error
    return encoded


def _load_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        content = path.read_bytes()
    except OSError as error:
        raise DataManifestError(f"cannot read manifest {path}: {error}") from error
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise DataManifestError(f"manifest must be UTF-8: {path}") from error
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except DataManifestError:
        raise
    except json.JSONDecodeError as error:
        raise DataManifestError(f"invalid JSON in {path}: {error}") from error
    if type(payload) is not dict:
        raise DataManifestError("manifest root must be an object")
    return payload, content


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _hash_string_set(values: Sequence[str]) -> str:
    """Hash a sorted string set using unambiguous canonical JSON."""

    encoded = json.dumps(
        sorted(values),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sidecar_path(manifest_path: Path) -> Path:
    return Path(f"{manifest_path}.sha256")


def _require_string(value: Any, location: str) -> str:
    if type(value) is not str or not value.strip():
        raise DataManifestError(f"{location} must be a non-empty string")
    if any(character in value for character in ("\x00", "\r", "\n")):
        raise DataManifestError(f"{location} contains a forbidden control character")
    return value


def _require_exact_object(
    value: Any,
    location: str,
    required_keys: set[str],
) -> dict[str, Any]:
    if type(value) is not dict:
        raise DataManifestError(f"{location} must be an object")
    keys = set(value)
    missing = sorted(required_keys - keys)
    extra = sorted(keys - required_keys)
    if missing:
        raise DataManifestError(
            f"{location} missing required fields: {', '.join(missing)}"
        )
    if extra:
        raise DataManifestError(f"{location} has unknown fields: {', '.join(extra)}")
    return value


def _absolute_lexical(path: Path) -> Path:
    try:
        expanded = path.expanduser()
    except (RuntimeError, OSError) as error:
        raise DataManifestError(f"cannot expand path {path}: {error}") from error
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    return Path(os.path.abspath(os.fspath(expanded)))


def _reject_symlink_components(path: Path, location: str) -> None:
    """Reject every existing symlink component in a lexical absolute path."""

    absolute = _absolute_lexical(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            break
        except OSError as error:
            raise DataManifestError(
                f"cannot inspect {location} path component {current}: {error}"
            ) from error
        if stat.S_ISLNK(info.st_mode):
            raise DataManifestError(
                f"{location} contains a symlink path component: {current}"
            )


def _resolve_root(raw_path: Path, location: str) -> Path:
    lexical = _absolute_lexical(raw_path)
    _reject_symlink_components(lexical, location)
    try:
        resolved = lexical.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise DataManifestError(f"{location} does not exist: {lexical}") from error
    try:
        mode = resolved.stat().st_mode
    except OSError as error:
        raise DataManifestError(
            f"cannot inspect {location} {resolved}: {error}"
        ) from error
    if not stat.S_ISDIR(mode):
        raise DataManifestError(f"{location} is not a directory: {resolved}")
    return resolved


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_relative_path(value: Any, location: str) -> str:
    relative = _require_string(value, location)
    try:
        relative.encode("utf-8")
    except UnicodeEncodeError as error:
        raise DataManifestError(f"{location} is not valid UTF-8") from error
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or relative in {".", ".."} or ".." in parsed.parts:
        raise DataManifestError(f"{location} must be a safe relative path")
    if parsed.as_posix() != relative or "\\" in relative:
        raise DataManifestError(f"{location} is not a canonical POSIX path")
    return relative


def _discover_files(root: Path, location: str) -> list[Path]:
    """Recursively enumerate regular files without following symlinks."""

    directories = [root]
    files: list[Path] = []
    while directories:
        directory = directories.pop()
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError as error:
            raise DataManifestError(
                f"cannot scan {location} directory {directory}: {error}"
            ) from error
        for entry in entries:
            path = Path(entry.path)
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise DataManifestError(
                    f"cannot inspect {location} entry {path}: {error}"
                ) from error
            if stat.S_ISLNK(info.st_mode):
                raise DataManifestError(
                    f"{location} directory tree contains a symlink: {path}"
                )
            if stat.S_ISDIR(info.st_mode):
                directories.append(path)
            elif stat.S_ISREG(info.st_mode):
                files.append(path)
            else:
                raise DataManifestError(
                    f"{location} contains a non-regular entry: {path}"
                )
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def _file_identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _hash_regular_file(
    path: Path,
    root: Path,
    location: str,
) -> tuple[dict[str, Any], tuple[int, int, int, int, int]]:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise DataManifestError(
            f"cannot safely open {location} file {path}: {error}"
        ) from error

    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise DataManifestError(f"{location} is not a regular file: {path}")
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        raise DataManifestError(
            f"cannot hash {location} file {path}: {error}"
        ) from error
    finally:
        os.close(descriptor)

    before_identity = _file_identity(before)
    if _file_identity(after) != before_identity:
        raise DataManifestError(f"{location} file changed while hashing: {path}")

    relative = path.relative_to(root).as_posix()
    _validate_relative_path(relative, f"{location}.relative_path")
    return (
        {
            "relative_path": relative,
            "sha256": digest.hexdigest(),
            "size_bytes": before.st_size,
        },
        before_identity,
    )


def _extension(relative_path: str) -> str:
    suffix = PurePosixPath(relative_path).suffix.lower()
    return suffix if suffix else NO_EXTENSION


def _stem(relative_path: str) -> str:
    return PurePosixPath(relative_path).stem


def _summarize_files(
    files: Sequence[Mapping[str, Any]], location: str
) -> dict[str, Any]:
    if not files:
        raise DataManifestError(f"{location} directory contains no regular files")
    stems: dict[str, str] = {}
    extensions: Counter[str] = Counter()
    for index, entry in enumerate(files):
        relative = _validate_relative_path(
            entry.get("relative_path"), f"{location}.files[{index}].relative_path"
        )
        stem = _stem(relative)
        if not stem:
            raise DataManifestError(f"{location} file has an empty stem: {relative}")
        if stem in stems:
            raise DataManifestError(
                f"{location} has duplicate stem {stem!r}: "
                f"{stems[stem]!r} and {relative!r}"
            )
        stems[stem] = relative
        extensions[_extension(relative)] += 1
    return {
        "extension_counts": dict(sorted(extensions.items())),
        "file_count": len(files),
        "stem_count": len(stems),
        "stem_set_sha256": _hash_string_set(list(stems)),
    }


def _scan_snapshot(root: Path, location: str) -> dict[str, Any]:
    discovered = _discover_files(root, location)
    records: list[dict[str, Any]] = []
    identities: dict[str, tuple[int, int, int, int, int]] = {}
    for path in discovered:
        record, identity = _hash_regular_file(path, root, location)
        records.append(record)
        identities[record["relative_path"]] = identity

    rediscovered = _discover_files(root, location)
    first_paths = [path.relative_to(root).as_posix() for path in discovered]
    second_paths = [path.relative_to(root).as_posix() for path in rediscovered]
    if first_paths != second_paths:
        raise DataManifestError(f"{location} directory changed while scanning")
    for path in rediscovered:
        relative = path.relative_to(root).as_posix()
        try:
            current = path.lstat()
        except OSError as error:
            raise DataManifestError(
                f"cannot recheck {location} file {path}: {error}"
            ) from error
        if not stat.S_ISREG(current.st_mode):
            raise DataManifestError(f"{location} changed while scanning: {path}")
        if _file_identity(current) != identities[relative]:
            raise DataManifestError(f"{location} file changed while scanning: {path}")

    summary = _summarize_files(records, location)
    return {"files": records, "root": str(root), "summary": summary}


def _validate_class_order(values: Sequence[str]) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise DataManifestError("class-order must be an ordered sequence of names")
    classes: list[str] = []
    for raw_value in values:
        if type(raw_value) is not str:
            raise DataManifestError("class-order entries must be strings")
        for item in raw_value.split(","):
            classes.append(_require_string(item.strip(), "class-order entry"))
    if not classes:
        raise DataManifestError("class-order must contain at least one class")
    if len(set(classes)) != len(classes):
        raise DataManifestError("class-order contains duplicate classes")
    return classes


def _alignment(
    annotation_files: Sequence[Mapping[str, Any]],
    image_files: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    annotation_stems = {
        _stem(str(entry["relative_path"])) for entry in annotation_files
    }
    image_stems = {_stem(str(entry["relative_path"])) for entry in image_files}
    if annotation_stems != image_stems:
        missing_images = sorted(annotation_stems - image_stems)
        missing_annotations = sorted(image_stems - annotation_stems)
        raise DataManifestError(
            "annotation/image stem sets differ: "
            f"missing_images={missing_images[:10]!r}, "
            f"missing_annotations={missing_annotations[:10]!r}"
        )
    return {
        "stem_count": len(annotation_stems),
        "stem_set_sha256": _hash_string_set(list(annotation_stems)),
        "stems_equal": True,
    }


def _build_payload_from_resolved_roots(
    resolved_ann: Path,
    resolved_images: Path,
    split: str,
    corruption: str,
    class_order: Sequence[str],
) -> dict[str, Any]:
    split_value = _require_string(split, "split")
    corruption_value = _require_string(corruption, "corruption")
    classes = _validate_class_order(class_order)

    annotations = _scan_snapshot(resolved_ann, "annotations")
    images = _scan_snapshot(resolved_images, "images")
    alignment = _alignment(annotations["files"], images["files"])
    return {
        "alignment": alignment,
        "annotations": annotations,
        "class_order": classes,
        "corruption": corruption_value,
        "images": images,
        "manifest_type": MANIFEST_TYPE,
        "schema_version": SCHEMA_VERSION,
        "split": split_value,
    }


def build_payload(
    ann_root: Path,
    image_root: Path,
    split: str,
    corruption: str,
    class_order: Sequence[str],
) -> dict[str, Any]:
    resolved_ann = _resolve_root(ann_root, "ann-root")
    resolved_images = _resolve_root(image_root, "image-root")
    return _build_payload_from_resolved_roots(
        resolved_ann,
        resolved_images,
        split,
        corruption,
        class_order,
    )


def _validate_file_entries(value: Any, location: str) -> list[dict[str, Any]]:
    if type(value) is not list or not value:
        raise DataManifestError(f"{location} must be a non-empty array")
    entries: list[dict[str, Any]] = []
    previous: str | None = None
    for index, raw_entry in enumerate(value):
        entry = _require_exact_object(
            raw_entry,
            f"{location}[{index}]",
            {"relative_path", "sha256", "size_bytes"},
        )
        relative = _validate_relative_path(
            entry["relative_path"], f"{location}[{index}].relative_path"
        )
        size = entry["size_bytes"]
        if type(size) is not int or size < 0:
            raise DataManifestError(
                f"{location}[{index}].size_bytes must be non-negative"
            )
        digest = entry["sha256"]
        if type(digest) is not str or SHA256_RE.fullmatch(digest) is None:
            raise DataManifestError(f"{location}[{index}].sha256 is invalid")
        if previous is not None and relative <= previous:
            raise DataManifestError(
                f"{location} must be strictly sorted by relative_path"
            )
        previous = relative
        entries.append(
            {"relative_path": relative, "sha256": digest, "size_bytes": size}
        )
    return entries


def _validate_summary(
    value: Any,
    files: Sequence[Mapping[str, Any]],
    location: str,
) -> None:
    summary = _require_exact_object(
        value,
        location,
        {"extension_counts", "file_count", "stem_count", "stem_set_sha256"},
    )
    if type(summary["file_count"]) is not int or summary["file_count"] <= 0:
        raise DataManifestError(f"{location}.file_count must be a positive integer")
    if type(summary["stem_count"]) is not int or summary["stem_count"] <= 0:
        raise DataManifestError(f"{location}.stem_count must be a positive integer")
    if (
        type(summary["stem_set_sha256"]) is not str
        or SHA256_RE.fullmatch(summary["stem_set_sha256"]) is None
    ):
        raise DataManifestError(f"{location}.stem_set_sha256 is invalid")
    extension_counts = summary["extension_counts"]
    if type(extension_counts) is not dict or not extension_counts:
        raise DataManifestError(f"{location}.extension_counts must be an object")
    for extension, count in extension_counts.items():
        if type(extension) is not str or not extension:
            raise DataManifestError(f"{location}.extension_counts has an invalid key")
        if type(count) is not int or count <= 0:
            raise DataManifestError(
                f"{location}.extension_counts[{extension!r}] must be positive"
            )
    expected = _summarize_files(files, location.rsplit(".", 1)[0])
    if summary != expected:
        raise DataManifestError(f"{location} does not match its file entries")


def _validate_snapshot(value: Any, location: str) -> tuple[Path, list[dict[str, Any]]]:
    snapshot = _require_exact_object(value, location, {"files", "root", "summary"})
    root_text = _require_string(snapshot["root"], f"{location}.root")
    root_candidate = Path(root_text)
    if not root_candidate.is_absolute():
        raise DataManifestError(f"{location}.root must be absolute")
    root = _resolve_root(root_candidate, f"{location}.root")
    if str(root) != root_text:
        raise DataManifestError(f"{location}.root is not a canonical resolved path")
    files = _validate_file_entries(snapshot["files"], f"{location}.files")
    _validate_summary(snapshot["summary"], files, f"{location}.summary")
    return root, files


def validate_payload(payload: Any) -> tuple[Path, Path]:
    manifest = _require_exact_object(
        payload,
        "manifest",
        {
            "alignment",
            "annotations",
            "class_order",
            "corruption",
            "images",
            "manifest_type",
            "schema_version",
            "split",
        },
    )
    if (
        type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != SCHEMA_VERSION
    ):
        raise DataManifestError(
            f"unsupported schema_version: {manifest['schema_version']!r}"
        )
    if manifest["manifest_type"] != MANIFEST_TYPE:
        raise DataManifestError(
            f"unsupported manifest_type: {manifest['manifest_type']!r}"
        )
    _require_string(manifest["split"], "manifest.split")
    _require_string(manifest["corruption"], "manifest.corruption")
    class_order = manifest["class_order"]
    if type(class_order) is not list:
        raise DataManifestError("manifest.class_order must be an array")
    _validate_class_order(class_order)

    ann_root, ann_files = _validate_snapshot(manifest["annotations"], "annotations")
    image_root, image_files = _validate_snapshot(manifest["images"], "images")
    expected_alignment = _alignment(ann_files, image_files)
    alignment = _require_exact_object(
        manifest["alignment"],
        "alignment",
        {"stem_count", "stem_set_sha256", "stems_equal"},
    )
    if type(alignment["stem_count"]) is not int or alignment["stem_count"] <= 0:
        raise DataManifestError("alignment.stem_count must be a positive integer")
    if (
        type(alignment["stem_set_sha256"]) is not str
        or SHA256_RE.fullmatch(alignment["stem_set_sha256"]) is None
    ):
        raise DataManifestError("alignment.stem_set_sha256 is invalid")
    if type(alignment["stems_equal"]) is not bool or not alignment["stems_equal"]:
        raise DataManifestError("alignment.stems_equal must be true")
    if alignment != expected_alignment:
        raise DataManifestError("alignment does not match declared file entries")
    return ann_root, image_root


def _atomic_write(path: Path, content: bytes, *, overwrite: bool) -> None:
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise DataManifestError(
                    f"output appeared while scanning; refusing to overwrite: {path}"
                ) from error
            temporary.unlink()
        temporary = None
    except OSError as error:
        raise DataManifestError(f"cannot atomically write {path}: {error}") from error
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _stage_output(path: Path, content: bytes) -> tuple[Path, tuple[int, int]]:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            info = os.fstat(handle.fileno())
        return temporary, (info.st_dev, info.st_ino)
    except OSError as error:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        raise DataManifestError(
            f"cannot stage data manifest {path}: {error}"
        ) from error


def _unlink_created_output(path: Path, identity: tuple[int, int]) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    except OSError as error:
        raise DataManifestError(
            f"cannot inspect published data manifest output {path}: {error}"
        ) from error
    if (info.st_dev, info.st_ino) != identity:
        return
    try:
        path.unlink()
    except OSError as error:
        raise DataManifestError(
            f"cannot roll back data manifest output {path}: {error}"
        ) from error


def _fsync_output_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise DataManifestError(
            f"cannot open output directory {path}: {error}"
        ) from error
    try:
        os.fsync(descriptor)
    except OSError as error:
        raise DataManifestError(
            f"cannot fsync output directory {path}: {error}"
        ) from error
    finally:
        os.close(descriptor)


def _atomic_create_bundle(
    output_path: Path,
    content: bytes,
    detached_path: Path,
    sidecar: bytes,
    expected_digest: str,
) -> str:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise DataManifestError(
            f"cannot create output directory {output_path.parent}: {error}"
        ) from error
    staged: list[Path] = []
    published: list[tuple[Path, tuple[int, int]]] = []
    try:
        staged_sidecar, sidecar_identity = _stage_output(detached_path, sidecar)
        staged.append(staged_sidecar)
        staged_manifest, manifest_identity = _stage_output(output_path, content)
        staged.append(staged_manifest)
        for staged_path, target, identity, role in (
            (
                staged_sidecar,
                detached_path,
                sidecar_identity,
                "detached sidecar",
            ),
            (staged_manifest, output_path, manifest_identity, "output"),
        ):
            try:
                os.link(staged_path, target)
            except FileExistsError as error:
                raise DataManifestError(
                    f"{role} appeared while scanning; refusing to overwrite: {target}"
                ) from error
            except OSError as error:
                raise DataManifestError(
                    f"cannot publish {role} {target}: {error}"
                ) from error
            published.append((target, identity))
        _fsync_output_directory(output_path.parent)
        verified_digest = verify_manifest(output_path)
        if verified_digest != expected_digest:
            raise DataManifestError(
                "post-create data manifest verification changed its digest"
            )
        return verified_digest
    except BaseException:
        rollback_error: DataManifestError | None = None
        for target, identity in reversed(published):
            try:
                _unlink_created_output(target, identity)
            except DataManifestError as error:
                rollback_error = rollback_error or error
        try:
            _fsync_output_directory(output_path.parent)
        except DataManifestError as error:
            rollback_error = rollback_error or error
        if rollback_error is not None:
            raise rollback_error
        raise
    finally:
        for temporary in staged:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _prepare_output(path: Path, overwrite: bool, location: str) -> None:
    _require_string(path.name, f"{location} filename")
    _reject_symlink_components(path.parent, f"{location} parent")
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    except OSError as error:
        raise DataManifestError(f"cannot inspect {location} {path}: {error}") from error
    if stat.S_ISLNK(info.st_mode):
        raise DataManifestError(f"{location} must not be a symlink: {path}")
    if not stat.S_ISREG(info.st_mode):
        raise DataManifestError(f"{location} is not a regular file: {path}")
    if not overwrite:
        raise DataManifestError(
            f"{location} already exists; pass --overwrite to replace it: {path}"
        )


def _validate_output_location(path: Path, roots: Sequence[Path]) -> None:
    absolute = _absolute_lexical(path)
    try:
        resolved_parent = absolute.parent.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise DataManifestError(
            f"cannot resolve output parent {absolute.parent}: {error}"
        ) from error
    resolved = resolved_parent / absolute.name
    for root in roots:
        if _is_within(resolved, root):
            raise DataManifestError(
                f"output must not be inside a dataset root: {resolved} within {root}"
            )


def build_manifest(
    ann_root: Path,
    image_root: Path,
    split: str,
    corruption: str,
    class_order: Sequence[str],
    output: Path,
    *,
    overwrite: bool = False,
) -> str:
    ann_resolved = _resolve_root(ann_root, "ann-root")
    image_resolved = _resolve_root(image_root, "image-root")
    output_path = _absolute_lexical(output)
    detached_path = sidecar_path(output_path)
    _validate_output_location(output_path, (ann_resolved, image_resolved))
    _validate_output_location(detached_path, (ann_resolved, image_resolved))
    _prepare_output(output_path, overwrite, "output")
    _prepare_output(detached_path, overwrite, "detached sidecar")

    payload = _build_payload_from_resolved_roots(
        ann_resolved,
        image_resolved,
        split,
        corruption,
        class_order,
    )
    content = canonical_json_bytes(payload)
    digest = _sha256_bytes(content)
    sidecar = f"{digest}  {output_path.name}\n".encode("ascii")
    if overwrite:
        # Explicit overwrite is a pre-freeze snapshot refresh.  Publish the
        # sidecar first and the manifest last as the visible commit marker.
        _atomic_write(detached_path, sidecar, overwrite=True)
        _atomic_write(output_path, content, overwrite=True)
        return digest
    return _atomic_create_bundle(
        output_path,
        content,
        detached_path,
        sidecar,
        digest,
    )


def _require_regular_non_symlink(path: Path, location: str) -> None:
    try:
        info = path.lstat()
    except OSError as error:
        raise DataManifestError(f"cannot inspect {location} {path}: {error}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise DataManifestError(
            f"{location} must be a regular non-symlink file: {path}"
        )


def _verify_detached_hash(manifest_path: Path, content: bytes) -> str:
    detached = sidecar_path(manifest_path)
    digest = _sha256_bytes(content)
    expected = f"{digest}  {manifest_path.name}\n".encode("ascii")
    try:
        actual = detached.read_bytes()
    except OSError as error:
        raise DataManifestError(
            f"cannot read detached sidecar {detached}: {error}"
        ) from error
    if actual != expected:
        raise DataManifestError(f"detached SHA-256 mismatch for {manifest_path}")
    return digest


def verify_manifest(manifest: Path) -> str:
    manifest_path = _absolute_lexical(manifest)
    _require_regular_non_symlink(manifest_path, "manifest")
    _require_regular_non_symlink(sidecar_path(manifest_path), "detached sidecar")
    payload, content = _load_json(manifest_path)
    digest = _verify_detached_hash(manifest_path, content)
    if canonical_json_bytes(payload) != content:
        raise DataManifestError(f"manifest is not canonical JSON: {manifest_path}")

    ann_root, image_root = validate_payload(payload)
    _validate_output_location(manifest_path, (ann_root, image_root))
    _validate_output_location(sidecar_path(manifest_path), (ann_root, image_root))
    recomputed = _build_payload_from_resolved_roots(
        resolved_ann=ann_root,
        resolved_images=image_root,
        split=payload["split"],
        corruption=payload["corruption"],
        class_order=payload["class_order"],
    )
    if canonical_json_bytes(recomputed) != content:
        raise DataManifestError(
            "manifest no longer matches the current dataset directory contents"
        )
    return digest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build or verify a canonical byte-level dataset manifest without "
            "parsing annotation semantics."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="scan roots and write a manifest")
    build.add_argument("--ann-root", type=Path, required=True)
    build.add_argument("--image-root", type=Path, required=True)
    build.add_argument("--split", required=True)
    build.add_argument("--corruption", required=True)
    build.add_argument(
        "--class-order",
        nargs="+",
        required=True,
        help="ordered class names, as space-separated or comma-separated values",
    )
    build.add_argument("--output", type=Path, required=True)
    build.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing regular manifest and sidecar",
    )

    verify = subparsers.add_parser(
        "verify", help="verify sidecar, canonical schema, and current directory bytes"
    )
    verify.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            output = _absolute_lexical(args.output)
            digest = build_manifest(
                ann_root=args.ann_root,
                image_root=args.image_root,
                split=args.split,
                corruption=args.corruption,
                class_order=args.class_order,
                output=output,
                overwrite=args.overwrite,
            )
            result = {
                "manifest": str(output),
                "sha256": digest,
                "sidecar": str(sidecar_path(output)),
                "status": "built",
            }
        else:
            manifest = _absolute_lexical(args.manifest)
            digest = verify_manifest(manifest)
            result = {
                "manifest": str(manifest),
                "sha256": digest,
                "status": "verified",
            }
    except DataManifestError as error:
        print(f"data manifest error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
