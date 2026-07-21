import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


RSAR_CLASSES = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor')
DIOR_CLASSES = (
    'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
    'chimney', 'expressway-service-area', 'expressway-toll-station', 'dam',
    'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium',
    'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill',
)
CLASSES = RSAR_CLASSES
save_img = False

# Visually-similar RSAR class groups (SAR crops). Within a group SARCLIP's
# class disagreement is NOT reliable evidence (ship/car/tank/aircraft look
# alike in AABB SAR patches), so evidence_veto never fires inside a group.
# Used by CGA_FILTER_MODE=evidence_veto. Override via CGA_CONFUSION_GROUPS
# env (";"-separated groups, ","-separated class names).
RSAR_CONFUSION_GROUPS = (
    ("ship", "aircraft", "car", "tank"),
)
# Classes whose recognition depends on scene context an AABB crop can't
# capture; optionally exempt from veto via CGA_VETO_SKIP_CONTEXT=1.
RSAR_CONTEXT_CLASSES = ("bridge", "harbor")


def _prob_entropy_normalized(prob):
    """Shannon entropy of a prob vector, normalized to [0,1] by log(K)."""
    p = np.clip(np.asarray(prob, dtype=np.float64), 1e-12, 1.0)
    ent = -np.sum(p * np.log(p))
    k = len(p)
    return float(ent / np.log(k)) if k > 1 else 0.0


def _zscore_np(sim):
    """Per-row (class-dim) z-score, population std, eps floor. Mirrors
    prototype_cga._zscore for the text-only degenerate path."""
    sim = np.asarray(sim, dtype=np.float64)
    mean = sim.mean(axis=-1, keepdims=True)
    std = sim.std(axis=-1, keepdims=True)
    return (sim - mean) / (std + 1e-6)


def _prepend_sys_path(path):
    path = str(path)
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)


def _log_cga_info(message):
    print(message, flush=True)
    try:
        from mmrotate.utils import get_root_logger

        logger = get_root_logger()
        logger.warning(message)
    except Exception:
        pass


def _ensure_sarclip_importable():
    repo_root = Path(__file__).resolve().parents[1]
    search_roots = [repo_root]

    sarclip_dir = os.environ.get("SARCLIP_DIR")
    if sarclip_dir:
        override_root = Path(sarclip_dir).expanduser().resolve()
        if not override_root.exists():
            raise FileNotFoundError(f"SARCLIP_DIR does not exist: {override_root}")
        search_roots.insert(0, override_root)

    for root in reversed(search_roots):
        _prepend_sys_path(root)

    try:
        import sar_clip
    except ImportError as exc:
        raise ImportError(
            "Unable to import sar_clip. Expected the vendored package at "
            f"{repo_root / 'sar_clip'} or set SARCLIP_DIR explicitly."
        ) from exc

    print("[CGA/SARCLIP] sar_clip imported from:", sar_clip.__file__)
    return sar_clip


def _ensure_clip_importable():
    import clip
    print("[CGA/CLIP] clip imported from:", clip.__file__)
    return clip


def _normalize_templates(templates, backend):
    if templates is None:
        if backend == "clip":
            templates = ("an aerial image of a {}",)
        else:
            templates = ("A SAR image of a {}", "This SAR patch shows a {}")
    if isinstance(templates, str):
        templates = [templates]
    return list(templates)


def _normalize_backend(backend, model):
    backend = (backend or "").strip().lower()
    if backend in ("", "auto"):
        env_backend = os.environ.get("CGA_BACKEND") or os.environ.get("CGA_SCORER")
        backend = (env_backend or "").strip().lower()
    if backend in ("", "none", "false", "0", "raw"):
        model_name = str(model or "")
        if model_name in ("RN50x4", "RN50x16", "RN50x64", "ViT-B/16", "ViT-B/32", "ViT-L/14"):
            return "clip"
        return "sarclip"
    if backend in ("openai", "optical", "optical_clip"):
        return "clip"
    if backend in ("sar_clip", "sar-clip"):
        return "sarclip"
    return backend


def _parse_exclude_ids(value):
    if value is None or value.strip() == "":
        return None
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _env_float(name, default):
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return float(default)
    return float(value)


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return int(default)
    return int(value)


CGA_SHUFFLE_SCORE_BINS = (0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.000001)


def _shuffle_score_bin_indices(scores, score_bins=CGA_SHUFFLE_SCORE_BINS):
    """Map detector scores to fixed, half-open shuffle strata."""

    values = np.asarray(scores, dtype=np.float64)
    boundaries = np.asarray(score_bins, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("shuffle detector scores must be one-dimensional")
    if (boundaries.ndim != 1 or len(boundaries) < 2
            or not np.all(np.diff(boundaries) > 0.0)):
        raise ValueError("shuffle score bins must be strictly increasing")
    if not np.all(np.isfinite(values)):
        raise ValueError("shuffle detector scores must be finite")
    indices = np.searchsorted(boundaries, values, side="right") - 1
    return np.clip(indices, 0, len(boundaries) - 2).astype(np.int64)


def _stable_shuffle_order_key(base_seed, group_key, identity):
    payload = (
        f"{int(base_seed)}\0{group_key[0]}\0{group_key[1]}\0{identity}"
    ).encode("utf-8", errors="surrogatepass")
    return hashlib.sha256(payload).digest()


def stratified_shuffle_probability_vectors(
    probabilities,
    detector_scores,
    detector_labels,
    identities,
    *,
    seed=0,
    exclude_ids=(),
    score_bins=CGA_SHUFFLE_SCORE_BINS,
):
    """Deterministically derange complete probability vectors within strata.

    A stratum is defined by detector label and a fixed detector-score bin.  Its
    members are put in a stable, seeded order and cyclically shifted by one.
    Therefore every stratum with more than one member has no fixed source row;
    singleton and excluded rows remain unchanged.  Since detector labels are
    constant inside a stratum, moving complete vectors preserves both the
    vector multiset and the agreement/disagreement count exactly.

    Returns the shuffled vectors and a destination-to-source index mapping.
    """

    vectors = np.asarray(probabilities, dtype=np.float64)
    scores = np.asarray(detector_scores, dtype=np.float64)
    labels = np.asarray(detector_labels, dtype=np.int64)
    identities = [str(identity) for identity in identities]
    if vectors.ndim != 2 or vectors.shape[1] == 0:
        raise ValueError("shuffle probabilities must be a non-empty 2D matrix")
    if not (len(vectors) == len(scores) == len(labels) == len(identities)):
        raise ValueError(
            "shuffle probabilities, scores, labels and identities must align"
        )
    if not np.all(np.isfinite(vectors)):
        raise ValueError("shuffle probability vectors must be finite")
    if np.any(labels < 0) or np.any(labels >= vectors.shape[1]):
        raise ValueError("shuffle detector label is outside probability columns")

    bin_indices = _shuffle_score_bin_indices(scores, score_bins)
    excluded = {int(class_id) for class_id in exclude_ids}
    groups = {}
    for index, (label, bin_index) in enumerate(zip(labels, bin_indices)):
        if int(label) in excluded:
            continue
        groups.setdefault((int(label), int(bin_index)), []).append(index)

    source_indices = np.arange(len(vectors), dtype=np.int64)
    for group_key, group_indices in groups.items():
        if len(group_indices) <= 1:
            continue
        ordered = sorted(
            group_indices,
            key=lambda index: (
                _stable_shuffle_order_key(
                    seed, group_key, identities[index]), identities[index], index
            ),
        )
        destinations = np.asarray(ordered, dtype=np.int64)
        sources = np.roll(destinations, -1)
        source_indices[destinations] = sources

    shuffled = vectors[source_indices].copy()
    real_agreement = np.argmax(vectors, axis=1) == labels
    operative_agreement = np.argmax(shuffled, axis=1) == labels
    for group_indices in groups.values():
        indices = np.asarray(group_indices, dtype=np.int64)
        if int(real_agreement[indices].sum()) != int(
                operative_agreement[indices].sum()):
            raise RuntimeError(
                "stratified shuffle failed to preserve agreement count"
            )
    return shuffled, source_indices


def _new_cga_diag_window(num_classes):
    return {
        "calls": 0,
        "total": 0,
        "agree": 0,
        "dropped": 0,
        "blended": 0,
        "boosted": 0,
        "multiplied": 0,
        "penalized": 0,
        "threshold_dropped": 0,
        "shuffled": 0,
        "moved": 0,
        "unmoved": 0,
        "real_agree": 0,
        "operative_agree": 0,
        "reweighted": 0,
        "sem_weight_sum": 0.0,
        "label_probs": [],
        "pred_counts": np.zeros(num_classes, dtype=np.int64),
        "det_total": np.zeros(num_classes, dtype=np.int64),
        "det_agree": np.zeros(num_classes, dtype=np.int64),
        "det_drop": np.zeros(num_classes, dtype=np.int64),
    }


def _format_class_counts(counts, class_names):
    parts = []
    for idx, count in enumerate(counts):
        count = int(count)
        if count <= 0:
            continue
        name = class_names[idx] if idx < len(class_names) else str(idx)
        parts.append(f"{name}:{count}")
    return ",".join(parts) if parts else "none"


def _format_detector_diag(total, agree, dropped, class_names):
    parts = []
    for idx, count in enumerate(total):
        count = int(count)
        if count <= 0:
            continue
        name = class_names[idx] if idx < len(class_names) else str(idx)
        parts.append(
            f"{name}:n={count},agree={int(agree[idx])},drop={int(dropped[idx])}"
        )
    return ";".join(parts) if parts else "none"


def _format_label_prob_percentiles(values):
    if not values:
        return "none"
    qs = np.percentile(np.asarray(values, dtype=np.float32), [0, 25, 50, 75, 100])
    return (
        f"min={qs[0]:.4f},p25={qs[1]:.4f},p50={qs[2]:.4f},"
        f"p75={qs[3]:.4f},max={qs[4]:.4f}"
    )


def _normalize_optical_clip_model_name(model):
    aliases = {
        "ViT-B-16": "ViT-B/16",
        "ViT-B-32": "ViT-B/32",
        "ViT-L-14": "ViT-L/14",
    }
    return aliases.get(str(model), model)


def obb2xyxy(rbboxes):
    w = rbboxes[:, 2::5]
    h = rbboxes[:, 3::5]
    a = rbboxes[:, 4::5]
    cosa = np.abs(np.cos(a))
    sina = np.abs(np.sin(a))
    hbbox_w = cosa * w + sina * h
    hbbox_h = sina * w + cosa * h
    dx = rbboxes[..., 0]
    dy = rbboxes[..., 1]
    dw = hbbox_w.reshape(-1)
    dh = hbbox_h.reshape(-1)
    x1 = dx - dw / 2
    y1 = dy - dh / 2
    x2 = dx + dw / 2
    y2 = dy + dh / 2
    return np.stack((x1, y1, x2, y2), -1)


class CGA:
    def __init__(
        self,
        class_names,
        model="ViT-B-32",
        pretrained="/home/storageSDA1/Dataset/SARCLIP/ViT-B-32/vit_b_32_model.safetensors",
        cache_dir="/home/storageSDA1/Dataset/SARCLIP/ViT-B-32",
        precision="fp32",
        templates=None,
        tau=100.0,
        expand_ratio=0.4,
        force_grayscale=False,
        backend="auto",
    ):
        super().__init__()
        self.backend = _normalize_backend(backend, model)
        self.class_names = list(class_names)
        self.device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.save_path = "_clip_img"
        self.expand_ratio = float(expand_ratio)
        self.tau = float(tau)
        self.force_grayscale = bool(force_grayscale)
        self._first_call_logged = False
        templates = _normalize_templates(templates, self.backend)

        if self.backend == "sarclip":
            self._init_sarclip(model, pretrained, cache_dir, precision, templates)
        elif self.backend == "clip":
            self._init_optical_clip(model, templates)
        else:
            raise ValueError(f"Unsupported CGA backend: {self.backend}")

    def _init_sarclip(self, model, pretrained, cache_dir, precision, templates):
        sar_clip = _ensure_sarclip_importable()
        print(
            f"[CGA/SARCLIP] building model={model}, "
            f"pretrained={pretrained}, cache_dir={cache_dir}"
        )
        self.clip = sar_clip.create_model_with_args(
            model,
            pretrained=pretrained,
            precision=precision,
            device=str(self.device),
            cache_dir=cache_dir,
            output_dict=True,
        )
        lora_path = os.environ.get("SARCLIP_LORA")
        if lora_path:
            lora_path = os.path.expanduser(lora_path)
            if not os.path.exists(lora_path):
                raise FileNotFoundError(f"SARCLIP_LORA does not exist: {lora_path}")
            print(f"[CGA/SARCLIP] using SARCLIP_LORA={lora_path}")
            from sarclip_adapter import load_adapter_checkpoint
            adapter_info = load_adapter_checkpoint(
                self.clip,
                lora_path,
                map_location=self.device,
            )
            print(
                "[CGA/SARCLIP] loaded SARCLIP_LORA "
                f"adapter_type={adapter_info.get('adapter_type')}"
            )
        self.clip.eval()

        self.tokenizer = sar_clip.get_tokenizer(model, cache_dir=cache_dir)
        self.classifier = sar_clip.build_zero_shot_classifier(
            self.clip,
            tokenizer=self.tokenizer,
            classnames=self.class_names,
            templates=[lambda c, t=t: t.format(c) for t in templates],
            num_classes_per_batch=None,
            device=self.device,
            use_tqdm=False,
        )
        self.classifier = self.classifier / self.classifier.norm(dim=0, keepdim=True)

        preprocess_cfg = sar_clip.get_model_preprocess_cfg(self.clip)
        self.preprocess = sar_clip.image_transform(
            preprocess_cfg.get("size", 224),
            is_train=False,
            mean=preprocess_cfg.get("mean"),
            std=preprocess_cfg.get("std"),
            interpolation=preprocess_cfg.get("interpolation"),
            resize_mode=preprocess_cfg.get("resize_mode"),
            fill_color=preprocess_cfg.get("fill_color", 0),
        )
        print(f"[CGA/SARCLIP] init OK, classes={self.class_names}")

    def _init_optical_clip(self, model, templates):
        clip = _ensure_clip_importable()
        model = _normalize_optical_clip_model_name(model)
        print(f"[CGA/CLIP] building model={model}")
        self.clip, self.preprocess = clip.load(model, device=self.device)
        self.clip.eval()

        texts = [
            template.format(class_name)
            for class_name in self.class_names
            for template in templates
        ]
        prompts = clip.tokenize(texts).to(self.device)
        with torch.no_grad():
            text_features = self.clip.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.reshape(
                len(self.class_names), len(templates), -1)
            text_features = text_features.mean(dim=1)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self.classifier = text_features.T
        print(f"[CGA/CLIP] init OK, classes={self.class_names}")

    def _crop_patches(self, img_path, boxes, scores, labels, expand_ratio=None,
                      image=None):
        if expand_ratio is None:
            expand_ratio = self.expand_ratio
        if image is None:
            image_mode = "L" if self.force_grayscale else "RGB"
            image = Image.open(img_path).convert(image_mode)

        image_list = []
        ori_image_list = []
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            x1, y1, x2, y2 = box
            h, w = y2 - y1, x2 - x1
            x1 = max(0, x1 - w * expand_ratio)
            y1 = max(0, y1 - h * expand_ratio)
            x2 = x2 + w * expand_ratio
            y2 = y2 + h * expand_ratio

            sub_image = image.crop((int(x1), int(y1), int(x2), int(y2)))
            if save_img:
                label_idx = int(label)
                label_name = (
                    self.class_names[label_idx]
                    if 0 <= label_idx < len(self.class_names)
                    else label_idx
                )
                os.makedirs(self.save_path, exist_ok=True)
                sub_image.save(
                    os.path.join(self.save_path, f"sub_image_{i}_{score:.3f}_{label_name}.jpg")
                )

            ori_image_list.append(sub_image)
            image_list.append(self.preprocess(sub_image).to(self.device))

        if not image_list:
            return None, None
        return torch.stack(image_list, dim=0), ori_image_list

    @torch.no_grad()
    def __call__(self, img_path, boxes, scores, labels):
        if not self._first_call_logged:
            print(f"[CGA/{self.backend.upper()}] first call, num_boxes={len(boxes)}")
            self._first_call_logged = True

        images, ori_image_list = self._crop_patches(img_path, boxes, scores, labels)
        if images is None:
            return np.empty((0, len(self.class_names))), []

        if self.backend == "sarclip":
            out = self.clip(image=images)
            image_features = out["image_features"] if isinstance(out, dict) else out[0]
        else:
            image_features = self.clip.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = (self.tau * (image_features @ self.classifier)).softmax(dim=-1)
        return logits.detach().cpu().numpy(), ori_image_list

    @torch.no_grad()
    def forward_aabb_embed(self, img_path, boxes, scores, labels):
        """Legacy AABB crop + one frozen SARCLIP encode per proposal.

        The crop path is intentionally the exact ``_crop_patches`` path used by
        ``__call__`` (same expansion, PIL behavior, preprocess, templates and
        proposal order).  Prototype-CGA v2 consumes all three outputs from this
        one image-encoder pass.
        """
        images, _ = self._crop_patches(
            img_path, boxes, scores, labels)
        if images is None:
            num_classes = len(self.class_names)
            return (
                np.empty((0, 0)),
                np.empty((0, num_classes)),
                np.empty((0, num_classes)),
            )

        if self.backend == "sarclip":
            out = self.clip(image=images)
            image_features = (
                out["image_features"] if isinstance(out, dict) else out[0])
        else:
            image_features = self.clip.encode_image(images)
        image_features = image_features / image_features.norm(
            dim=-1, keepdim=True)
        text_sim = image_features @ self.classifier
        text_prob = (self.tau * text_sim).softmax(dim=-1)
        return (
            image_features.detach().cpu().numpy(),
            text_sim.detach().cpu().numpy(),
            text_prob.detach().cpu().numpy(),
        )

    def _score_images(self, images):
        if self.backend == "sarclip":
            out = self.clip(image=images)
            image_features = out["image_features"] if isinstance(out, dict) else out[0]
        else:
            image_features = self.clip.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = (self.tau * (image_features @ self.classifier)).softmax(dim=-1)
        return logits.detach().cpu().numpy()

    @torch.no_grad()
    def forward_rotated_embed(self, img_path, obboxes, context_ratio=0.15):
        """Rotated-aligned crop + single SARCLIP encode per proposal.

        For prototype_legacy ONLY. Returns, in the SAME order as ``obboxes``:
          image_embedding : (N, D) L2-normalized, detached numpy
          text_sim        : (N, C) cosine sim to text prototypes (=image_feat @ classifier)
          text_prob       : (N, C) softmax(tau * text_sim)
        One image encode per proposal (embedding and text prob share it).
        ``obboxes`` are (N,5) (cx,cy,w,h,angle). Strict: any degenerate crop
        raises (no silent skip), so counts stay aligned.
        """
        from .prototype_cga import rotated_align_crop

        image_mode = "L" if self.force_grayscale else "RGB"
        pil = Image.open(img_path).convert(image_mode)
        image_np = np.asarray(pil)

        tensors = []
        for row in obboxes:
            cx, cy, w, h, angle = (float(row[0]), float(row[1]), float(row[2]),
                                   float(row[3]), float(row[4]))
            patch_np = rotated_align_crop(
                image_np, cx, cy, w, h, angle, context_ratio=context_ratio)
            patch_pil = Image.fromarray(patch_np)
            if self.force_grayscale and patch_pil.mode != "L":
                patch_pil = patch_pil.convert("L")
            tensors.append(self.preprocess(patch_pil).to(self.device))

        if not tensors:
            C = len(self.class_names)
            return (np.empty((0, 0)), np.empty((0, C)), np.empty((0, C)))

        images = torch.stack(tensors, dim=0)
        if self.backend == "sarclip":
            out = self.clip(image=images)
            image_features = out["image_features"] if isinstance(out, dict) else out[0]
        else:
            image_features = self.clip.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_sim = image_features @ self.classifier
        text_prob = (self.tau * text_sim).softmax(dim=-1)
        return (
            image_features.detach().cpu().numpy(),
            text_sim.detach().cpu().numpy(),
            text_prob.detach().cpu().numpy(),
        )

    def text_prototype_matrix(self):
        """(C, D) normalized text prototypes (classifier is (D, C))."""
        return self.classifier.detach().cpu().numpy().T

    @torch.no_grad()
    def forward_views(self, img_path, boxes, scores, labels, expand_ratios):
        """Score every box at several crop scales (multi-view SARCLIP).

        Opens the source image once, crops each box at each expand_ratio, and
        returns a (V, N, C) softmax array where V=len(expand_ratios), N=#boxes,
        C=#classes. Used by reliability_gated_mv to build a cross-scale
        consistency reliability score for RSAR AABB crops.
        """
        if not self._first_call_logged:
            print(
                f"[CGA/{self.backend.upper()}] first multi-view call, "
                f"num_boxes={len(boxes)}, views={list(expand_ratios)}"
            )
            self._first_call_logged = True

        image_mode = "L" if self.force_grayscale else "RGB"
        image = Image.open(img_path).convert(image_mode)

        num_classes = len(self.class_names)
        view_logits = []
        for expand_ratio in expand_ratios:
            images, _ = self._crop_patches(
                img_path, boxes, scores, labels,
                expand_ratio=float(expand_ratio), image=image)
            if images is None:
                view_logits.append(np.empty((0, num_classes)))
            else:
                view_logits.append(self._score_images(images))
        return np.stack(view_logits, axis=0)


class TestMixins:
    def __init__(self):
        self.cga = None

    def _get_cga_class_names(self, num_classes):
        class_names = getattr(self, "CLASSES", None)
        if class_names is not None and len(class_names) == num_classes:
            return list(class_names)
        if num_classes == len(DIOR_CLASSES):
            return list(DIOR_CLASSES)
        if num_classes == len(RSAR_CLASSES):
            return list(RSAR_CLASSES)
        return [str(i) for i in range(num_classes)]

    def _build_veto_groups(self, class_names):
        """Precompute class-id -> confusion-group-id and context-class-id set
        used by evidence_veto. Names not present in class_names are ignored."""
        name_to_id = {n: i for i, n in enumerate(class_names)}
        groups_env = os.environ.get("CGA_CONFUSION_GROUPS")
        if groups_env:
            groups = [tuple(g.split(",")) for g in groups_env.split(";") if g.strip()]
        else:
            groups = RSAR_CONFUSION_GROUPS
        self._veto_group_of = {}
        for gid, group in enumerate(groups):
            for name in group:
                cid = name_to_id.get(name.strip())
                if cid is not None:
                    self._veto_group_of[cid] = gid
        self._veto_context_ids = {
            name_to_id[n] for n in RSAR_CONTEXT_CLASSES if n in name_to_id
        }

    def _build_cga(self, num_classes):
        scorer = os.environ.get("CGA_SCORER", "").strip().lower()
        backend = os.environ.get("CGA_BACKEND", scorer).strip().lower()
        backend = _normalize_backend(backend, None)
        class_names = self._get_cga_class_names(num_classes)

        templates_env = os.environ.get("CGA_TEMPLATES")
        tau = float(os.environ.get("CGA_TAU", "100.0"))
        expand_ratio = float(os.environ.get("CGA_EXPAND_RATIO", "0.4"))
        force_grayscale = os.environ.get("CGA_FORCE_GRAYSCALE", "0").lower() in (
            "1",
            "true",
            "yes",
        )
        self.cga_filter_mode = os.environ.get("CGA_FILTER_MODE", "legacy").strip().lower()
        self.cga_gate_prob_thr = _env_float("CGA_GATE_PROB_THR", 0.5)
        self.cga_drop_score = _env_float("CGA_DROP_SCORE", 0.0)
        self.cga_blend_detector_weight = _env_float("CGA_BLEND_DET_WEIGHT", 0.7)
        self.cga_disagree_delta = _env_float("CGA_DISAGREE_DELTA", 0.1)
        if self.cga_disagree_delta < 0.0:
            raise ValueError("CGA_DISAGREE_DELTA must be non-negative")
        self.cga_disagree_score_thr = _env_float("CGA_DISAGREE_SCORE_THR", 0.95)
        self.cga_shuffle_seed = _env_int("CGA_SHUFFLE_SEED", 0)
        # veto_soft params: only drop when SARCLIP confidently disagrees, and
        # never drop detector-confident boxes (protects real targets buried in
        # chaff clouds that SARCLIP can no longer recognize).
        self.cga_veto_pred_thr = _env_float("CGA_VETO_PRED_THR", 0.7)
        self.cga_veto_label_thr = _env_float("CGA_VETO_LABEL_THR", 0.1)
        self.cga_protect_det_score = _env_float("CGA_PROTECT_DET_SCORE", 0.9)
        self.cga_boost_det_thr = _env_float("CGA_BOOST_DET_THR", 0.75)
        self.cga_boost_clip_thr = _env_float("CGA_BOOST_CLIP_THR", 0.8)
        self.cga_boost_strength = _env_float("CGA_BOOST_STRENGTH", 0.7)
        # adaptive_blend: detector-trust weight ramps linearly from w_min (at
        # det_score=0) to w_max (at det_score=1). Protects confident boxes,
        # lets SARCLIP veto marginal ones harder.
        self.cga_adapt_w_min = _env_float("CGA_ADAPT_W_MIN", 0.3)
        self.cga_adapt_w_max = _env_float("CGA_ADAPT_W_MAX", 0.95)
        # evidence_veto: only downweight on RELIABLE, low-uncertainty opposition.
        self.cga_veto_pred_hi = _env_float("CGA_VETO_PRED_HI", 0.90)   # SARCLIP conf on its own class
        self.cga_veto_label_lo = _env_float("CGA_VETO_LABEL_LO", 0.05)  # SARCLIP conf on detector class
        self.cga_veto_margin = _env_float("CGA_VETO_MARGIN", 0.60)      # top1-top2 margin
        self.cga_veto_entropy = _env_float("CGA_VETO_ENTROPY", 0.35)    # normalized entropy ceiling
        self.cga_veto_penalty = _env_float("CGA_VETO_PENALTY", 0.0)     # multiplicative factor (0=drop)
        self.cga_veto_skip_context = os.environ.get("CGA_VETO_SKIP_CONTEXT", "0").lower() in ("1", "true", "yes")
        # semantic_reweight (SRW): SARCLIP never removes or rescores a pseudo
        # box; the detector score alone controls admission. On disagreement we
        # compute a semantic reliability weight that ramps from (1 - lambda) at
        # low_thr up to 1.0 at high_thr, so near-threshold disagreements are
        # softened while high-confidence detector boxes are fully trusted.
        self.cga_sem_low_thr = _env_float("CGA_SEM_LOW_THR", 0.90)
        self.cga_sem_high_thr = _env_float("CGA_SEM_HIGH_THR", 0.95)
        self.cga_sem_lambda = _env_float("CGA_SEM_LAMBDA", 0.50)
        if self.cga_sem_high_thr <= self.cga_sem_low_thr:
            raise ValueError(
                "CGA_SEM_HIGH_THR must be strictly greater than CGA_SEM_LOW_THR")
        # reliability_gated / reliability_gated_mv: gate the downweight strength
        # by how RELIABLE SARCLIP's opposition is (r in [0,1]), so a bigger
        # lambda only bites when the semantic evidence is stable and confident.
        #   single-view: r = p_top1 * margin * (1 - H_norm)
        #   multi-view : r = 1[all view top1 agree] * mean(p_top1) * (1-mean H)
        #   w = 1 - lambda_gated * 1[det != clip] * r * g(s)
        #   g(s) = clip((high - s)/(high - low), 0, 1)  (protects confident dets)
        self.cga_sem_gated_lambda = _env_float("CGA_SEM_GATED_LAMBDA", 0.80)
        # reliability_gated_legacy: fire the FULL legacy score-blend, but only on
        # a RELIABLE disagreement below the high-confidence cap. Few boxes fire,
        # but each firing has legacy's full training-chain effect (blended score
        # can fall under the 0.9 admission threshold -> box removed from RPN, ROI
        # cls AND bbox reg). reliability r = opposition * (1 - H_norm),
        # opposition = clip((p_top1 - p_det_label)/0.5, 0, 1). tau=0 => cap-only
        # ablation (blend every disagreement with s_det < high_thr).
        self.cga_rel_legacy_tau = _env_float("CGA_REL_LEGACY_TAU", 0.10)
        # prototype_legacy: online per-class VISUAL prototypes in frozen SARCLIP
        # space, fused with TEXT prototypes to re-score, then plain-legacy blend.
        self.cga_proto_beta = _env_float("CGA_PROTO_BETA", 0.50)
        self.cga_proto_momentum = _env_float("CGA_PROTO_MOMENTUM", 0.95)
        self.cga_proto_min_count = _env_int("CGA_PROTO_MIN_COUNT", 20)
        self.cga_proto_context_ratio = _env_float("CGA_PROTO_CONTEXT_RATIO", 0.15)
        self.cga_proto_rotated_crop = os.environ.get(
            "CGA_PROTO_ROTATED_CROP", "1").lower() in ("1", "true", "yes")
        self.cga_proto_score_thr = _env_float("CGA_PROTO_SCORE_THR", 0.97)
        self.cga_proto_iou_thr = _env_float("CGA_PROTO_IOU_THR", 0.70)
        self.cga_strict = os.environ.get(
            "CGA_STRICT", "0").lower() in ("1", "true", "yes")
        view_ratios_env = os.environ.get("CGA_SEM_VIEW_RATIOS", "0.0,0.25,0.5")
        self.cga_sem_view_ratios = tuple(
            float(v.strip()) for v in view_ratios_env.split(",") if v.strip())
        if len(self.cga_sem_view_ratios) < 2:
            raise ValueError("CGA_SEM_VIEW_RATIOS needs >=2 comma-separated ratios")
        self._build_veto_groups(class_names)
        self.cga_filter_log_every = _env_int("CGA_FILTER_LOG_EVERY", 500)
        lora_path = os.environ.get("SARCLIP_LORA", "").strip()
        _log_cga_info(
            "[CGA] init "
            f"scorer={scorer or '<unset>'}, "
            f"backend={backend}, "
            f"lora={os.path.expanduser(lora_path) if lora_path else '<unset>'}, "
            f"filter_mode={self.cga_filter_mode}, "
            f"gate_prob_thr={self.cga_gate_prob_thr}, "
            f"drop_score={self.cga_drop_score}, "
            f"disagree_delta={self.cga_disagree_delta}, "
            f"disagree_score_thr={self.cga_disagree_score_thr}, "
            f"shuffle_seed={self.cga_shuffle_seed}, "
            f"veto_pred_thr={self.cga_veto_pred_thr}, "
            f"veto_label_thr={self.cga_veto_label_thr}, "
            f"protect_det_score={self.cga_protect_det_score}, "
            f"boost_det_thr={self.cga_boost_det_thr}, "
            f"boost_clip_thr={self.cga_boost_clip_thr}, "
            f"boost_strength={self.cga_boost_strength}"
        )

        if backend == "clip":
            model = os.environ.get("CLIP_MODEL", os.environ.get("CGA_CLIP_MODEL", "RN50x64"))
            templates = (templates_env or os.environ.get(
                "CLIP_TEMPLATES", "an aerial image of a {}")).split(";")
            cga = CGA(
                class_names,
                model=model,
                templates=templates,
                tau=tau,
                expand_ratio=expand_ratio,
                force_grayscale=force_grayscale,
                backend="clip",
            )
        elif backend == "sarclip":
            model = os.environ.get("SARCLIP_MODEL", "ViT-B-32")
            pretrained = os.environ.get(
                "SARCLIP_PRETRAINED",
                "/home/storageSDA1/Dataset/SARCLIP/ViT-B-32/vit_b_32_model.safetensors",
            )
            cache_dir = os.environ.get(
                "SARCLIP_CACHE_DIR",
                "/home/storageSDA1/Dataset/SARCLIP/ViT-B-32",
            )
            precision = os.environ.get("SARCLIP_PRECISION", "fp32")
            templates = (templates_env or
                         "A SAR image of a {};This SAR patch shows a {}").split(";")
            cga = CGA(
                class_names,
                model=model,
                pretrained=pretrained,
                cache_dir=cache_dir,
                precision=precision,
                templates=templates,
                tau=tau,
                expand_ratio=expand_ratio,
                force_grayscale=force_grayscale,
                backend="sarclip",
            )
        else:
            raise ValueError(f"Unsupported CGA backend: {backend}")

        exclude_ids = _parse_exclude_ids(os.environ.get("CGA_EXCLUDE_IDS"))
        if exclude_ids is None:
            exclude_ids = [7, 8, 11] if backend == "clip" and num_classes == len(DIOR_CLASSES) else []
        return cga, exclude_ids

    @staticmethod
    def _flatten_cga_inputs(image_results):
        boxes, scores, labels = [], [], []
        for class_id, result in enumerate(image_results):
            if len(result) == 0:
                continue
            result_xyxy = obb2xyxy(result)
            boxes.append(result_xyxy[:, :4])
            scores.append(result[:, -1])
            labels.append([class_id] * len(result))
        if not boxes:
            return None
        return (
            np.concatenate(boxes, axis=0),
            np.concatenate(scores, axis=0),
            np.concatenate(labels, axis=0),
        )

    @staticmethod
    def _flatten_cga_obb(image_results):
        """Original OBB (cx,cy,w,h,angle), raw score, label in the SAME flat
        order as _flatten_cga_inputs (per-class concat). Used by prototype_legacy
        to keep the rotated box (never obb2xyxy'd) and the raw detector score."""
        obb, scores, labels = [], [], []
        for class_id, result in enumerate(image_results):
            if len(result) == 0:
                continue
            obb.append(np.asarray(result[:, :5], dtype=np.float64))
            scores.append(np.asarray(result[:, -1], dtype=np.float64))
            labels.append(np.full(len(result), class_id, dtype=np.int64))
        if not obb:
            return None
        return (np.concatenate(obb, 0), np.concatenate(scores, 0),
                np.concatenate(labels, 0))

    def _get_proto_bank(self, num_classes):
        """Lazily create the per-run visual prototype bank + pending stash."""
        bank = getattr(self, "_proto_bank", None)
        if bank is None or bank.num_classes != num_classes:
            from .prototype_cga import VisualPrototypeBank
            bank = VisualPrototypeBank(
                num_classes,
                momentum=getattr(self, "cga_proto_momentum", 0.95),
                min_count=getattr(self, "cga_proto_min_count", 20))
            self._proto_bank = bank
            self._proto_cur_iter = 0
            self._proto_diag = {
                "fallback_count": 0,
                "prototype_update_error_count": 0,
                "nan_inf_count": 0,
                "alignment_error_count": 0,
                "top1_change": 0,
                "top1_total": 0,
                "det_agree_text": 0,
                "det_agree_fused": 0,
                "blended": 0,
                "legacy_blended": 0,
                "v2_blended": 0,
                "text_det_prob_sum": 0.0,
                "fused_det_prob_sum": 0.0,
                "det_prob_count": 0,
                # Fixed-memory streaming histogram over probability deltas in
                # [-1, 1].  Keeping every proposal delta made the diagnostic
                # consume hundreds of MB and repeatedly scan all history.
                "det_prob_delta_hist": np.zeros(4097, dtype=np.int64),
                "legacy_admitted": 0,
                "v2_admitted": 0,
                "legacy_hits": 0,
                "v2_hits": 0,
                "v2_newly_deleted": 0,
                "v2_newly_retained": 0,
            }
        return bank

    def reset_proto_pending(self):
        """Clear the per-iteration weak-proposal stash (called by teacher)."""
        self._proto_pending = []

    def _proto_strict(self):
        return getattr(self, "cga_strict", False)

    def _refine_single_prototype(self, image_results, img_meta):
        """prototype_legacy: rotated-crop embed -> text+visual fused prob ->
        plain-legacy blend. Scores use the prototype snapshot from BEFORE this
        iteration's update (teacher updates the bank at iteration end). Stashes
        per-image weak proposals (obb, raw score, label, embedding) for the
        teacher's weak/strong prototype-candidate matching."""
        from .prototype_cga import fuse_logits, softmax

        num_classes = len(image_results)
        obb_inputs = self._flatten_cga_obb(image_results)
        if obb_inputs is None:
            return image_results
        obb_list, raw_scores, labels_list = obb_inputs
        # RAW detector score saved separately; never overwritten before use.
        raw_scores = raw_scores.copy()

        if getattr(self, "cga", None) is None:
            self.cga, self.exclude_ids = self._build_cga(num_classes)
        bank = self._get_proto_bank(num_classes)
        diag = self._proto_diag
        det_weight = getattr(self, "cga_blend_detector_weight", 0.7)
        beta = getattr(self, "cga_proto_beta", 0.50)
        ctx = getattr(self, "cga_proto_context_ratio", 0.15)
        strict = self._proto_strict()

        filename = img_meta["filename"]
        try:
            embed, text_sim, text_prob = self.cga.forward_rotated_embed(
                filename, obb_list, context_ratio=ctx)
        except Exception as e:
            diag["alignment_error_count"] += 1
            if strict:
                raise
            _log_cga_info(f"[CGA][proto][WARN] rotated embed failed: {repr(e)}")
            diag["fallback_count"] += 1
            return image_results

        if embed.shape[0] != len(obb_list):
            diag["alignment_error_count"] += 1
            if strict:
                raise RuntimeError(
                    f"prototype: embed count {embed.shape[0]} != obb "
                    f"{len(obb_list)}")
            return image_results
        if bank.embed_dim is None:
            bank.embed_dim = embed.shape[1]
        if not (np.all(np.isfinite(embed)) and np.all(np.isfinite(text_sim))):
            diag["nan_inf_count"] += 1
            if strict:
                raise RuntimeError("prototype: NaN/Inf in embed/text_sim")
            return image_results

        # Fuse using the OLD prototype snapshot (no self-inclusion).
        active = bank.active_mask()
        if active.any():
            proto_mat = bank.matrix()                     # (C, D)
            sim_visual = embed @ proto_mat.T              # (N, C) cosine (both normed)
        else:
            sim_visual = np.zeros_like(text_sim)
        fused_logits = fuse_logits(text_sim, sim_visual, active, beta)
        if not np.all(np.isfinite(fused_logits)):
            diag["nan_inf_count"] += 1
            if strict:
                raise RuntimeError("prototype: NaN/Inf in fused logits")
            return image_results
        fused_prob = softmax(fused_logits, axis=-1)
        text_only_prob = softmax(_zscore_np(text_sim), axis=-1)

        fused_pred = fused_prob.argmax(axis=1)
        text_pred = text_only_prob.argmax(axis=1)

        # Plain-legacy blend, but with fused_prob replacing text prob.
        new_scores = raw_scores.copy()
        for i in range(len(obb_list)):
            det_label = int(labels_list[i])
            diag["top1_total"] += 1
            if fused_pred[i] != text_pred[i]:
                diag["top1_change"] += 1
            if text_pred[i] == det_label:
                diag["det_agree_text"] += 1
            if fused_pred[i] == det_label:
                diag["det_agree_fused"] += 1
            if int(fused_pred[i]) != det_label:
                new_scores[i] = (
                    det_weight * float(raw_scores[i])
                    + (1.0 - det_weight) * float(fused_prob[i, det_label]))
                diag["blended"] += 1

        # Stash weak proposals for the teacher's candidate matching.
        pending = getattr(self, "_proto_pending", None)
        if pending is None:
            self._proto_pending = pending = []
        pending.append({
            "filename": filename,
            "obb": obb_list,
            "raw_score": raw_scores,
            "label": labels_list,
            "embed": embed,
        })

        # Write blended scores back into image_results in the same flat order.
        j = 0
        for cls in range(num_classes):
            n = len(image_results[cls])
            for k in range(n):
                image_results[cls][k, -1] = new_scores[j]
                j += 1
        return image_results

    def _refine_single_prototype_v2(self, image_results, img_meta):
        """AABB-calibrated prototype-space fusion with plain-legacy scoring.

        This is an independent path.  It does not call the v1 rotated crop,
        ``_zscore_np`` or ``prototype_cga.fuse_logits``.
        """
        from .prototype_cga import (
            legacy_score_blend,
            prototype_legacy_v2_probabilities,
        )

        # One pending slot per weak image prevents batch alignment from shifting
        # when an earlier image has no proposals or takes a non-strict fallback.
        pending = getattr(self, "_proto_pending", None)
        if pending is None:
            self._proto_pending = pending = []
        pending.append(None)
        pending_index = len(pending) - 1

        num_classes = len(image_results)
        aabb_inputs = self._flatten_cga_inputs(image_results)
        if aabb_inputs is None:
            return image_results
        boxes, raw_scores, labels = aabb_inputs
        raw_scores = np.asarray(raw_scores, dtype=np.float64).copy()
        labels = np.asarray(labels, dtype=np.int64)

        obb_inputs = self._flatten_cga_obb(image_results)
        if getattr(self, "cga", None) is None:
            self.cga, self.exclude_ids = self._build_cga(num_classes)
        bank = self._get_proto_bank(num_classes)
        diag = self._proto_diag
        strict = self._proto_strict()

        try:
            if obb_inputs is None:
                raise RuntimeError("prototype_v2: missing OBB inputs")
            obb, obb_scores, obb_labels = obb_inputs
            if (len(obb) != len(boxes)
                    or not np.array_equal(obb_labels, labels)
                    or not np.allclose(
                        obb_scores, raw_scores, rtol=0.0, atol=0.0)):
                raise RuntimeError(
                    "prototype_v2: AABB/OBB proposal alignment mismatch")

            embedding, text_sim, text_prob = self.cga.forward_aabb_embed(
                img_meta["filename"], boxes, raw_scores, labels)
            expected_prob_shape = (len(boxes), num_classes)
            if (embedding.ndim != 2 or embedding.shape[0] != len(boxes)
                    or text_sim.shape != expected_prob_shape
                    or text_prob.shape != expected_prob_shape):
                raise RuntimeError(
                    "prototype_v2: SARCLIP output/proposal alignment mismatch")
            if not (np.all(np.isfinite(embedding))
                    and np.all(np.isfinite(text_sim))
                    and np.all(np.isfinite(text_prob))):
                raise RuntimeError(
                    "prototype_v2: NaN/Inf in embedding/text outputs")

            if bank.embed_dim is None:
                bank.embed_dim = embedding.shape[1]
            elif bank.embed_dim != embedding.shape[1]:
                raise RuntimeError(
                    "prototype_v2: prototype/embedding dimension mismatch")

            active = bank.active_mask()
            text_prototypes = self.cga.text_prototype_matrix()
            visual_prototypes = bank.matrix()
            _, fused_sim, _, fused_prob = (
                prototype_legacy_v2_probabilities(
                    embedding,
                    text_prototypes,
                    visual_prototypes,
                    active,
                    tau=self.cga.tau,
                    beta=getattr(self, "cga_proto_beta", 0.50),
                    text_sim=text_sim,
                    text_prob=text_prob,
                ))

            det_weight = getattr(self, "cga_blend_detector_weight", 0.70)
            exclude_ids = getattr(self, "exclude_ids", ())
            legacy_scores, text_pred, legacy_blended = legacy_score_blend(
                text_prob, raw_scores, labels,
                detector_weight=det_weight, exclude_ids=exclude_ids)
            v2_scores, fused_pred, v2_blended = legacy_score_blend(
                fused_prob, raw_scores, labels,
                detector_weight=det_weight, exclude_ids=exclude_ids)
        except Exception as error:
            if "NaN/Inf" in str(error):
                diag["nan_inf_count"] += 1
            else:
                diag["alignment_error_count"] += 1
            if strict:
                raise
            diag["fallback_count"] += 1
            _log_cga_info(
                f"[CGA][proto_v2][WARN] fallback: {error!r}")
            return image_results

        row = np.arange(len(labels), dtype=np.int64)
        text_det_prob = text_prob[row, labels]
        fused_det_prob = fused_prob[row, labels]
        diag["top1_total"] += len(labels)
        diag["top1_change"] += int(np.count_nonzero(text_pred != fused_pred))
        diag["det_agree_text"] += int(np.count_nonzero(text_pred == labels))
        diag["det_agree_fused"] += int(np.count_nonzero(fused_pred == labels))
        diag["legacy_blended"] += int(legacy_blended.sum())
        diag["v2_blended"] += int(v2_blended.sum())
        diag["blended"] += int(v2_blended.sum())
        diag["text_det_prob_sum"] += float(text_det_prob.sum())
        diag["fused_det_prob_sum"] += float(fused_det_prob.sum())
        diag["det_prob_count"] += len(labels)
        delta = np.asarray(
            fused_det_prob - text_det_prob, dtype=np.float64)
        delta_hist = diag["det_prob_delta_hist"]
        delta_index = np.rint(
            (np.clip(delta, -1.0, 1.0) + 1.0)
            * (len(delta_hist) - 1) / 2.0).astype(np.int64)
        delta_hist += np.bincount(
            delta_index, minlength=len(delta_hist))

        # Plain legacy writes its blended values back into the detector result
        # dtype (normally float32).  Mirror that cast before paired admission
        # diagnostics so threshold-edge counts describe the actual outputs.
        score_dtype = next(
            result.dtype for result in image_results if len(result))
        legacy_scores_written = np.asarray(
            legacy_scores, dtype=score_dtype)
        v2_scores_written = np.asarray(v2_scores, dtype=score_dtype)

        pending[pending_index] = {
            "filename": img_meta["filename"],
            "boxes": np.asarray(boxes).copy(),
            "obb": np.asarray(obb).copy(),
            "raw_score": raw_scores.copy(),
            "label": labels.copy(),
            "embed": np.asarray(embedding).copy(),
            "text_sim": np.asarray(text_sim).copy(),
            "fused_sim": np.asarray(fused_sim).copy(),
            "text_prob": np.asarray(text_prob).copy(),
            "fused_prob": np.asarray(fused_prob).copy(),
            "legacy_score": legacy_scores_written.copy(),
            "v2_score": v2_scores_written.copy(),
            "active_mask": np.asarray(active, dtype=bool).copy(),
        }

        flat_index = 0
        for class_results in image_results:
            num_detections = len(class_results)
            if num_detections:
                class_results[:, -1] = v2_scores_written[
                    flat_index:flat_index + num_detections]
            flat_index += num_detections
        return image_results

    @staticmethod
    def _validate_cga_logits(logits, labels, num_classes):
        probabilities = np.asarray(logits, dtype=np.float64)
        expected = (len(labels), num_classes)
        if probabilities.shape != expected:
            raise ValueError(
                "CGA scorer returned an unexpected probability shape: "
                f"got {probabilities.shape}, expected {expected}"
            )
        return probabilities

    @staticmethod
    def _empty_cga_meta(num_classes):
        """Per-class empty arrays, aligned to an image with no detections."""
        return {
            "semantic_weight": [np.ones(0, dtype=np.float64)
                                for _ in range(num_classes)],
            "agreement": [np.zeros(0, dtype=bool) for _ in range(num_classes)],
            "det_score": [np.zeros(0, dtype=np.float64)
                          for _ in range(num_classes)],
            "det_label": [np.zeros(0, dtype=np.int64)
                          for _ in range(num_classes)],
            "clip_top1": [np.zeros(0, dtype=np.int64)
                          for _ in range(num_classes)],
            "label_prob": [np.zeros(0, dtype=np.float64)
                           for _ in range(num_classes)],
            "top1_prob": [np.zeros(0, dtype=np.float64)
                          for _ in range(num_classes)],
            "margin": [np.zeros(0, dtype=np.float64)
                       for _ in range(num_classes)],
            "entropy": [np.zeros(0, dtype=np.float64)
                        for _ in range(num_classes)],
            "reliability": [np.zeros(0, dtype=np.float64)
                            for _ in range(num_classes)],
        }

    def refine_test(self, results, img_metas, return_cga_meta=False):
        if len(results) != len(img_metas):
            raise ValueError(
                "CGA batch mismatch: "
                f"len(results)={len(results)} != len(img_metas)={len(img_metas)}"
            )
        if not results:
            return (results, []) if return_cga_meta else results

        num_classes = len(results[0])
        for image_index, image_results in enumerate(results):
            if len(image_results) != num_classes:
                raise ValueError(
                    "CGA class-count mismatch inside batch: "
                    f"image 0 has {num_classes}, image {image_index} has "
                    f"{len(image_results)}"
                )

        mode = getattr(self, "cga_filter_mode", "legacy")
        if mode != "shuffled_legacy":
            metas = []
            for image_results, img_meta in zip(results, img_metas):
                out = self._refine_single(
                    image_results, img_meta, collect_meta=return_cga_meta)
                if return_cga_meta:
                    _, meta = out
                    metas.append(meta)
            if return_cga_meta:
                return results, metas
            return results

        # Shuffled-SARCLIP is defined over every detection in the current
        # inference batch.  Score each image normally first, then move complete
        # probability vectors across images inside label/score strata.
        contexts = []
        all_scores, all_labels, all_probabilities, all_identities = [], [], [], []
        for image_results, img_meta in zip(results, img_metas):
            inputs = self._flatten_cga_inputs(image_results)
            if inputs is None:
                continue
            boxes, scores, labels = inputs
            if getattr(self, "cga", None) is None:
                self.cga, self.exclude_ids = self._build_cga(num_classes)
            filename = img_meta["filename"]
            logits, _ = self.cga(filename, boxes, scores, labels)
            probabilities = self._validate_cga_logits(
                logits, labels, num_classes)
            start = sum(len(item) for item in all_scores)
            stop = start + len(scores)
            contexts.append(
                (image_results, img_meta, inputs, probabilities, start, stop)
            )
            all_scores.append(scores)
            all_labels.append(labels)
            all_probabilities.append(probabilities)
            all_identities.extend(
                f"{filename}\0{local_index}"
                for local_index in range(len(scores))
            )

        if not contexts:
            if return_cga_meta:
                return results, [self._empty_cga_meta(num_classes)
                                 for _ in results]
            return results
        scores = np.concatenate(all_scores, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        probabilities = np.concatenate(all_probabilities, axis=0)
        operative, source_indices = stratified_shuffle_probability_vectors(
            probabilities,
            scores,
            labels,
            all_identities,
            seed=getattr(self, "cga_shuffle_seed", 0),
            exclude_ids=getattr(self, "exclude_ids", ()),
        )
        moved = source_indices != np.arange(len(source_indices))
        initial_calls = getattr(self, "_cga_filter_calls", 0)
        log_every = getattr(self, "cga_filter_log_every", 500)
        future_calls = range(initial_calls + 1, initial_calls + len(contexts) + 1)
        log_after_batch = (
            1 in future_calls
            or (log_every > 0 and any(
                call % log_every == 0 for call in future_calls))
        )
        for context_index, (
            image_results, img_meta, inputs, real, start, stop
        ) in enumerate(contexts):
            is_last = context_index == len(contexts) - 1
            self._refine_single(
                image_results,
                img_meta,
                precomputed_inputs=inputs,
                precomputed_logits=real,
                operative_logits=operative[start:stop],
                shuffle_moved=moved[start:stop],
                defer_log=not is_last,
                force_log=is_last and log_after_batch,
            )
        if return_cga_meta:
            # shuffled_legacy does not collect per-box meta; return empty
            # per-image structures so the interface stays uniform.
            return results, [self._empty_cga_meta(num_classes)
                             for _ in results]
        return results

    def _refine_single(
        self,
        image_results,
        img_meta,
        *,
        precomputed_inputs=None,
        precomputed_logits=None,
        operative_logits=None,
        shuffle_moved=None,
        defer_log=False,
        force_log=False,
        collect_meta=False,
    ):
        """Apply CGA to one image while sharing diagnostics across the batch.

        When ``collect_meta`` is True, returns ``(image_results, meta)`` where
        ``meta`` is a dict whose per-class lists align 1:1 with
        ``image_results[class][box]`` (same order the scores are written back).
        Otherwise returns ``image_results`` for full backward compatibility.
        """
        num_classes = len(image_results)
        # On the first weak pass ``cga_filter_mode`` has not yet been populated
        # by ``_build_cga``.  Read the explicit run mode so v2 can reserve its
        # per-image pending slot (including for an empty first image) before
        # any scorer construction occurs.
        mode0 = getattr(
            self,
            "cga_filter_mode",
            os.environ.get("CGA_FILTER_MODE", "legacy").strip().lower(),
        )
        if mode0 == "prototype_legacy_v2":
            out = self._refine_single_prototype_v2(image_results, img_meta)
            if collect_meta:
                return out, self._empty_cga_meta(num_classes)
            return out

        inputs = precomputed_inputs or self._flatten_cga_inputs(image_results)
        if inputs is None:
            if collect_meta:
                return image_results, self._empty_cga_meta(num_classes)
            return image_results
        boxes_list, scores_list, labels_list = inputs

        if getattr(self, "cga", None) is None:
            self.cga, self.exclude_ids = self._build_cga(num_classes)
        if (not hasattr(self, "_cga_diag_window")
                or len(self._cga_diag_window["pred_counts"]) != num_classes):
            self._cga_diag_window = _new_cga_diag_window(num_classes)

        mode0 = getattr(self, "cga_filter_mode", "legacy")
        if mode0 == "prototype_legacy":
            out = self._refine_single_prototype(image_results, img_meta)
            if collect_meta:
                return out, self._empty_cga_meta(num_classes)
            return out

        filename = img_meta["filename"]
        if precomputed_logits is None:
            logits, _ = self.cga(
                filename, boxes_list, scores_list, labels_list)
            logits = self._validate_cga_logits(
                logits, labels_list, num_classes)
        else:
            logits = self._validate_cga_logits(
                precomputed_logits, labels_list, num_classes)

        mode = getattr(self, "cga_filter_mode", "legacy")
        gate_prob_thr = getattr(self, "cga_gate_prob_thr", 0.5)
        drop_score = getattr(self, "cga_drop_score", 0.0)
        det_weight = getattr(self, "cga_blend_detector_weight", 0.7)
        veto_pred_thr = getattr(self, "cga_veto_pred_thr", 0.7)
        veto_label_thr = getattr(self, "cga_veto_label_thr", 0.1)
        protect_det_score = getattr(self, "cga_protect_det_score", 0.9)
        boost_det_thr = getattr(self, "cga_boost_det_thr", 0.75)
        boost_clip_thr = getattr(self, "cga_boost_clip_thr", 0.8)
        boost_strength = getattr(self, "cga_boost_strength", 0.7)
        adapt_w_min = getattr(self, "cga_adapt_w_min", 0.3)
        adapt_w_max = getattr(self, "cga_adapt_w_max", 0.95)
        veto_pred_hi = getattr(self, "cga_veto_pred_hi", 0.90)
        veto_label_lo = getattr(self, "cga_veto_label_lo", 0.05)
        veto_margin = getattr(self, "cga_veto_margin", 0.60)
        veto_entropy = getattr(self, "cga_veto_entropy", 0.35)
        veto_penalty = getattr(self, "cga_veto_penalty", 0.0)
        veto_skip_context = getattr(self, "cga_veto_skip_context", False)
        disagree_delta = getattr(self, "cga_disagree_delta", 0.1)
        disagree_score_thr = getattr(self, "cga_disagree_score_thr", 0.95)
        sem_low_thr = getattr(self, "cga_sem_low_thr", 0.90)
        sem_high_thr = getattr(self, "cga_sem_high_thr", 0.95)
        sem_lambda = getattr(self, "cga_sem_lambda", 0.50)
        sem_gated_lambda = getattr(self, "cga_sem_gated_lambda", 0.80)
        sem_view_ratios = getattr(self, "cga_sem_view_ratios", (0.0, 0.25, 0.5))
        rel_legacy_tau = getattr(self, "cga_rel_legacy_tau", 0.10)
        if mode in ("blend", "rescore"):
            mode = "legacy"
        valid_modes = {
            "legacy",
            "shuffled_legacy",
            "fixed_disagreement_penalty",
            "disagreement_threshold",
            "disagree_gate",
            "gate",
            "agree_gate",
            "strict_gate",
            "prob_gate",
            "label_prob_gate",
            "multiply",
            "prob_multiply",
            "veto_soft",
            "consensus_boost",
            "adaptive_blend",
            "evidence_veto",
            "semantic_reweight",
            "reliability_gated",
            "reliability_gated_mv",
            "reliability_gated_legacy",
            "prototype_legacy",
            "prototype_legacy_v2",
        }
        if mode not in valid_modes:
            _log_cga_info(f"[CGA][WARN] unsupported CGA_FILTER_MODE={mode}, fallback to legacy")
            mode = "legacy"

        stats = {
            "total": len(logits),
            "agree": 0,
            "dropped": 0,
            "blended": 0,
            "boosted": 0,
            "multiplied": 0,
            "penalized": 0,
            "threshold_dropped": 0,
            "shuffled": 0,
            "moved": 0,
            "unmoved": 0,
            "real_agree": 0,
            "operative_agree": 0,
            "reweighted": 0,
            "sem_weight_sum": 0.0,
            "prob_sum": 0.0,
            "label_probs": [],
            "pred_counts": np.zeros(num_classes, dtype=np.int64),
            "det_total": np.zeros(num_classes, dtype=np.int64),
            "det_agree": np.zeros(num_classes, dtype=np.int64),
            "det_drop": np.zeros(num_classes, dtype=np.int64),
        }

        # Per-box metadata, collected in the same flat order as boxes_list.
        # Reshaped back into per-class arrays at the end so it aligns with
        # image_results[class][box]. Only populated when collect_meta=True.
        num_boxes = len(labels_list)
        meta_semantic_weight = np.ones(num_boxes, dtype=np.float64)
        meta_agreement = np.zeros(num_boxes, dtype=bool)
        meta_det_score = np.asarray(scores_list, dtype=np.float64).copy()
        meta_det_label = np.asarray(labels_list, dtype=np.int64).copy()
        meta_clip_top1 = np.zeros(num_boxes, dtype=np.int64)
        meta_label_prob = np.zeros(num_boxes, dtype=np.float64)
        meta_top1_prob = np.zeros(num_boxes, dtype=np.float64)
        meta_margin = np.zeros(num_boxes, dtype=np.float64)
        meta_entropy = np.zeros(num_boxes, dtype=np.float64)
        meta_reliability = np.zeros(num_boxes, dtype=np.float64)

        # Multi-view reliability (reliability_gated_mv only): score every box at
        # several crop scales, then reliability = cross-scale top-1 agreement *
        # mean top-1 prob * (1 - mean entropy). Computed once per image here.
        mv_top1 = None
        mv_reliability = None
        if mode == "reliability_gated_mv":
            view_logits = self.cga.forward_views(
                filename, boxes_list, scores_list, labels_list,
                sem_view_ratios)  # (V, N, C)
            view_top1 = np.argmax(view_logits, axis=2)          # (V, N)
            view_top1_prob = np.max(view_logits, axis=2)        # (V, N)
            all_agree = np.all(view_top1 == view_top1[0:1, :], axis=0)  # (N,)
            mean_top1_prob = view_top1_prob.mean(axis=0)        # (N,)
            mean_ent = np.array(
                [np.mean([_prob_entropy_normalized(view_logits[v, n])
                          for v in range(view_logits.shape[0])])
                 for n in range(num_boxes)],
                dtype=np.float64) if num_boxes > 0 else np.zeros(0)
            mv_top1 = view_top1[0]  # tight-crop (view 0) top-1 as the clip class
            mv_reliability = (all_agree.astype(np.float64)
                              * mean_top1_prob * (1.0 - mean_ent))

        real_predictions = np.asarray(
            np.argmax(logits, axis=1), dtype=np.int64)
        if mode == "shuffled_legacy":
            if operative_logits is None:
                identities = [
                    f"{filename}\0{index}" for index in range(len(labels_list))
                ]
                operative_logits, source_indices = (
                    stratified_shuffle_probability_vectors(
                        logits,
                        scores_list,
                        labels_list,
                        identities,
                        seed=getattr(self, "cga_shuffle_seed", 0),
                        exclude_ids=getattr(self, "exclude_ids", ()),
                    )
                )
                shuffle_moved = source_indices != np.arange(len(source_indices))
            else:
                operative_logits = self._validate_cga_logits(
                    operative_logits, labels_list, num_classes)
            if shuffle_moved is None:
                shuffle_moved = np.zeros(len(labels_list), dtype=bool)
            shuffle_moved = np.asarray(shuffle_moved, dtype=bool)
            if shuffle_moved.shape != (len(labels_list),):
                raise ValueError("shuffle moved mask must align with detections")
        else:
            operative_logits = logits
            shuffle_moved = np.zeros(len(labels_list), dtype=bool)

        operative_predictions = np.asarray(
            np.argmax(operative_logits, axis=1), dtype=np.int64)
        operative_label_probs = np.asarray(
            [
                float(prob[int(label)])
                for prob, label in zip(operative_logits, labels_list)
            ],
            dtype=np.float64,
        )

        for i in range(len(logits)):
            label = int(labels_list[i])
            real_pred = int(real_predictions[i])
            pred = int(operative_predictions[i])
            prob = operative_logits[i]
            label_prob = float(operative_label_probs[i])
            stats["prob_sum"] += label_prob
            stats["label_probs"].append(label_prob)
            stats["real_agree"] += int(label == real_pred)
            stats["operative_agree"] += int(label == pred)
            if mode == "shuffled_legacy":
                if bool(shuffle_moved[i]):
                    stats["moved"] += 1
                else:
                    stats["unmoved"] += 1
            if 0 <= pred < num_classes:
                stats["pred_counts"][pred] += 1
            if 0 <= label < num_classes:
                stats["det_total"][label] += 1
            if label == pred:
                stats["agree"] += 1
                if 0 <= label < num_classes:
                    stats["det_agree"][label] += 1

            if collect_meta:
                sorted_prob = np.sort(prob)[::-1]
                top1_prob = float(sorted_prob[0])
                margin = (
                    float(sorted_prob[0] - sorted_prob[1])
                    if len(sorted_prob) > 1 else float(sorted_prob[0])
                )
                # In multi-view mode the operative CLIP class is the cross-scale
                # (tight-crop) top-1, so agreement/clip_top1 reflect that.
                clip_pred = int(mv_top1[i]) if mv_top1 is not None else pred
                meta_agreement[i] = (label == clip_pred)
                meta_clip_top1[i] = clip_pred
                meta_label_prob[i] = label_prob
                meta_top1_prob[i] = top1_prob
                meta_margin[i] = margin
                meta_entropy[i] = _prob_entropy_normalized(prob)

            if label in getattr(self, "exclude_ids", ()):
                continue

            dropped = False
            if mode == "legacy":
                if label != pred:
                    scores_list[i] = scores_list[i] * det_weight + label_prob * (1.0 - det_weight)
                    stats["blended"] += 1
            elif mode == "reliability_gated_legacy":
                # Reliability-Gated Legacy: run the FULL legacy score-blend, but
                # only when SARCLIP's opposition is reliable AND the detector is
                # not already highly confident (protects the [0.95,1.0) region
                # the box audit shows legacy over-prunes). Non-firing boxes are
                # left exactly as detector output (== no-CGA). Score-only change;
                # no ROI head / loss / RPN changes.
                det_score = float(scores_list[i])
                if label != pred and det_score < sem_high_thr:
                    p_top1 = float(prob[pred])
                    p_det_label = label_prob
                    entropy = _prob_entropy_normalized(prob)
                    opposition = (p_top1 - p_det_label) / 0.50
                    if opposition < 0.0:
                        opposition = 0.0
                    elif opposition > 1.0:
                        opposition = 1.0
                    reliability = opposition * (1.0 - entropy)
                    meta_reliability[i] = reliability
                    if reliability >= rel_legacy_tau:
                        scores_list[i] = (
                            det_score * det_weight
                            + p_det_label * (1.0 - det_weight)
                        )
                        stats["blended"] += 1
            elif mode == "shuffled_legacy":
                if label != pred:
                    scores_list[i] = (
                        scores_list[i] * det_weight
                        + label_prob * (1.0 - det_weight)
                    )
                    stats["blended"] += 1
                    stats["shuffled"] += 1
            elif mode == "fixed_disagreement_penalty":
                if label != pred:
                    scores_list[i] = max(
                        0.0, float(scores_list[i]) - disagree_delta
                    )
                    stats["penalized"] += 1
            elif mode == "disagreement_threshold":
                if label != pred and float(scores_list[i]) < disagree_score_thr:
                    scores_list[i] = drop_score
                    stats["dropped"] += 1
                    stats["threshold_dropped"] += 1
                    dropped = True
            elif mode == "adaptive_blend":
                # Like legacy, but the detector-trust weight rises with the
                # detector score: high-confidence disagreements (real targets
                # buried in chaff) are protected, while marginal ones (FP-prone,
                # sitting near score_thr) get penalized harder by SARCLIP.
                if label != pred:
                    det_score = float(scores_list[i])
                    w_eff = adapt_w_min + (adapt_w_max - adapt_w_min) * det_score
                    if w_eff < 0.0:
                        w_eff = 0.0
                    elif w_eff > 1.0:
                        w_eff = 1.0
                    scores_list[i] = det_score * w_eff + label_prob * (1.0 - w_eff)
                    stats["blended"] += 1
            elif mode == "evidence_veto":
                # Only act when SARCLIP gives RELIABLE, low-uncertainty opposing
                # evidence. Default action on disagreement = DO NOTHING. Fire a
                # multiplicative penalty (scale-free vs detector score) only when
                # ALL reliability conditions hold. See RSAR_CONFUSION_GROUPS.
                if label != pred:
                    pred_prob = float(prob[pred])
                    sorted_p = np.sort(prob)[::-1]
                    margin = float(sorted_p[0] - sorted_p[1]) if len(sorted_p) > 1 else float(sorted_p[0])
                    ent = _prob_entropy_normalized(prob)
                    same_group = (pred in self._veto_group_of
                                  and label in self._veto_group_of
                                  and self._veto_group_of[pred] == self._veto_group_of[label])
                    skip_ctx = veto_skip_context and (
                        label in self._veto_context_ids or pred in self._veto_context_ids)
                    reliable = (
                        pred_prob >= veto_pred_hi
                        and label_prob <= veto_label_lo
                        and margin >= veto_margin
                        and ent <= veto_entropy
                        and not same_group
                        and not skip_ctx
                    )
                    if reliable:
                        scores_list[i] = float(scores_list[i]) * veto_penalty
                        if veto_penalty <= 1e-6:
                            stats["dropped"] += 1
                            dropped = True
                        else:
                            stats["blended"] += 1
            elif mode == "semantic_reweight":
                # SARCLIP never modifies the detector score (admission is
                # decided purely by raw s_det >= score_thr downstream). It only
                # produces a semantic reliability weight for the ROI positive
                # classification loss. Agreement => 1.0. Disagreement => ramp:
                #   g = clip((high - s_det) / (high - low), 0, 1)
                #   weight = 1 - lambda * g
                # so score 0.90 disagreement -> 1-lambda, score >=0.95 -> 1.0.
                det_score = float(scores_list[i])
                if label == pred:
                    sem_weight = 1.0
                else:
                    g = (sem_high_thr - det_score) / (sem_high_thr - sem_low_thr)
                    if g < 0.0:
                        g = 0.0
                    elif g > 1.0:
                        g = 1.0
                    sem_weight = 1.0 - sem_lambda * g
                    stats["reweighted"] += 1
                meta_semantic_weight[i] = sem_weight
                stats["sem_weight_sum"] += sem_weight
            elif mode == "reliability_gated":
                # Reliability-Gated SRW (single view). Like semantic_reweight but
                # the downweight is scaled by how RELIABLE SARCLIP's opposition
                # is, so a bigger lambda only bites on stable, confident, low-
                # entropy disagreements. Scores never modified; boxes never
                # dropped.  r = p_top1 * margin * (1 - H_norm).
                det_score = float(scores_list[i])
                if label == pred:
                    sem_weight = 1.0
                else:
                    sorted_p = np.sort(prob)[::-1]
                    p_top1 = float(sorted_p[0])
                    margin = (float(sorted_p[0] - sorted_p[1])
                              if len(sorted_p) > 1 else float(sorted_p[0]))
                    ent = _prob_entropy_normalized(prob)
                    r = p_top1 * margin * (1.0 - ent)
                    if r < 0.0:
                        r = 0.0
                    elif r > 1.0:
                        r = 1.0
                    g = (sem_high_thr - det_score) / (sem_high_thr - sem_low_thr)
                    if g < 0.0:
                        g = 0.0
                    elif g > 1.0:
                        g = 1.0
                    sem_weight = 1.0 - sem_gated_lambda * r * g
                    if sem_weight < 0.0:
                        sem_weight = 0.0
                    stats["reweighted"] += 1
                    meta_reliability[i] = r
                meta_semantic_weight[i] = sem_weight
                stats["sem_weight_sum"] += sem_weight
            elif mode == "reliability_gated_mv":
                # Reliability-Gated SRW (multi-view). Reliability comes from
                # cross-scale agreement of SARCLIP's top-1 over the crop views:
                #   r = 1[all views top1 agree] * mean(p_top1) * (1 - mean H).
                # A single unstable AABB crop can no longer trigger a strong
                # downweight; only multi-scale-consistent opposition does.
                det_score = float(scores_list[i])
                mv_pred = int(mv_top1[i]) if mv_top1 is not None else pred
                if label == mv_pred:
                    sem_weight = 1.0
                else:
                    r = float(mv_reliability[i]) if mv_reliability is not None else 0.0
                    if r < 0.0:
                        r = 0.0
                    elif r > 1.0:
                        r = 1.0
                    g = (sem_high_thr - det_score) / (sem_high_thr - sem_low_thr)
                    if g < 0.0:
                        g = 0.0
                    elif g > 1.0:
                        g = 1.0
                    sem_weight = 1.0 - sem_gated_lambda * r * g
                    if sem_weight < 0.0:
                        sem_weight = 0.0
                    stats["reweighted"] += 1
                    meta_reliability[i] = r
                meta_semantic_weight[i] = sem_weight
                stats["sem_weight_sum"] += sem_weight
            elif mode in ("multiply", "prob_multiply"):
                scores_list[i] = scores_list[i] * label_prob
                stats["multiplied"] += 1
            elif mode in ("disagree_gate", "gate"):
                if label != pred and label_prob < gate_prob_thr:
                    scores_list[i] = drop_score
                    stats["dropped"] += 1
                    dropped = True
                elif label != pred:
                    scores_list[i] = scores_list[i] * det_weight + label_prob * (1.0 - det_weight)
                    stats["blended"] += 1
            elif mode in ("agree_gate", "strict_gate"):
                if label != pred or label_prob < gate_prob_thr:
                    scores_list[i] = drop_score
                    stats["dropped"] += 1
                    dropped = True
            elif mode in ("prob_gate", "label_prob_gate"):
                if label_prob < gate_prob_thr:
                    scores_list[i] = drop_score
                    stats["dropped"] += 1
                    dropped = True
            elif mode == "veto_soft":
                det_score = float(scores_list[i])
                pred_prob = float(prob[pred])
                if det_score >= protect_det_score:
                    # detector very confident (likely a real target, possibly
                    # buried in chaff): never drop, only soft-downweight.
                    if label != pred:
                        scores_list[i] = det_score * (det_weight + (1.0 - det_weight) * label_prob)
                        stats["blended"] += 1
                elif (label != pred and pred_prob >= veto_pred_thr
                        and label_prob < veto_label_thr):
                    # SARCLIP confidently says this is a different class: veto.
                    scores_list[i] = drop_score
                    stats["dropped"] += 1
                    dropped = True
                elif label != pred:
                    # uncertain disagreement: soft-downweight, keep for student.
                    scores_list[i] = det_score * (det_weight + (1.0 - det_weight) * label_prob)
                    stats["blended"] += 1
            elif mode == "consensus_boost":
                det_score = float(scores_list[i])
                pred_prob = float(prob[pred])
                if (label == pred and det_score >= boost_det_thr
                        and label_prob >= boost_clip_thr):
                    scores_list[i] = det_score + (
                        boost_strength * (1.0 - det_score) * label_prob
                    )
                    stats["boosted"] += 1
                elif (label != pred and det_score < protect_det_score
                      and pred_prob >= veto_pred_thr
                      and label_prob <= veto_label_thr + 1e-6):
                    scores_list[i] = det_score * det_weight
                    stats["blended"] += 1
            if dropped and 0 <= label < num_classes:
                stats["det_drop"][label] += 1

        self._cga_filter_calls = getattr(self, "_cga_filter_calls", 0) + 1
        diag_window = getattr(self, "_cga_diag_window", None)
        if diag_window is None or len(diag_window["pred_counts"]) != num_classes:
            diag_window = _new_cga_diag_window(num_classes)
        diag_window["calls"] += 1
        diag_window["total"] += stats["total"]
        diag_window["agree"] += stats["agree"]
        diag_window["dropped"] += stats["dropped"]
        diag_window["blended"] += stats["blended"]
        diag_window["boosted"] += stats["boosted"]
        diag_window["multiplied"] += stats["multiplied"]
        diag_window["penalized"] += stats["penalized"]
        diag_window["threshold_dropped"] += stats["threshold_dropped"]
        diag_window["shuffled"] += stats["shuffled"]
        diag_window["moved"] += stats["moved"]
        diag_window["unmoved"] += stats["unmoved"]
        diag_window["real_agree"] += stats["real_agree"]
        diag_window["operative_agree"] += stats["operative_agree"]
        diag_window["reweighted"] += stats["reweighted"]
        diag_window["sem_weight_sum"] += stats["sem_weight_sum"]
        diag_window["label_probs"].extend(stats["label_probs"])
        diag_window["pred_counts"] += stats["pred_counts"]
        diag_window["det_total"] += stats["det_total"]
        diag_window["det_agree"] += stats["det_agree"]
        diag_window["det_drop"] += stats["det_drop"]
        self._cga_diag_window = diag_window

        log_every = getattr(self, "cga_filter_log_every", 500)
        should_log = (
            force_log
            or self._cga_filter_calls == 1
            or (log_every > 0 and self._cga_filter_calls % log_every == 0)
        )
        if stats["total"] > 0 and not defer_log and should_log:
            mean_prob = stats["prob_sum"] / stats["total"]
            _log_cga_info(
                "[CGA] filter "
                f"mode={mode}, calls={self._cga_filter_calls}, total={stats['total']}, "
                f"agree={stats['agree']}, dropped={stats['dropped']}, "
                f"blended={stats['blended']}, boosted={stats['boosted']}, "
                f"multiplied={stats['multiplied']}, penalized={stats['penalized']}, "
                f"threshold_dropped={stats['threshold_dropped']}, "
                f"shuffled={stats['shuffled']}, "
                f"moved={stats['moved']}, "
                f"unmoved={stats['unmoved']}, "
                f"real_agree={stats['real_agree']}, "
                f"operative_agree={stats['operative_agree']}, "
                f"reweighted={stats['reweighted']}, "
                f"mean_sem_weight={(stats['sem_weight_sum'] / stats['total']):.4f}, "
                f"mean_label_prob={mean_prob:.4f}"
            )
            class_names = getattr(
                self.cga, "class_names", self._get_cga_class_names(num_classes)
            )
            diag_mean = (
                sum(diag_window["label_probs"]) / len(diag_window["label_probs"])
                if diag_window["label_probs"] else 0.0
            )
            _log_cga_info(
                "[CGA] diag_window "
                f"calls={diag_window['calls']}, total={diag_window['total']}, "
                f"agree={diag_window['agree']}, dropped={diag_window['dropped']}, "
                f"blended={diag_window['blended']}, boosted={diag_window['boosted']}, "
                f"multiplied={diag_window['multiplied']}, "
                f"penalized={diag_window['penalized']}, "
                f"threshold_dropped={diag_window['threshold_dropped']}, "
                f"shuffled={diag_window['shuffled']}, "
                f"moved={diag_window['moved']}, "
                f"unmoved={diag_window['unmoved']}, "
                f"real_agree={diag_window['real_agree']}, "
                f"operative_agree={diag_window['operative_agree']}, "
                f"mean_label_prob={diag_mean:.4f}, "
                f"label_prob_pct={_format_label_prob_percentiles(diag_window['label_probs'])}, "
                f"argmax={_format_class_counts(diag_window['pred_counts'], class_names)}, "
                f"detector={_format_detector_diag(diag_window['det_total'], diag_window['det_agree'], diag_window['det_drop'], class_names)}"
            )
            self._cga_diag_window = _new_cga_diag_window(num_classes)

        meta = None
        if collect_meta:
            meta = {
                "semantic_weight": [],
                "agreement": [],
                "det_score": [],
                "det_label": [],
                "clip_top1": [],
                "label_prob": [],
                "top1_prob": [],
                "margin": [],
                "entropy": [],
                "reliability": [],
            }

        j = 0
        for i in range(num_classes):
            num_dets = len(image_results[i])
            if num_dets == 0:
                if collect_meta:
                    meta["semantic_weight"].append(
                        np.ones(0, dtype=np.float64))
                    meta["agreement"].append(np.zeros(0, dtype=bool))
                    meta["det_score"].append(np.zeros(0, dtype=np.float64))
                    meta["det_label"].append(np.zeros(0, dtype=np.int64))
                    meta["clip_top1"].append(np.zeros(0, dtype=np.int64))
                    meta["label_prob"].append(np.zeros(0, dtype=np.float64))
                    meta["top1_prob"].append(np.zeros(0, dtype=np.float64))
                    meta["margin"].append(np.zeros(0, dtype=np.float64))
                    meta["entropy"].append(np.zeros(0, dtype=np.float64))
                    meta["reliability"].append(np.zeros(0, dtype=np.float64))
                continue
            start = j
            stop = j + num_dets
            for k in range(num_dets):
                image_results[i][k, -1] = scores_list[j]
                j += 1
            if collect_meta:
                meta["semantic_weight"].append(
                    meta_semantic_weight[start:stop].copy())
                meta["agreement"].append(meta_agreement[start:stop].copy())
                meta["det_score"].append(meta_det_score[start:stop].copy())
                meta["det_label"].append(meta_det_label[start:stop].copy())
                meta["clip_top1"].append(meta_clip_top1[start:stop].copy())
                meta["label_prob"].append(meta_label_prob[start:stop].copy())
                meta["top1_prob"].append(meta_top1_prob[start:stop].copy())
                meta["margin"].append(meta_margin[start:stop].copy())
                meta["entropy"].append(meta_entropy[start:stop].copy())
                meta["reliability"].append(meta_reliability[start:stop].copy())

        if collect_meta:
            return image_results, meta
        return image_results
