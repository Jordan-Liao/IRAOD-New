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


def _new_cga_diag_window(num_classes):
    return {
        "calls": 0,
        "total": 0,
        "agree": 0,
        "dropped": 0,
        "blended": 0,
        "multiplied": 0,
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

    def _crop_patches(self, img_path, boxes, scores, labels):
        image_mode = "L" if self.force_grayscale else "RGB"
        image = Image.open(img_path).convert(image_mode)

        image_list = []
        ori_image_list = []
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            x1, y1, x2, y2 = box
            h, w = y2 - y1, x2 - x1
            x1 = max(0, x1 - w * self.expand_ratio)
            y1 = max(0, y1 - h * self.expand_ratio)
            x2 = x2 + w * self.expand_ratio
            y2 = y2 + h * self.expand_ratio

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
        self.cga_filter_log_every = _env_int("CGA_FILTER_LOG_EVERY", 500)
        lora_path = os.environ.get("SARCLIP_LORA", "").strip()
        _log_cga_info(
            "[CGA] init "
            f"scorer={scorer or '<unset>'}, "
            f"backend={backend}, "
            f"lora={os.path.expanduser(lora_path) if lora_path else '<unset>'}, "
            f"filter_mode={self.cga_filter_mode}, "
            f"gate_prob_thr={self.cga_gate_prob_thr}, "
            f"drop_score={self.cga_drop_score}"
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

    def refine_test(self, results, img_metas):
        if getattr(self, "cga", None) is None:
            self.cga, self.exclude_ids = self._build_cga(len(results[0]))
        if not hasattr(self, "_cga_diag_window"):
            self._cga_diag_window = _new_cga_diag_window(len(results[0]))

        boxes_list, scores_list, labels_list = [], [], []
        for cls_id, result in enumerate(results[0]):
            if len(result) == 0:
                continue
            result_xyxy = obb2xyxy(result)
            boxes_list.append(result_xyxy[:, :4])
            scores_list.append(result[:, -1])
            labels_list.append([cls_id] * len(result))

        if len(boxes_list) == 0:
            return results

        boxes_list = np.concatenate(boxes_list, axis=0)
        scores_list = np.concatenate(scores_list, axis=0)
        labels_list = np.concatenate(labels_list, axis=0)

        logits, _ = self.cga(img_metas[0]["filename"], boxes_list, scores_list, labels_list)

        mode = getattr(self, "cga_filter_mode", "legacy")
        gate_prob_thr = getattr(self, "cga_gate_prob_thr", 0.5)
        drop_score = getattr(self, "cga_drop_score", 0.0)
        det_weight = getattr(self, "cga_blend_detector_weight", 0.7)
        if mode in ("blend", "rescore"):
            mode = "legacy"
        valid_modes = {
            "legacy",
            "disagree_gate",
            "gate",
            "agree_gate",
            "strict_gate",
            "prob_gate",
            "label_prob_gate",
            "multiply",
            "prob_multiply",
        }
        if mode not in valid_modes:
            _log_cga_info(f"[CGA][WARN] unsupported CGA_FILTER_MODE={mode}, fallback to legacy")
            mode = "legacy"

        stats = {
            "total": len(logits),
            "agree": 0,
            "dropped": 0,
            "blended": 0,
            "multiplied": 0,
            "prob_sum": 0.0,
            "label_probs": [],
            "pred_counts": np.zeros(len(results[0]), dtype=np.int64),
            "det_total": np.zeros(len(results[0]), dtype=np.int64),
            "det_agree": np.zeros(len(results[0]), dtype=np.int64),
            "det_drop": np.zeros(len(results[0]), dtype=np.int64),
        }
        for i, prob in enumerate(logits):
            label = int(labels_list[i])
            pred = int(np.argmax(prob))
            label_prob = float(prob[label])
            stats["prob_sum"] += label_prob
            stats["label_probs"].append(label_prob)
            if 0 <= pred < len(results[0]):
                stats["pred_counts"][pred] += 1
            if 0 <= label < len(results[0]):
                stats["det_total"][label] += 1
            if label == pred:
                stats["agree"] += 1
                if 0 <= label < len(results[0]):
                    stats["det_agree"][label] += 1
            if label in self.exclude_ids:
                continue

            dropped = False
            if mode == "legacy":
                if label != pred:
                    scores_list[i] = scores_list[i] * det_weight + label_prob * (1.0 - det_weight)
                    stats["blended"] += 1
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
            if dropped and 0 <= label < len(results[0]):
                stats["det_drop"][label] += 1

        self._cga_filter_calls = getattr(self, "_cga_filter_calls", 0) + 1
        diag_window = getattr(self, "_cga_diag_window", None)
        if diag_window is None or len(diag_window["pred_counts"]) != len(results[0]):
            diag_window = _new_cga_diag_window(len(results[0]))
        diag_window["calls"] += 1
        diag_window["total"] += stats["total"]
        diag_window["agree"] += stats["agree"]
        diag_window["dropped"] += stats["dropped"]
        diag_window["blended"] += stats["blended"]
        diag_window["multiplied"] += stats["multiplied"]
        diag_window["label_probs"].extend(stats["label_probs"])
        diag_window["pred_counts"] += stats["pred_counts"]
        diag_window["det_total"] += stats["det_total"]
        diag_window["det_agree"] += stats["det_agree"]
        diag_window["det_drop"] += stats["det_drop"]
        self._cga_diag_window = diag_window

        log_every = getattr(self, "cga_filter_log_every", 500)
        if stats["total"] > 0 and (self._cga_filter_calls == 1 or (log_every > 0 and self._cga_filter_calls % log_every == 0)):
            mean_prob = stats["prob_sum"] / stats["total"]
            _log_cga_info(
                "[CGA] filter "
                f"mode={mode}, calls={self._cga_filter_calls}, total={stats['total']}, "
                f"agree={stats['agree']}, dropped={stats['dropped']}, "
                f"blended={stats['blended']}, multiplied={stats['multiplied']}, "
                f"mean_label_prob={mean_prob:.4f}"
            )
            class_names = getattr(self.cga, "class_names", self._get_cga_class_names(len(results[0])))
            diag_mean = (
                sum(diag_window["label_probs"]) / len(diag_window["label_probs"])
                if diag_window["label_probs"] else 0.0
            )
            _log_cga_info(
                "[CGA] diag_window "
                f"calls={diag_window['calls']}, total={diag_window['total']}, "
                f"agree={diag_window['agree']}, dropped={diag_window['dropped']}, "
                f"blended={diag_window['blended']}, multiplied={diag_window['multiplied']}, "
                f"mean_label_prob={diag_mean:.4f}, "
                f"label_prob_pct={_format_label_prob_percentiles(diag_window['label_probs'])}, "
                f"argmax={_format_class_counts(diag_window['pred_counts'], class_names)}, "
                f"detector={_format_detector_diag(diag_window['det_total'], diag_window['det_agree'], diag_window['det_drop'], class_names)}"
            )
            self._cga_diag_window = _new_cga_diag_window(len(results[0]))

        j = 0
        for i in range(len(results[0])):
            num_dets = len(results[0][i])
            if num_dets == 0:
                continue
            for k in range(num_dets):
                results[0][i][k, -1] = scores_list[j]
                j += 1

        return results
