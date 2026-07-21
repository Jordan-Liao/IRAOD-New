"""Target-Adaptive Prototype-Guided CGA (prototype_legacy mode).

Self-contained, testable building blocks for building online per-class VISUAL
prototypes in the frozen SARCLIP feature space from high-reliability target-domain
teacher proposals, and fusing them with the fixed TEXT prototypes to re-score
pseudo-labels.  Nothing here touches SARCLIP weights, gradients, GT, or any other
CGA mode.

Design invariants (see experiment spec):
  * SARCLIP fully frozen; embeddings are detached numpy.
  * Visual prototypes are runtime state only; reset per run; never use GT.
  * Prototype label == teacher detector label (never SARCLIP top-1 / fused).
  * Candidates gated by raw teacher score >= 0.97, weak/strong label agreement,
    and rotated IoU(weak, strong) >= 0.70.
  * A proposal is scored with the prototype snapshot taken BEFORE this
    iteration's update (no self-inclusion).
  * When no class prototype is active, the fused logits equal the text-only
    logits exactly (visual contributes nothing) -> degrades to plain legacy.
"""
import numpy as np

EPS = 1e-6


def rotated_align_crop(image_np, cx, cy, w, h, angle_rad, context_ratio=0.15):
    """Rotate the OBB to axis-aligned and crop a (w,h)*(1+context) patch.

    * Rotation is about the OBB centre using the project ``le90`` angle.
    * Border is replicated (never black-filled / masked).
    * Out-of-image regions are clipped to valid bounds.
    * Degenerate boxes (w<=0, h<=0) or empty crops raise ValueError so strict
      mode can terminate the run.

    Returns an HxWx3 (or HxW) uint8 numpy patch.
    """
    import cv2

    if image_np is None or image_np.size == 0:
        raise ValueError("rotated_align_crop: empty source image")
    if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(w)
            and np.isfinite(h) and np.isfinite(angle_rad)):
        raise ValueError("rotated_align_crop: non-finite OBB parameter")
    if w <= 0.0 or h <= 0.0:
        raise ValueError(f"rotated_align_crop: degenerate box w={w} h={h}")

    H, W = image_np.shape[:2]
    cw = float(w) * (1.0 + float(context_ratio))
    ch = float(h) * (1.0 + float(context_ratio))

    # le90: box major axis is rotated by angle_rad (CCW). Rotate the image by
    # +angle_deg about the centre so the box becomes axis-aligned, then crop.
    angle_deg = float(np.degrees(angle_rad))
    M = cv2.getRotationMatrix2D((float(cx), float(cy)), angle_deg, 1.0)
    rotated = cv2.warpAffine(
        image_np, M, (W, H), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE)

    x1 = cx - cw / 2.0
    y1 = cy - ch / 2.0
    x2 = cx + cw / 2.0
    y2 = cy + ch / 2.0
    xi1 = int(round(max(0.0, x1)))
    yi1 = int(round(max(0.0, y1)))
    xi2 = int(round(min(float(W), x2)))
    yi2 = int(round(min(float(H), y2)))
    if xi2 <= xi1 or yi2 <= yi1:
        raise ValueError(
            f"rotated_align_crop: empty patch after clip "
            f"(x1={xi1},y1={yi1},x2={xi2},y2={yi2}, W={W},H={H})")
    patch = rotated[yi1:yi2, xi1:xi2]
    if patch.size == 0 or patch.shape[0] < 1 or patch.shape[1] < 1:
        raise ValueError("rotated_align_crop: empty patch")
    return np.ascontiguousarray(patch)


def _zscore(sim):
    """Per-row (class-dim) z-score with unbiased=False and eps floor."""
    mean = sim.mean(axis=-1, keepdims=True)
    std = sim.std(axis=-1, keepdims=True)  # population std (ddof=0)
    return (sim - mean) / (std + EPS)


def fuse_logits(sim_text, sim_visual, active_mask, beta):
    """Fuse z-scored text/visual cosine sims per active class.

    sim_text, sim_visual : (N, C) cosine similarities.
    active_mask          : (C,) bool, True where visual prototype is usable.
    Inactive classes keep the text-only z-scored logit exactly.
    Returns fused_logits (N, C).
    """
    sim_text = np.asarray(sim_text, dtype=np.float64)
    sim_visual = np.asarray(sim_visual, dtype=np.float64)
    if sim_text.shape != sim_visual.shape:
        raise ValueError("fuse_logits: text/visual sim shape mismatch")
    z_text = _zscore(sim_text)
    fused = z_text.copy()
    active_mask = np.asarray(active_mask, dtype=bool)
    if active_mask.any():
        z_visual = _zscore(sim_visual)
        cols = np.where(active_mask)[0]
        fused[:, cols] = (1.0 - beta) * z_text[:, cols] + beta * z_visual[:, cols]
    return fused


def softmax(logits, axis=-1):
    logits = np.asarray(logits, dtype=np.float64)
    m = logits.max(axis=axis, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=axis, keepdims=True)


def _unit_rows(matrix, name):
    """Validate an already L2-normalized matrix without changing its values."""
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError(f"{name}: expected a 2-D matrix, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name}: NaN/Inf")
    norms = np.linalg.norm(matrix.astype(np.float64, copy=False), axis=1)
    if np.any(norms <= 0.0):
        raise ValueError(f"{name}: zero-norm row")
    if not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError(f"{name}: rows must be L2-normalized")
    return matrix


def mixed_prototype_matrix(text_prototypes, visual_prototypes, active_mask,
                           beta=0.50):
    """Build Prototype-CGA v2 class prototypes in embedding space.

    Inactive rows are copied directly from the normalized text classifier.
    Active rows are ``normalize((1-beta)*text + beta*visual)``.  The helper is
    deliberately separate from v1 ``fuse_logits``: no class-dimension z-score
    or probability-space interpolation is permitted in v2.
    """
    text = _unit_rows(text_prototypes, "text_prototypes")
    visual = np.asarray(visual_prototypes)
    if visual.shape != text.shape:
        raise ValueError(
            "visual_prototypes: shape mismatch "
            f"{visual.shape} != {text.shape}")
    active = np.asarray(active_mask, dtype=bool)
    if active.shape != (text.shape[0],):
        raise ValueError(
            f"active_mask: shape {active.shape} != {(text.shape[0],)}")
    beta = float(beta)
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")

    # Preserve inactive text rows byte-for-byte.  Besides satisfying the
    # experiment invariant, this avoids a second normalization changing the
    # final bits of the plain-legacy classifier.
    mixed = text.copy()
    for c in np.where(active)[0]:
        visual_c = np.asarray(visual[c])
        if not np.all(np.isfinite(visual_c)):
            raise ValueError(f"visual_prototypes[{c}]: NaN/Inf")
        visual_norm = float(np.linalg.norm(
            visual_c.astype(np.float64, copy=False)))
        if visual_norm <= 0.0:
            raise ValueError(f"visual_prototypes[{c}]: zero norm")
        visual_c = visual_c / visual_norm

        # These two calibrated controls must be numerically identical to the
        # original classifier, not merely mathematically equivalent.
        if beta == 0.0 or np.array_equal(visual[c], text[c]):
            mixed[c] = text[c]
            continue

        candidate = ((1.0 - beta) * text[c].astype(np.float64, copy=False)
                     + beta * visual_c.astype(np.float64, copy=False))
        candidate_norm = float(np.linalg.norm(candidate))
        if not np.isfinite(candidate_norm) or candidate_norm <= 0.0:
            raise ValueError(f"mixed_prototype[{c}]: zero norm")
        mixed[c] = candidate / candidate_norm

    _unit_rows(mixed, "mixed_prototypes")
    return mixed


def prototype_legacy_v2_probabilities(
        image_embeddings, text_prototypes, visual_prototypes, active_mask,
        tau, beta=0.50, text_sim=None, text_prob=None):
    """Return v2 mixed prototypes, similarities, scaled logits and probabilities.

    ``text_sim`` and ``text_prob`` should be the values returned by the same
    AABB SARCLIP forward.  Unchanged prototype rows reuse those values; when no
    row changes (all inactive, beta=0, or visual==text), the complete probability
    matrix is returned directly.  This is the numerical legacy-degradation
    guarantee and also avoids a duplicate image encode.
    """
    embedding = _unit_rows(image_embeddings, "image_embeddings")
    text = _unit_rows(text_prototypes, "text_prototypes")
    if embedding.shape[1] != text.shape[1]:
        raise ValueError(
            "embedding/prototype dimension mismatch: "
            f"{embedding.shape[1]} != {text.shape[1]}")
    active = np.asarray(active_mask, dtype=bool)
    mixed = mixed_prototype_matrix(
        text, visual_prototypes, active, beta=beta)

    expected_shape = (embedding.shape[0], text.shape[0])
    if text_sim is None:
        text_sim_arr = embedding @ text.T
    else:
        text_sim_arr = np.asarray(text_sim)
        if text_sim_arr.shape != expected_shape:
            raise ValueError(
                f"text_sim: shape {text_sim_arr.shape} != {expected_shape}")
        if not np.all(np.isfinite(text_sim_arr)):
            raise ValueError("text_sim: NaN/Inf")

    changed = active.copy()
    if float(beta) == 0.0:
        changed[:] = False
    else:
        for c in np.where(changed)[0]:
            if np.array_equal(mixed[c], text[c]):
                changed[c] = False

    fused_sim = text_sim_arr.copy()
    if changed.any():
        cols = np.where(changed)[0]
        fused_sim[:, cols] = embedding @ mixed[cols].T
    if not np.all(np.isfinite(fused_sim)):
        raise ValueError("fused_sim: NaN/Inf")

    tau = float(tau)
    if not np.isfinite(tau):
        raise ValueError("tau must be finite")
    fused_logits = tau * fused_sim

    if not changed.any() and text_prob is not None:
        fused_prob = np.asarray(text_prob).copy()
        if fused_prob.shape != expected_shape:
            raise ValueError(
                f"text_prob: shape {fused_prob.shape} != {expected_shape}")
    else:
        fused_prob = softmax(fused_logits, axis=-1)
    if not np.all(np.isfinite(fused_prob)):
        raise ValueError("fused_prob: NaN/Inf")
    return mixed, fused_sim, fused_logits, fused_prob


def legacy_score_blend(probabilities, detector_scores, detector_labels,
                       detector_weight=0.70, exclude_ids=()):
    """Apply the plain-legacy disagreement-only score blend."""
    probabilities = np.asarray(probabilities)
    scores = np.asarray(detector_scores, dtype=np.float64).copy()
    labels = np.asarray(detector_labels, dtype=np.int64)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a 2-D matrix")
    if probabilities.shape[0] != len(scores) or len(scores) != len(labels):
        raise ValueError("probability/score/label length mismatch")
    if not (np.all(np.isfinite(probabilities))
            and np.all(np.isfinite(scores))):
        raise ValueError("legacy_score_blend: NaN/Inf")
    detector_weight = float(detector_weight)
    if not 0.0 <= detector_weight <= 1.0:
        raise ValueError("detector_weight must be in [0, 1]")

    predictions = probabilities.argmax(axis=1)
    blended = np.zeros(len(scores), dtype=bool)
    excluded = set(int(c) for c in exclude_ids)
    for i, (label, prediction) in enumerate(zip(labels, predictions)):
        label = int(label)
        if label in excluded:
            continue
        if int(prediction) != label:
            scores[i] = (
                detector_weight * float(detector_scores[i])
                + (1.0 - detector_weight)
                * float(probabilities[i, label]))
            blended[i] = True
    return scores, predictions, blended


def greedy_match_weak_strong(weak_obb, weak_label, weak_score,
                             strong_obb, strong_label,
                             iou_fn, score_thr=0.97, iou_thr=0.70):
    """Return a boolean mask over weak proposals that qualify as prototype
    candidates: raw weak score >= score_thr AND greedily matched one-to-one to a
    strong proposal of the SAME label with rotated IoU >= iou_thr.

    weak_obb/strong_obb : (Nw,5)/(Ns,5) OBB (cx,cy,w,h,angle) in one frame.
    iou_fn(a, b) -> (len(a), len(b)) rotated IoU matrix.
    """
    nw = len(weak_obb)
    qualified = np.zeros(nw, dtype=bool)
    if nw == 0 or len(strong_obb) == 0:
        return qualified

    weak_label = np.asarray(weak_label)
    strong_label = np.asarray(strong_label)
    weak_score = np.asarray(weak_score, dtype=np.float64)

    strong_taken = np.zeros(len(strong_obb), dtype=bool)
    # Only consider weak proposals above the score gate; match per class.
    for c in np.unique(weak_label):
        w_idx = np.where((weak_label == c) & (weak_score >= score_thr))[0]
        s_idx = np.where(strong_label == c)[0]
        if len(w_idx) == 0 or len(s_idx) == 0:
            continue
        iou = np.asarray(iou_fn(weak_obb[w_idx], strong_obb[s_idx]), dtype=np.float64)
        # Greedy highest-IoU one-to-one matching within this class.
        pairs = [(iou[i, j], i, j)
                 for i in range(len(w_idx)) for j in range(len(s_idx))]
        pairs.sort(key=lambda t: t[0], reverse=True)
        w_used = np.zeros(len(w_idx), dtype=bool)
        s_used = np.zeros(len(s_idx), dtype=bool)
        for val, i, j in pairs:
            if val < iou_thr:
                break
            if w_used[i] or s_used[j] or strong_taken[s_idx[j]]:
                continue
            w_used[i] = True
            s_used[j] = True
            strong_taken[s_idx[j]] = True
            qualified[w_idx[i]] = True
    return qualified


class VisualPrototypeBank:
    """Per-class online visual prototype bank (EMA, frozen SARCLIP space)."""

    def __init__(self, num_classes, embed_dim=None, momentum=0.95,
                 min_count=20):
        self.num_classes = int(num_classes)
        self.embed_dim = embed_dim
        self.momentum = float(momentum)
        self.min_count = int(min_count)
        self.prototype = [None] * self.num_classes
        self.previous_prototype = [None] * self.num_classes
        self.prototype_count = np.zeros(self.num_classes, dtype=np.int64)
        self.prototype_update_count = np.zeros(self.num_classes, dtype=np.int64)
        self.prototype_initialized = np.zeros(self.num_classes, dtype=bool)
        self.first_active_iteration = -np.ones(self.num_classes, dtype=np.int64)

    def active_mask(self):
        return self.prototype_count >= self.min_count

    def is_active(self, c):
        return bool(self.prototype_count[c] >= self.min_count)

    def matrix(self):
        """(C, D) prototype matrix; inactive/uninit rows are zeros (never used
        because active_mask gates fusion)."""
        if self.embed_dim is None:
            raise ValueError("VisualPrototypeBank: embed_dim unknown")
        mat = np.zeros((self.num_classes, self.embed_dim), dtype=np.float64)
        for c in range(self.num_classes):
            if self.prototype[c] is not None:
                mat[c] = self.prototype[c]
        return mat

    def snapshot_previous(self):
        self.previous_prototype = [
            None if p is None else p.copy() for p in self.prototype]

    def update(self, class_to_embeddings, cur_iter):
        """EMA-update prototypes from this iteration's qualified embeddings.

        class_to_embeddings : dict[class_id] -> (k, D) normalized embeddings.
        Called at iteration END (after all scoring), so scoring used the old
        snapshot -> no self-inclusion.
        """
        for c, embs in class_to_embeddings.items():
            embs = np.asarray(embs, dtype=np.float64)
            if embs.ndim != 2 or embs.shape[0] == 0:
                continue
            if self.embed_dim is None:
                self.embed_dim = embs.shape[1]
            if embs.shape[1] != self.embed_dim:
                raise ValueError(
                    f"VisualPrototypeBank: embed dim mismatch "
                    f"{embs.shape[1]} != {self.embed_dim}")
            batch_proto = embs.mean(axis=0)
            n = np.linalg.norm(batch_proto)
            if not np.isfinite(n) or n <= 0.0:
                raise ValueError("VisualPrototypeBank: zero-norm batch proto")
            batch_proto = batch_proto / n

            if not self.prototype_initialized[c]:
                proto = batch_proto
                self.prototype_initialized[c] = True
            else:
                proto = (self.momentum * self.prototype[c]
                         + (1.0 - self.momentum) * batch_proto)
                pn = np.linalg.norm(proto)
                if not np.isfinite(pn) or pn <= 0.0:
                    raise ValueError("VisualPrototypeBank: zero-norm proto")
                proto = proto / pn
            self.prototype[c] = proto
            self.prototype_count[c] += embs.shape[0]
            self.prototype_update_count[c] += 1
            if (self.first_active_iteration[c] < 0
                    and self.prototype_count[c] >= self.min_count):
                self.first_active_iteration[c] = int(cur_iter)

    def drift(self, c):
        """1 - cos(current, previous); 0 if no previous."""
        if (self.prototype[c] is None
                or self.previous_prototype[c] is None):
            return 0.0
        a = self.prototype[c]
        b = self.previous_prototype[c]
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 0:
            return 0.0
        return float(1.0 - float(np.dot(a, b) / denom))

    def visual_text_cos(self, c, text_proto_c):
        if self.prototype[c] is None:
            return 0.0
        a = self.prototype[c]
        b = np.asarray(text_proto_c, dtype=np.float64)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 0:
            return 0.0
        return float(np.dot(a, b) / denom)
