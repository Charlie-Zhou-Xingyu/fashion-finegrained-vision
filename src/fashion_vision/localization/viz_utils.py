"""
Shared visualization utilities for 3.1.2 debug and calibration gallery scripts.

All drawing functions operate on BGR uint8 numpy arrays (OpenCV format).
Chinese text is rendered via PIL when available, with a unicode-escape fallback.
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def box_iou(a: list, b: list) -> float:
    """IoU of two xyxy bounding boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if not inter:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0

# ── Drawing constants ─────────────────────────────────────────────────────────
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.45
FONT_THICKNESS = 1
BOX_THICKNESS = 2
PANEL_PAD = 8
LABEL_H = 22


def draw_text(
    img: np.ndarray,
    text: str,
    xy: tuple[int, int],
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    """
    Draw text on img in-place.

    Uses cv2 for ASCII text; falls back to PIL for non-ASCII (Chinese etc.).
    If PIL is unavailable, non-ASCII characters are unicode-escaped before rendering.
    """
    if text.isascii():
        cv2.putText(img, text, xy, FONT, FONT_SCALE, (0, 0, 0), FONT_THICKNESS + 1, cv2.LINE_AA)
        cv2.putText(img, text, xy, FONT, FONT_SCALE, color, FONT_THICKNESS, cv2.LINE_AA)
        return

    try:
        from PIL import ImageFont, ImageDraw, Image as _PILImage  # type: ignore

        pil = _PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont
        for candidate in (
            "C:/Windows/Fonts/msyh.ttc",           # Windows Microsoft YaHei
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",  # Linux WenQuanYi
        ):
            try:
                font = ImageFont.truetype(candidate, 14)
                break
            except OSError:
                pass
        else:
            font = ImageFont.load_default()

        draw.text(xy, text, fill=(color[2], color[1], color[0]), font=font)
        img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except ImportError:
        safe = text.encode("unicode_escape").decode("ascii")[:60]
        cv2.putText(img, safe, xy, FONT, FONT_SCALE, (0, 0, 0), FONT_THICKNESS + 1, cv2.LINE_AA)
        cv2.putText(img, safe, xy, FONT, FONT_SCALE, color, FONT_THICKNESS, cv2.LINE_AA)


def draw_box(
    img: np.ndarray,
    bbox: list[float],
    color: tuple[int, int, int],
    label: str = "",
) -> None:
    """Draw a rectangle + optional label on img in-place."""
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, BOX_THICKNESS)
    if label:
        draw_text(img, label, (x1 + 2, max(y1 + 14, 14)), color)


def add_title_bar(panel: np.ndarray, title: str) -> np.ndarray:
    """Return a new image with a dark title bar prepended above panel."""
    bar = np.zeros((LABEL_H, panel.shape[1], 3), dtype=np.uint8)
    bar[:] = (40, 40, 40)
    draw_text(bar, title, (4, LABEL_H - 6))
    return np.vstack([bar, panel])


def resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return img
    scale = target_h / h
    return cv2.resize(img, (max(1, int(w * scale)), target_h), interpolation=cv2.INTER_LINEAR)


def pad_width(img: np.ndarray, target_w: int, fill: int = 50) -> np.ndarray:
    h, w = img.shape[:2]
    if w >= target_w:
        return img
    pad = np.full((h, target_w - w, 3), fill, dtype=np.uint8)
    return np.hstack([img, pad])


def panels_to_html(
    sections: list[tuple[str, list[tuple[str, np.ndarray]]]],
    out_path: Path,
) -> None:
    """
    Save a multi-section HTML file with embedded panel images.

    Args:
        sections: List of (section_title, [(panel_title, panel_img), ...]).
                  For a single-section layout pass a list with one entry.
        out_path: Path to write HTML file.
    """
    body = ""
    for section_title, panels in sections:
        if section_title:
            body += f'<h2 class="stitle">{section_title}</h2>\n<div class="grid">\n'
        else:
            body += '<div class="grid">\n'
        for title, img in panels:
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            b64 = base64.b64encode(buf).decode()
            body += (
                f'<div class="panel">'
                f'<div class="ptitle">{title}</div>'
                f'<img src="data:image/jpeg;base64,{b64}"/>'
                f'</div>\n'
            )
        body += "</div>\n"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  body{{background:#1a1a1a;margin:0;font-family:monospace;color:#ccc}}
  h2.stitle{{color:#adf;padding:8px 12px;margin:0;background:#2a2a3a;border-left:4px solid #55f}}
  .grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;padding:6px}}
  .panel{{background:#2a2a2a;border:1px solid #444}}
  .ptitle{{color:#ddd;font-size:12px;padding:4px 6px;background:#333;white-space:nowrap;overflow:hidden}}
  .panel img{{width:100%;display:block}}
</style></head>
<body>{body}</body></html>"""
    out_path.write_text(html, encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
# Inner-garment detection visualization (3.1.2)
# ═══════════════════════════════════════════════════════════════════════════════

# Colour constants
_PURPLE = (255, 0, 255)       # opening ROI
_MAGENTA = (255, 0, 128)     # torso ROI (BGR: 128, 0, 255)
_TEAL = (255, 255, 0)        # torso ROI overlay
_ORANGE = (64, 165, 255)     # before-refine bbox (dashed)
_BLUE = (255, 0, 0)          # after-refine bbox
_GREEN = (0, 255, 0)         # accepted
_RED = (0, 0, 255)           # rejected


def draw_dashed_box(
    img: np.ndarray,
    bbox: list[float],
    color: tuple[int, int, int],
    dash_len: int = 8,
    gap_len: int = 4,
    thickness: int = 2,
) -> None:
    """Draw a dashed rectangle on a BGR image in-place."""
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    for x in range(x1, x2, dash_len + gap_len):
        xe = min(x + dash_len, x2)
        cv2.line(img, (x, y1), (xe, y1), color, thickness)
        cv2.line(img, (x, y2), (xe, y2), color, thickness)
    for y in range(y1, y2, dash_len + gap_len):
        ye = min(y + dash_len, y2)
        cv2.line(img, (x1, y), (x1, ye), color, thickness)
        cv2.line(img, (x2, y), (x2, ye), color, thickness)


def draw_inner_garment_debug(
    img_bgr: np.ndarray,
    outer_bbox: list[int],
    opening_roi: list[int] | None = None,
    torso_mask: np.ndarray | None = None,
    torso_bbox: list[int] | None = None,
    before_bbox: list[int] | None = None,
    after_bbox: list[int] | None = None,
    inner_mask: np.ndarray | None = None,
    selected_scoring: dict | None = None,
    extension_debug: dict | None = None,
    refine_debug: dict | None = None,
    cleanup_debug: dict | None = None,
    cleanup_before_bbox: list[int] | None = None,
    alpha: float = 0.25,
) -> np.ndarray:
    """Draw inner-garment detection debug overlay.

    Draws in this order (most recent on top):
        1. Torso mask (magenta semi-transparent)
        2. Opening ROI (purple box, labelled)
        3. Before-refine bbox (orange dashed)
        4. Inner garment mask (teal semi-transparent)
        5. After-refine bbox (blue solid)
        6. Candidate subtitle table (top-left)

    Args:
        img_bgr: BGR uint8 H×W×3 image (modified in-place).
        outer_bbox: Outerwear bbox ``[x1, y1, x2, y2]``.
        opening_roi: Front-opening ROI box.
        torso_mask: Torso binary mask H×W.
        torso_bbox: Torso bbox for magenta box.
        before_bbox: Inner bbox before refinement (orange dashed).
        after_bbox: Inner bbox after refinement (blue solid).
        inner_mask: Inner garment mask H×W.
        selected_scoring: Scoring dict for the selected candidate.
        extension_debug: Extension debug dict from
            :func:`~fashion_vision.localization.inner_garment_detector._extend_inner_mask_downward`.
        refine_debug: Boundary refinement debug dict.

    Returns:
        Modified BGR image.
    """
    # 1. Torso mask overlay (magenta semi-transparent)
    if torso_mask is not None and torso_mask.sum() > 0:
        overlay = img_bgr.copy()
        overlay[torso_mask > 0] = _MAGENTA
        cv2.addWeighted(overlay, alpha, img_bgr, 1.0 - alpha, 0, img_bgr)

    # Torso bbox (magenta solid)
    if torso_bbox is not None:
        draw_box(img_bgr, torso_bbox, _MAGENTA, "torso")

    # 2. Opening ROI (purple box)
    if opening_roi is not None:
        draw_box(img_bgr, opening_roi, _PURPLE, "opening_roi")

    # 3. Before-refine bbox (orange dashed)
    if before_bbox is not None:
        draw_dashed_box(img_bgr, before_bbox, _ORANGE, dash_len=6, gap_len=3, thickness=2)
        draw_text(img_bgr, "before", (before_bbox[0], max(0, before_bbox[1] - 8)), _ORANGE)

    # 4. Inner garment mask (teal overlay)
    if inner_mask is not None and inner_mask.sum() > 0:
        overlay2 = img_bgr.copy()
        overlay2[inner_mask > 0] = _TEAL
        cv2.addWeighted(overlay2, 0.35, img_bgr, 0.65, 0, img_bgr)

    # 5. After-refine bbox (blue solid)
    if after_bbox is not None:
        draw_box(img_bgr, after_bbox, _BLUE, "inner_refined")

    # 5b. Before-cleanup bbox (green dashed) — shows what cleanup fixed
    if cleanup_before_bbox is not None and cleanup_debug and cleanup_debug.get("cleanup_accepted"):
        draw_dashed_box(img_bgr, cleanup_before_bbox, (0, 200, 100), dash_len=6, gap_len=3, thickness=2)
        draw_text(img_bgr, "before_cleanup", (cleanup_before_bbox[0], max(0, cleanup_before_bbox[1] - 8)), (0, 200, 100))

    # 6. Candidate subtitle (top-left text block)
    lines: list[str] = []
    if selected_scoring:
        sc = selected_scoring
        lines.append(
            f"score={sc.get('score', '?')}  "
            f"source={sc.get('selected_source', '?')}  "
            f"outside_out={sc.get('outside_outer_ratio', '?')}"
        )
        lines.append(
            f"rel_cx={sc.get('rel_cx', '?')}  "
            f"open_core={sc.get('opening_core_overlap', '?')}  "
            f"torso_ov={sc.get('torso_overlap', '?')}"
        )
        lines.append(
            f"bw_rat={sc.get('bbox_w_ratio', '?')}  "
            f"bh_rat={sc.get('bbox_h_ratio', '?')}  "
            f"area_rat={sc.get('area_ratio_bbox', '?')}"
        )

    if extension_debug:
        ext = extension_debug
        lines.append(
            f"extended={ext.get('extended', False)}  "
            f"matched={ext.get('num_matched', 0)}/{ext.get('num_opening_components', 0)}"
        )

    if refine_debug:
        rd = refine_debug
        lines.append(
            f"refine: h={rd.get('h_refined', False)} v={rd.get('v_refined', False)}"
        )

    if cleanup_debug:
        cd2 = cleanup_debug
        acc = "ACCEPT" if cd2.get("cleanup_accepted") else "REJECT"
        removed = cd2.get("removed_pixels", 0)
        reason = cd2.get("reason", "None") or "None"
        lines.append(
            f"cleanup={acc} removed={removed} reason={reason}"
        )

    # Draw text lines from top-left
    y_off = 16
    for line in lines:
        draw_text(img_bgr, line[:120], (6, y_off), (255, 255, 255))
        y_off += 16

    return img_bgr
