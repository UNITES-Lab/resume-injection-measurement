"""
HCD Stage 1: Rule-Based Hidden Content Detection in PDF Resumes

This script performs pixel-level analysis of PDF documents to detect hidden
content injection attacks. It implements four complementary detection methods:

  Method 1 - Tiny Font: flags text rendered below a readable font size.
  Method 2 - Solid Color Block: flags text regions with near-zero visual
             variance (text and background are visually indistinguishable).
  Method 3 - Same Color as Background: flags text whose color is within a
             small Euclidean distance of the estimated background color.
  Method 4 - Phantom Ink: flags text spans declared in the PDF structure but
             lacking corresponding ink pixels in the rendered image.

Output: a JSON report with label (0/1), detections, positions, and timing,
plus an annotated PDF with red bounding boxes around flagged regions.
"""

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np

# ============================================================
# Thresholds & Constants
# ============================================================

# Method 1: text smaller than this (in pt) is considered hidden
THRESHOLD_FONT_SIZE = 4.0

# Method 3: maximum Euclidean RGB distance to treat text as same-color-as-bg
THRESHOLD_COLOR_DISTANCE = 15.0

# Merged detection must exceed this character count to be reported
MIN_CONTENT_LENGTH = 3

# Spans starting with these prefixes are internal markers; skip them
IGNORE_PREFIXES = ["#HRJ#", ".#HRJ#"]

# Characters that carry no semantic payload (decorators, bullets, etc.)
SPECIAL_SYMBOLS_SET = set([
    '_', '-', '=', '*', '.', ' ', '~', '#', ',', ':', ';', '|', '/', '\\',
    '"', "'", '\u201c', '\u201d', '\u2018', '\u2019',
    '\u25cf', '\u2022', '\u00b7', '\u2027', '\u2026', '`', '^',
    '\u00b0', '\u00ba', '\u00a9', '\u00ae', '\u2122', '',
])

# Rendering resolution for pixel analysis (3x = 216 DPI)
ANALYSIS_MATRIX = fitz.Matrix(3, 3)

# Margin (in pixels) when sampling background around a text region
BACKGROUND_MARGIN = 2

# Quantization step for bucketing similar colors
COLOR_QUANTIZATION_STEP = 15

# Method 2: regions with grayscale std-dev below this are "solid blocks"
THRESHOLD_VISUAL_VARIANCE = 3.0

# Method 4: pixel-color tolerance when counting ink pixels
THRESHOLD_INK_TOLERANCE = 30.0

# Method 4: minimum fraction of ink pixels to consider text actually visible
THRESHOLD_INK_DENSITY = 0.015

# Method 4 is only applied when region std-dev is below this value
PHANTOM_CHECK_STD_THRESHOLD = 15.0


# ============================================================
# Helper: Zero-Width / Invisible Character Check
# ============================================================

ZERO_WIDTH_CHARS = {
    '\u200B',  # Zero Width Space
    '\u200C',  # Zero Width Non-Joiner
    '\u200D',  # Zero Width Joiner
    '\uFEFF',  # Zero Width No-Break Space (BOM)
    '\u2060',  # Word Joiner
    '\u180E',  # Mongolian Vowel Separator
    '\u061C',  # Arabic Letter Mark
}


def is_zero_width_or_invisible_chars(text):
    """Return True if text contains only zero-width chars, special symbols, or whitespace."""
    if not text:
        return True
    for char in text:
        if char in ZERO_WIDTH_CHARS or char in SPECIAL_SYMBOLS_SET or char.isspace():
            continue
        return False
    return True


# ============================================================
# Data Structures
# ============================================================

class HiddenSpan:
    """A single detected hidden-text span before merging."""

    def __init__(self, text, bbox, reason, page_num, debug_info=None):
        self.text = text
        self.bbox = bbox
        self.reason = reason
        self.page_num = page_num
        self.debug_info = debug_info


# ============================================================
# Color Utilities
# ============================================================

def get_rgb_from_int(color_int):
    """Convert a PyMuPDF color value (int or float tuple) to an (R, G, B) tuple in 0-255."""
    if color_int is None:
        return (0, 0, 0)
    if isinstance(color_int, int):
        return ((color_int >> 16) & 0xFF, (color_int >> 8) & 0xFF, color_int & 0xFF)
    if isinstance(color_int, (list, tuple)):
        if len(color_int) >= 3:
            return (int(color_int[0] * 255), int(color_int[1] * 255), int(color_int[2] * 255))
        elif len(color_int) == 1:
            val = int(color_int[0] * 255)
            return (val, val, val)
    return (0, 0, 0)


def calculate_color_distance(rgb1, rgb2):
    """Euclidean distance between two RGB tuples."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)))


def quantize_color(rgb, step=COLOR_QUANTIZATION_STEP):
    """Snap each channel to the nearest multiple of *step*."""
    return tuple(int(round(c / step) * step) for c in rgb)


# ============================================================
# Pixel-Region Analysis
# ============================================================

def get_region_stats(pixmap):
    """Return (grayscale_std_dev, dominant_color) for a rendered region.

    Uses NumPy vectorised operations and optional 2x down-sampling for speed.
    """
    if pixmap.width == 0 or pixmap.height == 0:
        return 0.0, (255, 255, 255)

    n = pixmap.n
    width = pixmap.width
    height = pixmap.height

    samples = np.frombuffer(pixmap.samples, dtype=np.uint8)
    pixels = samples.reshape((height, width, n))

    # Down-sample large regions to cap work at ~2500 pixels
    if width * height > 10000:
        pixels = pixels[::2, ::2, :]

    r = pixels[:, :, 0].astype(np.float32)
    g = pixels[:, :, 1].astype(np.float32)
    b = pixels[:, :, 2].astype(np.float32)

    gray = 0.299 * r + 0.587 * g + 0.114 * b
    std_dev = float(np.std(gray))

    # Dominant color via quantised histogram
    r_q = (np.round(r / COLOR_QUANTIZATION_STEP) * COLOR_QUANTIZATION_STEP).astype(np.uint8)
    g_q = (np.round(g / COLOR_QUANTIZATION_STEP) * COLOR_QUANTIZATION_STEP).astype(np.uint8)
    b_q = (np.round(b / COLOR_QUANTIZATION_STEP) * COLOR_QUANTIZATION_STEP).astype(np.uint8)

    color_codes = (r_q.flatten().astype(np.uint32) * 65536
                   + g_q.flatten().astype(np.uint32) * 256
                   + b_q.flatten().astype(np.uint32))

    unique_colors, counts = np.unique(color_codes, return_counts=True)
    dominant_code = unique_colors[np.argmax(counts)]

    dominant_color = (
        int((dominant_code >> 16) & 0xFF),
        int((dominant_code >> 8) & 0xFF),
        int(dominant_code & 0xFF),
    )
    return std_dev, dominant_color


def get_background_color_excluding_text(pixmap, text_rgb, tolerance=40.0):
    """Estimate the background color by excluding pixels close to text_rgb.

    Returns an (R, G, B) tuple, or None if every pixel is near the text color
    (which typically means the text is large/bold and dominates the bbox).
    """
    if pixmap.width == 0 or pixmap.height == 0:
        return None

    n = pixmap.n
    width = pixmap.width
    height = pixmap.height

    samples = np.frombuffer(pixmap.samples, dtype=np.uint8)
    pixels = samples.reshape((height, width, n))

    total_pixels = width * height
    if total_pixels > 5000:
        step = max(1, int(math.sqrt(total_pixels / 5000)))
        pixels = pixels[::step, ::step, :]

    r = pixels[:, :, 0].flatten().astype(np.float32)
    g = pixels[:, :, 1].flatten().astype(np.float32)
    b = pixels[:, :, 2].flatten().astype(np.float32)

    text_r, text_g, text_b = text_rgb

    dist = np.sqrt((r - text_r) ** 2 + (g - text_g) ** 2 + (b - text_b) ** 2)
    mask = dist >= tolerance

    if not np.any(mask):
        return None

    r_filtered = r[mask]
    g_filtered = g[mask]
    b_filtered = b[mask]

    r_q = (np.round(r_filtered / COLOR_QUANTIZATION_STEP) * COLOR_QUANTIZATION_STEP).astype(np.uint8)
    g_q = (np.round(g_filtered / COLOR_QUANTIZATION_STEP) * COLOR_QUANTIZATION_STEP).astype(np.uint8)
    b_q = (np.round(b_filtered / COLOR_QUANTIZATION_STEP) * COLOR_QUANTIZATION_STEP).astype(np.uint8)

    color_codes = r_q.astype(np.uint32) * 65536 + g_q.astype(np.uint32) * 256 + b_q.astype(np.uint32)
    unique_colors, counts = np.unique(color_codes, return_counts=True)
    bg_code = unique_colors[np.argmax(counts)]

    return (
        int((bg_code >> 16) & 0xFF),
        int((bg_code >> 8) & 0xFF),
        int(bg_code & 0xFF),
    )


def check_phantom_ink_density(pixmap, text_rgb, tolerance=THRESHOLD_INK_TOLERANCE):
    """Check whether ink pixels matching text_rgb actually exist in the rendered region.

    Returns (density, is_phantom) where density is the fraction of pixels
    that match the text color within *tolerance*, and is_phantom is True when
    that fraction falls below THRESHOLD_INK_DENSITY.
    """
    if pixmap.width == 0 or pixmap.height == 0:
        return 0.0, True

    n = pixmap.n
    width = pixmap.width
    height = pixmap.height
    total_pixels = width * height

    samples = np.frombuffer(pixmap.samples, dtype=np.uint8)
    pixels = samples.reshape((height, width, n))

    # Sub-sample large regions (cap at ~2000 pixels)
    max_sample_pixels = 2000
    if total_pixels > max_sample_pixels:
        sample_ratio = math.sqrt(total_pixels / max_sample_pixels)
        step = max(1, int(sample_ratio))
        pixels = pixels[::step, ::step, :]

    r = pixels[:, :, 0].astype(np.float32)
    g = pixels[:, :, 1].astype(np.float32)
    b = pixels[:, :, 2].astype(np.float32)

    text_r, text_g, text_b = text_rgb

    dist = np.sqrt((r - text_r) ** 2 + (g - text_g) ** 2 + (b - text_b) ** 2)
    found_ink = int(np.sum(dist < tolerance))
    total_scanned = dist.size

    if total_scanned == 0:
        return 0.0, True

    density = found_ink / total_scanned
    is_phantom = density < THRESHOLD_INK_DENSITY

    return float(density), is_phantom


# ============================================================
# Detection Merging
# ============================================================

def merge_detections(raw_detections):
    """Merge adjacent HiddenSpan objects into consolidated detection records.

    Returns (detections_list, positions_list).
    """
    if not raw_detections:
        return [], []

    merged_detections = []
    merged_positions = []

    raw_detections.sort(key=lambda x: (x.page_num, x.bbox[1], x.bbox[0]))
    current_group = []

    def finalize_group(group):
        if not group:
            return

        combined_text = " ".join(item.text for item in group)

        if len(combined_text.strip()) <= MIN_CONTENT_LENGTH:
            return
        for prefix in IGNORE_PREFIXES:
            if combined_text.strip().startswith(prefix):
                return

        # Single long token with no spaces is likely a garbled artifact, not real content
        if ' ' not in combined_text.strip() and len(combined_text.strip()) > 15:
            return

        first = group[0]
        min_x = min(item.bbox[0] for item in group)
        min_y = min(item.bbox[1] for item in group)
        max_x = max(item.bbox[2] for item in group)
        max_y = max(item.bbox[3] for item in group)
        merged_bbox = [min_x, min_y, max_x, max_y]

        reasons = sorted(set(item.reason for item in group))
        reason_str = ", ".join(reasons)

        debug_msg = f"Methods: {reason_str}"
        if group[0].debug_info:
            debug_msg += f" | {group[0].debug_info}"

        merged_detections.append({
            "excerpt": combined_text,
            "explanation": f"Hidden content detected via {reason_str}",
        })

        merged_positions.append({
            "excerpt": combined_text,
            "page": first.page_num,
            "bbox": merged_bbox,
            "explanation": f"Hidden content detected via {reason_str}",
            "debug": debug_msg,
        })

    for item in raw_detections:
        if not current_group:
            current_group.append(item)
            continue

        last = current_group[-1]
        # Merge spans on the same page whose vertical gap is small
        if item.page_num == last.page_num and (item.bbox[1] - last.bbox[3]) < 15.0:
            current_group.append(item)
        else:
            finalize_group(current_group)
            current_group = [item]

    finalize_group(current_group)
    return merged_detections, merged_positions


# ============================================================
# Core Analysis
# ============================================================

def analyze_pdf_content(pdf_path):
    """Run all four detection methods on every text span in the PDF.

    Returns (detections, positions, timing_stats).
    """
    doc = fitz.open(pdf_path)
    raw_detections = []

    timing_stats = {
        "text_extraction": 0.0,
        "method1_tiny_font": 0.0,
        "pixmap_rendering": 0.0,
        "region_stats_calculation": 0.0,
        "method2_solid_color": 0.0,
        "method3_same_color": 0.0,
        "method4_phantom_ink": 0.0,
        "merge_detections": 0.0,
        "total_spans_processed": 0,
        "total_pages": len(doc),
    }

    total_start = time.perf_counter()

    for page_idx, page in enumerate(doc):
        page_num = page_idx + 1

        t_start = time.perf_counter()
        blocks = page.get_text("dict")["blocks"]
        timing_stats["text_extraction"] += time.perf_counter() - t_start

        for block in blocks:
            if block.get("type") != 0:
                continue

            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "")

                    if not text.strip():
                        continue
                    if len(text.strip()) == 1 and not text.strip().isalnum():
                        continue

                    clean_text = text.strip()

                    if is_zero_width_or_invisible_chars(clean_text):
                        continue

                    is_ignored = False
                    for prefix in IGNORE_PREFIXES:
                        if clean_text.startswith(prefix):
                            is_ignored = True
                            break
                    if is_ignored:
                        continue

                    timing_stats["total_spans_processed"] += 1

                    bbox = span["bbox"]
                    size = span["size"]
                    color_int = span["color"]

                    # Pre-check: detect alignment padding (long runs of whitespace)
                    # to avoid false positives on phantom-ink for formatted spans
                    whitespace_count = sum(1 for c in text if c.isspace())
                    whitespace_ratio = whitespace_count / len(text) if len(text) > 0 else 0
                    is_alignment_padding = (len(text) > 10 and whitespace_ratio > 0.6)

                    # --- METHOD 1: Tiny Font ---
                    t_start = time.perf_counter()
                    if size < THRESHOLD_FONT_SIZE:
                        timing_stats["method1_tiny_font"] += time.perf_counter() - t_start
                        raw_detections.append(HiddenSpan(
                            text=clean_text, bbox=bbox, reason="tiny_font", page_num=page_num,
                            debug_info=f"size={round(size, 2)}pt",
                        ))
                        continue
                    timing_stats["method1_tiny_font"] += time.perf_counter() - t_start

                    # Prepare for visual checks
                    text_rgb = get_rgb_from_int(color_int)
                    clip_rect = fitz.Rect(bbox)

                    t_start = time.perf_counter()
                    try:
                        clip_pix = page.get_pixmap(matrix=ANALYSIS_MATRIX, clip=clip_rect, alpha=False)
                    except Exception:
                        timing_stats["pixmap_rendering"] += time.perf_counter() - t_start
                        continue
                    timing_stats["pixmap_rendering"] += time.perf_counter() - t_start

                    t_start = time.perf_counter()
                    std_dev, dominant_inner_color = get_region_stats(clip_pix)
                    timing_stats["region_stats_calculation"] += time.perf_counter() - t_start

                    # --- METHOD 2: Low Visual Variance (Solid Block) ---
                    t_start = time.perf_counter()
                    if std_dev < THRESHOLD_VISUAL_VARIANCE:
                        timing_stats["method2_solid_color"] += time.perf_counter() - t_start
                        raw_detections.append(HiddenSpan(
                            text=clean_text, bbox=bbox, reason="solid_color_block", page_num=page_num,
                            debug_info=f"std_dev={round(std_dev, 2)}",
                        ))
                        continue
                    timing_stats["method2_solid_color"] += time.perf_counter() - t_start

                    # --- METHOD 3: Text-Background Color Contrast ---
                    t_start = time.perf_counter()
                    background_color = get_background_color_excluding_text(clip_pix, text_rgb)

                    if background_color is not None:
                        dist_to_bg = calculate_color_distance(text_rgb, background_color)
                        if dist_to_bg < THRESHOLD_COLOR_DISTANCE:
                            timing_stats["method3_same_color"] += time.perf_counter() - t_start
                            raw_detections.append(HiddenSpan(
                                text=clean_text, bbox=bbox, reason="same_color_as_background", page_num=page_num,
                                debug_info=f"text_to_bg={round(dist_to_bg, 1)}, bg={background_color}",
                            ))
                            continue
                    # background_color is None -> text dominates bbox (large/bold); not hidden
                    timing_stats["method3_same_color"] += time.perf_counter() - t_start

                    # --- METHOD 4: Phantom Text (Ink Density) ---
                    if not is_alignment_padding and std_dev < PHANTOM_CHECK_STD_THRESHOLD:
                        t_start = time.perf_counter()
                        density, is_phantom = check_phantom_ink_density(clip_pix, text_rgb)
                        timing_stats["method4_phantom_ink"] += time.perf_counter() - t_start

                        if is_phantom:
                            raw_detections.append(HiddenSpan(
                                text=clean_text, bbox=bbox, reason="phantom_text_no_ink", page_num=page_num,
                                debug_info=f"ink_density={round(density * 100, 3)}%, std_dev={round(std_dev, 2)}",
                            ))
                            continue

    doc.close()

    t_start = time.perf_counter()
    merged_result = merge_detections(raw_detections)
    timing_stats["merge_detections"] = time.perf_counter() - t_start

    timing_stats["total_time"] = time.perf_counter() - total_start

    return merged_result[0], merged_result[1], timing_stats


# ============================================================
# PDF Marking
# ============================================================

def mark_pdf_visuals(pdf_path, output_path, positions):
    """Draw red rectangles around detected positions and save a new PDF."""
    doc = fitz.open(pdf_path)
    count = 0
    for pos in positions:
        page_idx = pos["page"] - 1
        if 0 <= page_idx < len(doc):
            page = doc[page_idx]
            rect = fitz.Rect(pos["bbox"])
            page.draw_rect(rect, color=(1, 0, 0), width=1.5)
            count += 1
    doc.save(output_path, garbage=4, deflate=True)
    doc.close()
    return count


# ============================================================
# End-to-End Processing
# ============================================================

def process_pdf(pdf_path, output_dir=None):
    """Analyse a single PDF and write the JSON report and marked PDF.

    Returns the result dict, or None on failure.
    """
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        print(f"Error: File not found: {pdf_path}")
        return None

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
    else:
        out_path = pdf_file.parent

    base_name = pdf_file.stem
    json_path = out_path / f"{base_name}_scan_result.json"
    marked_pdf_path = out_path / f"{base_name}_scan_marked.pdf"

    print(f"Scanning: {pdf_file.name}")

    try:
        overall_start = time.perf_counter()

        detections, positions, timing_stats = analyze_pdf_content(str(pdf_file))

        has_hidden = len(detections) > 0
        summary = "Hidden content detected." if has_hidden else "No hidden content injection detected."

        mark_pdf_time = 0.0
        if has_hidden:
            t_start = time.perf_counter()
            mark_pdf_visuals(str(pdf_file), str(marked_pdf_path), positions)
            mark_pdf_time = time.perf_counter() - t_start
            print(f"Saved marked PDF: {marked_pdf_path}")

            original_copy = out_path / f"{base_name}.pdf"
            if not original_copy.exists():
                shutil.copy2(pdf_file, original_copy)
        else:
            if marked_pdf_path.exists():
                try:
                    os.remove(marked_pdf_path)
                except OSError:
                    pass

        overall_time = time.perf_counter() - overall_start

        timing_ms = {
            "text_extraction_ms": round(timing_stats["text_extraction"] * 1000, 2),
            "pixmap_rendering_ms": round(timing_stats["pixmap_rendering"] * 1000, 2),
            "region_stats_calculation_ms": round(timing_stats["region_stats_calculation"] * 1000, 2),
            "method1_tiny_font_ms": round(timing_stats["method1_tiny_font"] * 1000, 2),
            "method2_solid_color_ms": round(timing_stats["method2_solid_color"] * 1000, 2),
            "method3_same_color_ms": round(timing_stats["method3_same_color"] * 1000, 2),
            "method4_phantom_ink_ms": round(timing_stats["method4_phantom_ink"] * 1000, 2),
            "merge_detections_ms": round(timing_stats["merge_detections"] * 1000, 2),
            "mark_pdf_ms": round(mark_pdf_time * 1000, 2),
            "analysis_total_ms": round(timing_stats["total_time"] * 1000, 2),
            "overall_total_ms": round(overall_time * 1000, 2),
            "total_pages": timing_stats["total_pages"],
            "total_spans_processed": timing_stats["total_spans_processed"],
        }

        print("\n===== Timing Statistics =====")
        print(f"  Total pages: {timing_ms['total_pages']}")
        print(f"  Total spans processed: {timing_ms['total_spans_processed']}")
        print(f"  Text extraction: {timing_ms['text_extraction_ms']:.2f} ms")
        print(f"  Pixmap rendering: {timing_ms['pixmap_rendering_ms']:.2f} ms")
        print(f"  Region stats calculation: {timing_ms['region_stats_calculation_ms']:.2f} ms")
        print(f"  Method 1 (tiny font): {timing_ms['method1_tiny_font_ms']:.2f} ms")
        print(f"  Method 2 (solid color): {timing_ms['method2_solid_color_ms']:.2f} ms")
        print(f"  Method 3 (same color): {timing_ms['method3_same_color_ms']:.2f} ms")
        print(f"  Method 4 (phantom ink): {timing_ms['method4_phantom_ink_ms']:.2f} ms")
        print(f"  Merge detections: {timing_ms['merge_detections_ms']:.2f} ms")
        print(f"  Mark PDF: {timing_ms['mark_pdf_ms']:.2f} ms")
        print(f"  --- Analysis total: {timing_ms['analysis_total_ms']:.2f} ms")
        print(f"  --- Overall total: {timing_ms['overall_total_ms']:.2f} ms")
        print("==============================\n")

        result_json = {
            "label": 1 if has_hidden else 0,
            "label_explanation": "hidden_text_injection" if has_hidden else "no_injection",
            "summary": summary,
            "detections": detections,
            "positions": positions,
            "stats": {
                "scan_method": "fitz_rule_based_v5_padding_aware",
                "total_detections": len(detections),
            },
            "timing_stats": timing_ms,
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result_json, f, indent=2, ensure_ascii=False)

        print(f"Result: Label {result_json['label']} - {summary}")
        return result_json

    except Exception as e:
        print(f"Critical error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="HCD Stage 1: Rule-based hidden content detection in PDF resumes.",
    )
    parser.add_argument(
        "--pdf", required=True,
        help="Path to the input PDF file.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Directory for output files (JSON report and marked PDF). "
             "Defaults to the same directory as the input PDF.",
    )
    args = parser.parse_args()
    process_pdf(args.pdf, args.output)


if __name__ == "__main__":
    main()
