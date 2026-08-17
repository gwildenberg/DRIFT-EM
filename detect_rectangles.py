#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import math  # For shape template geometry
from collections import namedtuple
import random  # For selecting random contours
from scipy.optimize import minimize  # For advanced optimization
import datetime  # For timestamped output filenames
import os  # For path operations
import sys  # For reporting flags that do not apply to the chosen shape
import pandas as pd  # For creating Excel files
from concurrent.futures import ProcessPoolExecutor

# Set fixed random seed for reproducibility
random.seed(42)  # Always use the same random seed for consistent results
np.random.seed(42)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION SHAPE TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════
# A "template" is an (N,2) float32 array of vertices in a local frame centred
# on the origin, with the section's WIDTH spanning x and HEIGHT spanning y.
#
# Convention note: cv2.boxPoints(((cx,cy),(w,h),angle)) rotates the w-side by
# `angle`.  polygon_at() below reproduces that convention exactly, so `angle`
# keeps the same meaning it had when this pipeline was rectangle-only: the
# direction of the WIDTH axis.  Height is expected to be the long axis (that
# is what the initial-angle logic assumes); pass_shape_params() warns if not.
# ═══════════════════════════════════════════════════════════════════════════

SHAPE_CHOICES = ('rect', 'square', 'trapezoid', 'ellipse', 'hull', 'custom')

# Picklable (module-level namedtuple) so it can cross the ProcessPoolExecutor
# boundary.  `kind` lets rectangles keep using cv2.boxPoints, which guarantees
# bit-identical results to the pre-shape version of this script.
SectionTemplate = namedtuple('SectionTemplate', 'verts kind width height')

# Defaults preserve the original hardcoded 110 x 284 px rectangle.
DEFAULT_SECTION_WIDTH = 110.0
DEFAULT_SECTION_HEIGHT = 284.0


def load_custom_shape(shape_file, width, height):
    """
    Load a custom section outline from a text file.

    Format: one vertex per line, "x y" or "x,y".  Blank lines and lines
    starting with # are ignored.  Units are arbitrary — the polygon is
    recentred on its bounding-box centre and rescaled so its bounding box is
    exactly width x height.  This means you can trace an outline in ImageJ at
    any magnification and reuse it at any section size.
    """
    pts = []
    with open(shape_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.replace(',', ' ').split()
            if len(parts) < 2:
                raise ValueError(f"Malformed vertex line in {shape_file}: {line!r}")
            pts.append((float(parts[0]), float(parts[1])))

    if len(pts) < 3:
        raise ValueError(f"{shape_file} needs at least 3 vertices, got {len(pts)}")

    arr = np.asarray(pts, dtype=np.float64)
    lo, hi = arr.min(axis=0), arr.max(axis=0)
    span = hi - lo
    if span[0] <= 0 or span[1] <= 0:
        raise ValueError(f"{shape_file} is degenerate (zero-width or zero-height)")

    arr = arr - (lo + hi) / 2.0          # recentre on bbox centre
    arr[:, 0] *= width / span[0]          # rescale to requested bbox
    arr[:, 1] *= height / span[1]
    return arr.astype(np.float32)


def make_shape_template(shape, width, height, taper=0.7, n_vertices=24,
                        shape_file=None):
    """
    Build the section-shape template.

    Args:
        shape: one of SHAPE_CHOICES.  'hull' returns None — that mode derives a
               per-contour template at optimization time (see hull_template).
        width, height: bounding-box dimensions in pixels.
        taper: trapezoid only.  Ratio of top edge to bottom edge (0 < taper <= 1).
               0.7 means the top is 70% as wide as the bottom.
        n_vertices: ellipse only.  Number of polygon vertices approximating it.
        shape_file: custom only.  Path to a vertex list.

    Returns:
        (N,2) float32 array, or None for 'hull'.
    """
    w2, h2 = width / 2.0, height / 2.0

    if shape in ('rect', 'square'):
        # Rectangles delegate to cv2.boxPoints at placement time (see
        # polygon_at), so the stored verts are only used for area/reporting.
        pts = cv2.boxPoints(((0.0, 0.0), (width, height), 0.0))
        return SectionTemplate(np.asarray(pts, np.float32), 'rect', width, height)

    elif shape == 'trapezoid':
        if not (0 < taper <= 1):
            raise ValueError(f"--taper must be in (0, 1], got {taper}")
        tw2 = w2 * taper
        pts = [(-w2, -h2), (w2, -h2), (tw2, h2), (-tw2, h2)]

    elif shape == 'ellipse':
        if n_vertices < 6:
            raise ValueError(f"--ellipse-vertices must be >= 6, got {n_vertices}")
        pts = [(w2 * math.cos(2 * math.pi * i / n_vertices),
                h2 * math.sin(2 * math.pi * i / n_vertices))
               for i in range(n_vertices)]

    elif shape == 'custom':
        if not shape_file:
            raise ValueError("--section-shape custom requires --shape-file")
        return SectionTemplate(load_custom_shape(shape_file, width, height),
                               'custom', width, height)

    elif shape == 'hull':
        return None  # built per-contour

    else:
        raise ValueError(f"Unknown shape {shape!r}; choose from {SHAPE_CHOICES}")

    return SectionTemplate(np.asarray(pts, dtype=np.float32), shape, width, height)


# process_image() subtracts a dilated Sobel edge band from the binary mask,
# which eats roughly this many pixels off every section boundary.  Measured
# against synthetic ground truth with the default thresholds and the 3x
# 4x4-kernel dilation.  Only matters for shape='hull', where the contour IS the
# template; parametric shapes are unaffected because their size is given, not
# measured.  Retune with --hull-dilate if you change --threshold or the kernel.
DEFAULT_HULL_DILATE_PX = 10.0


def hull_template(contour, simplify_eps_frac=0.01, dilate_px=DEFAULT_HULL_DILATE_PX):
    """
    Derive a template from the contour's own convex hull (shape='hull').

    Used when sections are irregular enough that no parametric template fits.
    The hull is grown by dilate_px to undo the edge-subtraction erosion (see
    DEFAULT_HULL_DILATE_PX), simplified with Douglas-Peucker to keep the vertex
    count sane, then recentred on its centroid.  Because it is already in image
    orientation, the optimizer starts it at angle 0.

    Returns a SectionTemplate whose width/height are the hull's minAreaRect
    dimensions, reported for bookkeeping only.
    """
    hull = cv2.convexHull(contour)

    if dilate_px > 0:
        # Grow via a local raster: exact for arbitrary convex outlines and far
        # simpler to reason about than an analytic polygon offset.
        pad = int(math.ceil(dilate_px)) + 2
        x, y, w, h = cv2.boundingRect(hull)
        mask = np.zeros((h + 2 * pad, w + 2 * pad), np.uint8)
        cv2.fillPoly(mask, [hull.reshape(-1, 2) - [x - pad, y - pad]], 255)
        k = int(2 * round(dilate_px) + 1)
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cs:
            hull = cv2.convexHull(max(cs, key=cv2.contourArea)) + [x - pad, y - pad]

    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, simplify_eps_frac * peri, True)
    if len(approx) < 3:
        approx = hull
    pts = approx.reshape(-1, 2).astype(np.float64)
    centre = pts.mean(axis=0)
    pts -= centre
    (_, _), (mw, mh), _ = cv2.minAreaRect(approx)
    w, h = (mw, mh) if mw <= mh else (mh, mw)
    return SectionTemplate(pts.astype(np.float32), 'hull', float(w), float(h))


def is_180_symmetric(template, tol=1e-3):
    """
    Is this template unchanged by a 180-degree rotation?

    Matters because cv2.minAreaRect only determines a contour's orientation
    modulo 180 degrees.  For symmetric shapes (rectangle, ellipse) that is
    harmless.  For asymmetric ones (trapezoid, most custom outlines, hulls) the
    template can land backwards, so the optimizer must try both orientations
    and keep whichever fits better -- see _optimize_worker.
    """
    v = template.verts.astype(np.float64)
    flipped = -v
    scale = max(np.abs(v).max(), 1.0)
    # Compare as unordered point sets: every flipped vertex must coincide with
    # some original vertex.
    for pt in flipped:
        if np.min(np.hypot(v[:, 0] - pt[0], v[:, 1] - pt[1])) > tol * scale:
            return False
    return True


def inflate_template(template, margin_x=0.0, margin_y=0.0, percent=0.0):
    """
    Grow a template outward, for use AFTER placement.

    Why after: the greedy overlap filter forbids ROIs from touching, so an
    oversized template makes neighbours collide and each accepted ROI blocks the
    next. Detection therefore wants a template at or below the true section
    size. Acquisition wants the opposite -- an ROI that overshoots, so that
    residual misalignment between the optical map and the stage does not clip
    the section. Inflating after placement satisfies both: the filter sees the
    tight template, the .magc records the generous one.

    margin_x/margin_y are per-side, in pixels, applied along the template's own
    width and height axes. percent grows both axes proportionally. Where both
    are given, the larger resulting dimension wins on each axis.

    Vertices are scaled about the template centroid. Exact for rectangles; for
    other shapes it is a proportional grow rather than a true polygon offset,
    which is the intended behaviour here.
    """
    w, h = float(template.width), float(template.height)
    if w <= 0 or h <= 0:
        return template

    new_w = max(w + 2.0 * margin_x, w * (1.0 + percent / 100.0))
    new_h = max(h + 2.0 * margin_y, h * (1.0 + percent / 100.0))
    if new_w == w and new_h == h:
        return template

    sx, sy = new_w / w, new_h / h
    v = template.verts.astype(np.float64)
    c = v.mean(axis=0)
    v = (v - c) * [sx, sy] + c
    return SectionTemplate(v.astype(np.float32), template.kind, new_w, new_h)


def polygon_at(template, cx, cy, angle_deg):
    """
    Place a template at (cx, cy) rotated by angle_deg.

    Rectangles are delegated to cv2.boxPoints so that rect mode reproduces the
    pre-shape version of this script exactly, down to vertex ordering and
    floating-point rounding.  Other shapes use the equivalent rotation, which
    follows the same convention: positive angle rotates the WIDTH (x) axis.
    """
    if template.kind == 'rect':
        return cv2.boxPoints(((cx, cy), (template.width, template.height),
                              angle_deg)).astype(np.float64)

    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    v = template.verts
    x, y = v[:, 0], v[:, 1]
    out = np.empty((v.shape[0], 2), dtype=np.float64)
    out[:, 0] = cx + x * ca - y * sa
    out[:, 1] = cy + x * sa + y * ca
    return out


def template_area(template):
    """Shoelace area of a template polygon, in px^2."""
    v = template.verts
    x = v[:, 0].astype(np.float64)
    y = v[:, 1].astype(np.float64)
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def describe_template(shape, template, width, height):
    """One-line human-readable summary for the run log."""
    if template is None:
        return "shape=hull (per-contour convex hull; --section-width/height ignored)"
    return (f"shape={shape}  bbox={width:.1f} x {height:.1f} px  "
            f"vertices={len(template.verts)}  area={template_area(template):.0f} px^2")

def process_image(image_path, threshold=86, edge_threshold=20, edge_dilate=3,
                  use_clahe=True, close_px=0):
    """
    Threshold an image and optionally subtract a dilated edge band.

    Edge subtraction exists to separate sections that touch: it carves a gap
    along strong gradients so neighbours become separate contours.

    On tissue with internal structure it can backfire. CLAHE amplifies local
    contrast, Sobel then finds edges INSIDE each section as readily as around
    it, and dilation smears those into bands that split one section into
    several fragments. Fragments fall below the area band (no ROI at all), and
    those that survive have centroids offset from the true section centre (an
    ROI that sits partly off the section). If you see either symptom, reduce
    edge_dilate or raise edge_threshold.

    Args:
        threshold: pixels darker than this are candidate section material.
        edge_threshold: gradient magnitude above which a pixel counts as an
            edge. The original value of 20 is very permissive. Raise it to
            subtract only strong section boundaries and ignore internal texture.
        edge_dilate: dilation iterations applied to the edge band. 0 disables
            edge subtraction entirely, which is the right choice when sections
            are well separated on the wafer.
        use_clahe: apply CLAHE before Sobel. Helps find faint section
            boundaries; also amplifies internal tissue texture.
        close_px: morphological closing applied after subtraction, in px. Heals
            fragments back into whole sections. Try roughly the edge band width
            (about 2 x edge_dilate + 2).

    Returns:
        tuple: (image, binary_minus_edges, contours, dilated_edges, binary_minus_edges)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    if edge_dilate <= 0:
        # No edge subtraction: contours come straight from the threshold.
        binary_minus_edges = binary.copy()
        dilated_edges = np.zeros_like(binary)
    else:
        if use_clahe:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            gray_for_edges = clahe.apply(gray)
        else:
            gray_for_edges = gray

        sobelx = cv2.Sobel(gray_for_edges, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray_for_edges, cv2.CV_64F, 0, 1, ksize=5)
        magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX,
                                  dtype=cv2.CV_8U)

        _, edges = cv2.threshold(magnitude, edge_threshold, 255, cv2.THRESH_BINARY)

        kernel = np.ones((4, 4), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=int(edge_dilate))

        binary_minus_edges = cv2.bitwise_and(binary, binary,
                                             mask=cv2.bitwise_not(dilated_edges))

    if close_px > 0:
        k = int(2 * round(close_px) + 1)
        binary_minus_edges = cv2.morphologyEx(
            binary_minus_edges, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    contours, _ = cv2.findContours(binary_minus_edges.copy(), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    return image, binary_minus_edges, contours, dilated_edges, binary_minus_edges

def optimize_rectangle_for_contour(image, contour, template=None,
                                   target_width=DEFAULT_SECTION_WIDTH,
                                   target_height=DEFAULT_SECTION_HEIGHT,
                                   occupied_mask=None):
    """
    Serial, full-image ROI optimizer.

    NOTE: this function is currently NOT called anywhere — save_debug_image()
    uses the parallel _optimize_worker() path instead.  It is kept because it is
    the readable reference implementation, and it is patched in step with the
    worker so the two cannot silently diverge.  If you edit the cost function,
    edit both.
    """
    if template is None:
        template = make_shape_template('rect', target_width, target_height)

    # Store the grayscale image for optimization
    # We will optimize to find the darkest region (lowest pixel values)
    if len(image.shape) == 3:
        gray_for_opt = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_for_opt = image.copy()
    
    # Use minAreaRect to get center and angle directly from the contour.
    # This avoids the 90-degree ambiguity of the moments formula.
    minrect = cv2.minAreaRect(contour)
    (cx, cy), (mw, mh), mangle = minrect

    # `angle` rotates the template's WIDTH (x) axis.  We want the template's
    # LONG axis aligned with the section's long axis, so when height is the long
    # side we point the width axis along the section's SHORT axis.
    # minAreaRect always makes the width the more-horizontal side, so:
    #   mw < mh  → mw is the short side, mangle is already the short-axis direction
    #   mw >= mh → mw is the long side, short axis is 90° away
    short_axis_angle = (mangle % 180) if mw < mh else ((mangle + 90) % 180)
    if target_height >= target_width:
        initial_angle = short_axis_angle
    else:
        initial_angle = (short_axis_angle + 90) % 180
    
    # Cache for optimization - prevents recreating the image on every iteration
    h, w = gray_for_opt.shape[:2]
    
    def rect_cost_components(params):
        """
        Calculate the individual cost components for a rectangle.
        
        Args:
            params: [center_x, center_y, angle_radians]
            
        Returns:
            tuple: (brightness_cost, angle_penalty_cost, overlap_penalty)
        """
        cx, cy, angle_rad = params
        angle_deg = np.degrees(angle_rad)
        
        # Constrain to image boundaries
        cx = np.clip(cx, 0, w-1)
        cy = np.clip(cy, 0, h-1)
        
        # Place the section-shape template at this pose
        box = polygon_at(template, cx, cy, angle_deg).astype(np.int32)
        
        # Check for overlap with existing rectangles using the occupied mask
        overlap_penalty = 0.0
        if occupied_mask is not None:
            # Create a mask for this candidate rectangle
            current_rect_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(current_rect_mask, [box], 1)
            
            # Check if this rectangle overlaps with any existing rectangles
            # by comparing with the occupied mask
            overlap = np.logical_and(current_rect_mask, occupied_mask).sum()
            
            # If there's any overlap, apply a very large penalty
            if overlap > 0:
                overlap_penalty = np.sum(overlap)
        
        # Create a mask for this rectangle for brightness calculation
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [box], 255)
        
        # Extract region using the mask and calculate the sum of pixel values
        # This will measure the total "brightness" inside the rectangle
        # For a grayscale image, lower values mean darker pixels
        masked_region = cv2.bitwise_and(gray_for_opt, gray_for_opt, mask=mask)
        
        # Count non-zero pixels (where mask is applied)
        non_zero_count = cv2.countNonZero(mask)
        
        # Default to high cost if no pixels in mask
        if non_zero_count == 0:
            return float('inf'), 0.0, float('inf')
            
        brightness = np.sum(masked_region) / non_zero_count

        current_angle = angle_deg % 180
        contour_angle = initial_angle % 180
        
        # Calculate angle deviation cost (penalty for rotating too far from initial angle)
        # Handle the discontinuity at 0/180 degrees
        angle_diff = min(
            abs(current_angle - contour_angle),  # Regular difference
            abs(current_angle - (contour_angle + 180) % 180),  # Wrap around one way
            abs((current_angle + 180) % 180 - contour_angle)   # Wrap around other way
        )
        
        angle_penalty = angle_diff * 0.75
        
        return float(brightness), float(angle_penalty), float(overlap_penalty)
    
    def rect_overlap_cost(params):
        """
        Cost function for optimization: sum of pixel values inside the rectangle
        (to be minimized). Lower pixel values (darker) are better.
        
        Args:
            params: [center_x, center_y, angle_radians]
            
        Returns:
            float: Sum of pixel values (lower is better, indicating darker area)
        """
        brightness, angle_penalty, overlap_penalty = rect_cost_components(params)
        
        # Combine the costs - brightness is our main term, angle penalty is secondary
        combined_cost = brightness + angle_penalty + overlap_penalty
        
        return combined_cost
    
    # Initial parameters [x, y, angle_radians]
    initial_params = [cx, cy, np.radians(initial_angle)]
    # Bounds for the parameters:
    # x: can move ±5 pixels from center
    # y: can move ±5 pixels from center
    # angle: can rotate ±90 degrees from initial angle
    bounds = [
        (max(0, cx - 5), min(w-1, cx + 5)),
        (max(0, cy - 5), min(h-1, cy + 5)),
        (np.radians(initial_angle - 90), np.radians(initial_angle + 90))
    ]
    
    # Use the minimize function for optimization
    result = minimize(
        rect_overlap_cost,
        initial_params,
        method='L-BFGS-B',  # Works well with bounds
        bounds=bounds,
        options={'maxiter': 100}
    )
    
    # Get the optimized parameters
    opt_cx, opt_cy, opt_angle_rad = result.x
    opt_angle_deg = np.degrees(opt_angle_rad) % 180
    
    # Calculate the final cost for reporting
    final_cost = rect_overlap_cost([opt_cx, opt_cy, opt_angle_rad])
    
    return {
        'center_x': int(opt_cx),
        'center_y': int(opt_cy),
        'angle': opt_angle_deg,
        'width': float(target_width),
        'height': float(target_height),
        'template': template,
        'score': final_cost,
        'success': result.success
    }


def _optimize_worker(args):
    """
    Top-level picklable worker for parallel ROI optimization.

    Operates on a small grayscale patch cropped around the contour, which is
    ~1000x faster than creating masks on the full 14k×14k image.  Returns
    results in global (full-image) pixel coordinates.
    """
    (contour, gray_patch, patch_x1, patch_y1,
     template, target_width, target_height,
     search_radius, angle_range, hull_dilate) = args

    ph, pw = gray_patch.shape

    # Initial position and angle directly from minAreaRect
    minrect = cv2.minAreaRect(contour)
    (gcx, gcy), (mw, mh), mangle = minrect

    # Convert global center to patch-local coordinates
    lcx = gcx - patch_x1
    lcy = gcy - patch_y1

    if template is None:
        # hull mode: template derives from this contour, already in image
        # orientation, so start at angle 0 and recentre on the hull centroid.
        template = hull_template(contour, dilate_px=hull_dilate)
        target_width, target_height = template.width, template.height
        hull_c = cv2.convexHull(contour).reshape(-1, 2).astype(np.float64)
        gcx, gcy = hull_c.mean(axis=0)
        lcx, lcy = gcx - patch_x1, gcy - patch_y1
        initial_angle = 0.0
    else:
        # Align the template's LONG axis with the contour's long axis.
        # minAreaRect always reports the more-horizontal side as `width`, so:
        #   mw < mh  -> mangle already points along the short axis
        #   mw >= mh -> the short axis is 90 degrees away
        # `angle` rotates the template's width (x) axis, so we want it pointing
        # along the section's short axis when height is the template's long side.
        short_axis_angle = (mangle % 180) if mw < mh else ((mangle + 90) % 180)
        if target_height >= target_width:
            initial_angle = short_axis_angle
        else:
            initial_angle = (short_axis_angle + 90) % 180

    def cost(params):
        lx, ly, angle_rad = params
        angle_deg = np.degrees(angle_rad)
        lx = float(np.clip(lx, 0, pw - 1))
        ly = float(np.clip(ly, 0, ph - 1))

        poly = polygon_at(template, lx, ly, angle_deg).astype(np.int32)

        mask = np.zeros((ph, pw), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        npix = cv2.countNonZero(mask)
        if npix == 0:
            return float('inf')

        brightness = float(gray_patch[mask > 0].sum()) / npix

        cur = angle_deg % 180
        ini = initial_angle % 180
        diff = min(abs(cur - ini),
                   abs(cur - (ini + 180) % 180),
                   abs((cur + 180) % 180 - ini))
        return brightness + diff * 0.75

    # cv2.minAreaRect fixes orientation only modulo 180 degrees.  If the
    # template is not 180-symmetric, the section could be the other way round,
    # so optimize from both starts and keep the better fit.  Symmetric shapes
    # skip this and cost exactly what they did before.
    starts = [initial_angle]
    if not is_180_symmetric(template):
        starts.append((initial_angle + 180.0) % 360.0)

    result = None
    for start_angle in starts:
        ip = [lcx, lcy, np.radians(start_angle)]
        bounds = [
            (max(0.0, lcx - search_radius), min(float(pw - 1), lcx + search_radius)),
            (max(0.0, lcy - search_radius), min(float(ph - 1), lcy + search_radius)),
            (np.radians(start_angle - angle_range), np.radians(start_angle + angle_range)),
        ]
        r = minimize(cost, ip, method='L-BFGS-B', bounds=bounds,
                     options={'maxiter': 100})
        if result is None or r.fun < result.fun:
            result = r

    opt_lx, opt_ly, opt_angle_rad = result.x
    return {
        'center_x': int(opt_lx + patch_x1),
        'center_y': int(opt_ly + patch_y1),
        # mod 360, not 180: for asymmetric shapes the flip is real information
        # and must reach the .magc file.  Symmetric shapes are unaffected
        # in practice because the optimizer never leaves the 0-180 branch.
        'angle': float(np.degrees(opt_angle_rad) % (180 if template.kind in ('rect', 'ellipse') else 360)),
        'width': float(target_width),
        'height': float(target_height),
        'template': template,   # carried through so the .magc writer can
                                # emit the true outline, not a bounding box
        'score': float(result.fun),
        'success': bool(result.success),
    }


def save_debug_image(image_path, min_area=0, max_area=float('inf'), num_rects=5,
                     save_excel=True, num_workers=None, threshold=86,
                     section_shape='rect',
                     section_width=DEFAULT_SECTION_WIDTH,
                     section_height=DEFAULT_SECTION_HEIGHT,
                     taper=0.7, ellipse_vertices=24, shape_file=None,
                     search_radius=5.0, angle_range=90.0,
                     hull_dilate=DEFAULT_HULL_DILATE_PX,
                     edge_threshold=20, edge_dilate=3, use_clahe=True, close_px=0,
                     inflate_margin=(0.0, 0.0), inflate_percent=0.0):
    """
    Save high-resolution debug images showing:
    A) Original image with optimized 24x36 rectangles
    B) Detected contours with random colors (contours outside size limits are shown in gray)
    C) Edge detection image (Sobel)
    D) Binary minus edges (the processed image used for detection)
    
    Args:
        image_path (str): Path to the input image
        min_area (int): Minimum contour area to highlight in color
        max_area (int): Maximum contour area to highlight in color
        num_rects (int): Number of random contours to optimize and display
        save_excel (bool): Whether to save contour data to Excel file

    Returns:
        str: Path to the saved debug image
    """
    # ── Build the section-shape template once, up front ──────────────────────
    template = make_shape_template(section_shape, section_width, section_height,
                                   taper=taper, n_vertices=ellipse_vertices,
                                   shape_file=shape_file)
    print(f"Section template: {describe_template(section_shape, template, section_width, section_height)}")

    # Reference figures used to rank which contours are worth optimizing.
    # For 'hull' there is no fixed template, so ranking falls back to area.
    if template is not None:
        target_aspect_ratio = (max(section_width, section_height) /
                               min(section_width, section_height))
        target_area = template_area(template)
    else:
        target_aspect_ratio = None
        target_area = None

    # Generate timestamp for unique filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get input file name without extension
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Diagnostics go in stats/, NOT beside the original image.
    #
    # magfinder asks for the folder holding the .magc and then looks for the
    # overview TIFF in it. If the debug TIFF sits in the same folder, magfinder
    # picks that up instead of the real overview. Keeping the .magc next to the
    # original image and pushing every diagnostic into stats/ leaves exactly one
    # TIFF where magfinder looks.
    stats_dir = os.path.join(os.path.dirname(os.path.abspath(image_path)), 'stats')
    os.makedirs(stats_dir, exist_ok=True)
    output_path = os.path.join(stats_dir, f"debug_{base_name}_{timestamp}.tiff")
    
    print(f"Processing debug image for {image_path}...")
    print(f"Output will be saved to {output_path}")
    

    # Process image with edge subtraction and Sobel method
    print(f"Steps 1-5: Processing image using edge subtraction with Sobel...")
    
    # Use edge subtraction method
    image, binary, contours, edges, binary_minus_edges = process_image(image_path, threshold=threshold)
    
    # Save contour data to Excel if requested
    if save_excel:
        print("Saving contour data to Excel file...")
        contour_data = []
        
        for i, cnt in enumerate(contours):
            # Calculate contour area
            area = cv2.contourArea(cnt)
            
            # Calculate center of mass using moments
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                # Fallback to bounding rect center if moments calculation fails
                x, y, w, h = cv2.boundingRect(cnt)
                cx, cy = x + w//2, y + h//2
            
            # Calculate bounding rectangle dimensions
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Calculate rectangle fit quality
            # Get the minimum area rectangle that fits the contour
            rect = cv2.minAreaRect(cnt)
            rect_width, rect_height = rect[1]
            
            # Make sure width is the smaller dimension and height is the larger one
            if rect_width > rect_height:
                rect_width, rect_height = rect_height, rect_width
                
            # Calculate area of the min area rectangle
            rect_area = rect_width * rect_height
            
            # Calculate fill ratio (how well the contour fills the rectangle)
            # Higher ratio means better rectangle fit
            fill_ratio = area / rect_area if rect_area > 0 else 0
            
            # Calculate aspect ratio of the rectangle (height/width)
            aspect_ratio = rect_height / rect_width if rect_width > 0 else 0
            
            # How well does this contour match the requested section shape?
            # (target_aspect_ratio / target_area computed once at function top)
            if target_area is None:
                # hull mode: no fixed template to compare against, so rank on
                # area alone — every contour in range is an equally valid target.
                rect_9x20_fit = 1.0
            else:
                aspect_ratio_fit = abs(aspect_ratio - target_aspect_ratio)

                if area == 0:
                    area_ratio = 0  # If area is zero, it's a poor fit
                else:
                    area_ratio = min(area / target_area, target_area / area)  # Between 0 and 1, higher is better

                # Combined fit score (higher is better); aspect weighted more heavily
                rect_9x20_fit = (0.7 * (1 - aspect_ratio_fit / 3.0)) + (0.3 * area_ratio)

                # Clamp the score between 0 and 1
                rect_9x20_fit = max(0, min(1, rect_9x20_fit))
            
            # Add data to the list
            contour_data.append({
                'Contour_ID': i,
                'Area': area,
                'Center_X': cx,
                'Center_Y': cy,
                'Bounding_Box_X': x,
                'Bounding_Box_Y': y,
                'Bounding_Box_Width': w,
                'Bounding_Box_Height': h,
                'Min_Rect_Width': round(rect_width, 2),
                'Min_Rect_Height': round(rect_height, 2),
                'Min_Rect_Area': round(rect_area, 2),
                'Rectangle_Fill_Ratio': round(fill_ratio, 4),
                'Aspect_Ratio': round(aspect_ratio, 4),
                'Rect_9x20_Fit_Score': round(rect_9x20_fit, 4)
            })
        
        # Create DataFrame and save to Excel
        df = pd.DataFrame(contour_data)
        
        # Sort by 9x20 rectangle fit score (highest first)
        df = df.sort_values(by='Rect_9x20_Fit_Score', ascending=False)
        
        xlsx_path = os.path.join(stats_dir, f"contours_{base_name}_{timestamp}.xlsx")
        df.to_excel(xlsx_path, index=False)
        print(f"Saved data for {len(contours)} contours to {xlsx_path}")
    
    # Create a copy of the original image for panel A
    panel_a_image = image.copy()
    
    # Create a 2x2 grid
    print("Step 6: Creating composite image grid...")
    # Get dimensions from the input image
    img_height, img_width = image.shape[:2]
    
    # Set fixed output dimensions for debug view
    height, width = binary.shape  # Standard debug size for all panels
    scale_factor = 1  # Scale factor for output resolution
    combined_height = height * 2 * scale_factor
    combined_width = width * 2 * scale_factor
    combined = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
    
    # Convert grayscale images to BGR for visualization
    print("Step 7: Converting images for visualization...")
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    binary_minus_edges_bgr = cv2.cvtColor(binary_minus_edges, cv2.COLOR_GRAY2BGR)
    
    # Ensure panel_a_image is in BGR format if it's grayscale
    if len(panel_a_image.shape) == 2 or panel_a_image.shape[2] == 1:
        panel_a_image = cv2.cvtColor(panel_a_image, cv2.COLOR_GRAY2BGR)
    
    # No need to resize if scale_factor is 1
    print("Step 8: Preparing images for visualization...")
    
    # Create contour visualization image
    print("Step 9a: Creating contour visualization image with size filtering...")
    contour_image = image.copy()
    
    # Count valid contours (within size limits)
    valid_contours = 0
    valid_contour_list = []
    
    # Generate a random color for each contour
    for i, contour in enumerate(contours):
        # Get contour area
        area = cv2.contourArea(contour)
        
        # Check if contour is within size limits
        if min_area <= area <= max_area:
            # Valid contour - use a random bright color
            color = np.random.randint(0, 255, size=3).tolist()
            
            # Ensure the color has at least one bright channel for visibility
            max_channel = max(color)
            if max_channel < 150:  # If all channels are dim
                bright_channel = np.random.randint(0, 3)  # Choose a random channel to make bright
                color[bright_channel] = 255  # Make that channel bright
                
            valid_contours += 1
            valid_contour_list.append((contour, color))
        else:
            # Invalid contour - use gray
            color = [120, 120, 120]  # Gray color for contours outside size limits
        
        # Draw contour outline slightly thicker for better visibility
        cv2.drawContours(contour_image, [contour], -1, color, 2)
    
    # Report the area filter's effect before anything else can short-circuit.
    # If nothing survives the band, the optimisation block below is skipped
    # entirely -- and that is exactly the case where the user most needs to be
    # told why, so the accounting cannot live only inside that block.
    if not valid_contour_list:
        areas = sorted(cv2.contourArea(c) for c in contours)
        print(f"\n  === Yield accounting ===")
        print(f"    contours found            {len(contours):7d}")
        print(f"    passed area band                0   "
              f"(band: {min_area:.0f} - {max_area:.0f} px^2)")
        print(f"    PLACED                          0")
        if areas:
            import bisect
            below = bisect.bisect_left(areas, min_area)
            above = len(areas) - bisect.bisect_right(areas, max_area)
            print(f"\n    {below} contours fell BELOW the band, {above} above it.")
            pcts = [50, 75, 90, 95, 99]
            qs = [areas[min(len(areas) - 1, int(len(areas) * p / 100))] for p in pcts]
            print(f"    Contour areas: " +
                  "  ".join(f"p{p}={q:.0f}" for p, q in zip(pcts, qs)))
            if below > above:
                print(f"    Most contours are SMALLER than the band. Either the "
                      f"section size you gave is")
                print(f"    too large, or edge subtraction is fragmenting sections. "
                      f"Try a smaller")
                print(f"    --section-width/--section-height, or "
                      f"--area-tolerance 0.3 1.6.")
            else:
                print(f"    Most contours are LARGER than the band. Sections may be "
                      f"merging together;")
                print(f"    try a lower --threshold, or widen with "
                      f"--area-tolerance 0.5 2.5.")

    # Optimize and draw rectangles if we have valid contours
    if valid_contour_list:
        # Create a list of contours to optimize, picking contours with highest 9x20 fit score
        valid_contours_with_score = []
        
        for contour, color in valid_contour_list:
            # Calculate how well this contour fits a 9x20 rectangle
            # Get the minimum area rectangle
            rect = cv2.minAreaRect(contour)
            rect_width, rect_height = rect[1]
            
            # Make sure width is the smaller dimension
            if rect_width > rect_height:
                rect_width, rect_height = rect_height, rect_width
                
            # Calculate aspect ratio
            aspect_ratio = rect_height / rect_width if rect_width > 0 else 0
            
            area = cv2.contourArea(contour)

            if target_area is None:
                # hull mode: rank purely by contour area (largest first)
                fit_score = area
            else:
                # Aspect ratio fit score (lower is better)
                aspect_ratio_fit = abs(aspect_ratio - target_aspect_ratio)

                # Avoid division by zero
                if area == 0:
                    area_ratio = 0  # If area is zero, it's a poor fit
                else:
                    area_ratio = min(area / target_area, target_area / area)  # Between 0 and 1, higher is better

                # Combined fit score (higher is better)
                fit_score = (0.7 * (1 - aspect_ratio_fit / 3.0)) + (0.3 * area_ratio)
                fit_score = max(0, min(1, fit_score))
            
            valid_contours_with_score.append((contour, color, fit_score))
        
        # Sort by fit score (highest first)
        valid_contours_with_score.sort(key=lambda x: x[2], reverse=True)
        
        # Select top contours (up to num_rects)
        num_to_optimize = min(num_rects, len(valid_contours_with_score))
        contours_to_optimize = [(c, color) for c, color, _ in valid_contours_with_score[:num_to_optimize]]
        
        # Log that optimization is being performed
        print(f"Optimizing {num_to_optimize} rectangles (sorted by 9x20 fit)...")
        
        # Create a transparent overlay for the optimized rectangles
        overlay = panel_a_image.copy()
        
        # Draw the original contours in blue
        for contour, _ in contours_to_optimize:
            cv2.drawContours(overlay, [contour], 0, (255, 0, 0), 1)  # Thin blue line
        
        # --- Parallel optimization ---
        # Each worker operates on a small grayscale patch instead of the full image,
        # which is orders of magnitude faster for mask creation.
        if num_workers is None:
            num_workers = os.cpu_count() or 1
        ih, iw = image.shape[:2]
        gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        target_w, target_h = section_width, section_height
        # Enough room for any rotation, plus the translation search radius.
        patch_margin = int(math.ceil(max(target_w, target_h) + search_radius + 10))

        worker_args = []
        for contour, _ in contours_to_optimize:
            bx, by, bw, bh = cv2.boundingRect(contour)
            px1 = max(0, bx - patch_margin)
            py1 = max(0, by - patch_margin)
            px2 = min(iw, bx + bw + patch_margin)
            py2 = min(ih, by + bh + patch_margin)
            worker_args.append((contour, gray_full[py1:py2, px1:px2], px1, py1,
                                template, target_w, target_h,
                                search_radius, angle_range, hull_dilate))

        n_workers = min(num_to_optimize, num_workers)
        chunk = max(1, num_to_optimize // n_workers)
        print(f"  Running parallel optimization: {num_to_optimize} ROIs across {n_workers} workers...")
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            raw_results = list(pool.map(_optimize_worker, worker_args, chunksize=chunk))

        # Greedy overlap filter: accept in order of brightness score (lower = darker = better)
        #
        # The overlap test is done inside the candidate's bounding box, not over
        # the whole image. The naive version allocated a full-image mask per
        # candidate and AND-ed it against the occupied mask; on a 200 Mpx
        # overview with ~1000 sections that is hundreds of GB of allocation
        # churn and minutes of wall time, for a comparison that only ever needs
        # to look at a few hundred px on a side.
        ih, iw = image.shape[:2]
        occupied_mask = np.zeros((ih, iw), dtype=np.uint8)
        optimized_rectangles = []
        verbose_placement = len(raw_results) <= 200
        n_failed_fit, n_offimage, n_overlap = 0, 0, 0
        for rect in sorted(raw_results, key=lambda r: r['score']):
            if rect['score'] >= float('inf'):
                n_failed_fit += 1
                continue
            box = polygon_at(rect['template'], rect['center_x'], rect['center_y'],
                             rect['angle']).astype(np.int32)

            bx, by, bw, bh = cv2.boundingRect(box)
            bx0, by0 = max(0, bx), max(0, by)
            bx1, by1 = min(iw, bx + bw), min(ih, by + bh)
            if bx1 <= bx0 or by1 <= by0:
                n_offimage += 1
                continue

            local = np.zeros((by1 - by0, bx1 - bx0), dtype=np.uint8)
            cv2.fillPoly(local, [box - [bx0, by0]], 1)
            if np.logical_and(local, occupied_mask[by0:by1, bx0:bx1]).any():
                n_overlap += 1
                continue

            optimized_rectangles.append(rect)
            occupied_mask[by0:by1, bx0:bx1] |= local
            cv2.drawContours(overlay, [box], 0, (0, 255, 0), 1)
            if verbose_placement:
                print(f"    Section at ({rect['center_x']}, {rect['center_y']}), "
                      f"angle: {rect['angle']:.1f}°, score: {rect['score']:.0f}")

        # ── Inflate for acquisition, after overlap filtering ────────────────
        # Detection used a tight template so neighbours would not block each
        # other. The .magc needs a generous one so residual misalignment
        # between the optical map and the SEM stage cannot clip a section.
        infl_x, infl_y = inflate_margin
        if infl_x or infl_y or inflate_percent:
            n_pairs_overlapping = 0
            inflated_boxes = []
            for rect in optimized_rectangles:
                tpl = inflate_template(rect['template'], infl_x, infl_y,
                                       inflate_percent)
                rect['template'] = tpl          # what gets written to .magc
                box = polygon_at(tpl, rect['center_x'], rect['center_y'],
                                 rect['angle']).astype(np.int32)
                inflated_boxes.append(box)
                # magenta: the inflated ROI actually recorded
                cv2.drawContours(overlay, [box], 0, (255, 0, 255), 2)

            # How much do the inflated ROIs now overlap each other? Some overlap
            # is the accepted cost of not clipping sections, but it should be a
            # deliberate choice rather than a surprise.
            infl_mask = np.zeros((ih, iw), dtype=np.uint8)
            overlap_px = 0
            total_px = 0
            for box in inflated_boxes:
                bx, by, bw2, bh2 = cv2.boundingRect(box)
                bx0, by0 = max(0, bx), max(0, by)
                bx1, by1 = min(iw, bx + bw2), min(ih, by + bh2)
                if bx1 <= bx0 or by1 <= by0:
                    continue
                loc = np.zeros((by1 - by0, bx1 - bx0), dtype=np.uint8)
                cv2.fillPoly(loc, [box - [bx0, by0]], 1)
                sub = infl_mask[by0:by1, bx0:bx1]
                hit = int(np.logical_and(loc, sub).sum())
                if hit:
                    n_pairs_overlapping += 1
                overlap_px += hit
                total_px += int(loc.sum())
                sub |= loc

            sw_ratio = (rect['template'].width /
                        max(section_width, 1e-9)) if optimized_rectangles else 1.0
            print(f"\n  === Inflation ===")
            if infl_x or infl_y:
                print(f"    margin                {infl_x:.1f} x {infl_y:.1f} px per side")
            if inflate_percent:
                print(f"    percent               {inflate_percent:g}%")
            if optimized_rectangles:
                t0w, t0h = section_width, section_height
                t1w = optimized_rectangles[0]['template'].width
                t1h = optimized_rectangles[0]['template'].height
                print(f"    ROI size              {t0w:.0f} x {t0h:.0f}  ->  "
                      f"{t1w:.0f} x {t1h:.0f} px")
            print(f"    ROIs now overlapping  {n_pairs_overlapping} of "
                  f"{len(inflated_boxes)}")
            if total_px:
                print(f"    overlapping area      {overlap_px / total_px:.1%} of "
                      f"total ROI area")
            print(f"    debug TIFF: green = detected, magenta = inflated "
                  f"(what the .magc records)")

        # Apply the overlay with transparency
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, panel_a_image, 1 - alpha, 0, panel_a_image)
        print(f"  Placed {len(optimized_rectangles)} non-overlapping rectangles")

        # Where the candidates went. This is the fastest way to tell which knob
        # matters: each rejection reason points at a different flag.
        total_contours = len(contours)
        in_band = len(valid_contours_with_score)
        n_optimised = len(contours_to_optimize)
        print(f"\n  === Yield accounting ===")
        print(f"    contours found            {total_contours:7d}")
        _ar = [cv2.contourArea(c) for c in contours]
        _below = sum(1 for a in _ar if a < min_area)
        _above = sum(1 for a in _ar if a > max_area)
        print(f"    rejected by area band     "
              f"{total_contours - in_band:7d}   "
              f"(outside {min_area:.0f} - {max_area:.0f} px^2)")
        print(f"        too small             {_below:7d}   "
              f"<- fragmenting, or entered size too large")
        print(f"        too large             {_above:7d}   "
              f"<- sections merging together")
        print(f"    in band                   {in_band:7d}")
        if n_optimised < in_band:
            print(f"    optimised                 {n_optimised:7d}   "
                  f"(capped by --num-rects)")
        else:
            print(f"    optimised                 {n_optimised:7d}")
        print(f"    dropped, no fit           {n_failed_fit:7d}")
        print(f"    dropped, off image        {n_offimage:7d}")
        print(f"    dropped, overlapped       {n_overlap:7d}   "
              f"<- template may be too large")
        print(f"    PLACED                    {len(optimized_rectangles):7d}")

        if n_optimised > 0:
            frac_overlap = n_overlap / n_optimised
            if frac_overlap > 0.25:
                print(f"\n    {frac_overlap:.0%} of optimised candidates were rejected for "
                      f"overlapping an already-placed")
                print(f"    ROI. On a densely packed wafer an oversized template makes "
                      f"neighbours collide,")
                print(f"    so each accepted ROI blocks the next. Try reducing "
                      f"--section-width/--section-height")
                print(f"    by 10%, which usually recovers most of them.")
        # Point at the right lever: merging and fragmenting need opposite fixes.
        _sig = max(3, int(0.05 * max(total_contours, 1)))
        if _above > _below and _above >= _sig:
            print(f"\n    {_above} contours are LARGER than the band: neighbouring "
                  f"sections are merging")
            print(f"    into one contour, so both are lost.")
            print(f"    First try widening the band:  --area-tolerance 0.5 2.5")
            print(f"      An ROI on a merged blob lands on one of the two sections "
                  f"and magfinder can")
            print(f"      fix the rest, which beats losing both.")
            print(f"    Then try separating them:  --edge-threshold 12   or   "
                  f"--edge-dilate 4")
            print(f"      This only works if the sections have a visible gap or "
                  f"gradient between them;")
            print(f"      it does nothing for sections that physically abut.")
        elif _below > _above and _below >= _sig:
            print(f"\n    {_below} contours are SMALLER than the band: sections are "
                  f"being split into")
            print(f"    fragments, or the size you entered is too large. Make edge "
                  f"subtraction gentler:")
            print(f"    --edge-threshold 60   or   --edge-dilate 1   or   "
                  f"--edge-dilate 0   or   --close 8")

        if total_contours > 0 and (total_contours - in_band) / total_contours > 0.98:
            print(f"\n    Over 98% of contours fell outside the area band. If you "
                  f"expected far more")
            print(f"    sections, widen it with --area-tolerance 0.35 1.6, or check "
                  f"--threshold.")
        
        # Save optimized rectangles to a .magc file
        if optimized_rectangles:
            print("Saving optimized rectangles to .magc file...")
            magc_output_path = os.path.join(
                os.path.dirname(os.path.abspath(image_path)),
                f"{base_name}_{timestamp}.magc")

            with open(magc_output_path, 'w') as f:
                f.write("[sections]\n")
                f.write(f"number = {len(optimized_rectangles)}\n\n")

                for i, rect in enumerate(optimized_rectangles):
                    center_x = rect['center_x']
                    center_y = rect['center_y']
                    angle = rect['angle']

                    # Emit the actual section outline (N vertices), not a
                    # 4-corner bounding box.  magfinder and the ATLAS
                    # Geometry/Polygon element both accept arbitrary N.
                    rect_points = polygon_at(rect['template'], center_x, center_y, angle)
                    polygon_str = ",".join([f"{p[0]:.1f},{p[1]:.1f}" for p in rect_points])
                    area = template_area(rect['template'])

                    f.write(f"[section.{i:04d}]\n")
                    f.write(f"polygon = {polygon_str}\n")
                    f.write(f"center = {center_x:.2f},{center_y:.2f}\n")
                    f.write(f"area = {area:.1f}\n")
                    f.write(f"angle = {angle}\n\n")

                f.write("[end_sections]\n\n")

                serial = ",".join(str(i) for i in range(len(optimized_rectangles)))
                f.write("[serial_order]\n")
                f.write(f"serial_order = {serial}\n\n")
                f.write("[stage_order]\n")
                f.write(f"stage_order = {serial}\n")

            print(f"Saved {len(optimized_rectangles)} rectangles to {magc_output_path}")
    
    # Ensure contour_image is in BGR format if it's grayscale
    if len(contour_image.shape) == 2 or contour_image.shape[2] == 1:
        contour_image = cv2.cvtColor(contour_image, cv2.COLOR_GRAY2BGR)
    
    # Place each image in the grid
    print("Step 9b: Arranging images in grid...")

    combined[0:height, 0:width] = panel_a_image              # A) Original with optimized rectangles
    combined[0:height, width:width*2] = contour_image        # B) Detected Contours
    combined[height:height*2, 0:width] = edges_bgr           # C) Edge detection
    combined[height:height*2, width:width*2] = binary_minus_edges_bgr  # D) Binary minus Edges
    
    # Add text labels
    print("Step 10: Adding text labels...")
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5  # Smaller font scale for fixed-size debug view
    font_thickness = 1
    cv2.putText(combined, "A) Optimized 24x36 Rectangles", (10, 20), font, font_scale, (0, 255, 255), font_thickness)
    cv2.putText(combined, "B) Detected Contours", (width + 10, 20), font, font_scale, (0, 255, 255), font_thickness)
    cv2.putText(combined, "C) Edge Detection (Sobel)", (10, height + 20), font, font_scale, (0, 255, 255), font_thickness)
    cv2.putText(combined, "D) Binary minus Edges", (width + 10, height + 20), font, font_scale, (0, 255, 255), font_thickness)
    
    # Add contour count as text overlay
    total_contours = len(contours)
    cv2.putText(combined, f"Total Contours: {total_contours} (Valid: {valid_contours})", 
               (10, combined_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 
               0.5, (255, 255, 255), 1)
    
    # Add size thresholds and method info
    cv2.putText(combined, f"Size range: {min_area}-{max_area}", 
               (10, combined_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 
               0.5, (255, 255, 255), 1)
    
    # Add timestamp to the image
    cv2.putText(combined, f"Time: {timestamp}", 
               (combined_width // 2, combined_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 
               0.5, (255, 255, 255), 1)
    
    cv2.putText(combined, "Method: Edge Subtraction (Sobel)", 
               (combined_width // 2, combined_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 
               0.5, (255, 255, 255), 1)
    

    # Save the debug image as TIFF
    print("Step 12: Writing output TIFF image...")
    cv2.imwrite(output_path, combined)  # TIFF format preserves full quality
    print(f"Debug image saved to {output_path}")
    print("Debug image processing completed successfully!")
    
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate debug image visualization')
    parser.add_argument('--image_path', required=True,
                        help='Path to the wafer overview image')
    parser.add_argument('--threshold', type=int, default=86,
                        help='Binary threshold for section detection (default: 86). '
                             'Raise to detect lighter sections; lower to reduce noise.')
    parser.add_argument('--min-area', type=int, default=None,
                        help='Minimum contour area. If omitted, derived from the '
                             'section dimensions you gave (see --area-tolerance).')
    parser.add_argument('--max-area', type=int, default=None,
                        help='Maximum contour area. If omitted, derived from the '
                             'section dimensions you gave (see --area-tolerance).')
    parser.add_argument('--num-rects', type=int, default=5000,
                        help='Number of contours to optimize (default: 5000)')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for consistent contour selection (default: 42)')
    parser.add_argument('--excel', action='store_true', default=True,
                        help='Save contour data to Excel file (default: True)')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of CPU workers for parallel optimization (default: all CPUs)')

    # ── Section shape ────────────────────────────────────────────────────────
    shape_group = parser.add_argument_group(
        'section shape',
        'Describes the physical shape of the sections on the wafer. Defaults '
        'reproduce the original hardcoded 110 x 284 px rectangle exactly.')
    shape_group.add_argument('--section-shape', choices=SHAPE_CHOICES, default='rect',
                             help='Section outline: rect, square, trapezoid, ellipse, '
                                  'hull (use each contour\'s own convex hull), or '
                                  'custom (from --shape-file). Default: rect')
    shape_group.add_argument('--section-width', type=float, default=DEFAULT_SECTION_WIDTH,
                             help=f'Section width, short axis (default: {DEFAULT_SECTION_WIDTH:g}). '
                                  'Pixels, unless --um-per-px is given, in which case microns.')
    shape_group.add_argument('--section-height', type=float, default=DEFAULT_SECTION_HEIGHT,
                             help=f'Section height, long axis (default: {DEFAULT_SECTION_HEIGHT:g}). '
                                  'Pixels, unless --um-per-px is given, in which case microns.')
    shape_group.add_argument('--um-per-px', type=float, default=None,
                             help='Image scale. If given, --section-width/--section-height '
                                  'are read as microns and converted to pixels. Lets you enter '
                                  'the dimensions you measured on the block face rather than '
                                  'recomputing them per magnification.')
    shape_group.add_argument('--taper', type=float, default=0.7,
                             help='Trapezoid only: top edge as a fraction of the bottom edge '
                                  '(0 < taper <= 1). Default: 0.7')
    shape_group.add_argument('--ellipse-vertices', type=int, default=24,
                             help='Ellipse only: number of polygon vertices. Default: 24')
    shape_group.add_argument('--shape-file', type=str, default=None,
                             help='Custom only: text file of "x y" vertices, one per line. '
                                  'Recentred and rescaled to the requested width x height, so '
                                  'the tracing magnification does not matter.')
    shape_group.add_argument('--hull-dilate', type=float, default=DEFAULT_HULL_DILATE_PX,
                             help='Hull only: grow each contour by this many px to undo the '
                                  'erosion caused by edge subtraction in process_image '
                                  f'(default: {DEFAULT_HULL_DILATE_PX:g}). Set 0 to disable.')
    shape_group.add_argument('--area-tolerance', type=float, nargs=2,
                             metavar=('LOW', 'HIGH'), default=(0.5, 1.3),
                             help='Area band as multiples of the template area, used '
                                  'when --min-area/--max-area are omitted '
                                  '(default: 0.5 1.3). Edge subtraction shrinks each '
                                  'contour to roughly 0.8x the template area, so the '
                                  'band is deliberately asymmetric. Widen LOW to '
                                  'catch partial sections; lower HIGH to reject '
                                  'merged pairs.')
    infl_group = parser.add_argument_group(
        'ROI inflation',
        'Grows every ROI outward AFTER placement, so the recorded ROI overshoots '
        'the section. Aligning the optical map to the SEM stage from a few corner '
        'reference images leaves residual error, and an ROI sized exactly to the '
        'section will then clip it. Inflation trades some wafer background, and '
        'some overlap between neighbours, for not having to reimage. Because it is '
        'applied after the overlap filter, it does not reduce detection yield -- '
        'you can keep an undersized detection template and still record a generous '
        'ROI.')
    infl_group.add_argument('--inflate-margin', type=float, nargs='+',
                            metavar='M', default=None,
                            help='Per-side margin. One value applies to both axes; '
                                 'two values are width then height. Pixels, unless '
                                 '--um-per-px is given, in which case microns. '
                                 'Absolute margins suit this problem because '
                                 'registration error is an absolute distance, not a '
                                 'fraction of section size.')
    infl_group.add_argument('--inflate-percent', type=float, default=0.0,
                            help='Grow both axes by this percentage (default: 0). '
                                 'Simpler when section sizes vary between '
                                 'experiments, but gives the short axis less '
                                 'absolute margin than the long one.')

    edge_group = parser.add_argument_group(
        'edge subtraction',
        'Edge subtraction separates touching sections by carving a gap along '
        'strong gradients. On tissue with internal structure it can instead '
        'split single sections into fragments, causing missing ROIs and ROIs '
        'that sit only partly over a section. These flags control it.')
    edge_group.add_argument('--edge-threshold', type=int, default=20,
                            help='Gradient magnitude counting as an edge (default: 20, '
                                 'very permissive). Raise to 50-80 to subtract only '
                                 'strong section boundaries and ignore internal texture.')
    edge_group.add_argument('--edge-dilate', type=int, default=3,
                            help='Dilation iterations on the edge band (default: 3). '
                                 'Lower to shrink the carved gap. 0 disables edge '
                                 'subtraction entirely, appropriate when sections are '
                                 'well separated on the wafer.')
    edge_group.add_argument('--no-clahe', action='store_true',
                            help='Skip CLAHE before edge detection. CLAHE helps find '
                                 'faint boundaries but also amplifies internal tissue '
                                 'texture, which is what causes fragmentation.')
    edge_group.add_argument('--close', type=float, default=0, dest='close_px',
                            help='Morphological closing in px after subtraction, to '
                                 'heal fragments back into whole sections (default: 0). '
                                 'Try about 2 x --edge-dilate + 2.')
    shape_group.add_argument('--search-radius', type=float, default=5.0,
                             help='Translation search radius in px around each contour centroid '
                                  '(default: 5). Increase for larger sections or coarser contours.')
    shape_group.add_argument('--angle-range', type=float, default=90.0,
                             help='Rotation search range in degrees either side of the initial '
                                  'estimate (default: 90).')

    args = parser.parse_args()
    random.seed(args.random_seed)

    # Resolve section dimensions to pixels
    section_width, section_height = args.section_width, args.section_height
    if args.um_per_px is not None:
        if args.um_per_px <= 0:
            parser.error('--um-per-px must be positive')
        section_width = section_width / args.um_per_px
        section_height = section_height / args.um_per_px
        print(f"Section size: {args.section_width:g} x {args.section_height:g} um "
              f"@ {args.um_per_px:g} um/px -> {section_width:.1f} x {section_height:.1f} px")

    # Resolve inflation margins, converting from microns if that is the unit in use
    infl_x = infl_y = 0.0
    if args.inflate_margin is not None:
        if len(args.inflate_margin) == 1:
            infl_x = infl_y = float(args.inflate_margin[0])
        elif len(args.inflate_margin) == 2:
            infl_x, infl_y = (float(v) for v in args.inflate_margin)
        else:
            parser.error('--inflate-margin takes one or two values')
        if min(infl_x, infl_y) < 0:
            parser.error('--inflate-margin must be non-negative')
        if args.um_per_px is not None:
            print(f"Inflation margin: {infl_x:g} x {infl_y:g} um per side "
                  f"-> {infl_x / args.um_per_px:.1f} x {infl_y / args.um_per_px:.1f} px")
            infl_x /= args.um_per_px
            infl_y /= args.um_per_px
    if args.inflate_percent < 0:
        parser.error('--inflate-percent must be non-negative')

    if args.section_shape == 'square':
        if section_width != section_height:
            side = min(section_width, section_height)
            print(f"--section-shape square: using {side:.1f} px for both axes "
                  f"(width and height differed)")
            section_width = section_height = side

    if args.section_shape != 'hull' and (section_width <= 0 or section_height <= 0):
        parser.error('--section-width and --section-height must be positive')

    # Validate shape-specific ranges up front, whatever shape was chosen, so a
    # typo is reported rather than silently ignored.
    if not (0 < args.taper <= 1):
        parser.error(f'--taper must be in (0, 1], got {args.taper}')
    if args.ellipse_vertices < 6:
        parser.error(f'--ellipse-vertices must be >= 6, got {args.ellipse_vertices}')
    if args.hull_dilate < 0:
        parser.error(f'--hull-dilate must be >= 0, got {args.hull_dilate}')
    if args.search_radius <= 0:
        parser.error(f'--search-radius must be positive, got {args.search_radius}')
    if not (0 < args.angle_range <= 180):
        parser.error(f'--angle-range must be in (0, 180], got {args.angle_range}')

    # Tell the user when a flag they typed does not apply to the chosen shape,
    # rather than quietly ignoring it.
    _relevant = {'--taper': ('trapezoid',),
                 '--ellipse-vertices': ('ellipse',),
                 '--shape-file': ('custom',),
                 '--hull-dilate': ('hull',)}
    _ignored = [f for f, shapes in _relevant.items()
                if any(a == f or a.startswith(f + '=') for a in sys.argv[1:])
                and args.section_shape not in shapes]
    if _ignored:
        print(f"Note: {', '.join(_ignored)} "
              f"{'has' if len(_ignored) == 1 else 'have'} no effect with "
              f"--section-shape {args.section_shape} and will be ignored.")

    if section_width > section_height and args.section_shape not in ('hull', 'custom'):
        print("Note: width > height. Width is treated as the SHORT axis for "
              "orientation, so the template will be fitted rotated 90 degrees "
              "relative to what you may expect. Swap the values if that is wrong.")

    # Derive the area band from the section size unless it was given explicitly.
    # This is why you only need to measure one section: the dimensions determine
    # the expected contour area, and edge subtraction reduces it by a predictable
    # ~20%, so the band follows from the template rather than being guessed.
    min_area, max_area = args.min_area, args.max_area
    if min_area is None or max_area is None:
        _t = make_shape_template(args.section_shape, section_width, section_height,
                                 taper=args.taper, n_vertices=args.ellipse_vertices,
                                 shape_file=args.shape_file)
        if _t is None:
            # hull: there is no template, so estimate from the bounding box.
            # Irregular sections fill only about 70% of their bbox, and using
            # the full bbox area puts the lower bound above real sections and
            # silently drops them.
            _ta = section_width * section_height * 0.70
        else:
            _ta = template_area(_t)
        lo, hi = args.area_tolerance
        if min_area is None:
            min_area = int(_ta * lo)
        if max_area is None:
            max_area = int(_ta * hi)
        print(f"Area band derived from section size: {min_area} to {max_area} px^2 "
              f"({lo:g}x to {hi:g}x the {_ta:.0f} px^2 template). "
              f"Override with --min-area / --max-area.")

    output_path = save_debug_image(
        args.image_path,
        min_area,
        max_area,
        args.num_rects,
        args.excel,
        args.num_workers,
        args.threshold,
        section_shape=args.section_shape,
        section_width=section_width,
        section_height=section_height,
        taper=args.taper,
        ellipse_vertices=args.ellipse_vertices,
        shape_file=args.shape_file,
        search_radius=args.search_radius,
        angle_range=args.angle_range,
        hull_dilate=args.hull_dilate,
        edge_threshold=args.edge_threshold,
        edge_dilate=args.edge_dilate,
        use_clahe=not args.no_clahe,
        close_px=args.close_px,
        inflate_margin=(infl_x, infl_y),
        inflate_percent=args.inflate_percent)