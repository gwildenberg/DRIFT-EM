# Connectomics Rectangle Detection - Claude Code Guide

This repository contains a computer vision pipeline for detecting and optimizing rectangular regions in microscopy images. The main script `detect_rectangles.py` is designed for detecting dark rectangles of specific dimensions, but can be adapted for different use cases.

## Quick Start

To run the default detection on your image:
```bash
python detect_rectangles.py --image_path your_image.tif
```

## Adapting for Different Rectangle Properties

### 1. Detecting Bright vs Dark Rectangles

**Current behavior:** Optimizes for dark rectangles (minimizes pixel intensity)

**To detect bright rectangles:**
- In `optimize_rectangle_for_contour()` function at line 176
- Change: `brightness = np.sum(masked_region) / non_zero_count`
- To: `brightness = -np.sum(masked_region) / non_zero_count` (negative to maximize brightness)

**To detect medium brightness rectangles:**
- Calculate target brightness from image statistics
- Modify cost function to minimize `abs(brightness - target_brightness)`

### 2. Changing Rectangle Dimensions

**Current target:** 24×36 pixels (width × height)

**To change dimensions:**
- Modify `target_width=36, target_height=24` parameters in function calls
- Update aspect ratio calculations at lines 330, 456: change `target_aspect_ratio = 20.0 / 9.0` to your ratio
- Update target area calculations at lines 336, 463: change `target_area = 9 * 20` to your area

### 3. Adjusting Detection Sensitivity

**Binary threshold (line 35):**
- Current: `threshold = 86`
- Lower values: detect more subtle features
- Higher values: detect only high-contrast features

**Edge detection threshold (line 52):**
- Current: `threshold = 20`
- Lower values: capture more edge details
- Higher values: only strong edges

**Contour area filtering:**
- Use `--min-area` and `--max-area` command line arguments
- Default: 1-400 pixels

### 4. Shape Flexibility

**Current:** Optimized for rectangles with specific aspect ratios

**For squares:**
- Set `target_width = target_height`
- Modify aspect ratio target to 1.0

**For different aspect ratios:**
- Calculate new `target_aspect_ratio = height / width`
- Update scoring functions accordingly

## Common Adaptations

### Detect Light Rectangles on Dark Background
```python
# In optimize_rectangle_for_contour(), line 176:
brightness = -np.sum(masked_region) / non_zero_count  # Maximize instead of minimize
```

### Change to 50×75 pixel rectangles
```python
# Update function calls with:
target_width=75, target_height=50
# Update aspect ratio:
target_aspect_ratio = 50.0 / 75.0  # or 2/3
# Update area:
target_area = 50 * 75
```

### More sensitive detection
```python
# Line 35: Lower binary threshold
_, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
# Line 52: Lower edge threshold  
_, edges = cv2.threshold(magnitude, 10, 255, cv2.THRESH_BINARY)
```

## Output Files

- **Debug TIFF:** `debug_[filename]_[timestamp].tiff` - 4-panel visualization
- **Excel data:** `d.xlsx` - contour analysis with fit scores
- **Coordinates:** `[filename]_[timestamp].magc` - rectangle coordinates (×10 scaled)

## Key Parameters to Tune

1. **Binary threshold** (line 35): Controls initial shape detection
2. **Edge threshold** (line 52): Controls edge sensitivity
3. **Target dimensions**: Rectangle width/height
4. **Optimization bounds** (lines 217-221): Search space for rectangle placement
5. **Area filters**: `--min-area` and `--max-area` command line options

## Troubleshooting

- **No rectangles detected:** Lower binary/edge thresholds or increase area range
- **Too many false positives:** Raise thresholds or tighten area constraints
- **Wrong brightness:** Check if optimizing for dark vs bright regions
- **Wrong shapes:** Verify aspect ratio targets match your rectangles

Ask Claude to help modify specific parameters or add new detection modes!