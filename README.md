# DRIFT-EM

Locate ultrathin sections randomly placed on a wafer, draw an imaging ROI
around each one, and emit a project file for ZEISS ATLAS.

```
wafer overview image
    |
    |  detect_rectangles.py          automatic section detection
    v
.magc  -------------------------->   magfinder (manual)
    |                                    add any sections the
    |                                    detector missed
    v
generate_and_fuse.py                 build the ATLAS project
    |
    v
.a5proj  ------------------------>   ATLAS
```

## CONTENTS

```
detect_rectangles.py            section detection, image -> .magc
generate_and_fuse.py            .magc -> .a5proj
check_a5proj.py                 validate a project before using it
create_blank_template.py        regenerate the ATLAS template
atlas_template/
    blank_template.a5proj       REQUIRED by generate_and_fuse.py
example_shape.txt               sample --shape-file input
requirements.txt
```

## INSTALL

```
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -c "import cv2, scipy, pandas, lxml, networkx; print('ok')"
```

Tested on Python 3.9 and 3.12, and on opencv-python 4.13 and 5.0.

If you use conda, deactivate it before activating the venv. Conda's base
environment takes priority on PATH and will shadow the venv even when the
prompt suggests otherwise.

Every new shell needs `source .venv/bin/activate` again, and the path is
relative, so run it from the folder holding .venv. Note that the (.venv) label
in your prompt is only a string the activate script sets -- it persists after
you cd elsewhere and does not track whether the venv is still on PATH. If
anything behaves oddly, `which python` is the reliable check: the path it
prints must contain .venv.

## 1.  MEASURE ONE SECTION

Measure a single representative section on your overview image, in pixels,
and note its shape. Everything else follows from that.

In ImageJ / Fiji: open the overview, drag the straight-line tool across the
narrow dimension of one clearly isolated section, and read the length off the
status bar. Repeat along the long dimension.

```
short axis (width)      the narrow dimension
long axis (height)      the long dimension
shape                   rect, square, trapezoid, ellipse, or irregular
```

You know these better than any algorithm can infer them, because you trimmed
the block. A single careful measurement beats statistical inference over an
image containing tens of thousands of debris contours.

### Overview magnification

What matters is the section's short axis in pixels, not the objective:

```
below 60 px      too coarse for a stable fit; use higher magnification
100 - 250 px     ideal
above 400 px     more resolution than detection can use
```

The overview only locates sections. Imaging resolution at acquisition is set
by the ATLAS protocol, so extra overview resolution buys nothing while costing
memory and runtime. Pixel count grows as the square of magnification: for a
4 inch wafer with sections a few hundred microns across, 2.5x is normally
right.

## 2.  DETECT SECTIONS

```bash
python detect_rectangles.py \
    --image_path [overview image] \
    --section-width [short axis] --section-height [long axis]
```

Dimensions are in pixels. To work in microns instead, add --um-per-px, which
switches --section-width, --section-height and --inflate-margin to microns at
once:

```bash
python detect_rectangles.py \
    --image_path [overview image] \
    --section-width [short axis um] --section-height [long axis um] \
    --um-per-px [image scale]
```

The script echoes what it resolved, so you can confirm the interpretation:

```
Section size: 1343 x 4400 um @ 7.14725 um/px -> 187.9 x 615.6 px
```

Outputs:

```
[image]_[timestamp].magc         beside the overview; open this in magfinder
stats/
    debug_[image]_[ts].tiff      4-panel diagnostic
    contours_[image]_[ts].xlsx   per-contour measurements
```

Diagnostics go in stats/ deliberately. magfinder asks for the folder holding
the .magc and then looks there for the overview image; if the debug image sits
beside it, magfinder loads that instead of the real overview.

Always inspect panel A of the debug TIFF, which draws every placed ROI in
green over the original. That is the real check.

### Section shape

Default is a rectangle. If your sections are not rectangular:

```
--section-shape trapezoid --taper 0.6     top edge 60% of the bottom
--section-shape ellipse
--section-shape square
--section-shape hull                      irregular; adapts per section
--section-shape custom --shape-file [file]
```

A custom shape file is a list of "x y" vertices, one per line; see
example_shape.txt. It is recentred and rescaled to the requested
width x height, so the magnification you traced at does not matter.

Use a parametric shape when every section is nominally the same known shape:
the template is rigid, which regularises the fit and makes results
reproducible. Use hull when sections are irregular or vary.

## 3.  TUNING FOR YIELD

A first run at the defaults rarely captures most sections. Expect to tune.
Four parameters matter, in order of impact.

### 3.1  --threshold          the largest single lever

Sections darker than --threshold enter the binary mask; anything lighter is
invisible to every later stage, so no other setting can recover it. The
default of 86 is often far too low for variably stained sections.

Diagnostic: compare the sections that got ROIs against those that did not in
the debug TIFF. If the missed ones are systematically lighter, raise it. Work
upward (86 -> 130 -> 160) until the missed sections no longer sort by
brightness, which is the sign that thresholding is no longer the constraint.

### 3.2  section dimensions, undersized about 10%

The least obvious change, and the second largest. Placement is greedy and
forbids overlap, so on a densely packed wafer a full-size template makes
neighbours collide and each accepted ROI blocks the next.

Entering roughly 90% of the true section size stops the collisions. Section 4
explains how to get the full size back for acquisition without losing this.

Diagnostic: a high "dropped, overlapped" count in the yield accounting.

### 3.3  --search-radius

Default 5 px, tuned for small sections. For sections several hundred px long,
15 costs almost nothing and gives the optimiser room to translate away from an
imperfect contour centroid.

### 3.4  --area-tolerance

The area band is derived from the section size you entered, so --min-area and
--max-area are optional. Widen the band when neighbouring sections merge into
one oversized contour and both are lost:

```
--area-tolerance 0.4 2.5      more permissive
--area-tolerance 0.7 1.15     stricter
```

### Reading the yield accounting

Every run prints where candidates were lost:

```
  contours found              95546
  rejected by area band       95306   (outside 57904 - 150550 px^2)
      too small                   4   <- fragmenting, or size too large
      too large                 231   <- sections merging together
  in band                       240
  optimised                     240
  dropped, no fit                 0
  dropped, off image              0
  dropped, overlapped            85   <- template may be too large
  PLACED                        155
```

Each line maps to a different flag:

```
contours found far below the number you know are on the wafer
    Sections are not becoming contours at all. Upstream of everything
    else, so fix it first: --threshold.

rejected, too large
    Neighbours merging into one contour. Widen the band first:
    --area-tolerance 0.4 2.5. An ROI on a merged blob still lands on one
    of the two sections, which beats losing both. More aggressive edge
    subtraction (--edge-threshold 12, --edge-dilate 4) is worth trying
    second, but only helps when there is a visible gap between them.

rejected, too small
    Sections split into fragments, or the entered size is too large.
    Edge subtraction is the usual cause: it amplifies internal tissue
    texture and carves sections apart. Make it gentler:
    --edge-threshold 60, --edge-dilate 1, --edge-dilate 0, or --close 8.
    This also produces ROIs that sit only partly over a section, because
    a fragment's centroid is offset from the section's.

dropped, overlapped
    Template too large for the packing density. Shrink about 10%.

optimised capped by --num-rects
    Raise it; the default is 5000.
```

### A worked example

One 4 inch wafer, roughly 1000 osmium-stained sections, imaged at 2.5x.
Sections measured 1343 x 4400 um at 7.14725 um/px; overview 14434 x 14497 px.

```
run   change                                        coverage
---   -------------------------------------------   --------
 1    defaults (--threshold 86, true size)          very low
 2    --threshold 130                                    65%
 3    + --section-width/height at 90% of true            87%
 4    + --threshold 160                                  90%
```

Final:

```bash
python detect_rectangles.py \
    --image_path [overview image] \
    --section-width 1200 --section-height 3960 \
    --um-per-px 7.14725 \
    --threshold 160 --search-radius 15
```

### Knowing when to stop

90% was the practical stopping point. The remaining misses no longer shared a
cause: torn sections, folds, debris-adjacent sections, tight clusters. Several
were damaged in ways where no rectangle would be correct.

magfinder exists for exactly this. Past roughly 90%, pushing the threshold
higher starts producing ROIs on scratches and dust, and a false positive costs
more than a miss: it sends the scope to image nothing. If a run's newly gained
sections are outnumbered by junk ROIs, back off and place the rest by hand.

## 4.  INFLATING ROIs FOR ACQUISITION

Registering the optical map to the SEM stage is never exact, and an ROI sized
to the section will then clip it. Reimaging afterwards costs far more than a
slightly generous ROI.

```
--inflate-margin [x] [y]        per-side margin, one or two values
--inflate-percent 10            proportional instead
```

With --um-per-px the margins are microns, which is the natural unit since
registration error is a physical distance. Absolute margins are usually better
than a percentage: a percentage gives the short axis less absolute margin than
the long one, while registration error is the same in both directions.

Inflation is applied AFTER the overlap filter, so it does not reduce detection
yield. That is what lets you keep an undersized detection template (3.2) and
still record a generous ROI.

Sizing the margin: it must cover the registration error AND any deliberate
undersizing.

```
margin per side = (true size - template size) / 2 + registration error
```

For the worked example, with a true section of 1343 x 4400 um, a template of
1200 x 3960, and registration error up to 100 um:

```
short axis   (1343 - 1200)/2 + 100 = 171.5 um per side
long  axis   (4400 - 3960)/2 + 100 = 320.0 um per side

--inflate-margin 172 320
```

The two axes differ, which is why per-axis values are supported.

The run reports how many ROIs now overlap and what fraction of total ROI area
is doubly covered. Some overlap is the accepted cost of not clipping sections,
but it should be a deliberate choice. In the debug TIFF, green is the detected
ROI and magenta is the inflated one that gets written to the .magc.

## 5.  MANUAL PASS IN MAGFINDER

Open the .magc in magfinder to add sections the detector missed. Point it at
the folder holding the .magc; it will find the overview image there.

## 6.  BUILD THE ATLAS PROJECT

```bash
python generate_and_fuse.py \
    --magc-file [magc file] \
    --scale [um per px] \
    --flip_y --center-on-wafer \
    --tif-background [overview image] \
    --no-embed-image \
    --output [project].a5proj
```

Then validate before taking it to the scope:

```bash
python check_a5proj.py [project].a5proj
```

All lines should read PASS, particularly "protocol references resolve" and
"geometry type is Polygon".

### Key options

```
--scale N            microns per pixel. THE most important value here:
                     it converts pixel positions to stage coordinates.
                     An error puts every section off by the same factor,
                     and it will not be obvious until the stage drives
                     somewhere wrong. Determine it from a known distance:
                     scale = um across / px across.
--flip_y             image Y runs downward, stage Y usually upward.
                     Symptom of needing it: sections appear mirrored
                     top-to-bottom relative to the real wafer.
--center-on-wafer    put the wafer at stage (0,0), which is where ATLAS
                     places the holder. Without it the ROIs sit offset by
                     half the image in each axis. Requires
                     --tif-background for the image dimensions.
--no-embed-image     do not write the overview into the project; import
                     it manually in ATLAS. See section 7.
--tif-background F   the overview image. With --no-embed-image only its
                     pixel dimensions are read, never the image itself,
                     so file size does not matter here.
--template PATH      ATLAS template. Default
                     atlas_template/blank_template.a5proj
--no-optimize        skip nearest-neighbour stage path ordering
```

Do not use --no-protocol for a real run. It builds a project with no imaging
protocol, whose sections reference protocols that do not exist. check_a5proj.py
flags this.

### The ATLAS template

generate_and_fuse.py needs atlas_template/blank_template.a5proj. It is
included because it cannot be regenerated from this repository alone:
create_blank_template.py works by stripping an existing real project down to a
reusable skeleton, so a fresh clone has nothing to strip.

The generators take from it the project wrapper and its four top-level
siblings, the carrier element, and the Protocol from the first section, which
is copied onto every generated section so all sections inherit the same
imaging settings. The shipped template's protocols are structurally complete
but have blank values, and it carries no scope configuration or file paths.

To regenerate it for a different scope:

```bash
python create_blank_template.py [real project].a5proj \
    atlas_template/blank_template.a5proj
```

Whatever you use as the source, the first section must still carry an
AcquisitionSpec/Protocol, because that is the element that gets copied.

### Two things that must change together

Change objective and BOTH of these change, or results are silently wrong:

  1. --section-width / --section-height for detection
  2. --scale for generate_and_fuse.py

--min-area and --max-area also scale, but as the square of magnification.

## 7.  LOADING IN ATLAS

The overview image is imported by hand. generate_and_fuse.py can embed it, and
the XML it writes matches a natively generated ATLAS project element for
element, but ATLAS still fails to build an image pyramid for it and shows a
placeholder box reading "GENERATING IMAGE PYRAMID". Importing through the
ATLAS UI works, so that is the documented route. Use --no-embed-image.

  1. Open the .a5proj. The ROIs appear, centred on the wafer holder, not yet

```
 tied to a stage position.
```

  2. Import the overview image through the ATLAS UI. tif, png, or jpg all

```
 work.

 If ATLAS shows "GENERATING IMAGE PYRAMID" and never resolves, the image
 is too large for it. A full-resolution wafer scan can be 200+ megapixels;
 a natively imported ATLAS overview is nearer 25. Downsample and import
 the reduced copy:

     sips -Z 3624 [overview].tif --out [overview]_small.jpg     (macOS)

 This affects only what you hand ATLAS. The project itself is generated
 from the full-resolution overview and does not change.
```

  3. Set the image size. Enter the physical size of the IMAGE, not the wafer.

```
 These differ because the image frame is larger than the wafer:

     image   14434 px x 7.14725 um/px  = 103163 um wide
     wafer   4 inch                    = 101600 um

 The wafer fills about 98.5% of the image width, so entering the wafer
 diameter where ATLAS wants the image width introduces a 1.5% scale error,
 roughly 1.5 mm across the wafer, which is enough to push edge sections
 out of their ROIs. Use the image width unless the dialog explicitly asks
 for wafer diameter and fits to the detected circle.

 For a downsampled overview the physical size is unchanged; only the pixel
 count differs.
```

  4. The overview should now sit on the ROIs, both centred on stage (0,0).

  5. Right-click the overview and choose Align for fine adjustment if needed.

  6. Register to the stage: take two or three SEM reference images at the

```
 wafer corners and pin them to the optical image.
```

Residual error after step 6 is expected, and is why ROIs should be inflated at
detection time (section 4).

## NOTES ON THE CODE

optimize_rectangle_for_contour() in detect_rectangles.py is not called.
save_debug_image() uses the parallel _optimize_worker() path instead. The
serial function is kept as the readable reference implementation and is
patched in step with the worker. If you change the cost function, change both.

Geometry is written as an arbitrary-N polygon throughout, so non-rectangular
shapes survive the whole pipeline.

Orientation comes from cv2.minAreaRect, which determines a contour's angle
only modulo 180 degrees. For symmetric templates that is harmless; for
asymmetric ones the optimiser starts from both orientations and keeps the
better fit.

Peak memory for a 14434 x 14497 overview with 186 x 596 px sections is roughly
1.8 GB at 150 sections and 4.6 GB at 1000. --num-workers does not reduce this:
all optimiser patches are built before the worker pool starts, so patch memory
is independent of worker count.

## LICENSE

MIT. See LICENSE.
