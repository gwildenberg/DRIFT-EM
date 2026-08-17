#!/usr/bin/env python3
import random
import argparse
import math
import copy
import numpy as np
from lxml import etree
from PIL import Image, ImageDraw
import networkx as nx
import os


def parse_magc_file(file_path):
    """Parse the MAGC file to extract polygon coordinates."""
    sections = []
    current_section = None

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('[section.'):
                if current_section:
                    sections.append(current_section)
                current_section = {'vertices': []}
            elif line.startswith('polygon = ') and current_section is not None:
                coords_str = line.replace('polygon = ', '')
                coords = [float(c) for c in coords_str.split(',')]
                current_section['vertices'] = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]

    if current_section:
        sections.append(current_section)

    return sections


def calculate_centroid(vertices):
    """Calculate the centroid of a polygon."""
    x_coords = [x for x, y in vertices]
    y_coords = [y for x, y in vertices]
    return (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))


def optimize_path(sections_data, method="simulated_annealing", time_limit=10):
    """
    Optimize the order of sections to minimize total stage travel distance.

    Methods: tsp, nearest_neighbor, simulated_annealing, two_opt
    """
    import time

    if len(sections_data) <= 1:
        return sections_data

    centroids = [calculate_centroid(s['vertices']) for s in sections_data]
    n = len(centroids)

    distances = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                x1, y1 = centroids[i]
                x2, y2 = centroids[j]
                distances[(i, j)] = math.sqrt((x2-x1)**2 + (y2-y1)**2)

    def path_distance(path):
        return sum(distances.get((path[i], path[i+1]), 0) for i in range(len(path)-1))

    def nearest_neighbor_path(start=0):
        unvisited = set(range(n))
        current = start
        path = [current]
        unvisited.remove(current)
        while unvisited:
            nearest = min(unvisited, key=lambda x: distances.get((current, x), float('inf')))
            path.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        return path

    if method == "tsp":
        G = nx.complete_graph(n)
        for i, j in G.edges():
            G[i][j]['weight'] = distances.get((i, j), 0)
        tsp_path = nx.approximation.traveling_salesman_problem(G, cycle=False)
        return [sections_data[i] for i in tsp_path]

    elif method == "nearest_neighbor":
        return [sections_data[i] for i in nearest_neighbor_path()]

    elif method == "simulated_annealing":
        print("Generating initial nearest neighbor path for simulated annealing...")
        current_path = nearest_neighbor_path()
        current_distance = path_distance(current_path)
        best_path = current_path.copy()
        best_distance = current_distance

        temp = 100.0
        cooling_rate = 0.995
        start_time = time.time()
        iterations = 0

        print(f"Starting simulated annealing optimization from nearest neighbor path (max {time_limit} seconds)...")
        while time.time() - start_time < time_limit:
            iterations += 1
            new_path = current_path.copy()
            i, j = sorted(random.sample(range(n), 2))
            new_path[i:j+1] = reversed(new_path[i:j+1])
            new_distance = path_distance(new_path)

            if new_distance < current_distance:
                current_path = new_path
                current_distance = new_distance
                if new_distance < best_distance:
                    best_path = new_path.copy()
                    best_distance = new_distance
            else:
                delta = new_distance - current_distance
                if random.random() < math.exp(-delta / temp):
                    current_path = new_path
                    current_distance = new_distance

            temp *= cooling_rate
            if temp < 0.1:
                temp = 100.0

        elapsed = time.time() - start_time
        print(f"Completed {iterations} iterations in {elapsed:.2f} seconds")
        print(f"Best path distance: {best_distance:.2f}")
        return [sections_data[i] for i in best_path]

    elif method == "two_opt":
        path = nearest_neighbor_path()
        best_distance = path_distance(path)
        start_time = time.time()
        iterations = 0

        print(f"Starting 2-opt optimization (max {time_limit} seconds)...")
        improvement = True
        while improvement and time.time() - start_time < time_limit:
            improvement = False
            for i in range(1, n-2):
                for j in range(i+1, n-1):
                    before = distances.get((path[i-1], path[i]), 0) + distances.get((path[j], path[j+1]), 0)
                    after = distances.get((path[i-1], path[j]), 0) + distances.get((path[i], path[j+1]), 0)
                    if after < before:
                        path[i:j+1] = reversed(path[i:j+1])
                        improvement = True
                        best_distance -= (before - after)
                        break
                if improvement:
                    break
                iterations += 1
                if iterations % 100 == 0 and time.time() - start_time >= time_limit:
                    break

        elapsed = time.time() - start_time
        print(f"Completed 2-opt optimization in {elapsed:.2f} seconds")
        print(f"Best path distance: {best_distance:.2f}")
        return [sections_data[i] for i in path]

    else:
        print(f"Unknown method: {method}, falling back to nearest_neighbor")
        return optimize_path(sections_data, "nearest_neighbor", time_limit)


def create_tif_overlay(sections_data, tif_path=None, output_path=None,
                       line_color=(255, 0, 0), line_width=3):
    """Create a PNG overlay showing ROI polygons and acquisition path sequence."""
    import datetime

    if tif_path and os.path.exists(tif_path):
        Image.MAX_IMAGE_PIXELS = None
        img = Image.open(tif_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
    else:
        if tif_path:
            print(f"Warning: Background file {tif_path} not found, using blank canvas.")
        img = Image.new('RGB', (5000, 5000), (255, 255, 255))

    draw = ImageDraw.Draw(img)
    centroids = [calculate_centroid(s['vertices']) for s in sections_data]

    for i, section in enumerate(sections_data):
        poly_points = [coord for x, y in section['vertices'] for coord in (x, y)]
        draw.polygon(poly_points, outline=(0, 255, 0), width=2)

        cx, cy = centroids[i]
        radius = 10
        draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius),
                     fill=(0, 0, 255), outline=(255, 255, 255))
        draw.text((cx+radius+5, cy-10), str(i+1), fill=(255, 255, 255))

    for i in range(len(centroids)-1):
        x1, y1 = centroids[i]
        x2, y2 = centroids[i+1]
        draw.line([(x1, y1), (x2, y2)], fill=line_color, width=line_width)

        mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
        dir_x, dir_y = x2-x1, y2-y1
        length = math.sqrt(dir_x**2 + dir_y**2)
        if length > 0:
            dir_x, dir_y = dir_x/length*20, dir_y/length*20
            draw.polygon([
                (mid_x, mid_y),
                (mid_x - dir_y - dir_x/2, mid_y + dir_x - dir_y/2),
                (mid_x + dir_y - dir_x/2, mid_y - dir_x - dir_y/2)
            ], fill=line_color)

    if not output_path:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"path_overlay_{timestamp}.png"

    img.save(output_path)
    print(f"Generated overlay image: {output_path}")
    return output_path


def _build_minimal_tree():
    """
    Build a minimal valid ATLAS F-BioSEM-Project XML tree from scratch.

    Used by --no-protocol so the script requires no template file at all.
    Contains all structural elements ATLAS needs (carrier, region sets, data
    sets, load-out) but no sessions, sections, protocols, or imported images.
    Users will need to assign an imaging Protocol in ATLAS before acquiring.
    """
    XSI = "http://www.w3.org/2001/XMLSchema-instance"

    loadout_uid  = random.randint(100000000, 2147483647)
    carrier_uid  = random.randint(100000000, 2147483647)
    slot_uid     = random.randint(100000000, 2147483647)
    rs_uid       = random.randint(100000000, 2147483647)
    brs_3d_uid   = random.randint(100000000, 2147483647)
    dbrs_uid     = random.randint(100000000, 2147483647)
    brs_ann_uid  = random.randint(100000000, 2147483647)
    brs_imp_uid  = random.randint(100000000, 2147483647)
    ds_imp_uid   = random.randint(100000000, 2147483647)
    brs_anal_uid = random.randint(100000000, 2147483647)
    fps_uid      = random.randint(100000000, 2147483647)
    sm_uid       = random.randint(100000000, 2147483647)
    ah_uid       = random.randint(100000000, 2147483647)
    proj_uid     = random.randint(100000000, 2147483647)

    root = etree.Element("F-BioSEM-Project", attrib={
        "Version": "2.10",
        f"{{{XSI}}}noNamespaceSchemaLocation":
            "http://fibics.com/xsd/FibicsAtlasProject_2_10.xsd",
    }, nsmap={"xsi": XSI})

    biosem = etree.SubElement(root, "BioSemProject")

    def _s(parent, tag, text=None):
        e = etree.SubElement(parent, tag)
        if text is not None:
            e.text = str(text)
        return e

    def _ipt(parent, tag="ParentTransform"):
        pt = etree.SubElement(parent, tag)
        for k, v in [("M11","1"),("M12","0"),("M13","0"),("M14","0"),
                     ("M21","0"),("M22","1"),("M23","0"),("M24","0"),
                     ("M31","0"),("M32","0"),("M33","1"),("M34","0"),
                     ("M41","0"),("M42","0"),("M43","0"),("M44","1"),
                     ("CenterLocalX","0"),("CenterLocalY","0")]:
            _s(pt, k, v)
        return pt

    def _da(parent):
        _ipt(etree.SubElement(parent, "DefaultAlignment"))

    # BioSemProject metadata (Name/DataFolder/XMLFile filled by generate_a5proj)
    _s(biosem, "LastUpdated", "2024-01-01T00:00:00.000+00:00")
    _s(biosem, "Name", "")
    _s(biosem, "UID", proj_uid)
    _ipt(biosem)
    _s(biosem, "IsVisible", "true")
    _da(biosem)
    _s(biosem, "DefaultCarrier", "0")
    _s(biosem, "ActiveLoadOut", loadout_uid)
    _s(biosem, "ExportIdMap", "false")
    _s(biosem, "AutoHist", "false")
    _s(biosem, "DataFolder", "")
    _s(biosem, "XMLFile", "")
    _s(biosem, "Type", "0")
    _s(biosem, "CreatedCarrierCount", "1")
    _s(biosem, "CreatedCrossSectionCount", "0")
    _s(biosem, "CreatedLamellaCount", "0")
    _s(biosem, "CreatedFastLamellaCount", "0")
    _s(biosem, "CreatedUserSectionSetCount", "0")
    _s(biosem, "CreatedRegionCount", "0")
    _s(biosem, "CreatedGrabsCount", "0")
    _s(biosem, "CreatedAnalyticsRegionCount", "0")
    _s(biosem, "CreatedRegionGroupCount", "0")

    # AtlasCarrier — UID cross-references must stay consistent
    carrier = _s(biosem, "AtlasCarrier")
    _s(carrier, "Name", "4 Inch Wafer 1")
    _s(carrier, "UID", carrier_uid)
    _ipt(carrier)
    _s(carrier, "IsVisible", "true")
    _da(carrier)
    _s(carrier, "UserRegionSet", rs_uid)
    _s(carrier, "Regions3D", brs_3d_uid)
    _s(carrier, "AnalyticsRegions", brs_anal_uid)
    _s(carrier, "DualBeamRegions", dbrs_uid)
    _s(carrier, "Annotations", brs_ann_uid)
    _s(carrier, "ImportedRegionSet", brs_imp_uid)
    _s(carrier, "ImportedDataSet", ds_imp_uid)
    _s(carrier, "ImportedFeaturePointSet", fps_uid)
    _s(carrier, "ActiveSession", "-1")
    _s(carrier, "Type", "1004")
    _s(carrier, "CreatedRegionSetCount", "4")
    _s(carrier, "CreatedSiteSetCount", "0")
    _s(carrier, "SimpleName", "4 Inch Wafer")
    _s(carrier, "CreatedIndex", "1")

    # RegionSet "2D Regions"
    rs = _s(carrier, "RegionSet")
    _s(rs, "Name", "2D Regions")
    _s(rs, "UID", rs_uid)
    _ipt(rs)
    _s(rs, "IsVisible", "true")
    _da(rs)
    _s(rs, "CreatedRegionCount", "0")
    _s(rs, "ChildNameRoot", "Region")

    # BasicRegionSet "3D Regions"
    b3d = _s(carrier, "BasicRegionSet")
    _s(b3d, "Name", "3D Regions")
    _s(b3d, "UID", brs_3d_uid)
    _ipt(b3d)
    _s(b3d, "IsVisible", "true")
    _da(b3d)

    # DualBeamRegionSet
    dbrs = _s(carrier, "DualBeamRegionSet")
    _s(dbrs, "Name", "FIBSEM Site Set")
    _s(dbrs, "UID", dbrs_uid)
    _ipt(dbrs)
    _s(dbrs, "IsVisible", "true")
    _da(dbrs)
    _s(dbrs, "CreatedBasicRegionCount", "0")

    # BasicRegionSet "Annotations"
    bann = _s(carrier, "BasicRegionSet")
    _s(bann, "Name", "Annotations")
    _s(bann, "UID", brs_ann_uid)
    _ipt(bann)
    _s(bann, "IsVisible", "true")
    _da(bann)

    # BasicRegionSet "Imported Regions"  ← needed by _update_optical_image
    bimp = _s(carrier, "BasicRegionSet")
    _s(bimp, "Name", "Imported Regions")
    _s(bimp, "UID", brs_imp_uid)
    _ipt(bimp)
    _s(bimp, "IsVisible", "true")
    _da(bimp)

    # DataSet "Imported Data"  ← needed by _update_optical_image
    dimp = _s(carrier, "DataSet")
    _s(dimp, "Name", "Imported Data")
    _s(dimp, "UID", ds_imp_uid)
    _ipt(dimp)
    _s(dimp, "IsVisible", "true")
    _da(dimp)
    _s(dimp, "ViewRotation", "0")
    _s(dimp, "AcquisitionSpecUID", "-1")
    _s(dimp, "FileName")
    _s(dimp, "ImagingTime", "0")

    # BasicRegionSet "Analytics Regions"
    banal = _s(carrier, "BasicRegionSet")
    _s(banal, "Name", "Analytics Regions")
    _s(banal, "UID", brs_anal_uid)
    _ipt(banal)
    _s(banal, "IsVisible", "true")
    _da(banal)

    # FeaturePointSet "Non-Image Data"
    fps = _s(carrier, "FeaturePointSet")
    _s(fps, "Name", "Non-Image Data")
    _s(fps, "UID", fps_uid)
    _ipt(fps)
    _s(fps, "IsVisible", "true")
    _da(fps)

    # LoadOut — UID must match ActiveLoadOut above
    lo = _s(biosem, "LoadOut")
    _s(lo, "Name", "LoadOut")
    _s(lo, "UID", loadout_uid)
    _s(lo, "HolderType", "1005")
    sp = _s(lo, "SlotPos")
    pos = _s(sp, "Position")
    _s(pos, "X", "0")
    _s(pos, "Y", "0")
    _s(sp, "Rotation", "0")
    _s(sp, "SlotIdx", "0")
    ma = _s(lo, "MicroscopeAdjust")
    _s(ma, "X", "0")
    _s(ma, "Y", "0")
    _s(ma, "R", "0")
    sl = _s(lo, "Slot")
    _s(sl, "Name", "Slot")
    _s(sl, "UID", slot_uid)
    _s(sl, "Index", "0")
    _s(sl, "CarrierID", carrier_uid)

    # StringMap and AutomationHistory
    sm = _s(biosem, "StringMap")
    _s(sm, "Name", "ExportIdMap")
    _s(sm, "UID", sm_uid)
    ah = _s(biosem, "AutomationHistory")
    _s(ah, "Name", "AutomationHistory")
    _s(ah, "UID", ah_uid)

    return etree.ElementTree(root)


def _identity_transform_elem(parent, tag):
    """Append an identity ParentTransform element to parent."""
    pt = etree.SubElement(parent, tag)
    for k, v in [("M11","1"),("M12","0"),("M13","0"),("M14","0"),
                 ("M21","0"),("M22","1"),("M23","0"),("M24","0"),
                 ("M31","0"),("M32","0"),("M33","1"),("M34","0"),
                 ("M41","0"),("M42","0"),("M43","0"),("M44","1"),
                 ("CenterLocalX","0"),("CenterLocalY","0")]:
        etree.SubElement(pt, k).text = v
    return pt


def _ensure_biosem_session(carrier):
    """
    ATLAS will not display sections unless the carrier has at least one
    BioSemSession and ActiveSession points to it.  Templates built from
    scratch (blank_template, --no-protocol) have neither; this function
    adds a minimal stub so blank_template works identically to a real
    session-bearing template.
    """
    if carrier.findall('BioSemSession'):
        return  # template already has session(s) — nothing to do

    sess_uid = random.randint(100000000, 2147483647)
    sess = etree.Element('BioSemSession')
    etree.SubElement(sess, 'Name').text = 'Session 1'
    etree.SubElement(sess, 'UID').text = str(sess_uid)
    _identity_transform_elem(sess, 'ParentTransform')
    etree.SubElement(sess, 'IsVisible').text = 'true'
    da = etree.SubElement(sess, 'DefaultAlignment')
    _identity_transform_elem(da, 'ParentTransform')
    etree.SubElement(sess, 'SessionType').text = 'Unknown'
    etree.SubElement(sess, 'InstrumentID').text = '-1'
    etree.SubElement(sess, 'SingleImages').text = '-1'
    etree.SubElement(sess, 'Atlas3DDataUID').text = '-1'
    etree.SubElement(sess, 'BeamType').text = 'Unknown'
    so = etree.SubElement(sess, 'SlotOffset')
    _identity_transform_elem(so, 'ParentTransform')
    ma = etree.SubElement(sess, 'MicroscopeAdjust')
    etree.SubElement(ma, 'X').text = '0'
    etree.SubElement(ma, 'Y').text = '0'
    etree.SubElement(ma, 'R').text = '0'

    # Insert immediately before the SectionSet so the XML ordering matches
    # real ATLAS files (BioSemSession precedes SectionSet in the carrier).
    ss = carrier.find('SectionSet')
    if ss is not None:
        carrier.insert(list(carrier).index(ss), sess)
    else:
        carrier.append(sess)

    active = carrier.find('ActiveSession')
    if active is not None:
        active.text = str(sess_uid)

    print(f"Note: added stub BioSemSession (required for ATLAS to display sections).")


def _image_transform_elem(parent, tag, m11, m22, m41, m42):
    """Append a scaled/translated ParentTransform for a PlaceableImage (CenterLocal=0.5)."""
    pt = etree.SubElement(parent, tag)
    for k, v in [("M11", f"{m11:.6f}"), ("M12", "0"), ("M13", "0"), ("M14", "0"),
                 ("M21", "0"), ("M22", f"{m22:.6f}"), ("M23", "0"), ("M24", "0"),
                 ("M31", "0"), ("M32", "0"), ("M33", "1"), ("M34", "0"),
                 ("M41", f"{m41:.6f}"), ("M42", f"{m42:.6f}"), ("M43", "0"), ("M44", "1"),
                 ("CenterLocalX", "0.5"), ("CenterLocalY", "0.5")]:
        etree.SubElement(pt, k).text = v
    return pt


def _update_optical_image(carrier, jpg_path, scale, flip_x, flip_y, shift_x, shift_y,
                          image_path_override=None, relative=False,
                          image_scale=None, image_flip_sign=True):
    """
    Register the optical overview image in the project's Imported Data DataSet
    and Imported Regions BasicRegionSet so ATLAS loads it as a background layer.

    Uses the same coordinate transform as sections so the image automatically
    aligns with the ROI outlines.

    ATLAS resolves FileName as an absolute path.  image_path_override lets the
    caller supply the exact Windows path (e.g. 'C:\\SEM\\wafer01.jpg').
    Without it, the absolute Mac path is stored; ATLAS will prompt once on
    Windows to relocate the file.
    """
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(jpg_path) as img:
        img_w, img_h = img.size

    # The overview may be a downsampled copy of the image the .magc refers to.
    # ATLAS has to build an image pyramid, and a full-resolution wafer scan can
    # be too large for it to process, so a reduced copy is often necessary. The
    # sections still use `scale` because their coordinates are in original
    # pixels; the background uses its own scale so it still covers the same
    # physical area.
    bg_scale = image_scale if image_scale else scale
    width_um  = img_w * bg_scale
    height_um = img_h * bg_scale

    # Image center in stage μm — same pipeline as build_section_element
    cx_px = (-img_w / 2.0) if flip_x else (img_w / 2.0)
    cy_px = (-img_h / 2.0) if flip_y else (img_h / 2.0)
    center_x = cx_px * bg_scale + shift_x
    center_y = cy_px * bg_scale + shift_y

    # Sign of M11/M22 encodes axis orientation; flip_y → negative M22 maps
    # image-Y-down to stage-Y-up, matching how section coords are transformed.
    # Sign of M11/M22 encodes axis orientation. Natively generated ATLAS
    # projects always use POSITIVE values here, even where the section
    # coordinates are flipped, so a negative scale may not be a representation
    # ATLAS handles. image_flip_sign=False writes positive values and relies on
    # the shift alone to position the image.
    if image_flip_sign:
        m11 = -width_um  if flip_x else  width_um
        m22 = -height_um if flip_y else  height_um
    else:
        m11, m22 = abs(width_um), abs(height_um)

    img_name      = os.path.splitext(os.path.basename(jpg_path))[0]
    # ATLAS stores absolute paths; use override (Windows path) when supplied,
    # otherwise the Mac absolute path so ATLAS can prompt once for relocation.
    if image_path_override:
        img_file_path = image_path_override
    elif relative:
        # Bare filename, so ATLAS looks beside the .a5proj. Keeps the project
        # portable between the Mac that generates it and the SEM PC that loads
        # it -- no absolute path to go stale.
        img_file_path = os.path.basename(jpg_path)
    else:
        img_file_path = os.path.abspath(jpg_path)

    image_uid = random.randint(100000000, 2147483647)
    link_uid  = random.randint(100000000, 2147483647)
    lr_uid    = random.randint(100000000, 2147483647)

    # ── DataSet "Imported Data" ───────────────────────────────────────────────
    dataset = next((ds for ds in carrier.findall("DataSet")
                    if ds.findtext("Name") == "Imported Data"), None)
    if dataset is not None:
        pi = dataset.find("PlaceableImage")
        if pi is not None:
            dataset.remove(pi)
        uid_elem = dataset.find("AcquisitionSpecUID")
        if uid_elem is not None:
            uid_elem.text = str(link_uid)

        pi = etree.SubElement(dataset, "PlaceableImage")
        etree.SubElement(pi, "Name").text = img_name
        etree.SubElement(pi, "UID").text = str(image_uid)
        _image_transform_elem(pi, "ParentTransform", m11, m22, center_x, center_y)
        # ATLAS writes IsVisible here, immediately after ParentTransform. Without
        # it the layer exists but is not rendered, which looks like a failure to
        # load the file. Verified against a natively generated project.
        etree.SubElement(pi, "IsVisible").text = "true"
        da = etree.SubElement(pi, "DefaultAlignment")
        _image_transform_elem(da, "ParentTransform", m11, m22, center_x, center_y)
        etree.SubElement(pi, "ViewRotation").text = "0"
        etree.SubElement(pi, "AcquisitionSpecUID").text = str(link_uid)
        etree.SubElement(pi, "FileName").text = img_file_path
        etree.SubElement(pi, "ImagingTime").text = "0"
        etree.SubElement(pi, "ActiveChannelFunction").text = "0"
        cf = etree.SubElement(pi, "ChannelFunctions")
        si = etree.SubElement(cf, "ShaderInstance")
        etree.SubElement(si, "InterfaceUID").text = "1000"
        etree.SubElement(si, "CustomName")
        db = etree.SubElement(si, "DisplayBlock")
        etree.SubElement(db, "Brightness").text = "0"
        etree.SubElement(db, "Contrast").text = "1"
        etree.SubElement(db, "Invert").text = "false"
        d = etree.SubElement(si, "Data")
        etree.SubElement(d, "Kind").text = "ChannelDropdownData"
        etree.SubElement(d, "Selected").text = "0"
        etree.SubElement(pi, "Alpha").text = "255"
        etree.SubElement(pi, "CpxMultiCh").text = "false"
        etree.SubElement(pi, "Width").text = str(img_w)
        etree.SubElement(pi, "Height").text = str(img_h)
        etree.SubElement(pi, "TransPixVal").text = "-1"
        etree.SubElement(pi, "DwellTime").text = "-1"
        etree.SubElement(pi, "LineAvg").text = "-1"
        etree.SubElement(pi, "Timestamp").text = "-1"
        chs = etree.SubElement(pi, "Channels")
        ch  = etree.SubElement(chs, "Channel")
        etree.SubElement(ch, "FileName").text = img_file_path
        etree.SubElement(ch, "DetectorName").text = "Unknown"
        etree.SubElement(ch, "CorrectionInvert").text = "false"
        etree.SubElement(ch, "CorrectionBrightness").text = "0"
        etree.SubElement(ch, "CorrectionContrast").text = "1"
        etree.SubElement(ch, "CorrectionXSlope").text = "0"
        etree.SubElement(ch, "CorrectionYSlope").text = "0"
        etree.SubElement(ch, "CorrectionRSlope").text = "0"
    else:
        print("Warning: 'Imported Data' DataSet not found in template — optical image not embedded.")

    # ── BasicRegionSet "Imported Regions" ────────────────────────────────────
    imported_brs = next((brs for brs in carrier.findall("BasicRegionSet")
                         if brs.findtext("Name") == "Imported Regions"), None)
    if imported_brs is not None:
        for lr in imported_brs.findall("LinkedRegion"):
            imported_brs.remove(lr)

        lr = etree.SubElement(imported_brs, "LinkedRegion")
        etree.SubElement(lr, "CreatedIndex").text = "0"
        etree.SubElement(lr, "CustomName").text = "false"
        etree.SubElement(lr, "Name").text = img_name
        etree.SubElement(lr, "UID").text = str(lr_uid)
        _image_transform_elem(lr, "ParentTransform", m11, m22, center_x, center_y)
        etree.SubElement(lr, "IsVisible").text = "true"
        da = etree.SubElement(lr, "DefaultAlignment")
        _image_transform_elem(da, "ParentTransform", m11, m22, center_x, center_y)
        etree.SubElement(lr, "MinZ").text = "0"
        etree.SubElement(lr, "MaxZ").text = "0"
        geom = etree.SubElement(lr, "Geometry")
        etree.SubElement(geom, "Type").text = "Polygon"
        for vx, vy in [(0, 0), (1, 0), (1, 1), (0, 1)]:
            v = etree.SubElement(geom, "Vertex")
            etree.SubElement(v, "X").text = str(vx)
            etree.SubElement(v, "Y").text = str(vy)
        rc = etree.SubElement(lr, "RotationCentre")
        etree.SubElement(rc, "X").text = "0"
        etree.SubElement(rc, "Y").text = "0"
        etree.SubElement(lr, "LinkUID").text = str(link_uid)
        etree.SubElement(lr, "GeometryAcquired").text = "false"
        al = etree.SubElement(lr, "AcquisitionLink")
        etree.SubElement(al, "Name").text = "Acquisition Link"
        etree.SubElement(al, "UID").text = str(link_uid)
        etree.SubElement(al, "PlaceableDataUID").text = str(image_uid)
        etree.SubElement(al, "AcquisitionTypeEnum").text = "ForImport"
        etree.SubElement(al, "AcquisitionDataTypeEnum").text = "2DData"
        etree.SubElement(al, "AcquisitionStateEnum").text = "Acquired"
        etree.SubElement(al, "AcquisitionMode").text = "Undefined"
    else:
        print("Warning: 'Imported Regions' BasicRegionSet not found in template — optical image not embedded.")

    # Register the image in LayerFolder as well.
    #
    # Embedding a PlaceableImage in the "Imported Data" DataSet makes the image
    # part of the project, but LayerFolder is ATLAS's registry of what to
    # actually display. Without an entry here the image is present in the file
    # and never drawn, which looks exactly like ATLAS "failing to find" it --
    # the geometry is correct but nothing appears, and you end up importing the
    # overview by hand.
    root = carrier.getroottree().getroot()
    layer_folder = root.find("LayerFolder")
    if layer_folder is not None:
        existing = {e.text for e in layer_folder.iter("PlaceableRef")}
        if str(image_uid) not in existing:
            sub = etree.SubElement(layer_folder, "LayerFolder")
            etree.SubElement(sub, "Name").text = img_name
            etree.SubElement(sub, "UID").text = str(random.randint(100000000, 2147483647))
            etree.SubElement(sub, "PlaceableRef").text = str(image_uid)
            print(f"  Registered as a display layer in LayerFolder.")
    else:
        print("  Warning: no LayerFolder in the template — ATLAS may not display "
              "the image even though it is embedded.")

    print(f"Optical image registered: {img_file_path} "
          f"({img_w}×{img_h} px = {width_um:.0f}×{abs(height_um):.0f} μm, "
          f"center ({center_x:.0f}, {center_y:.0f}) μm)")
    if image_scale:
        print(f"  Background scale {bg_scale:g} um/px (sections use {scale:g})")
    if img_w * img_h > 80e6:
        print(f"  Note: {img_w * img_h / 1e6:.0f} megapixels. ATLAS may fail to build "
              f"an image pyramid this large.")
        print(f"  If it shows a placeholder box instead of the image, supply a "
              f"downsampled overview and")
        print(f"  set --image-scale to match (halving the image doubles the value).")
    if relative:
        print(f"  Stored as a bare filename. Copy {os.path.basename(jpg_path)} into "
              f"the same folder as the .a5proj.")
        print("  If ATLAS still prompts to locate it, it needs an absolute path — "
              "use --image-path instead.")
    elif not image_path_override:
        print("  Note: stored as an absolute path on this machine. On Windows, ATLAS "
              "will prompt once to relocate — point it to the image beside the "
              ".a5proj. Use --image-relative, or --image-path for the Windows path.")


def build_section_element(index, magc_vertices, scale, shift_x, shift_y,
                          rotation_deg, flip_x, flip_y, protocol_template, section_set_name):
    """
    Build a complete ATLAS Section XML element from MAGC polygon data.

    Coordinate flow:
      pixel coords  ×scale→  stage μm  +shift→  final stage position
      flip_y negates Y before scaling (image Y↓ vs stage Y↑ correction)
      Local geometry = centered vertices in stage μm (mean subtracted)
      M41/M42 = centroid in stage μm (the section's stage position)
    """
    rotation_rad = math.radians(rotation_deg)

    # Transform MAGC vertices → stage μm
    transformed = []
    for x, y in magc_vertices:
        if flip_x:
            x = -x
        if flip_y:
            y = -y
        sx, sy = x * scale, y * scale
        rx = sx * math.cos(rotation_rad) - sy * math.sin(rotation_rad)
        ry = sx * math.sin(rotation_rad) + sy * math.cos(rotation_rad)
        transformed.append((rx + shift_x, ry + shift_y))

    # Section centroid = stage position
    cx = sum(v[0] for v in transformed) / len(transformed)
    cy = sum(v[1] for v in transformed) / len(transformed)

    # Local (centered) vertices in μm
    local_verts = [(x - cx, y - cy) for x, y in transformed]

    section_uid = random.randint(100000000, 2147483647)
    link_uid = random.randint(100000000, 2147483647)
    protocol_uid = random.randint(100000000, 2147483647)

    section = etree.Element("Section")
    etree.SubElement(section, "CreatedIndex").text = str(index + 1)
    etree.SubElement(section, "CustomName").text = "false"
    etree.SubElement(section, "Name").text = f"{section_set_name} - {index + 1}"
    etree.SubElement(section, "UID").text = str(section_uid)

    # ParentTransform: identity rotation, centroid as translation
    pt = etree.SubElement(section, "ParentTransform")
    for k, v in [("M11","1"),("M12","0"),("M13","0"),("M14","0"),
                 ("M21","0"),("M22","1"),("M23","0"),("M24","0"),
                 ("M31","0"),("M32","0"),("M33","1"),("M34","0"),
                 ("M41",str(cx)),("M42",str(cy)),("M43","0"),("M44","1"),
                 ("CenterLocalX","0"),("CenterLocalY","0")]:
        etree.SubElement(pt, k).text = v

    etree.SubElement(section, "IsVisible").text = "true"
    da = etree.SubElement(section, "DefaultAlignment")
    _identity_transform_elem(da, "ParentTransform")

    etree.SubElement(section, "MinZ").text = "0"
    etree.SubElement(section, "MaxZ").text = "0"

    geom = etree.SubElement(section, "Geometry")
    etree.SubElement(geom, "Type").text = "Polygon"
    for lx, ly in local_verts:
        v = etree.SubElement(geom, "Vertex")
        etree.SubElement(v, "X").text = f"{lx:.6f}"
        etree.SubElement(v, "Y").text = f"{ly:.6f}"

    rc = etree.SubElement(section, "RotationCentre")
    etree.SubElement(rc, "X").text = "0"
    etree.SubElement(rc, "Y").text = "0"

    etree.SubElement(section, "LinkUID").text = str(link_uid)
    etree.SubElement(section, "GeometryAcquired").text = "false"
    etree.SubElement(section, "SectionIndex").text = str(index)

    at = etree.SubElement(section, "AlignmentTransform")
    _identity_transform_elem(at, "ParentTransform")
    etree.SubElement(section, "Aligned").text = "false"

    acq = etree.SubElement(section, "AcquisitionSpec")
    etree.SubElement(acq, "Name").text = f"Acquisition Spec {index + 1}"
    etree.SubElement(acq, "UID").text = str(link_uid)
    etree.SubElement(acq, "PlaceableDataUID").text = "-1"
    etree.SubElement(acq, "AcquisitionTypeEnum").text = "ForAcquisition"
    etree.SubElement(acq, "AcquisitionDataTypeEnum").text = "2DData"
    etree.SubElement(acq, "AcquisitionStateEnum").text = "Fresh"
    etree.SubElement(acq, "AcquisitionMode").text = "Mosaic"
    etree.SubElement(acq, "WorkingProtocolUID").text = str(protocol_uid)

    if protocol_template is not None:
        protocol = copy.deepcopy(protocol_template)
        uid_elem = protocol.find("UID")
        if uid_elem is not None:
            uid_elem.text = str(protocol_uid)
        acq.append(protocol)

    return section


def generate_a5proj(template_path, magc_file_path, output_path,
                    shift_x=0.0, shift_y=0.0, rotation=0.0, scale=7.14725,
                    flip_x=False, flip_y=False, optimize=True,
                    optimize_method="simulated_annealing",
                    time_limit=10, create_overlay=True, overlay_output=None,
                    tif_background=None, image_path_override=None,
                    image_relative=False, image_scale=None,
                    image_positive_scale=False, no_embed_image=False):
    """
    Generate a complete ATLAS .a5proj file by injecting detected sections into
    a template project file (or a minimal built-in skeleton if template_path is
    None, i.e. --no-protocol mode).

    template_path=None activates --no-protocol mode: the ATLAS carrier XML is
    built from scratch and sections are generated without an imaging Protocol.
    Users must assign acquisition protocols in ATLAS before running.

    Scale: converts MAGC pixel coordinates to stage μm.
      Default 7.14725 = 1 / 0.13992 pixels/μm
    Shift: offset applied after scaling.  Use --center-on-wafer to auto-compute
      shifts that place the image center at stage (0,0) — the standard ATLAS
      wafer origin.  Fine-tune at the scope with explicit --shift_x / --shift_y.
    flip_y: negates Y before scaling to correct for image-Y-down vs stage-Y-up.
    """
    sections_data = parse_magc_file(magc_file_path)
    print(f"Loaded {len(sections_data)} sections from {magc_file_path}")

    if optimize and len(sections_data) > 1:
        orig_centroids = [calculate_centroid(s['vertices']) for s in sections_data]
        orig_dist = sum(
            math.sqrt((orig_centroids[i+1][0] - orig_centroids[i][0])**2 +
                      (orig_centroids[i+1][1] - orig_centroids[i][1])**2)
            for i in range(len(orig_centroids) - 1)
        )
        sections_data = optimize_path(sections_data, method=optimize_method, time_limit=time_limit)
        opt_centroids = [calculate_centroid(s['vertices']) for s in sections_data]
        opt_dist = sum(
            math.sqrt((opt_centroids[i+1][0] - opt_centroids[i][0])**2 +
                      (opt_centroids[i+1][1] - opt_centroids[i][1])**2)
            for i in range(len(opt_centroids) - 1)
        )
        improvement = (1 - opt_dist / orig_dist) * 100 if orig_dist > 0 else 0
        print(f"Path optimized using {optimize_method}: "
              f"{orig_dist * scale / 1000:.0f} mm → {opt_dist * scale / 1000:.0f} mm total stage travel "
              f"({improvement:.1f}% reduction)")

    if create_overlay:
        overlay_path = create_tif_overlay(sections_data, tif_path=tif_background, output_path=overlay_output)
        print(f"TIF overlay created at {overlay_path}")

    # Build or parse the ATLAS XML tree
    if template_path is None:
        print("--no-protocol: building minimal ATLAS skeleton (no imaging protocol).")
        print("  Assign an acquisition Protocol in ATLAS before running.")
        tree = _build_minimal_tree()
        root = tree.getroot()
        biosem = root.find("BioSemProject")
        carrier = biosem.find("AtlasCarrier")
        protocol_template = None
    else:
        tree = etree.parse(template_path)
        root = tree.getroot()

        biosem = root.find("BioSemProject")
        if biosem is None:
            raise ValueError(f"Template {template_path} has no <BioSemProject> element")

        carrier = biosem.find("AtlasCarrier")
        if carrier is None:
            raise ValueError(f"Template {template_path} has no <AtlasCarrier> element")

        # Extract Protocol XML from first SectionSet's first Section to reuse imaging settings
        protocol_template = None
        existing_section_sets = carrier.findall("SectionSet")
        if existing_section_sets:
            first_section = existing_section_sets[0].find("Section")
            if first_section is not None:
                acq_spec = first_section.find("AcquisitionSpec")
                if acq_spec is not None:
                    protocol_template = acq_spec.find("Protocol")
        print(f"Using imaging protocol from template: "
              f"{protocol_template.findtext('Name') if protocol_template is not None else 'none found'}")

    # ── Strip experiment-specific content from the template ──────────────────
    # Only remove what is experiment-specific; keep all structural carrier
    # elements (BioSemSession, BasicRegionSet, FeaturePointSet, DataSet,
    # DualBeamRegionSet, ActiveSession) that ATLAS requires to function.

    # Remove old SectionSets — replaced with ours below
    for ss in carrier.findall("SectionSet"):
        carrier.remove(ss)

    # Clear grab regions from "2D Regions" RegionSet (keep the RegionSet itself)
    region_set = carrier.find("RegionSet")
    if region_set is not None:
        for ar in region_set.findall("AtlasRegion"):
            region_set.remove(ar)
        rc = region_set.find("CreatedRegionCount")
        if rc is not None:
            rc.text = "0"

    # Remove only UserSectionSet — it is replaced with ours below.
    # LoadOut, StringMap, AutomationHistory and all other BioSemProject elements
    # must be kept: ActiveLoadOut references the LoadOut UID and ATLAS will fail
    # to initialise (silently, showing no sections) if that reference is dangling.
    for elem in biosem.findall("UserSectionSet"):
        biosem.remove(elem)

    # ── Update project identity metadata ─────────────────────────────────────
    import datetime as _dt
    output_basename = os.path.basename(output_path)
    base_no_ext = os.path.splitext(output_basename)[0]

    for tag, val in [
        ("Name",        output_basename),
        ("UID",         str(random.randint(100000000, 2147483647))),
        ("LastUpdated", _dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.000+00:00")),
        ("DataFolder",  base_no_ext + "_data"),
        ("XMLFile",     output_basename),
        ("CreatedUserSectionSetCount", "0"),
        ("CreatedGrabsCount",          "0"),
    ]:
        elem = biosem.find(tag)
        if elem is not None:
            elem.text = val

    # Register optical overview image (updates DataSet + Imported Regions)
    if tif_background and not no_embed_image:
        _update_optical_image(carrier, tif_background, scale, flip_x, flip_y, shift_x, shift_y,
                              image_path_override=image_path_override,
                              relative=image_relative, image_scale=image_scale,
                              image_flip_sign=not image_positive_scale)

    # Build new SectionSet
    section_set_name = "Section Set 1"
    ss = etree.SubElement(carrier, "SectionSet")
    etree.SubElement(ss, "Name").text = section_set_name
    etree.SubElement(ss, "UID").text = str(random.randint(100000000, 2147483647))
    _identity_transform_elem(ss, "ParentTransform")
    etree.SubElement(ss, "IsVisible").text = "true"
    da = etree.SubElement(ss, "DefaultAlignment")
    _identity_transform_elem(da, "ParentTransform")
    etree.SubElement(ss, "CreatedRegionCount").text = str(len(sections_data))
    etree.SubElement(ss, "ChildNameRoot").text = "Region "

    print(f"Building {len(sections_data)} Section elements...")
    for i, section in enumerate(sections_data):
        elem = build_section_element(
            i, section['vertices'], scale, shift_x, shift_y,
            rotation, flip_x, flip_y, protocol_template, section_set_name
        )
        ss.append(elem)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(sections_data)} sections written...")

    # Build UserSectionSet at BioSemProject level — ATLAS uses this to find and
    # display sections; SectionList2 must reference every Section UID.
    # Must be inserted before StringMap/AutomationHistory — ATLAS is order-sensitive.
    all_section_uids = [s.findtext('UID') for s in ss.findall('Section')]
    uss = etree.Element('UserSectionSet')
    etree.SubElement(uss, 'Name').text = section_set_name
    etree.SubElement(uss, 'UID').text = str(random.randint(100000000, 2147483647))
    _identity_transform_elem(uss, 'ParentTransform')
    etree.SubElement(uss, 'IsVisible').text = 'true'
    uss_da = etree.SubElement(uss, 'DefaultAlignment')
    _identity_transform_elem(uss_da, 'ParentTransform')
    sl2 = etree.SubElement(uss, 'SectionList2')
    for uid in all_section_uids:
        etree.SubElement(sl2, 'SectionUID').text = uid
    etree.SubElement(uss, 'CreatedSectionCount').text = str(len(all_section_uids))

    string_map = biosem.find('StringMap')
    if string_map is not None:
        biosem.insert(list(biosem).index(string_map), uss)
    else:
        biosem.append(uss)

    count_elem = biosem.find('CreatedUserSectionSetCount')
    if count_elem is not None:
        count_elem.text = '1'

    _ensure_biosem_session(carrier)

    tree.write(output_path, xml_declaration=True, encoding='utf-8', pretty_print=True)
    print(f"\nGenerated {output_path} with {len(sections_data)} sections")
    return output_path


def main():
    import datetime

    parser = argparse.ArgumentParser(
        description="Generate an ATLAS .a5proj file from a .magc ROI file and a template project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Coordinate calibration
----------------------
--scale converts MAGC pixel coordinates to stage μm.
  Default: 7.14725  =  1 / 0.13992 pixels/μm

--shift_x / --shift_y are the stage coordinates (in μm) of the top-left corner
  of the optical overview image.  Determine these at the scope by loading the
  .a5proj, finding a recognisable section in ATLAS, noting its stage position,
  then computing: shift = stage_pos - (MAGC_centroid × scale).
  Start with 0 / 0 for a first test load.
""")
    parser.add_argument('--magc-file', type=str, required=True,
                        help='Path to input .magc file (output of detect_rectangles.py)')
    parser.add_argument('--template', type=str,
                        default='atlas_template/blank_template.a5proj',
                        help='Path to ATLAS .a5proj template file '
                             '(default: atlas_template/blank_template.a5proj). '
                             'Ignored when --no-protocol is set.')
    parser.add_argument('--no-protocol', action='store_true',
                        help='Build the ATLAS project skeleton from scratch without any '
                             'template file. Sections will have no imaging Protocol — '
                             'assign one in ATLAS before acquiring. Use this when you '
                             'have no prior .a5proj file to use as a template.')
    parser.add_argument('--output', type=str, default=None,
                        help='Output .a5proj filename (default: auto-generated with timestamp)')
    parser.add_argument('--shift_x', type=float, default=0.0,
                        help='Stage X offset (μm) for image origin (default: 0)')
    parser.add_argument('--shift_y', type=float, default=0.0,
                        help='Stage Y offset (μm) for image origin (default: 0)')
    parser.add_argument('--rotation', type=float, default=0.0,
                        help='Rotation correction in degrees (default: 0)')
    parser.add_argument('--scale', type=float, default=7.14725,
                        help='μm per pixel (default: 7.14725 = 1/0.13992 px/μm camera)')
    parser.add_argument('--flip_x', action='store_true',
                        help='Flip X coordinates (mirror left-right)')
    parser.add_argument('--flip_y', action='store_true',
                        help='Flip Y coordinates (mirror top-bottom, corrects image-Y-down vs stage-Y-up)')
    parser.add_argument('--center-on-wafer', action='store_true',
                        help='Auto-compute shifts so the image center maps to stage (0,0). '
                             'Requires --tif-background. Overrides --shift_x / --shift_y.')
    parser.add_argument('--no-optimize', action='store_true',
                        help='Disable path optimisation')
    parser.add_argument('--optimize-method', type=str, default='simulated_annealing',
                        choices=['tsp', 'nearest_neighbor', 'simulated_annealing', 'two_opt'],
                        help='Path optimisation method (default: simulated_annealing)')
    parser.add_argument('--time-limit', type=int, default=10,
                        help='Time limit in seconds for path optimisation (default: 10)')
    parser.add_argument('--no-overlay', action='store_true',
                        help='Skip creating the PNG acquisition path overlay')
    parser.add_argument('--overlay-output', type=str,
                        help='Custom path for the PNG overlay output')
    parser.add_argument('--tif-background', type=str, default=None,
                        help='Path to background image for overlay (e.g. wafer01.jpg)')
    parser.add_argument('--no-embed-image', action='store_true',
                        help='Do not embed the overview image in the project; import it '
                             'manually in ATLAS instead. --tif-background is still read '
                             'for its pixel dimensions, so --center-on-wafer keeps '
                             'working. Use this when ATLAS cannot build an image pyramid '
                             'for the embedded overview and shows a placeholder box.')
    parser.add_argument('--image-negative-scale', action='store_true',
                        help='Encode --flip_y as a negative scale in the background '
                             'image transform. Natively generated ATLAS projects always '
                             'use positive values, so this is off by default. Turn it on '
                             'only if the overview loads but appears vertically mirrored '
                             'relative to the ROIs.')
    parser.add_argument('--image-scale', type=float, default=None,
                        help='Microns per pixel of the BACKGROUND IMAGE, when it is a '
                             'downsampled copy of the image the .magc refers to. '
                             'Defaults to --scale. ATLAS builds an image pyramid and '
                             'can fail on very large overviews; halving the overview '
                             'means doubling this value. Sections are unaffected, as '
                             'their coordinates stay in original pixels.')
    parser.add_argument('--image-relative', action='store_true',
                        help='Store only the image filename, not a full path, so ATLAS '
                             'looks for it beside the .a5proj. Put the overview image '
                             'in the same folder as the project. Keeps the project '
                             'portable; use --image-path instead if ATLAS insists on '
                             'an absolute path.')
    parser.add_argument('--image-path', type=str, default=None,
                        help='Exact path ATLAS should use to find the background image on the SEM PC '
                             '(e.g. "C:\\\\SEM\\\\wafer01.jpg"). Without this, the Mac absolute '
                             'path is stored and ATLAS will prompt once to relocate the file on Windows.')

    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or f"atlas_{timestamp}.a5proj"

    template_path = None if args.no_protocol else args.template
    if args.no_protocol and args.template != 'atlas_template/blank_template.a5proj':
        print("Note: --no-protocol is set; --template is ignored.")

    shift_x = args.shift_x
    shift_y = args.shift_y

    if args.center_on_wafer:
        if not args.tif_background:
            print("Error: --center-on-wafer requires --tif-background to read image dimensions.")
            raise SystemExit(1)
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(args.tif_background) as img:
            img_w, img_h = img.size
        # Use the background image's own scale: when it is a downsampled copy,
        # its pixel dimensions no longer correspond to --scale, and using the
        # wrong one puts the centre off by the downsample factor.
        _bg = args.image_scale if args.image_scale else args.scale
        shift_x = (1 if args.flip_x else -1) * (img_w / 2.0) * _bg
        shift_y = (1 if args.flip_y else -1) * (img_h / 2.0) * _bg
        print(f"--center-on-wafer: image {img_w}×{img_h} px @ {_bg:g} um/px → "
              f"shift_x={shift_x:.1f} μm, shift_y={shift_y:.1f} μm")

    generate_a5proj(
        template_path=template_path,
        magc_file_path=args.magc_file,
        output_path=output_path,
        shift_x=shift_x,
        shift_y=shift_y,
        rotation=args.rotation,
        scale=args.scale,
        flip_x=args.flip_x,
        flip_y=args.flip_y,
        optimize=not args.no_optimize,
        optimize_method=args.optimize_method,
        time_limit=args.time_limit,
        create_overlay=not args.no_overlay,
        overlay_output=args.overlay_output,
        tif_background=args.tif_background,
        image_path_override=args.image_path,
        image_relative=args.image_relative,
        image_scale=args.image_scale,
        image_positive_scale=not args.image_negative_scale,
        no_embed_image=args.no_embed_image,
    )


if __name__ == "__main__":
    main()
