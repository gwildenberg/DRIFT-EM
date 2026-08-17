#!/usr/bin/env python3
"""
Strip an ATLAS .a5proj file to a minimal blank template for shipping with
generate_and_fuse.py.

What is kept:
  - All ATLAS structural elements (carrier, region sets, data sets, load-out)
  - The imaging Protocol from the first section (so generate_and_fuse.py can
    inherit pixel size, dwell time, detector, etc.)

What is removed:
  - All sections / section sets
  - All BioSemSessions (instrument-session specific)
  - Grab regions (AtlasRegion) from the 2D RegionSet
  - Linked/imported image data (LinkedRegion, PlaceableImage)
  - Experiment-specific metadata (Name, UID, timestamps, paths)

Usage:
    python create_blank_template.py [input.a5proj] [output.a5proj]

Defaults:
    input:  a real ATLAS project to strip (required)
    output: atlas_template/blank_template.a5proj
"""

import sys
import copy
import random
from lxml import etree


def _identity_pt(parent, tag="ParentTransform"):
    pt = etree.SubElement(parent, tag)
    for k, v in [("M11","1"),("M12","0"),("M13","0"),("M14","0"),
                 ("M21","0"),("M22","1"),("M23","0"),("M24","0"),
                 ("M31","0"),("M32","0"),("M33","1"),("M34","0"),
                 ("M41","0"),("M42","0"),("M43","0"),("M44","1"),
                 ("CenterLocalX","0"),("CenterLocalY","0")]:
        etree.SubElement(pt, k).text = v
    return pt


def create_blank_template(input_path, output_path):
    tree = etree.parse(input_path)
    root = tree.getroot()
    biosem = root.find("BioSemProject")
    carrier = biosem.find("AtlasCarrier")

    # ── 1. Extract Protocol before removing sections ──────────────────────────
    protocol_template = None
    for ss in carrier.findall("SectionSet"):
        sec = ss.find("Section")
        if sec is not None:
            acq = sec.find("AcquisitionSpec")
            if acq is not None:
                proto = acq.find("Protocol")
                if proto is not None:
                    protocol_template = copy.deepcopy(proto)
                    print(f"Protocol extracted: '{proto.findtext('Name')}'")
                    break

    if protocol_template is None:
        print("Warning: no Protocol found in input — blank template will have no protocol.")

    # ── 2. Strip carrier experiment data ─────────────────────────────────────
    for ss in carrier.findall("SectionSet"):
        carrier.remove(ss)

    for sess in carrier.findall("BioSemSession"):
        carrier.remove(sess)

    active_sess = carrier.find("ActiveSession")
    if active_sess is not None:
        active_sess.text = "-1"

    rs = carrier.find("RegionSet")
    if rs is not None:
        for ar in rs.findall("AtlasRegion"):
            rs.remove(ar)
        rc = rs.find("CreatedRegionCount")
        if rc is not None:
            rc.text = "0"

    for brs in carrier.findall("BasicRegionSet"):
        if brs.findtext("Name") == "Imported Regions":
            for lr in brs.findall("LinkedRegion"):
                brs.remove(lr)

    for ds in carrier.findall("DataSet"):
        if ds.findtext("Name") == "Imported Data":
            for pi in ds.findall("PlaceableImage"):
                ds.remove(pi)
            asu = ds.find("AcquisitionSpecUID")
            if asu is not None:
                asu.text = "-1"

    # ── 3. Strip BioSemProject experiment data ────────────────────────────────
    for uss in biosem.findall("UserSectionSet"):
        biosem.remove(uss)

    # ── 4. Embed Protocol in a stub SectionSet ────────────────────────────────
    # generate_and_fuse.py reads the Protocol from the first SectionSet's first
    # Section. The stub has no geometry — it is stripped and replaced entirely
    # when generate_and_fuse.py runs. Without this stub, users using --template
    # would get sections with no imaging Protocol.
    if protocol_template is not None:
        proto_uid = random.randint(100000000, 2147483647)
        link_uid  = random.randint(100000000, 2147483647)
        sec_uid   = random.randint(100000000, 2147483647)
        ss_uid    = random.randint(100000000, 2147483647)

        stub_ss = etree.SubElement(carrier, "SectionSet")
        etree.SubElement(stub_ss, "Name").text = "_ProtocolStub"
        etree.SubElement(stub_ss, "UID").text = str(ss_uid)
        _identity_pt(stub_ss)
        etree.SubElement(stub_ss, "IsVisible").text = "true"
        _identity_pt(etree.SubElement(stub_ss, "DefaultAlignment"))
        etree.SubElement(stub_ss, "CreatedRegionCount").text = "1"
        etree.SubElement(stub_ss, "ChildNameRoot").text = "Region "

        stub_sec = etree.SubElement(stub_ss, "Section")
        etree.SubElement(stub_sec, "CreatedIndex").text = "1"
        etree.SubElement(stub_sec, "CustomName").text = "false"
        etree.SubElement(stub_sec, "Name").text = "_ProtocolStub - 1"
        etree.SubElement(stub_sec, "UID").text = str(sec_uid)
        _identity_pt(stub_sec)
        etree.SubElement(stub_sec, "IsVisible").text = "true"
        _identity_pt(etree.SubElement(stub_sec, "DefaultAlignment"))
        etree.SubElement(stub_sec, "MinZ").text = "0"
        etree.SubElement(stub_sec, "MaxZ").text = "0"
        geom = etree.SubElement(stub_sec, "Geometry")
        etree.SubElement(geom, "Type").text = "Polygon"
        rc2 = etree.SubElement(stub_sec, "RotationCentre")
        etree.SubElement(rc2, "X").text = "0"
        etree.SubElement(rc2, "Y").text = "0"
        etree.SubElement(stub_sec, "LinkUID").text = str(link_uid)
        etree.SubElement(stub_sec, "GeometryAcquired").text = "false"
        etree.SubElement(stub_sec, "SectionIndex").text = "0"
        _identity_pt(etree.SubElement(stub_sec, "AlignmentTransform"))
        etree.SubElement(stub_sec, "Aligned").text = "false"

        acq = etree.SubElement(stub_sec, "AcquisitionSpec")
        etree.SubElement(acq, "Name").text = "Acquisition Spec 1"
        etree.SubElement(acq, "UID").text = str(link_uid)
        etree.SubElement(acq, "PlaceableDataUID").text = "-1"
        etree.SubElement(acq, "AcquisitionTypeEnum").text = "ForAcquisition"
        etree.SubElement(acq, "AcquisitionDataTypeEnum").text = "2DData"
        etree.SubElement(acq, "AcquisitionStateEnum").text = "Fresh"
        etree.SubElement(acq, "AcquisitionMode").text = "Mosaic"
        etree.SubElement(acq, "WorkingProtocolUID").text = str(proto_uid)

        proto_copy = copy.deepcopy(protocol_template)
        uid_e = proto_copy.find("UID")
        if uid_e is not None:
            uid_e.text = str(proto_uid)
        acq.append(proto_copy)

    # ── 5. Reset metadata ─────────────────────────────────────────────────────
    for tag, val in [
        ("Name",        "blank_template"),
        ("UID",         str(random.randint(100000000, 2147483647))),
        ("LastUpdated", "2024-01-01T00:00:00.000+00:00"),
        ("DataFolder",  "blank_template_data"),
        ("XMLFile",     "blank_template.a5proj"),
        ("CreatedUserSectionSetCount", "0"),
        ("CreatedGrabsCount",          "0"),
        ("CreatedRegionCount",         "0"),
    ]:
        elem = biosem.find(tag)
        if elem is not None:
            elem.text = val

    tree.write(output_path, xml_declaration=True, encoding="utf-8", pretty_print=True)
    size_kb = __import__("os").path.getsize(output_path) // 1024
    print(f"Blank template written: {output_path} ({size_kb} KB)")
    print("Users can pass this to generate_and_fuse.py with --template, or")
    print("use --no-protocol to skip the template entirely.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: create_blank_template.py <input.a5proj> [output.a5proj]\n"
                 "  input must be a real ATLAS project whose first SectionSet's\n"
                 "  first Section carries an AcquisitionSpec/Protocol.")
    input_path  = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "atlas_template/blank_template.a5proj"
    create_blank_template(input_path, output_path)
