#!/usr/bin/env python3
"""
Sanity-check a generated ATLAS .a5proj before loading it on the scope.

    python check_a5proj.py wafer.a5proj

Checks structural completeness rather than schema conformance -- it does not
need the Fibics .xsd. Written after finding that --no-protocol produces files
whose sections reference imaging protocols that do not exist anywhere in the
file. Exits nonzero if anything failed, so it can gate a script.

If you have the schema locally, this is complementary, not a substitute:

    xmllint --schema FibicsAtlasProject_2_10.xsd wafer.a5proj --noout
"""
import sys
from lxml import etree

# Siblings of BioSemProject that a template-derived project has and a
# from-scratch one does not. Their absence is the clearest signal that a file
# was built with --no-protocol.
TOP_LEVEL = ['OriginalRoot', 'BioSemProject', 'LayerFolder',
             'ProtocolCache', 'RecipeCache']


def check(path):
    try:
        root = etree.parse(path).getroot()
    except (OSError, etree.XMLSyntaxError) as e:
        print(f"FAIL  cannot parse {path}: {e}")
        return False

    ok = True

    def report(passed, label, detail=''):
        nonlocal ok
        print(f"{'PASS' if passed else 'FAIL'}  {label}"
              f"{'  ' + detail if detail else ''}")
        if not passed:
            ok = False

    print(f"=== {path} ===")

    report(root.tag == 'F-BioSEM-Project', 'root element is F-BioSEM-Project',
           f"got <{root.tag}>")

    present = [c.tag for c in root]
    missing = [t for t in TOP_LEVEL if t not in present]
    report(not missing, 'top-level elements present',
           f"missing {missing} (was this built with --no-protocol?)" if missing
           else f"{len(TOP_LEVEL)}/{len(TOP_LEVEL)}")

    # Every WorkingProtocolUID must resolve to a Protocol defined in the file.
    refs = {e.text for e in root.iter('WorkingProtocolUID') if e.text}
    defined = {e.findtext('UID') for e in root.iter('Protocol')}
    defined.discard(None)
    dangling = refs - defined
    report(not dangling, 'protocol references resolve',
           f"{len(dangling)} of {len(refs)} dangling -- sections point at "
           f"protocols that do not exist" if dangling
           else f"{len(refs)} refs, {len(defined)} protocols defined")

    sections = list(root.iter('Section'))
    report(bool(sections), 'contains sections', f"{len(sections)} found")

    # Geometry: each section should carry a Polygon with at least 3 vertices.
    vcounts, bad_type = [], []
    for s in sections:
        g = s.find('Geometry')
        if g is None:
            continue
        if (g.findtext('Type') or '') != 'Polygon':
            bad_type.append(g.findtext('Type'))
        vcounts.append(len(g.findall('Vertex')))

    report(len(vcounts) == len(sections), 'every section has Geometry',
           f"{len(vcounts)}/{len(sections)}")
    report(not bad_type, 'geometry type is Polygon',
           f"unexpected types: {set(bad_type)}" if bad_type else '')
    if vcounts:
        report(min(vcounts) >= 3, 'vertex counts sane',
               f"per-section vertices: {sorted(set(vcounts))}")

    # Section sets, and whether each declares its membership list.
    sets = list(root.iter('SectionSet'))
    user_sets = list(root.iter('UserSectionSet'))
    print(f"INFO  {len(sets)} SectionSet, {len(user_sets)} UserSectionSet")
    for us in user_sets:
        name = us.findtext('Name') or '(unnamed)'
        lst = us.find('SectionList2')
        # Membership is written either as child <SectionUID> elements or as
        # comma-delimited text, depending on the generator. Accept both.
        n = 0
        if lst is not None:
            n = len(lst.findall('SectionUID'))
            if n == 0 and lst.text and lst.text.strip():
                n = len([x for x in lst.text.split(',') if x.strip()])
        declared = us.findtext('CreatedSectionCount')
        flag = '' if str(n) == (declared or '') else f"  <-- disagrees with CreatedSectionCount={declared}"
        print(f"INFO    '{name}': {n} sections listed{flag}")

    # Duplicate UIDs anywhere in the file are a corruption signature.
    all_uids = [e.text for e in root.iter('UID') if e.text]
    dupes = {u for u in all_uids if all_uids.count(u) > 1}
    report(not dupes, 'UIDs unique',
           f"{len(dupes)} duplicated" if dupes else f"{len(all_uids)} UIDs")

    print(f"\n{'OK' if ok else 'PROBLEMS FOUND'}: {path}")
    return ok


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__.strip())
    results = [check(p) for p in sys.argv[1:]]
    sys.exit(0 if all(results) else 1)
