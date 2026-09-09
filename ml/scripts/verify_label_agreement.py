"""Directly test the label-noise hypothesis.

If SARTAJ (our training source) carries label errors that the external
compilation corrected, then the SAME physical image should appear under
different class labels in the two datasets. Byte-identical (MD5) matches
let us check that with zero ambiguity.
"""
import csv
from collections import Counter

A_CLASSES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
B_MAP = {
    "glioma": "glioma_tumor",
    "meningioma": "meningioma_tumor",
    "notumor": "no_tumor",
    "pituitary": "pituitary_tumor",
}


def norm(p):
    return p.replace("\\", "/")


def lab_a(p):
    p = norm(p)
    for c in A_CLASSES:
        if f"/merged/{c}/" in p:
            return c
    return "?"


def lab_b(p):
    p = norm(p)
    for k, v in B_MAP.items():
        if f"/{k}/" in p:
            return v
    return "?"


rows = list(csv.DictReader(open("ml/artifacts/brain_mri_overlap.csv")))
exact = [r for r in rows if r["kind"] == "exact_md5"]

unparsed = sum(1 for r in exact if lab_a(r["a_path"]) == "?" or lab_b(r["b_path"]) == "?")
print(f"unparsed labels: {unparsed} (must be 0)")

agree = sum(1 for r in exact if lab_a(r["a_path"]) == lab_b(r["b_path"]))
dis = len(exact) - agree
print(f"\nBYTE-IDENTICAL images present in BOTH datasets: {len(exact)}")
print(f"  labels agree:    {agree}  ({100*agree/len(exact):.2f}%)")
print(f"  labels DISAGREE: {dis}  ({100*dis/len(exact):.2f}%)")

if dis:
    pairs = Counter((lab_a(r["a_path"]), lab_b(r["b_path"]))
                    for r in exact if lab_a(r["a_path"]) != lab_b(r["b_path"]))
    print("\n  disagreement pairs (ours -> external):")
    for (a, b), c in pairs.most_common(10):
        print(f"    {a:18s} -> {b:18s}  {c}")
else:
    print("\n  => No label disagreement on any shared image.")
    print("     The label-noise hypothesis is NOT supported by this evidence.")
