"""Does class composition explain 'external beats in-domain'?

Accuracy is a weighted average of per-class recall, weighted by each class's
share of the test set. If the external set happens to contain more of the
classes this model is good at, its accuracy rises without any real gain in
generalization.

Test: reweight the external per-class recalls to the IN-DOMAIN class
proportions. If the gap collapses, composition explains it.
"""
import json

indomain = json.load(open("ml/artifacts/plain_effnet_seed3/metrics.json"))
external = json.load(open("ml/artifacts/xdata_single_clean/metrics.json"))

CLASSES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

id_sup = {c: indomain["per_class"][c]["support"] for c in CLASSES}
ex_sup = {c: external["per_class"][c]["support"] for c in CLASSES}
id_rec = {c: indomain["per_class"][c]["recall"] for c in CLASSES}
ex_rec = {c: external["per_class"][c]["recall"] for c in CLASSES}

id_n, ex_n = sum(id_sup.values()), sum(ex_sup.values())

print(f"{'class':20s} {'in-dom share':>13s} {'ext share':>11s} | {'in-dom recall':>14s} {'ext recall':>11s}")
for c in CLASSES:
    print(f"{c:20s} {100*id_sup[c]/id_n:12.1f}% {100*ex_sup[c]/ex_n:10.1f}% | "
          f"{100*id_rec[c]:13.1f}% {100*ex_rec[c]:10.1f}%")

print(f"\nraw accuracy   in-domain {100*indomain['accuracy']:.2f}%   external {100*external['accuracy']:.2f}%"
      f"   gap {100*(external['accuracy']-indomain['accuracy']):+.2f} pts")

# Reweight external recalls to in-domain class proportions.
adj = sum((id_sup[c] / id_n) * ex_rec[c] for c in CLASSES)
print(f"\nexternal accuracy REWEIGHTED to in-domain class mix: {100*adj:.2f}%")
print(f"  gap after adjustment: {100*(adj-indomain['accuracy']):+.2f} pts")
explained = (external["accuracy"] - adj) / (external["accuracy"] - indomain["accuracy"]) * 100
print(f"  share of the raw gap explained by class composition alone: {explained:.0f}%")

# And the reverse: in-domain reweighted to external mix.
adj2 = sum((ex_sup[c] / ex_n) * id_rec[c] for c in CLASSES)
print(f"\nin-domain accuracy REWEIGHTED to external class mix: {100*adj2:.2f}%")
print(f"  (vs external {100*external['accuracy']:.2f}%)")
