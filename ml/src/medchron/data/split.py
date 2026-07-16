
from __future__ import annotations

import copy
from collections import Counter, defaultdict
from typing import Dict, Hashable, List

from sklearn.model_selection import train_test_split

from .manifest import Sample


def _three_way(keys: List[Hashable], label_of: Dict, val_size: float, test_size: float, seed: int):

    def _split(ks: List[Hashable], frac: float):
        if frac <= 0 or len(ks) < 2:
            return ks, []
        lbls = [label_of[k] for k in ks]
        stratify = lbls if min(Counter(lbls).values()) >= 2 else None
        return train_test_split(ks, test_size=frac, random_state=seed, stratify=stratify)

    train_keys, test_keys = _split(keys, test_size)
    val_rel = 0.0 if (1 - test_size) <= 0 else val_size / (1 - test_size)
    train_keys, val_keys = _split(train_keys, val_rel)
    return set(train_keys), set(val_keys), set(test_keys)


def stratified_split(
    samples: List[Sample],
    *,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    group_by_patient: bool = True,
) -> List[Sample]:
    out = [copy.copy(s) for s in samples]
    have_patients = group_by_patient and all(s.patient_id for s in out)

    if have_patients:
        by_patient: Dict[str, List[str]] = defaultdict(list)
        for s in out:
            by_patient[s.patient_id].append(s.label)
        patients = list(by_patient)
        label_of = {p: Counter(lbls).most_common(1)[0][0] for p, lbls in by_patient.items()}
        train, val, test = _three_way(patients, label_of, val_size, test_size, seed)
        assign = {p: ("train" if p in train else "val" if p in val else "test") for p in patients}
        for s in out:
            s.split = assign[s.patient_id]
    else:
        keys = list(range(len(out)))
        label_of = {i: out[i].label for i in keys}
        train, val, test = _three_way(keys, label_of, val_size, test_size, seed)
        for i, s in enumerate(out):
            s.split = "train" if i in train else "val" if i in val else "test"

    return out


def assert_no_patient_leakage(samples: List[Sample]) -> None:
    seen: Dict[str, str] = {}
    for s in samples:
        if not s.patient_id:
            continue
        if s.patient_id in seen and seen[s.patient_id] != s.split:
            raise AssertionError(
                f"Patient {s.patient_id!r} leaks across splits "
                f"{seen[s.patient_id]!r} and {s.split!r}"
            )
        seen[s.patient_id] = s.split
