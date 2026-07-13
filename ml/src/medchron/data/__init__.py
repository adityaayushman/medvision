"""Dataset layer: registry, manifests, and leakage-safe splitting."""

from .manifest import (
    LABEL_DELIM,
    Sample,
    build_manifest_from_csv,
    build_manifest_from_folders,
    build_multilabel_class_index,
    class_distribution,
    read_manifest,
    split_distribution,
    write_manifest,
)
from .registry import REGISTRY, DatasetSpec, get_spec, recommended_v1
from .split import assert_no_patient_leakage, stratified_split

__all__ = [
    "Sample",
    "build_manifest_from_folders",
    "build_manifest_from_csv",
    "write_manifest",
    "read_manifest",
    "class_distribution",
    "split_distribution",
    "build_multilabel_class_index",
    "LABEL_DELIM",
    "stratified_split",
    "assert_no_patient_leakage",
    "REGISTRY",
    "DatasetSpec",
    "get_spec",
    "recommended_v1",
]
