"""Registry of candidate datasets — the project's dataset shortlist, as code.

Encoding the shortlist here (instead of only in a doc) means the training and
CLI code can reason about a dataset's task, modality, ROI support and access
constraints programmatically. V1 targets chest X-ray.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal

Task = Literal["classification", "multilabel", "detection", "segmentation"]
Access = Literal["open", "kaggle", "credentialed"]


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    name: str
    modality: str
    task: Task
    access: Access
    roi_support: bool                 # does it provide masks / boxes for real ROI?
    approx_images: str
    url: str
    notes: str = ""
    recommended_for: List[str] = field(default_factory=list)


REGISTRY: Dict[str, DatasetSpec] = {
    # ---- Chest X-ray classification / localization ----
    "rsna_pneumonia": DatasetSpec(
        key="rsna_pneumonia",
        name="RSNA Pneumonia Detection Challenge",
        modality="Chest X-ray",
        task="detection",
        access="kaggle",
        roi_support=True,             # bounding boxes on lung opacities
        approx_images="~26k",
        url="https://www.kaggle.com/datasets/parin30/rsna-pneumonia-detection",
        notes="Cleanest path to an MVP: clear labels + boxes for ROI/localization.",
        recommended_for=["v1", "mvp", "roi", "grad-cam demo"],
    ),
    "nih_cxr14": DatasetSpec(
        key="nih_cxr14",
        name="NIH ChestX-ray14",
        modality="Chest X-ray",
        task="multilabel",
        access="kaggle",
        roi_support=False,
        approx_images="112k+",
        url="https://www.kaggle.com/datasets/nih-chest-xrays/data",
        notes="Large-scale multi-disease benchmark; weaker localization support.",
        recommended_for=["scale", "multi-disease"],
    ),
    "vindr_cxr": DatasetSpec(
        key="vindr_cxr",
        name="VinDr-CXR",
        modality="Chest X-ray",
        task="detection",
        access="credentialed",
        roi_support=True,
        approx_images="18k",
        url="https://physionet.org/content/vindr-cxr/1.0.0/",
        notes="Radiologist boxes, DICOM; strongest scope but heavier + credentialed.",
        recommended_for=["professional scope", "localization"],
    ),
    "chexpert": DatasetSpec(
        key="chexpert",
        name="CheXpert",
        modality="Chest X-ray",
        task="multilabel",
        access="credentialed",
        roi_support=False,
        approx_images="224k",
        url="https://stanfordmlgroup.github.io/competitions/chexpert/",
        notes="Stanford multi-label benchmark; strong research credibility.",
        recommended_for=["benchmark"],
    ),
    # ---- Lung-field ROI / segmentation (fixes whole-image Otsu limitation) ----
    "montgomery_shenzhen": DatasetSpec(
        key="montgomery_shenzhen",
        name="Montgomery + Shenzhen Lung Segmentation",
        modality="Chest X-ray",
        task="segmentation",
        access="kaggle",
        roi_support=True,
        approx_images="~800",
        url="https://www.kaggle.com/datasets/iamtapendu/chest-x-ray-lungs-segmentation",
        notes="Lung masks -> train a lung-field ROI so cropping is anatomical, not Otsu.",
        recommended_for=["roi", "lung segmentation"],
    ),
    "crd_lung_masks": DatasetSpec(
        key="crd_lung_masks",
        name="CRD: Chest X-ray with Lung Segmented Masks",
        modality="Chest X-ray",
        task="segmentation",
        access="kaggle",
        roi_support=True,
        approx_images="3,311",
        url="https://www.kaggle.com/datasets",
        notes="Image+mask pairs for ROI/preprocessing experiments.",
        recommended_for=["roi", "lung segmentation"],
    ),
}


def get_spec(key: str) -> DatasetSpec:
    try:
        return REGISTRY[key]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset {key!r}. Known: {sorted(REGISTRY)}") from exc


def recommended_v1() -> List[DatasetSpec]:
    """The V1 pairing: one classifier dataset + one lung-ROI dataset."""
    return [REGISTRY["rsna_pneumonia"], REGISTRY["montgomery_shenzhen"]]
