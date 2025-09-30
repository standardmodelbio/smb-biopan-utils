#! /usr/bin/env python3
#
# Copyright © 2025 Standard Model Biomedicine, Inc. <zach@standardmodel.bio>
#
# Distributed under terms of the Apache License 2.0 license.
import torch

# ==========================
# Medical imaging utilities
# ==========================

CT_DEFAULT_A_MIN = -1000
CT_DEFAULT_A_MAX = 1000
CT_DEFAULT_B_MIN = 0.0
CT_DEFAULT_B_MAX = 1.0


def _require_monai() -> None:
    try:
        import monai  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "MONAI is required for medical imaging preprocessing. Install with: pip install monai nibabel"
        ) from e


def get_ct_transforms(
    spatial_size: tuple[int, int, int] = (384, 384, 256),
    pixdim: tuple[float, float, float] = (1.0, 1.0, 1.5),
    a_min: float = CT_DEFAULT_A_MIN,
    a_max: float = CT_DEFAULT_A_MAX,
    b_min: float = CT_DEFAULT_B_MIN,
    b_max: float = CT_DEFAULT_B_MAX,
):
    """Create a MONAI Compose for CT NIfTI preprocessing.

    Returns a transform that produces a tensor with shape (D, C, H, W), where C=1.
    """
    _require_monai()
    from monai.transforms import (
        CenterSpatialCropd,
        Compose,
        EnsureChannelFirstd,
        LoadImaged,
        MapTransform,
        Orientationd,
        ScaleIntensityRanged,
        Spacingd,
        SpatialPadd,
        ToTensord,
    )

    class PermuteImage(MapTransform):
        """Permute the dimensions for VJEPA2 input: (D, C, H, W)."""

        def __init__(self, keys=["image"], allow_missing_keys=False):
            MapTransform.__init__(self, keys, allow_missing_keys)

        def __call__(self, data):
            data["image"] = data["image"].permute(3, 0, 1, 2)
            return data

    ct_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=pixdim, mode=("bilinear")),
            ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, clip=True),
            SpatialPadd(keys=["image"], spatial_size=list(spatial_size)),
            CenterSpatialCropd(roi_size=list(spatial_size), keys=["image"]),
            ToTensord(keys=["image"]),
            PermuteImage(),
        ]
    )
    return ct_transforms


def preprocess_ct_nifti(
    nifti_path: str,
    spatial_size: tuple[int, int, int] = (384, 384, 256),
    pixdim: tuple[float, float, float] = (1.0, 1.0, 1.5),
    a_min: float = CT_DEFAULT_A_MIN,
    a_max: float = CT_DEFAULT_A_MAX,
    b_min: float = CT_DEFAULT_B_MIN,
    b_max: float = CT_DEFAULT_B_MAX,
) -> torch.Tensor:
    """Load and preprocess a CT NIfTI file to VJEPA2-ready shape [1, D, C, H, W]."""
    transforms = get_ct_transforms(
        spatial_size=spatial_size, pixdim=pixdim, a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max
    )
    data_dict = {"image": nifti_path}
    transformed = transforms(data_dict)
    volume_dc_hw = transformed["image"]  # (D, C, H, W)
    if volume_dc_hw.dim() != 4:
        raise ValueError(f"Expected 4D tensor (D, C, H, W), got shape {tuple(volume_dc_hw.shape)}")
    # Add batch dimension -> (B, D, C, H, W)
    return volume_dc_hw.unsqueeze(0)


def fetch_medical_volume(ele: dict) -> torch.Tensor:
    """Convenience wrapper to preprocess medical volumes.

    Supported keys:
      - "nifti_path" or "image": path to .nii/.nii.gz CT volume
      - Optional overrides: spatial_size, pixdim, a_min, a_max, b_min, b_max

    Returns: torch.Tensor with shape [1, D, C, H, W]
    """
    nifti_path = ele.get("nifti_path") or ele.get("image")
    if not isinstance(nifti_path, str):
        raise ValueError("fetch_medical_volume expects 'nifti_path' or 'image' string path to a NIfTI file")
    spatial_size = tuple(ele.get("spatial_size", (384, 384, 256)))
    pixdim = tuple(ele.get("pixdim", (1.0, 1.0, 1.5)))
    a_min = float(ele.get("a_min", CT_DEFAULT_A_MIN))
    a_max = float(ele.get("a_max", CT_DEFAULT_A_MAX))
    b_min = float(ele.get("b_min", CT_DEFAULT_B_MIN))
    b_max = float(ele.get("b_max", CT_DEFAULT_B_MAX))
    return preprocess_ct_nifti(
        nifti_path=nifti_path,
        spatial_size=spatial_size,
        pixdim=pixdim,
        a_min=a_min,
        a_max=a_max,
        b_min=b_min,
        b_max=b_max,
    )


def extract_imaging_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    """Extract medical imaging entries from a conversation-like structure.

    Accepts either a list of dicts, or a list of list of dicts (messages with content).
    Looks for elements containing 'image' or 'nifti_path' pointing to a .nii/.nii.gz.
    """
    imaging_infos: list[dict] = []
    if not conversations:
        return imaging_infos
    if isinstance(conversations[0], dict):
        conversations = [conversations]  # type: ignore[assignment]
    for conversation in conversations:  # type: ignore[assignment]
        for message in conversation:
            if isinstance(message.get("content"), list):
                for ele in message["content"]:
                    path_val = ele.get("image") or ele.get("nifti_path")
                    if isinstance(path_val, str) and (path_val.endswith(".nii") or path_val.endswith(".nii.gz")):
                        imaging_infos.append(ele)
                    elif ele.get("type", "") in ("image", "nifti"):
                        imaging_infos.append(ele)
    return imaging_infos


def process_imaging_info(
    conversations: list[dict] | list[list[dict]],
) -> list[torch.Tensor] | None:
    """Process medical imaging info into preprocessed tensors.

    Returns a list of tensors with shape [1, D, C, H, W] or None if no entries.
    """
    imaging_infos = extract_imaging_info(conversations)
    print(imaging_infos)
    volume_inputs: list[torch.Tensor] = []
    for info in imaging_infos:
        volume_inputs.append(fetch_medical_volume(info))
    if len(volume_inputs) == 0:
        return None
    return volume_inputs
