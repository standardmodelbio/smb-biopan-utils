#! /usr/bin/env python3
#
# Copyright © 2025 Standard Model Biomedicine, Inc. <zach@standardmodel.bio>
#
# Distributed under terms of the Apache License 2.0 license.
import os
import tempfile

import boto3
import torch


# ==========================
# Medical imaging utilities
# ==========================

CT_DEFAULT_SPATIAL_SIZE = (320, 320, 176)
CT_DEFAULT_PIXDIM = (1.0, 1.0, 1.5)
CT_DEFAULT_A_MIN = -1000
CT_DEFAULT_A_MAX = 1000
CT_DEFAULT_B_MIN = 0.0
CT_DEFAULT_B_MAX = 1.0
DEPTH_PATCH_SIZE = 16
PATCH_SIZE = 16


def _require_monai() -> None:
    try:
        import monai  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "MONAI is required for medical imaging preprocessing. Install with: pip install monai nibabel"
        ) from e


def get_ct_transforms(
    spatial_size: tuple[int, int, int] = CT_DEFAULT_SPATIAL_SIZE,
    pixdim: tuple[float, float, float] = CT_DEFAULT_PIXDIM,
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
            data["image"] = data["image"].permute(0, 3, 1, 2)
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
            ToTensord(keys=["image"], track_meta=False),
            PermuteImage(),
        ]
    )
    return ct_transforms


def download_nifti_file(nifti_path: str, save_dir: str = "/tmp") -> str:
    """Download a NIfTI file from a s3 or r2 object to local temp directory."""
    # get the bucket and object name from the nifti path
    # Parse S3/R2 URL to extract bucket and object key
    if nifti_path.startswith("s3://"):
        # Remove s3:// prefix and split into bucket and object key
        path_without_prefix = nifti_path[5:]  # Remove "s3://"
        bucket, object_name = path_without_prefix.split("/", 1)
    elif "/" in nifti_path:
        # Handle other URL formats or direct bucket/object paths
        bucket, object_name = nifti_path.split("/", 1)
    else:
        raise ValueError(f"Invalid S3/R2 path format: {nifti_path}")

    # create s3 client
    session = boto3.Session(profile_name="smb-dev")
    s3 = session.client("s3")

    # create temp directory and get local file path
    local_path = os.path.join(save_dir, os.path.basename(object_name))

    # download the nifti file from the s3 or r2 object to temp directory
    s3.download_file(bucket, object_name, local_path)
    return local_path


def preprocess_ct_nifti(
    nifti_path: str,
    spatial_size: tuple[int, int, int] = CT_DEFAULT_SPATIAL_SIZE,
    pixdim: tuple[float, float, float] = CT_DEFAULT_PIXDIM,
    a_min: float = CT_DEFAULT_A_MIN,
    a_max: float = CT_DEFAULT_A_MAX,
    b_min: float = CT_DEFAULT_B_MIN,
    b_max: float = CT_DEFAULT_B_MAX,
    depth_patch_size: int = 16,
    patch_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load and preprocess a CT NIfTI file to VJEPA2-ready shape [1, C, D, H, W]."""
    transforms = get_ct_transforms(
        spatial_size=spatial_size, pixdim=pixdim, a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max
    )
    data_dict = {"image": nifti_path}
    transformed = transforms(data_dict)
    volume_dc_hw = transformed["image"]  # (C, D, H, W)
    if volume_dc_hw.dim() != 4:
        raise ValueError(f"Expected 4D tensor (C, D, H, W), got shape {tuple(volume_dc_hw.shape)}")

    # get grid_thw
    grid_thw = torch.tensor(
        [
            volume_dc_hw.shape[1] // depth_patch_size,
            volume_dc_hw.shape[2] // patch_size,
            volume_dc_hw.shape[3] // patch_size,
        ]
    )

    # 1. Chain unfold calls for a cleaner look
    patches = (
        volume_dc_hw.unfold(1, depth_patch_size, depth_patch_size)
        .unfold(2, patch_size, patch_size)
        .unfold(3, patch_size, patch_size)
    )

    # 2. Permute to group grid dimensions and patch dimensions separately
    # Initial shape: (C, nD, nH, nW, d_p, p, p)
    # Target shape:  (nD, nH, nW, C, d_p, p, p)
    patches = patches.permute(1, 2, 3, 0, 4, 5, 6)

    # 3. Explicitly create a contiguous tensor, then flatten
    # This is the key optimization step.
    # The first three dimensions (nD, nH, nW) are flattened into `total_patches`.
    # The last four dimensions (C, d_p, p, p) are flattened into the feature dimension.
    patches = patches.contiguous().view(-1, volume_dc_hw.shape[0] * depth_patch_size * patch_size * patch_size)
    return patches, grid_thw


def fetch_medical_volume(ele: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Convenience wrapper to preprocess medical volumes.

    Supported keys:
      - "nifti_path" or "image": path to .nii/.nii.gz CT volume
      - Optional overrides: spatial_size, pixdim, a_min, a_max, b_min, b_max

    Returns: tuple[torch.Tensor, torch.Tensor]
    """
    nifti_path = ele.get("nifti_path") or ele.get("image")
    if not isinstance(nifti_path, str):
        raise ValueError("fetch_medical_volume expects 'nifti_path' or 'image' string path to a NIfTI file")
    spatial_size = tuple(ele.get("spatial_size", CT_DEFAULT_SPATIAL_SIZE))
    pixdim = tuple(ele.get("pixdim", CT_DEFAULT_PIXDIM))
    a_min = float(ele.get("a_min", CT_DEFAULT_A_MIN))
    a_max = float(ele.get("a_max", CT_DEFAULT_A_MAX))
    b_min = float(ele.get("b_min", CT_DEFAULT_B_MIN))
    b_max = float(ele.get("b_max", CT_DEFAULT_B_MAX))
    depth_patch_size = int(ele.get("depth_patch_size", DEPTH_PATCH_SIZE))
    patch_size = int(ele.get("patch_size", PATCH_SIZE))

    # download nifti file if it is not a local path
    if not os.path.exists(nifti_path):
        with tempfile.TemporaryDirectory() as temp_dir:
            nifti_path = download_nifti_file(nifti_path, temp_dir)
            patches, grid_thw = preprocess_ct_nifti(
                nifti_path=nifti_path,
                spatial_size=spatial_size,
                pixdim=pixdim,
                a_min=a_min,
                a_max=a_max,
                b_min=b_min,
                b_max=b_max,
                depth_patch_size=depth_patch_size,
                patch_size=patch_size,
            )
    else:
        patches, grid_thw = preprocess_ct_nifti(
            nifti_path=nifti_path,
            spatial_size=spatial_size,
            pixdim=pixdim,
            a_min=a_min,
            a_max=a_max,
            b_min=b_min,
            b_max=b_max,
            depth_patch_size=depth_patch_size,
            patch_size=patch_size,
        )
    return patches, grid_thw


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
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Process medical imaging info into preprocessed tensors.

    Returns a list of tensors with shape [1, D, C, H, W] or None if no entries.
    """
    imaging_infos = extract_imaging_info(conversations)
    volume_inputs: list[torch.Tensor] = []
    grid_thws: list[torch.Tensor] = []
    for info in imaging_infos:
        volume, grid_thw = fetch_medical_volume(info)
        volume_inputs.append(volume)
        grid_thws.append(grid_thw)
    if len(volume_inputs) == 0:
        return None
    return torch.cat(volume_inputs, dim=0), torch.stack(grid_thws)
