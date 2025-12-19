#! /usr/bin/env python3
#
# Copyright © 2025 Standard Model Biomedicine, Inc. <zach@standardmodel.bio>
#
# Distributed under terms of the Apache License 2.0 license.

import torch

from .ehr_process import process_ehr_info
from .imaging_process import extract_imaging_info, fetch_medical_volume, process_imaging_info
from .imaging_text_pairs import create_imaging_text_pairs


def process_mm_info(conversations: list[dict] | list[list[dict]]) -> tuple[torch.Tensor, torch.Tensor] | None:
    imaging, grid_thw = process_imaging_info(conversations)
    return imaging, grid_thw


__all__ = [
    "fetch_medical_volume",
    "extract_imaging_info",
    "process_imaging_info",
    "process_mm_info",
    "process_ehr_info",
    "create_imaging_text_pairs",
]
