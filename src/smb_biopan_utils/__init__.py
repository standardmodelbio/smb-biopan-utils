#! /usr/bin/env python3
#
# Copyright © 2025 Standard Model Biomedicine, Inc. <zach@standardmodel.bio>
#
# Distributed under terms of the Apache License 2.0 license.
import torch

from .imaging_process import extract_imaging_info, fetch_medical_volume, process_imaging_info


def process_mm_info(conversations: list[dict] | list[list[dict]]) -> list[torch.Tensor] | None:
    imaging = process_imaging_info(conversations)
    return imaging


__all__ = ["fetch_medical_volume", "extract_imaging_info", "process_imaging_info", "process_mm_info"]
