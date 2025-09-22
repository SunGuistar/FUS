#!/usr/bin/env python3
"""Multi-H5 Dataset Loader utilities for FUS project."""

import os
from pathlib import Path
from utils.loader import SSFrameDataset

def load_multi_h5_dataset(data_path, json_filename, h5_file_names, num_samples=None):
    """Load dataset from multiple H5 files."""
    if isinstance(h5_file_names, str):
        h5_file_names = [h5_file_names]
    
    datasets = []
    for h5_file_name in h5_file_names:
        try:
            dataset = SSFrameDataset.read_json(data_path, json_filename, h5_file_name, num_samples)
            datasets.append(dataset)
            print(f"  Loaded data from {h5_file_name}: {len(dataset)} samples")
        except Exception as e:
            print(f"  Warning: Failed to load {h5_file_name}: {e}")
            continue
    
    if not datasets:
        raise ValueError("No valid datasets loaded!")
    
    # Combine all datasets
    combined_dataset = datasets[0]
    for dataset in datasets[1:]:
        combined_dataset = combined_dataset + dataset
    
    return combined_dataset

def get_h5_file_mapping(use_dataAll=True):
    """Get mapping from fold to H5 files."""
    if use_dataAll:
        return {
            "fold_00": ["scans_res0_forth.h5"],
            "fold_01": ["scans_res1_forth.h5"], 
            "fold_02": ["scans_res2_forth.h5"],
            "fold_03": ["scans_res3_forth.h5"],
            "fold_04": ["scans_res4_forth.h5"],
        }
    else:
        return {
            "fold_00": ["scans_res4_forth.h5"],
            "fold_01": ["scans_res4_forth.h5"],
            "fold_02": ["scans_res4_forth.h5"],
            "fold_03": ["scans_res4_forth.h5"],
            "fold_04": ["scans_res4_forth.h5"],
        }

