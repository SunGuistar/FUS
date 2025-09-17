#!/usr/bin/env python3
"""
Quick verification script for calibration matrix reading.
"""

import os
import sys
import torch

# Add current directory to path
sys.path.append(os.getcwd())

from data.calib import read_calib_matrices

def verify_calib_files():
    """Verify that calibration files can be read correctly."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resample_factor = 4
    
    files_to_test = [
        'data/calib_matrix.csv',
        'data/calib_matrix_test.csv'
    ]
    
    print("=== Calibration Matrix Verification ===\n")
    
    for filename in files_to_test:
        if os.path.exists(filename):
            try:
                tform_calib_scale, tform_calib_R_T, tform_calib = read_calib_matrices(
                    filename, resample_factor, device
                )
                print(f"✅ {filename}: Successfully loaded")
                print(f"   Scale matrix: {tform_calib_scale.shape}")
                print(f"   R_T matrix: {tform_calib_R_T.shape}")
                print(f"   Combined matrix: {tform_calib.shape}")
            except Exception as e:
                print(f"❌ {filename}: Error - {str(e)}")
        else:
            print(f"⚠️  {filename}: File not found")
        print()

if __name__ == "__main__":
    verify_calib_files()
