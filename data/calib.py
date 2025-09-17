
import csv
import torch
import numpy as np
import os

def read_calib_matrices(filename_calib, resample_factor,device):
    # T{image->tool} = T{image_mm -> tool} * T{image_pix -> image_mm} * T{resampled_image_pix -> image_pix}
    tform_calib = np.empty((8,4), np.float32)
    
    with open(os.path.join(os.getcwd(),filename_calib)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        row_index = 0
        
        for row in csv_reader:
            # Skip header rows (non-numeric rows)
            if not row or not row[0].replace('.', '').replace('-', '').replace(',', '').isdigit():
                continue
            
            # Only process numeric rows
            if row_index < 8:  # We need exactly 8 rows of data
                try:
                    tform_calib[row_index,:] = list(map(float, row))
                    row_index += 1
                except ValueError:
                    # Skip rows that can't be converted to float
                    continue
    
    # Ensure we have exactly 8 rows of data
    if row_index != 8:
        raise ValueError(f"Expected 8 rows of numeric data, but found {row_index} rows in {filename_calib}")
    
    return torch.tensor(tform_calib[0:4,:],device=device),torch.tensor(tform_calib[4:8,:],device=device), torch.tensor(tform_calib[4:8,:] @ tform_calib[0:4,:] @ np.array([[resample_factor,0,0,0], [0,resample_factor,0,0], [0,0,1,0], [0,0,0,1]], np.float32),device=device)