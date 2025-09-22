# Multi-H5 Dataset Support for FUS Project

## Overview

This implementation extends the original FUS project to support loading data from multiple H5 files based on fold configurations. This allows for better data organization and more flexible training setups.

## Key Features

- **Automatic Detection**: Automatically detects if multi-H5 format is available
- **Backward Compatibility**: Falls back to original single H5 file format if multi-H5 not available
- **Flexible Mapping**: Maps different folds to different H5 files
- **Error Handling**: Gracefully handles missing files with informative warnings

## File Structure

```
data/
├── scans_res0_forth.h5          # Training fold 0 data
├── scans_res1_forth.h5          # Training fold 1 data  
├── scans_res2_forth.h5          # Training fold 2 data
├── scans_res3_forth.h5          # Validation fold data
├── scans_res4_forth.h5          # Test fold data
├── fold_00_seqlen100_sub_forth.json  # Training fold 0 config
├── fold_01_seqlen100_sub_forth.json  # Training fold 1 config
├── fold_02_seqlen100_sub_forth.json  # Training fold 2 config
├── fold_03_seqlen100_sub_forth.json  # Validation fold config
└── fold_04_seqlen100_sub_forth.json  # Test fold config
```

## Modified Scripts

### Training Scripts
- `train_ete_multi_h5.py` - End-to-end training with multi-H5 support
- `train_meta_multi_h5.py` - Meta learning training with multi-H5 support

### Testing Script
- `test_multi_h5.py` - Testing with multi-H5 support

## Usage

### End-to-End Training
```bash
python3 train_ete_multi_h5.py --config config/config_ete.json
```

### Meta Learning Training  
```bash
python3 train_meta_multi_h5.py --config config/config_meta.json
```

### Testing
```bash
python3 test_multi_h5.py --config config/config_ete.json
```

## How It Works

### Automatic Detection
The scripts automatically detect if multi-H5 format is available by checking:
1. If `data/` directory exists
2. If `scans_res*_forth.h5` files exist in the directory

### Fold to H5 File Mapping
```python
h5_mapping = {
    "fold_00": ["scans_res0_forth.h5"],  # Training fold 0
    "fold_01": ["scans_res1_forth.h5"],  # Training fold 1  
    "fold_02": ["scans_res2_forth.h5"],  # Training fold 2
    "fold_03": ["scans_res3_forth.h5"],  # Validation fold
    "fold_04": ["scans_res4_forth.h5"],  # Test fold
}
```

### Data Loading Strategy

#### End-to-End Training (`train_ete_multi_h5.py`)
- **Training**: Combines data from fold_00, fold_01, fold_02
- **Validation**: Uses data from fold_03  
- **Testing**: Uses data from fold_04

#### Meta Learning Training (`train_meta_multi_h5.py`)
- **Training**: Uses data from fold_00, fold_01
- **Validation**: Combines data from fold_02, fold_03
- **Testing**: Uses data from fold_04

## Backward Compatibility

If multi-H5 files are not detected, the scripts automatically fall back to the original single H5 file format:

```
=== Using Legacy Single H5 Format ===
using scans_res4_forth.h5
```

## Error Handling

The implementation includes robust error handling:
- Warns if individual H5 files cannot be loaded
- Continues with available files
- Fails gracefully if no valid datasets are found

## Benefits

1. **Better Data Organization**: Each fold has its own H5 file
2. **Memory Efficiency**: Only loads required data for each fold
3. **Flexibility**: Easy to modify fold assignments
4. **Scalability**: Can easily add more folds or H5 files
5. **Debugging**: Easier to identify issues with specific folds

## Migration from Single H5

The multi-H5 format is fully backward compatible. To migrate:

1. Generate multi-H5 files using `generate_folds_and_h5.py`
2. Place files in `data/` directory
3. Use the new `*_multi_h5.py` scripts
4. Original scripts continue to work with single H5 files

## Example Output

```
=== Using Multi-H5 Format from data directory ===
Loading Training Data:
  Loading fold_00_seqlen100_sub_forth.json from ['scans_res0_forth.h5']
  Loaded data from scans_res0_forth.h5: 240 samples
  Loading fold_01_seqlen100_sub_forth.json from ['scans_res1_forth.h5']  
  Loaded data from scans_res1_forth.h5: 240 samples
  Loading fold_02_seqlen100_sub_forth.json from ['scans_res2_forth.h5']
  Loaded data from scans_res2_forth.h5: 240 samples

Loading Validation Data:
  Loading fold_03_seqlen100_sub_forth.json from ['scans_res3_forth.h5']
  Loaded data from scans_res3_forth.h5: 240 samples

Loading Test Data:
  Loading fold_04_seqlen100_sub_forth.json from ['scans_res4_forth.h5']
  Loaded data from scans_res4_forth.h5: 240 samples

Dataset Summary:
  Training samples: 720
  Validation samples: 240  
  Test samples: 240
```
