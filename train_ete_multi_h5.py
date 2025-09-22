

# train reconstruction and registration network at the same time.
# use one optimiser, train together
# loss is the sum of reconstruction loss and registartion loss 

import os
import torch,sys
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from utils.loader import SSFrameDataset
from options.train_options import TrainOptions
from utils.utils_grid_data import *
from utils.utils_ete import *

sys.path.append(os.getcwd()+'/utils')


opt = TrainOptions().parse()
writer = SummaryWriter(os.path.join(opt.SAVE_PATH))

# if not opt.multi_gpu:
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



opt.FILENAME_VAL=opt.FILENAME_VAL+'_seqlen'+str(opt.NUM_SAMPLES)+'_'+opt.split_type+'_'+opt.train_set+'.json'
opt.FILENAME_TEST=opt.FILENAME_TEST+'_seqlen'+str(opt.NUM_SAMPLES)+'_'+opt.split_type+'_'+opt.train_set+'.json'
opt.FILENAME_TRAIN=[opt.FILENAME_TRAIN[i]+'_seqlen'+str(opt.NUM_SAMPLES)+'_'+opt.split_type+'_'+opt.train_set+'.json' for i in range(len(opt.FILENAME_TRAIN))]

# Multi-H5 Dataset Loading Support
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
    combined_dataset = datasets[0]
    for dataset in datasets[1:]:
        combined_dataset = combined_dataset + dataset
    return combined_dataset

# Check if we should use multi-H5 format
dataAll_path = Path(os.getcwd()) / "data"
use_dataAll = dataAll_path.exists() and any(dataAll_path.glob("scans_res*_forth.h5"))

if use_dataAll:
    print("=== Using Multi-H5 Format from data directory ===")
    opt.DATA_PATH = "/data"  # Switch to dataAll directory
    
    # Define H5 file mapping
    h5_mapping = {
        "fold_00": ["scans_res0_forth.h5"],  # Training fold 0
        "fold_01": ["scans_res1_forth.h5"],  # Training fold 1
        "fold_02": ["scans_res2_forth.h5"],  # Training fold 2
        "fold_03": ["scans_res3_forth.h5"],  # Validation fold
        "fold_04": ["scans_res4_forth.h5"],  # Test fold
    }
    
    data_path = Path(os.getcwd()).as_posix() + opt.DATA_PATH
    
    # Load training data from multiple folds (fold_00, fold_01, fold_02)
    print("Loading Training Data:")
    train_datasets = []
    for i, train_filename in enumerate(opt.FILENAME_TRAIN):
        fold_key = f"fold_{i:02d}"
        h5_files = h5_mapping.get(fold_key, [opt.h5_file_name])
        print(f"  Loading {train_filename} from {h5_files}")
        dataset = load_multi_h5_dataset(data_path, train_filename, h5_files)
        train_datasets.append(dataset)
    dset_train = train_datasets[0]
    for dataset in train_datasets[1:]:
        dset_train = dset_train + dataset
    
    # Load validation data (fold_03)
    print("Loading Validation Data:")
    val_h5_files = h5_mapping.get("fold_03", [opt.h5_file_name])
    print(f"  Loading {opt.FILENAME_VAL} from {val_h5_files}")
    dset_val = load_multi_h5_dataset(data_path, opt.FILENAME_VAL, val_h5_files)
    
    # Load test data (fold_04)
    print("Loading Test Data:")
    test_h5_files = h5_mapping.get("fold_04", [opt.h5_file_name])
    print(f"  Loading {opt.FILENAME_TEST} from {test_h5_files}")
    dset_test = load_multi_h5_dataset(data_path, opt.FILENAME_TEST, test_h5_files)
    
    print(f"Dataset Summary:")
    print(f"  Training samples: {len(dset_train)}")
    print(f"  Validation samples: {len(dset_val)}")
    print(f"  Test samples: {len(dset_test)}")
    
else:
    print("=== Using Legacy Single H5 Format ===")
    dset_val = SSFrameDataset.read_json(Path(os.getcwd()).as_posix()+opt.DATA_PATH,opt.FILENAME_VAL,opt.h5_file_name)
    dset_test = SSFrameDataset.read_json(Path(os.getcwd()).as_posix()+opt.DATA_PATH,opt.FILENAME_TEST,opt.h5_file_name)
    dset_train_list = [SSFrameDataset.read_json(Path(os.getcwd()).as_posix()+opt.DATA_PATH,opt.FILENAME_TRAIN[i],opt.h5_file_name) for i in range(len(opt.FILENAME_TRAIN))]
    dset_train = dset_train_list[0]+dset_train_list[1]+dset_train_list[2]
    print("using %s"%opt.h5_file_name)
            

train_rec_reg_model = Train_Rec_Reg_Model(opt = opt,
                        non_improve_maxmum = 1e10, 
                        reg_loss_weight = 1000,
                        val_loss_min = 1e10,
                        val_dist_min = 1e10,
                        val_loss_min_reg = 1e10,
                        dset_train = dset_train,
                        dset_val = dset_val,
                        dset_train_reg = None,
                        dset_val_reg = None,
                        device = device,
                        writer = writer,
                        option = 'common_volume')



# load pre-trained model
# train_rec_reg_model.load_recon_model_initial()
# train_rec_reg_model.load_def_model_initial()

train_rec_reg_model.multi_model()
train_rec_reg_model.train_rec_model()

