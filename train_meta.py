

import os
import torch
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from utils.loader import SSFrameDataset
from options.train_options import TrainOptions
from utils.utils_grid_data import *
from utils.utils_meta import *
import sys
sys.path.append(os.getcwd()+'/utils')

opt = TrainOptions().parse()
writer = SummaryWriter(os.path.join(opt.SAVE_PATH))
# if not opt.multi_gpu:
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load datasets directly without JSON files - use all available data
dset_val = SSFrameDataset(
    min_scan_len=opt.MIN_SCAN_LEN,
    data_path=Path(os.getcwd()).as_posix()+opt.DATA_PATH,
    h5_file_name=opt.h5_file_name,
    indices_in_use=None,  # Use all available data
    num_samples=opt.NUM_SAMPLES,
    sample_range=opt.SAMPLE_RANGE,
    split_type='val'
)

dset_test = SSFrameDataset(
    min_scan_len=opt.MIN_SCAN_LEN,
    data_path=Path(os.getcwd()).as_posix()+opt.DATA_PATH,
    h5_file_name=opt.h5_file_name,
    indices_in_use=None,  # Use all available data
    num_samples=opt.NUM_SAMPLES,
    sample_range=opt.SAMPLE_RANGE,
    split_type='test'
)

dset_train_list = []
for i in range(len(opt.FILENAME_TRAIN)):
    dset_train = SSFrameDataset(
        min_scan_len=opt.MIN_SCAN_LEN,
        data_path=Path(os.getcwd()).as_posix()+opt.DATA_PATH,
        h5_file_name=opt.h5_file_name,
        indices_in_use=None,  # Use all available data
        num_samples=opt.NUM_SAMPLES,
        sample_range=opt.SAMPLE_RANGE,
        split_type='train'
    )
    dset_train_list.append(dset_train)
dset_train = dset_train_list[0]+dset_train_list[1]
dset_val = dset_val+dset_train_list[2]
print('使用新的数据加载方式：从train/test/val目录读取多个h5文件')
print('训练集扫描数: %d, 验证集扫描数: %d, 测试集扫描数: %d' % (dset_train.num_scans, dset_val.num_scans, dset_test.num_scans))
               

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
# train_rec_reg_model.load_rec_model_initial()
# train_rec_reg_model.load_reg_model_initial()

train_rec_reg_model.multi_model()
train_rec_reg_model.train_rec_model()



