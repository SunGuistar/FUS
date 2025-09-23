import torch,os,csv
from scipy import stats
from utils.visualizer import Visualizer_plot_volume

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from matplotlib import cm
from utils.loader import SSFrameDataset
from options.train_options import TrainOptions
from utils.utils import str2list

# 像素到毫米转换比例（基于校准矩阵）
PIXEL_TO_MM_X = 0.229389190673828
PIXEL_TO_MM_Y = 0.220979690551758
PIXEL_TO_MM_Z = 1.0
PIXEL_TO_MM_AVG = (PIXEL_TO_MM_X + PIXEL_TO_MM_Y) / 2  # 用于全局距离转换

def convert_pixel_to_mm(pixel_value, is_global=True):
    """
    将像素值转换为毫米值
    Args:
        pixel_value: 像素值
        is_global: 是否为全局距离（True）或局部距离（False）
    Returns:
        毫米值
    """
    if is_global:
        return pixel_value * PIXEL_TO_MM_AVG
    else:
        return pixel_value * PIXEL_TO_MM_Z  # 局部距离已经是毫米单位

from utils.utils_grid_data import *
from utils.utils_meta import *
sys.path.append(os.getcwd()+'/utils')


opt = TrainOptions().parse()
opt_test = opt

writer = SummaryWriter(os.path.join(opt.SAVE_PATH))
os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu_ids
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_pairs = data_pairs_adjacent(opt.NUM_SAMPLES)
data_pairs=torch.tensor(data_pairs)

opt.FILENAME_VAL=opt.FILENAME_VAL+'_seqlen'+str(opt.NUM_SAMPLES)+'_'+opt.split_type+'_'+opt.train_set+'.json'
opt.FILENAME_TEST=opt.FILENAME_TEST+'_seqlen'+str(opt.NUM_SAMPLES)+'_'+opt.split_type+'_'+opt.train_set+'.json'
opt.FILENAME_TRAIN=[opt.FILENAME_TRAIN[i]+'_seqlen'+str(opt.NUM_SAMPLES)+'_'+opt.split_type+'_'+opt.train_set+'.json' for i in range(len(opt.FILENAME_TRAIN))]

dset_val = SSFrameDataset.read_json(Path(os.getcwd()).as_posix()+opt.DATA_PATH,opt.FILENAME_VAL,opt.h5_file_name,num_samples = -1)
dset_test = SSFrameDataset.read_json(Path(os.getcwd()).as_posix()+opt.DATA_PATH,opt.FILENAME_TEST,opt.h5_file_name,num_samples = -1)
dset_train_list = [SSFrameDataset.read_json(Path(os.getcwd()).as_posix()+opt.DATA_PATH,opt.FILENAME_TRAIN[i],opt.h5_file_name,num_samples = -1) for i in range(len(opt.FILENAME_TRAIN))]

dset_train = dset_train_list[0]+dset_train_list[1]+dset_train_list[2]
# dset_val = dset_val+dset_train_list[2]
print('using %s'%opt.h5_file_name)




viridis = cm.get_cmap('viridis', 10)

options = ['generate_reg_volume_data','reconstruction_vlume']

test_folders = 'models_all' # creat a new "models_all" folder, and move the generated folder after python trainXXX.py, into "models_all" folder
csv_name = 'metrics.csv'
fd_name_save = 'results'

folders = [f for f in os.listdir(os.getcwd()+'/'+test_folders) if f.startswith('seq_len')  and not os.path.isfile(os.path.join(os.getcwd()+'/'+test_folders, f))]
folders = sorted(folders)


# csv file
if not os.path.exists(os.getcwd()+'/'+test_folders+'/'+fd_name_save):
    os.makedirs(os.getcwd()+'/'+test_folders+'/'+fd_name_save)

# 定义规范的CSV列名（合并均值和标准差）
csv_headers = [
    'file_name',
    'model_name', 
    'global_T_all_points_mm',
    'global_T_R_all_points_mm',
    'global_T_4_points_mm',
    'global_T_R_4_points_mm',
    'local_T_all_points_mm',
    'local_T_4_points_mm'
]

# 创建CSV文件并写入表头
csv_path = os.path.join(os.getcwd(), test_folders, fd_name_save, csv_name)
with open(csv_path, 'w', newline='', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(csv_headers)

for sub_fd in folders:
    # get paramteters from folder name
    print(sub_fd)
    sub_fd_split = sub_fd.split('__')
    fn = sub_fd.split('__')[0]
    opt.NUM_SAMPLES = int(fn[len(fn.rstrip('0123456789')):])
    
    
    ind_inc = None
    for i in range(len(sub_fd_split)):
        if 'inc_reg' in sub_fd_split[i] or 'in_ch_reg' in sub_fd_split[i]:
            ind_inc = i
    try:
        opt.in_ch_reg = int(sub_fd_split[ind_inc].split('_')[-1])
        
    except:
            
        opt.in_ch_reg = 2
    print('opt.in_ch_reg = %d'%opt.in_ch_reg)

    opt.ddf_dirc = None
    if 'Move' in sub_fd:
        opt.ddf_dirc = 'Move'
    
    print('opt.ddf_dirc = %s'%opt.ddf_dirc)


    opt.saved_results = os.getcwd()+'/'+test_folders+'/'+sub_fd
    opt.SAVE_PATH = opt.saved_results

    models_all = [f for f in os.listdir(opt.SAVE_PATH+'/saved_model') if f.startswith('best_val_dist') and os.path.isfile(os.path.join(opt.SAVE_PATH+'/saved_model', f))]
    
    if len(models_all)==4:
        # non-meta training models
        models_name = [['best_val_dist_R_T','best_val_dist_R_R'],['best_val_dist_T_T','best_val_dist_T_R']]
    
    elif len(models_all)==8:
        # meta training models
        models_name = [['best_val_dist_R_R_T','best_val_dist_R_R_R'],\
                       ['best_val_dist_R_T_T','best_val_dist_R_T_R'],\
                        ['best_val_dist_T_R_T','best_val_dist_T_R_R'],\
                       ['best_val_dist_T_T_T','best_val_dist_T_T_R']]                 
    
    elif len(models_all)==0:
        # previous trained model
        models_pre = [f for f in os.listdir(opt.SAVE_PATH+'/saved_model') if f.startswith('best_validation') and os.path.isfile(os.path.join(opt.SAVE_PATH+'/saved_model', f))]
        if len(models_pre) ==4:
            models_name = [['best_validation_loss_model','best_validation_loss_model_reg'],\
                           ['best_validation_dist_model','best_validation_dist_model_reg']]
        elif len(models_pre) ==2:
            models_name = [['best_validation_loss_model'],['best_validation_dist_model']]

    else:
        raise('Not implenment')




    for i_m in range(len(models_name)):
        model_name = models_name[i_m]
        saved_folder_test = os.path.join(os.getcwd()+'/'+test_folders+'/'+fd_name_save+'/'+sub_fd, str(model_name)+'__TestSet')
        if not os.path.exists(saved_folder_test):
            os.makedirs(saved_folder_test)
    


        # visualizer_scan_train = Visualizer_plot_volume(opt,device, dset_train,model_name,data_pairs,options[1])
        # visualizer_scan_val = Visualizer_plot_volume(opt,device, dset_val,model_name,data_pairs,options[1])
        visualizer_scan_test = Visualizer_plot_volume(opt,device, dset_test,model_name,data_pairs,options[1])

        # test - test set
        for scan_index in range(len(dset_test)):
            visualizer_scan_test.generate_volume_data(
                                                        scan_index,
                                                        saved_folder_test,
                                                        based_volume = 'common_volume',# use common volume to reconstruct volume both for registartion use and for final vlulisation use
                                                        
                                                        )
        # save value for future use
        metric1 = np.array(visualizer_scan_test.T_Global_AllPts_Dist)[None,...]
        metric2 = np.array(visualizer_scan_test.T_R_Warp_Global_AllPts_Dist)[None,...]
        metric3 = np.array(visualizer_scan_test.T_Global_FourPts_Dist)[None,...]
        metric4 = np.array(visualizer_scan_test.T_R_Warp_Global_FourPts_Dist)[None,...]
        metric5 = np.array(visualizer_scan_test.T_Local_AllPts_Dist)[None,...]
        metric6 = np.array(visualizer_scan_test.T_Local_FourPts_Dist)[None,...]
        
        metrics = np.concatenate((metric1, metric2, metric3, metric4,metric5, metric6), axis=0)

        with open(os.getcwd()+'/'+test_folders+'/'+fd_name_save+'/'+sub_fd + '__' + str(model_name)+'.npy', 'wb') as f:
            np.save(f, metrics)
        
        
        # 计算毫米单位的指标
        # 全局T变换-所有点
        global_T_all_mean = convert_pixel_to_mm(np.array(visualizer_scan_test.T_Global_AllPts_Dist).mean(), True)
        global_T_all_std = convert_pixel_to_mm(np.array(visualizer_scan_test.T_Global_AllPts_Dist).std(), True)
        
        # 全局T+R变换-所有点
        global_T_R_all_mean = convert_pixel_to_mm(np.array(visualizer_scan_test.T_R_Warp_Global_AllPts_Dist).mean(), True)
        global_T_R_all_std = convert_pixel_to_mm(np.array(visualizer_scan_test.T_R_Warp_Global_AllPts_Dist).std(), True)
        
        # 全局T变换-4个点
        global_T_4_mean = convert_pixel_to_mm(np.array(visualizer_scan_test.T_Global_FourPts_Dist).mean(), True)
        global_T_4_std = convert_pixel_to_mm(np.array(visualizer_scan_test.T_Global_FourPts_Dist).std(), True)
        
        # 全局T+R变换-4个点
        global_T_R_4_mean = convert_pixel_to_mm(np.array(visualizer_scan_test.T_R_Warp_Global_FourPts_Dist).mean(), True)
        global_T_R_4_std = convert_pixel_to_mm(np.array(visualizer_scan_test.T_R_Warp_Global_FourPts_Dist).std(), True)
        
        # 局部T变换-所有点
        local_T_all_mean = convert_pixel_to_mm(np.array(visualizer_scan_test.T_Local_AllPts_Dist).mean(), False)
        local_T_all_std = convert_pixel_to_mm(np.array(visualizer_scan_test.T_Local_AllPts_Dist).std(), False)
        
        # 局部T变换-4个点
        local_T_4_mean = convert_pixel_to_mm(np.array(visualizer_scan_test.T_Local_FourPts_Dist).mean(), False)
        local_T_4_std = convert_pixel_to_mm(np.array(visualizer_scan_test.T_Local_FourPts_Dist).std(), False)
        
        # 准备CSV行数据（合并均值和标准差）
        row_data = [
            sub_fd,
            str(model_name),
            f"{global_T_all_mean:.2f}±{global_T_all_std:.2f}",
            f"{global_T_R_all_mean:.2f}±{global_T_R_all_std:.2f}",
            f"{global_T_4_mean:.2f}±{global_T_4_std:.2f}",
            f"{global_T_R_4_mean:.2f}±{global_T_R_4_std:.2f}",
            f"{local_T_all_mean:.2f}±{local_T_all_std:.2f}",
            f"{local_T_4_mean:.2f}±{local_T_4_std:.2f}"
        ]
        
        # 写入CSV
        with open(csv_path, 'a', newline='', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)
        
        # 打印结果
        print(f"  全局T变换-所有点: {global_T_all_mean:.2f}±{global_T_all_std:.2f} mm")
        print(f"  全局T+R变换-所有点: {global_T_R_all_mean:.2f}±{global_T_R_all_std:.2f} mm")
        print(f"  局部T变换-所有点: {local_T_all_mean:.2f}±{local_T_all_std:.2f} mm")

print(f"\n测试完成！结果已保存到: {csv_path}")
print(f"转换比例:")
print(f"  全局距离: {PIXEL_TO_MM_AVG:.6f} mm/pixel")
print(f"  局部距离: {PIXEL_TO_MM_Z:.1f} mm/pixel")
print(f"CSV格式:")
print(f"  - 使用标准CSV格式（逗号分隔）")
print(f"  - 均值和标准差合并显示（格式：mean±std）")
print(f"  - 所有距离单位转换为毫米")
print(f"  - 列名简洁明确")


