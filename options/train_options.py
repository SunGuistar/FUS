from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        self.parser.add_argument('--PRED_TYPE', type=str,default='parameter',help='网络输出类型：{"transform", "parameter", "point","quaternion"}')
        self.parser.add_argument('--LABEL_TYPE', type=str,default='transform',help='标签类型：{"point", "parameter"}')

        self.parser.add_argument('--NUM_SAMPLES', type=int,default=100,help='输入帧数/图像数量')
        self.parser.add_argument('--SAMPLE_RANGE', type=int,default=100,help='输入帧数/图像的选择范围')
        self.parser.add_argument('--NUM_PRED', type=int,default=99,help='对这些帧/图像预测变换矩阵的数量')
        self.parser.add_argument('--sample', type=bool,default=True,help='False - 使用所有数据对进行训练；True - 仅使用采样的数据对进行训练')
        self.parser.add_argument('--MIN_SCAN_LEN', type=int, default=108,help='扫描长度大于此值的数据可用于训练和验证')
        self.parser.add_argument('--train_set', type=str, default='forth',help='loop: h5文件中的所有数据；forth: 仅前向数据；back: 仅后向数据；forth_back: 前向和后向数据分别处理')
        self.parser.add_argument('--split_type', type=str, default='sub',help='sub: 在受试者级别分割数据集；scan: 在扫描级别分割数据集')
        self.parser.add_argument('--Loss_type', type=str, default='rec_reg',help='MSE_points: 点的MSE损失;\
                                  Plane_norm: MSE损失和平面法向量的损失;\
                                  reg: 仅配准损失; rec_reg: 重建损失和配准损失;\
                                  rec_volume: 重建损失和体积损失; \
                                 rec_volume10000: 体积损失权重为10000; volume_only: 仅使用体积损失\
                                 wraped: 变形预测和真实值之间的MSE' )
        self.parser.add_argument('--intepoletion_method', type=str, default='bilinear',help='bilinear: 三轴差值相乘；IDW: 反距离加权')
        self.parser.add_argument('--Conv_Coords', type=str, default='optimised_coord',help='optimised_coord: 将体积转换为优化坐标；ori_coords: 不转换坐标' )
        self.parser.add_argument('--intepoletion_volume', type=str, default='fixed_interval',help='fixed_volume_size: 设置体积尺寸为128*128*128；fixed_interval: 设置体积间隔为1毫米' )
        self.parser.add_argument('--img_pro_coord', type=str, default='pro_coord',help='img_coord: 将坐标系转换为第一幅图像的图像坐标；pro_coord: 将坐标转换为第一幅图像的投影坐标系' )
        self.parser.add_argument('--in_ch_reg', type=int,default=1,help='配准网络的输入通道数')
        self.parser.add_argument('--ddf_dirc', type=str, default='Move',help='Move: 基于移动图像然后生成变形的固定图像；Fix: 基于固定图像然后生成变形的移动图像')


        self.parser.add_argument('--model_name', type=str,default='efficientnet_b1',help='网络名称：{"efficientnet_b1", "resnet", "LSTM_0", "LSTM", "LSTM_GT"}')
        self.parser.add_argument('--MAXNUM_PAIRS', type=int,default=50,help='保存到标量的最大变换对数量')
        self.parser.add_argument('--retain', type=bool,default=False,help='是否加载预训练模型')
        self.parser.add_argument('--retain_epoch', type=str,default='00000000',help='是否加载预训练模型：{0: 从头训练；数字，如1000，从第1000轮开始训练}')
        self.parser.add_argument('--MINIBATCH_SIZE_rec', type=int,default=1,help='重建网络的输入批次大小')
        self.parser.add_argument('--MINIBATCH_SIZE_reg', type=int,default=1,help='配准网络的输入批次大小')

        self.parser.add_argument('--LEARNING_RATE_rec',type=float,default=1e-4,help='重建网络的学习率')
        self.parser.add_argument('--LEARNING_RATE_reg',type=float,default=1e-4,help='配准网络的学习率')

        self.parser.add_argument('--NUM_EPOCHS',type =int,default=int(100),help='训练轮数')
        self.parser.add_argument('--max_rec_epoch_each_interation',type =int,default=int(2),help='每次迭代中训练重建或配准的最大轮数')
        self.parser.add_argument('--max_inter_rec_reg',type =int,default=int(500),help='训练重建-配准模型的最大迭代次数')
        self.parser.add_argument('--inter',type =str,default='iteratively',help='nointer/iteratively: 是否迭代训练')
        self.parser.add_argument('--meta',type =str,default='meta',help='meta: 使用验证集训练配准')
        self.parser.add_argument('--initial',type =str,default='noninitial',help='noninitial/InitialBest/InitialHalf: 初始化方式')
        self.parser.add_argument('--BatchNorm',type =str,default='BNoff',help='BNoff/BNon: 是否关闭批归一化')

        
        self.parser.add_argument('--FREQ_INFO', type=int, default=1,help='打印信息的频率')
        self.parser.add_argument('--FREQ_SAVE', type=int, default=100,help='保存模型的频率')
        self.parser.add_argument('--val_fre', type=int, default=1,help='验证的频率')
        #######used in testing##################################################
        self.parser.add_argument('--FILENAME_VAL', type=str, default="fold_03", help='验证JSON文件')
        self.parser.add_argument('--FILENAME_TEST', type=str, default="fold_04", help='测试JSON文件')
        self.parser.add_argument('--FILENAME_TRAIN', type=list,
                                 default=["fold_00", "fold_01", "fold_02"],
                                 help='训练JSON文件')
        self.parser.add_argument('--MODEL_FN', type=str, default="saved_model/", help='可视化用的模型路径')
        self.parser.add_argument('--use_bash_shell', type=bool, default=False,
                                 help='True: 使用bash shell脚本进行测试')

        self.isTrain= True
