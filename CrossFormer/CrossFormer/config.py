import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files选择模型文件
_C.BASE = ['/storage/c_xbw/project/CrossFormer/configs/crossformer/base_patch4_group7_224.yaml']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()

# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 64
# Path to dataset, 数据集路径和标签文件路径
_C.DATA.DATA_PATH = '/storage/c_xbw/Datasets/crossformer/crops'
_C.DATA.TRAIN_idx= '/storage/c_xbw/Datasets/crossformer/label_idx/train.txt'
_C.DATA.VAL_idx = '/storage/c_xbw/Datasets/crossformer/label_idx/val.txt'
# Dataset name
_C.DATA.DATASET = 'crops'
# Input image size#图像尺寸
_C.DATA.IMG_SIZE = 256
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.ZIP_MODE = False
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = 'part'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8
# Whether to use token labeling and settings
_C.DATA.TOKEN_LABEL = False
_C.DATA.TOKEN_LABEL_SIZE = 7
_C.DATA.PREFETCH = False
_C.DATA.MIX_GROUNDTRUTH = False
_C.DATA.CLS_WEIGHT = 1.0
_C.DATA.DENSE_WEIGHT = 0.5

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'cross-scale'
_C.MODEL.IMPL_TYPE = ''
_C.MODEL.CONV_BLOCKS = ''
# Model name
_C.MODEL.NAME = 'tiny_patch4_group7_224'
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
_C.MODEL.FROM_PRETRAIN = ''
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 2
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1
_C.MODEL.RETURN_DENSE = True
_C.MODEL.MIX_TOKEN = True

# CrossFormer parameters
_C.MODEL.CROS = CN()
_C.MODEL.CROS.PATCH_SIZE = [4, 8, 16, 32]
_C.MODEL.CROS.MERGE_SIZE = [[2, 4], [2,4], [2, 4]]
_C.MODEL.CROS.IN_CHANS = 6
_C.MODEL.CROS.EMBED_DIM = 48
_C.MODEL.CROS.DEPTHS = [2, 2, 6, 2]
_C.MODEL.CROS.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.CROS.GROUP_SIZE = [7, 7, 7, 7]
_C.MODEL.CROS.INTERVAL = [8, 4, 2, 1]
_C.MODEL.CROS.MLP_RATIO = [4., 4., 4., 4.]
_C.MODEL.CROS.QKV_BIAS = True
_C.MODEL.CROS.QK_SCALE = None
_C.MODEL.CROS.APE = False
_C.MODEL.CROS.PATCH_NORM = True

# CrossFormer++ parameters
_C.MODEL.CROS.GROUP_TYPE = 'constant'
_C.MODEL.CROS.USE_ACL = True
_C.MODEL.CROS.USE_CPE = False
_C.MODEL.CROS.PAD_TYPE = 0
_C.MODEL.CROS.NO_MASK = False
_C.MODEL.CROS.ADAPT_INTER = False

# Loss settings for supervision on intermediate feature maps
_C.MODEL.LOSS = CN()
_C.MODEL.LOSS.ALPHA2 = 0.1
_C.MODEL.LOSS.ALPHA3 = 0.1
_C.MODEL.LOSS.ALPHA4 = 0.25 # impl_type == 15

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
#学习率
_C.TRAIN.LEARNING_RATE = 1e-3

_C.TRAIN.BASE_LR = 2e-3
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 0
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = 'batch'

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Mixed precision opt level, if O0, no amp is used ('O0', 'O1', 'O2')
# overwritten by command line argument
_C.AMP_OPT_LEVEL = ''
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Divide log and model into different directory
_C.LOG_OUTPUT    = ''
_C.WEIGHT_OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 1000
# Frequency to logging info
_C.PRINT_FREQ = 50
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    # merge from specific arguments
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.data_path:
        config.DATA.DATA_PATH = args.data_path
    if args.zip:
        config.DATA.ZIP_MODE = True
    if args.cache_mode:
        config.DATA.CACHE_MODE = args.cache_mode
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.accumulation_steps:
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if args.use_checkpoint:
        config.TRAIN.USE_CHECKPOINT = True
    if args.amp_opt_level:
        config.AMP_OPT_LEVEL = args.amp_opt_level
    if args.output:
        config.OUTPUT = args.output
    if args.tag:
        config.TAG = args.tag
    if args.eval:
        config.EVAL_MODE = True
    if args.num_workers >= 0:
        config.DATA.NUM_WORKERS = args.num_workers
    if args.throughput:
        config.THROUGHPUT_MODE = True
    if args.embed_dim:
        config.MODEL.CROS.EMBED_DIM = args.embed_dim

    # if args.patch_size:
    #     config.MODEL.CROS.PATCH_SIZE = args.patch_size

    # config.MODEL.MERGE_SIZE_AFTER = [args.merge_size_after1, args.merge_size_after2, args.merge_size_after3, []]
    config.DATA.DATASET = args.data_set
    config.DATA.IMG_SIZE = args.img_size
    config.TRAIN.WARMUP_EPOCHS = args.warmup_epochs
    config.TRAIN.EPOCHS = args.epochs
    config.TRAIN.WEIGHT_DECAY = args.weight_decay
    config.TRAIN.BASE_LR = args.lr
    config.TRAIN.WARMUP_LR = args.warmup_lr
    config.TRAIN.MIN_LR = args.min_lr
    config.MODEL.IMPL_TYPE = args.impl_type
    # set local rank for distributed training
    # config.LOCAL_RANK = args.local_rank
    # config.LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    # output folder
    # config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)
    # config.LOG_OUTPUT    = os.path.join(config.OUTPUT, 'log',    config.TAG, config.MODEL.NAME)
    # config.WEIGHT_OUTPUT = os.path.join(config.OUTPUT, 'weight', config.TAG, config.MODEL.NAME)
    config.LOG_OUTPUT    = os.path.join(config.OUTPUT, 'log',    config.TAG)
    config.WEIGHT_OUTPUT = os.path.join(config.OUTPUT, 'weight', config.TAG)

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    # local_rank = os.getenv('LOCAL_RANK', '0')
    # config.LOCAL_RANK = int(local_rank)
    update_config(config, args)

    return config