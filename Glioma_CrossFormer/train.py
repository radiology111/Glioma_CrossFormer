import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' #指定gpu盘号

import time
import warnings
import argparse
import datetime
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch
import torch.backends.cudnn as cudnn
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter, NativeScaler
from torch.cuda.amp import GradScaler
from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_checkpoint_only, save_checkpoint
from datetime import datetime

try:
    from apex import amp
except ImportError:
    amp = None

warnings.filterwarnings("ignore", module="PIL")

def parse_option():
    parser = argparse.ArgumentParser('CrossFormer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str,  metavar="FILE", help='path to config file', default='./configs/crossformer/tiny_patch4_group7_224.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification#
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-set', type=str, default='imagenet', help='dataset to use')
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='native', choices=['native', 'O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', default='debug', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--num_workers', type=int, default=8, help="")
    parser.add_argument('--warmup_epochs', type=int, default=20, help="#epoches for warm up")
    parser.add_argument('--epochs', type=int, default=300, help="#epoches")
    parser.add_argument('--lr', type=float, default=5e-3, help="max learning rate for training")
    parser.add_argument('--min_lr', type=float, default=5e-6, help="min learning rate for training")
    parser.add_argument('--warmup_lr', type=float, default=5e-7, help="learning rate to start warmup")
    parser.add_argument('--weight_decay', type=float, default=5e-2, help="l2 reguralization")

    # local rank is obtained using os.environ in newr version
    # parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    parser.add_argument("--img_size", type=int, default=224, help='input resolution for image')
    parser.add_argument("--embed_dim", type=int, nargs='+', default=None, help='size of embedding')
    parser.add_argument("--impl_type", type=str, default='', help='options to use for different methods')

    # arguments relevant to our experiment
    parser.add_argument('--group_type', type=str, default='constant', help='group size type')
    parser.add_argument('--use_cpe', action='store_true', help='whether to use conditional positional encodings')
    parser.add_argument('--pad_type', type=int, default=0, help='0 to pad in one direction, otherwise 1')
    parser.add_argument('--no_mask', action='store_true', help='whether to use mask after padding')
    parser.add_argument('--adaptive_interval', action='store_true', help='interval change with the group size')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(args, config):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    #初始化模型
    model = build_model(config, args)
    model.cuda()
    # logger.info(str(model))
    #选择优化器 损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAIN.LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0

    logger.info("Start training")
    start_time = time.time()
    #创建权重文件保存路径
    now_time = datetime.now()
    time_str = now_time.strftime('%Y-%m-%d_%H-%M')
    save_dir=os.path.join(config.WEIGHT_OUTPUT,time_str)
    os.makedirs(save_dir,exist_ok=True)
    save_path=save_dir+ '/ckpt_best.pt'



    for epoch in range(config.TRAIN.EPOCHS):

        #训练阶段
        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn)
        #验证阶段
        acc1, acc5, val_loss = validate(config, data_loader_val, model)

        #验证精度高于最优精度 保存权重
        if acc1>=max_accuracy:
            max_accuracy=acc1
            max_epoch=epoch
            save_checkpoint(config, epoch, model, max_accuracy, optimizer,  logger,save_path)
        # logger.info(f'Epoch: {epoch:d},val_accuracy: {acc1:.1f}%,val_loss{val_loss},Max val_accuracy: {max_accuracy:.2f}%, Best epoch:{max_epoch} ')
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn):
    model.train()
    optimizer.zero_grad()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()

    start = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        end = time.time()

        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        targets = F.one_hot(targets, num_classes=2).float()
        optimizer.zero_grad()

        outputs = model(samples)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), samples.size(0))
        batch_time.update(time.time() - end)


    memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    epoch_time = time.time() - start

    logger.info(
        f'Train: [{epoch}/{config.TRAIN.EPOCHS}], '
        f'time {epoch_time:.2f} , '
        f'loss {loss_meter.val:.4f} , '
        f'mem {memory_used:.0f}MB')




@torch.no_grad()
def validate(config, data_loader, model, epoch=0):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output = model(images)
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        loss_meter.update(loss.item(), images.size(0))
        acc1_meter.update(acc1.item(), images.size(0))
        acc5_meter.update(acc5.item(), images.size(0))

        # if idx % config.PRINT_FREQ == 0 or idx == len(data_loader) - 1:
        memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    logger.info(
        'Validate:'
        f'Epoch {epoch:d}\t'
        f'Loss {loss_meter.val:.3f} ({loss_meter.avg:.3f})\t'
        f'Acc(Acc_average) {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
        f'Mem {memory_used:.0f}MB')
    # logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg

if __name__ == '__main__':
    args, config = parse_option()
    if config.AMP_OPT_LEVEL != "O0" and config.AMP_OPT_LEVEL != "native":
        assert amp is not None, "amp not installed!"
    seed = config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    #创建log文件
    os.makedirs(config.LOG_OUTPUT, exist_ok=True)
    os.makedirs(config.WEIGHT_OUTPUT, exist_ok=True)
    
    logger = create_logger(output_dir=config.LOG_OUTPUT, name=f"{config.MODEL.NAME}")
    path = os.path.join(config.LOG_OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")
    logger.info(config.dump())
    main(args, config)