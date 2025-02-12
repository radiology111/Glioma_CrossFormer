import os
import time
import warnings
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter, NativeScaler

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_checkpoint_only, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

warnings.filterwarnings("ignore", module="PIL")

def parse_option():
    parser = argparse.ArgumentParser('CrossFormer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str,  metavar="FILE", help='path to config file', default='./configs/crossformer/base_patch4_group7_224.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
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
    parser.add_argument('--lr', type=float, default=5e-4, help="max learning rate for training")
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
    # create token_label dataset
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    # logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config, args)
    model.cuda()
    # logger.info(str(model))

    optimizer = build_optimizer(config, model)
    if config.AMP_OPT_LEVEL != "O0" and config.AMP_OPT_LEVEL != "native":
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    model_without_ddp = model.module
    loss_scaler = NativeScaler()

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint_only(config, model_without_ddp, optimizer, lr_scheduler, logger)
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.WEIGHT_OUTPUT)
        if resume_file:
            if config.MODEL.RESUME and os.path.getmtime(config.MODEL.RESUME) >= os.path.getmtime(resume_file):
                pass
            else:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
                config.defrost()
                config.MODEL.RESUME = resume_file
                config.freeze()
                logger.info(f'auto resuming from {resume_file}')
                max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
                acc1, acc5, loss = validate(config, data_loader_val, model)
                logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
                if config.EVAL_MODE:
                    return
        else:
            logger.info(f'no checkpoint found in {config.WEIGHT_OUTPUT}, ignoring auto resume')

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    if config.MODEL.FROM_PRETRAIN:
        config.defrost()
        config.MODEL.RESUME = config.MODEL.FROM_PRETRAIN
        config.freeze()
        load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler)
        # if dist.get_rank() == 0 and epoch == config.TRAIN.EPOCHS - 1:
        save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger)
        # if config.DATA.DATASET != "ImageNet22K" or epoch % 10 == 0 or epoch == config.TRAIN.EPOCHS - 1:
        #     acc1, acc5, loss = validate(config, data_loader_val, model, epoch)
        #     logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        #     if dist.get_rank() == 0 and acc1 >= max_accuracy: ## save best
        #         save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger, best=True)
        #     if dist.get_rank() == 0:
        #         save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger, last=True)
        #     max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Epoch: {epoch:d}, Max accuracy: {max_accuracy:.2f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=(config.AMP_OPT_LEVEL=="native")):
            outputs = model(samples)
            loss = criterion(outputs, targets)

            if config.TRAIN.ACCUMULATION_STEPS > 1:
                loss = loss / config.TRAIN.ACCUMULATION_STEPS
                if config.AMP_OPT_LEVEL != "O0" and config.AMP_OPT_LEVEL != "native":
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    if config.TRAIN.CLIP_GRAD:
                        grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                    else:
                        grad_norm = get_grad_norm(amp.master_params(optimizer))
                else:
                    loss.backward()
                    if config.TRAIN.CLIP_GRAD:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                    else:
                        grad_norm = get_grad_norm(model.parameters())
                if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step_update(epoch * num_steps + idx)
            else:
                optimizer.zero_grad()
                if config.AMP_OPT_LEVEL != "O0" and config.AMP_OPT_LEVEL != "native":
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    if config.TRAIN.CLIP_GRAD:
                        grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                    else:
                        grad_norm = get_grad_norm(amp.master_params(optimizer))
                    optimizer.step()
                elif config.AMP_OPT_LEVEL == "native":
                    loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD, parameters=model.parameters())
                    grad_norm = 0
                    for p in model.parameters():
                        param_norm = p.grad.data.norm(2)
                        grad_norm += param_norm.item() ** 2
                    grad_norm = grad_norm ** 0.5
                else:
                    loss.backward()
                    if config.TRAIN.CLIP_GRAD:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                    else:
                        grad_norm = get_grad_norm(model.parameters())
                    optimizer.step()
                lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), samples.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0 or idx == len(data_loader) - 1:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}], '
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}, '
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f}), '
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}), '
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f}), '
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model, epoch=0):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=(config.AMP_OPT_LEVEL=="native")):
            output = model(images)

            # measure accuracy and record loss
            loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0 or idx == len(data_loader) - 1:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Epoch {epoch:d}\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    args, config = parse_option()

    if config.AMP_OPT_LEVEL != "O0" and config.AMP_OPT_LEVEL != "native":
        assert amp is not None, "amp not installed!"

    # if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
    #     rank = int(os.environ["RANK"])
    #     world_size = int(os.environ['WORLD_SIZE'])
    #     print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    # else:
    #     rank = -1
    #     world_size = -1
    # torch.cuda.set_device(config.LOCAL_RANK)
    # torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    # torch.distributed.barrier()

    seed = config.SEED #+ dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # # linear scale the learning rate according to total batch size, may not be optimal
    # linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0


    # gradient accumulation also need to scale the learning rate
    # if config.TRAIN.ACCUMULATION_STEPS > 1:
    #     linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
    #     linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
    # #     linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    # config.defrost()
    # config.TRAIN.BASE_LR = linear_scaled_lr
    # config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    # config.TRAIN.MIN_LR = linear_scaled_min_lr
    # config.freeze()

    os.makedirs(config.LOG_OUTPUT,    exist_ok=True)
    os.makedirs(config.WEIGHT_OUTPUT, exist_ok=True)
    # logger = create_logger(output_dir=config.LOG_OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")
    logger = create_logger(output_dir=config.LOG_OUTPUT, dist_rank=0, name=f"{config.MODEL.NAME}")


    # if dist.get_rank() == 0:
    path = os.path.join(config.LOG_OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(args, config)
