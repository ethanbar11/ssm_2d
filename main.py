import wandb

from utils.memory_counter import print_model_memory_consumption
from utils.mix import cutmix_data, mixup_data, mixup_criterion
import numpy as np
import random
import logging as log
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from colorama import Fore, Style
from utils.losses import LabelSmoothingCrossEntropy
import os
from utils.sampler import RASampler
# import models.create_model as m
from utils.logger_dict import Logger_dict
from utils.print_progress import progress_bar
from utils.training_functions import accuracy
import argparse
from utils.scheduler import build_scheduler
from utils.dataloader import datainfo, dataload
from models.create_model import create_model
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings
from datetime import datetime

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

warnings.filterwarnings("ignore", category=Warning)

best_acc1 = 0
MODELS = ['vit', 'swin', 'pit', 'cait', 't2t', 'mega', 'convit', 'convnext', 'convnext-small']


def create_optimization_groups(model, args):
    decay = []
    no_decay = []
    # skip_list = ['pos_embedding','pos_embed']
    skip_list = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if hasattr(param, 'should_be_without_weight_decay') and param.should_be_without_weight_decay:
            no_decay.append(param)
        elif name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': args.weight_decay}]


def init_parser():
    parser = argparse.ArgumentParser(description='CIFAR10 quick training script')

    # Data args
    parser.add_argument('--data_path', default='./dataset', type=str, help='dataset path')
    parser.add_argument('--name', default=None, type=str, help='run name')
    parser.add_argument('--project', default='', type=str, help='project name')

    parser.add_argument('--dataset', default='CIFAR10',
                        choices=['CIFAR10', 'CIFAR100', 'CIFAR100-224px', 'T-IMNET', 'SVHN'], type=str,
                        help='Dataset')
    parser.add_argument('--ema', default=None, choices=['ema', 'ssm_2d', 's4nd', 'none', None], type=str,
                        help='EMA type')

    parser.add_argument('--s4nd_config', default=None, type=str,
                        help='S4nd config file path')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--print-freq', default=1, type=int, metavar='N', help='log frequency (by iteration)')
    parser.add_argument('--save_directory', default=None, type=str)
    parser.add_argument('--no_dropout_mega', default=False, type=bool)

    # Optimization hyperparams
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')

    parser.add_argument('--warmup', default=10, type=int, metavar='N', help='number of warmup epochs')

    parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)',
                        dest='batch_size')
    parser.add_argument('--use_mega_gating', default=False, type=bool)

    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')

    parser.add_argument('--weight-decay', default=5e-2, type=float, help='weight decay (default: 1e-4)')

    parser.add_argument('--model', type=str, default='vit', choices=MODELS)

    parser.add_argument('--disable-cos', action='store_true', help='disable cosine lr schedule')

    parser.add_argument('--enable_aug', action='store_true', help='disable augmentation policies for training')

    # parser.add_argument('--gpu', default=0, type=int)

    parser.add_argument('--no_cuda', action='store_true', help='disable cuda')

    parser.add_argument('--ls', action='store_false', help='label smoothing')

    parser.add_argument('--channel', type=int, help='disable cuda')

    parser.add_argument('--heads', type=int, help='disable cuda')

    parser.add_argument('--depth', type=int, help='disable cuda')

    parser.add_argument('--tag', type=str, help='tag', default='')

    parser.add_argument('--seed', type=int, default=0, help='seed')

    parser.add_argument('--sd', default=0.1, type=float, help='rate of stochastic depth')

    parser.add_argument('--resume', default=False, help='Version')

    parser.add_argument('--aa', action='store_false', help='Auto augmentation used'),

    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')

    parser.add_argument('--cm', action='store_false', help='Use Cutmix')

    # Mega params
    parser.add_argument('--embed_dim', type=int, default=192, help='seed')
    parser.add_argument('--zdim_ratio', type=float, default=None, help='seed')
    parser.add_argument('--hidden_dim_ratio', type=float, default=None, help='seed')
    parser.add_argument('--ffn_hidden_dim', type=int, default=192, help='seed')
    parser.add_argument('--ndim', type=int, default=16, help='seed')

    parser.add_argument('--beta', default=1.0, type=float,
                        help='hyperparameter beta (default: 1)')

    parser.add_argument('--mu', action='store_false', help='Use Mixup')

    parser.add_argument('--alpha', default=1.0, type=float,
                        help='mixup interpolation coefficient (default: 1)')

    parser.add_argument('--mix_prob', default=0.5, type=float,
                        help='mixup probability')

    parser.add_argument('--ra', type=int, default=3, help='repeated augmentation')

    parser.add_argument('--re', default=0.25, type=float, help='Random Erasing probability')

    parser.add_argument('--re_sh', default=0.4, type=float, help='max erasing area')

    parser.add_argument('--re_r1', default=0.3, type=float, help='aspect of erasing area')

    parser.add_argument('--is_LSA', action='store_true', help='Locality Self-Attention')

    parser.add_argument('--is_SPT', action='store_true', help='Shifted Patch Tokenization')

    parser.add_argument('--n_ssm', type=int, default=1, help='number of internal ssms. In general, H is supposed to serve as the number of internal SSMs, but it is a common practice to create lower amount of actual SSMs. Thus, the params shape are: A_1... A_4 \in [number of directions * N_ssm], B_1, B_2 \in [number of directions * N_ssm, N,2], C in [number of directions * H, N,], D \in [H]')
    parser.add_argument('--complex_ssm', action='store_true', default=False, help='Use complex ssm')
    parser.add_argument('--use_positional_encoding', type=bool, default=False, help='Use complex ssm')

    parser.add_argument('--directions_amount', type=int, default=4, help='number directions, can be 2 or 4. In order to allow bidirectional causality, we flip the kernels in 2 directions (two opposite corners of the image) or 4 (all the corners).')
    parser.add_argument('--save_kernels_and_exit', type=bool, default=False, help='Should')
    parser.add_argument('--dataset_percentage', type=float, default=1.0, help='Should')

    ## Vit params
    parser.add_argument('--smooth_v_as_well', action='store_true', default=False,
                        help='Whether to smooth the v or just k&q in ViT')
    parser.add_argument('--use_relative_pos_embedding', action='store_true', default=False,
                        help='Whether to rel pos embed or abs pos embed in ViT')
    parser.add_argument('--normalize', action='store_true', default=False,
                        help='Perform normalization after SSM')
    parser.add_argument('--use_residual_inside_ssm', action='store_true', default=False,
                        help='Uses residual inside the SSM. Some backbones (Like ConvNext) do it outside as well.')

    parser.add_argument('--no_pos_embedding', action='store_true', default=False,
                        help='Whether to rel pos embed or abs pos embed in ViT')

    parser.add_argument('--ssm_kernel_size', type=int, default=None,
                        help='Use shorter than the default ssm length')

    parser.add_argument('--use_mix_ffn', action='store_true', default=False,
                        help='Use mix ffn')

    return parser


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    data_info = datainfo(logger, args)
    model, run_name_with_model = create_model(data_info['img_size'], data_info['n_classes'], args)
    model.to(device)

    if args.project != '':
        run_name_with_model = run_name_with_model if not args.name else args.name
        # should_resume = True if args.resume != None and args.resume != '' else False
        wandb.init(project=args.project, name=run_name_with_model, config=args)
        args.wandb = True
    else:
        args.wandb = False
    global best_acc1

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_model_memory_consumption(model,logger)
    if args.wandb:
        wandb.log({'n_parameters': n_parameters}, commit=False)
    logger.debug(Fore.GREEN + '*' * 80)
    logger.debug(f"Creating model: {run_name_with_model}")
    logger.debug(str(model))
    logger.debug(f'Number of params: {format(n_parameters, ",")}')
    logger.debug(f'Initial learning rate: {args.lr:.6f}')
    logger.debug(f"Start training for {args.epochs} epochs")
    logger.debug('*' * 80 + Style.RESET_ALL)

    if args.ls:
        print(Fore.YELLOW + '*' * 80)
        logger.debug('label smoothing used')
        print('*' * 80 + Style.RESET_ALL)
        criterion = LabelSmoothingCrossEntropy()

    else:
        criterion = nn.CrossEntropyLoss()

    if args.sd > 0.:
        print(Fore.YELLOW + '*' * 80)
        logger.debug(f'Stochastic depth({args.sd}) used ')
        print('*' * 80 + Style.RESET_ALL)

    criterion = criterion.to(device)

    normalize = [transforms.Normalize(mean=data_info['stat'][0], std=data_info['stat'][1])]

    if args.cm:
        print(Fore.YELLOW + '*' * 80)
        logger.debug('Cutmix used')
        print('*' * 80 + Style.RESET_ALL)
    if args.mu:
        print(Fore.YELLOW + '*' * 80)
        logger.debug('Mixup used')
        print('*' * 80 + Style.RESET_ALL)
    if args.ra > 1:
        print(Fore.YELLOW + '*' * 80)
        logger.debug(f'Repeated Aug({args.ra}) used')
        print('*' * 80 + Style.RESET_ALL)

    '''
        Data Augmentation
    '''
    augmentations = []

    augmentations += [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(data_info['img_size'], padding=4)
    ]

    if args.aa == True:
        print(Fore.YELLOW + '*' * 80)
        logger.debug('Autoaugmentation used')

        if 'CIFAR' in args.dataset:
            print("CIFAR Policy")
            from utils.autoaug import CIFAR10Policy
            augmentations += [
                CIFAR10Policy()
            ]

        elif 'SVHN' in args.dataset:
            print("SVHN Policy")
            from utils.autoaug import SVHNPolicy
            augmentations += [
                SVHNPolicy()
            ]


        else:
            from utils.autoaug import ImageNetPolicy
            augmentations += [
                ImageNetPolicy()
            ]

        print('*' * 80 + Style.RESET_ALL)

    augmentations += [
        transforms.ToTensor(),
        *normalize]

    if args.re > 0:
        from utils.random_erasing import RandomErasing
        print(Fore.YELLOW + '*' * 80)
        logger.debug(f'Random erasing({args.re}) used ')
        print('*' * 80 + Style.RESET_ALL)

        augmentations += [
            RandomErasing(probability=args.re, sh=args.re_sh, r1=args.re_r1, mean=data_info['stat'][0])
        ]

    augmentations = transforms.Compose(augmentations)

    train_dataset, val_dataset = dataload(args, augmentations, normalize, data_info)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, num_workers=args.workers, pin_memory=True,
        batch_sampler=RASampler(len(train_dataset), args.batch_size, 1, args.ra, shuffle=True, drop_last=True))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    '''
        Training
    '''
    groups = create_optimization_groups(model, args)
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(args, optimizer, len(train_loader))

    # summary(model, (3, data_info['img_size'], data_info['img_size']))

    logger.debug("\nBeginning training\n")

    lr = optimizer.param_groups[0]["lr"]

    if args.resume:
        if args.resume == 'auto':
            args.resume = os.path.join(save_path, 'checkpoint.pth')
        checkpoint = torch.load(args.resume)  # , map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        final_epoch = args.epochs
        args.epochs = final_epoch - (checkpoint['epoch'] + 1)
    for epoch in tqdm(range(args.epochs)):
        lr = train(train_loader, model, criterion, optimizer, epoch, scheduler, args)
        acc1 = validate(val_loader, model, criterion, lr, args, epoch=epoch)
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        },
            os.path.join(save_path, 'checkpoint.pth'))

        logger_dict.print()

        if acc1 > best_acc1:
            print('* Best model upate *')
            best_acc1 = acc1

            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, os.path.join(save_path, 'best.pth'))

        print(f'Best acc1 {best_acc1:.2f}')
        if args.wandb:
            # Log best accuracy
            wandb.log({"best_acc1": best_acc1}, commit=False)
        print('*' * 80)
        print(Style.RESET_ALL)

        writer.add_scalar("Learning Rate", lr, epoch)

    print(Fore.RED + '*' * 80)
    logger.debug(f'best top-1: {best_acc1:.2f}, final top-1: {acc1:.2f}')
    print('*' * 80 + Style.RESET_ALL)
    torch.save(model.state_dict(), os.path.join(save_path, 'checkpoint.pth'))


def train(train_loader, model, criterion, optimizer, epoch, scheduler, args):
    global device
    model.train()
    loss_val, acc1_val = 0, 0
    n = 0

    for i, (images, target) in enumerate(train_loader):
        if (not args.no_cuda) and torch.cuda.is_available():
            images = images.to(device)
            target = target.to(device)

        # Cutmix only
        if args.cm and not args.mu:
            r = np.random.rand(1)
            if r < args.mix_prob:
                slicing_idx, y_a, y_b, lam, sliced = cutmix_data(images, target, args)
                images[:, :, slicing_idx[0]:slicing_idx[2], slicing_idx[1]:slicing_idx[3]] = sliced
                output = model(images)

                loss = mixup_criterion(criterion, output, y_a, y_b, lam)


            else:
                output = model(images)

                loss = criterion(output, target)


        # Mixup only
        elif not args.cm and args.mu:
            r = np.random.rand(1)
            if r < args.mix_prob:
                images, y_a, y_b, lam = mixup_data(images, target, args)
                output = model(images)

                loss = mixup_criterion(criterion, output, y_a, y_b, lam)



            else:
                output = model(images)

                loss = criterion(output, target)


        # Both Cutmix and Mixup
        elif args.cm and args.mu:
            r = np.random.rand(1)
            if r < args.mix_prob:
                switching_prob = np.random.rand(1)

                # Cutmix
                if switching_prob < 0.5:
                    slicing_idx, y_a, y_b, lam, sliced = cutmix_data(images, target, args)
                    images[:, :, slicing_idx[0]:slicing_idx[2], slicing_idx[1]:slicing_idx[3]] = sliced
                    output = model(images)

                    loss = mixup_criterion(criterion, output, y_a, y_b, lam)


                # Mixup
                else:
                    images, y_a, y_b, lam = mixup_data(images, target, args)
                    output = model(images)

                    loss = mixup_criterion(criterion, output, y_a, y_b, lam)

            else:
                output = model(images)

                loss = criterion(output, target)

                # No Mix
        else:
            output = model(images)

            loss = criterion(output, target)

        acc = accuracy(output, target, (1,))
        acc1 = acc[0]
        n += images.size(0)
        loss_val += float(loss.item() * images.size(0))
        acc1_val += float(acc1[0] * images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        if args.print_freq >= 0 and i % args.print_freq == 0:
            avg_loss, avg_acc1 = (loss_val / n), (acc1_val / n)
            progress_bar(i, len(train_loader),
                         f'[Epoch {epoch + 1}/{args.epochs}][T][{i}]   Loss: {avg_loss:.4e}   Top-1: {avg_acc1:6.2f}   LR: {lr:.7f}' + ' ' * 10)

    logger_dict.update(keys[0], avg_loss)
    logger_dict.update(keys[1], avg_acc1)
    if args.wandb:
        wandb.log({"Train Loss": avg_loss, "Train Acc": avg_acc1, "Learning Rate": lr})
    writer.add_scalar("Loss/train", avg_loss, epoch)
    writer.add_scalar("Acc/train", avg_acc1, epoch)

    return lr


def validate(val_loader, model, criterion, lr, args, epoch=None):
    model.eval()
    global device
    loss_val, acc1_val = 0, 0
    n = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if (not args.no_cuda) and torch.cuda.is_available():
                images = images.to(device)
                target = target.to(device)

            output = model(images)
            loss = criterion(output, target)

            acc = accuracy(output, target, (1, 5))
            acc1 = acc[0]
            n += images.size(0)
            loss_val += float(loss.item() * images.size(0))
            acc1_val += float(acc1[0] * images.size(0))

            if args.print_freq >= 0 and i % args.print_freq == 0:
                avg_loss, avg_acc1 = (loss_val / n), (acc1_val / n)
                progress_bar(i, len(val_loader),
                             f'[Epoch {epoch + 1}][V][{i}]   Loss: {avg_loss:.4e}   Top-1: {avg_acc1:6.2f}   LR: {lr:.6f}')
    print()

    print(Fore.BLUE)
    print('*' * 80)

    logger_dict.update(keys[2], avg_loss)
    logger_dict.update(keys[3], avg_acc1)
    if args.wandb:
        wandb.log({"Val Loss": avg_loss, "Val Acc": avg_acc1})
    writer.add_scalar("Loss/val", avg_loss, epoch)
    writer.add_scalar("Acc/val", avg_acc1, epoch)

    return avg_acc1


if __name__ == '__main__':

    parser = init_parser()
    args = parser.parse_args()
    global save_path
    global writer

    # random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    np.random.seed(args.seed)  # Numpy module.
    random.seed(args.seed)  # Python random module.
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Get current time
    current_time = datetime.now()

    # Format as day_month_year_hour_minute
    formatted_time = current_time.strftime("%d_%m_%Y_%H_%M")

    save_path = os.path.join(os.getcwd(), args.dataset, args.model, formatted_time)
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    writer = SummaryWriter(os.path.join(os.getcwd(), 'tensorboard'))

    # logger

    log_dir = os.path.join(save_path, 'history.csv')
    logger = log.getLogger(__name__)
    formatter = log.Formatter('%(message)s')
    streamHandler = log.StreamHandler()
    fileHandler = log.FileHandler(log_dir, 'a')
    streamHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)
    logger.setLevel(level=log.DEBUG)

    global logger_dict
    global keys

    logger_dict = Logger_dict(logger, save_path)
    keys = ['T Loss', 'T Top-1', 'V Loss', 'V Top-1']
    # Set cuda visible devices to 6

    main(args)
