import argparse
import importlib
import itertools

import torch
import math

from collections import OrderedDict

import common

from torch.utils.data import DataLoader

from utils import lr_scheduler, gradual_warmup_scheduler
from utils.prefetch_dataloader import CUDAPrefetcher
from models.model import Model
from utils.init_utils import master_only, when_attr_is_true

__all__ = ["create_dataloader",
           "create_criterions",
           "load_ckpt",
           "state_dict_saver",
           'ckpt_saver',
           "create_optim_scheduler",
           # "clip_gradient"
           ]


def create_dataloader(args):
    dataset_module = importlib.import_module(f'datasets.{args.dataset}' if args.dataset else 'datasets')
    train_dataset = dataset_module.get_dataset(common.modes.TRAIN, args)

    # Load eval dataset
    if args.eval_datasets:
        eval_datasets = []
        for eval_dataset in args.eval_datasets:
            eval_dataset_module = importlib.import_module(f'datasets.{eval_dataset}')
            eval_datasets.append((eval_dataset, eval_dataset_module.get_dataset(common.modes.EVAL, args)))
    else:
        eval_datasets = [(args.dataset, dataset_module.get_dataset(common.modes.EVAL, args))]

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    eval_sampler = None
    prefetch_factor = 4
    # Dataloader
    train_data_loader = DataLoader(dataset=train_dataset,
                                   num_workers=args.num_data_threads,
                                   batch_size=args.train_batch_size,
                                   shuffle=(train_sampler is None),
                                   drop_last=True,
                                   pin_memory=True,
                                   sampler=train_sampler,
                                   prefetch_factor=prefetch_factor)

    train_data_loader = CUDAPrefetcher(loader=train_data_loader, args=args)

    eval_kwargs = {"num_workers": args.num_data_threads,
                   "batch_size": args.eval_batch_size,
                   'shuffle': False,
                   'drop_last': False,
                   'pin_memory': True,
                   'sampler': eval_sampler}
    eval_data_loaders = [(data_name, DataLoader(dataset=dataset, **eval_kwargs)) for data_name, dataset in
                         eval_datasets]

    args.total_iterations = int(args.epochs * len(train_data_loader))
    if hasattr(args, 'bn_iterations'):
        args.bn_iterations = int(args.bn_iterations * args.total_iterations)

    if args.log_steps == 0:
        args.log_steps = min(len(train_data_loader) // 100, 100)
    return train_data_loader, train_sampler, eval_data_loaders, eval_sampler


def get_subdict(dict: dict, exception: str) -> dict:
    # exceptions = exceptions.split(',')
    return {k: v for k, v in dict.items() if k not in exception}


def create_criterions(args: argparse.Namespace):
    assert isinstance(args, argparse.Namespace), 'args should be an argparse.Namespace object'
    assert hasattr(args, 'losses'), "Missing losses in model config"
    assert isinstance(args.losses, dict), "Losses in model config should be a dictionary"
    losses_module = importlib.import_module("losses")
    criterions = OrderedDict()
    for name, kwargs in args.losses.items():
        loss = getattr(losses_module, kwargs.get('type'))
        if kwargs.get('type') == 'BitPerPixelLoss':
            kwargs['lambda_schedule']['steps'][0] *= args.total_iterations
            kwargs['target_schedule']['steps'][0] *= args.total_iterations
        criterions[name] = loss(**get_subdict(kwargs, exception='type')).to(args.local_rank)

    return criterions


def create_optim_scheduler(model_list: [torch.nn.Module], args: argparse.Namespace, num_iters: float):
    if not isinstance(model_list, list):
        model_list = [model_list]
    assert isinstance(args, argparse.Namespace), 'args should be an argparse.Namespace object'
    assert hasattr(args, 'optim'), "Missing optim in model config"
    assert isinstance(args.optim, dict), "optim in model config should be a dictionary"

    optim_module = getattr(torch.optim, args.optim.get('type'))
    optim_dict = get_subdict(args.optim, exception='type')

    if args.distributed:
        if not hasattr(args, 'no_scale_lr') or not args.no_scale_lr:  # works due to python lazy boolean
            original_lr = optim_dict.get('lr')
            # scalar = args.world_size if args.world_size <= 4 else 4 + math.sqrt(args.world_size - 4)
            scalar = args.world_size
            optim_dict['lr'] *= scalar
            args.logger.info(f"Scale learning from {original_lr} ==x{scalar:.02f}==> {optim_dict.get('lr')}")

    total_iters = args.epochs * num_iters

    if args.warmup_lr:
        if hasattr(args, 'warmup_iters'):
            warmup_iters = int(total_iters * args.warmup_iters)
        else:
            args.logger.info('warmup_iters not set, using default setting')
            warmup_iters = total_iters // args.epochs
        args.logger.info(f"Warm up lr for"
                         f" {warmup_iters}/{total_iters}={warmup_iters / total_iters * 100:.02f}% iterations")
        total_iters -= warmup_iters

    assert hasattr(args, 'scheduler'), "Missing scheduler in model config"
    assert isinstance(args.scheduler, dict), "scheduler in config should be a dictionary"

    scheduler_module = getattr(lr_scheduler, args.scheduler.get('type'))
    if issubclass(scheduler_module, torch.optim.lr_scheduler.MultiStepLR):
        args.scheduler['milestones'] = [int(math.ceil(total_iters * i)) for i in args.scheduler['milestones']]
    elif issubclass(scheduler_module, lr_scheduler.CosineAnnealingRestartLR):
        # evenly divide periods
        args.scheduler['periods'] = [total_iters // len(args.scheduler['restart_weights']) for _ in
                                     range(len(args.scheduler['restart_weights']))]

        # periods base on percentage e.g. epochs=40 [0, 0.25, 0.50, 0.75, 1] --> [10, 10, 10, 10]
        # args.scheduler['periods'] = [
        #     num_iters * args.epochs * (args.scheduler['periods'][idx + 1] - args.scheduler['periods'][id]) for idx in
        #     range(len(args.scheduler['periods']) - 1)
        # ]
    elif issubclass(scheduler_module, torch.optim.lr_scheduler.CosineAnnealingLR):
        args.scheduler["T_max"] = total_iters
    else:
        NotImplementedError(f"Method {scheduler_module.__class__} is not implemented")
    optimizer_list = []
    scheduler_list = []
    for model in model_list:
        if hasattr(model, 'amortization_models'):
            amortization_parameters = itertools.chain.from_iterable(
                [am.parameters() for am in model.amortization_models])
            optimizer = optim_module(filter(lambda p: p.requires_grad, amortization_parameters),
                                     **optim_dict)
        else:
            optimizer = optim_module(filter(lambda p: p.requires_grad, model.parameters()),
                                     **optim_dict)
        scheduler = scheduler_module(optimizer, **get_subdict(args.scheduler, exception='type'))

        if args.warmup_lr:
            scheduler = gradual_warmup_scheduler.GradualWarmupScheduler(optimizer=optimizer,
                                                                        multiplier=1,
                                                                        total_epoch=warmup_iters,
                                                                        after_scheduler=scheduler)
        optimizer_list.append(optimizer)
        scheduler_list.append(scheduler)

    return optimizer_list, scheduler_list


def load_ckpt(ckpt, **kwargs):
    for k, v in kwargs.items():
        if hasattr(v, 'load_state_dict'):
            v.load_state_dict(ckpt[k])


@master_only
def state_dict_saver(path, model):
    state_dict = model.state_dict() if not hasattr(model, 'module') else model.module.state_dict()
    torch.save(state_dict, path)


@master_only
def ckpt_saver(path, **kwargs):
    ckpt = {}
    for k, v in kwargs.items():
        if isinstance(v, int):
            pass
        elif hasattr(v, 'module'):
            v = v.module
        if hasattr(v, 'state_dict'):
            v = v.state_dict()
        ckpt[k] = v
    torch.save(ckpt, path)


# def clip_gradient(optimizer, grad_clip):
#     for group in optimizer.param_groups:
#         for param in group["params"]:
#             if param.grad is not None:
#                 param.grad.data.clamp_(-grad_clip, grad_clip)


if __name__ == "__main__":
    test_params = argparse.Namespace()
    # test_params.losses = {
    #     'psnr': {
    #         "type": "PixelWiseLoss",
    #         "criterion": 'l1',
    #         "loss_weight": 1.0,
    #     },
    #     # 'vgg': {
    #     #     "type": "PerceptualLoss",
    #     #     "layer_weights":{'34': 1.0},
    #     #     "vgg_type": 'vgg19',
    #     #     "norm_img": False,
    #     #     "criterion": 'l1',
    #     #     "pretrained": 'torchvision://vgg19',
    #     #     "perceptual_weight": 1,
    #     # },
    # }
    #
    # test_params.local_rank = 0
    #
    # out = create_criterions(test_params)
    # for k, v in out.items():
    #     print(k, v)
