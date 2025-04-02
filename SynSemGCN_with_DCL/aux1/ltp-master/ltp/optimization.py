#! /usr/bin/env python
# -*- coding: utf-8 -*_
# Author: Yunlong Feng <ylfeng@ir.hit.edu.cn>

import re
import inspect
from argparse import ArgumentParser

from transformers import optimization
from transformers.optimization import *


def get_layer_lrs(
        named_parameters,
        transformer_preffix,
        learning_rate,
        layer_decay,
        n_layers,
        **kwargs
):
    groups = []
    temp_groups = [None] * (n_layers + 3)
    temp_no_decay_groups = [None] * (n_layers + 3)
    for name, parameters in named_parameters:
        regex = rf"^{transformer_preffix}\.(embeddings|encoder)\w*\.(layer.(\d+))?.+"
        m = re.match(regex, name)

        is_transformer = True
        if m is None:
            depth = n_layers + 2
            is_transformer = False
        elif m.group(1) == 'embeddings':
            depth = 0
        elif m.group(1) == 'encoder':
            depth = int(m.group(3)) + 1
        else:
            raise Exception("Not Recommand!!!")

        if is_transformer and any(x in name for x in ['bias', 'LayerNorm.bias', 'LayerNorm.weight']):
            if temp_no_decay_groups[depth] is None:
                temp_no_decay_groups[depth] = []
            temp_no_decay_groups[depth].append(parameters)
        else:
            if temp_groups[depth] is None:
                temp_groups[depth] = []
            temp_groups[depth].append(parameters)

    for depth, parameters in enumerate(temp_no_decay_groups):
        if parameters:
            groups.append({
                'params': parameters, 'weight_decay': 0.0, 'lr': learning_rate * (layer_decay ** (n_layers + 2 - depth))
            })
    for depth, parameters in enumerate(temp_groups):
        if parameters:
            groups.append({
                'params': parameters, 'lr': learning_rate * (layer_decay ** (n_layers + 2 - depth))
            })
    return groups


def get_layer_lrs_with_crf(
        named_parameters,
        transformer_preffix,
        learning_rate,
        layer_decay,
        n_layers,
        crf_preffix,
        crf_rate=10.0,
        **kwargs
):
    groups = []
    crf_groups = []
    temp_groups = [None] * (n_layers + 3)
    temp_no_decay_groups = [None] * (n_layers + 3)
    for name, parameters in named_parameters:
        regex = rf"^{transformer_preffix}\.(embeddings|encoder)\w*\.(layer.(\d+))?.+"
        m = re.match(regex, name)

        is_transformer = True
        if m is None:
            depth = n_layers + 2
            is_transformer = False
        elif m.group(1) == 'embeddings':
            depth = 0
        elif m.group(1) == 'encoder':
            depth = int(m.group(3)) + 1
        else:
            raise Exception("Not Recommand!!!")

        if is_transformer and any(x in name for x in ['bias', 'LayerNorm.bias', 'LayerNorm.weight']):
            if temp_no_decay_groups[depth] is None:
                temp_no_decay_groups[depth] = []
            temp_no_decay_groups[depth].append(parameters)
        elif not is_transformer and crf_preffix in name:
            crf_groups.append(parameters)
        else:
            if temp_groups[depth] is None:
                temp_groups[depth] = []
            temp_groups[depth].append(parameters)

    for depth, parameters in enumerate(temp_no_decay_groups):
        if parameters:
            groups.append({
                'params': parameters, 'weight_decay': 0.0, 'lr': learning_rate * (layer_decay ** (n_layers + 2 - depth))
            })
    for depth, parameters in enumerate(temp_groups):
        if parameters:
            groups.append({
                'params': parameters, 'lr': learning_rate * (layer_decay ** (n_layers + 2 - depth))
            })
    if crf_groups:
        groups.append({
            'params': crf_groups, 'lr': learning_rate * crf_rate
        })

    return groups


scheduler_register = {
    'constant_schedule': get_constant_schedule,
    'constant_schedule_with_warmup': get_constant_schedule_with_warmup,
    'linear_schedule_with_warmup': get_linear_schedule_with_warmup,
    'cosine_schedule_with_warmup': get_cosine_schedule_with_warmup,
    'cosine_with_hard_restarts_schedule_with_warmup': get_cosine_with_hard_restarts_schedule_with_warmup,
    'polynomial_decay_schedule_with_warmup': get_polynomial_decay_schedule_with_warmup
}


def get_scheduler(optimizer, name, **kwargs) -> LambdaLR:
    scheduler = scheduler_register.get(name, get_linear_schedule_with_warmup)
    signature = inspect.signature(scheduler)
    kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return scheduler(optimizer, **kwargs)


layer_lrs_getters_register = {
    'get_layer_lrs': get_layer_lrs,
    'get_layer_lrs_with_crf': get_layer_lrs_with_crf,
}


def get_layer_lrs_getters(named_parameters, name, **kwargs):
    layer_lrs_getter = layer_lrs_getters_register.get(name, get_layer_lrs)
    signature = inspect.signature(layer_lrs_getter)
    kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return layer_lrs_getter(named_parameters, **kwargs)


def create_optimizer(
        model, lr,
        num_train_steps,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.0,
        warmup_steps=0,
        warmup_proportion=0.1,
        layerwise_lr_decay_power=0.8,
        transformer_preffix="transformer",
        n_transformer_layers=12,
        layer_lrs_getter='get_layer_lrs',
        layer_lrs_getter_kwargs=None,
        lr_scheduler='linear_schedule_with_warmup',
        lr_scheduler_kwargs=None,
):
    """

    Args:
        model:
        lr: 3e-4 for Small, 1e-4 for Base, 5e-5 for Large
        num_train_steps:
        weight_decay: 0
        warmup_steps: 0
        warmup_proportion: 0.1
        lr_decay_power: 1.0
        layerwise_lr_decay_power: 0.8 for Base/Small, 0.9 for Large

    Returns:

    """
    if lr_scheduler_kwargs is None:
        lr_scheduler_kwargs = {}
    if layer_lrs_getter_kwargs is None:
        layer_lrs_getter_kwargs = {}
    if lr_scheduler_kwargs is None:
        lr_scheduler_kwargs = {}
    if layerwise_lr_decay_power > 0:
        parameters = get_layer_lrs_getters(
            named_parameters=list(model.named_parameters()),
            name=layer_lrs_getter,
            transformer_preffix=transformer_preffix,
            learning_rate=lr,
            layer_decay=layerwise_lr_decay_power,
            n_layers=n_transformer_layers,
            **layer_lrs_getter_kwargs
        )
    else:
        parameters = model.parameters()

    optimizer = optimization.AdamW(
        parameters,
        lr=lr,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
        eps=1e-6,
        correct_bias=False,
    )

    warmup_steps = max(num_train_steps * warmup_proportion, warmup_steps)
    scheduler = get_scheduler(
        optimizer,
        name=lr_scheduler,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_train_steps,
        **lr_scheduler_kwargs
    )
    return optimizer, scheduler


def add_optimizer_specific_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)

    # 3e-4 for Small, 1e-4 for Base, 5e-5 for Large
    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument(
        '--lr_scheduler', type=str, default='linear_schedule_with_warmup', choices=scheduler_register.keys()
    )
    parser.add_argument('--lr_end', type=float, default=1e-7)
    parser.add_argument('--lr_num_cycles', type=float, default=0.5)
    parser.add_argument('--lr_decay_power', type=float, default=1.0)

    parser.add_argument(
        '--lr_layers_getter', type=str, default='get_layer_lrs_with_crf', choices=layer_lrs_getters_register.keys()
    )
    parser.add_argument('--lr_crf_preffix', type=str, default='crf')
    parser.add_argument('--lr_crf_rate', type=float, default=10.0)

    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.0)

    # 0.8 for Base/Small, 0.9 for Large
    parser.add_argument('--layerwise_lr_decay_power', type=float, default=0.8)
    return parser


def from_argparse_args(args, model, num_train_steps, n_transformer_layers=12, **kwargs):
    return create_optimizer(
        model,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        num_train_steps=num_train_steps,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        warmup_proportion=args.warmup_proportion,
        layerwise_lr_decay_power=args.layerwise_lr_decay_power,
        n_transformer_layers=n_transformer_layers,
        layer_lrs_getter=args.lr_layers_getter,
        layer_lrs_getter_kwargs={
            'crf_preffix': args.lr_crf_preffix,
            'crf_rate': args.lr_crf_rate,
        },
        lr_scheduler=args.lr_scheduler,
        lr_scheduler_kwargs={
            'lr_end': args.lr_end,
            'power': args.lr_decay_power,
            'num_cycles': args.lr_num_cycles
        }, **kwargs
    )
