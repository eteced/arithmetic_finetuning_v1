#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This software may be used and distributed according to the terms of the GNU General Public License version 3.
"""Small Models for Arthmetic
"""
__author__ = "Yingdi Guo"
__license__ = "GPLv3"
__email__ = "eteced@gmail.com"

import argparse
import copy
import datetime
import json
import os
import sys
import time
from pathlib import Path

import arthmetic_model.model
import numpy as np
import timm.optim.optim_factory as optim_factory
import torch
import torch.backends.cudnn as cudnn
import util.misc as misc
from torch.utils.data import Dataset
from dataclasses import field
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import random
from llama import Tokenizer
from torch.utils.tensorboard import SummaryWriter
from typing import Iterable
import util.lr_sched as lr_sched
import util.misc as misc
import math
from arthmetic_model.model import ArthModelArgs, Arth_Model
from arthmodel_train import SimpleNumberDataset


def get_args_parser():
    parser = argparse.ArgumentParser("ArthModel Inference", add_help=False)
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )

    # Model parameters
    parser.add_argument("--llama_model_path", default="/home/eteced/dl_workspace/model_repo.folder/llama_ord/", type=str, help="path of llama model")
    parser.add_argument("--checkpoint", default="./checkpoint/checkpoint-99.pth", type=str, help="path of llama model")
    return parser

def arthmodel_load(args, args_for_model : ArthModelArgs, **kwargs):
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    arth_model = Arth_Model(args_for_model)
    arth_model.load_state_dict(checkpoint['model'], strict=True)
    return arth_model

def transfer_token_ids(list_tokens : list, dict_vocb_map : dict):
    list_final = []
    for x in list_tokens:
        if x in dict_vocb_map:
            list_final.append(dict_vocb_map[x])
        else:
            list_final.append(20)
    return list_final

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    default_arth_args = ArthModelArgs()
    default_arth_args.max_batch_size = 1
    default_arth_args.dim=16
    default_arth_args.max_seq_len=16
    default_arth_args.output_steps = True
    default_arth_args.debug_trace = True
    default_arth_args.device="cpu"
    model = arthmodel_load(args, default_arth_args)
    tokenizer = Tokenizer(model_path=args.llama_model_path + "/tokenizer.model")
    text="1.111"
    model.eval()
    list_tokens=tokenizer.encode(text,bos=True,eos=True)
    list_arth_tokens=transfer_token_ids(list_tokens, default_arth_args.dict_vocb_map)
    print("list_arth_tokens", list_arth_tokens)
    token_tensor = torch.tensor([list_arth_tokens], dtype=torch.int)
    if default_arth_args.output_steps ==True:
        trans_valid, trans_dense, trans_op, steps_ignore_logits, steps_tmp_moved_logits, steps_dense_op_logits, steps_dense_map_logits, steps_decimal_start_logits, steps_op_pred = model(token_tensor, start_pos = 0)
        print("trans_valid", trans_valid)
        print("trans_dense", trans_dense)
        print("steps_ignore_logits", steps_ignore_logits)
        print("steps_tmp_moved_logits", steps_tmp_moved_logits)
        print("steps_dense_op_logits", steps_dense_op_logits)
        print("steps_dense_map_logits", steps_dense_map_logits)
        print("steps_decimal_start_logits", steps_decimal_start_logits)
        print("steps_op_pred", steps_op_pred)
    else:
        trans_valid, trans_dense, trans_op = model(token_tensor, start_pos = 0)
        print("trans_valid", trans_valid)
        print("trans_dense", trans_dense)
        print("trans_op", trans_op)