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
from arthmetic_model.model import ArthModelArgs, Arth_Model, ArthCalcModule
from arthmodel_train import SimpleNumberDataset, gen_manual_aux_info

def gen_legal_output_for_arthcalc(texts: list, max_len=128):
    op_safe=['+' , '-' , "*", '/', '^']
    l_out_trans_valid=[]
    l_out_trans_dense=[]
    l_out_trans_op_pred=[]
    for x_str in texts:
        arr = x_str.split(" ")
        list_op_pred=[]
        list_trans_valid=[]
        list_trans_dense=[]
        for arr_x in arr:
            arr_x = arr_x.strip()
            list_op=[-100, -100, -100, -100, -100, -100, -100]
            if arr_x == "":
                continue
            elif arr_x in op_safe:
                index = op_safe.index(arr_x)
                list_op[index + 2] = 100
                list_trans_valid.append(1)
                list_trans_dense.append(0.0)
                list_op_pred.append(list_op)
            else:
                try:
                    dense=float(arr_x)
                    list_op[0] = 100
                    list_trans_valid.append(1)
                    list_trans_dense.append(dense)
                    list_op_pred.append(list_op)
                except:
                    continue
        if len(list_trans_valid) > max_len:
            list_op_pred=list_op_pred[:max_len, :]
            list_trans_valid=list_trans_valid[:max_len]
            list_trans_dense=list_trans_dense[:max_len]
        while len(list_op_pred) < max_len:
            list_op=[100, -100, -100, -100, -100, -100, -100]
            list_op_pred.append(list_op)
            list_trans_valid.append(0)
            list_trans_dense.append(0)
        l_out_trans_valid.append(torch.tensor(list_trans_valid, dtype=torch.float32))
        l_out_trans_dense.append(torch.tensor(list_trans_dense, dtype=torch.float64))
        l_out_trans_op_pred.append(torch.tensor(list_op_pred, dtype=torch.float32))
    trans_valid = torch.stack(l_out_trans_valid)
    trans_dense = torch.stack(l_out_trans_dense)
    trans_op_pred = torch.stack(l_out_trans_op_pred)
    return trans_valid, trans_dense, trans_op_pred

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
    parser.add_argument("--checkpoint", default="./checkpoint/arth_model_128dim.pth", type=str, help="path of llama model")
    #parser.add_argument("--checkpoint", default="./checkpoint_final_202307142226/checkpoint-0.pth", type=str, help="path of llama model")
    return parser

def arthmodel_load(args, args_for_model : ArthModelArgs, tokenizer, **kwargs):
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    arth_model = Arth_Model(args_for_model, tokenizer)
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

def test_arth_to_dense(args):
    default_arth_args = ArthModelArgs()
    default_arth_args.max_batch_size = 1
    default_arth_args.dim=128
    default_arth_args.max_seq_len=128
    default_arth_args.output_steps = False
    default_arth_args.debug_trace = False
    default_arth_args.device="cuda"
    tokenizer = Tokenizer(model_path=args.llama_model_path + "/tokenizer.model")
    model = arthmodel_load(args, default_arth_args, tokenizer)
    model.to(default_arth_args.device)
    # text="13.45"
    # text_list=["90.12", "1.234", "567.9"]
    # text_list=["32716.022586 * 48959.059241 71809.071167 41330.026497 ^ 62365.021137 ^ * 27002.081354 + 88707.069639 / ^ 65903.013672"]
    # text_list=["24895.045487 * 59404.090586 ^ 39265.011407 87531.080102 82825.058262 57437.093692 + 26169.032728 50103.073226 ^ - *"]
    # text_list=["0.8207 905980.60908 + 65.31 - 58.0 27.0 * 565768.81514 / - 495431.354828 825.66628 / 8242.644686 * 1930.3 / +"]
    text_list=["16 11 * 10001 + 3 -  "]
    model.eval()
    lst_tokens=[]
    for x in text_list:
        list_tokens=tokenizer.encode(x,bos=True,eos=True)
        # list_tokens = [1, 29871, 29941, 29889, 29945, 29871, 29906, 29889, 29955,   334,
        #  29871, 29947, 29889, 29953,   718,     2]
        list_arth_tokens=transfer_token_ids(list_tokens, default_arth_args.dict_vocb_map)
        lst_tokens.append(list_arth_tokens)
    print("lst_tokens", lst_tokens)
    token_tensor = torch.tensor(lst_tokens, dtype=torch.int).to(default_arth_args.device)
    aux_lal_steps_ignore_logits, l_steps_tmp_moved_logits, l_steps_dense_op_logits, l_steps_dense_map_logits, l_steps_decimal_start_logits, l_steps_op_pred = gen_manual_aux_info(token_tensor, 0)
    print("l_steps_op_pred", l_steps_op_pred)
    final_tokens = model(token_tensor, start_pos = 0, tokens_is_index=True)
    print("final_tokens: ", final_tokens)
    if default_arth_args.output_steps ==True:
        arth_tokens, steps_ignore_logits, steps_tmp_moved_logits, steps_dense_op_logits, steps_dense_map_logits, steps_decimal_start_logits, steps_op_pred = model(token_tensor, start_pos = 0, tokens_is_index=True)
        print("arth_tokens", arth_tokens)
        print("steps_ignore_logits", steps_ignore_logits)
        print("steps_tmp_moved_logits", steps_tmp_moved_logits)
        print("steps_dense_op_logits", steps_dense_op_logits)
        print("steps_dense_map_logits", steps_dense_map_logits)
        print("steps_decimal_start_logits", steps_decimal_start_logits)
        print("steps_op_pred", steps_op_pred)
    else:
        arth_tokens = model(token_tensor, start_pos = 0, tokens_is_index=True)
        print("arth_tokens", arth_tokens)

def test_arth_calc(args):
    default_arth_args = ArthModelArgs()
    default_arth_args.max_batch_size = 1
    default_arth_args.dim=8
    default_arth_args.max_seq_len=8
    default_arth_args.output_steps = True
    default_arth_args.debug_trace = True
    default_arth_args.device="cpu"
    default_arth_args.tdtype = torch.float64
    default_arth_args.ndtype = torch.float32
    arth_calc_model = ArthCalcModule(default_arth_args)
    text="3 5 +"
    trans_valid, trans_dense, trans_op_pred = gen_legal_output_for_arthcalc([text], default_arth_args.max_seq_len)
    trans_valid, trans_dense, trans_op_pred, if_finished,  if_valid= arth_calc_model.forward(trans_valid, trans_dense, trans_op_pred)
    print("trans_valid", trans_valid)
    print("trans_dense", trans_dense)
    print("trans_op_pred", trans_op_pred)
    print("if_finished", if_finished)
    print("if_valid", if_valid)

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    test_arth_to_dense(args)
    # test_arth_calc(args)