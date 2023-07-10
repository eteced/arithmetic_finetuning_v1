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
from arthmetic_model.model import ArthModelArgs

def arth_train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
):

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))
    for data_iter_step, (Text, labels) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        Text=Text.to(device)
        labels=labels.to(device)
        if args.step_mode == True:
            trans_valid, trans_dense, trans_op, steps_ignore_logits, steps_tmp_moved_logits, steps_dense_op_logits, steps_dense_map_logits, steps_decimal_start_logits, steps_op_pred = model(Text, start_pos = 0)
        else:
            trans_valid, trans_dense, trans_op = model(Text, start_pos = 0)
        
        if args.step_mode == False:
            label_valid = torch.zeros_like(trans_valid)
            label_valid[:, 0] = 1
            label_trans_op = torch.zeros_like(trans_op)
            label_trans_op[:, 0] = 1

            pred_dense = trans_dense[:, 0]
            loss_mse = torch.nn.MSELoss()
            loss_cp = torch.nn.CrossEntropyLoss()
            loss_bce = torch.nn.BCELoss()
            loss_smooth_l1 = torch.nn.SmoothL1Loss(beta=0.01)
            # print("max(label_valid)", torch.max(trans_valid, dim=1))
            # loss = 0.01 * loss_mse(pred_dense, labels) + 50 * loss_cp(trans_valid, label_valid) + 50 * loss_cp(trans_op, label_trans_op)
            loss = 50 * loss_smooth_l1(trans_valid, label_valid) + 50 * loss_cp(trans_op, label_trans_op)
            # print("trans_op", trans_op[:, 0])
            loss_value = loss.item()
        else:
            loss = None
            for tt in range(Text.shape[0]):
                l_steps_ignore_logits, l_steps_tmp_moved_logits, l_steps_dense_op_logits, l_steps_dense_map_logits, l_steps_decimal_start_logits, l_steps_op_pred = gen_manual_aux_info(Text, tt)
                o_steps_ignore_logits=[]
                o_steps_tmp_moved_logits=[]
                o_steps_dense_op_logits=[]
                o_steps_dense_map_logits=[]
                o_steps_decimal_start_logits=[]
                o_steps_op_pred=[]
                loss_cp = torch.nn.CrossEntropyLoss()
                for x in steps_ignore_logits:
                    o_steps_ignore_logits.append(x[tt])
                for x in steps_tmp_moved_logits:
                    o_steps_tmp_moved_logits.append(x[tt])
                for x in steps_dense_op_logits:
                    o_steps_dense_op_logits.append(x[tt])
                for x in steps_dense_map_logits:
                    o_steps_dense_map_logits.append(x[tt])
                for x in steps_decimal_start_logits:
                    o_steps_decimal_start_logits.append(x[tt])
                for x in steps_op_pred:
                    o_steps_op_pred.append(x[tt])
                for i in range(len(l_steps_ignore_logits)):
                    if loss is None:
                        loss = loss_cp(o_steps_ignore_logits[i], torch.tensor(l_steps_ignore_logits[i], dtype=torch.long))
                    else:
                        loss += loss_cp(o_steps_ignore_logits[i], torch.tensor(l_steps_ignore_logits[i], dtype=torch.long))
                    loss += loss_cp(o_steps_tmp_moved_logits[i], torch.tensor(l_steps_tmp_moved_logits[i], dtype=torch.long))
                    loss += loss_cp(o_steps_dense_op_logits[i], torch.tensor(l_steps_dense_op_logits[i], dtype=torch.long))
                    loss += loss_cp(o_steps_dense_map_logits[i], torch.tensor(l_steps_dense_map_logits[i], dtype=torch.long))
                    loss += loss_cp(o_steps_decimal_start_logits[i], torch.tensor(l_steps_decimal_start_logits[i], dtype=torch.long))
                    loss += loss_cp(o_steps_op_pred[i], torch.tensor(l_steps_op_pred[i], dtype=torch.long))
                    # print('>> ', i, "steps_decimal_start_logits", steps_decimal_start_logits[i], "text", Text,"l_steps_decimal_start_logits", l_steps_decimal_start_logits[i])
            loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        
        metric_logger.update(closs=loss_value)
        loss /= accum_iter

        if args.device != 'cpu':
            loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=(data_iter_step + 1) % accum_iter == 0)
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()
            torch.cuda.synchronize()
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("train_loss", loss_value, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def arth_val_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    accum_iter = args.accum_iter

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))
    for data_iter_step, (Text, labels) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):

        with torch.no_grad():
            Text=Text.to(device)
            labels=labels.to(device)
            if args.step_mode == True:
                trans_valid, trans_dense, trans_op, steps_ignore_logits, steps_tmp_moved_logits, steps_dense_op_logits, steps_dense_map_logits, steps_decimal_start_logits, steps_op_pred = model(Text, start_pos = 0)
            else:
                trans_valid, trans_dense, trans_op = model(Text, start_pos = 0)
            label_valid = torch.zeros_like(trans_valid)
            label_valid[:, 1] = 1
            label_trans_op = torch.zeros_like(trans_op)

            pred_dense = trans_dense[:, 1]
            loss_mse = torch.nn.MSELoss()
            loss_cp = torch.nn.CrossEntropyLoss()
            loss_bce = torch.nn.BCELoss()
            loss_smooth_l1 = torch.nn.SmoothL1Loss(beta=0.01)
            print("max(trans_valid)", torch.max(trans_valid, dim=1))
            loss = 0.01 * loss_smooth_l1(pred_dense, labels) + 50 * loss_cp(trans_valid, label_valid) + 50 * loss_cp(trans_op, label_trans_op)
            # loss = 50 * loss_mse(trans_valid, label_valid) + 50 * loss_cp(trans_op, label_trans_op)
            print("pred_dense", pred_dense)
            print("trans_dense", trans_dense)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        metric_logger.update(closs=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        misc.all_reduce_mean(loss_value)
        c_loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("eval, c_val_loss", c_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Val Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# class SimpleNumberDataset(Dataset):
#     def __init__(self, tokenizer_path, args: ArthModelArgs, record_size=100000, max_integer=100000, decimal_precise=0.000001, record_to_file_path=None):
#         self.f1 = None
#         if record_to_file_path is not None:
#             self.f1 = open(record_to_file_path, "w")
#         self.record_size = record_size
#         self.max_integer = max_integer
#         self.decimal_precise = decimal_precise
#         self.records={}
#         self.tokenizer = Tokenizer(model_path=tokenizer_path + "/tokenizer.model")
#         self.dict_vocb_map = args.dict_vocb_map
    
#     def __len__(self):
#         return self.record_size

#     def __getitem__(self, index):
#         if index in self.records:
#             number=self.records[index]
#             text=str(number)
#         else:
#             number=random.randrange(0, self.max_integer) + random.randrange(0, self.max_integer) * self.decimal_precise
#             text=str(number)
#             self.records[index]=number
#             self.f1.write(str(index)+'\t'+str(number)+'\n')

#         list_tokens = self.tokenizer.encode(text, bos=True, eos=True)
#         list_final = []
#         for x in list_tokens:
#             if x in self.dict_vocb_map:
#                 list_final.append(self.dict_vocb_map[x])
#             else:
#                 list_final.append(20)
#         ts_1 = torch.tensor(list_final, dtype=torch.float)
#         ts_2 = torch.tensor(number, dtype=torch.float)
#         return ts_1, ts_2

class SimpleNumberDataset(Dataset):
    def __init__(self, tokenizer_path, args: ArthModelArgs, record_size=100000, max_integer=100000, decimal_precise=0.000001, record_to_file_path='./snd.txt', partition="train"):
        f1 = open(record_to_file_path, "r")
        self.lst_data = []
        for line in f1:
            line=line.strip()
            if line=="":
                continue
            self.lst_data.append(line)
        f1.close()
        train_index=int(len(self.lst_data) * 0.1)
        print("train_index", train_index)
        if partition == "train":
            self.lst_data = self.lst_data[train_index:]
        else:
            self.lst_data = self.lst_data[:train_index]
        self.record_size = record_size
        self.max_integer = max_integer
        self.decimal_precise = decimal_precise
        self.tokenizer = Tokenizer(model_path=tokenizer_path + "/tokenizer.model")
        self.dict_vocb_map = args.dict_vocb_map
        self.max_words = args.max_seq_len
        self.ndtype = args.ndtype
        self.tdtype = args.tdtype
        self.device = args.device
    
    def __len__(self):
        return len(self.lst_data)

    def __getitem__(self, index):
        text = self.lst_data[index]
        number = float(text)
        list_tokens = self.tokenizer.encode(text, bos=True, eos=True)
        list_final = []
        for x in list_tokens:
            if x in self.dict_vocb_map:
                list_final.append(self.dict_vocb_map[x])
            else:
                list_final.append(20)
        ts_1 = torch.tensor(list_final, dtype=torch.int)
        padding = self.max_words - ts_1.shape[0]
        if padding > 0:
            ts_1 = torch.cat((ts_1, torch.ones(padding, dtype=torch.int) * 20))
        elif padding < 0:
            ts_1 = ts_1[: self.max_words]
        ts_2 = torch.tensor(number, dtype=self.tdtype)
        return ts_1, ts_2

# class SimpleNumberDatasetStepMode(Dataset):
#     def __init__(self, tokenizer_path, args: ArthModelArgs, record_size=100000, max_integer=100000, decimal_precise=0.000001, record_to_file_path='./snd.txt', partition="train"):
#         f1 = open(record_to_file_path, "r")
#         self.lst_data = []
#         for line in f1:
#             line=line.strip()
#             if line=="":
#                 continue
#             self.lst_data.append(line)
#         f1.close()
#         train_index=int(len(self.lst_data) * 0.1)
#         if partition == "train":
#             self.lst_data = self.lst_data[train_index:]
#         else:
#             self.lst_data = self.lst_data[:train_index]
#         self.record_size = record_size
#         self.max_integer = max_integer
#         self.decimal_precise = decimal_precise
#         self.tokenizer = Tokenizer(model_path=tokenizer_path + "/tokenizer.model")
#         self.dict_vocb_map = args.dict_vocb_map
#         self.max_words = args.max_seq_len
#         self.ndtype = args.ndtype
#         self.tdtype = args.tdtype
#         self.device = args.device
    
#     def __len__(self):
#         return len(self.lst_data)

#     def __getitem__(self, index):
#         text = self.lst_data[index]
#         number = float(text)
#         list_tokens = self.tokenizer.encode(text, bos=True, eos=True)
#         list_final = []
#         for x in list_tokens:
#             if x in self.dict_vocb_map:
#                 list_final.append(self.dict_vocb_map[x])
#             else:
#                 list_final.append(20)
#         ts_1 = torch.tensor(list_final, dtype=torch.int)
#         padding = self.max_words - ts_1.shape[0]
#         while len(list_final) < self.max_words:
#             list_final.append(20)
#         if padding > 0:
#             ts_1 = torch.cat((ts_1, torch.ones(padding, dtype=torch.int) * 20))
#         elif padding < 0:
#             ts_1 = ts_1[: self.max_words]
#             list_final = list_final[:self.max_words]
#         ts_2 = torch.tensor(number, dtype=self.tdtype)
#         return ts_1, ts_2
        

def gen_manual_aux_info(text : torch.Tensor, batch_index=1):
    steps_ignore_logits=[]
    steps_tmp_moved_logits=[]
    steps_dense_op_logits=[]
    steps_dense_map_logits=[]
    steps_decimal_start_logits=[]
    steps_op_pred=[]
    fp_flag = 0
    one_text = text[0, :]
    for i in range(one_text.shape[0]):
        xx=one_text[i]
        if xx in range(21, 23):
            steps_ignore_logits.append(1)
            steps_tmp_moved_logits.append(0)
            steps_dense_op_logits.append(0)
            steps_dense_map_logits.append(0)
            steps_decimal_start_logits.append(0)
            steps_op_pred.append(0)
        elif xx == 23:
            fp_flag = 0
            steps_ignore_logits.append(0)
            steps_tmp_moved_logits.append(1)
            steps_dense_op_logits.append(0)
            steps_dense_map_logits.append(0)
            steps_decimal_start_logits.append(0)
            steps_op_pred.append(0)
        elif xx in range(0, 10):
            steps_ignore_logits.append(0)
            steps_tmp_moved_logits.append(0)
            if fp_flag == 1:
                steps_dense_op_logits.append(3)
            else:
                steps_dense_op_logits.append(2)
            steps_dense_map_logits.append(xx)
            steps_decimal_start_logits.append(0)
            steps_op_pred.append(0)
        elif xx == 10:
            fp_flag = 1
            steps_ignore_logits.append(0)
            steps_tmp_moved_logits.append(0)
            steps_dense_op_logits.append(0)
            steps_dense_map_logits.append(0)
            steps_decimal_start_logits.append(1)
            steps_op_pred.append(0)
        elif xx in range(11, 15):
            fp_flag = 0
            steps_ignore_logits.append(0)
            steps_tmp_moved_logits.append(2)
            steps_dense_op_logits.append(0)
            steps_dense_map_logits.append(0)
            steps_decimal_start_logits.append(0)
            steps_op_pred.append(xx - 9)
        elif xx == 16:
            fp_flag = 0
            steps_ignore_logits.append(0)
            steps_tmp_moved_logits.append(2)
            steps_dense_op_logits.append(0)
            steps_dense_map_logits.append(0)
            steps_decimal_start_logits.append(0)
            steps_op_pred.append(6)
    return steps_ignore_logits, steps_tmp_moved_logits, steps_dense_op_logits, steps_dense_map_logits, steps_decimal_start_logits, steps_op_pred

def get_args_parser():
    parser = argparse.ArgumentParser("ArthModel Training", add_help=False)
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    # Model parameters
    parser.add_argument("--llama_model_path", default="./llama", type=str, help="path of llama model")
    parser.add_argument("--model", default="llama7B_adapter", type=str, metavar="MODEL", help="Name of model to train")

    parser.add_argument("--adapter_layer", type=int, default=30, metavar="LENGTH", help="the number of adapter layer")

    parser.add_argument("--adapter_len", type=int, default=10, metavar="LENGTH", help="the adapter length")

    parser.add_argument("--max_seq_len", type=int, default=512, metavar="LENGTH", help="the maximum sequence length")

    # Optimizer parameters
    parser.add_argument("--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)")

    parser.add_argument("--lr", type=float, default=None, metavar="LR", help="learning rate (absolute lr)")
    parser.add_argument(
        "--blr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--min_lr", type=float, default=0.0, metavar="LR", help="lower lr bound for cyclic schedulers that hit 0"
    )

    parser.add_argument("--warmup_epochs", type=int, default=40, metavar="N", help="epochs to warmup LR")

    # Dataset parameters
    parser.add_argument("--data_path", default="/instruction_dataset/", type=str, help="dataset path")

    parser.add_argument("--output_dir", default="./output_dir", help="path where to save, empty for no saving")
    parser.add_argument("--log_dir", default="./output_dir", help="path where to tensorboard log")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--step_mode", default=False, type=bool, help="if use step mode training")
    parser.add_argument("--eval", default=False, type=bool, help="clear optimizer")

    return parser


def main(args):

    # misc.init_distributed_mode(args)
    args.distributed = False

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    torch.multiprocessing.set_start_method("spawn")
    device = torch.device(args.device)
    

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    arth_args = ArthModelArgs()
    arth_args.device = args.device
    arth_args.max_batch_size = args.batch_size
    arth_args.dim=16
    arth_args.max_seq_len=16
    arth_args.output_steps = args.step_mode

    dataset_train = SimpleNumberDataset(
        args.llama_model_path, arth_args, record_size=100000, max_integer=100000, decimal_precise=0.000001, partition='train'
    )
    dataset_val = SimpleNumberDataset(
        args.llama_model_path, arth_args, record_size=10000, max_integer=100000, decimal_precise=0.000001, partition='val'
    )

    print(dataset_train)
    print(dataset_val)

    if args.distributed:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.RandomSampler(dataset_val)

    os.makedirs(args.log_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=args.log_dir)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # define the model
    model = arthmetic_model.model.Arth_Model(arth_args)

    model = model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    for name, param in model.named_parameters():
        print(name, ", requires_grad", param.requires_grad)
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
            data_loader_val.sampler.set_epoch(epoch)

        with torch.set_grad_enabled(True):
            train_stats = arth_train_one_epoch(
                model, data_loader_train, optimizer, device, epoch, loss_scaler, log_writer=log_writer, args=args
            )

        val_stats = arth_val_one_epoch(
            model, data_loader_val, optimizer, device, epoch, loss_scaler, log_writer=log_writer, args=args
        )

        if args.output_dir and (epoch % 8 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
            **{f"val_{k}": v for k, v in val_stats.items()},
        }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":

    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
