#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This software may be used and distributed according to the terms of the GNU General Public License version 3.
"""ArthModel with LLaMA train one epoch
"""

import math
import sys
from typing import Iterable

import torch
import util.lr_sched as lr_sched
import util.misc as misc

output_steps = True
big_debug = False
VAL_DEBUG = False
TRAINING_DEBUG = False

def gen_manual_aux_info(text : torch.Tensor, batch_index=1):
    steps_ignore_logits=[]
    steps_tmp_moved_logits=[]
    steps_dense_op_logits=[]
    steps_dense_map_logits=[]
    steps_decimal_start_logits=[]
    steps_op_pred=[]
    fp_flag = 0
    one_text = text[0, :]
    dict_vocb_map = {
        1:21, 2:22, 29871:23, 29900:0, 29896:1, 29906:2, 29941:3, 29946:4, 29945:5, 29953:6, 29955:7, 29947:8, 29929:9, 29889: 10, 718: 11, 448:12, 334:13, 847:14, 321:15,
        6228:16, 313:17, 1723:18
    }
    for i in range(one_text.shape[0]):
        ax=one_text[i]
        if isinstance(ax, torch.Tensor):
            ax=ax.item()
        if ax in dict_vocb_map:
            xx=dict_vocb_map[ax]
        else:
            xx=20
        if big_debug:
            print(">> i ", i , "ax:", ax, "xx:", xx)
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
        else:
            steps_ignore_logits.append(1)
            steps_tmp_moved_logits.append(0)
            steps_dense_op_logits.append(0)
            steps_dense_map_logits.append(0)
            steps_decimal_start_logits.append(0)
            steps_op_pred.append(0)
    if big_debug:
        print("one_text", one_text)
        print("in_steps_ignore_logits", steps_ignore_logits)
        print("in_steps_tmp_moved_logits", steps_tmp_moved_logits)
        print("in_steps_dense_op_logits", steps_dense_op_logits)
        print("in_steps_dense_map_logits", steps_dense_map_logits)
        print("in_steps_decimal_start_logits", steps_decimal_start_logits)
        print("in_steps_op_pred", steps_op_pred)
    return steps_ignore_logits, steps_tmp_moved_logits, steps_dense_op_logits, steps_dense_map_logits, steps_decimal_start_logits, steps_op_pred

def loss_generate(output, h_gate_logits, h_arth_output, labels, example_mask, swift_tokens, swift_valids, enable_arthstep=False, steps_ignore_logits=None, steps_tmp_moved_logits=None, steps_dense_op_logits=None, steps_dense_map_logits=None, steps_decimal_start_logits=None, steps_op_pred=None):
    normal_criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    aux_criterion = torch.nn.CrossEntropyLoss() # c is always in dim 1
    
    output = output.reshape(-1, output.shape[2])
    labels = labels[:, 1:].flatten().to(output.device)
    if big_debug:
        print('output.shape', output.shape)
        print('labels.shape', labels)
    loss_normal = torch.mean(normal_criterion(output, labels)) *  10
    swift_valids = swift_valids.type(torch.LongTensor).flatten()
    if big_debug:
        print('h_gate_logits.shape', h_gate_logits.shape)
        print('swift_valids.shape', swift_valids.shape)
        print('h_gate_logits', h_gate_logits)
        print('swift_valids', swift_valids)
    loss_arth_gate = torch.mean(aux_criterion(h_gate_logits, swift_valids.to(h_gate_logits.device)))
    # loss_normal = torch.tensor([0.0], requires_grad=True).to(loss_arth_gate)
    h_arth_output = h_arth_output.reshape(-1, h_arth_output.shape[2])
    swift_tokens_calc = swift_tokens.type(torch.LongTensor).flatten()
    if big_debug:
        print('h_arth_output.shape', h_arth_output.shape)
        print('swift_tokens_calc.shape', swift_tokens_calc.shape)
        print('swift_tokens.shape', swift_tokens.shape)
        print('h_arth_output', torch.argmax(h_arth_output, dim=-1))
        print("swift_tokens_calc", swift_tokens_calc)
        print('swift_valids', swift_valids)
    if torch.sum(swift_tokens_calc) > 0:
        # all ignore labels will let normal_criterion produce nan
        # loss_arth_mid_result = torch.mean(normal_criterion(h_arth_output, swift_tokens_calc.to(h_arth_output.device)) * swift_valids.to(h_arth_output))  *  50
        loss_arth_mid_result = torch.mean(aux_criterion(h_arth_output, swift_tokens_calc.to(h_arth_output.device)) * swift_valids.to(h_arth_output))  *  50
    else:
        loss_arth_mid_result = torch.tensor([0.0], requires_grad=True).to(loss_arth_gate)
    if big_debug:
        print("loss_arth_mid_result", loss_arth_mid_result)
    arth_tau = 0.1
    if enable_arthstep:
        loss_cp = torch.nn.CrossEntropyLoss()
        for tt in range(swift_tokens.shape[0]):
            l_steps_ignore_logits, l_steps_tmp_moved_logits, l_steps_dense_op_logits, l_steps_dense_map_logits, l_steps_decimal_start_logits, l_steps_op_pred = gen_manual_aux_info(swift_tokens, tt)
            if big_debug:
                print("swift_tokens", swift_tokens)
                print("l_steps_ignore_logits", l_steps_ignore_logits)
                print("l_steps_tmp_moved_logits", l_steps_tmp_moved_logits)
                print("l_steps_dense_op_logits", l_steps_dense_op_logits)
                print("l_steps_dense_map_logits", l_steps_dense_map_logits)
                print("l_steps_decimal_start_logits", l_steps_decimal_start_logits)
                print("l_steps_op_pred", l_steps_op_pred)
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
            loss_steps_ignore_logits = None
            loss_steps_tmp_moved_logits = None
            loss_steps_dense_op_logits = None
            loss_steps_dense_map_logits = None
            loss_steps_steps_decimal_start_logits = None
            loss_steps_steps_op_pred = None
            for i in range(len(l_steps_ignore_logits)):
                if loss_steps_ignore_logits is None:
                    loss_steps_ignore_logits = loss_cp(o_steps_ignore_logits[i] / arth_tau, torch.tensor(l_steps_ignore_logits[i], dtype=torch.long).to(o_steps_ignore_logits[i].device))
                    loss_steps_tmp_moved_logits = loss_cp(o_steps_tmp_moved_logits[i] / arth_tau, torch.tensor(l_steps_tmp_moved_logits[i], dtype=torch.long).to(o_steps_tmp_moved_logits[i].device))
                    loss_steps_dense_op_logits = loss_cp(o_steps_dense_op_logits[i] / arth_tau, torch.tensor(l_steps_dense_op_logits[i], dtype=torch.long).to(o_steps_dense_op_logits[i].device))
                    loss_steps_dense_map_logits = loss_cp(o_steps_dense_map_logits[i] / arth_tau, torch.tensor(l_steps_dense_map_logits[i], dtype=torch.long).to(o_steps_dense_map_logits[i].device))
                    loss_steps_steps_decimal_start_logits = loss_cp(o_steps_decimal_start_logits[i] / arth_tau, torch.tensor(l_steps_decimal_start_logits[i], dtype=torch.long).to(o_steps_decimal_start_logits[i].device))
                    loss_steps_steps_op_pred = loss_cp(o_steps_op_pred[i] / arth_tau, torch.tensor(l_steps_op_pred[i], dtype=torch.long).to(o_steps_op_pred[i].device))
                else:
                    loss_steps_ignore_logits += loss_cp(o_steps_ignore_logits[i] / arth_tau, torch.tensor(l_steps_ignore_logits[i], dtype=torch.long).to(o_steps_ignore_logits[i].device))
                    loss_steps_tmp_moved_logits += loss_cp(o_steps_tmp_moved_logits[i] / arth_tau, torch.tensor(l_steps_tmp_moved_logits[i], dtype=torch.long).to(o_steps_tmp_moved_logits[i].device))
                    loss_steps_dense_op_logits += loss_cp(o_steps_dense_op_logits[i] / arth_tau, torch.tensor(l_steps_dense_op_logits[i], dtype=torch.long).to(o_steps_dense_op_logits[i].device))
                    loss_steps_dense_map_logits += loss_cp(o_steps_dense_map_logits[i] / arth_tau, torch.tensor(l_steps_dense_map_logits[i], dtype=torch.long).to(o_steps_dense_map_logits[i].device))
                    loss_steps_steps_decimal_start_logits += loss_cp(o_steps_decimal_start_logits[i] / arth_tau, torch.tensor(l_steps_decimal_start_logits[i], dtype=torch.long).to(o_steps_decimal_start_logits[i].device))
                    loss_steps_steps_op_pred += loss_cp(o_steps_op_pred[i] / arth_tau, torch.tensor(l_steps_op_pred[i], dtype=torch.long).to(o_steps_op_pred[i].device))
                if big_debug:
                    print('>> l >', i, "steps_decimal_start_logits", steps_decimal_start_logits[i],"l_steps_decimal_start_logits", l_steps_decimal_start_logits[i])
                    print('>> l >', i, "o_steps_dense_op_logits[i]", o_steps_dense_op_logits[i], "l_steps_dense_op_logits[i]", l_steps_dense_op_logits[i])
            del o_steps_ignore_logits, o_steps_tmp_moved_logits, o_steps_dense_op_logits, o_steps_dense_map_logits, o_steps_decimal_start_logits, o_steps_op_pred
            del normal_criterion, aux_criterion
        loss_arthstep = loss_steps_ignore_logits * 5 + loss_steps_tmp_moved_logits + loss_steps_dense_op_logits + loss_steps_dense_map_logits + loss_steps_steps_decimal_start_logits * 5 + loss_steps_steps_op_pred
        if big_debug:
            print("loss_normal", loss_normal, "loss_arth_gate", loss_arth_gate, "loss_arthstep", loss_arthstep, "loss_steps_ignore_logits", loss_steps_ignore_logits, "loss_steps_tmp_moved_logits", loss_steps_tmp_moved_logits, "loss_steps_dense_op_logits", loss_steps_dense_op_logits, "loss_steps_dense_map_logits", loss_steps_dense_map_logits, "loss_steps_steps_decimal_start_logits", loss_steps_steps_decimal_start_logits, "loss_steps_steps_op_pred", loss_steps_steps_op_pred)
        loss_arth_mid_result = loss_arth_mid_result + loss_arthstep
        # loss_arth_mid_result = loss_arthstep
    if big_debug:
        print("loss_normal", loss_normal, "loss_arth_gate", loss_arth_gate, "loss_arth_mid_result", loss_arth_mid_result)
    loss = loss_normal + loss_arth_gate + loss_arth_mid_result
    if big_debug:
        print("loss", loss, "loss_normal", loss_normal, "loss_arth_gate", loss_arth_gate, "loss_arth_mid_result", loss_arth_mid_result)
    return loss, loss_normal, loss_arth_gate, loss_arth_mid_result

def joined_train_one_epoch(
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
    for data_iter_step, (examples, labels, example_mask, swift_tokens, swift_valids) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
        # data_loader
    ):
        # print('examples', examples)
        # print('labels', labels)
        # print('swift_tokens', swift_tokens)
        # print('swift_valids', swift_valids)
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        if output_steps:
            # print(model)
            output, h_gate_logits, h_arth_output, steps_ignore_logits, steps_tmp_moved_logits, steps_dense_op_logits, steps_dense_map_logits, steps_decimal_start_logits, steps_op_pred, arth_result_tokens = model(examples, example_mask)
            loss, loss_normal, loss_arth_gate, loss_arth_mid_result = loss_generate(output, h_gate_logits, h_arth_output, labels, example_mask, swift_tokens, swift_valids, True, steps_ignore_logits, steps_tmp_moved_logits, steps_dense_op_logits, steps_dense_map_logits, steps_decimal_start_logits, steps_op_pred)
        else:
            output, h_gate_logits, h_arth_output = model(examples, example_mask)
            loss, loss_normal, loss_arth_gate, loss_arth_mid_result = loss_generate(output, h_gate_logits, h_arth_output, labels, example_mask, swift_tokens, swift_valids)
        loss_value = loss.item()
        loss_normal_value = loss_normal.item()
        loss_arth_gate_value = loss_arth_gate.item()
        loss_arth_mid_result_value = loss_arth_mid_result.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter

        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=(data_iter_step + 1) % accum_iter == 0)
        optimizer.zero_grad()

        torch.cuda.synchronize()

        if output_steps:
            del output, h_gate_logits, h_arth_output, steps_ignore_logits, steps_tmp_moved_logits, steps_dense_op_logits, steps_dense_map_logits, steps_decimal_start_logits, steps_op_pred
            del loss, loss_normal, loss_arth_gate, loss_arth_mid_result
            # del examples, labels, example_mask, swift_tokens, swift_valids
        else:
            del output, h_gate_logits, h_arth_output
            del loss, loss_normal, loss_arth_gate, loss_arth_mid_result
            # del examples, labels, example_mask, swift_tokens, swift_valids
        metric_logger.update(closs=loss_value)
        metric_logger.update(loss_normal_value=loss_normal_value)
        metric_logger.update(loss_arth_gate_value=loss_arth_gate_value)
        metric_logger.update(loss_arth_mid_result=loss_arth_mid_result_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        misc.all_reduce_mean(loss_value)
        misc.all_reduce_mean(loss_normal_value)
        misc.all_reduce_mean(loss_arth_gate_value)
        misc.all_reduce_mean(loss_arth_mid_result_value)
        c_loss_value_reduce = misc.all_reduce_mean(loss_value)
        c_loss_normal_value_reduce = misc.all_reduce_mean(loss_normal_value)
        c_loss_arth_gate_value_reduce = misc.all_reduce_mean(loss_arth_gate_value)
        c_loss_arth_mid_result_reduce = misc.all_reduce_mean(loss_arth_mid_result_value)


        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("c_train_loss", c_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("c_loss_normal_value", c_loss_normal_value_reduce, epoch_1000x)
            log_writer.add_scalar("c_loss_arth_gate_value", c_loss_arth_gate_value_reduce, epoch_1000x)
            log_writer.add_scalar("c_loss_arth_mid_result", c_loss_arth_mid_result_reduce, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)
        # finalize
        del examples, labels, example_mask, swift_tokens, swift_valids

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def joined_val_one_epoch(
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
    for data_iter_step, (examples, labels, example_mask, swift_tokens, swift_valids) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
        # data_loader
    ):
        # examples[0, :] = torch.Tensor([    1,  3529,   274,   562,  5987,   445, 29889,   396, 29871, 29941,
        #  29889, 29945,   334, 29871, 29906, 29889, 29955,   718, 29871, 29947,
        #  29889, 29953,   353,  1577,   395,     29871,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0]).to(examples)
        # labels = torch.zeros_like(examples)
        # example_mask[0, :] = torch.Tensor([    1,  1,   1,   1,  1,   1, 1,   1, 1, 1,
        #      1, 1,   1, 1, 1, 1, 1,   1, 1, 1,
        #      1, 1,   1,  1,   1,     1,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #      0,     0]).to(example_mask)
        with torch.no_grad():
            if output_steps:
                output, h_gate_logits, h_arth_output, steps_ignore_logits, steps_tmp_moved_logits, steps_dense_op_logits, steps_dense_map_logits, steps_decimal_start_logits, steps_op_pred, arth_result_tokens = model(examples, example_mask)
                loss, loss_normal, loss_arth_gate, loss_arth_mid_result = loss_generate(output, h_gate_logits, h_arth_output, labels, example_mask, swift_tokens, swift_valids, True, steps_ignore_logits, steps_tmp_moved_logits, steps_dense_op_logits, steps_dense_map_logits, steps_decimal_start_logits, steps_op_pred)
            else:
                output, h_gate_logits, h_arth_output = model(examples, example_mask)
                loss, loss_normal, loss_arth_gate, loss_arth_mid_result = loss_generate(output, h_gate_logits, h_arth_output, labels, example_mask, swift_tokens, swift_valids)
        loss_value = loss.item()
        loss_normal_value = loss_normal.item()
        loss_arth_gate_value = loss_arth_gate.item()
        loss_arth_mid_result = loss_arth_mid_result.item()
        if VAL_DEBUG:
            print("examples", examples)
            print("labels", labels)
            print("example_mask * example", (examples * example_mask).long())
            print("swift_tokens", swift_tokens)
            print("h_gate_logits", h_gate_logits)
            print("h_arth_output", torch.argmax(h_arth_output, dim=-1))
            print("arth_result_tokens", arth_result_tokens)
            print("output_argmax", torch.argmax(output, dim=-1))
            print("steps_ignore_logits", steps_ignore_logits)
            print("steps_tmp_moved_logits", steps_tmp_moved_logits)
            print("steps_dense_map_logits", steps_dense_map_logits)
            print("steps_decimal_start_logits", steps_decimal_start_logits)
            print("steps_op_pred", steps_op_pred)
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        metric_logger.update(closs=loss_value)
        metric_logger.update(loss_normal_value=loss_normal_value)
        metric_logger.update(loss_arth_gate_value=loss_arth_gate_value)
        metric_logger.update(loss_arth_mid_result=loss_arth_mid_result)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        misc.all_reduce_mean(loss_value)
        misc.all_reduce_mean(loss_normal_value)
        misc.all_reduce_mean(loss_arth_gate_value)
        misc.all_reduce_mean(loss_arth_mid_result)
        c_loss_value_reduce = misc.all_reduce_mean(loss_value)
        c_loss_normal_value_reduce = misc.all_reduce_mean(loss_normal_value)
        c_loss_arth_gate_value_reduce = misc.all_reduce_mean(loss_arth_gate_value)
        c_loss_arth_mid_result_reduce = misc.all_reduce_mean(loss_arth_mid_result)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("c_train_loss", c_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("c_loss_normal_value", c_loss_normal_value_reduce, epoch_1000x)
            log_writer.add_scalar("c_loss_arth_gate_value", c_loss_arth_gate_value_reduce, epoch_1000x)
            log_writer.add_scalar("c_loss_arth_mid_result", c_loss_arth_mid_result_reduce, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
