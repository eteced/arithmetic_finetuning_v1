#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This software may be used and distributed according to the terms of the GNU General Public License version 3.
"""Small Models for Arthmetic
"""
__author__ = "Yingdi Guo"
__license__ = "GPLv3"
__email__ = "eteced@gmail.com"

from typing import Optional, Tuple
from dataclasses import dataclass, field
import math

import torch
from torch import nn
import torch.nn.functional as F

import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ParallelEmbedding,
    RowParallelLinear,
    ColumnParallelLinear,
)


@dataclass
class ArthModelArgs:
    dim: int = 128
    trans_n_layers: int = 1
    trans_n_heads: int = 1
    calc_n_layers: int = 1
    calc_n_heads: int = 1
    vocab_size: int = 25  # 0 1 2 3 4 5 6 7 8 9 . + - * / e ^ ( ) [BOS]{21} [EOS]{22} [SEP/space]{23}
    norm_eps: float = 1e-5
    arth_tau: float = 0.001

    max_batch_size: int = 32
    max_seq_len: int = 128
    dict_vocb_rev_map: dict = field(default_factory=lambda: {
        21:1, 22:2, 23:29871, 0:29900, 1:29896, 2:29906, 3:29941, 4:29946, 5:29945, 6:29953, 7:29955, 8:29947, 9:29929, 10: 29889, 11: 718, 12: 448, 13:334, 14: 847, 15: 321,
        16:6228, 17: 313, 18: 1723
    })
    dict_vocb_map: dict = field(default_factory=lambda: {
        1:21, 2:22, 29871:23, 29900:0, 29896:1, 29906:2, 29941:3, 29946:4, 29945:5, 29953:6, 29955:7, 29947:8, 29929:9, 29889: 10, 718: 11, 448:12, 334:13, 847:14, 321:15,
        6228:16, 313:17, 1723:18
    })
    tdtype = torch.float64
    ndtype = torch.float16
    llm_emb_dtype = torch.float16
    device: str = "cuda"
    output_steps: bool = False
    debug_trace: bool =  False
    use_argmax: bool = True

class ArthTextToDenseBlock(nn.Module):
    def __init__(self, layer_id: int, args: ArthModelArgs):
        super().__init__()
        self.dim = args.dim
        self.layer_id = layer_id
        self.max_batch_size = args.max_batch_size
        self.arth_tau = args.arth_tau
        self.device=args.device
        self.output_steps = args.output_steps
        self.create_fixed_values(args.tdtype, args.llm_emb_dtype)
        self.create_math_nn_s(args.ndtype)
        self.args = args
        self.norm_eps = args.norm_eps
        self.use_argmax = args.use_argmax


    def create_fixed_values(self, tdype, llm_emb_dtype):
        self.next_mask = torch.zeros(self.dim, self.dim, dtype=tdype, device=self.device) # dim(T) x dim(T)
        for i in range(self.dim - 1):
            self.next_mask[i, i + 1] = 1
        # which arth slot is we currently worked on
        self.pos_handle_mask = torch.zeros(self.max_batch_size, self.dim, dtype=tdype, device=self.device)
        for i in range(self.max_batch_size):
            self.pos_handle_mask[i, 0] = 1
        self.trans_dense = torch.zeros(self.max_batch_size, self.dim, dtype=tdype, device=self.device) # B x dim(T)
        self.trans_valid = torch.zeros(self.max_batch_size, self.dim, dtype=tdype, device=self.device) # B x dim(T)
        # valid arth op, current [empty] 1 + - * / ^
        self.trans_op = torch.zeros(self.max_batch_size, self.dim, 7, dtype=tdype, device=self.device) # B x dim(T) x 7
        # if it's decimal part
        self.float_point_mem = torch.ones(self.max_batch_size, dtype=tdype, device=self.device)
        # if decimal start
        self.float_decimal_start = torch.zeros(self.max_batch_size, dtype=llm_emb_dtype, device=self.device)
        # oh, numbers
        self.fixed_map_tensor = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=tdype, device=self.device)
    
    def create_math_nn_s(self, ndtype):
        # token valid nn or skip nn, skip that token without any changes
        self.token_valid_nn = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.LeakyReLU(),
            nn.Linear(self.dim, self.dim // 2),
            nn.LeakyReLU(),
            nn.Linear(self.dim // 2, 2),
        )
        # if move to next or not, 0 remain, 1 move to next, 2 move to next, sets op, and move to next again
        self.moved_nn = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.LeakyReLU(),
            nn.Linear(self.dim, self.dim // 2),
            nn.LeakyReLU(),
            nn.Linear(self.dim // 2, 3),
        )
        # the operator predict, here we directly move the predict result to final
        self.op_nn = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.LeakyReLU(),
            nn.Linear(self.dim, self.dim // 2),
            nn.LeakyReLU(),
            nn.Linear(self.dim // 2, 7),
        )
        # if decimal start
        self.decimal_start_nn = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.LeakyReLU(),
            nn.Linear(self.dim, self.dim // 2),
            nn.LeakyReLU(),
            nn.Linear(self.dim // 2, 2),
        )
        # dense handle, three op, trans_dense directly add pred_num, trans_dense * 10 and add pred_num, float_now * 0.1 and trans_dense + float_now * pred_num
        self.dense_op_nn = nn.Sequential(
            nn.Linear(self.dim + self.dim, self.dim),
            nn.LeakyReLU(),
            nn.Linear(self.dim, self.dim // 2),
            nn.LeakyReLU(),
            nn.Linear(self.dim // 2, 4),
        )

        # dense pred, 0~9, because there are many gates, here we directly predict 0~9 without extra invalid flag
        self.dense_pred_nn =  nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.LeakyReLU(),
            nn.Linear(self.dim, self.dim // 2),
            nn.LeakyReLU(),
            nn.Linear(self.dim // 2, 10),
        )

    def argmax_onehot(self, logits):
        arg_a=torch.argmax(logits, dim=-1)
        return F.one_hot(arg_a, num_classes=logits.shape[-1])

    def forward(self, x: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor]):
        # the dense value for trans
        trans_dense = self.trans_dense.clone().detach().requires_grad_(True).to(self.device)
        # so the invalid mask (padding area) should have value zero
        trans_valid = self.trans_valid.clone().detach().requires_grad_(True).to(self.device)
        # operators, note that op 1 is identical op, means the position is a dense value
        trans_op = self.trans_op.clone().detach().requires_grad_(True).to(self.device)

        # internal gates/variables
        pos_handle_mask = self.pos_handle_mask.clone().detach().requires_grad_(True).to(self.device) # B x dim(T)
        float_point_mem = self.float_point_mem.clone().detach().requires_grad_(True).to(self.device) # B
        float_decimal_start = self.float_decimal_start.clone().detach().requires_grad_(True).to(self.device) # B
        steps_ignore_logits=[]
        steps_tmp_moved_logits=[]
        steps_dense_op_logits=[]
        steps_dense_map_logits=[]
        steps_decimal_start_logits=[]
        steps_op_pred=[]

        # x should be a 3D tensor, batch_size X token_nums X dims
        for i in range(start_pos, x.shape[1]):
            token_vec = x[:,i,:] # B x dim
            ignore_logits = self.token_valid_nn(token_vec)
            # normalize significantly prevent the simple training from coveraged, thus I hold the normalize operator
            ignore_logits = torch.nn.functional.normalize(ignore_logits, p=2, dim=-1, eps=self.norm_eps)
            if self.use_argmax:
                ignore_gate = self.argmax_onehot(ignore_logits)[:, 1]
            else:
                ignore_gate = F.gumbel_softmax(ignore_logits, tau=self.arth_tau, hard=True)[:, 1] # B
            move_logits = self.moved_nn(token_vec)
            move_logits = torch.nn.functional.normalize(move_logits, p=2, dim=-1, eps=self.norm_eps)
            if self.use_argmax:
                tmp_moved_gate = self.argmax_onehot(move_logits)
            else:
                tmp_moved_gate = F.gumbel_softmax(move_logits, tau=self.arth_tau, hard=True) # B x 3
            move_next_gate = torch.max(tmp_moved_gate[:, 1:], dim=1).values  # B
            op_set_and_move_gate = tmp_moved_gate[:, 2] # B
            remain_gate = tmp_moved_gate[:, 0] # B
            float_decimal_start_flag = float_decimal_start.view(-1, 1)
            float_decimal_start_flag = torch.tile(float_decimal_start_flag, (1, self.dim))
            dense_op_logits_input = torch.cat((token_vec, float_decimal_start_flag), dim=-1)
            if self.args.debug_trace == True:
                print("> -------------")
                print('>> ', i, "dense_op_logits_input", dense_op_logits_input)
            dense_op_logits = self.dense_op_nn(dense_op_logits_input)
            dense_op_logits = torch.nn.functional.normalize(dense_op_logits, p=2, dim=-1, eps=self.norm_eps)
            if self.use_argmax:
                dense_op_gate = self.argmax_onehot(dense_op_logits)
            else:
                dense_op_gate = F.gumbel_softmax(dense_op_logits, tau=self.arth_tau, hard=True) # B x 4
            dense_map_logits = self.dense_pred_nn(token_vec)
            dense_map_logits = torch.nn.functional.normalize(dense_map_logits, p=2, dim=-1, eps=self.norm_eps)
            if self.use_argmax:
                dense_maps = self.argmax_onehot(dense_map_logits)
            else:
                dense_maps = F.gumbel_softmax(dense_map_logits, tau=self.arth_tau, hard=True) # B x 10
            op_pred = self.op_nn(token_vec) # B x op_dims
            op_pred = torch.nn.functional.normalize(op_pred, p=2, dim=-1, eps=self.norm_eps)
            decimal_start_logits = self.decimal_start_nn(token_vec)
            decimal_start_logits = torch.nn.functional.normalize(decimal_start_logits, p=2, dim=-1, eps=self.norm_eps)
            if self.use_argmax:
                decimal_start_gate = self.argmax_onehot(decimal_start_logits)[:, 1]
            else:
                decimal_start_gate = F.gumbel_softmax(decimal_start_logits, tau=self.arth_tau, hard=True)[:, 1]

            if self.args.debug_trace == True:
                print("> ", i, " > ignore_logits", ignore_logits)
                print("> ", i, " > move_logits", move_logits)
                print("> ", i, " > dense_op_logits", dense_op_logits)
                print("> ", i, " > dense_map_logits", dense_map_logits)
                print("> ", i, " > op_pred", op_pred)
                print("> ", i, " > decimal_start_logits", decimal_start_logits)
                print("> ", i, " > move_next_gate", move_next_gate)
                print("> ", i, " > ignore_gate", ignore_gate)
                print("> ", i, " > op_set_and_move_gate", op_set_and_move_gate)
                print("> ", i, " > dense_op_gate", dense_op_gate)
                print("> ", i, " > dense_maps", dense_maps)
                print("> ", i, " > decimal_start_gate", decimal_start_gate)
                print("> ", i, " > 1: float_decimal_start", float_decimal_start)
            
            # if this token is an op, we would like to move next and set op_pred
            pos_tmp_gate = (torch.ones_like(ignore_gate) - ignore_gate) * op_set_and_move_gate
            pos_tmp_apply_gate = pos_tmp_gate.view(-1, 1)
            pos_tmp_apply_gate = torch.tile(pos_tmp_apply_gate, (1, self.dim))
            pos_tmp_gate = pos_tmp_gate.view(-1, 1)
            pos_tmp_gate = torch.tile(pos_tmp_gate, (1, self.dim))
            pos_handle_mask_update = torch.matmul(pos_handle_mask, self.next_mask) * pos_tmp_gate
            pos_handle_mask = pos_tmp_apply_gate * pos_handle_mask_update + (torch.ones_like(pos_tmp_apply_gate) - pos_tmp_apply_gate) * pos_handle_mask
            if self.args.debug_trace == True:
                print("> ", i, " > 1: pos_handle_mask", pos_handle_mask)
            # save to trans op
            tmp_trans_op_gate = (torch.ones_like(ignore_gate) - ignore_gate) * op_set_and_move_gate
            tmp_trans_op_gate = tmp_trans_op_gate.view(-1, 1)
            tmp_trans_op_gate = torch.tile(tmp_trans_op_gate, (1, self.dim))
            tmp_trans_op_gate = tmp_trans_op_gate * pos_handle_mask
            tmp_trans_op_gate = tmp_trans_op_gate.view(-1, self.dim, 1)
            tmp_trans_op_gate = torch.tile(tmp_trans_op_gate, (1, 1, 7))
            op_pred_pos = torch.tile(op_pred.view(-1, 1, 7), (1, self.dim, 1))
            trans_op = trans_op + tmp_trans_op_gate * op_pred_pos

            # update float point mem, when move to next, the float_point_mem is reset to one, when dense_op_gate[:, 3] == 1, float_point_mem = float_point_mem * 0.1
            float_point_mem_update = torch.ones_like(float_point_mem) * (move_next_gate) + (torch.ones_like(move_next_gate) - move_next_gate) * ( \
                              dense_op_gate[:, 3] * 0.1 * float_point_mem + (torch.ones_like(dense_op_gate[:, 3]) - dense_op_gate[:, 3]) * float_point_mem)
            if self.args.debug_trace == True:
                print("> ", i, " > 1: float_point_mem_update", float_point_mem_update)
                print("> ", i, " > float_point_mem", float_point_mem)
            float_point_mem = ignore_gate * float_point_mem + (torch.ones_like(ignore_gate) - ignore_gate) * float_point_mem_update
            # also reset
            float_decimal_start_update = torch.zeros_like(float_decimal_start) * (move_next_gate) + (torch.ones_like(move_next_gate) - move_next_gate) * float_decimal_start
            float_decimal_start = ignore_gate * float_decimal_start_update + (torch.ones_like(ignore_gate) - ignore_gate) * float_decimal_start_update

            # update decimal start
            float_decimal_start_update = decimal_start_gate * torch.ones_like(float_decimal_start) + (torch.ones_like(decimal_start_gate) - decimal_start_gate) * float_decimal_start
            float_decimal_start = ignore_gate * float_decimal_start + (torch.ones_like(ignore_gate) - ignore_gate) * float_decimal_start_update

            if self.args.debug_trace == True:
                print("> ", i, " > float_point_mem", float_point_mem)
                print("> ", i, " > 2: float_decimal_start", float_decimal_start)

            # update dense
            tmp_dense_op_identical = torch.tile(((torch.ones_like(ignore_gate) - ignore_gate) * dense_op_gate[:, 1]).view(-1, 1), (1,self.dim)) * pos_handle_mask
            tmp_dense_op_add = torch.tile(((torch.ones_like(ignore_gate) - ignore_gate) * dense_op_gate[:, 2]).view(-1, 1), (1,self.dim)) * pos_handle_mask
            tmp_dense_op_decimal = torch.tile(((torch.ones_like(ignore_gate) - ignore_gate) * dense_op_gate[:, 3]).view(-1, 1), (1,self.dim)) * pos_handle_mask
            tmp_dense_no_change = torch.tile(((torch.ones_like(ignore_gate) - ignore_gate) * dense_op_gate[:, 0]).view(-1, 1), (1,self.dim)) * pos_handle_mask
            new_value = tmp_dense_op_identical * torch.tile(torch.sum(dense_maps * self.fixed_map_tensor, dim=-1).view(-1, 1), (1, self.dim)) \
                        + tmp_dense_op_add * (10 * trans_dense + torch.tile(torch.sum(dense_maps * self.fixed_map_tensor, dim=-1).view(-1, 1), (1, self.dim))) \
                        + tmp_dense_op_decimal * (trans_dense + torch.tile(torch.sum(dense_maps * self.fixed_map_tensor, dim=-1).view(-1, 1), (1, self.dim)) * torch.tile(float_point_mem.view(-1, 1), (1,self.dim))) \
                        + tmp_dense_no_change * trans_dense + torch.tile(ignore_gate.view(-1, 1), (1, self.dim)) * trans_dense
            trans_dense = pos_handle_mask * new_value + (torch.ones_like(pos_handle_mask) - pos_handle_mask) * trans_dense

            if self.args.debug_trace == True:
                print("> ", i, " > trans_dense", trans_dense)
            # update the valid flag
            trans_valid_ord = trans_valid * pos_handle_mask
            op_valid = torch.tile(op_set_and_move_gate.view(-1, 1), (1, self.dim)) * pos_handle_mask
            dense_valid = torch.tile(torch.max(dense_op_gate[:,1:], dim=-1).values.view(-1, 1), (1, self.dim)) * pos_handle_mask
            trans_valid_gate_stack = torch.concat([trans_valid_ord, op_valid, dense_valid], dim=-1)
            if self.args.debug_trace == True:
                print("> ", i, " > trans_valid", trans_valid)
                print("> ", i, " > trans_valid_ord", trans_valid_ord)
                print("> ", i, " > op_valid", op_valid)
                print("> ", i, " > dense_valid", dense_valid)
            trans_valid_update = torch.tile(torch.max(trans_valid_gate_stack, dim=-1).values.view(-1, 1), (1, self.dim)) * pos_handle_mask + (torch.ones_like(pos_handle_mask) - pos_handle_mask) * trans_valid
            trans_valid = torch.tile(ignore_gate.view(-1, 1), (1, self.dim)) * trans_valid + torch.tile((torch.ones_like(ignore_gate) - ignore_gate).view(-1, 1), (1,self.dim)) * trans_valid_update
            if self.args.debug_trace == True:
                print("> ", i, " > 2trans_valid", trans_valid)
            # move to next
            pos_tmp_gate = (torch.ones_like(ignore_gate) - ignore_gate) * move_next_gate
            pos_tmp_apply_gate = pos_tmp_gate.view(-1, 1)
            pos_tmp_apply_gate = torch.tile(pos_tmp_apply_gate, (1, self.dim))
            pos_tmp_gate = pos_tmp_gate.view(-1, 1)
            pos_tmp_gate = torch.tile(pos_tmp_gate, (1, self.dim))
            pos_handle_mask_update = torch.matmul(pos_handle_mask, self.next_mask) * pos_tmp_gate
            pos_handle_mask = pos_tmp_apply_gate * pos_handle_mask_update + (torch.ones_like(pos_tmp_apply_gate) - pos_tmp_apply_gate) * pos_handle_mask
            if self.args.debug_trace == True:
                print("> ", i, " > 2: pos_handle_mask", pos_handle_mask)

            # reset again to ensure no history value when reset
            # update float point mem, when move to next, the float_point_mem is reset to one, when dense_op_gate[:, 3] == 1, float_point_mem = float_point_mem * 0.1
            float_point_mem_update = torch.ones_like(float_point_mem) * (move_next_gate) + (torch.ones_like(move_next_gate) - move_next_gate) * float_point_mem
            float_point_mem = ignore_gate * float_point_mem + (torch.ones_like(ignore_gate) - ignore_gate) * float_point_mem_update
            if self.args.debug_trace == True:
                print("> ", i, " > 4: float_point_mem_update", float_point_mem_update)
                print("> ", i, " > float_point_mem", float_point_mem)
            # also reset
            float_decimal_start_update = torch.zeros_like(float_decimal_start) * (move_next_gate) + (torch.ones_like(move_next_gate) - move_next_gate) * float_decimal_start_update
            float_decimal_start = ignore_gate * float_decimal_start_update + (torch.ones_like(ignore_gate) - ignore_gate) * float_decimal_start_update

            if self.args.debug_trace == True:
                print("> ", i, " > op_pred", op_pred)

            if (self.output_steps):
                steps_ignore_logits.append(ignore_logits)
                steps_tmp_moved_logits.append(move_logits)
                steps_dense_op_logits.append(dense_op_logits)
                steps_dense_map_logits.append(dense_map_logits)
                steps_decimal_start_logits.append(decimal_start_logits)
                steps_op_pred.append(op_pred)
        if self.output_steps:
            return trans_valid, trans_dense, trans_op, steps_ignore_logits, steps_tmp_moved_logits, steps_dense_op_logits, steps_dense_map_logits, steps_decimal_start_logits, steps_op_pred
        else:
            return trans_valid, trans_dense, trans_op

class ArthDenseCalcToDenseBlock(nn.Module):
    def __init__(self, params: ArthModelArgs):
        super().__init__()
        self.args = params
        self.device = params.device
        self.arth_tau = params.arth_tau
        self.create_fixed_values()

    def create_fixed_values(self):
        self.hold_numbers = torch.zeros(self.args.max_batch_size, 2, dtype=self.args.tdtype)
        self.hold_status = torch.zeros(self.args.max_batch_size, 2, dtype=self.args.ndtype)
        self.hold_index = torch.zeros(self.args.max_batch_size, 2, dtype=torch.long)
        self.meet_op = torch.zeros(self.args.max_batch_size, dtype=self.args.ndtype)
        self.next_mask = torch.zeros(2, 2, dtype=self.args.ndtype, device=self.device) # 2 x 2
        self.next_mask[0, 1] = 1
        self.next_mask_long = torch.zeros(2, 2, dtype=torch.long, device=self.device) # 2 x 2
        self.next_mask_long[0, 1] = 1
    
    def argmax_onehot(self, logits):
        arg_a=torch.argmax(logits, dim=-1)
        return F.one_hot(arg_a, num_classes=logits.shape[-1])

    def forward(self, trans_valid : torch.Tensor, trans_dense : torch.Tensor, trans_op : torch.Tensor, if_finished : torch.Tensor, if_valid: torch.Tensor, start_pos = 0):
        # if_finished size = B, if_valid size = B, should be ether zeros or ones.
        hold_numbers = self.hold_numbers.clone().detach().requires_grad_(True).to(self.device) # B x dim(T)
        hold_status = self.hold_status.clone().detach().requires_grad_(True).to(self.device) # B x 2
        hold_index = self.hold_index.clone().detach().requires_grad_(False).to(self.device) # B x 2
        meet_op = torch.zeros(self.args.max_batch_size, dtype=self.args.ndtype).to(self.device)

        if self.args.debug_trace == True:
            print("---------------ArthDenseCalcToDenseBlock-------------------")
            print("trans_valid", trans_valid)
            print("trans_dense", trans_dense)
            print("trans_op", trans_op)
            print("---------------ArthDenseCalcToDenseBlock-Handle-BEGIN--------------")
        for i in range(start_pos, trans_valid.shape[1]):
            i_trans_valid = trans_valid[:, i] * (torch.ones_like(meet_op) - meet_op) * (torch.ones_like(if_finished) - if_finished)
            i_trans_dense = trans_dense[:, i]
            i_trans_op = trans_op[:, i, :]
            if self.args.use_argmax:
                i_op_which = self.argmax_onehot(i_trans_op)
            else:
                i_op_which = F.gumbel_softmax(i_trans_op, tau=self.arth_tau, hard=True) # B x op_dims
            not_op_gate = i_op_which[:, 0]

            # save denses
            dense_gate = not_op_gate * i_trans_valid
            dense_applied_gate = if_valid * dense_gate
            dense_applied = dense_applied_gate.view(-1, 1)
            dense_applied = torch.tile(dense_applied, (1, 2))
            hold_numbers = dense_applied.to(self.args.tdtype) *  torch.matmul(hold_numbers, self.next_mask.to(self.args.tdtype)) + (torch.ones_like(dense_applied.to(self.args.tdtype)) - dense_applied.to(self.args.tdtype)) * hold_numbers
            hold_status = dense_applied.to(hold_status) * torch.matmul(hold_status, self.next_mask.to(hold_status)) + (torch.ones_like(dense_applied.to(hold_status)) - dense_applied.to(hold_status)) * hold_status
            hold_index = dense_applied.long().cpu() *  torch.matmul(hold_index.cpu(), self.next_mask_long.cpu()) + (torch.ones_like(dense_applied.long().cpu()) - dense_applied.long().cpu()) * hold_index.cpu()
            hold_index = hold_index.to(self.args.device)
            
            if self.args.debug_trace == True:
                print("> ", i, "> not_op_gate", not_op_gate)
                print("> ", i, "> dense_applied_gate", dense_applied_gate)
                print("> ", i, "> hold_numbers", hold_numbers)
                print("> ", i, "> hold_status", hold_status)
                print("> ", i, "> hold_index", hold_index)

            hold_index[:, 0] = hold_index[:, 0] * (torch.ones_like(dense_applied_gate.long()) - dense_applied_gate.long()) + dense_applied_gate.long() * i
            hold_status[:, 0] = hold_status[:, 0] * (torch.ones_like(dense_applied_gate) - dense_applied_gate) + torch.ones_like(dense_applied_gate) * dense_applied_gate
            hold_numbers[:, 0] = hold_numbers[:, 0] * (torch.ones_like(dense_applied_gate) - dense_applied_gate) + torch.ones_like(dense_applied_gate) * dense_applied_gate * i_trans_dense

            if self.args.debug_trace == True:
                print("> ", i, "> 2 hold_numbers", hold_numbers)
                print("> ", i, "> 2 hold_status", hold_status)
                print("> ", i, "> 2 hold_index", hold_index)

            # handle ops
            op_gate = (torch.ones_like(not_op_gate) - not_op_gate) * i_trans_valid * if_valid
            if self.args.debug_trace == True:
                print("> ", i, "> op_gate", op_gate)
            meet_op = torch.ones_like(op_gate) * op_gate * (torch.ones_like(meet_op) - meet_op) + meet_op
            if self.args.debug_trace == True:
                print("> ", i, "> meet_op", meet_op)
            if_valid = op_gate * hold_status[:, 0] * hold_status[:, 1] + (torch.ones_like(op_gate) - op_gate) * if_valid
            dense_add = hold_numbers[:, 1] + hold_numbers[:, 0]
            dense_minus = hold_numbers[:, 1] - hold_numbers[:, 0]
            dense_mul = hold_numbers[:, 1] * hold_numbers[:, 0]
            dense_div = hold_numbers[:, 1] / (hold_numbers[:, 0] + 0.0000001)
            dense_pow = torch.pow(hold_numbers[:, 1], hold_numbers[:, 0])
            dense_zero = torch.zeros_like(hold_numbers[:, 0])
            dense_one = torch.zeros_like(hold_numbers[:, 0])
            dense_ops = torch.stack([dense_zero, dense_one, dense_add, dense_minus, dense_mul, dense_div, dense_pow], dim=-1)
            dense_calc = torch.sum(dense_ops * i_op_which, dim=-1)
            ## with if_valid update, we put the dense calc result to the place of second number and set the place of first number invalid
            final_op_gate = (torch.ones_like(not_op_gate) - not_op_gate) * i_trans_valid * if_valid
            one_hot_number2 = F.one_hot(hold_index[:, 0], num_classes = self.args.max_seq_len).float()
            one_hot_number1 = F.one_hot(hold_index[:, 1], num_classes = self.args.max_seq_len).float()
            final_op_gate_tile = torch.tile(final_op_gate.view(-1, 1), (1, self.args.max_seq_len))
            
            trans_valid = trans_valid * (torch.ones_like(final_op_gate_tile) - final_op_gate_tile) + trans_valid * final_op_gate_tile * (torch.ones_like(one_hot_number1) - one_hot_number1)
            dense_calc_tile = torch.tile(dense_calc.view(-1, 1), (1, self.args.max_seq_len))
            trans_dense = trans_dense * (torch.ones_like(final_op_gate_tile) - final_op_gate_tile) + trans_dense * final_op_gate_tile * (torch.ones_like(one_hot_number2) - one_hot_number2) + \
                          dense_calc_tile * final_op_gate_tile * one_hot_number2
            if self.args.debug_trace == True:
                print("> ", i, "> trans_valid", trans_valid)
            trans_valid[:, i] = trans_valid[:, i] * (torch.ones_like(final_op_gate) - final_op_gate)
            if self.args.debug_trace == True:
                print("> ", i, "> 2 trans_valid", trans_valid)
            # reset hold_status[:, 0]
            hold_status[:, 0] = hold_status[:, 0] * (torch.ones_like(final_op_gate) - final_op_gate)
            if self.args.debug_trace == True:
                print("> ", i, "> trans_dense", trans_dense)

        if_finished_update = if_valid * (torch.ones_like(meet_op) - meet_op) * (hold_status[:, 0]) * (torch.ones_like(hold_status[:, 0]) - hold_status[:, 1])
        if_finished = (torch.ones_like(if_finished) - if_finished) * if_finished_update + if_finished
        return trans_valid, trans_dense, trans_op, if_finished, if_valid

class ArthCalcModule(nn.Module):
    def __init__(self, params: ArthModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size

        self.tok_embeddings = torch.nn.Embedding(
            params.vocab_size, params.dim, device=params.device)

        self.output_steps = params.output_steps
        self.max_batch_size = params.max_batch_size
        self.max_seq_len = params.max_seq_len
        self.acc_blocks = []
        self.init_acc_block()
    
    def init_acc_block(self):
        for i in range(0, self.max_seq_len + 1):
            self.acc_blocks.append(ArthDenseCalcToDenseBlock(self.params))
    
    def forward(self, trans_valid: torch.Tensor, trans_dense: torch.Tensor, trans_op: torch.Tensor, start_pos = 0):
        if_finished = torch.zeros(self.max_batch_size)
        if_valid = torch.ones(self.max_batch_size)
        i = 0
        for layer in self.acc_blocks:
            if self.args.debug_trace == True:
                print("----------------- ", i, " -----------------")
            i=i+1
            trans_valid, trans_dense, trans_op, if_finished, if_valid = layer(trans_valid, trans_dense, trans_op, if_finished, if_valid, start_pos)
        return trans_valid, trans_dense, trans_op, if_finished, if_valid            

# class ArthDenseToTextEmbeddingModule(nn.Module):
#     def __init__(self, params: ArthModelArgs):
#         super().__init__()
#         self.params = params
#         self.vocab_size = params.vocab_size

#         self.tok_embeddings = torch.nn.Embedding(
#             params.vocab_size, params.dim, device=params.device)
    
#     def init_fixed_values(self):
#         self.next_mask = torch.zeros(self.params.max_seq_len, self.max_seq_len, dtype=torch.long, device=self.device) # dim(T) x dim(T)
#         for i in range(self.max_seq_len - 1):
#             self.next_mask[i, i + 1] = 1
#         self.result_put = torch.zeros(self.params.max_batch_size, self.params.max_seq_len, dtype=torch.long)
#         self.dense_convert_finish = torch.zeros(self.params.max_batch_size, dtype=self.params.ndtype)
    
#     def forward(self,  trans_valid: torch.Tensor, trans_dense: torch.Tensor, trans_op: torch.Tensor, start_pos = 0):
#         result_put = self.result_put.clone().detach().to(self.params.device)
#         for i in range(start_pos, trans_valid.shape[1]):

#         return result_put

class ArthDenseToTextEmbeddingModuleSimp(nn.Module):
    def __init__(self, params: ArthModelArgs, tokenizer):
        super().__init__()
        self.params = params
        self.tokenizer = tokenizer
        self.max_seq_len = params.max_seq_len

    def forward(self,  trans_valid: torch.Tensor, trans_dense: torch.Tensor, trans_op: torch.Tensor, start_pos = 0):
        float_values = torch.sum(trans_valid * trans_dense, -1).cpu().detach()
        float_str = float_values.detach().numpy().astype("str")
        lst_final_tokens=[]
        for i in range(len(float_str)):
            if self.params.debug_trace == True:
                print("> i > float_str", float_str)
            token = self.tokenizer.encode(str(float_str[i]), bos=True, eos=True)
            while len(token) < self.max_seq_len:
                token.append(0)
            token = token[:self.max_seq_len]
            lst_final_tokens.append(token)
        return torch.tensor(lst_final_tokens, dtype=torch.int64)

class Arth_Model(nn.Module):
    def __init__(self, params: ArthModelArgs, tokenizer):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size

        self.tok_embeddings = torch.nn.Embedding(
            params.vocab_size, params.dim, device=params.device)

        self.artd = ArthTextToDenseBlock(0, params)
        self.calc_block = ArthDenseCalcToDenseBlock(params)
        self.artt = ArthDenseToTextEmbeddingModuleSimp(params, tokenizer)
        self.output_steps = params.output_steps

    def forward(self, tokens: torch.Tensor, start_pos: int, tokens_is_index: True):
        if tokens_is_index:
            _bsz, seqlen = tokens.shape
            h = self.tok_embeddings(tokens)
        else:
            _bsz, seqlen, dim_ = tokens.shape
            h = tokens

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        if self.output_steps == False:
            trans_valid, trans_dense, trans_op = self.artd(h, start_pos, mask)
        else:
            trans_valid, trans_dense, trans_op, steps_ignore_logits, steps_tmp_moved_logits, steps_dense_op_logits, steps_dense_map_logits, steps_decimal_start_logits, steps_op_pred = self.artd(h, start_pos, mask)
        
        trans_valid, trans_dense, trans_op, if_finished, if_valid = self.calc_block(trans_valid, trans_dense, trans_op, torch.zeros(_bsz, dtype=self.params.ndtype).to(self.params.device), torch.ones(_bsz, dtype=self.params.ndtype).to(self.params.device))
        arth_tokens = self.artt(trans_valid, trans_dense, trans_op)
        if self.output_steps == False:
            return arth_tokens
        else:
            return arth_tokens, steps_ignore_logits, steps_tmp_moved_logits, steps_dense_op_logits, steps_dense_map_logits, steps_decimal_start_logits, steps_op_pred 