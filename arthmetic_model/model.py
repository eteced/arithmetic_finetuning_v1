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
    arth_tau: float = 0.2

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
    device: str = "gpu"

class ArthTextToDenseBlock(nn.Module):
    def __init__(self, layer_id: int, args: ArthModelArgs):
        super().__init__()
        self.dim = args.dim
        self.layer_id = layer_id
        self.max_batch_size = args.max_batch_size
        self.arth_tau = args.arth_tau
        self.device=args.device
        self.create_fixed_values(args.tdtype, args.llm_emb_dtype)
        self.create_math_nn_s(args.ndtype)


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
            nn.SiLU(),
            nn.Linear(self.dim, self.dim // 2),
            nn.SiLU(),
            nn.Linear(self.dim // 2, 2),
        )
        # if move to next or not, 0 remain, 1 move to next, 2 move to next, sets op, and move to next again
        self.moved_nn = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim // 2),
            nn.SiLU(),
            nn.Linear(self.dim // 2, 3),
        )
        # the operator predict, here we directly move the predict result to final
        self.op_nn = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim // 2),
            nn.SiLU(),
            nn.Linear(self.dim // 2, 7),
        )
        # if decimal start
        self.decimal_start_nn = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim // 2),
            nn.SiLU(),
            nn.Linear(self.dim // 2, 2),
        )
        # dense handle, three op, trans_dense directly add pred_num, trans_dense * 10 and add pred_num, float_now * 0.1 and trans_dense + float_now * pred_num
        self.dense_op_nn = nn.Sequential(
            nn.Linear(self.dim + 1, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim // 2),
            nn.SiLU(),
            nn.Linear(self.dim // 2, 4),
        )

        # dense pred, 0~9, because there are many gates, here we directly predict 0~9 without extra invalid flag
        self.dense_pred_nn =  nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim // 2),
            nn.SiLU(),
            nn.Linear(self.dim // 2, 10),
        )

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
        # x should be a 3D tensor, batch_size X token_nums X dims
        for i in range(start_pos, x.shape[1]):
            token_vec = x[:,i,:] # B x dim
            ignore_gate = F.gumbel_softmax(self.token_valid_nn(token_vec), tau=self.arth_tau, hard=True)[:, 1] # B
            tmp_moved_gate = F.gumbel_softmax(self.moved_nn(token_vec), tau=self.arth_tau, hard=True) # B x 3
            move_next_gate = torch.max(tmp_moved_gate[:, 1:], dim=1).values  # B
            op_set_and_move_gate = tmp_moved_gate[:, 2] # B
            remain_gate = tmp_moved_gate[:, 0] # B
            float_decimal_start_flag = float_decimal_start.view(-1, 1)
            dense_op_gate = F.gumbel_softmax(self.dense_op_nn(torch.cat((token_vec, float_decimal_start_flag), dim=-1)), tau=self.arth_tau, hard=True) # B x 4
            dense_map_logits = self.dense_pred_nn(token_vec)
            dense_maps = F.gumbel_softmax(dense_map_logits, tau=self.arth_tau, hard=True) # B x 10
            op_pred = self.op_nn(token_vec) # B x op_dims
            
            # if this token is an op, we would like to move next and set op_pred
            pos_tmp_gate = (torch.ones_like(ignore_gate) - ignore_gate) * op_set_and_move_gate
            pos_tmp_apply_gate = pos_tmp_gate.view(-1, 1)
            pos_tmp_apply_gate = torch.tile(pos_tmp_apply_gate, (1, self.dim))
            pos_tmp_gate = pos_tmp_gate.view(-1, 1)
            pos_tmp_gate = torch.tile(pos_tmp_gate, (1, self.dim))
            pos_handle_mask_update = torch.matmul(pos_handle_mask, self.next_mask) * pos_tmp_gate
            pos_handle_mask = pos_tmp_apply_gate * pos_handle_mask_update + (torch.ones_like(pos_tmp_apply_gate) - pos_tmp_apply_gate) * pos_handle_mask
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
            float_point_mem_update = torch.ones_like(float_point_mem) * (move_next_gate) + (torch.ones_like(move_next_gate) - move_next_gate) * float_point_mem \
                              + torch.ones_like(dense_op_gate[:, 3]) * (torch.ones_like(move_next_gate) - move_next_gate) * float_point_mem * 0.1 + (torch.ones_like(dense_op_gate[:, 3]) - dense_op_gate[:, 3]) * (torch.ones_like(move_next_gate) - move_next_gate) * float_point_mem
            float_point_mem_update = dense_op_gate[:, 0] * float_point_mem + (torch.ones_like(dense_op_gate[:, 0]) - dense_op_gate[:, 0]) * float_point_mem_update
            float_point_mem = ignore_gate * float_point_mem + (torch.ones_like(ignore_gate) - ignore_gate) * float_point_mem_update
            # also reset
            float_decimal_start_update = torch.zeros_like(float_decimal_start) * (move_next_gate) + (torch.ones_like(move_next_gate) - move_next_gate) * float_decimal_start
            float_decimal_start = ignore_gate * float_decimal_start_update + (torch.ones_like(ignore_gate) - ignore_gate) * float_decimal_start_update

            # update dense
            tmp_dense_op_identical = torch.tile(((torch.ones_like(ignore_gate) - ignore_gate) * dense_op_gate[:, 1]).view(-1, 1), (1,self.dim)) * pos_handle_mask
            tmp_dense_op_add = torch.tile(((torch.ones_like(ignore_gate) - ignore_gate) * dense_op_gate[:, 2]).view(-1, 1), (1,self.dim)) * pos_handle_mask
            tmp_dense_op_decimal = torch.tile(((torch.ones_like(ignore_gate) - ignore_gate) * dense_op_gate[:, 3]).view(-1, 1), (1,self.dim)) * pos_handle_mask
            tmp_dense_no_change = torch.tile(((torch.ones_like(ignore_gate) - ignore_gate) * dense_op_gate[:, 0]).view(-1, 1), (1,self.dim)) * pos_handle_mask
            new_value = tmp_dense_op_identical * torch.tile(torch.sum(dense_maps * self.fixed_map_tensor).view(-1, 1), (1, self.dim)) \
                        + tmp_dense_op_add * (10 * trans_dense + torch.tile(torch.sum(dense_maps * self.fixed_map_tensor).view(-1, 1), (1, self.dim))) \
                        + tmp_dense_op_decimal * (trans_dense + torch.tile(torch.sum(dense_maps * self.fixed_map_tensor).view(-1, 1), (1, self.dim)) * torch.tile(float_point_mem.view(-1, 1), (1,self.dim))) \
                        + tmp_dense_no_change * trans_dense + torch.tile(ignore_gate.view(-1, 1), (1, self.dim)) * trans_dense
            trans_dense = pos_handle_mask * new_value + (torch.ones_like(pos_handle_mask) - pos_handle_mask) * trans_dense

            # update the valid flag
            trans_valid_ord = trans_valid * pos_handle_mask
            op_valid = torch.tile(op_set_and_move_gate.view(-1, 1), (1, self.dim)) * pos_handle_mask
            dense_valid = torch.tile(torch.max(dense_op_gate[:,1:], dim=-1).values.view(-1, 1), (1, self.dim)) * pos_handle_mask
            trans_valid_gate_stack = torch.concat([trans_valid_ord, op_valid, dense_valid], dim=-1)
            trans_valid_update = torch.tile(torch.max(trans_valid_gate_stack, dim=-1).values.view(-1, 1), (1, self.dim)) * pos_handle_mask
            trans_valid = torch.tile(ignore_gate.view(-1, 1), (1, self.dim)) * trans_dense + torch.tile((torch.ones_like(ignore_gate) - ignore_gate).view(-1, 1), (1,self.dim)) * trans_valid_update

            # move to next
            pos_tmp_gate = (torch.ones_like(ignore_gate) - ignore_gate) * move_next_gate
            pos_tmp_apply_gate = pos_tmp_gate.view(-1, 1)
            pos_tmp_apply_gate = torch.tile(pos_tmp_apply_gate, (1, self.dim))
            pos_tmp_gate = pos_tmp_gate.view(-1, 1)
            pos_tmp_gate = torch.tile(pos_tmp_gate, (1, self.dim))
            pos_handle_mask_update = torch.matmul(pos_handle_mask, self.next_mask) * pos_tmp_gate
            pos_handle_mask = pos_tmp_apply_gate * pos_handle_mask_update + (torch.ones_like(pos_tmp_apply_gate) - pos_tmp_apply_gate) * pos_handle_mask

            # reset again to ensure no history value when reset
            # update float point mem, when move to next, the float_point_mem is reset to one, when dense_op_gate[:, 3] == 1, float_point_mem = float_point_mem * 0.1
            float_point_mem_update = torch.ones_like(float_point_mem) * (move_next_gate) + (torch.ones_like(move_next_gate) - move_next_gate) * float_point_mem \
                              + torch.ones_like(dense_op_gate[:, 3]) * (torch.ones_like(move_next_gate) - move_next_gate) * float_point_mem * 0.1 + (torch.ones_like(dense_op_gate[:, 3]) - dense_op_gate[:, 3]) * (torch.ones_like(move_next_gate) - move_next_gate) * float_point_mem
            float_point_mem_update = dense_op_gate[:, 0] * float_point_mem + (torch.ones_like(dense_op_gate[:, 0]) - dense_op_gate[:, 0]) * float_point_mem_update
            float_point_mem = ignore_gate * float_point_mem + (torch.ones_like(ignore_gate) - ignore_gate) * float_point_mem_update
            # also reset
            float_decimal_start_update = torch.zeros_like(float_decimal_start) * (move_next_gate) + (torch.ones_like(move_next_gate) - move_next_gate) * float_decimal_start_update
            float_decimal_start = ignore_gate * float_decimal_start_update + (torch.ones_like(ignore_gate) - ignore_gate) * float_decimal_start_update

        return trans_valid, trans_dense, trans_op

class Arth_Model(nn.Module):
    def __init__(self, params: ArthModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size

        self.tok_embeddings = torch.nn.Embedding(
            params.vocab_size, params.dim, device=params.device)

        self.artd = ArthTextToDenseBlock(0, params)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        trans_valid, trans_dense, trans_op = self.artd(h, start_pos, mask)
        return trans_valid, trans_dense, trans_op
