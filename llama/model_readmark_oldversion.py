# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

# Modified By Yingdi Guo as part of `arithmetric_finetuning_v1` project which is also under license GPLv3
# email: eteced@gmail.com
# llama_adapter code from https://github.com/OpenGVLab/LLaMA-Adapter

from typing import Optional, Tuple
from dataclasses import dataclass
import math
import sys
sys.path.append("..")
import torch
from torch import nn
import torch.nn.functional as F
from arthmetic_model.model import *
DEBUG_ENABLE = False

@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048
    arth_influence_layers: list = field(default_factory=lambda: [x for x in range(16, 32)])
    arth_math_layers: list = field(default_factory=lambda: [x for x in range(16, 24)])
    # arth_influence_layers: list = field(default_factory=lambda: [])
    arth_insert_layer_after: int = 23
    arth_extra_think_len: int = 32
    arth_extra_token_begin: list = field(default_factory=lambda: [1, 1738, 1433, 386, 3195, 7867, 29901, 259])
    adapter_len: int = 64

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.gate = torch.nn.Parameter(torch.zeros(1, self.n_local_heads, 1, 1))

    def forward(
        self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None
    ):

        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if adapter is not None: # adapter is global, same for all instances in a batch and all position
            adapter_len = adapter.shape[1]
            adapter_k = self.wk(adapter).view(1, adapter_len, self.n_local_heads, self.head_dim).repeat(bsz, 1, 1, 1)
            adapter_v = self.wv(adapter).view(1, adapter_len, self.n_local_heads, self.head_dim).repeat(bsz, 1, 1, 1)
            xk = torch.cat([adapter_k, xk], dim=1)
            xv = torch.cat([adapter_v, xv], dim=1)
            extra_mask = torch.zeros(1, 1, seqlen, adapter_len).to(mask) # mask is all zero for adapter_len
            mask = torch.cat([extra_mask, mask], dim=-1) # 1 x 1 x seqlen x (adpater_len+seqlen)
        keys = xk # bs x (adpater_len+seqlen) x n_local_heads x head_dim
        values = xv # bs x (adpater_len+seqlen) x n_local_heads x head_dim

        xq = xq.transpose(1, 2) # bs x n_local_heads x seqlen x head_dim
        keys = keys.transpose(1, 2) # bs x n_local_heads x (adpater_len+seqlen) x head_dim
        values = values.transpose(1, 2) # bs x n_local_heads x (adpater_len+seqlen) x head_dim
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim) # bs x n_local_heads x seqlen x (adpater_len + seqlen)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, adpater_len + slen)
        if adapter is not None:
            scores = torch.cat(
                [
                    self.gate.tanh().half() * F.softmax(scores[:, :, :, :adapter_len].float(), dim=-1).type_as(xq),
                    F.softmax(scores[:, :, :, adapter_len:].float(), dim=-1).type_as(xq),
                ],
                dim=-1,
            ) # adapter_len has extra gate (also global) to control whether attention or not
        else:
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # [bs x n_local_heads x seqlen x (adpater_len + seqlen)] x [bs x n_local_heads x (adpater_len + seqlen) x head_dim]
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class ArthTransferBlock(nn.Module):
    def __init__(self, layer_id: int, args: ArthModelArgs, tokenizer):
        super().__init__()
        self.args = args
        self.arth_model_frozen = Arth_Model(args, tokenizer)
    
    def forward(self, token_dim:torch.Tensor, start_pos : int):
        if self.args.output_steps == True:
            arth_tokens, steps_ignore_logits, steps_tmp_moved_logits, steps_dense_op_logits, steps_dense_map_logits, steps_decimal_start_logits, steps_op_pred = self.arth_model_frozen(token_dim, start_pos, tokens_is_index=False)
            return arth_tokens, steps_ignore_logits, steps_tmp_moved_logits, steps_dense_op_logits, steps_dense_map_logits, steps_decimal_start_logits, steps_op_pred 
        else:
            arth_token_index = self.arth_model_frozen(token_dim, start_pos, tokens_is_index=False)
            return arth_token_index

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, adapter)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs, arth_params: ArthModelArgs, tokenizer):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.arth_params = arth_params

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.norm_gate = RMSNorm(2, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        self.output_arth = nn.Linear(self.arth_params.dim, params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

        self.freqs_cis_math = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, (self.params.max_seq_len + self.arth_params.max_seq_len) * 2
        )

        self.tokenizer = tokenizer

        self.emb_transfer_nn = nn.Sequential(
            nn.Linear(self.params.dim, self.params.dim, bias=False),
            nn.LeakyReLU(),
            nn.Linear(self.params.dim, self.params.dim // 2, bias=False),
            nn.LeakyReLU(),
            nn.Linear(self.params.dim // 2, self.arth_params.dim, bias=False)
        )
        self.arth_gate_nn = nn.Sequential(
            nn.Linear(self.params.dim, self.params.dim, bias=False),
            nn.LeakyReLU(),
            nn.Linear(self.params.dim, self.params.dim // 2, bias=False),
            nn.LeakyReLU(),
            nn.Linear(self.params.dim // 2, 2, bias=False)
        )
        self.arth_block = ArthTransferBlock(params.n_layers + 1,arth_params, tokenizer)

        self.adapter_len = self.params.adapter_len
        self.adapter_layer = len(self.params.arth_influence_layers)
        self.adapter_layer_math = len(self.params.arth_math_layers)
        # print("self.adapter_layer", self.adapter_layer)

        self.adapter_query = nn.Embedding(self.adapter_len * self.adapter_layer, params.dim)
        self.adapter_query_math = nn.Embedding(self.adapter_len * self.adapter_layer_math, params.dim)

    def argmax_onehot(self, logits):
        arg_a=torch.argmax(logits, dim=-1)
        return F.one_hot(arg_a, num_classes=logits.shape[-1])

    def forward(self, tokens: torch.Tensor, example_mask: torch.Tensor, start_pos = 0):
        _bsz, seqlen = tokens.shape
        if DEBUG_ENABLE:
            print('tokens', tokens)
            print('example_mask', example_mask)
        h = self.tok_embeddings(tokens).detach().requires_grad_(False)
        arth_fill = self.tok_embeddings(torch.Tensor([self.arth_params.padding_token]).long().to(h.device))
        # h = self.tok_embeddings(tokens)
        arth_prefix_len = len(self.params.arth_extra_token_begin)
        arth_token_prefix = torch.tile(torch.Tensor(self.params.arth_extra_token_begin).long().view(1, -1), (_bsz, 1)).to(h.device)
        h_ord = h
        adapter_index = -1
        math_adapter_index = -1
        adapter = self.adapter_query.weight.reshape(-1, self.adapter_len, self.params.dim).unsqueeze(1)
        math_adapter = self.adapter_query_math.weight.reshape(-1, self.adapter_len, self.params.dim).unsqueeze(1)
        example_mask_tile = torch.tile(example_mask.view(-1, seqlen, 1), (1, 1, self.params.dim)).to(h)
        ### use prompt inject ###
        # use only the prompt to predict math interaction
        h_for_math = h * example_mask_tile
        non_ord_prompt_begin = torch.sum(example_mask, dim=-1).long()
        extra_padding = torch.tile(arth_fill.view(1, 1, -1), (_bsz, self.params.arth_extra_think_len, 1))
        for i in range(_bsz):
            h_for_math[:, non_ord_prompt_begin[i] : non_ord_prompt_begin[i] + self.params.arth_extra_think_len,: ] = extra_padding[i, : ,:]
        # forward math part
        for layer_index in range(0, self.params.arth_insert_layer_after + 1):
            freqs_cis = self.freqs_cis.to(h.device)
            freqs_cis = freqs_cis[:seqlen]
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
            mask = torch.triu(mask, diagonal=0 + 1).type_as(h_for_math)
            if layer_index not in self.params.arth_math_layers:
                with torch.no_grad():
                    h_for_math = self.layers[layer_index](h_for_math, start_pos, freqs_cis, mask)
            else:
                math_adapter_index = math_adapter_index + 1
                h_for_math = self.layers[layer_index](h_for_math, start_pos, freqs_cis, mask, math_adapter[math_adapter_index].half())
        lst_h_gate_logits = []
        lst_h_arth = []
        for i in range(_bsz):
            h_gate_logits = h_for_math[i, non_ord_prompt_begin[i] + self.params.arth_extra_think_len,:]
            h_arth = h_for_math[i, non_ord_prompt_begin[i] + self.params.arth_extra_think_len + 1 : non_ord_prompt_begin[i] + self.params.arth_extra_think_len + 1 + self.arth_params.max_seq_len, :]
            lst_h_gate_logits.append(h_gate_logits)
            lst_h_arth.append(h_arth)
        m_h_gate_logits = torch.stack(lst_h_gate_logits)
        m_h_arth = torch.stack(lst_h_arth)
        # m_h_arth = self.norm(m_h_arth)
        # now call arth math
        h_for_arth = self.emb_transfer_nn(m_h_arth.float())
        h_gate_logits = self.arth_gate_nn(m_h_gate_logits.float())
        h_gate_logits = self.norm_gate(h_gate_logits)  # also learns whether enable arth model
        h_arth_output = self.output_arth(h_for_arth.half()) # for supervised reverse polish notation
        if DEBUG_ENABLE:
            print(">>> h_gate_logits", h_gate_logits)
            print(">>> h_arth_output", torch.argmax(h_arth_output, dim=-1))
        if self.arth_params.output_steps == False:
            arth_result_tokens = self.arth_block(h_for_arth.half(), start_pos=0)
        else:
            arth_result_tokens, steps_ignore_logits, steps_tmp_moved_logits, steps_dense_op_logits, steps_dense_map_logits, steps_decimal_start_logits, steps_op_pred = self.arth_block(h_for_arth.half(), start_pos=0)
        arth_result_tokens = torch.concat([arth_token_prefix, arth_result_tokens.to(h.device)], axis=-1)
        if DEBUG_ENABLE:
            print(">>> arth_result_tokens", arth_result_tokens)
        # cut the tokens
        arth_result_tokens = arth_result_tokens[:, : self.arth_params.max_seq_len]
        extra_padding_arth_result = torch.tile(arth_fill.view(1, 1, -1), (_bsz, self.arth_params.max_seq_len, 1))
        q_new = self.tok_embeddings(arth_result_tokens.to(self.arth_params.device))
        q_update = self.argmax_onehot(h_gate_logits)[:, 1] # B
        q_update_tile = torch.tile(q_update.view(-1, 1, 1), (1, self.arth_params.max_seq_len, self.params.dim)).to(h)
        q_new_final = q_new * q_update_tile + (torch.ones_like(q_update_tile) - q_update_tile) * extra_padding_arth_result
        # inject prompt
        h_new = torch.zeros(_bsz, seqlen + self.arth_params.max_seq_len, self.params.dim)
        q_update_t = torch.tile(q_update.view(-1, 1), (1, self.arth_params.max_seq_len))
        mq_token = arth_result_tokens.to(h.device) * q_update_t.to(h.device)
        if DEBUG_ENABLE:
            print(">>> mq_token", mq_token)
        max_q_index = torch.sum((mq_token != 0), axis=-1)
        for i in range(_bsz):
            h_new[i, :non_ord_prompt_begin[i], :] = h_ord[i, :non_ord_prompt_begin[i], :] # copy the ord token
            mq_index = max_q_index[i].item()
            if (mq_index > 0):
                h_new[i, non_ord_prompt_begin[i]:non_ord_prompt_begin[i] + mq_index, :] = q_new_final[i, :mq_index, :] # copy the arth tokens
            h_new[i, non_ord_prompt_begin[i] + mq_index : mq_index + seqlen, :] = h_ord[i, non_ord_prompt_begin[i]:, :] # copy the answers
        h_new = h_new.to(h)
        for layer_index in range(0, len(self.layers)):
            freqs_cis_math = self.freqs_cis_math.to(h_new.device)
            freqs_cis_math = freqs_cis_math[:seqlen + self.arth_params.max_seq_len]
            # mask = None
            mask_math = torch.full((1, 1, seqlen + self.arth_params.max_seq_len, seqlen + self.arth_params.max_seq_len), float("-inf"), device=h_new.device)
            mask_math = torch.triu(mask_math, diagonal=0 + 1).type_as(h)
            if layer_index not in self.params.arth_influence_layers:
                with torch.no_grad():
                    h_new = self.layers[layer_index](h_new, start_pos, freqs_cis_math, mask_math)
            else:
                adapter_index = adapter_index + 1
                h_new = self.layers[layer_index](h_new, start_pos, freqs_cis_math, mask_math, adapter[adapter_index].half())
        del mask, freqs_cis, mask_math, freqs_cis_math
        h = h[:, :seqlen, :]
        # should know that we have extra self.arth_params.max_seq_len dims before the label. we would like exclude it
        for i in range(_bsz):
            mq_index = max_q_index[i].item()
            h[i, :non_ord_prompt_begin[i], :] = h_new[i, :non_ord_prompt_begin[i], :] # copy the ord token
            h[i, non_ord_prompt_begin[i]:, :] = h_new[i, non_ord_prompt_begin[i] + mq_index: mq_index + seqlen, :] # copy the exclude arth token
        h = self.norm(h)
        output = self.output(h[:, :-1, :])
        if self.arth_params.output_steps == True:
            return output.float(), h_gate_logits.float(), h_arth_output.float(), steps_ignore_logits, steps_tmp_moved_logits, steps_dense_op_logits, steps_dense_map_logits, steps_decimal_start_logits, steps_op_pred, arth_result_tokens
        else:
            return output.float(), h_gate_logits.float(), h_arth_output.float()
    
    @torch.inference_mode()
    def forward_inference(self, tokens: torch.Tensor, example_mask: torch.Tensor, start_pos: int, cur_pos: int, full_mode=False):
        _bsz, seqlen = tokens.shape
        if DEBUG_ENABLE:
            print('tokens', tokens)
        h = self.tok_embeddings(tokens).detach().requires_grad_(False)
        arth_fill = self.tok_embeddings(torch.Tensor([self.arth_params.padding_token]).long().to(h.device))
        # h = self.tok_embeddings(tokens)
        arth_prefix_len = len(self.params.arth_extra_token_begin)
        arth_token_prefix = torch.tile(torch.Tensor(self.params.arth_extra_token_begin).long().view(1, -1), (_bsz, 1)).to(h.device)
        h_ord = h
        adapter_index = -1
        math_adapter_index = -1
        adapter = self.adapter_query.weight.reshape(-1, self.adapter_len, self.params.dim).unsqueeze(1)
        math_adapter = self.adapter_query_math.weight.reshape(-1, self.adapter_len, self.params.dim).unsqueeze(1)
        example_mask_tile = torch.tile(example_mask.view(-1, seqlen, 1), (1, 1, self.params.dim)).to(h)
        ### use prompt inject ###
        # use only the prompt to predict math interaction
        h_for_math = h * example_mask_tile
        non_ord_prompt_begin = torch.sum(example_mask, dim=-1).long()
        extra_padding = torch.tile(arth_fill.view(1, 1, -1), (_bsz, self.params.arth_extra_think_len, 1))
        for i in range(_bsz):
            h_for_math[:, non_ord_prompt_begin[i] : non_ord_prompt_begin[i] + self.params.arth_extra_think_len,: ] = extra_padding[i, : ,:]
        # forward math part
        for layer_index in range(0, self.params.arth_insert_layer_after + 1):
            freqs_cis = self.freqs_cis.to(h.device)
            freqs_cis = freqs_cis[:seqlen]
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
            mask = torch.triu(mask, diagonal=0 + 1).type_as(h_for_math)
            if layer_index not in self.params.arth_math_layers:
                with torch.no_grad():
                    h_for_math = self.layers[layer_index](h_for_math, start_pos, freqs_cis, mask)
            else:
                math_adapter_index = math_adapter_index + 1
                h_for_math = self.layers[layer_index](h_for_math, start_pos, freqs_cis, mask, math_adapter[math_adapter_index].half())
        lst_h_gate_logits = []
        lst_h_arth = []
        for i in range(_bsz):
            h_gate_logits = h_for_math[i, non_ord_prompt_begin[i] + self.params.arth_extra_think_len,:]
            h_arth = h_for_math[i, non_ord_prompt_begin[i] + self.params.arth_extra_think_len + 1 : non_ord_prompt_begin[i] + self.params.arth_extra_think_len + 1 + self.arth_params.max_seq_len, :]
            lst_h_gate_logits.append(h_gate_logits)
            lst_h_arth.append(h_arth)
        m_h_gate_logits = torch.stack(lst_h_gate_logits)
        m_h_arth = torch.stack(lst_h_arth)
        # now call arth math
        h_for_arth = self.emb_transfer_nn(m_h_arth.float())
        h_gate_logits = self.arth_gate_nn(m_h_gate_logits.float())
        h_gate_logits = self.norm_gate(h_gate_logits)  # also learns whether enable arth model
        h_arth_output = self.output_arth(h_for_arth.half()) # for supervised reverse polish notation
        if DEBUG_ENABLE:
            print(">>> h_gate_logits", h_gate_logits)
            print(">>> h_arth_output", torch.argmax(h_arth_output, dim=-1))
        if self.arth_params.output_steps == False:
            arth_result_tokens = self.arth_block(h_for_arth.half(), start_pos=0)
        else:
            arth_result_tokens, steps_ignore_logits, steps_tmp_moved_logits, steps_dense_op_logits, steps_dense_map_logits, steps_decimal_start_logits, steps_op_pred = self.arth_block(h_for_arth.half(), start_pos=0)
        arth_result_tokens = torch.concat([arth_token_prefix, arth_result_tokens.to(h.device)], axis=-1)
        if DEBUG_ENABLE:
            print(">>> arth_result_tokens", arth_result_tokens)
        # cut the tokens
        arth_result_tokens = arth_result_tokens[:, : self.arth_params.max_seq_len]
        extra_padding_arth_result = torch.tile(arth_fill.view(1, 1, -1), (_bsz, self.arth_params.max_seq_len, 1))
        q_new = self.tok_embeddings(arth_result_tokens.to(self.arth_params.device))
        q_update = self.argmax_onehot(h_gate_logits)[:, 1] # B
        q_update_tile = torch.tile(q_update.view(-1, 1, 1), (1, self.arth_params.max_seq_len, self.params.dim)).to(h)
        q_new_final = q_new * q_update_tile + (torch.ones_like(q_update_tile) - q_update_tile) * extra_padding_arth_result
        # inject prompt
        h_new = torch.zeros(_bsz, seqlen + self.arth_params.max_seq_len, self.params.dim)
        q_update_t = torch.tile(q_update.view(-1, 1), (1, self.arth_params.max_seq_len))
        mq_token = arth_result_tokens.to(h.device) * q_update_t.to(h.device)
        max_q_index = torch.sum((mq_token != 0), axis=-1)
        for i in range(_bsz):
            h_new[i, :non_ord_prompt_begin[i], :] = h_ord[i, :non_ord_prompt_begin[i], :] # copy the ord token
            mq_index = max_q_index[i].item()
            if (mq_index > 0):
                h_new[i, non_ord_prompt_begin[i]:non_ord_prompt_begin[i] + mq_index, :] = q_new_final[i, :mq_index, :] # copy the arth tokens
            h_new[i, non_ord_prompt_begin[i] + mq_index : mq_index + seqlen, :] = h_ord[i, non_ord_prompt_begin[i]:, :] # copy the answers
        h_new = h_new.to(h)
        for layer_index in range(0, len(self.layers)):
            freqs_cis_math = self.freqs_cis_math.to(h_new.device)
            freqs_cis_math = freqs_cis_math[:seqlen + self.arth_params.max_seq_len]
            # mask = None
            mask_math = torch.full((1, 1, seqlen + self.arth_params.max_seq_len, seqlen + self.arth_params.max_seq_len), float("-inf"), device=h_new.device)
            mask_math = torch.triu(mask_math, diagonal=0 + 1).type_as(h)
            if layer_index not in self.params.arth_influence_layers:
                with torch.no_grad():
                    h_new = self.layers[layer_index](h_new, start_pos, freqs_cis_math, mask_math)
            else:
                adapter_index = adapter_index + 1
                h_new = self.layers[layer_index](h_new, start_pos, freqs_cis_math, mask_math, adapter[adapter_index].half())
        del mask, freqs_cis, mask_math, freqs_cis_math
        h = h[:, :seqlen, :]
        # should know that we have extra self.arth_params.max_seq_len dims before the label. we would like exclude it
        for i in range(_bsz):
            mq_index = max_q_index[i].item()
            h[i, :non_ord_prompt_begin[i], :] = h_new[i, :non_ord_prompt_begin[i], :] # copy the ord token
            h[i, non_ord_prompt_begin[i]:, :] = h_new[i, non_ord_prompt_begin[i] + mq_index: mq_index + seqlen, :] # copy the exclude arth token
        h = self.norm(h)
        if full_mode == True:
            output = self.output(h)
        else:
            output = self.output(h[:, cur_pos, :])
        return output.float()

    def enable_cache(self):
        for layer in self.layers:
            layer.attention.enable_cache()

    def disable_cache(self):
        for layer in self.layers:
            layer.attention.disable_cache()
