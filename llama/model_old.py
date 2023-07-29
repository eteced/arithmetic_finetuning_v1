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
    # arth_influence_layers: list = field(default_factory=lambda: [])
    arth_insert_layer_after: int = 23
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

        if adapter is not None:
            adapter_len = adapter.shape[1]
            adapter_k = self.wk(adapter).view(1, adapter_len, self.n_local_heads, self.head_dim).repeat(bsz, 1, 1, 1)
            adapter_v = self.wv(adapter).view(1, adapter_len, self.n_local_heads, self.head_dim).repeat(bsz, 1, 1, 1)
            xk = torch.cat([adapter_k, xk], dim=1)
            xv = torch.cat([adapter_v, xv], dim=1)
            extra_mask = torch.zeros(1, 1, seqlen, adapter_len).to(mask)
            mask = torch.cat([extra_mask, mask], dim=-1)
        keys = xk
        values = xv

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        if adapter is not None:
            scores = torch.cat(
                [
                    self.gate.tanh().half() * F.softmax(scores[:, :, :, :adapter_len].float(), dim=-1).type_as(xq),
                    F.softmax(scores[:, :, :, adapter_len:].float(), dim=-1).type_as(xq),
                ],
                dim=-1,
            )
        else:
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
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
        # print("self.adapter_layer", self.adapter_layer)

        self.adapter_query = nn.Embedding(self.adapter_len * self.adapter_layer, params.dim)

    def argmax_onehot(self, logits):
        arg_a=torch.argmax(logits, dim=-1)
        return F.one_hot(arg_a, num_classes=logits.shape[-1])

    def forward(self, tokens: torch.Tensor, example_mask: torch.Tensor, start_pos = 0):
        _bsz, seqlen = tokens.shape
        # print('tokens', tokens)
        # print('example_mask', example_mask)
        h = self.tok_embeddings(tokens)
        adapter_index = -1
        adapter = self.adapter_query.weight.reshape(-1, self.adapter_len, self.params.dim).unsqueeze(1)
        example_mask_tile = torch.tile(example_mask.view(-1, seqlen, 1), (1, 1, self.params.dim)).to(h)
        for layer_index in range(0, len(self.layers)):
            freqs_cis = self.freqs_cis.to(h.device)
            freqs_cis = freqs_cis[:seqlen]
            freqs_cis_math = self.freqs_cis_math.to(h.device)
            freqs_cis_math = freqs_cis_math[:seqlen + self.arth_params.max_seq_len]
            # mask = None
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
            mask = torch.triu(mask, diagonal=0 + 1).type_as(h)
            mask_math = torch.full((1, 1, seqlen + self.arth_params.max_seq_len, seqlen + self.arth_params.max_seq_len), float("-inf"), device=h.device)
            mask_math = torch.triu(mask_math, diagonal=0 + 1).type_as(h)
            if layer_index not in self.params.arth_influence_layers:
                with torch.no_grad():
                    h = self.layers[layer_index](h, start_pos, freqs_cis, mask)
            else:
                adapter_index = adapter_index + 1
                # print("adapter_index", adapter_index)
                # print("layer_index", layer_index)
                if layer_index <= self.params.arth_insert_layer_after:
                    h = self.layers[layer_index](h, start_pos, freqs_cis, mask, adapter[adapter_index].half())
                else:
                    h = self.layers[layer_index](h, start_pos, freqs_cis_math, mask_math, adapter[adapter_index].half())
            if layer_index == self.params.arth_insert_layer_after:
                # only top self.arth_params.max_seq_len (default 128) will be send to ArthTransferBlock
                h_for_arth_tmp = h * example_mask_tile
                h_for_arth = self.emb_transfer_nn(h_for_arth_tmp[:, :self.arth_params.max_seq_len, :].float())
                h_gate_logits = self.arth_gate_nn(torch.sum(h[:, :self.arth_params.max_seq_len, :].float(), dim=1))
                h_gate_logits = self.norm_gate(h_gate_logits)  # also learns whether enable arth model
                h_arth_output = self.output_arth(h_for_arth[:, :, :].half()) # for supervised reverse polish notation
                # print("h_arth_output.shape", h_arth_output.shape)
                # print("h_for_arth.shape", h_for_arth.shape)
                if self.arth_params.output_steps == False:
                    arth_result_tokens = self.arth_block(h_for_arth.half(), start_pos=0)
                else:
                    arth_result_tokens, steps_ignore_logits, steps_tmp_moved_logits, steps_dense_op_logits, steps_dense_map_logits, steps_decimal_start_logits, steps_op_pred = self.arth_block(h_for_arth.half(), start_pos=0)
                # print('arth_result_tokens', arth_result_tokens)
                q_new = self.tok_embeddings(arth_result_tokens.to(self.arth_params.device))
                q_update = self.argmax_onehot(h_gate_logits)[:, 1] # B
                q_update_tile = torch.tile(q_update.view(-1, 1, 1), (1, self.arth_params.max_seq_len, self.params.dim)).to(h)
                # remix h
                # h[:, :self.arth_params.max_seq_len, :] = (torch.ones_like(q_update_tile) - q_update_tile) * h[:, :self.arth_params.max_seq_len, :] + q_update_tile * q_new
                # print('h.shape', h.shape)
                # h[:, :self.arth_params.max_seq_len, :] = h[:, :self.arth_params.max_seq_len, :] + q_update_tile * q_new
                h = torch.concat([q_update_tile * q_new, h], dim = 1)
                del q_new, q_update, q_update_tile
            del mask, freqs_cis
        h = h[:, :seqlen, :]
        h = self.norm(h)
        output = self.output(h[:, :-1, :])
        if self.arth_params.output_steps == True:
            return output.float(), h_gate_logits.float(), h_arth_output.float(), steps_ignore_logits, steps_tmp_moved_logits, steps_dense_op_logits, steps_dense_map_logits, steps_decimal_start_logits, steps_op_pred
        else:
            return output.float(), h_gate_logits.float(), h_arth_output.float()
    
    @torch.inference_mode()
    def forward_inference(self, tokens: torch.Tensor, start_pos: int, full_mode=True):
        _bsz, seqlen = tokens.shape
        # print('tokens', tokens)
        h = self.tok_embeddings(tokens)
        adapter_index = -1
        adapter = self.adapter_query.weight.reshape(-1, self.adapter_len, self.params.dim).unsqueeze(1)
        for layer_index in range(0, len(self.layers)):
            freqs_cis = self.freqs_cis.to(h.device)
            freqs_cis = freqs_cis[:seqlen]
            freqs_cis_math = self.freqs_cis_math.to(h.device)
            freqs_cis_math = freqs_cis_math[:seqlen + self.arth_params.max_seq_len]
            # mask = None
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
            mask = torch.triu(mask, diagonal=0 + 1).type_as(h)
            mask_math = torch.full((1, 1, seqlen + self.arth_params.max_seq_len, seqlen + self.arth_params.max_seq_len), float("-inf"), device=h.device)
            mask_math = torch.triu(mask_math, diagonal=0 + 1).type_as(h)
            if layer_index not in self.params.arth_influence_layers:
                with torch.no_grad():
                    h = self.layers[layer_index](h, start_pos, freqs_cis, mask)
            else:
                adapter_index = adapter_index + 1
                # print("adapter_index", adapter_index)
                # print("layer_index", layer_index)
                if layer_index <= self.params.arth_insert_layer_after:
                    h = self.layers[layer_index](h, start_pos, freqs_cis, mask, adapter[adapter_index].half())
                else:
                    h = self.layers[layer_index](h, start_pos, freqs_cis_math, mask_math, adapter[adapter_index].half())
            if layer_index == self.params.arth_insert_layer_after:
                # only top self.arth_params.max_seq_len (default 128) will be send to ArthTransferBlock
                h_for_arth = self.emb_transfer_nn(h[:, :self.arth_params.max_seq_len, :].float())
                h_gate_logits = self.arth_gate_nn(torch.sum(h[:, :self.arth_params.max_seq_len, :].float(), dim=1))
                h_gate_logits = self.norm_gate(h_gate_logits)  # also learns whether enable arth model
                h_arth_output = self.output_arth(h_for_arth[:, :, :].half()) # for supervised reverse polish notation
                # print("h_arth_output.shape", h_arth_output.shape)
                # print("h_for_arth.shape", h_for_arth.shape)
                if self.arth_params.output_steps == False:
                    arth_result_tokens = self.arth_block(h_for_arth.half(), start_pos=0)
                else:
                    arth_result_tokens, steps_ignore_logits, steps_tmp_moved_logits, steps_dense_op_logits, steps_dense_map_logits, steps_decimal_start_logits, steps_op_pred = self.arth_block(h_for_arth.half(), start_pos=0)
                # print('arth_result_tokens', arth_result_tokens)
                q_new = self.tok_embeddings(arth_result_tokens.to(self.arth_params.device))
                q_update = self.argmax_onehot(h_gate_logits)[:, 1] # B
                q_update_tile = torch.tile(q_update.view(-1, 1, 1), (1, self.arth_params.max_seq_len, self.params.dim)).to(h)
                q_final = q_update_tile * q_new
                print("arth_result_tokens", arth_result_tokens)
                print("h_gate_logits", h_gate_logits)
                # remix h
                # h[:, :self.arth_params.max_seq_len, :] = (torch.ones_like(q_update_tile) - q_update_tile) * h[:, :self.arth_params.max_seq_len, :] + q_update_tile * q_new
                # print('h.shape', h.shape)
                # h[:, :self.arth_params.max_seq_len, :] = h[:, :self.arth_params.max_seq_len, :] + q_update_tile * q_new
                h = torch.concat([q_final, h], dim = 1)
                del q_new, q_update, q_update_tile
            del mask, freqs_cis
        h = h[:, :seqlen, :]
        h = self.norm(h)
        if full_mode == True:
            output = self.output(h)
        else:
            output = self.output(h[:, -1, :])
        return output.float()

    def enable_cache(self):
        for layer in self.layers:
            layer.attention.enable_cache()

    def disable_cache(self):
        for layer in self.layers:
            layer.attention.disable_cache()
