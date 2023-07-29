# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import List

import torch

from llama.tokenizer import Tokenizer
from llama.model import Transformer


class LLaMA:
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[str]:
        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        # print(prompts)
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        # prompt_tokens = [[  1,  3529,   274,   562,  5987,   445, 29889,   396, 29871, 29941,
        #  29889, 29945,   334, 29871, 29906, 29889, 29955,   718, 29871, 29947,
        #  29889, 29953,   353,  1577,   395]]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)
        # print("total_len", total_len)

        pad_id = 0
        tokens = torch.full((bsz, total_len), 0).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != pad_id
        input_prompt_len = torch.sum(input_text_mask, dim=-1)
        max_start_pos = torch.max(input_prompt_len)
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(max_start_pos, total_len):
            print(" >> inference...", cur_pos, " / ", total_len)
            logits = self.model.forward_inference(tokens, 0, cur_pos - 1, full_mode=False)[0]
            print('logits.shape', logits.shape)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            # next_token = torch.where(
            #     input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            # )
            tokens[:, cur_pos] = next_token
            print('in', tokens)
            prev_pos = cur_pos
        print("tokens.shape", tokens.shape)
        print("tokens", tokens)
        print('max_start_pos', max_start_pos)
        # logits = self.model.forward_inference(tokens, 0, 0, full_mode=True)
        # for i in range(logits.shape[1]):
        #     logit = logits[:, i, :]
        #     next_token = torch.argmax(logit, dim=-1)
        #     tokens[:, i] = next_token
        # print("tokens", tokens)
        # print("start token", tokens[:, max_start_pos])

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        return decoded
        # return tokens


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
