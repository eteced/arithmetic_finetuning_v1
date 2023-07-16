#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This software may be used and distributed according to the terms of the GNU General Public License version 3.
"""ArthModel with LLaMA
"""
__author__ = "Yingdi Guo"
__license__ = "GPLv3"
__email__ = "eteced@gmail.com"

import json

import torch

from llama import ModelArgs, Tokenizer, Transformer
from arthmetic_model.model import *

def Arth_Llama7B(args, **kwargs):
    llama_model_path = args.llama_model_path
    model_name = "7B"

    checkpoint = torch.load(llama_model_path + model_name + "/consolidated.00.pth", map_location="cpu")
    print(llama_model_path + model_name + "/consolidated.00.pth")

    arth_model_path = args.arth_model_path

    with open(llama_model_path + model_name + "/params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=args.max_seq_len,
        max_batch_size=1,
        **params
    )
    tokenizer = Tokenizer(model_path=llama_model_path + "/tokenizer.model")

    arth_params = ArthModelArgs()
    arth_params.output_steps = True
    arth_params.max_batch_size = 1

    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    arth_llama_7b = Transformer(model_args, arth_params, tokenizer)
    torch.set_default_tensor_type(torch.FloatTensor)
    arth_llama_7b.load_state_dict(checkpoint, strict=False)
    checkpoint_arth = torch.load(arth_model_path + '/arth_model_128dim.pth', map_location="cpu")
    arth_llama_7b.arth_block.arth_model_frozen.load_state_dict(checkpoint_arth, strict=False)

    for name, param in arth_llama_7b.named_parameters():
        if "adapter" not in name and 'emb_transfer_nn' not in name and 'arth_gate_nn' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            param.data = param.data.float()
            print("requires_grad, name: ", name)
    

    for name, param in arth_llama_7b.layers.named_parameters():
        if "arth_block" not in name and ("gate" in name or "adapter" in name):
            param.data = param.data.float()
            param.requires_grad = True
            print("requires_grad, name: ", name)
    
    return arth_llama_7b

Arth_Llama7B = Arth_Llama7B