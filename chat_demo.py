import argparse
import os
import sys

import fairscale.nn.model_parallel.initialize as fs_init
import model_arthllama
import torch
import torch.distributed as dist
from conversation import conv_templates, SeparatorStyle
from util import misc

from llama import LLaMA, Tokenizer


def load_model(args, load_8bit=False):
    model = model_arthllama.__dict__[args.model_name](args)
    model.eval()
    if args.model_path is None:
        print("Warning: not loading instruct tuned weights.")
    else:
        print("Using instruct tuned weights from:", args.model_path)
        checkpoint = torch.load(args.model_path, map_location="cpu")
        for k, v in checkpoint["model"].items():
            if (
                k.endswith(".wq_bias")
                or k.endswith(".wk_bias")
                or k.endswith(".wv_bias")
                or k.endswith(".wo_scale")
                or k.endswith(".w1_bias")
                or k.endswith(".w3_bias")
                or k.endswith(".w2_scale")
            ):
                assert v.ndim == 1
                mp_size = 1
                mp_rank = 0
                shard_size = v.size(0) // mp_size
                checkpoint["model"][k] = v[shard_size * mp_rank : shard_size * (mp_rank + 1)]
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)

    generator = LLaMA(
        model,
        Tokenizer(model_path=os.path.join(args.llama_model_path, "tokenizer.model")),
    )

    return generator


@torch.inference_mode()
def generate_stream(model, params):
    """Adapted from fastchat/serve/model_worker.py::generate_stream"""

    prompt = params["prompt"]
    len(prompt)
    temperature = float(params.get("temperature", -1.0))
    top_p = float(params.get("top_p", 0.95))
    max_new_tokens = int(params.get("max_new_tokens", 256))
    stop_str = params.get("stop", None)
    with torch.cuda.amp.autocast():
        decoded = model.generate(
            [prompt],
            max_gen_len=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
        )
    decoded = decoded[0]

    pos = decoded.find(stop_str)
    if pos != -1:
        decoded = decoded[:pos]

    return decoded


def main(args):
    misc.init_distributed_mode(args)
    # fs_init.initialize_model_parallel(dist.get_world_size())
    torch.manual_seed(1)

    # Model
    model = load_model(args)

    # Chat
    conv = conv_templates[args.conv_template].copy()
    while True:
        # if dist.get_rank() == 0:
        try:
            sys.stdout.write(f"\n{conv.roles[0]}: ")
            sys.stdout.flush()
            inp = input()
        except EOFError:
            inp = ""
        #     dist.broadcast_object_list([inp], src=0)
        # else:
        #     recv_obj = [None]
        #     dist.broadcast_object_list(recv_obj, src=0)
        #     inp = recv_obj[0]

        if not inp:
            print("exit...")
            break

        # conv.append_message(conv.roles[0], inp)
        # conv.append_message(conv.roles[1], None)
        # prompt = conv.get_prompt()
        prompt = inp

        params = {
            "model": "Arth_Llama7B",
            "prompt": prompt,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "stop": conv.sep if conv.sep_style == SeparatorStyle.SINGLE else conv.sep2,
        }

        # if dist.get_rank() == 0:
        sys.stdout.write(f"{conv.roles[1]}: ")
        sys.stdout.flush()
        outputs = generate_stream(model, params)
        outputs = outputs.strip()
        # if dist.get_rank() == 0:
        sys.stdout.write(outputs + "\n")
        sys.stdout.flush()

        conv.messages[-1][-1] = outputs

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")

def get_args_parser():
    parser = argparse.ArgumentParser("ArthLLaMA Inference", add_help=False)
    parser.add_argument(
        "--batch_size",
        default=1,
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
    parser.add_argument("--model_name", default="Arth_Llama7B", type=str, metavar="MODEL", help="Name of model to train")

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
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--arth_model_path", default="./checkpoint/", type=str, help="dataset path")
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
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--conv_template", type=str, default="v1")
    parser.add_argument("--temperature", type=float, default=-0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--debug", action="store_true")

    return parser

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
