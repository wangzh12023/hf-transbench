#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import models


CONFIGS = [
    "configs/my_llama_tiny_sigmoid.json",
    "configs/my_llama_tiny_sigmoid_with_b.json",
    "configs/my_llama_tiny_linear.json",
    "configs/head_softmax.json",
    "configs/head_softmax_with_b.json",
    "configs/softmax_and_head_softmax.json",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--intermediate-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    args = parser.parse_args()

    all_ok = True
    for config_path in CONFIGS:
        print(f"[smoke] {config_path}", flush=True)
        try:
            cfg = AutoConfig.from_pretrained(config_path)
            cfg.hidden_size = args.hidden_size
            cfg.intermediate_size = args.intermediate_size
            cfg.num_hidden_layers = args.num_layers
            cfg.num_attention_heads = args.num_heads
            cfg.num_key_value_heads = args.num_kv_heads

            model = AutoModelForCausalLM.from_config(cfg, attn_implementation="eager")
            input_ids = torch.randint(0, cfg.vocab_size, (1, 8))
            attention_mask = torch.ones_like(input_ids)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            print(f"  ok: logits_shape={tuple(logits.shape)}", flush=True)
        except Exception as exc:
            all_ok = False
            print(f"  FAIL: {type(exc).__name__}: {exc}", flush=True)

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
