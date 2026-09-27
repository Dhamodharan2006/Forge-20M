import os
import time
import math
import numpy as np
import torch 
from Model import GPTConfig, GPT
import inspect
from dataclasses import dataclass
import torch.nn as nn
from torch.nn import functional as F

def ask_gpt_with_penalty(prompt, max_new_tokens=150, temperature=0.7, top_k=25, repetition_penalty=1.3):
    start_ids = enc.encode(prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    idx = x
    generated_ids = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
            logits, _ = f_model(idx_cond)
            logits = logits[:, -1, :]

            # apply repetition penalty against everything generated so far (prompt + new tokens)
            for token_id in set(idx[0].tolist()):
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty

            logits = logits / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            if idx_next.item() == 50256:
                break

            idx = torch.cat((idx, idx_next), dim=1)
            generated_ids.append(idx_next.item())

    return enc.decode(generated_ids)


print("--- Forge-20M Completion Chat started ---")
print("Type 'exit' or 'quit' to stop.\n")

while True:
    user_input = input("\nYour Prompt: ")
    if user_input.lower() in ['exit', 'quit']:
        break
    
    response = ask_gpt_with_penalty(user_input, repetition_penalty=1.3)
    
    print(f"\nForge-20M: {user_input}{response}")
    print("-" * 60)