import os
import tiktoken
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

DATASET_NAME = "HuggingFaceFW/fineweb-edu"
SAMPLE_NAME = "sample-10BT"
TARGET_TOKENS = 300_000_000
SKIP_TOKENS = 100_002_100

enc = tiktoken.get_encoding("gpt2")

def process(example):
    ids = enc.encode_ordinary(example['text'])
    ids.append(enc.eot_token)
    return {'ids': ids, 'len': len(ids)}

if __name__ == "__main__":
    print(f"Loading streaming dataset {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME, name=SAMPLE_NAME, split='train', streaming=True)

    all_tokens = []
    total_tokens = 0
    skipped = 0
    pbar_skip = tqdm(total=SKIP_TOKENS, desc="Skipping already-used tokens")
    for example in dataset:
        tokens = process(example)['ids']
        skipped += len(tokens)
        pbar_skip.update(len(tokens))
        if skipped >= SKIP_TOKENS:
            break
    pbar_skip.close()

    pbar = tqdm(total=TARGET_TOKENS, desc="Collecting NEW tokens")
    for example in dataset:
        tokens = process(example)['ids']
        all_tokens.extend(tokens)
        total_tokens += len(tokens)
        pbar.update(len(tokens))
        if total_tokens >= TARGET_TOKENS:
            break
    pbar.close()

    n = len(all_tokens)
    train_data = all_tokens[:int(n*0.95)]
    val_data = all_tokens[int(n*0.95):]

    for name, d in [('train', train_data), ('val', val_data)]:
        arr = np.array(d, dtype=np.uint16)
        filename = f"{name}.bin"
        arr.tofile(filename)
        print(f"Saved {filename} with {len(d):,} tokens.")

    print("\nDone!")