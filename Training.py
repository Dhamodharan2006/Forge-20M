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

load_dir = "new_llm_v1"
out_dir = "new_llm_v2"

eval_interval = 250
log_interval = 1
eval_iters = 20
eval_only = False
always_save_checkpoint = True

init_from = 'resume'

gradient_accumulation_steps = 4
batch_size = 32
block_size = 512

n_layer = 10
n_head = 8
n_embd = 256
dropout = 0.0
bias = False

learning_rate = 5e-4             
max_iters = 330 + 4600           
lr_decay_iters = 330 + 4600      

weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
warmup_iters = 100
min_lr = 5e-5

device = 'cuda'
dtype = 'float16'
compile = False

data_dir = ""

tokens_per_iter=gradient_accumulation_steps * batch_size * block_size
print(f"tokens per iteration: {tokens_per_iter:,}")
print(f"total tokens: {tokens_per_iter * max_iters:,}")

os.makedirs(out_dir,exist_ok=True)
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type='cuda' if 'cuda' in device else 'cpu'
ptdtype={'float32':torch.float32,'bfloat16':torch.bfloat16,'float16':torch.float16}[dtype]
ctx=torch.amp.autocast(device_type=device_type,dtype=ptdtype)

def get_batch(split):
    if split == 'train':
        data=np.memmap(os.path.join(data_dir,'train.bin'),dtype=np.uint16,mode='r')
    else :
        data=np.memmap(os.path.join(data_dir,'val.bin'),dtype=np.uint16,mode='r')
    ix=torch.randint(len(data)-block_size, (batch_size,))
    x=torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y=torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type=='cuda':
        x,y=x.pin_memory().to(device,non_blocking=True),y.pin_memory().to(device,non_blocking=True)
    else:
        x,y=x.to(device),y.to(device)
    return x,y

iter_num=0
best_val_loss=1e9

model_args=dict(
    n_layer=n_layer,n_head=n_head,n_embd=n_embd, block_size=block_size,bias=bias,vocab_size=50304,dropout=dropout
)

if init_from == 'scratch':
    print("Initializing a new Model from Scratch")
    gptconf=GPTConfig(**model_args)
    model=GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {load_dir}")
    ckpt_path="/kaggle/input/datasets/dhamodharan18/tiny-web-20m-final-checkpoint/ckpt_final_iter4931_valloss4.211.pt"
    checkpoint=torch.load(ckpt_path,map_location=device)
    checkpoint_model_args=checkpoint['model_args']
    for k in ['n_layer','n_head','n_embd','block_size','bias','vocab_size']:
        model_args[k]=checkpoint_model_args[k]
    gptconf=GPTConfig(**model_args)
    model=GPT(gptconf)
    state_dict=checkpoint['model']
    unwanted_prefix='_orig_mod.'
    for k,v in list(state_dict.items()):
        if k .startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]]=state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num=checkpoint['iter_num']
    best_val_loss=checkpoint['best_val_loss']
    
model.to(device)

scaler=torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

optimizer=model.configure_optimizers(weight_decay,learning_rate,(beta1,beta2),device_type)
if init_from =='resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint=None

print("Loaded iter_num:", iter_num)
print("Loaded best_val_loss:", best_val_loss)

print(model)

@torch.no_grad()
def estimate_loss():
    out={}
    model.eval()
    for split in ['train','val']:
        losses=torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            with ctx:
                logits, loss=model(X,Y)
            losses[k]=loss.item()
        out[split]=losses.mean()
    model.train()
    return out

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it+1)/(warmup_iters +1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio=(it-warmup_iters)/(lr_decay_iters-warmup_iters)
    assert 0 <= decay_ratio<=1
    coeff=0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr+coeff*(learning_rate - min_lr)

X,Y = get_batch('train')
t0=time.time()
local_iter_num=0
running_mfu=-1.0

print("---------------------------------Started Training--------------------------------------")

while(True):
    lr=get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr']=lr

    if iter_num % eval_interval==0:
        losses=estimate_loss()
        print(f"Step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses['val']<best_val_loss or always_save_checkpoint:
            best_val_loss=losses['val']
            if iter_num > 0:
                checkpoint={
                    'model':model.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    'model_args':model_args,
                    'iter_num':iter_num,
                    'best_val_loss':best_val_loss,
                }
                print(f"Saving checkpoint to {out_dir}")
                torch.save(checkpoint,os.path.join(out_dir,f'ckpt.pt'))
    if iter_num==0 and eval_only:
        break
    if iter_num % 250 ==0 and iter_num > 0 :
        model.eval()
        start_ids=[50256]
        x_sample=torch.tensor(start_ids,dtype=torch.long,device=device)[None,...]
        with torch.no_grad():
            with ctx:
                y_sample=model.generate(x_sample,100,temperature=0.8,top_k=200)
                out_ids=y_sample[0].tolist()
                #import tiktoken
                #enc=tiktoken.get_encoding("gpt2")
                print(enc.decode(out_ids))
        print("--------------------------------------------\n")
        model.train()

    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits,loss=model(X,Y)
            loss=loss/gradient_accumulation_steps
        X,Y=get_batch('train')
        scaler.scale(loss).backward()

    if grad_clip!=0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),grad_clip)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1=time.time()
    dt=t1-t0
    t0=t1
    if iter_num % log_interval ==0:
        lossf=loss.item()*gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:.2e}")
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

print("-----------------------------------------------------Training Completed ------------------------------------------------")

checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'model_args': model_args,
    'iter_num': iter_num,
    'best_val_loss': best_val_loss,
}
torch.save(checkpoint, os.path.join(out_dir, 'ckpt_latest.pt'))
final_name = f'ckpt_final_iter{iter_num}_valloss{best_val_loss:.3f}.pt'
torch.save(checkpoint, os.path.join(out_dir, final_name))
print(f"Final checkpoint saved to {out_dir}/{final_name}")





