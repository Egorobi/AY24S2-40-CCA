import ImageBind.data as data
from ImageBind.models.x2touch_model_part import x2touch,imagebind_huge
import numpy as np
import torch
import os
import torchvision.transforms.functional as F
from torch.distributed.pipeline.sync import Pipe
from torchvision import transforms
import torch.distributed as dist
import Augmix
from PIL import Image
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam


torch.cuda.empty_cache()
torch.cuda.ipc_collect()

llama_dir = "./unitouch/llama_ori"
dist.init_process_group(backend="nccl")
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

deepspeed_config = {
    "train_micro_batch_size_per_gpu": 1,

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4
        }
    },

    "zero_optimization": {
        "stage": 2,  # ZeRO Stage 2
        "offload_optimizer": {"device": "cpu"},
        "contiguous_gradients": False ,
       "reduce_bucket_size": 5e8
    },
    "fp16": {"enabled": False,
             "loss_scale": 0,
             "initial_scale_power": 16,
             "loss_scale_window": 1000,
             "hysteresis": 2
             }
}


def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return logits[idx], idx

def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - torch.log(torch.tensor(logits.shape[0], dtype=torch.float32, device=logits.device))
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

def test_time_tuning(model, inputs, optimizer, args):
    selected_idx = None
    print("torch.cuda.device_count:",torch.cuda.device_count())
    for j in range(args.tta_steps):
        for key in inputs:
            inputs[key] = inputs[key].to(model.device)
        embeddings = model(inputs)
        similarity = embeddings["touch"] @ embeddings["text"].T
        logits = torch.softmax(similarity, dim=-1)
        if selected_idx is not None:
            logits = logits[selected_idx]
        else:
            logits, selected_idx = select_confident_samples(logits, args.selection_p)
        loss = avg_entropy(logits)
        model.backward(loss,retain_graph=True)
        model.step()
        model.zero_grad()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        print(f"{j+1}/{args.tta_steps} time loss值: {loss.item():.4f}")
        torch.cuda.empty_cache()

    return model


vision_paths = ["./unitouch/rock_rgb.jpg"]
vision = data.load_and_transform_vision_data(vision_paths, device=device).to(torch.float16)


touch_paths=["./datasets/objectfolder_2/touch/974/85.png"]
touch11 = data.load_and_transform_touch_data(touch_paths, device=device)
#print(touch.shape)

base_transform = transforms.Compose(
            [
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224)])
preprocess = transforms.Compose([transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),)
                ])
with open(touch_paths[0], "rb") as fopen:
    touch_image = Image.open(fopen).convert("RGB")
data_transform = Augmix.AugMixAugmenter(base_transform, preprocess,  n_views=3, augmix=True)
touch1 = data_transform(touch_image)
touch = torch.stack(touch1)


class_labels = ["wood","denim","grass","rock","Brick","Leather","cotton"]
class_label = data.load_and_transform_text(class_labels, device=device)

inputs = {
    "touch": touch,
    "text": class_label
}
inputs11 = {
    "touch": touch11,
    "text": class_label
}
model = x2touch(pretrained=True)
model.eval()

# **DeepSpeed initiate**
params = [p for p in model.parameters() if p.requires_grad]
optimizer = DeepSpeedCPUAdam(model.parameters(), lr=1e-4,weight_decay=0.01,adamw_mode=True)
model, optimizer, _, _ = deepspeed.initialize(model=model, optimizer=optimizer,config=deepspeed_config)
model.to(device)


args = type('', (), {})()
args.tta_steps = 4
args.selection_p = 0.5
model = test_time_tuning(model, inputs, optimizer, args)

embeddings = model(inputs11)

similarity = embeddings["touch"] @ embeddings["text"].T
logits = torch.softmax(similarity, dim=-1)
preds = torch.argmax(logits, dim=-1)

print("Similarity matrix:", similarity)
print("Logits:", logits)


