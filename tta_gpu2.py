import unitouch.ImageBind.data as data
import llama
from llama.tokenizer import Tokenizer
from unitouch.ImageBind.models.x2touch_model_part import x2touch,imagebind_huge
import numpy as np
import torch
from scipy.interpolate import interp1d
import os
import torchvision.transforms.functional as F
from torch.distributed.pipeline.sync import Pipe
from torchvision import transforms
import torch.distributed as dist
from torch.nn import Sequential
import gzip
import Augmix
from PIL import Image
import torch.nn as nn
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam


torch.cuda.empty_cache()
torch.cuda.ipc_collect()

llama_dir = "./unitouch/llama_ori"
dist.init_process_group(backend="nccl")
#device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
#os.environ["CUDA_HOME"] = "/usr/lib/cuda"
#os.environ["LD_LIBRARY_PATH"] = "/usr/lib/cuda/lib64"
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")  # 关键：每个进程绑定到对应GPU

deepspeed_config = {
    "train_micro_batch_size_per_gpu": 1,

    #"optimizer": {
    #    "type": "AdamW",
    #    "params": {
    #        "lr": 1e-4
    #    }
    #},

    "zero_optimization": {
        "stage": 2,  # ZeRO Stage 2 优化
        "offload_optimizer": {"device": "cpu"},
        "contiguous_gradients": False , # 关闭连续梯度优化以节省内存
       "reduce_bucket_size": 5e8
    },
    "fp16": {"enabled": False,   #是否启用混合精度训练
             "loss_scale": 0,  # 动态损失缩放
             "initial_scale_power": 16,
             "loss_scale_window": 1000,
             "hysteresis": 2
             }
}


def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    # 计算每个 touch image 的熵
    #batch_entropy = -(logits * logits.log()).sum(1)  # 形状: [1, N]
    # 确保 logits 至少选 1 个
    #num_selected = max(1, int(logits.shape[1] * top))  # 现在按列计算，不是按 batch 计算
    #idx = torch.argsort(batch_entropy, descending=False)[:num_selected]  # 选最有信心的 touch image
    return logits[idx], idx

def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - torch.log(torch.tensor(logits.shape[0], dtype=torch.float32, device=logits.device))
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

def test_time_tuning(model, inputs, optimizer, args):
    selected_idx = None
    #scaler = torch.cuda.amp.GradScaler()
    print("torch.cuda.device_count:",torch.cuda.device_count())
    for j in range(args.tta_steps):
        for key in inputs:
            inputs[key] = inputs[key].to(model.device)
        embeddings = model(inputs)
        #stability = pressure_stability(embeddings['touch'], inputs['touch'])
        #conf_idx = torch.topk(stability, k=int(len(stability) * 0.5)).indices
        #similarity = embeddings["touch"][conf_idx] @ embeddings["text"].T
        similarity = embeddings["touch"] @ embeddings["text"].T
        logits = torch.softmax(similarity, dim=-1)
        #print(len(logits))
        if selected_idx is not None:
            logits = logits[selected_idx]
        else:
            logits, selected_idx = select_confident_samples(logits, args.selection_p)
        #print(len(logits))
        loss = avg_entropy(logits)
        # DeepSpeed 自动处理梯度更新
        model.backward(loss,retain_graph=True)
        # 梯度裁剪
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        model.step()
        model.zero_grad()
        # 添加进程同步
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        #optimizer.zero_grad()
        #scaler.scale(loss).backward()
        #torch.cuda.empty_cache()
        #scaler.step(optimizer)
        #scaler.update()
        print(f"第 {j+1}/{args.tta_steps} 次循环 loss值: {loss.item():.4f}")
        torch.cuda.empty_cache()  # 释放显存

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
data_transform = Augmix.AugMixAugmenter(base_transform, preprocess,  n_views=5, augmix=True)
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
# **模型初始化**
model = x2touch(pretrained=True)
model.eval()

# **DeepSpeed 初始化**
params = [p for p in model.parameters() if p.requires_grad]
optimizer = DeepSpeedCPUAdam(model.parameters(), lr=1e-4,weight_decay=0.01,adamw_mode=True)
model, optimizer, _, _ = deepspeed.initialize(model=model, optimizer=optimizer,config=deepspeed_config)
model.to(device)


args = type('', (), {})()
args.tta_steps = 4
args.selection_p = 0.5
model = test_time_tuning(model, inputs, optimizer, args)

# **计算相似度**
embeddings = model(inputs11)
#augmented_embeddings = model(augmented_inputs)

similarity = embeddings["touch"] @ embeddings["text"].T
logits = torch.softmax(similarity, dim=-1)
preds = torch.argmax(logits, dim=-1)

#augmented_similarity = augmented_embeddings["vision"] @ augmented_embeddings["touch"].T
#augmented_logits = torch.softmax(augmented_similarity, dim=-1)

print("Similarity matrix:", similarity)
print("Logits:", logits)
#print("Augmented Logits:", augmented_logits)
#print(f"Predicted image: {predicted_image}")

'''
# checkpoint will be automatically downloaded
model = x2touch(pretrained=True)
model.eval()
model = model.to(device)
inputs = {}

vision_paths=["./unitouch/rock_rgb.jpg"]
vision=data.load_and_transform_vision_data(vision_paths, device=device)
augmented_vision = apply_augmentations(vision)

touch_paths=["./unitouch/wood.jpg","./unitouch/metal.jpg","./unitouch/cotton.jpg","./unitouch/grass.jpg",
             "./unitouch/rock.jpg"]
touch = data.load_and_transform_vision_data(touch_paths, device=device)
augmented_touch = apply_augmentations(touch)

class_labels = ["wood","cotton","metal","Brick","Tile","Leather","Synthetic Fabric"]
class_label=data.load_and_transform_text(class_labels, device=device)
print(class_label.shape)

inputs = {
    "text": class_label,
    "touch": touch,
    "vision":vision
}
embeddings = model(inputs)
#print(model)

augmented_inputs = {
    "text": class_label,
    "touch": touch,
    "vision":augmented_vision
}
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = torch.cuda.amp.GradScaler()

# Apply Test-Time Tuning
args = type('', (), {})()
args.tta_steps = 5
args.selection_p = 0.5
model = test_time_tuning(model, augmented_inputs, optimizer, scaler, args)
augmented_embeddings = model(augmented_inputs)

print(embeddings["touch"].shape) # 形状: [2, 1024]
print(embeddings["text"].shape)# 形状: [2, 77,1024]
text_features = embeddings["text"].max(dim=1).values  #序列维度取平均 形状: [2, 1024]
similarity1 = embeddings["touch"] @ text_features.T  # 形状: [2, 2]

similarity = embeddings["vision"] @ embeddings["touch"].T
logits = torch.softmax(similarity, dim=-1)
preds = torch.argmax(logits, dim=-1)  # 选择相似度最高的类别
pred_index = preds.item()  # 获取索引值
predicted_image = touch_paths[pred_index]  # 取出对应的路径

augmented_similarity = augmented_embeddings["vision"] @ augmented_embeddings["touch"].T
augmented_logits = torch.softmax(augmented_similarity, dim=-1)

print("Similarity matrix:", similarity)
print("Logits:", logits)
print("augmented_Logits:", augmented_logits)
print(f"Predicted image: {predicted_image}")
'''