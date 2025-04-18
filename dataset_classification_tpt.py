import ImageBind.data as data
from ImageBind.models.x2touch_model_part import x2touch
import torch
import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torchvision import transforms
import time
import datetime
import Augmix
from PIL import Image
import torch.nn as nn
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam


device = "cuda" if torch.cuda.is_available() else "cpu"


dataset = "objectfolder_2.0"
progress_dir = f"./datasets/evaluation/{dataset}/"
dataset = pd.read_csv(f"./datasets/objectfolder_real/path_material.csv")

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
    "fp16": {"enabled": True,
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
        model.backward(loss)
        model.step()
        model.zero_grad()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        #print(f"第 {j+1}/{args.tta_steps} times loss: {loss.item():.4f}")
        torch.cuda.empty_cache()

    return model

# basic zero shot classification, no tta
def classification_basic(model, inputs, classes):
    # get embeddings
    embeddings = model.module(inputs)
    touch_embeddings = embeddings["touch"]
    text_embeddings = embeddings["text"]

    # compute similarity
    similarity = touch_embeddings @ text_embeddings.T
    probs = torch.softmax(similarity, dim=-1)

    return classes[probs.argmax()]

def save_results(predictions, targets):
    results_dict = predictions
    results_dict["target"] = targets
    results_df = pd.DataFrame(results_dict)
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime(r'%Y%m%d-%H-%M-%S')
    results_df.to_csv(os.path.join(progress_dir, timestamp+".csv"), index=False)

model = x2touch(pretrained=True)
model.eval()
model = model.to(device)
inputs = {}
text_params = []

text_params = [model.modality_preprocessors.text.token_embedding.weight]
optimizer = DeepSpeedCPUAdam(text_params, lr=1e-4, weight_decay=0.01, adamw_mode=True)
model, optimizer, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=deepspeed_config)

dataset = pd.read_csv("./datasets/objectfolder_real/path_material.csv")
print(dataset.head())

label_to_material_dict = {0: "wood",
                          1: "steel",
                          2: "polycarbonate",
                          3: "plastic",
                          4: "iron",
                          5: "ceramic",
                          6: "glass"}

targets = []

all_materials = []
for _, row in dataset.iterrows():
    material = label_to_material_dict[row["label"]]
    if not material in all_materials:
        all_materials.append(material)

material_prompts = [f"This feels like {m}" for m in all_materials]
material_prompts = data.load_and_transform_text(material_prompts, device=device)

predictions_basic = []

# slicing for testing
dataset=dataset.sample(frac=1).reset_index(drop=True)
dataset = dataset.iloc[:]

for idx, row in tqdm(dataset.iterrows(), total=len(dataset)):
    touch_path = [row["path"]]
    touch11 = data.load_and_transform_vision_data(touch_path, device=device)
    base_transform = transforms.Compose(
        [
            transforms.Resize(
                224, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(224)])
    preprocess = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711), )
                                     ])
    with open(touch_path[0], "rb") as fopen:
        touch_image = Image.open(fopen).convert("RGB")
    data_transform = Augmix.AugMixAugmenter(base_transform, preprocess, n_views=2, augmix=True)
    touch1 = data_transform(touch_image)
    touch = torch.stack(touch1)

    # correct label is saved for evaluation
    label = int(row["label"])
    material = label_to_material_dict[label]
    targets.append(material)

    # inputs to classifier are the touch image and the set of all possible classes
    inputs = {
        "touch": touch,
        "text": material_prompts
    }

    args = type('', (), {})()
    args.tta_steps = 3
    args.selection_p = 0.5
    model1 = test_time_tuning(model, inputs, optimizer, args)

    inputs11 = {
        "touch": touch11,
        "text": material_prompts
    }
    prediction = classification_basic(model1, inputs11, all_materials)
    predictions_basic.append(prediction)
    if (idx + 1) % 500 == 0:
        #save_results(predictions_basic, targets)
        accuracy = accuracy_score(targets, predictions_basic)
        print("CLASSIFICATION ACCURACY: ")
        print(accuracy)

accuracy = accuracy_score(targets, predictions_basic)

# for i in range(len(predictions_basic)):
#     print(f"{targets[i]} {predictions_basic[i]}")

print("CLASSIFICATION ACCURACY: ")
print(accuracy)