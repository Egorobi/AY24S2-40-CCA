import unitouch.ImageBind.data as data
from unitouch.ImageBind.models.x2touch_model_part import x2touch
import torch
import torch.nn as nn
import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np
import time
import datetime

from classification_tta_zero import classify_tta_zero_single

device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

os.environ["CUDA_HOME"] = "/usr/lib/cuda"
os.environ["LD_LIBRARY_PATH"] = "/usr/lib/cuda/lib64"

dataset = "objectfolder_2"

progress_dir = f"./datasets/evaluation/{dataset}/"

dataset = pd.read_csv(f"./datasets/{dataset}/path_material.csv")

# basic zero shot classification, no tta
def classification_basic(model, inputs, classes):
    # get embeddings
    embeddings = model(inputs)
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

# if torch.cuda.device_count() > 1:
#     model = nn.parallel.DistributedDataParallel(model)

model = model.to(device)
inputs = {}

# shuffle dataset
dataset = dataset.sample(frac=1, random_state=24).reset_index(drop=True)

label_to_material_dict = {0: "wood", 1: "steel", 2: "polycarbonate", 3: "plastic", 4: "iron", 5: "ceramic", 6: "glass"}

targets = []

all_materials = []
for _, row in dataset.iterrows():
    material = label_to_material_dict[row["label"]]
    if not material in all_materials:
        all_materials.append(material)

material_prompts = [f"This feels like {m}" for m in all_materials]
material_prompts = data.load_and_transform_text(material_prompts, device=device)

predictions = {"basic": [], "zero": []}

# slicing for testing
dataset = dataset.iloc[7000:10000]

try:
    for idx, row in tqdm(dataset.iterrows(), total=len(dataset)):
        touch_path = [row["path"]]
        touch_image = data.load_and_transform_vision_data(touch_path, device=device)

        # inputs to classifier are the touch image and the set of all possible classes
        inputs = {
            "touch": touch_image,
            "text": material_prompts
        }

        # make classifications
        prediction = classification_basic(model, inputs, all_materials)
        # predictions_basic.append(prediction)
        predictions["basic"].append(prediction)

        # add prediction for zero tta
        prediction_zero = classify_tta_zero_single(model, inputs, all_materials)
        predictions["zero"].append(prediction_zero)

        # correct label is saved for evaluation
        label = int(row["label"])
        material = label_to_material_dict[label]
        targets.append(material)

        # backup results
        if (idx+1) % 1000 == 0:
            save_results(predictions, targets)
        
    save_results(predictions, targets)
except KeyboardInterrupt:
    pass

accuracy_basic = accuracy_score(targets, predictions["basic"])
accuracy_zero = accuracy_score(targets, predictions["zero"])

print("CLASSIFICATION ACCURACY BASIC: ")
print(accuracy_basic)

print("CLASSIFICATION ACCURACY ZERO: ")
print(accuracy_zero)