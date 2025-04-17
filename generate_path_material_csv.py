import csv
import os
import json
import pandas as pd

dataset = "objectfolder_real"

data_dir = f"./datasets/{dataset}/touch"
label_file = f"./datasets/{dataset}/label.json"

target_file = f"./datasets/{dataset}/path_material.csv"

label_dict = {}

with open(label_file, 'r') as file:
    data = file.read()
    label_dict = json.loads(data)

objects = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

material_labels = {"path": [], "label": []}

for object in objects:
    object_path = os.path.join(data_dir, object)
    for img_path in os.listdir(object_path):
        if os.path.isfile(os.path.join(object_path, img_path)):
            material_labels["path"].append(os.path.join(object_path, img_path))
            material_labels["label"].append(label_dict[object])


# with open(target_file, "w", newline="") as f:
#     w = csv.DictWriter(f, material_labels.keys())
#     w.writeheader()
#     for i in range(len(material_labels["path"])):
#         w.writerow(material_labels)

df = pd.DataFrame(material_labels)
df.to_csv(target_file, index=False)