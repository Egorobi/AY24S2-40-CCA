import os
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score

dataset = "objectfolder_2"

backups_dir = f"./datasets/evaluation/{dataset}/"

# get all backups and sort by timestamp
backups = [d for d in os.listdir(backups_dir) if os.path.isfile(os.path.join(backups_dir, d))]
backups.sort(key=lambda date: datetime.strptime(date[:-4], r'%Y%m%d-%H-%M-%S'))

# get latest backup
latest = backups[-1]
print(f"Latest backup: {latest[:-4]}")
latest_df = pd.read_csv(os.path.join(backups_dir, latest))
print(f"Total samples: {len(latest_df)}")

# calculate metrics
for method in ["basic", "zero"]:
    accuracy = accuracy_score(latest_df[method].tolist(), latest_df["target"].tolist())

    print(f"{method} accuracy:")
    print(accuracy)
