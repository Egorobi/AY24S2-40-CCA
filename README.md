# AY24S2-40-CCA
Project files for EE6008

# Usage
This code is developed in Python 3.8 on a Linux system and is based on UniTouch (https://github.com/cfeng16/UniTouch/tree/main)

## Structure
In order to work, the unitouch repository folder should be placed in the same directory as the files contained here.

Datasets should be placed in a "datasets" folder in the same directory, below is an example of the structure for the ObjectFolder-Real dataset (https://objectfolder.stanford.edu/objectfolder-real-download).

The generate_path_material_csv.py file can be used to generate the csv used by dataset evaluation code to get image paths and ground truth labels.

```
datasets 
└───objectfolder_real
    └───touch
unitouch
classification_tta_zero.py
...
```

## Files
classification_tta_zero.py - classification code and demo using the ZERO TTA method
tta_gpu2.py - classification demo using the TPT method
Augmix.py - augmentaion functions used by TPT
dataset_classification_zero.py - runs ZERO aided classification on a given dataset and saves results
dataset_classification_tpt.py - runs TPT aided classification on a given dataset
evaluate_backup.py - calculates metrics for saved classificaiton results