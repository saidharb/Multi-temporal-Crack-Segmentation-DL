# Models

This directory contains all files related to the Swin UNETR and U-Net model, as well as some figures which were created for the report.

## U-Net
- `UNet.ipynb`: Notebook containing the workflow to train the U-Net on the mono-temporal dataset.
- `augmentations.py`: Contains the data augmentation used for some trainings.
- `dataset.py`: Contains the classes for `UNet.ipynb`.
- `helper.py`: Contains functions for preprocessing the dataset.
- `metrics.py`: Contains the script to calculate the metrics during training.
- `preprocessing.py`: Preprocessing pipeline.
- `train.py`: Training scrip for the U-Net.
- `unet.py`: Contains the script for the U-Net.
- `utils.py`: Contains additional functions used in `UNet.ipynb`.

## Swin UNETR
- `Swin_UNETR.ipynb`: Notebook containing the workflow to train the Swin UNETR on the multi-temporal dataset. This notebook also includes the evaluation script for both models and some scripts to create figures in the report.
- `visualization.ipynb`: Notebook containing the workflow to create predictions of complete Weimar experiment images and to create many figures used in the report.
- `augmentations_UNETR.py`: Contains the data augmentation used for some trainings.
- `augmentations_exp3.py`: Contains the data augmentation for the U-Net in experiment 3.
- `dataset_UNETR.py`: Contains the classes for `Swin_UNETR.ipynb`.
- `metrics_UNETR.py`: Contains the script to calculate the metrics during training.
- `train_UNETR.py`: Training scrip for the Swin UNETR.
- `unet.py`: Contains the script for the U-Net.
- `utils_UNETR.py`: Contains additional functions used in `Swin_UNETR.ipynb`.
