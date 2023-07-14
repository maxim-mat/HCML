import torch
import torchvision
from models.DiffusionModel import DiffusionModel
from dataset.dataset import ImageDataset
from easydict import EasyDict as edict
import json
import pickle as pkl
from tqdm import tqdm


def main():
    CREATE_DATASET = False

    with open("config/example_EXP.json", "r") as f:
        cfg = edict(json.load(f))
    if CREATE_DATASET:
        label_path = r"D:/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0/train.csv"
        chexpert_dataset = ImageDataset(label_path, cfg)
        with open("dev/chexpert_dataset_train.pkl", "wb") as f:
            pkl.dump(chexpert_dataset, f)
    else:
        with open("dev/chexpert_dataset_train.pkl", "rb") as f:
            chexpert_dataset = pkl.load(f)

    for _ in tqdm(chexpert_dataset):
        pass

    print("finished")


if __name__ == '__main__':
    main()
