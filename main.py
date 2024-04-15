import argparse
import os.path

import numpy as np
import pandas as pd
import torch
import torchvision
from Classification.model.classifier import Classifier
from Diffusion.ddpm_conditional import Diffusion
from Diffusion.modules import UNet_conditional
from dataset.dataset import ImageDataset
from easydict import EasyDict as edict
import json
import pickle as pkl
from tqdm import tqdm
from torchinfo import summary
import plotly.express as ex
from PIL import Image

BATCH_SIZE = 5
LABEL_NAMES = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']


def save_images(images, labels, path, label_path, index):
    for i, (img, label) in enumerate(zip(images, labels)):
        full_img_path = os.path.join(path, f"{index}_{i}.jpg")
        img_array = np.transpose(img.cpu().numpy(), (1, 2, 0))
        Image.fromarray(img_array).save(full_img_path)
        header = ["Path", "Sex", "Age", "Frontal/Lateral", "AP/PA", "No Finding",
                  "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
                  "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
                  "Support Devices"]
        new_row = {k: None for k in header}
        new_row['Path'] = full_img_path
        for l_name, l_value in zip(LABEL_NAMES, label):
            new_row[l_name] = l_value.item()
        if os.path.exists(label_path):
            pd.DataFrame([new_row]).to_csv(label_path, mode='a', header=False, index=False)
        else:
            pd.DataFrame([new_row]).to_csv(label_path, mode='a', header=True, index=False)


def generate_images(probs, amounts, model, diffuser, args, device='cuda:0'):
    for prob, amount in zip(probs, amounts):
        print(amount)
        for i in tqdm(range(amount)):
            print(i)
            labels = torch.tensor((np.random.uniform(size=(BATCH_SIZE, prob.size)) < prob).astype(int)).long().to(
                device)
            sampled_images = diffuser.sample(model, n=len(labels), labels=labels)
            save_images(sampled_images, labels, args.img_aug_dir, args.label_aug_dir, i)


def dataset_stats():
    CREATE_DATASET = False

    with open("config/example_EXP.json", "r") as f:
        cfg = edict(json.load(f))

    if CREATE_DATASET:
        label_path = r"D:/chexpert/chexpertchestxrays-u20210408/CheXpert-v1.0/train.csv"
        chexpert_dataset = ImageDataset(label_path, cfg)
        with open("dev/datasets/chexpert_128_aug.pkl", "wb") as f:
            pkl.dump(chexpert_dataset, f)
    else:
        with open("dev/datasets/chexpert_128_aug.pkl", "rb") as f:
            chexpert_dataset = pkl.load(f)

    label_names = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']
    labels_total = np.sum(np.array(chexpert_dataset._labels), axis=0)
    total_sum = sum(labels_total)
    fig = ex.bar(x=label_names, y=labels_total)
    fig.show()
    print([x / total_sum for x in labels_total])
    print(labels_total)
    print(total_sum)
    l = chexpert_dataset._labels
    l_0 = [x for x in l if x[0] == 1]
    l_2 = [x for x in l if x[2] == 1]
    l_0_conditional_prob = np.sum(np.array(l_0), axis=0) / 54000
    l_2_conditional_prob = np.sum(np.array(l_2), axis=0) / 29566
    print(l_0_conditional_prob)
    print(l_2_conditional_prob)
    # chexpert_classifier = Classifier(cfg)
    # summary(chexpert_classifier, input_size=(1, 3, 512, 512))


def main(args):
    device = args.device
    with open(args.cfg_path, 'r') as f:
        cfg = edict(json.load(f))

    model = UNet_conditional(input_dim=cfg.long_side, num_classes=len(cfg.num_classes), device=device).to(device)
    diffusion = Diffusion(img_size=cfg.long_side, device=device)
    ckpt = torch.load(args.model, map_location=device)
    model.load_state_dict(ckpt)

    # np.array([1.0, 0.50851852, 0.05318519, 0.29859259, 0.43837037]),
    probs = (
        np.array([0.09713861, 0.26719881, 1.0, 0.2916864, 0.50077792])
    )
    # 3000,
    amounts = [5000]

    generate_images(probs, amounts, model, diffusion, args)
    # TODO new dataset with added images


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', default=r"dev/diffusion/runs/conditional_0/config.json",
                        help="configuration path")
    parser.add_argument('--model', default=f"dev/diffusion/models/conditional_0/ckpt_99.pt",
                        help="generation model weights")
    parser.add_argument('-d', '--device', default="cuda:0", help="device for networks")
    parser.add_argument('--img_aug_dir', default=r"D:\chexpert\chexpertchestxrays-u20210408\CheXpert-v1.0\generated_2")
    parser.add_argument('--label_aug_dir',
                        default=r"D:\chexpert\chexpertchestxrays-u20210408\CheXpert-v1.0\generated_2.csv")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
