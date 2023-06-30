import torch
import torchvision
from models.DiffusionModel import DiffusionModel


def main():
    t = torch.rand((3, 3, 3)).to("cuda:0")
    print(t)
    sd = DiffusionModel()


if __name__ == '__main__':
    main()
