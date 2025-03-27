import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloader(dataset="mnist", batch_size=64):
    """
    下载并加载数据集
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 归一化到 [-1, 1]
    ])
    
    if dataset == "mnist":
        data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    elif dataset == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 适用于 CIFAR-10
        ])
        data = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    else:
        raise ValueError("Unknown dataset")

    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    return dataloader
