"""Generate synthetic data for experiments."""

import argparse
import os
from e2cnn import gspaces
from e2cnn import nn as gnn
from scipy.special import softmax
import numpy as np
import torch
from torch import nn
from layers import LocallyConnected1d
from torchvision import datasets
import matplotlib.pyplot as plt



def generate_mnist_tasks_torch(out_path, num_tasks=20000, samples_per_task=20):
    train_dataset = datasets.MNIST(root='./mnist_data', train=True, download=True)
    
    all_images = train_dataset.data.float() / 255.0  # (60000, 28, 28)
    all_images = all_images.reshape(-1, 784)  # (60000, 784)
    all_labels = train_dataset.targets
    
    linear_layer = nn.Linear(784, 10, bias=True)
    
    xs, ys, ws= [], [], []
    
    for task_idx in range(num_tasks):
        # Réinitialiser les poids aléatoirement
        nn.init.normal_(linear_layer.weight, mean=0, std=0.01)
        nn.init.zeros_(linear_layer.bias)
        
        # Sauvegarder les poids
        weights = linear_layer.weight.detach().cpu().numpy()  # (10, 784)
        
        # Sélectionner des exemples
        indices = torch.randint(0, len(all_images), (samples_per_task,))
        task_x = all_images[indices]  # (20, 784)
        ls = all_labels[indices]
        # Appliquer la couche
        with torch.no_grad():
            task_y = linear_layer(task_x)  # (20, 10)
        
        xs.append(task_x.numpy())
        ys.append(task_y.numpy())
        ws.append(weights)
        
        if task_idx % 100 == 0:
            print(f"Finished generating task {task_idx}")
    

        


    xs = np.stack(xs)
    ys = np.stack(ys)
    ws = np.stack(ws)
    
    
    np.savez(out_path, x=xs, y=ys, w=ws)
    #print(f"Shapes - x: {xs.shape}, y: {ys.shape}, w: {ws.shape}")




TYPE_2_PATH = {
    
    "mnist": "./data/mnist_tasks.npz"
}


def main():
   
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="mnist")
    args = parser.parse_args()
    out_path = TYPE_2_PATH[args.problem]
    
    generate_mnist_tasks_torch(out_path,num_tasks=20000)
  


if __name__ == "__main__":
    main()

