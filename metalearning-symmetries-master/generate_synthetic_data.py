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
import torchvision
import matplotlib.pyplot as plt
import rotated_mnist
from rotated_mnist import flattened_rotMNIST, tasks_rotMNIST
import utils
from random import sample



def generate_mnist_tasks_torch(out_path, num_tasks=20000, samples_per_task=20):
    train_dataset = datasets.MNIST(root='./mnist_data', train=True, download=True)
    #print(type(train_dataset.data))
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


def generate_rotated_mnist_tasks_torch(out_path, rot_percent, angle_rot, num_tasks=20000, samples_per_task=20):


    train_dataset = datasets.MNIST(root='./mnist_data', train=True, download=True)
    all_images = train_dataset.data.float() / 255.0  # (60000, 28, 28)
    all_images = all_images.reshape(-1, 784)  # (60000, 784)
    all_labels = train_dataset.targets


    #Generer données rotated 
    #train_dataset_rotated, train_dataset_rotated = flattened_rotMNIST(360/angle_rot,angle_rot,1) #nb position possible - angle
    all_images_rotated = []
    all_labels_rotated = []
    for i in flattened_rotMNIST(360/angle_rot,angle_rot,1)[0]:
        img, label, angle = i # Déclenche __getitem__ et donc la rotation
        all_images_rotated.append(img)  # Filtrer les images avec label 6 ou 9
        all_labels_rotated.append(label)
    all_images_rotated = torch.stack(all_images_rotated).reshape(-1, 784)
    all_labels_rotated = torch.tensor(all_labels_rotated)
    

    
    rot_percent = rot_percent
    #print(all_labels_rotated.shape[0])

    indices = sample(list(i for i in range(all_labels_rotated.shape[0])), int(rot_percent/100 * len(all_images)))
    all_images[indices] = all_images_rotated[indices]
    all_labels[indices] = all_labels_rotated[indices]


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
        #print(task_x)
        ls = all_labels[indices]
        #print(ls)
        # Appliquer la couche
        with torch.no_grad():
            task_y = linear_layer(task_x)  # (20, 10)
            #print(task_y)
            
        xs.append(task_x.numpy())
        ys.append(task_y.numpy())
        ws.append(weights)
        
        #if task_idx % 100 == 0:
            #print(f"Finished generating task {task_idx}")
    
    xs = np.stack(xs)
    ys = np.stack(ys)
    ws = np.stack(ws)
    
    np.savez(out_path, x=xs, y=ys, w=ws)
    #print(f"Shapes - x: {xs.shape}, y: {ys.shape}, w: {ws.shape}")






def generate_dsprite_tasks_torch(out_path, num_tasks=20000, samples_per_task=20):
    
    train_dataset = np.load('./data/dSprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', allow_pickle=True, encoding='bytes')
    

    '''
    all_images = train_dataset.data.float() / 255.0  # (60000, 28, 28)
    all_images = all_images.reshape(-1, 784)  # (60000, 784)
    all_labels = train_dataset.targets
    
    linear_layer = nn.Linear(784, 10, bias=True)
    
    xs, ys, ws= [], [], []
    '''



    all_images_np = train_dataset['imgs'].astype(np.float32)
    # 'latents_classes' contient les facteurs latents. L'indice 1 correspond à la forme (3 classes: 0, 1, 2).
    all_labels_np = train_dataset['latents_classes'][:, 1].astype(np.long)

    # Convertir en tenseurs PyTorch (sans division par 255.0 car c'est déjà binaire 0/1)
    all_images = torch.from_numpy(all_images_np) 
    all_labels = torch.from_numpy(all_labels_np)

    # Aplatir les images pour la couche linéaire : (N, 64, 64) -> (N, 4096)
    input_size = 64 * 64
    num_classes = 3 # Forme: Carré, Ellipse, Cœur
    all_images_flat = all_images.reshape(-1, input_size)

    # --- 3. Définition du Méta-Modèle Linéaire ---
    # Le modèle doit s'adapter à la taille dSprites (4096) et aux 3 classes.
    linear_layer = nn.Linear(4096, 3, bias=True)

    # --- 4. Initialisation des listes de sortie ---
    xs, ys, ws= [], [], []




    
    for task_idx in range(num_tasks):
        # Réinitialiser les poids aléatoirement
        nn.init.normal_(linear_layer.weight, mean=0, std=0.01)
        nn.init.zeros_(linear_layer.bias)
        
        # Sauvegarder les poids
        weights = linear_layer.weight.detach().cpu().numpy()  # (10, 784)
        
        # Sélectionner des exemples
        indices = torch.randint(0, len(all_images), (samples_per_task,))
        task_x = all_images_flat[indices]  # (20, 784)
        ls = all_labels[indices]
        # Appliquer la couche
        with torch.no_grad():
            task_y = linear_layer(task_x)  # (20, 10)
        
        xs.append(task_x.numpy())
        ys.append(task_y.numpy())
        ws.append(weights)
        
        #if task_idx % 100 == 0:
            #print(f"Finished generating task {task_idx}")
    

        


    xs = np.stack(xs)
    ys = np.stack(ys)
    ws = np.stack(ws)
    
    
    np.savez(out_path, x=xs, y=ys, w=ws)
    #print(f"Shapes - x: {xs.shape}, y: {ys.shape}, w: {ws.shape}")







TYPE_2_PATH = {
    
    "mnist": "./data/mnist_tasks.npz",
    "dsprite": "./data/dsprite_tasks.npz",
    "rotated_mnist": "./data/rotated_mnist_tasks.npz",

}


def main():
   
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="rotated_mnist")
    parser.add_argument("--rot_percent", type=int, default=20)
    parser.add_argument("--angle_rot", type=int, default=45)
    args = parser.parse_args()
    out_path = TYPE_2_PATH[args.problem]
    
    if args.problem == "dsprite":
        generate_dsprite_tasks_torch(out_path,num_tasks=20000)
    elif args.problem == "rotated_mnist":
        generate_rotated_mnist_tasks_torch(out_path,num_tasks=20000, rot_percent=args.rot_percent, angle_rot=args.angle_rot)
    elif args.problem == "mnist":
        generate_mnist_tasks_torch(out_path,num_tasks=20000)
  


if __name__ == "__main__":
    main()

