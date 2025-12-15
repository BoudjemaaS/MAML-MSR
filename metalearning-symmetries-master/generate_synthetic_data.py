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
from rotated_mnist import  RotatedMNISTDataset
import utils
from random import sample
from rotated_dSprites import dSpritesPerRotationTask




def generate_mnist_tasks_torch(out_path, rot_percent, angle_rot, num_tasks=20000, samples_per_task=20):
    '''
    Génération et stockage de données MNIST avec une certaine proportion d'images pivotées.
    
    :param out_path: Localisation de sauvegarde des données générées
    :param rot_percent: Pourcentage d'images classiques à pivoter
    :param angle_rot: Facteur d'angle de rotation des images pivotées en degrés
    :param num_tasks: Nombre de tâches à générer
    :param samples_per_task: Nombre d'exemples par tâche
    '''

    #Génération des données "classiques"
    input_size = 28 * 28
    num_classes = 10
    train_dataset = datasets.MNIST(root='./mnist_data', train=True, download=True)
    all_images = train_dataset.data.float() / 255.0  #Normalisation pixel -> [0 - 1]
    all_images = all_images.reshape(-1, input_size)  # (60000, 28, 28) -> flattened -> (60000, 784)
    all_labels = train_dataset.targets 

    #Generation des données rotated 
    if rot_percent > 0: #Seulement si nécessaire (pourcentage > 0)
        all_images_rotated = []
        all_labels_rotated = []
        
        for i in RotatedMNISTDataset(per_task_rotation=angle_rot): #Parcours des images pivotées générées
            img, label = i # Déclenche __getitem__ et donc la rotation
            all_images_rotated.append(img)  
            all_labels_rotated.append(label)

        all_images_rotated = torch.stack(all_images_rotated).reshape(-1, input_size) # Conversion en tenseur et aplatissement
        all_labels_rotated = torch.tensor(all_labels_rotated) # Conversion en tenseur

        # Sélection aléatoire des indices à remplacer
        data_indices = list(i for i in range (all_labels_rotated.shape[0])) #Indices de toutes les données
        data_to_rotate = int(rot_percent/100 * len(data_indices)) #Calcul du nombre de données à pivoter
        indices = sample(data_indices, data_to_rotate) #Sélection aléatoire des indices
        
        # Remplacement des images et labels originaux par des données pivotées
        all_images[indices] = all_images_rotated[indices]
        all_labels[indices] = all_labels_rotated[indices]

    linear_layer = nn.Linear(input_size, num_classes, bias=True) 
    
    xs, ys, ws= [], [], []
    
    for task_idx in range(num_tasks): 
        #Pour chaque taches

        # Réinitialisation aléatoirement et suavegarde des poids 
        nn.init.normal_(linear_layer.weight, mean=0, std=0.01)
        nn.init.zeros_(linear_layer.bias)
        weights = linear_layer.weight.detach().cpu().numpy()  # (10, 784)
        ws.append(weights)

        # Sélection et stockage des exemples (images)
        indices = torch.randint(0, len(all_images), (samples_per_task,))
        task_x = all_images[indices]  # (samples_per_task, 784)
        xs.append(task_x.numpy())
        
        # Appliquer la couche et stocker les logits
        with torch.no_grad():
            task_y = linear_layer(task_x)  # (20, 10)
        ys.append(task_y.numpy())
    
    xs = np.stack(xs)
    ys = np.stack(ys)
    ws = np.stack(ws)

    #Sauvegarde des données générées
    np.savez(out_path, x=xs, y=ys, w=ws) 
    

def generate_dsprite_tasks_torch(out_path, num_tasks=20000, samples_per_task=20,angle_rot=45, rot_percent=20):
    '''
    Génération et stockage de données Dsprite avec une certaine proportion d'images pivotées.
    
    :param out_path: Localisation de sauvegarde des données générées
    :param num_tasks: Nombre de tâches à générer
    :param samples_per_task: Nombre d'exemples par tâche
    :param angle_rot: Facteur d'angle de rotation des images pivotées en degrés
    :param rot_percent: Pourcentage d'images classiques à pivoter
    
    '''
    #Génération des données "classiques"
    train_dataset = np.load('./data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', allow_pickle=True, encoding='bytes')
    
    # Récupération et convertion (float + tensor) des images
    all_images_np = train_dataset['imgs'].astype(np.float32)
    all_images = torch.from_numpy(all_images_np) 

    # Récupération et convertion (float + tensor) des labels
    all_labels_np = train_dataset['latents_classes'][:, 1].astype(np.long)
    all_labels = torch.from_numpy(all_labels_np)

    input_size = 64 * 64 #Format des images
    num_classes = 3 
    all_images_flat = all_images.reshape(-1, input_size) #(737280,64,64) -> flattened -> (737280,4096) 


    

    #Generation des données rotated 
    if rot_percent > 0: #Seulement si nécessaire (pourcentage > 0)
        all_images_rotated = []
        all_labels_rotated = []

        for i in dSpritesPerRotationTask(angle_rot=angle_rot): #Parcours des images pivotées générées
            img, label = i[0],i[1] # Déclenche __getitem__ et donc la rotation
            all_images_rotated.append(img)  
            all_labels_rotated.append(label)

        all_images_rotated = torch.stack(all_images_rotated).reshape(-1, 64,64)
        all_labels_rotated = torch.stack(all_labels_rotated)

        # Sélection aléatoire des indices à remplacer
        data_indices = list(i for i in range (all_labels_rotated.shape[0])) #Indices de toutes les données
        data_to_rotate = int(rot_percent/100 * len(data_indices)) #Calcul du nombre de données à pivoter
        indices = sample(data_indices, data_to_rotate) #Sélection aléatoire des indices

        # Remplacement des images et labels originaux par des données pivotées
        all_images[indices] = all_images_rotated[indices]
        all_labels[indices] = all_labels_rotated[indices].int()

    linear_layer = nn.Linear(input_size, num_classes, bias=True)

    xs, ys, ws= [], [], []
    
    for task_idx in range(num_tasks):
        #Pour chaque taches

        # # Réinitialisation aléatoirement et suavegarde des poids
        nn.init.normal_(linear_layer.weight, mean=0, std=0.01)
        nn.init.zeros_(linear_layer.bias)
        weights = linear_layer.weight.detach().cpu().numpy()  # (3, 4096)
        ws.append(weights)

        # Sélection et stockage des exemples (images)
        indices = torch.randint(0, len(all_images), (samples_per_task,))
        task_x = all_images_flat[indices]  # (samples_per_task, 4096)
        xs.append(task_x.numpy())


        # Appliquer la couche et stocker les logits
        with torch.no_grad():
            task_y = linear_layer(task_x)  # (20, 3)
        ys.append(task_y.numpy())
       
    xs = np.stack(xs)
    ys = np.stack(ys)
    ws = np.stack(ws)
    
    #Sauvegarde des données générées
    np.savez(out_path, x=xs, y=ys, w=ws)
   
#Emplacement des fichiers générés
TYPE_2_PATH = {
    "mnist": "./data/mnist_tasks.npz",
    "dsprite": "./data/dsprite_tasks.npz",
}


def main():
   
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="mnist") #Dataset ["mnist","dsprite"]
    parser.add_argument("--rot_percent", type=int, default=0) #Pourcentage de données à pivoter [0-100]
    parser.add_argument("--angle_rot", type=int, default=45) #Rotation à appliquer [0-360] Rotations possibles : angle_rot * n avec n = [0 - 360/angle_rot] 
    args = parser.parse_args()
    out_path = TYPE_2_PATH[args.problem]
    
    if args.problem == "dsprite":
        generate_dsprite_tasks_torch(out_path, angle_rot=args.angle_rot)

    elif args.problem == "mnist":
        generate_mnist_tasks_torch(out_path,num_tasks=20000, rot_percent=args.rot_percent, angle_rot=args.angle_rot)

    else:
        raise ValueError(f"Invalid problem {args.problem}")
    
if __name__ == "__main__":
    main()

