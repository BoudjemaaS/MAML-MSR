import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import matplotlib.pyplot as plt # NOUVEAU: Import pour l'affichage
from random import choice
import random
import math
class dSpritesPerRotationTask(Dataset):
    def __init__(self,angle_rot):
        """
        Renvoie une image avec un pivot qui correspond
        :param per_task_rotation: Angle à appliquer
        """
        data = np.load('.\\data\\dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', allow_pickle=True, encoding='bytes')
        self.images = data['imgs']  # Binary images (64x64)
        self.latents = data['latents_classes']  # Latent classes
        self.transform = []

        #Choix aléatoire d'une image (et label) avec le bon pivot

        possible_rotations = []
        for i in range(int(360 / angle_rot)):
            possible_rotations.append(int(i * angle_rot/9))

        mask = self.latents[:, 3] == choice(possible_rotations)  # Filtrer par rotation
        self.images = self.images[mask]
        self.labels = self.latents[mask][:, 1]  # Shape labels (0: square, 1: ellipse, 2: heart)

        if len(self.images) == 0:
            raise ValueError(f"No data found for rotation bin {angle_rot}.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = self.images[idx].astype(np.float32)
        label = self.labels[idx]

        image = torch.tensor(image).unsqueeze(0)  # Reshape (1, 64, 64)
        label = torch.tensor(label, dtype=torch.long)  # Convertion ->  Tensor

        return image, label






