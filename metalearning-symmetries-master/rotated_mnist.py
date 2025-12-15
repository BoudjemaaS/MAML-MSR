import torch.utils.data
import torchvision
import torchvision.transforms.functional as F

import numpy as np
from PIL import Image

class RotatedMNISTDataset(torch.utils.data.Dataset):
    '''
        Retourne une version pivotée de l'image
        :param per_task_rotation: Angle à appliquer
    '''
    def __init__(self, per_task_rotation=45):

        transform = []
        extended_transform = transform.copy()
        extended_transform.extend([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
        transforms = torchvision.transforms.Compose(extended_transform)

        #Chargement du dataset
        self.dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transforms, download=True)
        
        #Rotations possibles 
        self.rotation_angles = []
        for task in range(int(360 / per_task_rotation)):
            self.rotation_angles.append(float((task) * per_task_rotation))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        '''
        Retourne une image retournée avec son label
        :param idx: Index de l'image dans le dataset
        '''
        g = torch.Generator()
        g.manual_seed(0) 

        image, label = self.dataset[idx]

        angle = np.random.choice(self.rotation_angles) #Choix aléatoire de l'angle
        rotated_image = F.rotate(image, angle, fill=(0,)) #Rotation de l'image

        train_loader = torch.utils.data.DataLoader((rotated_image, label), batch_size=1000, shuffle=False, num_workers=0, pin_memory=True, generator=g)
        
        return train_loader.dataset #Retour de l'image pivotée et de son label

