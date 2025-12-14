import torch.utils.data
import torchvision
import torchvision.transforms.functional as F

import numpy as np
from PIL import Image

class RotatedMNISTDataset(torch.utils.data.Dataset):
    '''
        This class provides MNIST images with random rotations sampled from
        a list of rotation angles. This list is dependent of the number of tasks
        `num_tasks` and the distance (measured in degrees) between tasks
        `per_task_rotation`.
    '''
    def __init__(self, train=True, transform=None, download=True, num_tasks=5, per_task_rotation=45,problem="mnist"):

        transform = []
        extended_transform = transform.copy()
        extended_transform.extend([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
        transforms = torchvision.transforms.Compose(extended_transform)

        #Chargement des datasets
        self.problem = problem
        if problem == "mnist":
            self.dataset = torchvision.datasets.MNIST(root="./data", train=train, transform=transforms, download=download)
        elif problem == "dsprite":
            self.dataset = np.load("metalearning-symmetries-master\\data\\dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz", allow_pickle=True, encoding='bytes')
        
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

        if self.problem == "dsprite":

            image_np = self.dataset['imgs'][idx].astype(np.float32) #Image originale
            label = self.dataset['latents_classes'][:, 1][idx] #Label original 
            
            pil_image = Image.fromarray((image_np * 255).astype(np.uint8)) #Convertion en PIL.Image 
            image, label = pil_image, label
            
        elif self.problem == "mnist":
            image, label = self.dataset[idx]

        else:
            raise NotImplementedError("Problem not implemented")

        angle = np.random.choice(self.rotation_angles) #Choix aléatoire de l'angle
        rotated_image = F.rotate(image, angle, fill=(0,)) #Rotation de l'image

        train_loader = torch.utils.data.DataLoader((rotated_image, label), batch_size=1000, shuffle=False, num_workers=0, pin_memory=True, generator=g)
        
        return train_loader.dataset #retour de l'image pivotée et de son label

