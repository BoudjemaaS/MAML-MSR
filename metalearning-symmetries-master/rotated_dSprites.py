import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import matplotlib.pyplot as plt # NOUVEAU: Import pour l'affichage

class dSpritesPerRotationTask(Dataset):
    def __init__(self, path, target_rotation, num_shapes=3, transform=None):
        """
            path (str): Folder where the dSprites dataset is located (default: './data/').
            target_rotation (int): Target rotation bin (0-39, corresponding to specific degrees).
            num_shapes (int): Number of shapes to classify (default: 3).
        """
        data = np.load(path+'dSprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', allow_pickle=True, encoding='bytes')
        self.images = data['imgs']  # Binary images (64x64)
        self.latents = data['latents_classes']  # Latent classes
        self.transform = transform

        # Extract relevant data for the target rotation
        mask = self.latents[:, 3] == target_rotation
        self.images = self.images[mask]
        self.labels = self.latents[mask][:, 1]  # Shape labels (0: square, 1: ellipse, 2: heart)

        if len(self.images) == 0:
            raise ValueError(f"No data found for rotation bin {target_rotation}.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32)
        label = self.labels[idx]

        # Apply transform if provided
        if self.transform:
            image = self.transform(image)

        # Add channel dimension (for grayscale)
        image = torch.tensor(image).unsqueeze(0)  # Shape: (1, 64, 64)
        label = torch.tensor(label, dtype=torch.long)  # Shape label as LongTensor

        return image, label


def create_rotated_dsprites_task(num_tasks, per_task_rotation, batch_size, transform=[], test_batch_size=256):
    train_loaders = []
    test_loaders = []

    g = torch.Generator()
    g.manual_seed(0)  # check: always setting generator to 0 ensures the same ordering of data

    for task in range(num_tasks):
        # In the dSprites dataset, as the range 0°-360° is devided into 40 bins, we get [0, 5, 10, ..., 35] for 8 tasks
        #rotations = [i * (40 // num_tasks) for i in range(num_tasks)]
        rotation_degree = (task-1) * per_task_rotation
        rotation_bin = (rotation_degree // 9) % 40

        print(f"Creating DataLoader for Task {task}: Rotation degree {rotation_degree}° (dSprite's rotation_bin {rotation_bin})")
        #print()
        dataset = dSpritesPerRotationTask(
            path='./data/',
            target_rotation=rotation_bin,
            transform=transform
        )

        train_size = int(0.8 * len(dataset))  # 80% train, 20% test
        test_size = len(dataset) - train_size
        print(f'train_size: {train_size} test_size: {test_size}')
        train, test = random_split(dataset, [train_size, test_size], generator=g.manual_seed(0))

        train_loader = torch.utils.data.DataLoader(train,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, generator=g)  #.manual_seed(0))
        test_loader = torch.utils.data.DataLoader(test,  batch_size=test_batch_size, shuffle=False, num_workers=0, pin_memory=True, generator=g)  #.manual_seed(0))

        train_loaders.append({
            'loader':train_loader,
            'task':task,
            'rot':rotation_degree})
        test_loaders.append({
            'loader':test_loader,
            'task':task,
            'rot':rotation_degree})

    return train_loaders, test_loaders





def display_dsprites_samples(dataset, num_samples=5):
    """
    Affiche un nombre spécifié d'images aléatoires du jeu de données.
    
    Args:
        dataset (dSpritesPerRotationTask): Le jeu de données à visualiser.
        num_samples (int): Le nombre d'échantillons à afficher.
    """
    
    # Définir les noms des classes (labels[:, 1] est la forme)
    shape_names = {0: 'Carré', 1: 'Ellipse', 2: 'Cœur'}
    
    # Déterminer les indices à afficher (aléatoirement)
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    # Définir le sous-graphique
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    # Assurez-vous que axes est iterable même pour un seul échantillon
    if num_samples == 1:
        axes = [axes]

    #print(f"Affichage de {num_samples} échantillons pour la Rotation Cible: {dataset.rotation_degree}° (Bin {dataset.target_rotation})")

    for i, idx in enumerate(indices):
        # Récupérer l'image et le label
        image, label_tensor = dataset[idx]
        
        # Le tenseur est sur CPU (par défaut dans __getitem__), mais nous devons retirer la dimension du canal (1)
        image_np = image.squeeze().numpy() 
        
        label = label_tensor.item()
        
        # Affichage
        axes[i].imshow(image_np, cmap='gray')
        axes[i].set_title(f"Shape: {shape_names.get(label, label)}", fontsize=10)
        axes[i].axis('off')

    plt.suptitle(f"Échantillons dSprites (Rotation Fixe: °)", y=1.05)
    plt.tight_layout()
    plt.show()
    return 0
