"""Meta dataloader for synthetic problems."""

import numpy as np
import torch
from generate_synthetic_data import TYPE_2_PATH
import matplotlib.pyplot as plt

class SyntheticLoader:
    def __init__(self, device, model ="default", problem="default", k_spt="default", k_qry="default"):
        self.device = device
        self.problem = problem
        self.model = model
        self.samples_per_task = 20
        self.show_num = 0
        
        data = np.load(TYPE_2_PATH[problem])
        self.xs, self.ys, self.ws = data["x"], data["y"], data["w"]
        # xs shape: (10000, 20, c_i, ...)
        # ys shape: (10000, 20, c_o, ...)

        if problem == "mnist" and model in ["conv", "share_conv"]:
            # xs shape: (10000, 20, 784) -> (10000, 20, 1, 28, 28)
            n_tasks, n_samples, features = self.xs.shape
            self.xs = self.xs.reshape(n_tasks, n_samples, 1, 28, 28)



        self.c_i, self.c_o = self.xs.shape[2], self.ys.shape[2]
        self.k_spt, self.k_qry = k_spt, k_qry
        assert k_spt + k_qry <= 20, "Max 20 k_spt + k_20"
        train_cutoff = int(0.8 * self.xs.shape[0])
        self.train_range = range(train_cutoff)
        self.test_range = range(train_cutoff, self.xs.shape[0])

    def next(self, n_tasks, mode="train"):
        rnge = self.train_range if mode == "train" else self.test_range
        task_idcs = np.random.choice(rnge, n_tasks, replace=True)
        xs, ys, ws = self.xs[task_idcs], self.ys[task_idcs], self.ws[task_idcs]
        num_examples = xs.shape[1]
        x_spt, y_spt, x_qry, y_qry = [], [], [], []
        for i in range(n_tasks):
            example_idcs = np.random.choice(num_examples, self.k_spt + self.k_qry, replace=True)
            spt_idcs, qry_idcs = example_idcs[: self.k_spt], example_idcs[self.k_spt :]
            x_spt.append(xs[i][spt_idcs])
            y_spt.append(ys[i][spt_idcs])
            x_qry.append(xs[i][qry_idcs])
            y_qry.append(ys[i][qry_idcs])
            #print(x_spt[i][0].shape)
       
            
        x_spt = np.stack(x_spt)
        y_spt = np.stack(y_spt)
        x_qry = np.stack(x_qry)
        y_qry = np.stack(y_qry)
        data = [x_spt, y_spt, x_qry, y_qry]
        data = [torch.from_numpy(x.astype(np.float32)).to(self.device) for x in data]


        fig, axes = plt.subplots(2, self.samples_per_task // 2, figsize=(15, 6))
        axes = axes.flatten() # Pour itérer facilement
        
        if self.show_num < 0:
            # Afficher chaque image sélectionnée
            for i in range(self.samples_per_task):
                
                # Remodeler l'image de (784,) à (28, 28)
                image_28x28 = x_spt[i][0].reshape(28, 28)
                
                # Afficher l'image dans le sous-graphique
                axes[i].imshow(image_28x28, cmap='gray')
                #axes[i].set_title(ls[i].item())
                axes[i].axis('off') # Masquer les axes
                
            plt.suptitle(f"Images MNIST sélectionnées pour la Tâche {self.show_num}")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajuster pour le titre
            plt.show() # Afficher la figure
            self.show_num += 1
        else:
            plt.close()
            



        return data, ws

