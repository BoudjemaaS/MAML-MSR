"""Main training script for synthetic problems."""

import argparse
import os
import time
import scipy.stats as st
import wandb
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import higher

import layers
from synthetic_loader import SyntheticLoader
from inner_optimizers import InnerOptBuilder
import matplotlib.pyplot as plt
#from visualize import visualize_predictions

OUTPUT_PATH = "./outputs/synthetic_outputs"


def train(step_idx, data, net, inner_opt_builder, meta_opt, n_inner_iter):
    '''
    Fonction d'entrainement du modele a chaque step
    
    :param step_idx: Step actuel de l'entrainement 
    :param data: Ensemble des taches (images,labels) split en support et query
    :param net: Reseau de neuronne
    :param inner_opt_builder: Optimiseur de la boucle interieure
    :param meta_opt: Optimiseur des metaparametres
    :param n_inner_iter: Nombre d'execution de la boucle interieure
    '''
    
    x_spt, y_spt, x_qry, y_qry = data #unpack des data
    task_num = x_spt.size()[0] #taille du train_set
    inner_opt = inner_opt_builder.inner_opt
    qry_losses = [] #Historique des couts

    meta_opt.zero_grad() #Réinitialisation des gradients pour la boucle exterieure

    for i in range(task_num):
        

        with higher.innerloop_ctx(net, inner_opt, copy_initial_weights=False, override=inner_opt_builder.overrides,
        ) as (fnet, diffopt,): #Hight Level Optimizaiton
            
            for _ in range(n_inner_iter):
                spt_pred = fnet(x_spt[i]) #Calcul des prédictions sur train_set
                spt_loss = F.mse_loss(spt_pred, y_spt[i]) #Calcul de la fonction de cout
                diffopt.step(spt_loss) #MAJ des poids pour la tache

            qry_pred = fnet(x_qry[i]) #Calcul des prédictions sur test_set
            qry_loss = F.mse_loss(qry_pred, y_qry[i]) #Calcul de la fonction de cout
            qry_losses.append(qry_loss.detach().cpu().numpy()) #MAJ de l'historique du cout
            qry_loss.backward() #Le gradient remonte dans la boucle interieure


    # Plot de la loss sur "Weights and Biases"
    metrics = {"train_loss": np.mean(qry_losses)}
    wandb.log(metrics, step=step_idx)

    # Acumulation des gradient sur les parametres initiaux
    all_metaparams = inner_opt_builder.metaparams.values()
    torch.nn.utils.clip_grad_norm_(all_metaparams, max_norm=5.0)
    
    meta_opt.step() # MAJ finale aveec les gradients accumulés


def test(step_idx, data, net, inner_opt_builder, n_inner_iter,problem):
    '''
    Fonction de test du modele
    
    :param step_idx: Step actuel de l'entrainement 
    :param data: Ensemble des taches (images,labels) split en support et query
    :param net: Reseau de neuronne
    :param inner_opt_builder: Optimiseur de la boucle interieure
    :param n_inner_iter: Nombre d'execution de la boucle interieure
    :param problem: Type du dataset
    
    '''

    x_spt, y_spt, x_qry, y_qry = data # Unpack des data
    task_num = x_spt.size()[0] #taille du test_set
    inner_opt = inner_opt_builder.inner_opt

    qry_losses = [] #Historique des couts
    total_acc = 0 #Accuracy moyenne du test_set
    all_true_label = [] # Labels reels  
    all_pred = [] # Labels prédits

    #Création label de classe selon le dataset
    if problem == "mnist":
        class_name = [str(i) for i in range(10)]
    else:
        class_name = ['square', 'ellipse', 'heart']


    for i in range(task_num):
        
        with higher.innerloop_ctx(net, inner_opt, track_higher_grads=False, override=inner_opt_builder.overrides,
        ) as (fnet, diffopt,): #Hight Level Optimizaiton
            
            for _ in range(n_inner_iter):
                spt_pred = fnet(x_spt[i]) #Calcul des prédictions sur train_set
                spt_loss = F.mse_loss(spt_pred, y_spt[i]) #Calcul de la fonction de cout
                diffopt.step(spt_loss) #MAJ des poids pour la tache


            qry_pred = fnet(x_qry[i]) #Calcul des prédictions sur test_set
            qry_loss = F.mse_loss(qry_pred, y_qry[i]) # Calcul de la fonction de cout

            num_acc = 0 #nombre de taches correctement prédites

            predicted_labels = torch.argmax(qry_pred, dim=1)
            real_labels = torch.argmax(y_qry[i], dim=1)
            for a,b in zip(predicted_labels,real_labels):
                if a==b:
                    #si le label est correctemetn prédit
                    num_acc+=1

            total_acc += num_acc / len(qry_pred) #MAJ de l'accuracy moyenne de l'ensemble des tests
            all_true_label.extend(real_labels.cpu().numpy()) #Inrémentaion liste vrais labels
            all_pred.extend(predicted_labels.cpu().numpy()) #Inrémentaion liste labels prédits
            
            qry_losses.append(qry_loss.detach().cpu().numpy()) #Incrémentation de l'historique de cout


    total_acc = (total_acc / task_num)*100 #Calcul de l'accuracy moyenne de l'ensemble des tests (en pourcentage)
    avg_qry_loss = np.mean(qry_losses) #Calcul du cout moyenn de l'ensemble des tests

    _low, high = st.t.interval(0.95, len(qry_losses) - 1, loc=avg_qry_loss, scale=st.sem(qry_losses)) #intervalle de confiance

    #Plot du cout et de l'accuracy sur "Weights and Biases"
    test_metrics = {"test_loss": avg_qry_loss, "test_err": high - avg_qry_loss}
    test_metrics_acc = {"test_accuracy": total_acc}
    wandb.log(test_metrics, step=step_idx)
    wandb.log(test_metrics_acc, step=step_idx)
    wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,y_true=all_true_label, preds=all_pred,class_names=class_name)})
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_inner_lr", type=float, default=0.001) # Learning rate de la boucle interieure
    parser.add_argument("--outer_lr", type=float, default=0.001) # Learning rate de la boucle exterieure
    parser.add_argument("--k_spt", type=int, default=10) # Taille d'echantillon de train par tache  k_spt+k_qry <= 20
    parser.add_argument("--k_qry", type=int, default=10) # Taille d'echantillon de test par tache
    parser.add_argument("--lr_mode", type=str, default="per_param") #Type de Learning rate ["per_param","per_layer"]
    parser.add_argument("--num_inner_steps", type=int, default=5) #Nombre d'itération de la boucle interieure
    parser.add_argument("--num_outer_steps", type=int, default=300) #Nombre d'itération de la boucle exterieure
    parser.add_argument("--inner_opt", type=str, default="maml") #Optimiseur de la boucle interieure 
    parser.add_argument("--outer_opt", type=str, default="Adam") #Optimiseur de la boucle exterieure
    parser.add_argument("--problem", type=str, default="mnist") #Dataset ["mnist","dsprite"]
    parser.add_argument("--model", type=str, default="conv") #Modele du reseau de neuronnes ["fc","share_fc","conv","share_conv"]
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu") #device d'execution ["cuda","cpu"]

    #Création du fichier de sortie
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    #Connection et initialisation de "Weights and Biases" 
    wandb.init(project="weight_sharing_toy", dir=OUTPUT_PATH)
    args = parser.parse_args()
    wandb.config.update(args)
    cfg = wandb.config
    device = torch.device(cfg.device)
    db = SyntheticLoader(device, model=cfg.model, problem=cfg.problem, k_spt=cfg.k_spt, k_qry=cfg.k_qry)

    # Initialisation des réseau de neuronnes en fonction du model choisi
    if cfg.problem == "mnist":
        if cfg.model == "fc": #MAML+FC
            net = nn.Sequential(nn.Linear(784, 10, bias=True)).to(device)
        elif cfg.model == "share_fc": #MSR+FC
            net = nn.Sequential(layers.ShareLinearFull(784, 10, bias=True, latent_size=50)).to(device)
        elif cfg.model == "conv": #MAML+Conv
            net = nn.Sequential(nn.Conv2d(1, 32, 3, bias=True), nn.Flatten(), nn.Linear(21632, 10, bias=True)).to(device)
        elif cfg.model == "share_conv": #MSR+Conv
            net = nn.Sequential(layers.ShareConv2d(1, 32, 3, bias=True), nn.Flatten(), nn.Linear(21632, 10, bias=True)).to(device)
        else:
            raise ValueError(f"Invalid model {cfg.model} for mnist")

    if cfg.problem == "dsprite":
        if cfg.model == "fc": #MAML+FC
            net = nn.Sequential(nn.Linear(4096, 3, bias=True)).to(device)
        elif cfg.model == "share_fc": #MSR+FC
            net = nn.Sequential(layers.ShareLinearFull(4096, 3, bias=True, latent_size=50)).to(device)
        elif cfg.model == "conv": #MAML+Conv
            net = nn.Sequential(nn.Conv2d(1, 32, 3, bias=True), nn.Flatten(), nn.Linear(123008, 3, bias=True)).to(device)
        elif cfg.model == "share_conv": #MSR+Conv
            net = nn.Sequential(layers.ShareConv2d(1, 32, 3, bias=True), nn.Flatten(), nn.Linear(123008, 3, bias=True)).to(device)
        else:
            raise ValueError(f"Invalid model {cfg.model} for dsprite")

    #Optimiseur boucle interne
    inner_opt_builder = InnerOptBuilder(
        net, device, cfg.inner_opt, cfg.init_inner_lr, "learned", cfg.lr_mode)
    
    #Optimiseur boucle externe
    if cfg.outer_opt == "SGD":
        meta_opt = optim.SGD(inner_opt_builder.metaparams.values(), lr=cfg.outer_lr)
    else:
        meta_opt = optim.Adam(inner_opt_builder.metaparams.values(), lr=cfg.outer_lr)


    start_time = time.time()
    for step_idx in range(cfg.num_outer_steps):
        data, _filters = db.next(32, "train") #Nombre de taches d'entrainement
        train(step_idx, data, net, inner_opt_builder, meta_opt, cfg.num_inner_steps)


        if step_idx == 0 or (step_idx + 1) % 50 == 0: #Frequence de test, tout les x steps
            test_data, _filters  = db.next(500, "test") #Nombre de taches de test
            test(
                step_idx,
                test_data,
                net,
                inner_opt_builder,
                cfg.num_inner_steps,
                problem=cfg.problem
            )
            if step_idx > 0:
                #Calcul du temps et de la vitesse d'execution des taches
                steps_p_sec = (step_idx + 1) / (time.time() - start_time)
                wandb.log({"steps_per_sec": steps_p_sec}, step=step_idx)
                print(f"Step: {step_idx}. Steps/sec: {steps_p_sec:.2f}")

if __name__ == "__main__":
    main()
