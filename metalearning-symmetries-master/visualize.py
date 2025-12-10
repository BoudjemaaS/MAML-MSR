import argparse
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import higher

# Imports pour l'évaluation
from synthetic_loader import SyntheticLoader
from inner_optimizers import InnerOptBuilder
import layers # Assurez-vous d'importer votre module layers

# Utilisation de matplotlib et seaborn pour l'affichage uniquement
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_PATH = "./outputs/synthetic_outputs"

# --- Fonction de remplacement pour la matrice de confusion (sans sklearn) ---
def numpy_confusion_matrix(y_true, y_pred, labels):
    """
    Calcule la matrice de confusion à l'aide de NumPy.
    
    y_true (np.array): Vraies étiquettes.
    y_pred (np.array): Étiquettes prédites.
    labels (list/array): Liste des classes (ex: [0, 1, ..., 9]).
    """
    K = len(labels)
    # Initialiser la matrice de taille KxK
    cm = np.zeros((K, K), dtype=int)
    
    # Assurer que les entrées sont du même type et de la même longueur
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true et y_pred doivent avoir la même longueur.")

    # Remplir la matrice
    for i in range(len(y_true)):
        true_label = y_true[i]
        pred_label = y_pred[i]
        # On suppose que les labels sont 0..9, ce qui correspond directement à l'index
        # cm[vrai][prédit]
        if true_label in labels and pred_label in labels:
            cm[true_label, pred_label] += 1
            
    return cm
# --------------------------------------------------------------------------

def load_model_and_optimizer(net, inner_opt_builder, checkpoint_path):
    # ... (le corps de cette fonction reste inchangé, voir réponse précédente)
    """Charge l'état du modèle et des metaparamètres de l'optimiseur depuis un checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint non trouvé à l'emplacement : {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path)
    
    net.load_state_dict(checkpoint['model_state_dict'])
    print(f"Modèle chargé. Étape : {checkpoint['step']}")
    
    for name, param_tensor in checkpoint['inner_opt_state_dict'].items():
        if name in inner_opt_builder.ext_metaparams:
            inner_opt_builder.ext_metaparams[name].data.copy_(param_tensor.data)
            
    return checkpoint['step']


def evaluate_checkpoint(cfg, checkpoint_path, db, net, inner_opt_builder, n_inner_iter, plot=False):
    """Effectue l'évaluation et affiche la matrice de confusion."""
    n_tasks_to_test = cfg.num_outer_steps
    step_idx = load_model_and_optimizer(net, inner_opt_builder, checkpoint_path)
    
    test_data, _ = db.next(n_tasks_to_test, "test")
    x_spt, y_spt, x_qry, y_qry = test_data
    

    task_num = x_spt.size()[0]

    # Préparation des étiquettes (labels)
    true_labels_logits = y_qry.detach().cpu().numpy().reshape(-1, y_qry.size(-1))
    true_labels = np.argmax(true_labels_logits, axis=1)
    predicted_labels = []
    qry_losses = []
    
    inner_opt = inner_opt_builder.inner_opt

    print(f"Démarrage de l'évaluation avec {task_num} tâches...")

    for i in range(task_num):
        # ... (Adaptation et Évaluation restent inchangées)
        with higher.innerloop_ctx(
            net, inner_opt, track_higher_grads=False, override=inner_opt_builder.overrides,
        ) as (
            fnet,
            diffopt,
        ):
            # 1. Inner loop (Adaptation)
            for _ in range(n_inner_iter):
                spt_pred = fnet(x_spt[i])
                spt_loss = F.mse_loss(spt_pred, y_spt[i])
                diffopt.step(spt_loss)
                
            # 2. Query step (Evaluation)
            qry_pred = fnet(x_qry[i])
            qry_loss = F.mse_loss(qry_pred, y_qry[i])
            qry_losses.append(qry_loss.detach().cpu().numpy())
            
            # Calcul des prédictions (argmax du logit)
            task_preds = qry_pred.detach().cpu().numpy()
            predicted_labels.append(np.argmax(task_preds, axis=1))

    predicted_labels = np.concatenate(predicted_labels)
    
    # Tronquer pour assurer la même taille
    min_len = min(len(true_labels), len(predicted_labels))

    
    true_labels = true_labels[:min_len]
    predicted_labels = predicted_labels[:min_len]
    
    # Calcul des métriques
    avg_qry_loss = np.mean(qry_losses)
    accuracy = np.sum(true_labels == predicted_labels) / len(true_labels)
    
    print("\n--- Résultats d'Évaluation ---")
    print(f"Perte Moyenne (MSE) sur Query Set: {avg_qry_loss:.4f}")
    print(f"Précision (Accuracy): {accuracy * 100:.2f}%")
    
    # --- Visualisation de la Matrice de Confusion 🖼️ ---
    # Utilisation de la fonction NumPy personnalisée
    if plot:
        cm = numpy_confusion_matrix(true_labels, predicted_labels, labels=range(10)) 
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt="d", 
            cmap="Blues", 
            xticklabels=range(10), 
            yticklabels=range(10)
        )
        plt.title(f"Matrice de Confusion (Checkpoint Étape {step_idx})")
        plt.ylabel("Vraie Étiquette (True Label)")
        plt.xlabel("Étiquette Prédite (Predicted Label)")
        
        # Sauvegarde de l'image
        cm_path = os.path.join(OUTPUT_PATH, f"confusion_matrix_checkpoint_{step_idx}_no_sklearn.png")
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        plt.savefig(cm_path)
        plt.show()
        plt.close()
        
        print(f"Matrice de confusion enregistrée dans : {cm_path}")
        print("---------------------------------")
    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default="outputs\\checkpoints\\checkpoint_step_5000.pt")
    parser.add_argument("--init_inner_lr", type=float, default=0.1)
    parser.add_argument("--k_spt", type=int, default=5)
    parser.add_argument("--k_qry", type=int, default=10)
    parser.add_argument("--lr_mode", type=str, default="per_layer")
    parser.add_argument("--num_inner_steps", type=int, default=1)
    parser.add_argument("--inner_opt", type=str, default="maml")
    parser.add_argument("--problem", type=str, default="mnist")
    parser.add_argument("--model", type=str, default="conv")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_outer_steps", type=int, default=5000)

    args = parser.parse_args()
    cfg = args
    device = torch.device(cfg.device)

    # Initialisation du DataLoader et du modèle (doit correspondre à l'entraînement)
    db = SyntheticLoader(device, model=cfg.model, problem=cfg.problem, k_spt=cfg.k_spt, k_qry=cfg.k_qry)

    # Définition du modèle
    if cfg.problem == "mnist":
        if cfg.model == "fc":
            net = nn.Sequential(nn.Linear(784, 10, bias=True)).to(device)
        elif cfg.model == "share_fc":
            net = nn.Sequential(layers.ShareLinearFull(784, 10, bias=True, latent_size=50)).to(device)
        elif cfg.model == "conv":
            net = nn.Sequential(nn.Conv2d(1, 32, 3, bias=True), nn.Flatten(), nn.Linear(21632, 10, bias=True)).to(device)
        elif cfg.model == "share_conv":
            net = nn.Sequential(layers.ShareConv2d(1, 32, 3, bias=True), nn.Flatten(), nn.Linear(21632, 10, bias=True)).to(device)
        else:
            raise ValueError(f"Invalid model {cfg.model} for mnist")
    else:
         raise ValueError(f"Evaluation script currently only supports 'mnist'")


    # Initialisation de l'Inner Optimizer Builder (pour les overrides/LRS apprises)
    inner_opt_builder = InnerOptBuilder(
        net, device, cfg.inner_opt, cfg.init_inner_lr, "learned", cfg.lr_mode
    )

    # Lancer l'évaluation et la visualisation
    evaluate_checkpoint(
        cfg,
        cfg.checkpoint_path, 
        db, 
        net, 
        inner_opt_builder, 
        cfg.num_inner_steps
    )


if __name__ == "__main__":
    main()