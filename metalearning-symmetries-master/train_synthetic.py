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


def train(step_idx, data, net, inner_opt_builder, meta_opt, n_inner_iter,problem):
    """Main meta-training step."""
    x_spt, y_spt, x_qry, y_qry = data
    


    
    




    task_num = x_spt.size()[0]
    querysz = x_qry.size(1)

    inner_opt = inner_opt_builder.inner_opt

    qry_losses = []
    meta_opt.zero_grad()
    for i in range(task_num):
        with higher.innerloop_ctx(
            net,
            inner_opt,
            copy_initial_weights=False,
            override=inner_opt_builder.overrides,
        ) as (
            fnet,
            diffopt,
        ):
            for _ in range(n_inner_iter):
                spt_pred = fnet(x_spt[i])
                spt_loss = F.mse_loss(spt_pred, y_spt[i])
                diffopt.step(spt_loss)
            qry_pred = fnet(x_qry[i])
            qry_loss = F.mse_loss(qry_pred, y_qry[i])
            qry_losses.append(qry_loss.detach().cpu().numpy())
            qry_loss.backward()
    metrics = {"train_loss": np.mean(qry_losses)}
    wandb.log(metrics, step=step_idx)
    all_metaparams = inner_opt_builder.metaparams.values()
    torch.nn.utils.clip_grad_norm_(all_metaparams, max_norm=5.0)
    meta_opt.step()


def test(step_idx, data, net, inner_opt_builder, n_inner_iter,problem):
    """Main meta-training step."""
    x_spt, y_spt, x_qry, y_qry = data
    task_num = x_spt.size()[0]
    querysz = x_qry.size(1)

    inner_opt = inner_opt_builder.inner_opt

    qry_losses = []
    total_acc = 0

    if problem=="mnist":
        class_name = [str(i) for i in range(10)]
    else:
        class_name = ['square', 'ellipse', 'heart']
    all_true_label = []
    all_pred = []   
    for i in range(task_num):
        #print("task ", i)
        with higher.innerloop_ctx(
            net, inner_opt, track_higher_grads=False, override=inner_opt_builder.overrides,
        ) as (
            fnet,
            diffopt,
        ):
            for _ in range(n_inner_iter):
                spt_pred = fnet(x_spt[i])
                spt_loss = F.mse_loss(spt_pred, y_spt[i])
                diffopt.step(spt_loss)
            qry_pred = fnet(x_qry[i])

            num_acc = 0 #accuracy par tache
            for a,b in zip(torch.argmax(qry_pred, dim=1),torch.argmax(y_qry[i], dim=1)):
                if a==b:
                    num_acc+=1
            total_acc += num_acc / len(qry_pred)
            all_true_label.extend(torch.argmax(y_qry[i], dim=1).cpu().numpy())
            all_pred.extend(torch.argmax(qry_pred, dim=1).cpu().numpy())
            #print("pred" , torch.argmax(qry_pred, dim=1)) #valeurs prédites
            #print("true" , torch.argmax(y_qry[i], dim=1)) #valeurs réelles
            qry_loss = F.mse_loss(qry_pred, y_qry[i])
            qry_losses.append(qry_loss.detach().cpu().numpy())

    total_acc = (total_acc / task_num)*100
    print("Accuracy totale: ", total_acc, "%")
    avg_qry_loss = np.mean(qry_losses)
    _low, high = st.t.interval(
        0.95, len(qry_losses) - 1, loc=avg_qry_loss, scale=st.sem(qry_losses)
    )
    test_metrics = {"test_loss": avg_qry_loss, "test_err": high - avg_qry_loss}
    test_metrics_acc = {"test_accuracy": total_acc}
    wandb.log(test_metrics, step=step_idx)
    wandb.log(test_metrics_acc, step=step_idx)
    wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,y_true=all_true_label, preds=all_pred,class_names=class_name)})

    return avg_qry_loss



def save_checkpoint(net, inner_opt_builder, step_idx, output_dir="./outputs/checkpoints"):
    """Save model checkpoint."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    checkpoint = {
        'step': step_idx,
        'model_state_dict': net.state_dict(),
        'inner_opt_state_dict': inner_opt_builder.ext_metaparams,
    }
    path = os.path.join(output_dir, f"checkpoint_step_{step_idx}.pt")
    torch.save(checkpoint, path)
    #print(f"Checkpoint saved to {path}")
    return path




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_inner_lr", type=float, default=0.001)
    parser.add_argument("--outer_lr", type=float, default=0.001)
    parser.add_argument("--k_spt", type=int, default=10)
    parser.add_argument("--k_qry", type=int, default=10)
    parser.add_argument("--lr_mode", type=str, default="per_param")
    parser.add_argument("--num_inner_steps", type=int, default=3)
    parser.add_argument("--num_outer_steps", type=int, default=3000)
    parser.add_argument("--inner_opt", type=str, default="maml")
    parser.add_argument("--outer_opt", type=str, default="Adam")
    parser.add_argument("--problem", type=str, default="dsprite")
    parser.add_argument("--model", type=str, default="fc")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    wandb.init(project="weight_sharing_toy", dir=OUTPUT_PATH)
    args = parser.parse_args()
    wandb.config.update(args)
    cfg = wandb.config
    device = torch.device(cfg.device)
    db = SyntheticLoader(device, model=cfg.model, problem=cfg.problem, k_spt=cfg.k_spt, k_qry=cfg.k_qry)

    # AJOUT: Section pour MNIST
    if cfg.problem == "mnist":
        if cfg.model == "fc":
            net = nn.Sequential(nn.Linear(784, 10, bias=True)).to(device)
        elif cfg.model == "share_fc":
            # Utiliser une couche partagée si disponible
            net = nn.Sequential(layers.ShareLinearFull(784, 10, bias=True, latent_size=50)).to(device)
        elif cfg.model == "conv":
         
            net = nn.Sequential(nn.Conv2d(1, 32, 3, bias=True), nn.Flatten(), nn.Linear(21632, 10, bias=True)).to(device)
        elif cfg.model == "share_conv":
            # Version avec weight sharing
            net = nn.Sequential(layers.ShareConv2d(1, 32, 3, bias=True), nn.Flatten(), nn.Linear(21632, 10, bias=True)).to(device)
        else:
            raise ValueError(f"Invalid model {cfg.model} for mnist")

    # AJOUT: Section pour DSprite
    if cfg.problem == "dsprite":
        if cfg.model == "fc":
            net = nn.Sequential(nn.Linear(4096, 3, bias=True)).to(device)
        elif cfg.model == "share_fc":
            # Utiliser une couche partagée si disponible
            net = nn.Sequential(layers.ShareLinearFull(784, 10, bias=True, latent_size=50)).to(device)
        elif cfg.model == "conv":
         
            net = nn.Sequential(nn.Conv2d(1, 32, 3, bias=True), nn.Flatten(), nn.Linear(21632, 10, bias=True)).to(device)
        elif cfg.model == "share_conv":
            # Version avec weight sharing
            net = nn.Sequential(layers.ShareConv2d(1, 32, 3, bias=True), nn.Flatten(), nn.Linear(21632, 10, bias=True)).to(device)
        else:
            raise ValueError(f"Invalid model {cfg.model} for dsprite")













    inner_opt_builder = InnerOptBuilder(
        net, device, cfg.inner_opt, cfg.init_inner_lr, "learned", cfg.lr_mode
    )
    if cfg.outer_opt == "SGD":
        meta_opt = optim.SGD(inner_opt_builder.metaparams.values(), lr=cfg.outer_lr)
    else:
        meta_opt = optim.Adam(inner_opt_builder.metaparams.values(), lr=cfg.outer_lr)

    
    checkpoint_path = None

    start_time = time.time()
    for step_idx in range(cfg.num_outer_steps):
        data, _filters = db.next(32, "train")
        train(step_idx, data, net, inner_opt_builder, meta_opt, cfg.num_inner_steps,problem=cfg.problem)

        if step_idx == 0 or (step_idx + 1) % 50 == 0:
            test_data, _filters  = db.next(600, "test")
            val_loss = test(
                step_idx,
                test_data,
                net,
                inner_opt_builder,
                cfg.num_inner_steps,
                problem=cfg.problem
            )
            if step_idx > 0:
                steps_p_sec = (step_idx + 1) / (time.time() - start_time)
                wandb.log({"steps_per_sec": steps_p_sec}, step=step_idx)
                print(f"Step: {step_idx}. Steps/sec: {steps_p_sec:.2f}")

            if (step_idx + 1) % 100 == 0 or (step_idx + 1) == cfg.num_outer_steps:
                    checkpoint_path = save_checkpoint(net, inner_opt_builder, step_idx + 1)
                    
    accuracy_tab = []


    from visualize import evaluate_checkpoint

    '''
    dossier_path = "./outputs/checkpoints"    

    for file in os.listdir(dossier_path):
        if file.endswith(".pt"):
            checkpoint_path = os.path.join(dossier_path, file)

            if str(cfg.num_outer_steps) not in checkpoint_path:
                accuracy_tab.append(evaluate_checkpoint(cfg,checkpoint_path, db, net, inner_opt_builder, cfg.num_inner_steps))
            else:
                accuracy_tab.append(evaluate_checkpoint(cfg,checkpoint_path, db, net, inner_opt_builder, cfg.num_inner_steps, plot=True))
    '''
if __name__ == "__main__":
    main()
