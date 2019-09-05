import os
import argparse
import logging
import time
import numpy as np
import numpy.random as npr
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import pickle
from scipy.stats import norm
from class_model import ODEF, ODEAdjoint, NeuralODE, LinearODEF, RandomLinearODEF, TestODEF, NNODEF, ODEVAE, \
    SpiralFunctionExample, ODE
from IPython.display import clear_output
import seaborn as sns
sns.color_palette("bright")


def to_np(x):
    return x.detach().cpu().numpy()

def extrapolation(
        PATH=None,
        VAE=True,
        seed=1,
        itr=False,
        step=0,
        loss=False
):
    torch.cuda.manual_seed(seed)
    device = torch.device("cpu")

    #with open("e.txt", "rb") as fp:
     #       [input, nhidden, latent] = pickle.load(fp)
    [input, nhidden, latent] = [2, 64, 6]
    with open("test_orig_trajs.txt", "rb") as fp:
            orig_trajs = pickle.load(fp)

    with open("test_samp_trajs.txt", "rb") as fp:
            samp_trajs = pickle.load(fp)

    with open("test_samp_ts.txt", "rb") as fp:
            samp_ts = pickle.load(fp)

    with open(PATH+"_losses.txt", "rb") as fp:
        losses = pickle.load(fp)

    with open(PATH+"_kl_losses.txt", "rb") as fp:
        kl_losses = pickle.load(fp)

    with open(PATH+"_logp_losses.txt", "rb") as fp:
        logp_losses = pickle.load(fp)

    """with open(PATH+"_memory.txt", "rb") as fp:
        memory = pickle.load(fp)"""

    if VAE:
        vae = ODEVAE(input, nhidden, latent)
    else:
        vae = ODE(output_dim=input)
        checkpoint = torch.load("/share/home/fpainblanc/virtualenvironment/virtenv/bin/" + PATH + ".pth")
    if itr:
        checkpoint = torch.load("/share/home/fpainblanc/virtualenvironment/virtenv/bin/" + PATH + "_itr" + str(step) + ".pth")
        vae.load_state_dict(checkpoint["ode_trained_itr"+str(step)])
    elif loss:
        checkpoint = torch.load("/share/home/fpainblanc/virtualenvironment/virtenv/bin/" + PATH + "_loss_" + str(step) + ".pth")
        vae.load_state_dict(checkpoint["ode_trained_loss_"+str(step)])
    else:
        checkpoint = torch.load("/share/home/fpainblanc/virtualenvironment/virtenv/bin/" + PATH + ".pth")
        vae.load_state_dict(checkpoint["ode_trained"])

    plt.clf()
    plt.plot(losses)
    plt.savefig(PATH+"_loss.png")
    plt.clf()
    plt.plot(kl_losses)
    plt.savefig(PATH+"_kl_loss.png")
    plt.clf()
    plt.plot(logp_losses)
    plt.savefig(PATH+"_logp_loss.png")
    frm, to, to_seed = 0, 200, 50
    seed_trajs = samp_trajs[frm:to_seed]
    ts = samp_ts[frm:to]
    samp_trajs_p = to_np(vae.generate_with_seed(seed_trajs, ts))
    plt.clf()

    fig, axes = plt.subplots(nrows=2, ncols=2)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.scatter(to_np(seed_trajs[:, i, 0]), to_np(seed_trajs[:, i, 1]), c=to_np(ts[frm:to_seed, i, 0]), cmap=cm.plasma, label="Data points")
        ax.plot(to_np(orig_trajs[frm:to, i, 0]), to_np(orig_trajs[frm:to, i, 1]), label="Ground Truth")
        ax.plot(samp_trajs_p[:, i, 0], samp_trajs_p[:, i, 1], label="Reconstruction")
        if i == 3:
            ax.legend(loc='lower left', bbox_to_anchor=(-1.2, -0.35), ncol=3, fancybox=True, shadow=True)

    if itr:
        plt.savefig(PATH + "itr" + str(step) + ".png")
    elif loss:
        plt.savefig(PATH + "loss" + str(step) + ".png")
    else:
        plt.savefig(PATH + ".png")
    clear_output(wait=True)
    print(PATH + str(step) + " has been produced")
    return plt


plot = extrapolation(PATH="spiral_1_SGD2", VAE=True, seed=1, itr=True, step=8000)
