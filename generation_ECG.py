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
from class_model_ECG import ODEF, ODEAdjoint, NeuralODE, LinearODEF, RandomLinearODEF, TestODEF, NNODEF, ODEVAE, \
    SpiralFunctionExample
from IPython.display import clear_output
import seaborn as sns
sns.color_palette("bright")
import tslearn.datasets
from VAERNN import VAE as VAERNN


def to_np(x):
    return x.detach().cpu().numpy()


def extrapolation(PATH=None,
                  step="",
                  name="",
                  new_gen=False,
                  itr=False,
                  loss=False,
):

    if new_gen:
        with open(PATH+"_variable.txt", "rb") as fp:
                [input, e_nhidden, l_nhidden, d_nhidden, latent, n_epochs] = pickle.load(fp)
        vae = ODEVAE(input, e_nhidden, l_nhidden, d_nhidden, latent)

    else:
        with open(PATH+"_variable.txt", "rb") as fp:
                [input, nhidden, latent, n_epochs] = pickle.load(fp)
        vae = ODEVAE(input, nhidden, nhidden, nhidden, latent)

    X_train, Y_train, X_test, Y_test = tslearn.datasets.UCR_UEA_datasets().load_dataset('ECG5000')
    X_ts = np.linspace(1, 140, num=140)
    X_ts = np.reshape(X_ts, (140, 1))
    X_test = np.transpose(X_test, (1, 0, 2))
    X_test = torch.from_numpy(X_test)
    X_test = X_test[:, :500]
    X_ts = np.tile(X_ts, (500, 1, 1))
    X_ts = np.transpose(X_ts, (1, 0, 2))
    X_ts = torch.from_numpy(X_ts)

    checkpoint = torch.load("/share/home/fpainblanc/virtualenvironment/virtenv/bin/" + PATH + step +".pth")
    vae.load_state_dict(checkpoint["ode_trained"])

    frm, to, to_seed = 0, 140, 140

    seed_trajs = X_test[frm:to_seed]
    ts = X_ts[frm:to]
    samp_trajs_p = vae.generate_with_seed(seed_trajs, ts)
    samp_trajs_p = np.asarray(samp_trajs_p.detach())
    X_test = X_test.detach().numpy()

    plt.clf()
    fig, axes = plt.subplots(nrows=2, ncols=1)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.plot(X_test[:, i], label="Ground Truth not seen by the model")
        ax.plot(X_test[:50, i], label="Ground Truth seen by the model")
        ax.plot(np.linspace(1, 140, 140), samp_trajs_p[:, i], label="Reconstruction")
        if i == 1:
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.32), ncol=2, fancybox=True, shadow=True)

    if itr:
        plt.savefig(PATH + "itr" + str(step) + ".png")
    else:
        plt.savefig(PATH + ".png")
    if loss:
        plt.savefig(PATH + "loss" + str(step) + ".png")

    clear_output(wait=True)
    plt.clf()
    return plt


extrapolation(PATH="ECG4_lat6_h32", new_gen=True, itr=True)
