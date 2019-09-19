import os
import math
import arff
import numpy as np
import numpy.random as npr
from IPython.display import clear_output
from tqdm import tqdm_notebook as tqdm
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
#matplotlib inline
#import seaborn as sns
#sns.color_palette("bright")
import matplotlib as mpl
import matplotlib.cm as cm
import psutil
import torch
from scipy import stats
import scipy.io.arff
from torch import Tensor
import tslearn.datasets
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from class_model_ECG import ODE, ODEF, ODEAdjoint, NeuralODE, LinearODEF, RandomLinearODEF, TestODEF, NNODEF, ODEVAE, \
    SpiralFunctionExample


use_cuda = torch.cuda.is_available()


def gen_batch(batch_size, n_sample=140, samp_trajs=None, samp_ts=None):
    n_batches = samp_trajs.shape[1] // batch_size
    time_len = samp_trajs.shape[0]
    n_sample = min(n_sample, time_len)
    for i in range(n_batches):
        if n_sample > 0:
            t0_idx = npr.multinomial(1, [1. / (time_len - n_sample)] * (time_len - n_sample))
            t0_idx = np.argmax(t0_idx)
            tM_idx = t0_idx + n_sample
        else:
            t0_idx = 0
            tM_idx = time_len

        frm, to = batch_size*i, batch_size*(i+1)
        yield samp_trajs[t0_idx:tM_idx, frm:to], samp_ts[t0_idx:tM_idx, frm:to]


def ecg_training(input=1, e_nhidden=64, l_nhidden=64, d_nhidden=64, latent=6, n_epochs=1000, lr=0.001, batch_size=100,
                 name="ECG", saving_step=100):

    variable = [input, e_nhidden, l_nhidden, d_nhidden, latent, n_epochs]

    vae = ODEVAE(input, e_nhidden, l_nhidden, d_nhidden, latent)
    vae = vae.cuda()

    optim = torch.optim.Adam(vae.parameters(), betas=(0.9, 0.999), lr=lr)

    with open(name+"_variable.txt", "wb") as fp:
        pickle.dump(variable, fp)

    X_train, Y_train, X_test, Y_test = tslearn.datasets.UCR_UEA_datasets().load_dataset('ECG5000')
    X_t = np.linspace(1, 140, num=140)
    X_ts = np.reshape(X_t, (140, 1))
    memory = []
    losses = []
    X_train = np.transpose(X_train, (1, 0, 2))
    X_train = torch.from_numpy(X_train)
    X_ts = np.tile(X_ts, (500, 1, 1))
    X_ts = np.transpose(X_ts, (1, 0, 2))
    X_ts = torch.from_numpy(X_ts)

    for epoch_idx in range(n_epochs):
        process = psutil.Process(os.getpid())

        batch = gen_batch(batch_size=batch_size, samp_trajs=X_train, samp_ts=X_ts, n_sample=0)
        for x, t in batch:
            optim.zero_grad()

            x, t = x.cuda(), t.cuda()
            max_len = np.random.choice([30, 50, 100])
            permutation = np.random.permutation(t.shape[0])
            np.random.shuffle(permutation)
            permutation = np.sort(permutation[:max_len])
            x, t = x[permutation], t[permutation]
            x_p, z, z_mean, z_log_var = vae(x, t)
            kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean**2 - torch.exp(z_log_var), -1)

            x_p = x_p.double()
            kl_loss = kl_loss.double()
            x = x.double()
            x = x.cuda()
            x_p = x_p.cuda()
            kl_loss = kl_loss.cuda()
            loss = 0.5 * ((x-x_p)**2).sum(-1).sum(0) + kl_loss
            loss = torch.mean(loss)
            loss /= max_len
            loss.backward()
            optim.step()
            losses.append(loss.item())

        print("Epoch "+str(epoch_idx))
        memory.append(process.memory_info().rss)
        print("Memory: "+str(memory[-1]))
        if epoch_idx % saving_step == 0:
            torch.save({"ode_trained": vae.state_dict()},
                       name+str(epoch_idx)+".pth")

            with open(name+"_losses.txt", "wb") as fp:
                    pickle.dump(losses, fp)

            with open(name+"_memory.txt", "wb") as fp:
                    pickle.dump(memory, fp)

    torch.save({"ode_trained": vae.state_dict()},
               name+".pth")

    with open(name+"_losses.txt", "wb") as fp:
        pickle.dump(losses, fp)

    with open(name+"_memory.txt", "wb") as fp:
        pickle.dump(memory, fp)


ecg_training(name="ECG", latent=6, e_nhidden=64, l_nhidden=20, d_nhidden=32,
             n_epochs=30000, saving_step=2000)
