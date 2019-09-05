import os
import math
import numpy as np
import numpy.random as npr
from IPython.display import clear_output
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
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from class_model import ODE, ODEF, ODEAdjoint, NeuralODE, LinearODEF, RandomLinearODEF, TestODEF, NNODEF, ODEVAE, \
    SpiralFunctionExample

use_cuda = torch.cuda.is_available()


def gen_batch(batch_size, n_sample=100, samp_trajs=None, samp_ts=None):
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


def log_normal_pdf(x, mean, logvar):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl


def to_np(x):
    return x.detach().cpu().numpy()


def train_NODE(input=2, nhidden=64, latent=6, n_epochs=20000, batch_size=100, PATH=None, noise_std=0.02, n_points=200,
               saving_steps=None, VAE=True, opti_SGD=False, seed=1, itr=False):

    torch.cuda.manual_seed(seed)

    if VAE:
        vae = ODEVAE(input, nhidden, latent)
    else:
        vae = ODE(output_dim=input)

    if use_cuda:
        vae = vae.cuda()

    if opti_SGD:
        optim = torch.optim.SGD(vae.parameters(), lr=0.00001)
    else:
        optim = torch.optim.Adam(vae.parameters(), betas=(0.9, 0.999), lr=0.001)

    if itr:
        checkpoint = torch.load("/share/home/fpainblanc/virtualenvironment/virtenv/bin/" + PATH + "_itr" + str(step_itr)
                                + ".pth")
        vae.load_state_dict(checkpoint["ode_trained_itr" + str(step_itr)])

    n_epochs = n_epochs
    noise_std = noise_std

    #extract the training data
    with open("train_orig_trajs.txt", "rb") as fp:
        orig_trajs = pickle.load(fp)

    with open("train_samp_trajs.txt", "rb") as fp:
        samp_trajs = pickle.load(fp)

    with open("train_samp_ts.txt", "rb") as fp:
        samp_ts = pickle.load(fp)

    with open("test_orig_trajs.txt", "rb") as fp:
        orig_test = pickle.load(fp)

    with open("test_samp_trajs.txt", "rb") as fp:
        samp_test = pickle.load(fp)

    with open("test_samp_ts.txt", "rb") as fp:
        samp_ts_test = pickle.load(fp)

    samp_test = samp_test[:, 500:]
    samp_test_ts = samp_ts[:, 500:]
    samp_test = samp_test.cuda()
    samp_test_ts = samp_test_ts.cuda()

    #Save the setting of the experiment
    variable = [input, nhidden, latent]

    with open(PATH+"_variable.txt", "wb") as fp:
        pickle.dump(variable, fp)

    with open(PATH+"_orig_trajs.txt", "wb") as fp:
        pickle.dump(orig_trajs, fp)

    with open(PATH+"_samp_trajs.txt", "wb") as fp:
        pickle.dump(samp_trajs, fp)

    with open(PATH+"_samp_ts.txt", "wb") as fp:
        pickle.dump(samp_ts, fp)

    losses = []
    kl_losses = []
    logp_losses = []
    memory = []
    mse_tot = []
    loss_epoch = []

    for epoch_idx in range(n_epochs):
        train_iter = gen_batch(batch_size=batch_size, samp_trajs=samp_trajs, samp_ts=samp_ts)
        process = psutil.Process(os.getpid())
        for x, t in train_iter:
            optim.zero_grad()
            if use_cuda:
                x, t = x.cuda(), t.cuda()

            max_len = 50
            permutation = np.random.permutation(t.shape[0])
            np.random.shuffle(permutation)
            permutation = np.sort(permutation[:max_len])

            x, t = x[permutation], t[permutation]

            if VAE:
                x_p, z, z_mean, z_log_var = vae(x, t)
                kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean**2 - torch.exp(z_log_var), -1)
                logp_loss = 0.5 * ((x-x_p)**2).sum(-1).sum(0) / noise_std**2
                logp_loss = torch.mean(logp_loss)
                kl_loss = torch.mean(kl_loss)
                logp_loss /= max_len
                kl_loss /= max_len
                loss = logp_loss + kl_loss

            else:
                x_vae = vae(x[0], t)
                loss = ((x_vae - x)**2).sum(0).sum(-1).sum(-1).sum(-1)

            loss.backward()
            if opti_SGD:
                torch.nn.utils.clip_grad_value_(vae.parameters(), 1)
            optim.step()
            torch.save({"ode_trained_loss_i1": vae.state_dict()},
                       PATH + "_loss_i1.pth")

            if loss > 3e12:
                torch.save({"ode_trained_loss_i": vae.state_dict()},
                           PATH + "_loss_i.pth")

            if len(losses) > 1 and losses[-1] > 3e12:
                torch.save({"ode_trained_loss_i2": vae.state_dict()},
                           PATH + "_loss_i2.pth")
                print("intermediary saving done")
                print(loss)

            losses.append(loss.item())
            kl_losses.append(kl_loss.item())
            logp_losses.append(logp_loss.item())

        print("Epoch "+str(epoch_idx))
        memory.append(process.memory_info().rss)
        print("Memory: " + str(memory[-1]))
        loss_epoch.append(loss.item())

        if epoch_idx % 10 == 0:
            frm, to, to_seed = 0, 200, 200
            seed_trajs = samp_test[frm:to_seed]
            ts = samp_test_ts[frm:to]
            samp_trajs_p = to_np(vae.generate_with_seed(seed_trajs, ts))
            samp_trajs_p = torch.from_numpy(samp_trajs_p)
            samp_trajs_p = samp_trajs_p.cuda()
            mse_train = ((samp_test - samp_trajs_p) ** 2).sum(1).sum(0).sum(-1) / 500
            print(mse_train.item())
            mse_tot.append(mse_train.item())
            with open(PATH + "_mse" + str(epoch_idx) + ".txt", "wb") as fp:
                pickle.dump(mse_tot, fp)

        if epoch_idx % saving_steps == 0:
            torch.save({"ode_trained_itr"+str(epoch_idx): vae.state_dict()},
                        PATH + "_itr" + str(epoch_idx) + ".pth")

            with open(PATH+"_memory"+str(epoch_idx)+".txt", "wb") as fp:
                pickle.dump(memory, fp)

            with open(PATH+"_losses.txt", "wb") as fp:
                pickle.dump(losses, fp)

            with open(PATH+"_kl_losses.txt", "wb") as fp:
                pickle.dump(kl_losses, fp)

            with open(PATH+"_logp_losses.txt", "wb") as fp:
                pickle.dump(logp_losses, fp)

            with open(PATH+"_loss_epoch.txt", "wb") as fp:
                    pickle.dump(loss_epoch, fp)

    torch.save({"ode_trained": vae.state_dict()},
               PATH + ".pth")

    with open(PATH+"_losses.txt", "wb") as fp:
        pickle.dump(losses, fp)

    with open(PATH+"2_kl_losses.txt", "wb") as fp:
        pickle.dump(kl_losses, fp)

    with open(PATH+"2_logp_losses.txt", "wb") as fp:
        pickle.dump(logp_losses, fp)

    with open(PATH+"2_memory.txt", "wb") as fp:
        pickle.dump(memory, fp)

    with open(PATH+"2_logp_losses.txt", "wb") as fp:
        pickle.dump(logp_losses, fp)

    print("Model saved in "+PATH)


train_NODE(PATH="spiral_2_SGD2", n_epochs=10000, saving_steps=2000, batch_size=100, VAE=True, opti_SGD=True, seed=2)
train_NODE(PATH="spiral_3_SGD2", n_epochs=10000, saving_steps=2000, batch_size=100, VAE=True, opti_SGD=True, seed=3)
train_NODE(PATH="spiral_4_SGD2", n_epochs=10000, saving_steps=2000, batch_size=100, VAE=True, opti_SGD=True, seed=4)
train_NODE(PATH="spiral_5_SGD2", n_epochs=10000, saving_steps=2000, batch_size=100, VAE=True, opti_SGD=True, seed=5)

