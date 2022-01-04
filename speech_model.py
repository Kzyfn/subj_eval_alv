

from nnmnkwii.datasets import FileDataSource, FileSourceDataset
from nnmnkwii.datasets import PaddedFileSourceDataset, MemoryCacheDataset  # これはなに？
from nnmnkwii.preprocessing import trim_zeros_frames, remove_zeros_frames
from nnmnkwii.preprocessing import minmax, meanvar, minmax_scale, scale
from nnmnkwii import paramgen
from nnmnkwii.io import hts
from nnmnkwii.frontend import merlin as fe
from nnmnkwii.postfilters import merlin_post_filter

from os.path import join, expanduser, basename, splitext, basename, exists
import os
from glob import glob
import numpy as np
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
import pyworld
import pysptk
import librosa
import librosa.display
#import IPython
#from IPython.display import Audio

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import time
import numpy as np
from tqdm import tnrange, tqdm
#import optuna
import os
import random

import pandas as pd
y_stats = pd.read_csv('static/data/y_stats.csv')


from models import VQVAE

device = "cuda" if torch.cuda.is_available() else "cpu"

mgc_dim = 180  # メルケプストラム次数　？？
lf0_dim = 3  # 対数fo　？？ なんで次元が３？
vuv_dim = 1  # 無声or 有声フラグ　？？
bap_dim = 15  # 発話ごと非周期成分　？？

duration_linguistic_dim = 438  # question_jp.hed で、ラベルに対する言語特徴量をルールベースで記述してる
acoustic_linguisic_dim = 442  # 上のやつ+frame_features とは？？
duration_dim = 1
acoustic_dim = mgc_dim + lf0_dim + vuv_dim + bap_dim  # aoustice modelで求めたいもの

fs = 48000
frame_period = 5
fftlen = pyworld.get_cheaptrick_fft_size(fs)
alpha = pysptk.util.mcepalpha(fs)
hop_length = int(0.001 * frame_period * fs)

mgc_start_idx = 0
lf0_start_idx = 180
vuv_start_idx = 183
bap_start_idx = 184

windows = [
    (0, 0, np.array([1.0])),
    (1, 1, np.array([-0.5, 0.0, 0.5])),
    (1, 1, np.array([1.0, -2.0, 1.0])),
]

use_phone_alignment = True
acoustic_subphone_features = "coarse_coding" if use_phone_alignment else "full"  # とは？

import random, string


def randomname(n):
   randlst = [random.choice(string.ascii_letters + string.digits) for i in range(n)]
   return ''.join(randlst)


def gen_parameters(y_predicted, verbose=True):
    # Number of time frames
    T = y_predicted.shape[0]
    
    # Split acoustic features
    mgc = y_predicted[:,:lf0_start_idx]
    lf0 = y_predicted[:,lf0_start_idx:vuv_start_idx]
    #lf0 = Y['acoustic']['train'][90][:, lf0_start_idx:vuv_start_idx]
    #lf0 = np.zeros(lf0.shape)
    vuv = y_predicted[:,vuv_start_idx]
    bap = y_predicted[:,bap_start_idx:]
    
    # Perform MLPG
    ty = "acoustic"
    mgc_variances = np.tile(y_stats['var'][:lf0_start_idx], (T, 1))#np.tile(np.ones(Y_var[ty][:lf0_start_idx].shape), (T, 1))#
    mgc = paramgen.mlpg(mgc, mgc_variances, windows)
    lf0_variances = np.tile(y_stats['var'][lf0_start_idx:vuv_start_idx], (T,1))#np.tile(np.ones(Y_var[ty][lf0_start_idx:vuv_start_idx].shape), (T,1))#
    lf0 = paramgen.mlpg(lf0, lf0_variances, windows)
    bap_variances = np.tile(y_stats['var'][bap_start_idx:], (T, 1))#np.tile(np.ones(Y_var[ty][bap_start_idx:].shape), (T, 1))#
    bap = paramgen.mlpg(bap, bap_variances, windows)
    
    return mgc, lf0, vuv, bap
def gen_waveform(y_predicted, do_postfilter=True, verbose=True):  
    y_predicted = trim_zeros_frames(y_predicted)
        
    # Generate parameters and split streams
    mgc, lf0, vuv, bap = gen_parameters(y_predicted, verbose=verbose)
    
    if do_postfilter:
        mgc = merlin_post_filter(mgc, alpha)
        
    spectrogram = pysptk.mc2sp(mgc, fftlen=fftlen, alpha=alpha)
    aperiodicity = pyworld.decode_aperiodicity(bap.astype(np.float64), fs, fftlen)
    f0 = lf0.copy()
    f0[vuv < 0.5] = 0
    f0[np.nonzero(f0)] = np.exp(f0[np.nonzero(f0)])
    
    generated_waveform = pyworld.synthesize(f0.flatten().astype(np.float64),
                                            spectrogram.astype(np.float64),
                                            aperiodicity.astype(np.float64),
                                            fs, frame_period)
    return generated_waveform

def class2value(cl, model):
    codebook = np.sort(model.quantized_vectors.weight.detach().cpu().numpy().reshape(-1))
    return codebook[cl]


def synthesize(z, rate=1):

    vqvae = VQVAE(num_layers=2, z_dim=1, num_class=4, input_linguistic_dim = 442+3+2, use_attention=True).to(device)
    vqvae.load_state_dict(torch.load('static/vqvae_model_best.pth', map_location=torch.device(device)))


    data = [np.loadtxt('seqs_for_alv_control_test/linguistic_F.csv'), np.loadtxt('seqs_for_alv_control_test/acoustic_F.csv'), np.loadtxt('seqs_for_alv_control_test/mora_index.csv').reshape(-1),]#水をマレーシアから買わなくてはな
    #水をマレーシアから買わなくてはならないのですのデータ

    z_tf = np.array([class2value(int(cl), vqvae) for cl in z]).reshape(1, -1, 1)





    with torch.no_grad():
        f0_mean, f0_std = np.array(data[1][:, 180].mean()).reshape(1,1), np.array(data[1][:, 180].std()).reshape(1,1)
        indices = np.concatenate([np.arange(285), np.arange(335, 488), np.arange(531, 535)])
        linguistic_f = data[0][:, indices]
        linguistic_f = np.concatenate([linguistic_f, f0_mean.repeat(linguistic_f.shape[0], axis=0), f0_std.repeat(linguistic_f.shape[0], axis=0), 
        np.zeros([linguistic_f.shape[0], 1]), np.ones([linguistic_f.shape[0], 1]), np.zeros([linguistic_f.shape[0], 1])], axis=1)

        linguistic_f = torch.from_numpy(linguistic_f).float().to(device)
        # print(linguistic_f.size()) #torch.Size([344, 447])
        pred_lf0 = vqvae.decode(torch.from_numpy(z_tf).float().to(device), linguistic_f, data[2], tokyo=False).cpu().numpy().reshape(-1)


    y_base = data[1].copy()

    y_base[:, lf0_start_idx] = pred_lf0
    y_base[:, lf0_start_idx+1:lf0_start_idx+3] = 0

    waveform = gen_waveform(y_base)

    filepath = './static/wav/BASIC5000_0001_{}.wav'.format(randomname(10))

    wavfile.write(filepath, rate=int(fs*rate), data=waveform.astype(np.int16))

    return filepath


def recon_f0(data, vqvae_model):
    tmp = [torch.from_numpy(np.concatenate([data[0], np.ones([data[0].shape[0], 1]), np.zeros([data[0].shape[0], 1])], axis=1)).float().to(device), 
          torch.from_numpy(data[1]).float().to(device)]
    recon_y, z, z_uq = vqvae_model(tmp[0], tmp[1], data[2], 0)

    return recon_y.detach().cpu().numpy().reshape(-1)
