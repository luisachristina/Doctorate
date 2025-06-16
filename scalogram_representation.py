import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.signal import resample
from PIL import Image
import io
import pandas as pd
import os
import cv2

def create_scalogram(signal, fs, wavelet='cmor1.5-1.0'):
    scales = np.arange(1, 120)
  
    coefficients, frequencies = pywt.cwt(signal, scales, wavelet, sampling_period=1/fs)

    # Criar imagem
    fig, ax = plt.subplots(figsize=(2.8, 2.1), dpi=100)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    tms = np.arange(len(signal)) / fs
    ax.imshow(np.abs(coefficients), extent=[tms[0], tms[-1], frequencies[-1], frequencies[0]],
              aspect='auto', cmap='jet', interpolation='bilinear')

    ax.set_yscale('log')
    ax.axis('off')
    ax.set_position([0, 0, 1, 1])

    fig.canvas.draw()
    img_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_data = img_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return img_data, coefficients, frequencies

# CRIAR GRAY SCALOGRAMA

def create_gray_scalogram(signal, fs, wavelet='cmor1.5-1.0'):

    scales = np.arange(1, 120)
    coefficients, frequencies = pywt.cwt(signal, scales, wavelet, sampling_period=1/fs)

    # Cria figura
    fig, ax = plt.subplots(figsize=(2.8, 2.1), dpi=100)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    tms = np.arange(len(signal)) / fs
    ax.imshow(np.abs(coefficients), extent=[tms[0], tms[-1], frequencies[-1], frequencies[0]],
              aspect='auto', cmap='gray', interpolation='bilinear')

    ax.set_yscale('log')
    ax.axis('off')
    ax.set_position([0, 0, 1, 1])

    # Extrai imagem da figura em tons de cinza (1 canal)
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
    
    # Converte direto para grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    plt.close(fig)
    return gray_img, coefficients, frequencies

path_data = '/mnt/sda2/LUISA/DOUTORADO/projeto/IMGS_ADL/'

for id in os.listdir(path_data):
    signal = pd.read_csv(os.path.join(path_data, id))
    signal = signal[~signal["Chunks"].isin(["Start", "Last event", "End"])]

    ID = id.split('_')[1]
    print(f'Processing ID: {ID}')
    fs = 400  # sampling frequency
    K = 60  # window size in seconds
    L = fs * K  # samples per window
    H = int(len(signal) / L)  # number of windows
    P, Q = 210,280  # target image dimensions
    scalograms = np.empty((H, P, Q, 4), dtype=np.uint8)


    for j, nome_sinal in enumerate(['EDA', 'ECG', 'Resp', 'BVP']):
        sinal = signal[nome_sinal].values
        signal[nome_sinal] = sinal
    for i in range(1, H):
        start, end = L * (i - 1), L * i

        for j, nome_sinal in enumerate(['EDA', 'ECG', 'Resp', 'BVP']):
            sinal = signal[nome_sinal].values[start:end]
            gray_img, _, _ = create_gray_scalogram(sinal, fs)
            scalograms[i, :, :, j] = gray_img
            
    # Save the scalograms as numpy array
    np.save(f'/home/luisa/Área de trabalho/DOC/new_scalograms/scalograms_{ID}.npy', scalograms)