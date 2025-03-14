import numpy as np
from scipy.fft import fft

"""
Funções criadas para extrair características dos sinais tanto no domínio do tempo como no domínio da frequência.
Esse processo foi realizado com o intuito de diminuir a necessidade de trabalhar com um grande número de dados.
Outros dados estatíticos poderiam ser extraidos e utilizados, como curtose, além de entropia, fator de crista
dentre outros. 

Caso o modelo apresente resultados com métricas abaixo de 95%, será realizada uma análise para extrair maiores 
caracteristicas.

"""
def calculate_rms(signal):
    return np.sqrt(np.mean(signal**2))

def time_domain_features(signal):
    return {
        'min': np.min(signal),
        'max': np.max(signal),
        'mean': np.mean(signal),
        'std': np.std(signal),
        'rms': calculate_rms(signal)
    }

def frequency_domain_features(signal, sampling_rate):
    n = len(signal)
    fft_vals = fft(signal)
    fft_freqs = np.fft.fftfreq(n, d=1/sampling_rate)
    positive_freqs = fft_freqs[:n//2]
    positive_fft_vals = np.abs(fft_vals[:n//2])
    
    return {
        'min_freq': np.min(positive_fft_vals),
        'max_freq': np.max(positive_fft_vals),
        'mean_freq': np.mean(positive_fft_vals),
        'std_freq': np.std(positive_fft_vals),
        'rms_freq': calculate_rms(positive_fft_vals)
    }
