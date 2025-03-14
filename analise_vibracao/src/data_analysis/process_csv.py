import os
import pandas as pd
from analise_vibracao.src.data_analysis.feature_extraction import time_domain_features, frequency_domain_features

"""
Funções para abrir os arquivos .csv da base de dados e exrair as características a partir dos
dados brutos de vibração. 

Retorna o dataframe concatenado da base de dados do motor em funcionamento normal e desbalanceado
com as caracteristicas extraídas e os rótulos para cada amostra. 
"""

# Função para analisar os dados e gerar a base com o rótulo fornecido
def analyze_data(csv_folder, label):
    results = []
    sampling_rate = 50000  # Taxa de amostragem de 50 kHz
    
    # Verificar se a pasta possui subpastas ou não
    for root, dirs, files in os.walk(csv_folder):  # Usando os.walk() para percorrer subpastas
        for filename in files:
            if filename.endswith('.csv'):
                # Obtendo o caminho completo do arquivo
                file_path = os.path.join(root, filename)
                df = pd.read_csv(file_path, header=None)
                
                # Assume que as colunas de dados estão da 2ª à 7ª posição (colunas 1 a 6)
                signals = df.iloc[:, 1:7].values  
                
                features = {}

                for i, signal in enumerate(signals.T):  # Iterando pelos sinais (colunas)
                    axis_label = f'axis_{i+1}'
                    
                    # Características no domínio do tempo
                    time_features = time_domain_features(signal)
                    for key, value in time_features.items():
                        features[f'{axis_label}_time_{key}'] = value
                    
                    # Características no domínio da frequência
                    freq_features = frequency_domain_features(signal, sampling_rate)
                    for key, value in freq_features.items():
                        features[f'{axis_label}_freq_{key}'] = value
                
                # Adicionar a coluna 'rótulo' com o valor passado como parâmetro
                features['rótulo'] = label
                
                results.append(features)

    return pd.DataFrame(results)


# Função para gerar o dataframe final
def generate_final_dataframe(csv_folder_normal, csv_folder_imbalance):
    
    df_bom = analyze_data(csv_folder_normal, 'bom')
    df_desbalanceado = analyze_data(csv_folder_imbalance, 'desbalanceado')
    
    final_df = pd.concat([df_bom, df_desbalanceado], ignore_index=True)
    
    # Salva o dataframe final no caminho desejado
    final_df.to_csv('analise_vibracao/data/df_tratado.csv', index=False)
    
    return final_df