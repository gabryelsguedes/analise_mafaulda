from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler


"""
Aplica balanceamento de dados para relizar teste. 

A função aplica uma subamostragem: Diminui a quantidade de atributos de um rótulo que 
esteja com uma quantidade maior, se comparado ao outro. 
A funçao também aplica uma sobreamostragem: Cria dados sintéticos para o rótulo de menor 
atributo.

O intuito dessa função é verificar a influencia do desbalanceamento dos dados em relação 
a qualidade da saída do modelo.

"""

def balance_data(X, y, method='undersample'):
    if method == 'undersample':
        sampler = RandomUnderSampler(random_state=42)
    else:
        sampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    return X_resampled, y_resampled
