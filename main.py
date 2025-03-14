from analise_vibracao.src.data_analysis.process_csv import generate_final_dataframe
from analise_vibracao.src.model_training.preprocessing import preprocess_data
from analise_vibracao.src.model_training.model_training import train_models
from analise_vibracao.src.model_training.utils import balance_data
from sklearn.model_selection import train_test_split
import pandas as pd

# Caminhos para os diretórios de dados
csv_folder_imbalance = r'analise_vibracao\data\imbalance\imbalance' 
csv_folder_normal = r'analise_vibracao\data\normal\normal' 

# Gerar o dataframe final, com rótulos 'bom' e 'desbalanceado'
df = generate_final_dataframe(csv_folder_normal, csv_folder_imbalance)

# Carregamento dos dados
X, y = preprocess_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Balanceamento dos dados
X_under, y_under = balance_data(X_train, y_train, method='undersample')
X_over, y_over = balance_data(X_train, y_train, method='oversample')

# Treinamento dos modelos
results_unbalanced = train_models(X_train, y_train, X_test, y_test)
results_undersampled = train_models(X_under, y_under, X_test, y_test)
results_oversampled = train_models(X_over, y_over, X_test, y_test)

# Antes de concatenar, convertendo os resultados para DataFrame
df_unbalanced = pd.DataFrame(results_unbalanced)
df_unbalanced['BalanceType'] = 'Unbalanced'
df_undersampled = pd.DataFrame(results_undersampled)
df_undersampled['BalanceType'] = 'Undersampling'
df_oversampled = pd.DataFrame(results_oversampled)
df_oversampled['BalanceType'] = 'Oversampling'

# Concatenar os resultados
final_results = pd.concat([df_unbalanced, df_undersampled, df_oversampled])

# Salvar o DataFrame final em um arquivo CSV
final_results.to_csv('resultado_modelos.csv', index=True)

print(final_results)
