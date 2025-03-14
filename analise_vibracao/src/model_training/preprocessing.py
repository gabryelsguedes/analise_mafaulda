from sklearn.preprocessing import StandardScaler, LabelEncoder

"""
Função para retornar os dados após a aplicação da padronização e a binarização da base trabalhada.

Aplicação da padronizaçao dos dados para evitar vies dos atributos com valores de ordem de grandezas maiores
e que afetem outros atributos, de ordem de grandeza menores.

Aplicação da binarização dos rótulos, transforma os rótulos bom e desbalanceado para 0 e 1, respectivamente.
"""
def preprocess_data(df):
    X = df.drop(columns=['rótulo']) 
    y = df['rótulo']
    
    # Aplicação da padronizaçao dos dados para evitar vies de atributos com valores de ordem de grandezas que afetem outros atributos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Aplicação da binarização dos rótulos
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    return X_scaled, y_encoded
