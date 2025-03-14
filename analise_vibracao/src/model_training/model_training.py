from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

"""
Aplicação de diferentes modelos de aprendizado de máquina para comparar qual apresentará o melhor resultado 
em relação as métricas utilizadas.

A função recebe os dados divididos em treino e teste e realiza um laço para executar o treinamento e teste 
dos modelos.

Os modelos analisados são SVM para modelos de classificação, Regressão logística, XGBoost e uma 
rede neural com duas camadas ocultas.

A função tem como saída os resultados dos testes dos modelos em relação a Acurácia, F1-Score e área
sob a curva ROC.
"""

def train_models(X_train, y_train, X_test, y_test):
    models = {
        'SVM': SVC(probability=True),
        'Logistic Regression': LogisticRegression(),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        'Neural Network': Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.5),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
    }

    results = {}
    
    for name, model in models.items():
        if name == 'Neural Network':
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0, validation_data=(X_test, y_test))
            y_pred = (model.predict(X_test) > 0.5).astype('int32')
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        results[name] = {
            'Test Accuracy': accuracy, 'Test F1-score': f1, 'Test ROC-AUC': roc_auc
        }
    
    return results
