# Analise MAFAULDA - Análise de Vibração e Modelagem

Este projeto realiza a análise de dados e aplicação de diferentes modelos de aprendizado de máquina com o uso da base de dados MAFAULDA(Machinary Fault Database), coletado através do link [MAFAULDA](https://www.kaggle.com/datasets/uysalserkan/fault-induction-motor-dataset/data). Essa base de dados simula diferentes estados de um motor elétrico, para maiores informações acessar esse [link](https://www02.smt.ufrj.br/~offshore/mfs/page_01.html).

No projeto em questão, foram levados em consideração apenas dois estados: Motor em funcionamento normal e motor com desbalanceamento.

A base de dados é composta de 49 amostras do motor em funcionamento normal e 333 amostras do motor com desbalanceamento. O dataset em questão é composto por 8 atributos.

| Coluna           | Descrição                                                          |
| ---------------- | ------------------------------------------------------------------ |
| **Coluna 1**     | Tacômetro                                                          |
| **Coluna 2 a 4** | Sensor de vibração triaxial (Dados de vibração dos eixos X, Y e Z) |
| **Coluna 5 a 7** | Sensor de vibração triaxial (Dados de vibração dos eixos X, Y e Z) |
| **Coluna 8**     | Microfone                                                          |

Para o desenvolvimento do projeto, foram utilizados apenas os sinais de vibração dos dois sensores presentes no dataset. De acordo com a norma ISO 10816, através dos sinais de vibração é possível realizar a identificação de diferentes tipos de falhas, dentre elas o desbalanceamento.

## Estrutura do Projeto

```plaintext
/analise_vibracao
│
├── /data
│   ├── df_tratado.csv        # Arquivo de dados tratado (gerado no código)
│   ├── /normal               # Dados normais
│   └── /imbalance            # Dados desbalanceados
│
├── /src
│   ├── /data_analysis
│   │   ├── init.py
│   │   ├── feature_extraction.py   # Funções de extração de características
│   │   └── process_csv.py          # Funções para processamento de arquivos CSV
│   │
│   ├── /model_training
│   │   ├── init.py
│   │   ├── model_training.py     # Funções para treinamento e avaliação de modelos
│   │   ├── preprocessing.py      # Funções de pré-processamento de dados
│   │   └── utils.py              # Funções auxiliares (como balanceamento)
│   │
│   └── init.py
│
├── main.py                   # Programa principal
└── README.md
```

## Dependências

Este projeto utiliza as seguintes bibliotecas Python:

- pandas - Manipulação de dados
- numpy - Operações numéricas
- scikit-learn - Modelagem e pré-processamento
- imbalanced-learn - Técnicas de balanceamento de dados
- xgboost - Modelo XGBoost
- tensorflow - Redes neurais

Para instalar as dependências, execute, no ambiente de desenvolvimento:

```bash
pip install -r requirements.txt
```

## Etapas de funcionamento do Projeto

1. Preparar os Dados

Faça o download da base de dados no link citado acima. Extraia as pastas normal e imbalance do arquivo baixado e adicione-as nas pasta data, onde cada uma das pastas extraidas conterá os arquivos CSV correspondentes. O formato esperado dos dados é que as colunas 2 a 7 sejam os sinais de vibração dos eixos X, Y e Z.

Se a base for baixada no site do kaggle, nao há necessidade de realizar qualquer modificação, apenas adicionar as pastas conforme mencionado anteriormente.

No script principal, verifique o caminho

Todas as outras etapas são realizadas automaticamente ao executar o script principal.

```bash
python main.py
```

2. Pré-processamento

- Acessar todos os arquivos .csv presentes nas pastas.
- Extrair características do sinal no domínio do tempo e da frequência.
- Rotular as amostras com os status de "Bom" ou "Desbalanceado".
- Criar um novo dataframe com as características extraídas(Mínimo, Máximo, Média, Desvio padrão e RMS) e seus rótulos.
- Normalização dos dados com StandardScaler.
- Codificação dos rótulos usando LabelEncoder.

3. Preparação e execução dos modelos

- São criadas três bases de dados para testar o desempenho dos modelos. Uma base com os dados balanceados criando dados sintéticos. A segunda base removendo dados do rótulo de maior número de amostras e uma terceira base com os dados desbalanceados.
- Realiza o treinamento e teste de diferentes modelos de aprendizado de máquina (SVM, Regressão Logística, XGBoost, Redes Neurais) nos dados balanceados e desbalanceados.
- Exibe os resultados de teste da acurácia, F1-score e ROC-AUC para cada modelo e tipo de dado em relação ao balanceamento.

4. Resultados

O programa gera um arquivo .csv, de nome 'resultado_modelos.csv', e uma tabela de resultados que será impressa no terminal, com a performance de cada modelo (accuracy, F1-score e ROC-AUC) para cada técnica de balanceamento (desbalanceado, undersampling, oversampling).

Saída esperada:

| Metric            | SVM      | Logistic Regression | XGBoost  | Neural Network | BalanceType   |
| ----------------- | -------- | ------------------- | -------- | -------------- | ------------- |
| **Test Accuracy** | 0.987013 | 1.0                 | 0.987013 | 0.987013       | Unbalanced    |
| **Test F1-score** | 0.992000 | 1.0                 | 0.992000 | 0.992000       | Unbalanced    |
| **Test ROC-AUC**  | 0.966667 | 1.0                 | 0.966667 | 0.966667       | Unbalanced    |
| **Test Accuracy** | 0.961039 | 1.0                 | 0.922078 | 0.974026       | Undersampling |
| **Test F1-score** | 0.975610 | 1.0                 | 0.949153 | 0.983607       | Undersampling |
| **Test ROC-AUC**  | 0.950538 | 1.0                 | 0.951613 | 0.983871       | Undersampling |
| **Test Accuracy** | 0.987013 | 1.0                 | 1.000000 | 1.000000       | Oversampling  |
| **Test F1-score** | 0.992000 | 1.0                 | 1.000000 | 1.000000       | Oversampling  |
| **Test ROC-AUC**  | 0.966667 | 1.0                 | 1.000000 | 1.000000       | Oversampling  |

5. Conclusão

Observando os valores das métricas geradas, em relação ao balanceamento dos dados, a aplicação de oversampling apresentou os melhores resultados, porém existe a necessidade de atenção, pois ao gerar dados sintéticos, é possível criar overfiting do modelo de aprendizado, retirando seu poder de generalização.

A remoção de dados da base original se mostrou menos eficiente, se comparada as outras duas bases, trazendo uma redução em todas as métricas analisadas, essa situação pode ter ocorrido pelo fato da classe de menor quantidade de atributos ter apenas 49 amostras, que é um valor consideravelmente baixo quando se quer criar um modelo de aprendizado de máquina.

Observando os resultados gerados com a quantidade de atributos desbalanceados, se comparado aos outros resultados, não teve um valor discrepante que leve a rejeitar o modelo. Em uma base de dados maior, com valores mais discrepantes em relação ao desbalanceamento, fique mais visivel esse viés, afetando o modelo.

Em relação aos modelos aplicados, A regressão logística apresentou o valor de 100% para todos os resultados, o que precisa de maior tempo de estudo para analisar o que causou tal fato. Os outros modelos aplicados apresentaram resultados próximos, denotando que qualquer método teria sua eficiência em relação a base de dados em questão, tendo em vista que podem ser melhorados aplicando testes em seus hiperparâmetros.
