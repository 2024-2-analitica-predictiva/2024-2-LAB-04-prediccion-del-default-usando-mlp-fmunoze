# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

def load_data(name):
    dataframe = pd.read_csv(
        f"files/input/{name}",
        index_col=False,
        compression="zip",
    )

    

    return dataframe

#nombre de archivos
name_test="test_data.csv.zip"
name_train="train_data.csv.zip"

#carga de datos train
df_train=load_data(name_train)
#carga de datos test
df_test=load_data(name_test)

df_train['EDUCATION']=df_train['EDUCATION'].replace(0,np.nan)
df_train['MARRIAGE']=df_train['MARRIAGE'].replace(0,np.nan)

df_test['EDUCATION']=df_test['EDUCATION'].replace(0,np.nan)
df_test['MARRIAGE']=df_test['MARRIAGE'].replace(0,np.nan)

df_train.dropna(inplace=True)
df_test.dropna(inplace=True)

#renombrar Columnas
df_train.rename(columns={"default payment next month":"default"},inplace=True)

#renombrar Columnas test
df_test.rename(columns={"default payment next month":"default"},inplace=True)

#quitar ID 
df_train=df_train.drop(columns="ID")
df_test=df_test.drop(columns="ID")


# añadir la categoria others con los valores de eduación mayores a 4 =1 y menores y menores o iguales con valor 0
df_train.loc[df_train["EDUCATION"]>4,"EDUCATION"]=4
df_test["EDUCATION"]=df_test["EDUCATION"].apply(lambda x: 4 if x>4 else x)

#Dividir el dataframe
from sklearn.model_selection import train_test_split
x_train=df_train.drop("default",axis=1)
y_train=df_train["default"]

x_test=df_test.drop("default",axis=1)
y_test=df_test["default"]

categorical_features=['SEX', 'EDUCATION', 'MARRIAGE']
numerical_features= ['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2',
       'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6' ]

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.model_selection import GridSearchCV
from sklearn import pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score,balanced_accuracy_score,recall_score,f1_score
import mlflow
from sklearn.decomposition import PCA

# Preprocesamiento
preprocessor = ColumnTransformer(
    transformers = [
        ("encoder", OneHotEncoder(), categorical_features)
    ],
    remainder = StandardScaler()
)

# Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("pca",PCA(n_components=5)),
    ("select_kbest",SelectKBest(score_func=f_classif,k=5)),
    ("mlp", MLPClassifier())
])


mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

# Create a new MLflow Experiment
mlflow.set_experiment("MLflow lab 4")

def optimize(pipeline,x_train,y_train,x_test,y_test):
    param_grid = {
        "mlp__hidden_layer_sizes":[(600)],
        "mlp__activation":['relu'],
        "mlp__solver":["adam"],
        "mlp__alpha":[0.01231232271],#0.00000000000057873 #0.0123123224 0.0123123227
        "mlp__batch_size":[1050],
        #"mlp__learning_rate":['constant'],
        "mlp__learning_rate_init":[0.0500046],
        #"mlp__power_t":[0.05],
        "mlp__max_iter":[46],
        #"mlp__shuffle":[True],
        "mlp__random_state":[45],
        "mlp__tol":[0.0001],
        #"mlp__momentum":[0.2],
        #"mlp__n_iter_no_change":[10],
        "mlp__early_stopping":[False],
        "select_kbest__k":[12],
        "pca__n_components":[21],
        #'mlp__beta_1':[0.9],
        #'mlp__beta_2':[0.7]
       
    }

    grid_search = GridSearchCV(
    pipeline, param_grid, cv=10, scoring="precision", verbose=2, n_jobs=-1 #"balanced_accuracy"
    )
    
    with mlflow.start_run():

        grid_search.fit(x_train, y_train)

        # Log parameters

        best_params=grid_search.best_params_

        for param,value in best_params.items():
            mlflow.log_param(param,value)


        
        
        #log metrics
        y_train_pred=grid_search.predict(x_train)
        precision=precision_score(y_train, y_train_pred)
        balanced_accuracy= balanced_accuracy_score(y_train, y_train_pred)
        recall= recall_score(y_train, y_train_pred)
        f1_score_mod= f1_score(y_train, y_train_pred)    
        score_Train=grid_search.score(x_train, y_train)

        mlflow.log_metric("precision",precision)
        mlflow.log_metric("balanced_accuracy",balanced_accuracy)
        mlflow.log_metric("recall",recall)
        mlflow.log_metric("f1_score",f1_score_mod)
        mlflow.log_metric("Score_train",score_Train)
        
        # Log test metrics
        y_pred_test = grid_search.predict(x_test)
        precision_test = precision_score(y_test, y_pred_test)
        balanced_acc_test = balanced_accuracy_score(y_test, y_pred_test)
        recall_test = recall_score(y_test, y_pred_test)
        f1_test = f1_score(y_test, y_pred_test)
        score_test=grid_search.score(x_test, y_test)

        mlflow.log_metric("precision_test", precision_test)
        mlflow.log_metric("balanced_accuracy_test", balanced_acc_test)
        mlflow.log_metric("recall_test", recall_test)
        mlflow.log_metric("f1_score_test", f1_test)
        mlflow.log_metric("Score_test",score_test)

        #log_model
        mlflow.sklearn.log_model(grid_search.best_estimator_,"model")

    return grid_search



grid_search=optimize(pipeline,x_train,y_train,x_test,y_test)

best_model = grid_search

import gzip

import os

file_path = "files/models/model.pkl.gz"  # Define el archivo completo con su ruta

def save_estimator(best_model):
    import os
    import pickle
    import gzip

    # Crear los directorios necesarios si no existen
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Guardar el modelo en un archivo comprimido
    with gzip.open(file_path, "wb") as file:
        pickle.dump(best_model, file)

save_estimator(best_model)

# Predicciones
y_train_pred = best_model.predict(x_train)
y_test_pred = best_model.predict(x_test)

def load_estimator():

    import os
    import pickle

    if not os.path.exists("files/models/model.pkl"):
        return None
    with open("files/models/model.pkl", "rb") as file:
        estimator = pickle.load(file)

    return estimator

cm_train=confusion_matrix(y_train, y_train_pred)
cm_test=confusion_matrix(y_test, y_test_pred)

scores=[best_model.score(x_train, y_train),best_model.score(x_train, y_train)]

# Métricas
metrics_train={
        "type":"metrics",
        "dataset":"train",
        "precision":precision_score(y_train, y_train_pred),
        "balanced_accuracy": balanced_accuracy_score(y_train, y_train_pred),
        "recall": recall_score(y_train, y_train_pred),
        "f1_score": f1_score(y_train, y_train_pred)
    }

metrics_test={
        "type":"metrics",
        "dataset":"test",
        "precision": precision_score(y_test, y_test_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_test_pred),
        "recall": recall_score(y_test, y_test_pred),
        "f1_score": f1_score(y_test, y_test_pred)
    }
cm_matrix_train={
    "type": "cm_matrix",
    "dataset": "train",
    "true_0": {
        "predicted_0": int(cm_train[0][0]),  # Verdaderos Negativos
        "predicted_1": int(cm_train[0][1])   # Falsos Positivos
    },
    "true_1": {
        "predicted_0": int(cm_train[1][0]),  # Falsos Negativos
        "predicted_1": int(cm_train[1][1])   # Verdaderos Positivos
    }
}
cm_matrix_test={
    "type": "cm_matrix",
    "dataset": "test",
    "true_0": {
        "predicted_0": int(cm_test[0][0]),  # Verdaderos Negativos
        "predicted_1": int(cm_test[0][1])   # Falsos Positivos
    },
    "true_1": {
        "predicted_0": int(cm_test[1][0]),  # Falsos Negativos
        "predicted_1": int(cm_test[1][1])   # Verdaderos Positivos
    }

}
    
metrics=[metrics_train,metrics_test,cm_matrix_train,cm_matrix_test]
os.makedirs("files/output", exist_ok=True)
pd.DataFrame(metrics).to_json('files/output/metrics.json',orient='records',lines=True)

print(metrics)
