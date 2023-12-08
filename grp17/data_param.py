# Data and Parameters

import numpy  as np
import pandas as pd
#
def load_config():
    ruta_archivo = 'config.csv'
    with open(ruta_archivo, 'r') as archivo_csv:
        param = [int(i) if '.' not in i else float(i) for i in archivo_csv if i != '\n']
    return param

# training data load
def load_dtrain():
    x = np.genfromtxt('Data/xtrain.csv', delimiter=',', dtype=float, encoding=None)
    y = np.genfromtxt('Data/ytrain.csv', delimiter=',', dtype=float, encoding=None)
    return x,y


def save_ws_costo(W, costo):

    np.savez('ws.npz', W[0].T, W[1].T)                  
    df = pd.DataFrame(costo)
    df.to_csv('costo_avg.csv',index=False, header = False ) # Save Cost csv

#load pretrained weights
def load_ws():
    ws = np.load('ws.npz')                     # Load AEs weights
    ws = [ws[i] for i in ws.files]            # Save AEs weights in ws
    return ws
#


# training data load
def load_data():
    x = np.genfromtxt('Data/xtest.csv', delimiter=',', dtype=float, encoding=None)
    y = np.genfromtxt('Data/ytest.csv', delimiter=',', dtype=float, encoding=None)
    return x,y





def save_metric(fscores, cm):
    # Guardar el archivo CSV con el F-score
    df_fscores = pd.DataFrame(fscores)
    df_fscores.to_csv('fscores.csv', index=False, header=False)

    # Guardar la matriz de confusión en un archivo CSV
    df_cm = pd.DataFrame(cm)
    df_cm.to_csv('cmatrix.csv', index=False ,header=False)


