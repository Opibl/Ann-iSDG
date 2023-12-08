import pandas as pd
import numpy as np
import nnetwork as nn
import data_param as dpar


def main():      
    x, y = dpar.load_data()
    y = np.squeeze(y)  # Asegurarse de que y sea un vector
    param = dpar.load_config()
    W = dpar.load_ws() 
    for i in range(len(W)):
        W[i] = W[i].T
      

    zv = nn.forward(x, W, param[5])
    
    f_score = []
    matriz_cm = []
    for i in range(y.shape[0]):
        y_prob = zv[-1][i,:].T  # Probabilidades de la clase i
        cm, f = nn.metricas(y[i,:], y_prob) 
        print(f'Confusion Matrix:{i+1}')
        print(cm)
        matriz_cm.append(cm)
        f_score.append(f)

    cm_total = 0 
    for i in range(len(matriz_cm)):
        cm_total += matriz_cm[i]
        print("matriz total:")
        print(cm_total)

    f_score = np.array(f_score)
    dpar.save_metric(f_score,cm_total)
    print('F1 Score: {:.5f}'.format(np.mean(f_score) * 100))

if __name__ == '__main__':   
    main()
