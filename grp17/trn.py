# MLP's Training 
import pandas as pd
import numpy as np
import nnetwork as nn
import data_param as dpar




def data_norm(x):
    a = 0.01
    b = 0.99

    max_vals = np.max(x, axis=1, keepdims=True)
    min_vals = np.min(x, axis=1, keepdims=True)

    data_norm = (((x - min_vals) / (max_vals - min_vals)) * (b - a)) + a
    return data_norm



def train_miniBatch(xe, ye, W, V, learning_rate, batch_size, activation_functions, B):
    costo = []  # Inicializa la variable cost aquí
    
    
    for i in range(0, xe.shape[0], batch_size):
       
        X_batch = xe[i:i+batch_size]
        y_batch = ye[i:i+batch_size]
        Act = nn.forward(X_batch, W, activation_functions)
        grad_W,Cost = nn.gradWs(Act, W,y_batch,activation_functions,batch_size)
        W, V = nn.updWs(W, V, grad_W, learning_rate,i,B)
        costo.append(Cost)
        

    return W, V, costo

# MLP's training 
def train_mlp(x, y, param):     
    inshape = x.shape[0]  # Assumes that x is your input data set
    output_node = y.shape[0]
    layer_node = param[3]  # Number of nodes in the hidden layer   
    layer2_node = param[4]
    num_capas = param[2]
    x = data_norm(x)
    W, V = nn.iniWs(inshape, layer_node,layer2_node,num_capas,output_node) 
    costo = []      
    num_capas = param[2]
    batch_size = param[1] 
    activation_functions = param[5]   
    learning_rate = param[6]
    B = np.int16(np.floor(x.shape[1] / batch_size))
    for Iter in range(1, param[0]):  # param[0] epochs
        xe, ye = nn.randpermute(x, y)
        W, V, cost = train_miniBatch(xe, ye, W, V, learning_rate, batch_size, activation_functions, B)
        costo.append(np.mean(cost))  
        if (Iter % 20) == 0:
            print('Iter={} Cost={:.5f}'.format(Iter, costo[-1]))    
       
    return W, costo 

# Beginning ...
def main():
    param = dpar.load_config()        
    x, y = dpar.load_dtrain()   
    W, costo = train_mlp(x, y, param)         
    dpar.save_ws_costo(W, costo)

       
if __name__ == '__main__':   
    main()
