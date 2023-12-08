# Neural Network: functions

import numpy  as np
import pandas as pd

# Inicialización de pesos y vectores V
def iniWs(inshape, layer1_node, layer2_node,num_capas,output_node):
    
    

    if num_capas > 2:
        W1 = randW(layer1_node, inshape)
        V1 = np.zeros_like(W1)
        W2 = randW(layer2_node, layer1_node)
        V2 = np.zeros_like(W2)
        W_output = randW(output_node, layer2_node)
        V_output = np.zeros_like(W_output)
       
        W = [W1, W2, W_output]
        V = [V1, V2, V_output]
    else:
        W1 = randW(layer1_node, inshape)
        V1 = np.zeros_like(W1)
        W_output = randW(output_node, layer1_node)
        V_output = np.zeros_like(W_output)
        W = [W1, W_output]
        V = [V1, V_output]


    return W, V

# Rand values for W    
def randW(next, prev):
    r = np.sqrt(6 / (next + prev))
    w = np.random.rand(next, prev)
    w = w * 2 * r - r    
    return w

# Random location for data
def randpermute(xe,ye):

    # Obtener una permutación aleatoria de los índices
    indices_permuted = np.random.permutation((xe.shape[1]))
    # Aplicar la permutación a los datos
    xe_permuted = xe[:,indices_permuted]
    # Aplicar la permutación a los datos
    ye_permuted = ye[:,indices_permuted]

    return xe_permuted,ye_permuted


# Activation function
def act_functions(x, act):
    # Default Values
    a_SELU = 1.6732
    lambd = 1.0507

    # Sigmoid
    if act == 1:
        return 1 / (1 + np.exp(-x))

    # Tanh
    if act == 2:
        return np.tanh(x)

    # Relu
    if act == 3:
        return np.maximum(0, x)

    # LRelu
    if act == 4:
        return np.where(x >= 0, x, x * 0.01)

    # SELU
    if act == 5:
        return lambd * np.where(x > 0, x, a_SELU * (np.exp(x) - 1))

    return x


# Derivatives of the activation function
def deriva_act(x, act):
    # Default Values
    a_SELU = 1.6732
    lambd = 1.0507

    # Sigmoid
    if act == 1:
        sigmoid_x = act_functions(x, act)
        return sigmoid_x * (1 - sigmoid_x)
    
    # Tanh
    if act == 2:
        tanh_x = np.tanh(x)
        return 1 - tanh_x**2

    #RELU
    if act == 3:
       return np.where(x > 0, 1, 0)


    # LRelu
    if act == 4:
        return np.where(x > 0, 1, 0.01)

    # SELU
    if act == 5:
        return lambd * np.where(x > 0, 1, a_SELU * np.exp(x))

    return x



# Feed-forward of SNN
def forward(X, W, type_activation):
    """
    Realiza la propagación hacia adelante en una red neuronal.

    Parámetros:
    - X: Datos de entrada
    - W: Lista de matrices de pesos para cada capa
    - type_activation: Tipo de activación para las capas ocultas

    Devuelve:
    - Act: Lista que contiene las activaciones y las salidas de cada capa
    """

    Act = []  # Lista que contendrá las activaciones y salidas de cada capa
    A = X  # Inicializa A con los datos de entrada

    Act.append(A)  # Agrega los datos de entrada a la lista Act

    # Itera sobre las capas de la red
    for i in range(len(W)):
        Z = np.dot(W[i], A)  # Calcula la entrada ponderada

        # Aplica la función de activación correspondiente, excepto en la última capa
        if i == len(W) - 1:
            A = act_functions(Z, 1)  # Utiliza la función sigmoide en la última capa
        else:
            A = act_functions(Z, type_activation)

        Act.append(A)  # Agrega la salida de la capa a la lista Act

    return Act



def gradWs(Act, W, y_true, type_activation, batch_size):
    L = len(Act) - 1  # Número de capas en la red
    M = batch_size  # Tamaño del lote

    # Calcula el costo
    Cost = np.sum(np.sum(np.square(Act[L] - y_true), axis=1) / 2) / M

    gW = []  # Lista que contendrá los gradientes de los pesos

    # Calcula el gradiente para la capa de salida
    delta = (Act[L] - y_true) * deriva_act(Act[L], 1)  # Siempre usa la derivada de sigmoid
    gW_L = np.dot(delta, Act[L - 1].T) / M
    gW.append(gW_L)

    # Calcula los gradientes para las capas ocultas
    for l in reversed(range(1, L)):
        delta = np.dot(W[l].T, delta) * deriva_act(Act[l], type_activation)
        gW_l = np.dot(delta, Act[l - 1].T) / M
        gW.append(gW_l)

    gW.reverse()

    return gW, Cost





def updWs(W, V, grad_W, learning_rate,epoch,total_epochs):
    """
    Actualiza los pesos utilizando Descenso de Gradiente con Momento que varía con el tiempo.

    Parámetros:
    - W: Lista de matrices de pesos para cada capa.
    - V: Lista de vectores de velocidad para cada capa.
    - grad_W: Lista de gradientes de los pesos para cada capa.
    - learning_rate: Tasa de aprendizaje.
    - epoch: Época actual.
    - total_epochs: Número total de épocas.

    Devuelve:
    - W: Lista actualizada de matrices de pesos para cada capa.
    - V: Lista actualizada de vectores de velocidad para cada capa.
    """
    eps = 1e-20
    teta = 1 - (epoch/total_epochs) 
    current_momentum = (0.9 * teta) / (0.1 + (0.9 * teta)) + eps   
    # Actualiza los pesos y la velocidad
    for i in range(len(W)):

        # Calcula el factor de momentum
        W[i] = W[i] + V[i]
        V[i] = current_momentum * V[i] - learning_rate * grad_W[i]

    return W, V




# Función para calcular la matriz de confusión
def confusion_matrix(y_true, y_pred):
    unique_classes = np.unique(np.concatenate((y_true, y_pred)))
    num_classes = len(unique_classes)
    confusion_mat = np.zeros((num_classes, num_classes), dtype=int)

    for i in range(len(y_true)):
        true_class = np.where(unique_classes == y_true[i])[0][0]
        pred_class = np.where(unique_classes == y_pred[i])[0][0]
        confusion_mat[true_class, pred_class] += 1

    return confusion_mat

# Función para calcular el puntaje F1
def f1_score_custom(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    precision = np.diag(cm) / np.sum(cm, axis=0)
    recall = np.diag(cm) / np.sum(cm, axis=1)
    
    # Evitar divisiones por cero
    precision = np.nan_to_num(precision, nan=0.0)
    recall = np.nan_to_num(recall, nan=0.0)
    
    f1 = 2 * (precision * recall) / (precision + recall)
    weighted_f1 = np.average(f1, weights=np.sum(cm, axis=1) / np.sum(cm))
    
    return weighted_f1

def metricas(y_true, y_prob):
    y_pred = (y_prob > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    f1 = f1_score_custom(y_true, y_pred)
    return cm, f1

