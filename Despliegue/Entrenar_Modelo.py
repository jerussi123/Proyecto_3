import pandas as pd

data_completa = pd.read_csv("datos_limpios.csv")

data_encoded = pd.get_dummies(data_completa, columns=['cole_area_ubicacion', 'cole_caracter', 'cole_genero', 'cole_jornada', 'cole_mcpio_ubicacion', 
                                                      'cole_naturaleza', 'estu_genero', 'estu_mcpio_presentacion', 'estu_mcpio_reside',
                                                      'fami_cuartoshogar', 'fami_educacionmadre', 'fami_educacionpadre', 'fami_estratovivienda', 
                                                      'fami_personashogar', 'fami_tieneautomovil', 'fami_tienecomputador', 'fami_tieneinternet', 
                                                      'fami_tienelavadora', 'desemp_ingles'], drop_first=True)
data_encoded = data_encoded.astype(int)

data_x = data_encoded.drop(['periodo', 'punt_ingles', 'punt_matematicas', 'punt_sociales_ciudadanas', 'punt_c_naturales', 'punt_lectura_critica',
                            'punt_global'], axis=1)
data_y = data_encoded['punt_global']

import sklearn
from sklearn.model_selection import train_test_split

x_train_full, x_test, y_train_full, y_test = train_test_split(
    data_x, data_y, test_size=0.2, random_state=42) #Dividir el conjunto de datos en entrenamiento y valaidacion, 80% entrenamiento, 20% validacion

x_train, x_valid, y_train, y_valid = train_test_split(
    x_train_full, y_train_full, test_size=0.2, random_state=42) #Dividir el conjunto en  entrenamiento

# Usaremos argparse para pasarle argumentos a las funciones de entrenamiento
import argparse

parser = argparse.ArgumentParser(description='Entrenamiento de una red feed-forward para el problema de clasificación con datos MNIST en TensorFlow/Keras')
parser.add_argument('--batch_size', '-b', type=int, default=32)
parser.add_argument('--epochs', '-e', type=int, default=40)
parser.add_argument('--learning_rate', '-l', type=float, default=0.001)
parser.add_argument('--num_hidden_units', '-n', type=int, default=128)
parser.add_argument('--num_hidden_layers', '-N', type=int, default=2)
parser.add_argument('--regularizers', '-r', type=float, default=0.01)
parser.add_argument('--dropout', '-d', type=float, default=0.20)
parser.add_argument('--activation', '-a', type=str, default='relu')
parser.add_argument('--same_neurons', '-s', type=int, default=0)
parser.add_argument('--normalization', '-t', type=int, default=0)
args = parser.parse_args([])

import mlflow 
import mlflow.keras
import keras
import tensorflow as tf
import tensorflow.keras as tk
from keras import models
from keras import layers
from keras import regularizers

def get_optimizer():
    """
    :return: Keras optimizer
    """
    optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    return optimizer

args = parser.parse_args(["--batch_size", '256', "--epochs", '30',"--num_hidden_units", '128', "--num_hidden_layers", '1',
                           "--regularizers", '0.2', "--same_neurons", '1'])

if args.activation == 'relu':
    act_func = tf.nn.relu
elif args.activation == 'tanh':
    act_func = tf.nn.tanh
elif args.activation == 'prelu':
    act_func = layers.PReLU()

# definimos la capa de entrada
input_layer = layers.Input(shape=(148,))  # 148 variables de entrada
x = input_layer

neurons = [0] * 4
# Agregamos capas ocultas a la red
for n in range(0, args.num_hidden_layers):
    if args.normalization == 0:
        if args.same_neurons == 0:
            neurons[n] = args.num_hidden_units
            # agregamos una capa densa (completamente conectada)
            x = layers.Dense(neurons[n], activation=act_func,kernel_regularizer=regularizers.l2(args.regularizers))(x)
            # agregamos dropout o normalizacion como método de regularización para aleatoriamente descartar una capa si los gradientes son muy pequeños
            # x = layers.Dropout(args.dropout)(x)
            x = layers.BatchNormalization()(x)
        else: 
            neurons[n] = args.num_hidden_units/(2 ** (n - 1))
            # agregamos una capa densa (completamente conectada)
            x = layers.Dense(neurons[n], activation=act_func,kernel_regularizer=regularizers.l2(args.regularizers))(x)
            # agregamos dropout o normalizacion como método de regularización para aleatoriamente descartar una capa si los gradientes son muy pequeños
            # x = layers.Dropout(args.dropout)(x)
            x = layers.BatchNormalization()(x)
    else:
        if args.same_neurons == 0:
            neurons[n] = args.num_hidden_units
            # agregamos una capa densa (completamente conectada)
            x = layers.Dense(neurons[n], activation=act_func)(x)
            # agregamos dropout o normalizacion como método de regularización para aleatoriamente descartar una capa si los gradientes son muy pequeños
            x = layers.Dropout(args.dropout)(x)
            # x = layers.BatchNormalization()(x)
        else: 
            neurons[n] = args.num_hidden_units/(2 ** (n - 1))
            # agregamos una capa densa (completamente conectada)
            x = layers.Dense(neurons[n], activation=act_func)(x)
            # agregamos dropout o normalizacion como método de regularización para aleatoriamente descartar una capa si los gradientes son muy pequeños
            x = layers.Dropout(args.dropout)(x)
            # x = layers.BatchNormalization()(x)

# capa final con 1 nodo de salida y sin activacion por lo que es de regresion
output_layer = layers.Dense(1)(x)

# Se arma el modelo:
model = keras.Model(input_layer, output_layer)
# https://keras.io/optimizers/
optimizer = get_optimizer()

# compilamos el modelo y definimos la función de pérdida  
# otras funciones de pérdida comunes para problemas de clasificación
# 1. sparse_categorical_crossentropy
# 2. binary_crossentropy
model.compile(optimizer=optimizer,
                loss='mean_absolute_error',
                metrics=['mse'])

# entrenamos el modelo
model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size, validation_data=(x_valid, y_valid))

model.save('modelo_proyecto3.keras')
