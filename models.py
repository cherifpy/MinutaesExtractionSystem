from keras.models import  Sequential
from keras.layers import Conv2D, Dense, Input, MaxPool2D, Dropout, Activation, Flatten
from keras.activations import relu, softmax


def CreateFirstModel(input_shape=(), individual=[]):

    my_model = Sequential(name="Model")
    my_model.add(Input(shape=input_shape))

    conv_layers = individual[0]
    dense_layers = individual[1]

 
    for i,layer in enumerate(conv_layers):
        my_model.add(Conv2D(layer[0], layer[1], name=f'conv{i}',padding="same")) #Add conv parameters
        if layer[3] == 1: my_model.add(Activation(relu))
        if layer[2] != 0: my_model.add(MaxPool2D((layer[2],layer[2]),padding="same"))

    my_model.add(Flatten())

    for i,layer in enumerate(dense_layers):
        my_model.add(Dense(layer[0]))
        if layer[1] == 1:  my_model.add(Activation(relu))

    my_model.add(Dense(2,activation=softmax))

    return my_model
def CreateSecondModel(input_shape=(),individual=[]):

    my_model = Sequential(name="Model")
    my_model.add(Input(shape=input_shape))

    conv_layers = individual[0]
    dense_layers = individual[1]

    for i,layer in enumerate(conv_layers):
        my_model.add(Conv2D(layer[0], layer[1], name=f'conv{i}',padding="same")) #Add conv parameters
        my_model.add(Dropout(0.25))
        if layer[3] == 1: my_model.add(Activation(relu))
        if layer[2] != 0: my_model.add(MaxPool2D((layer[2],layer[2]),padding="same"))

    my_model.add(Flatten())

    for i,layer in enumerate(dense_layers):
        my_model.add(Dense(layer[0]))
        if layer[1] == 1:  my_model.add(Activation(relu))

    my_model.add(Dense(9,activation=softmax))

    return my_model
