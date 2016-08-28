from __future__ import print_function
import numpy as np
from keras.models import Model
from keras import backend as K
# Orthonorm init code is taked from Lasagne
# https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py
def svd_orthonormal(shape):
    if len(shape) < 2:
        raise RuntimeError("Only shapes of length 2 or more are supported.")
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.standard_normal(flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return q
def get_activations(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output,])
    activations = get_activations([X_batch,0])
    return activations

def LSUVinit(model,batch):
    margin = 0.1
    max_iter = 10
    i=-1
    for layer in model.layers:
        i+=1
        print(layer.get_config()['name'])
        if (('convolution' not in layer.get_config()['name']) and 'clf' not in layer.get_config()['name'] and 'dense' not in layer.get_config()['name']):
            continue
        w_all=layer.get_weights();
        weights = np.array(w_all[0])
        weights = svd_orthonormal(weights.shape)
        biases = np.array(w_all[1])
        w_all_new = [weights,biases]
        layer.set_weights(w_all_new)
        acts1=get_activations(model,i,batch)
        var1=np.var(acts1)
        iter1=0
        needed_variance = 1.0
        print(var1)
        while (abs(needed_variance - var1) > margin):
            w_all=layer.get_weights();
            weights = np.array(w_all[0])
            biases = np.array(w_all[1])
            weights /= np.sqrt(var1)/np.sqrt(needed_variance)
            w_all_new = [weights,biases]
            layer.set_weights(w_all_new)
            acts1=get_activations(model,i,batch)
            var1=np.var(acts1)
            iter1+=1
            print(var1)
            if iter1 > max_iter:
                break
    return model