import numpy as np


class Layer(object):
    
    def __init__(self, input_size, output_size, activation):
        
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.empty(0)
        
        activations = {
            "none": lambda z: z,
            "relu": lambda z: max(0, z),
            "sigmoid": lambda z: 1/(1 + np.exp(-z)),
            "softmax": lambda z: np.exp(z)/np.exp(z).sum()
        }
        
        activation_derivs = {
            "none": lambda z: 1,
            "relu": lambda z: 1 if z > 0 else 1,
            "sigmoid": lambda z: 1/(1 + np.exp(-z)) * (1 - 1/(1 + np.exp(-z))),
            "softmax": lambda z: z # TODO
        }
        
        self.activation_name = activation
        self.activation_fun = activations.get(activation)
        self.activation_deriv_fun = activation_derivs.get(activation)
    
    

    def get_weights(self):
        return self.weights.flatten()




class Input(Layer):
    
    def __init__(self, output_size, activation="none"):
        super().__init__(output_size, output_size, activation)

    
    def process(self, inputs):
        self.raw_output = inputs
        self.output = self.activation_fun(self.raw_output)
        return self.output



class Dense(Layer):
    
    def __init__(self, input_size, output_size, activation="sigmoid", add_bias=True):
        super().__init__(input_size, output_size, activation)
        self.add_bias = add_bias
        self.weights = np.random.uniform(
            -0.1, 0.1,
            size=(
                self.output_size,
                self.input_size + 1 if self.add_bias else self.input_size
            )
        )
    


    def process(self, inputs):
        m = inputs.shape[0]
        self.raw_output = np.matmul(
            np.hstack((np.ones((m, 1)), inputs)) if self.add_bias else inputs,
            self.weights.T
        )
        self.output = self.activation_fun(self.raw_output)
        return self.output
    
    

    def get_error(self, next_layer):
        # Omit bias weights
        self.error = np.matmul(
            next_layer.error,
            next_layer.weights[:, 1:] if next_layer.add_bias else next_layer.weights
        ) * self.activation_deriv_fun(self.raw_output)
        return self.error
    
    

    def get_grads(self, data, labels, prev_layer, penalty=0):
        m = data.shape[0]
        
        grads = 1/m * np.matmul(
            self.error.T,
            np.hstack((np.ones((m, 1)), prev_layer.output)) if self.add_bias else prev_layer.output
        )

        # Compute regularisation term
        reg = (penalty/m) * self.weights
        if self.add_bias:
            reg[:, 0] = 0
        
        self.grads = grads + reg
        return self.grads
    
    

    def update_weights(self, meh):
        pass