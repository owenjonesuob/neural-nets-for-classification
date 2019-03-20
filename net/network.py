import numpy as np

class Network(object):
    
    def __init__(self, layers):
        self.layers = layers
        self.output_size = layers[-1].output_size
    
    
    def show(self):
        k = 0
        for layer in self.layers:
            print("Layer {0}: {1}, {2} -> {3}, {4}".format(
                k, 
                type(layer).__name__,
                layer.input_size,
                layer.output_size,
                layer.activation_name
            ))
            k = k+1
    
    

    def predict(self, data, classes=True):
        outputs = data
        for k in range(len(self.layers)):
            outputs = self.layers[k].process(outputs)
        return np.argmax(outputs, axis=1) if classes else outputs
    
    

    def get_cost(self, data, labels, penalty=0):
        """
        Multiclass logistic cost function
        """
        m = data.shape[0]
        preds = self.predict(data, classes=False)
        
        cost = (-1/m) * (np.log(preds) * np.eye(self.output_size)[:, labels].T +
                         np.log(1-preds) * (1 - np.eye(self.output_size)[:, labels]).T).sum()
        
        # Add regularisation
        cost = cost + (penalty/m) * sum([(layer.weights**2).sum() for layer in self.layers])
        return cost
    
    

    def get_all_weights(self):
        return np.concatenate([layer.get_weights() for layer in self.layers])
    

    
    def get_all_grads(self, data, labels, penalty=0):
        
        preds = self.predict(data, classes=False)
        
        # The "k+1st" error actually gets stored in Input layer - all fine!
        error = preds - np.eye(self.output_size)[:, labels].T
        self.layers[-1].error = error
        all_grads = np.empty(0)

        for k in range(len(self.layers)-2):
            
            error = self.layers[-k-1].get_error(self.layers[-k])
            
            grads = self.layers[-k-1].get_grads(data, labels, self.layers[-k-2].output)
            
            all_grads = np.concatenate((grads.flatten(), all_grads))
            
        return all_grads
    
    
    def update_weights(self, new_weights):
        pass
    