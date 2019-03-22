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
    


    def set_all_weights(self, new_weights):
        cursor = 0

        for layer in self.layers:
            weights = layer.get_weights()
            new_cursor = cursor + len(weights)
            success = layer.set_weights(new_weights[cursor:new_cursor])
            cursor = new_cursor

            if not success:
                raise ValueError("Weights in layer", layer, "not set")

        return True



    
    def get_all_grads(self, data, labels, penalty=0):
        
        preds = self.predict(data, classes=False)
        
        # The final layer's error is special!
        error = preds - np.eye(self.output_size)[:, labels].T
        self.layers[-1].error = error
        all_grads = self.layers[-1].get_grads(data, labels, self.layers[-2])


        for k in range(len(self.layers)-2):
            
            error = self.layers[-k-2].get_error(self.layers[-k-1])
            
            grads = self.layers[-k-2].get_grads(data, labels, self.layers[-k-3])
            
            all_grads = np.concatenate((grads, all_grads))
            
        return all_grads
    
    

    def train(self, data, labels, max_iters=1000, learning_rate=0.1, penalty=0, tolerance=1e-6, report_level=100):
        
        cost = self.get_cost(data, labels, penalty)

        if report_level > 0:
            print("Training in progress...")
            print("")
            print("Iter | Cost")
            print("---- | ----")
            print("0000 |", cost)

        self.cost_history = np.empty(max_iters+1)
        self.cost_history[0] = cost
        

        for k in range(1, max_iters+1):
            
            # Update weights based on gradient of cost function
            grads = self.get_all_grads(data, labels, penalty)
            
            
            current_weights = self.get_all_weights()
            weights = current_weights - learning_rate*grads
            success = self.set_all_weights(weights)
            if not success:
                raise ValueError("New weights were not set successfully :(")

            new_cost = self.get_cost(data, labels, penalty)
            print("NC:", new_cost)
            

            while new_cost > cost: #np.isnan(new_cost) or :

                if report_level > 0:
                    print(" Rate shift!", learning_rate, "->", learning_rate/2)
                learning_rate = learning_rate/2
                
                weights = current_weights - learning_rate*grads
                success = self.set_all_weights(weights)
                if not success:
                    raise ValueError("New weights were not set successfully :(")

                new_cost = self.get_cost(data, labels, penalty)
                print("NC:", new_cost)

            # Grow rate
            learning_rate = learning_rate*1.1


            # Stopping criterion
            if abs(cost - new_cost) < tolerance:
                if report_level > 0:
                    print("Tolerance reached - terminating early!")
                
                self.cost_history = self.cost_history[:k]
                break
                
            else:        
                cost = new_cost
                self.cost_history[k] = cost
            
                if report_level > 0 and k % report_level == 0:
                    print(str(k).zfill(4), "|", cost)
        
        if report_level > 0:
            print("")
            print("Training complete!")
            print("Iterations completed:", k)
            print("Final cost:", self.cost_history[-1])
        
        return True

