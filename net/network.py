import numpy as np
from utils import make_sets

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


    def get_accuracy(self, data, labels):
        preds = self.predict(data, classes=True)
        return np.mean(preds == labels)
    
    

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

        return None



    
    def get_all_grads(self, data, labels, penalty=0):
        
        # Get predictions (this carries out a full forward pass)
        preds = self.predict(data, classes=False)
        
        # The final layer's error is special!
        error = preds - np.eye(self.output_size)[:, labels].T
        self.layers[-1].error = error
        all_grads = self.layers[-1].get_grads(data, labels, self.layers[-2])

        
        # Now we work backwards through the layers
        for k in range(len(self.layers)-2):
            
            error = self.layers[-k-2].get_error(self.layers[-k-1])
            
            grads = self.layers[-k-2].get_grads(data, labels, self.layers[-k-3])
            
            all_grads = np.concatenate((grads, all_grads))
            
        return all_grads
    
    


    def train(self, data, labels, epochs=10, batch_size=100, learning_rate=0.1, penalty=0, tolerance=1e-8, verbose=True):

        cost = self.get_cost(data, labels, penalty)

        self.cost_history = np.empty(epochs+1)
        self.cost_history[0] = cost


        print("Training in progress...")
        print("")
        print("Epoch | Cost")
        print("----- | ----")
        print(" 0000 |", cost)


        for epoch in range(1, epochs+1):

            # Split data into batches
            batches = data.shape[0]//batch_size
            p = batch_size / data.shape[0]
            props = batches*[p]
            if data.shape[0] % batch_size != 0:
                props.append(1 - batches*p)                
                batches = batches + 1
            data_sets = make_sets(data, labels, props)


            for batch in range(batches):

                # Isolate batch
                X, y = data_sets[2*batch], data_sets[2*batch + 1]
                batch_cost = self.get_cost(X, y, penalty)
                    
                # Update weights based on gradient of cost function
                grads = self.get_all_grads(X, y, penalty)
                
                
                current_weights = self.get_all_weights()
                weights = current_weights - learning_rate*grads
                self.set_all_weights(weights)

                new_batch_cost = self.get_cost(X, y, penalty)
                

                # Limit to at most [10] new attempts with smaller rates
                for k in range(10): 
                    
                    if new_batch_cost > batch_cost:

                        learning_rate = learning_rate/2

                        weights = current_weights - learning_rate*grads
                        self.set_all_weights(weights)

                        new_batch_cost = self.get_cost(X, y, penalty)
                    
                    else:
                        break
                    

                # Grow rate
                learning_rate = learning_rate*1.1


            cost = self.get_cost(data, labels, penalty)
            self.cost_history[epoch] = cost

            # Stopping criterion - consistently small decrease over 5 epochs
            if all(np.diff(-self.cost_history[max(0, epoch-5):(epoch+1)]) < tolerance):
                if verbose:
                    print("Tolerance reached - terminating early!")
                    self.cost_history = self.cost_history[:(epoch+1)]
                    
                break
                
            else:
                if verbose:
                    print(" " + str(epoch).zfill(4) + " | " + str(cost))
        
        
        if verbose:
            print("")
            print("Training complete!")
            print("Epochs completed:", epoch)
            print("Final cost:", self.cost_history[-1])
            print("Training accuracy:", self.get_accuracy(data, labels))

        return True





# folds=5, val_prop=0.2
# print("Fold " + str(fold+1) + "/" + str(folds))
# # Split data in preparation for cross-validation...
# data_sets = make_sets(data, labels, [1/folds]*folds)


# for fold in range(folds):
    
#     # Isolate validation set
#     X_val, y_val = data_sets[2*fold], data_sets[2*fold + 1]
#     # Training set is ever-so-slightly trickier!
#     X_train = np.vstack([data_set for j, data_set in enumerate(data_sets[::2]) if j != fold])
#     y_train = np.concatenate([data_set for j, data_set in enumerate(data_sets[1::2]) if j != fold])

