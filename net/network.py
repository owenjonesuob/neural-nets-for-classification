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
    
    
    def reset(self):
        for layer in self.layers[1:]:
            layer.__init__(layer.input_size, layer.output_size, layer.activation_name, layer.add_bias)
        return None



    def predict(self, data, classes=True):
        outputs = data
        for layer in self.layers:
            outputs = layer.process(outputs)
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
    
    


    def train(self, train_data, train_labels, val_data=None, val_labels=None, epochs=100, batch_size=128, learning_rate=0.1, penalty=0, early_stopping=100, verbose=True):

        train_cost = self.get_cost(train_data, train_labels, penalty)
        self.train_cost_history = np.empty(epochs+1)
        self.train_cost_history[0] = train_cost

        
        if val_data is not None:
            val_cost = self.get_cost(val_data, val_labels, penalty)
            val_acc = self.get_accuracy(val_data, val_labels)
            self.val_cost_history = np.empty(epochs+1)
            self.val_cost_history[0] = val_cost
            min_val_cost = val_cost
            self.best_epoch = (0, self.get_all_weights(), val_acc)



        if verbose:
            print("Training in progress...")
            print("")

            if val_data is not None:
                print("Epoch | Train cost | Validation cost | Validation accuracy %")
                print("----- | ---------- | --------------- | ---------------------")
                print(" 0000 |", "%.4f" % train_cost, "|", "%.4f" % val_cost, "|", "%.2f" % (val_acc*100))
            else:
                print("Epoch | Cost")
                print("----- | ----")
                print(" 0000 |", "%.4f" % train_cost)


        for epoch in range(1, epochs+1):

            # Split data into batches
            batches = train_data.shape[0]//batch_size
            p = batch_size / train_data.shape[0]
            props = batches*[p]
            if train_data.shape[0] % batch_size != 0:
                props.append(1 - batches*p)                
                batches = batches + 1
            data_sets = make_sets(train_data, train_labels, props)


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
                

                #Limit to at most [20] new attempts with smaller rates
                for _ in range(20): 
                    
                    if new_batch_cost > batch_cost:

                        learning_rate = learning_rate/2

                        weights = current_weights - learning_rate*grads
                        self.set_all_weights(weights)

                        new_batch_cost = self.get_cost(X, y, penalty)
                    
                    else:
                        break
                    

                # Grow rate
                learning_rate = learning_rate*1.2


            train_cost = self.get_cost(train_data, train_labels, penalty)
            self.train_cost_history[epoch] = train_cost

            if val_data is not None:
                val_cost = self.get_cost(val_data, val_labels, penalty)
                val_acc = self.get_accuracy(val_data, val_labels)
                self.val_cost_history[epoch] = val_cost
                if val_cost < min_val_cost:
                    min_val_cost = val_cost
                    self.best_epoch = (epoch, self.get_all_weights(), val_acc)
            

            # Early stopping criterion - validation cost starts increasing...
            if (
                early_stopping > 0 and
                val_data is not None and
                epoch > 2*early_stopping and
                (self.val_cost_history[(epoch-early_stopping+1):(epoch+1)].mean() >= 
                    self.val_cost_history[(epoch-(2*early_stopping+1)):(epoch-early_stopping+1)].mean())
            ) or np.isnan(train_cost):

                if verbose:
                    print("Validation cost not decreasing - terminating early!")
                    print("Resetting to 'best' model (epoch " + str(self.best_epoch[0]) + ")")
                    print("")
                    self.train_cost_history = self.train_cost_history[:(epoch+1)]
                    self.val_cost_history = self.val_cost_history[:(epoch+1)]
                    self.set_all_weights(self.best_epoch[1])
                    
                break
                
            else:
                if verbose:
                    if val_data is not None:
                        print(" " + str(epoch).zfill(4) + " | " + "%.4f" % train_cost +
                                " | " + "%.4f" % val_cost + " | " + "%.2f" % (val_acc*100))
                    else:
                        print(" " + str(epoch).zfill(4) + " | " + "%.4f" % train_cost)
        
        
        if verbose:
            print("")
            print("Training complete!")
            print("Epochs completed:", epoch)
            print("Final training cost:", self.train_cost_history[-1])
            print("Final training accuracy:", self.get_accuracy(train_data, train_labels))
            if val_data is not None:
                print("Best validation cost:", min_val_cost)
                print("Best validation accuracy:", self.get_accuracy(val_data, val_labels))

        return True
    


    def cross_validate(self, data, labels, folds=10, val_prop=0.2, val_data=None, val_labels=None, epochs=100, batch_size=128, learning_rate=0.1, penalty=0, early_stopping=100, verbose=False):

        print("Performing {}-fold cross-validation...".format(folds))

        # Split data in preparation for cross-validation...
        data_sets = make_sets(data, labels, [1/folds]*folds)

        accs = np.zeros(folds)

        for fold in range(folds):

            # Isolate validation set
            X_val, y_val = data_sets[2*fold], data_sets[2*fold + 1]
            # Training set is ever-so-slightly trickier!
            X_train = np.vstack([data_set for j, data_set in enumerate(data_sets[::2]) if j != fold])
            y_train = np.concatenate([data_set for j, data_set in enumerate(data_sets[1::2]) if j != fold])

            self.reset()

            success = self.train(X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate, penalty, early_stopping, verbose)

            print("Fold: {0} | Best epoch: {1} | Validation accuracy: {2}%".format(
                fold+1,
                self.best_epoch[0],
                self.best_epoch[2]*100
            ))

            accs[fold] = self.best_epoch[2]
        
        print("Average validation accuracy of {}%".format(round(accs.mean()*100, 2)))
        print("")

        return accs.mean()
    






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

