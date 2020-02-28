class NeuralNetwork:
    def __init__(self, seed=1492):
        self.seed = seed;

    def model(self, elements_per_layer, data_shape, residual = False):
        from keras.layers import Dense, Input, Activation, Dropout, BatchNormalization, Add
        from keras.models import Model
        from numpy.random import seed
        from keras import regularizers
        seed(self.seed)
        
        x_in = Input(shape=data_shape);
        for i in range(len(elements_per_layer)-1):
            if i == 0:
                x = Dense(elements_per_layer[i], 
#                          kernel_regularizer=regularizers.l1_l2(0.1)
                          )(x_in);                
                x = BatchNormalization()(x);
#                if residual == True:
#                    x = Add()([x, x_in]);
            else:
                x_inner = Dense(elements_per_layer[i],
#                          kernel_regularizer=regularizers.l1_l2(0.1)
                          )(x);            
                x_inner = BatchNormalization()(x_inner)
                if (residual == True) & (i+1 != len(elements_per_layer)):
                    x = Add()([x, x_inner])                
                    
            
            x = Activation('relu')(x);
            x = Dropout(.25)(x);
        
        x = Dense(elements_per_layer[-1])(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x);
            

        self.model = Model(inputs = x_in, outputs = x);

    def compile(self, opt_name, lr, loss_list):
        if opt_name == 'Adam':
            from keras.optimizers import Adam
            opt = Adam(lr = lr);

        self.model.compile(optimizer = opt, loss= loss_list);

    def fit_model(self, x, y, batch_size, n_epochs, validation_split = 0):
        self.h = self.model.fit(x, y, batch_size = batch_size, epochs = n_epochs, validation_split = validation_split, verbose=0);

    def return_prediction(self, x):
        return self.model.predict(x);

    def plot_history(self):
        import matplotlib.pyplot as plt
        import numpy as np
        plt.figure(figsize=(10,5));
        plt.subplot(211)
        plt.plot(self.h.history['loss'], label='Train loss')
        plt.plot(self.h.history['val_loss'], label= 'Val loss')
        plt.xlabel('N epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(212)
        plt.plot(np.log(self.h.history['loss'][-100:]), label='Train loss')
        plt.plot(np.log(self.h.history['val_loss'][-100:]), label= 'Val loss')
        plt.xlabel('N epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.show()
        
    def plot_results(self, y_test, y_pred, score_function, verbose = False):
        import matplotlib.pyplot as plt
        if verbose == True:
            plt.figure(figsize=(10,5));
            plt.plot(y_pred, label='Prediction')
            plt.plot(y_test, label='Label')
            plt.legend()
            plt.show()

        scoring = score_function(y_test, y_pred);
        print('{}. Score: {}'.format('DNN', scoring))
        return scoring
