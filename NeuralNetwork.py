class NeuralNetwork:
    def __init__(self, seed=1492):
        self.seed = seed;

    def model(self, elements_per_layer, data_shape):
        from keras.layers import Dense
        from keras.models import Model
        from keras.layers import Input

        x_in = Input(shape=data_shape);
        for i in range(len(elements_per_layer)):
            if i == 0:
                x = Dense(elements_per_layer[i], activation='relu')(x_in);
            else:
                x = Dense(elements_per_layer[i], activation='relu')(x);

        self.model = Model(inputs = x_in, outputs = x);

    def compile(self, opt_name, lr, loss_list):
        if opt_name == 'Adam':
            from keras.optimizers import Adam
            opt = Adam(lr = lr);

        self.model.compile(optimizer = opt_name, loss= loss_list);

    def fit_model(self, x, y, batch_size, n_epochs):
        self.model.fit(x, y, batch_size = batch_size, epochs = n_epochs);

    def return_prediction(self, x):
        return self.model.predict(x);


    def plot_results(self, y_test, y_pred, score_function, verbose = False):
        import matplotlib.pyplot as plt
        if verbose == True:
            plt.figure(figsize=(15,5));
            plt.plot(y_pred, label='Prediction')
            plt.plot(y_test, label='Label')
            plt.legend()

        scoring = score_function(y_test, y_pred);
        print('{}. Score: {}'.format('DNN', scoring))
        return scoring
