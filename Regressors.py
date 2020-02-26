class Regressors:
    def __init__(self, seed = 1492, CV=True):
        self.seed = seed;
        self.CV = CV;

    def get_regressor(self,regressor_name):
        import random
        random.seed(self.seed)
        # ------------------------------------------------
        if regressor_name == 'LinearRegression':
            from sklearn.linear_model import LinearRegression
            clf = LinearRegression();
            param_grid = {}
        # ------------------------------------------------
        if regressor_name == 'KernelRidge':
            from sklearn.kernel_ridge import KernelRidge
            clf = KernelRidge()
            param_grid = {
                'kernel': ['rbf', 'poly'],
                'coef0': [0, 1.0, 2.0],
                'degree': [1, 2, 3, 4],
                'alpha': [1, .1, .01, .001, .0001],
                'gamma': [.1, .01, .001, .0001, .00001]
            }
        # ------------------------------------------------
        if regressor_name == 'BayesianRidge':
            from sklearn.linear_model import BayesianRidge
            clf = BayesianRidge()
            param_grid = {
                'alpha_1': [1e-10, 1e-9, 1e-8, 1e-7, 1e-06, 1e-5],
                'alpha_2': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                'lambda_1': [1e-4, 1e-3, 1e-2, 1e-1, 1],
                'lambda_2': [1e-9, 1e-8, 1e-7, 1e-06, 1e-5, 1e-4, 1e-3],
            }
        # ------------------------------------------------
        if regressor_name == 'SVR':
            from sklearn.svm import SVR
            clf = SVR()
            param_grid = {
                'kernel': ['rbf', 'poly', 'linear'],
                'C': [.1, .01, .001],  # [1, .1, .01, .001, .0001],
                'degree': [2, 3, 4],
                'coef0': [0, 1.0, 2.0],
                'gamma': [.01, .001, .0001]
            }
        # ------------------------------------------------
        if regressor_name == 'AdaBoost':
            from sklearn.ensemble import AdaBoostRegressor
            clf = AdaBoostRegressor(random_state = self.seed)
            param_grid = {
                'n_estimators': [50, 100, 200, 400],
                'learning_rate': [.1, .01],
                'loss': ['linear', 'square', 'exponential'],
            }
        # ------------------------------------------------
        if regressor_name == 'GradientBoosting':
            from sklearn.ensemble import GradientBoostingRegressor
            clf = GradientBoostingRegressor(random_state = self.seed)
            param_grid = {
                'loss': ['ls', 'lad', 'huber', 'quantile'],
                'n_estimators': [50, 100],
                'max_depth': [ 3, 5] + [None],
                'min_samples_split': [2, 4, 8],
                'min_samples_leaf': [1, 3, 5],
                'learning_rate': [.1, .01, .001, .0001],
                'max_features': [.25, .5],
            }
        # ------------------------------------------------
        if regressor_name == 'KNN':
            from sklearn.neighbors import KNeighborsRegressor
            clf = KNeighborsRegressor();
            param_grid = {
                'n_neighbors': [1,3,5, 10, 20, 30, 40, 50, 60, 70],
                'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                'weights': ['uniform', 'distance'],
            }
        # ------------------------------------------------
        if regressor_name == 'RandomForest':
            from sklearn.ensemble import RandomForestRegressor
            clf = RandomForestRegressor(random_state = self.seed);
            param_grid = {
                'n_estimators': [20, 50, 75, 100, 200, 400],
                'max_depth': [1, 2, 3, 5, None],
                'min_samples_split': [2, 3,  4, 8],
                'min_samples_leaf': [1, 3, 5, 7, 9],
            }
        # ------------------------------------------------
        if regressor_name == 'MLP':
            from sklearn.neural_network import MLPRegressor
            clf = MLPRegressor(random_state = self.seed)
            param_grid = {
                'hidden_layer_sizes': [2, 8, 16, 32, 64, 128],
                'activation': ['identity', 'logistic', 'tanh', 'relu'],
                'batch_size': [16, 32, 64]
            }

        self.clf = clf;
        self.param_grid = param_grid;
        self.regressor_name = regressor_name;

    def get_TimeSeries_CV(self, score, n_splits=5):
        from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
        self.splitter = TimeSeriesSplit(n_splits);
        self.model = GridSearchCV(self.clf,
                     self.param_grid,
                     scoring = score,
                     n_jobs = -1,
                     cv = self.splitter,
                     return_train_score = True,
                     verbose = 0
                     )

    def fit_model(self, x, y):
        self.model.fit(x, y)

    def return_prediction(self, x):
        if self.CV == True:
            return self.model.best_estimator_.predict(x)
        else:
            return self.model.predict(x)

    def plot_results(self, y_test, y_pred, score_function, verbose = False):
        import matplotlib.pyplot as plt
        if verbose == True:
            plt.figure(figsize=(15,5));
            plt.plot(y_pred, label='Prediction')
            plt.plot(y_test, label='Label')
            plt.legend()

        scoring = score_function(y_test, y_pred);
        print('{}. Score: {}'.format(self.regressor_name, scoring))
        return scoring