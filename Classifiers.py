def classifiers(classifier_name, seed, n_iter_search):
    import random

    from sklearn.multiclass import OneVsRestClassifier

    random.seed(seed)
    if classifier_name == 'knn':
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier();
        param_dist = {
            'n_neighbors': [2 ** random.randint(0, 6) for i in range(n_iter_search)],
            'algorithm': [random.sample(['ball_tree', 'kd_tree', 'brute'], 1)[0] for i in range(n_iter_search)],
            'leaf_size': [2 ** random.randint(0, 8) for i in range(n_iter_search)],
            'p': [random.randint(1, 5) for i in range(n_iter_search)]
        }

    if classifier_name == 'SVM_linear':
        from sklearn.svm import SVC
        base_estim = SVC(random_state=seed, kernel='linear');
        clf = OneVsRestClassifier(base_estim);
        param_dist = {
            'estimator__C': [10 ** random.randint(-5, 0) for i in range(n_iter_search)],
            'estimator__gamma': [10 ** random.randint(-5, 2) for i in range(n_iter_search)],
        }

    if classifier_name == 'rforest':
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(random_state=seed);
        param_dist = {
            'n_estimators': [random.randint(100, 200) for i in range(n_iter_search)],
            'max_depth': [random.randint(1, 5) for i in range(n_iter_search - 1)] + [None],
            'min_samples_split': [random.randint(2, 10) for i in range(n_iter_search)],
            'min_samples_leaf': [random.randint(1, 5) for i in range(n_iter_search)],
        }

    if classifier_name == 'gboost':
        from sklearn.ensemble import GradientBoostingClassifier
        # Gradient boost
        clf = GradientBoostingClassifier(random_state=seed)
        param_dist = {
            'n_estimators': [random.randint(1, 100) for i in range(n_iter_search)],
            'max_depth': [random.randint(1, 5) for i in range(n_iter_search - 1)] + [None],
            'min_samples_split': [random.randint(2, 10) for i in range(n_iter_search)],
            'min_samples_leaf': [random.randint(1, 5) for i in range(n_iter_search)],
            'learning_rate': [10 ** random.randint(-5, -1) for i in range(n_iter_search)],
            # 'subsample':[random.randint(1,4)/4 for i in range(n_iter_search)],
            'max_features': [random.randint(1, 4) / 4 for i in range(n_iter_search)],
        }

    if classifier_name == 'SVM':
        from sklearn.svm import SVC
        base_estim = SVC(random_state=seed);
        clf = OneVsRestClassifier(base_estim);
        param_dist = {
            'estimator__C': [10 ** random.randint(-5, 0) for i in range(n_iter_search)],
            'estimator__gamma': [10 ** random.randint(-5, 2) for i in range(n_iter_search)],
        }

    if classifier_name == 'logistic_regression':
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(solver='saga');
        param_dist = {
            'penalty': [random.sample(['l1', 'l2', 'elasticnet'], 1)[0] for i in range(n_iter_search)],
            'C': [random.random() * (10 ** random.randint(0, 4)) for i in range(n_iter_search)],
            'l1_ratio': [random.random() for i in range(n_iter_search)]
        }

    if classifier_name == 'MLP':
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(max_iter=5)
        param_dist = {
            'hidden_layer_sizes': [10 ** random.randint(1, 3) for i in range(n_iter_search)],
            'activation': [random.sample(['identity', 'logistic', 'tanh', 'relu'], 1)[0] for i in range(n_iter_search)],
            'batch_size': [2 ** random.randint(3, 4) for i in range(n_iter_search)]
        }

    if classifier_name == 'adaboost':
        from sklearn.ensemble import AdaBoostClassifier
        clf = AdaBoostClassifier()
        param_dist = {
            'n_estimators': [random.randint(1, 300) for i in range(n_iter_search)],
            'learning_rate': [random.random() * (10 ** random.randint(-2, 2)) for i in range(n_iter_search - 1)] + [
                None],
        }

    if classifier_name == 'decision_tree':
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier()
        param_dist = {
            'criterion': [random.sample(['gini', 'entropy'], 1)[0] for i in range(n_iter_search)],
            'max_depth': [random.randint(1, 5) for i in range(n_iter_search - 1)] + [None],
            'min_samples_split': [random.randint(2, 10) for i in range(n_iter_search)],
            'min_samples_leaf': [random.randint(1, 5) for i in range(n_iter_search)],
        }

    if classifier_name == 'passive_aggressive':
        # Tutorial: https://www.youtube.com/watch?v=TJU8NfDdqNQ
        from sklearn.linear_model import PassiveAggressiveClassifier
        clf = PassiveAggressiveClassifier(max_iter=1000)
        param_dist = {
            'C': [10 ** random.randint(-2, 2) for i in range(n_iter_search)],
            'loss': [random.sample(['hinge', 'squared_hinge'], 1)[0] for i in range(n_iter_search)],
            'fit_intercept': [random.sample([True, False], 1)[0] for i in range(n_iter_search)]
        }

    return clf, param_dist