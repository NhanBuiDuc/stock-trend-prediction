rf_cf = {
    "model": {
        'n_estimators': 5,  # Number of trees in the forest
        'criterion': 'gini',  # Splitting criterion (can be 'gini' or 'entropy')
        'max_depth': 7,  # Maximum depth of the tree
        'min_samples_leaf': 2,  # Minimum number of samples required to be at a leaf node  # Whether to use out-of-bag samples to estimate the generalization accuracy
        'random_state': 42,  # Random seed for reproducibility
        "window_size": 7,
        "output_step": 3,
        "data_mode": 1,
        "topk": 20,
        "symbol": "AAPL",
        "max_string_length": 0,
        # 'param_grid': {
        #     'data_mode': [0, 1, 2],
        #     'window_size': [3, 7, 14],
        #     'output_size': [3, 7, 14],
        #     'n_estimators': [5, 10, 100],
        #     'criterion': ['gini', 'entropy'],
        #     'min_samples_leaf': [2, 4, 6, 50],
        #     'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        #     'max_string_length': [500, 1000, 10000, 20000]
        # }
        'param_grid': {
            'data_mode': [0, 1, 2],
            'window_size': [3, 7, 14],
            'output_size': [3, 7, 14],
            'n_estimators': [5, 10, 100],
            'criterion': ['gini', 'entropy'],
            'min_samples_leaf': [2, 4, 6, 50],
            'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'max_string_length': [500, 1000, 10000, 20000]
        }
    },
    "training": {
        "device": "cuda",  # "cuda" or "cpu"
        "batch_size": 64,
        "num_epoch": 300,
        "learning_rate": 0.001,
        "loss": "bce",
        "evaluate": ["bce", "accuracy", "precision", "f1"],
        "optimizer": "adam",
        "scheduler_step_size": 50,
        "patient": 500,
        "start": "2022-07-01",
        "end": "2023-05-01",
        "best_model": True,
        "early_stop": True,
        "train_shuffle": True,
        "val_shuffle": True,
        "test_shuffle": True,
        "weight_decay": 0.0001
    }
}
