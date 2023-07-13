svm_cf = {
    "model": {
        "C": 1000,
        "kernel": 'poly',
        "degree": 400,
        "gamma": "scale",
        "coef0": 100,
        "class_weight": {0: 0.5, 1: 0.5},
        "window_size": 14,
        "output_step": 7,
        "data_mode": 1,
        "topk": 10,
        "symbol": "GOOGL",
        "max_string_length": 1000,
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
        "weight_decay": 0.0001,
        "param_grid": {
            'data_mode': [0, 1, 2],
            'window_size': [3, 7, 14],
            'output_size': [3, 7, 14],
            'C': [1e3, 1e4, 1e5],
            'gamma': [0.001, 0.01, 0.1],
            'max_string_length': [500, 1000, 10000, 20000]
        },
    }
}
