transformer_cf = {
    "model": {
        "symbol": "AMZN",
        "nhead": 3,
        "num_encoder_layers": 20,
        "dim_feedforward": 20,
        "drop_out": 0.5,
        "window_size": 14,
        "output_step": 3,
        "data_mode":2,
        "topk": 10,
        "max_string_length": 500,
        "svm_drop_out_rate": 0.2,
        "rfc_drop_out_rate": 0.2,
        "xgboost_drop_out_rate": 0.2,
        "lstm_drop_out_rate": 0.2,
        "news_drop_out_rate": 0.2,
        "ensembled_model": {
            "svm": 0,
            "random_forest": 1,
            "xgboost": 2, 
            "lstm": 2,
            "news": 0
        },
        
    },
    "training": {
        "device": "cuda",  # "cuda" or "cpu"
        "batch_size": 64,
        "num_epoch": 50,
        "learning_rate": 0.001,
        "loss": "bce",
        "evaluate": ["bce", "accuracy", "precision", "f1"],
        "optimizer": "adam",
        "scheduler_step_size": 100,
        "patient": 600,
        "start": "2022-07-01",
        "end": "2023-05-01",
        "best_model": True,
        "early_stop": True,
        "train_shuffle": True,
        "val_shuffle": True,
        "test_shuffle": True,
        "weight_decay": 0.001,
        'param_grid': {
            "drop_out": [0, 0.2, 0.5],
            "AAPL":{
                3: {
                    'window_size': 7,
                    "ensembled_model": {
                        "random_forest": 1,
                        "svm": 1,
                        "xgboost": 1,
                        "lstm": 0,
                        "news": 2
                    },
                },
                7: {
                    'window_size': 7,
                    "ensembled_model": {
                        "random_forest": 2,
                        "svm": 1,
                        "xgboost": 1, 
                        "lstm": 0,
                        "news": 2
                    },
                },
                14: {
                    'window_size': 14,
                    "ensembled_model": {
                        "random_forest": 0,
                        "svm": 1,
                        "xgboost": 1, 
                        "lstm": 2,
                        "news": 0
                    },
                },
            },
            "AMZN":{
                3: {
                    'window_size': 7,
                    "ensembled_model": {
                        "svm": 2,
                        "random_forest": 1,
                        "xgboost": 2, 
                        "lstm": 2,
                        "news": 2
                    },
                },
                7: {
                    'window_size': 14,
                    "ensembled_model": {
                        "svm": 1,
                        "random_forest": 1,
                        "xgboost": 1, 
                        "lstm": 1,
                        "news": 2
                    },
                },
                14: {
                    'window_size': 7,
                    "ensembled_model": {
                        "svm": 2,
                        "random_forest": 0,
                        "xgboost": 1,
                        "lstm": 2,
                        "news": 2
                    },
                },
            },
            "GOOGL":{
                3: {
                    'window_size': 3,
                    "ensembled_model": {
                        "svm": 0,
                        "random_forest": 1,
                        "xgboost": 2, 
                        "lstm": 2,
                        "news": 0
                    },
                },
                7: {
                    'window_size': 3,
                    "ensembled_model": {
                        "svm": 0,
                        "random_forest": 1,
                        "xgboost": 2, 
                        "lstm": 2,
                        "news": 0
                    },
                },
                14: {
                    'window_size': 3,
                    "ensembled_model": {
                        "svm": 0,
                        "random_forest": 1,
                        "xgboost": 2, 
                        "lstm": 2,
                        "news": 0
                    },
                },
            },
            "TSLA":{
                3: {
                    'window_size': 3,
                    "ensembled_model": {
                        "svm": 0,
                        "random_forest": 1,
                        "xgboost": 2, 
                        "lstm": 2,
                        "news": 0
                    },
                },
                7: {
                    'window_size': 3,
                    "ensembled_model": {
                        "svm": 0,
                        "random_forest": 1,
                        "xgboost": 2, 
                        "lstm": 2,
                        "news": 0
                    },
                },
                14: {
                    'window_size': 3,
                    "ensembled_model": {
                        "svm": 0,
                        "random_forest": 1,
                        "xgboost": 2, 
                        "lstm": 2,
                        "news": 0
                    },
                },
            },
            "MSFT":{
                3: {
                    'window_size': [3],
                    "ensembled_model": {
                        "svm": 0,
                        "random_forest": 1,
                        "xgboost": 2, 
                        "lstm": 2,
                        "news": 0
                    },
                },
                7: {
                    'window_size': [3],
                    "ensembled_model": {
                        "svm": 0,
                        "random_forest": 1,
                        "xgboost": 2, 
                        "lstm": 2,
                        "news": 0
                    },
                },
                14: {
                    'window_size': [3],
                    "ensembled_model": {
                        "svm": 0,
                        "random_forest": 1,
                        "xgboost": 2, 
                        "lstm": 2,
                        "news": 0
                    },
                },
            },
        },
        "dropout_list":{
            "svm": [0, 0.2, 0,5],
            "random_forest":  [0, 0.2, 0,5],
            "xgboost":  [0, 0.2, 0,5], 
            "lstm":  [0, 0.2, 0,5],
            "news":  [0, 0.2, 0,5]
        },
    }
}