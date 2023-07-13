config = {
    "alpha_vantage": {
        "api_key": "XOLA7URKCZHU7C9X",  # Claim your free API key here: https://www.alphavantage.co/support/#api-key
        "output_size": "full",
        "url": "https://www.alphavantage.co",
        "key_adjusted_close": "5. adjusted close"
    },
    "data": {
        "train_test_split_size": 0.7,
        "train_val_split_size": 0.7,
        "smoothing": 2,
    },
    "plots": {
        "show_plots": False,
        "xticks_interval": 90,
        "color_actual_val": "#001f3f",
        "color_actual_test": "#4D1BF3",
        "color_train": "#3D9970",
        "color_val": "#0074D9",
        "color_pred_train": "#3D9970",
        "color_pred_val": "#0074D9",
        "color_pred_test": "#FF4136",
    },
    "model": {
        "magnitude_1": {
            "lstm_num_layers": 4.5,
            "lstm_hidden_layer_size": 64,
            "drop_out": 0.4,
            "output_step": 1,
            "window_size": 14,
            "conv1d_kernel_size": 4,
            "dilation_base": 3
        },
        "LSTM_bench_mark_1": {
            "lstm_num_layer": 1,
            "lstm_hidden_layer_size": 64,
            "drop_out": 0.2,
            "output_step": 1,
            "window_size": 14,
        },
        "GRU_bench_mark_1": {
            "hidden_size": 64,
            "output_step": 1,
            "window_size": 14,
        },
    },
    "training": {
        "magnitude_1":
            {
                "device": "cuda",  # "cuda" or "cpu"
                "batch_size": 64,
                "num_epoch": 1,
                "learning_rate": 0.001,
                "loss": "mse",
                "evaluate": ["mse", "mae"],
                "optimizer": "adam",
                "scheduler_step_size": 50,
                "patient": 200,
                "from": "2021-01-01",
                "to": None,
                "best_model": True,
                "early_stop": True,
                "train_shuffle": True,
                "val_shuffle": True,
                "test_shuffle": True
            },
        "LSTM_bench_mark_1": {
                "device": "cuda",  # "cuda" or "cpu"
                "batch_size": 64,
                "num_epoch": 1000,
                "learning_rate": 0.01,
                "loss": "bce",
                "evaluate": ["bce", "accuracy", "precision", "f1"],
                "optimizer": "adam",
                "scheduler_step_size": 500,
                "patient": 1000,
                "start": "2000-5-01",
                "end": None,
                "best_model": True,
                "early_stop": True,
                "train_shuffle": True,
                "val_shuffle": True,
                "test_shuffle": True,
                "weight_decay": 0.1
        },
        "svm_1": {
            "batch_size": 64,
            "num_epoch": 300,
            "learning_rate": 0.01,
            "loss": "bce",
            "evaluate": ["bce", "accuracy", "precision", "f1"],
            "optimizer": "adam",
            "scheduler_step_size": 50,
            "patient": 1000,
            "start": "2022-07-01",
            "end": "2023-05-01",
            "best_model": True,
            "early_stop": True,
            "train_shuffle": True,
            "val_shuffle": True,
            "test_shuffle": True,
            "weight_decay": 0.5
        }

    },
    "pytorch_timeseries_model_type_dict": {
            1: "movement",
            2: "magnitude",
            3: "assembler",
            4: "lstm",
            5: "gru",
            6: "ensembler",
            7: "pred_price_LSTM",
    },
    "tensorflow_timeseries_model_type_dict": {
        1: "svm",
        2: "random_forest",
        3: "xgboost"
    }
}


