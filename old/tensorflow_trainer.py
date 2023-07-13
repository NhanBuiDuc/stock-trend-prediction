# from trainer import Trainer,run_epoch, check_best_loss, is_early_stop,
# import pandas as pd
# import torch
# from model import Model
# from configs import configs as cf
# from joblib import dump, load
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# import json
# import numpy as np
# import os
# import tensorflow as tf
#
#
# class TensorflowTrainer(Trainer):
#     def __init__(self):
#         super(TensorflowTrainer, self).__init__()
#         self.model_type_dict = cf["tensorflow_timeseries_model_type_dict"]
#
#     def train(self, model_name, new_data=False):
#         training_param = cf["training"][model_name]
#
#         batch_size = training_param["batch_size"]
#         num_epoch = training_param["num_epoch"]
#         learning_rate = training_param["learning_rate"]
#         loss = training_param["loss"]
#         evaluate = training_param["evaluate"]
#         optimizer = training_param["optimizer"]
#         scheduler_step_size = training_param["scheduler_step_size"]
#         patient = training_param["patient"]
#         start = training_param["start"]
#         end = training_param["end"]
#         best_model = training_param["best_model"]
#         early_stop = training_param["early_stop"]
#         train_shuffle = training_param["train_shuffle"]
#         val_shuffle = training_param["val_shuffle"]
#         test_shuffle = training_param["test_shuffle"]
#         weight_decay = training_param["weight_decay"]
#         model_param = cf["model"][model_name]
#
#         for t in self.model_type_dict:
#             if self.model_type_dict[t] or self.model_type_dict[t].upper() in model_name:
#                 model_type = self.model_type_dict[t]
#
#         window_size = model_param["window_size"]
#         output_step = model_param["output_step"]
#         model_full_name = cf["alpha_vantage"]["symbol"] + "_" + model_name
#         train_dataloader, valid_dataloader, test_dataloader, \
#             num_feature, num_data_points, \
#             train_date, valid_date, test_date = prepare_data(model_type, model_full_name, window_size, start, end,
#                                                              new_data,
#                                                              output_step, batch_size, train_shuffle, val_shuffle,
#                                                              test_shuffle)
#
#         model = Model(name=model_name, num_feature=num_feature, model_type=model_type)
#         model.full_name = model_full_name
#
#         X_train = train_dataloader.dataset.x
#         X_valid = valid_dataloader.dataset.x
#         X_test = test_dataloader.dataset.x
#         y_train = train_dataloader.dataset.y
#         y_valid = valid_dataloader.dataset.y
#         y_test = test_dataloader.dataset.y
#
#         if "mse" in loss:
#             loss = tf.keras.losses.MeanSquaredError()
#         elif "mae" in loss:
#             loss = tf.keras.losses.MeanAbsoluteError()
#         elif "bce" in loss:
#             loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
#
#         if "adam" in optimizer:
#             optimizer = "adam"
#         elif "sgd" in optimizer:
#             optimizer = "sgd"
#         reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
#         early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patient)
#         checkpoint_path = "./models/" + model_full_name + ".h5"
#         model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss',
#                                                               save_best_only=True, verbose=1)
#         callbacks_list = []
#         if hasattr(model.structure, 'compile'):
#             model.structure.compile(optimizer=optimizer,
#                                     loss=loss,
#                                     metrics=[evaluate])
#             if early_stop:
#                 callbacks_list.append(early_stop)
#             if best_model:
#                 callbacks_list.append(model_checkpoint)
#
#             model.structure.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size,
#                                 epochs=num_epoch,
#                                 callbacks=callbacks_list)
#         else:
#             model.structure.fit(X_train, y_train)
#             dump(model, "./models/" + model_full_name + '.pkl')
#         return model
#
#     def eval(self, model):
#         model_full_name = model.full_name
#         model_name = model.name
#         train_file_name = f"./csv/train_{model_full_name}.csv"
#         valid_file_name = f"./csv/valid_{model_full_name}.csv"
#         test_file_name = f"./csv/test_{model_full_name}.csv"
#         data_param = cf["data"]
#         training_param = cf["training"][model_name]
#         device = training_param["device"]
#         batch_size = training_param["batch_size"]
#         evaluate = training_param["evaluate"]
#         start = training_param["start"]
#         end = training_param["end"]
#         train_shuffle = training_param["train_shuffle"]
#         val_shuffle = training_param["val_shuffle"]
#         test_shuffle = training_param["test_shuffle"]
#         model_type = model.model_type
#
#         model_param = cf["model"][model_name]
#         window_size = model_param["window_size"]
#         output_step = model_param["output_step"]
#         train_dataloader, valid_dataloader, test_dataloader, train_date, valid_date, test_date = \
#             prepare_eval_data(model_type, model_full_name, train_file_name, valid_file_name, test_file_name, batch_size,
#                               evaluate, start, end, train_shuffle, val_shuffle, test_shuffle, window_size, output_step)
#
#         X_train = train_dataloader.x
#         X_valid = valid_dataloader.x
#         X_text = test_dataloader.x
#         y_train = train_dataloader.y
#         y_valid = valid_dataloader.y
#         y_test = test_dataloader.y
#         loss, evaluate_list = model.structure.evaluate(X_text, y_test)
#         model.structure.to(device)
#         for i in range(0, 3, 1):
#             if i == 0:
#                 dataloader = train_dataloader
#                 print_string = "Train evaluate " + model_full_name
#             if i == 1:
#                 dataloader = valid_dataloader
#                 print_string = "Valid evaluate " + model_full_name
#             elif i == 2:
#                 dataloader = test_dataloader
#                 print_string = "Test evaluate " + model_full_name
#
#             X = dataloader.x
#             y = dataloader.y
#
#             output_list = model.predict(X)
#             if "accuracy" or "precision" or "f1" in evaluate:
#                 classification_output_list = (output_list > 0.5).float()
#
#             if "accuracy" or "precision" or "f1" in evaluate:
#                 # Compute classification report
#                 target_names = ["DOWN", "UP"]  # Add target class names here
#                 report = classification_report(y, classification_output_list, target_names=target_names)
#                 # Create the saving folder if it does not exist
#                 save_folder = "./eval/"
#                 if not os.path.exists(save_folder):
#                     os.makedirs(save_folder)
#
#                 # Open the file in write mode
#                 save_path = os.path.join(save_folder, model_full_name + "_eval")
#                 # Open the file in write mode
#                 with open(save_path, "a") as f:
#                     # Write the classification report to the file
#                     f.write(print_string + "\n" + "Classification report:\n")
#                     f.write(report)
#
#                     # Write the configs dictionary to the file
#                     f.write("\nTraining Config:\n")
#                     f.write(json.dumps(training_param, indent=4))
#                     f.write("\nModel Config:\n")
#                     f.write(json.dumps(model_param, indent=4))
#                     # Compute confusion matrix
#                     cm = confusion_matrix(y, classification_output_list)
#
#                     # Write the confusion matrix to the file
#                     f.write("\nConfusion matrix:\n")
#                     f.write(np.array2string(cm))
#                     f.write("\n")
#                     f.write("-" * 100)
#                     f.write("\n")
#                 # Print a message to confirm that the file was written successfully
#                 print("Results written to " + model_full_name + "_eval.txt")
#
#             # temp_evaluate = np.array(evaluate)
#             #
#             # temp_evaluate = temp_evaluate[temp_evaluate != "accuracy"]
#             # temp_evaluate = temp_evaluate[temp_evaluate != "precision"]
#             # temp_evaluate = temp_evaluate[temp_evaluate != "f1"]
#             # for c in temp_evaluate:
#             #     if "mse" in evaluate:
#             #         criterion = nn.MSELoss()
#             #         c_name = "MSE"
#             #     elif "mae" in evaluate:
#             #         criterion = nn.L1Loss()
#             #         c_name = "MAE"
#             #     elif "bce" in evaluate:
#             #         criterion = nn.BCELoss()
#             #         c_name = "BCE"
#             #
#             #     target_list = target_list.reshape(-1)
#             #     output_list = output_list.reshape(-1)
#             #     loss = criterion(target_list, output_list)
#             #     loss_str = f"{c_name} loss: {loss.item()}"
#             #
#             #     # Create the saving folder if it does not exist
#             #     save_folder = "./eval/"
#             #     if not os.path.exists(save_folder):
#             #         os.makedirs(save_folder)
#             #
#             #     # Open the file in append mode
#             #     save_path = os.path.join(save_folder, model_full_name + "_eval")
#             #     with open(save_path, "a") as f:
#             #         # Write the loss to the file
#             #         f.write(print_string + " " + loss_str + "\n")
#             #         f.write("-" * 100)
#             #         f.write("-")
#             #         f.write("\n")
#             #     # Print a message to confirm that the file was written successfully
#             #     print(f"Loss written to {save_path}.")
