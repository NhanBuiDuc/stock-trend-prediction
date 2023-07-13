from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from config import config as cf
import numpy as np
import torch.nn as nn
import torch
from old.model import Diff_1, Assemble_1 ,Movement_1, Movement_3, Movement_7,Movement_14, Magnitude_1,Magnitude_3, Magnitude_7, Magnitude_14
from sklearn.metrics import confusion_matrix


def test_random_forest_classfier(model, X_test, y_test):

    y_pred = model.predict(X_test)

    # calculate the accuracy of the predictions
    accuracy = accuracy_score(y_test, y_pred)
    # Calculate the F1 score of the model's predictions
    f1 = f1_score(y_test, y_pred)

    return accuracy, f1

def evalute_diff_1(dataset_val, features):

    # here we re-initialize dataloader so the data doesn't shuffled, so we can plot the values by date
    # load the saved model weights from a file
    model = Diff_1(
        input_size = len(features),
        window_size = cf["model"]["diff_1"]["window_size"],
        lstm_hidden_layer_size = cf["model"]["diff_1"]["lstm_hidden_layer_size"], 
        lstm_num_layers = cf["model"]["diff_1"]["lstm_num_layers"], 
        output_steps = cf["model"]["diff_1"]["output_steps"]
    )
    model_name = cf["alpha_vantage"]["symbol"] + "_" + "diff_1"
    
    checkpoint = torch.load('./models/' + model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Epoch: ", checkpoint["epoch"], "Valid loss: ", checkpoint["valid_loss"], "Training loss: ", checkpoint["training_loss"])
    model.eval()
    model.to("cuda")

    # create `DataLoader`
    val_dataloader = DataLoader(dataset_val, batch_size=cf["training"]["diff_1"]["batch_size"], shuffle=True, drop_last=True)
    num_data = len(val_dataloader) * val_dataloader.batch_size
    # define optimizer, scheduler and loss function
    criterion = nn.MSELoss()
    criterion2 = nn.L1Loss()
    criterion3 = nn.NLLLoss()
    
    MSE_val_loss = 0
    MAE_val_loss = 0
    RMSE_val_loss = 0
    y_true = []
    y_pred = []
    model.eval()

    for idx, (x, y) in enumerate(val_dataloader):

        batchsize = x.shape[0]

        x = x.to("cuda")
        y = y.to("cuda")
        y_true.append(y)

        out = model(x)
        y_pred.append(out)

        # y = torch.tensor(y).to("cuda")
        # out = torch.tensor(out).to("cuda")
        loss = criterion(out, y)
        loss2 = criterion2(out, y)
        loss3 = torch.sqrt(loss)
        
        MSE_val_loss += loss.detach().item()  / batchsize
        MAE_val_loss += loss2.detach().item()  / batchsize
        RMSE_val_loss += loss3.detach().item()  / batchsize


    print('Diff 1 MSE Valid loss:{:.6f}%, MAE Valid loss:{:.6f}%, RMSE Valid loss:{:.6f}%'
                    .format(MSE_val_loss * 100 / num_data, 
                            MAE_val_loss * 100 / num_data,
                            RMSE_val_loss * 100 / num_data))
    print('Diff MSE Valid loss:{:.6f}, MAE Valid loss:{:.6f}, RMSE Valid loss:{:.6f}'
                    .format(MSE_val_loss, 
                            MAE_val_loss,
                            RMSE_val_loss))
    # print('Valid loss:{:.6f}%'
    #                 .format(val_loss))
    return MSE_val_loss, MAE_val_loss, RMSE_val_loss, y_true, y_pred

def evalute_assembly_regression(dataset_val, features):
    model_name = cf["alpha_vantage"]["symbol"] + "_" + "assemble_1"
    batch_size = cf["training"]["assemble_1"]["batch_size"]
    # here we re-initialize dataloader so the data doesn't shuffled, so we can plot the values by date
    # load the saved model weights from a file
    model = Assemble_1()
    checkpoint = torch.load('./models/' + model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Epoch: ", checkpoint["epoch"], "Valid loss: ", checkpoint["valid_loss"], "Training loss: ", checkpoint["training_loss"])
    model.eval()
    model.to("cuda")
    # create `DataLoader`
    val_dataloader = DataLoader(dataset_val, batch_size=cf["training"]["assemble_1"]["batch_size"], shuffle=True, drop_last=True)
    num_data = len(val_dataloader) * val_dataloader.batch_size
    # define optimizer, scheduler and loss function
    criterion = nn.MSELoss()
    criterion2 = nn.L1Loss()
    criterion3 = nn.NLLLoss()
    
    MSE_val_loss = 0
    MAE_val_loss = 0
    RMSE_val_loss = 0
    y_true = []
    y_pred = []
    model.eval()

    for idx, (x, y) in enumerate(val_dataloader):

        batchsize = x.shape[0]

        x = x.to("cuda")
        y = y.to("cuda")
        y_true.append(y)

        out = model(x)
        y_pred.append(out)

        # y = torch.tensor(y).to("cuda")
        # out = torch.tensor(out).to("cuda")
        loss = criterion(out, y)
        loss2 = criterion2(out, y)
        loss3 = torch.sqrt(loss)
        
        MSE_val_loss += loss.detach().item() / batch_size
        MAE_val_loss += loss2.detach().item()
        RMSE_val_loss += loss3.detach().item()


    print('Assemble MSE Average Valid loss:{:.6f}, MAE Valid loss:{:.6f}, RMSE Valid loss:{:.6f}'
                    .format(MSE_val_loss / num_data, 
                            MAE_val_loss / num_data,
                            RMSE_val_loss / num_data))
    print('Assemble MSE Valid loss:{:.6f}, MAE Valid loss:{:.6f}, RMSE Valid loss:{:.6f}'
                    .format(MSE_val_loss, 
                            MAE_val_loss,
                            RMSE_val_loss))
    return MSE_val_loss, MAE_val_loss, RMSE_val_loss, y_true, y_pred

def evalute_Movement_1(dataset_val, features):
    batch_size = cf["training"]["movement_1"]["batch_size"]
    model_name = cf["alpha_vantage"]["symbol"] +  "_"  + "movement_1"
    # here we re-initialize dataloader so the data doesn't shuffled, so we can plot the values by date
    # load the saved model weights from a file
    model = Movement_1(
        input_size = len(features),
        window_size = cf["model"]["movement_1"]["window_size"],
        lstm_hidden_layer_size = cf["model"]["movement_1"]["lstm_hidden_layer_size"], 
        lstm_num_layers = cf["model"]["movement_1"]["lstm_num_layers"], 
        output_steps = cf["model"]["movement_1"]["output_steps"],
        kernel_size=4,
        dilation_base=3

    )
    checkpoint = torch.load('./models/' + model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Epoch: ", checkpoint["epoch"], "Valid loss: ", checkpoint["valid_loss"], "Training loss: ", checkpoint["training_loss"])
    model.eval()
    model.to("cuda")
    # create `DataLoader`
    val_dataloader = DataLoader(dataset_val, batch_size=cf["training"]["movement_1"]["batch_size"], shuffle=True, drop_last=True)
    num_data = len(val_dataloader) * val_dataloader.batch_size
    # define optimizer, scheduler and loss function
    criterion1 = nn.BCELoss()    
    binary_cross_entropy_val_loss = 0
    accuracy_score = 0
    y_true = np.empty(0)
    y_pred = np.empty(0)
    model.eval()

    for idx, (x, y) in enumerate(val_dataloader):
        batchsize = x.shape[0]

        x = x.to("cuda")
        y = y.to("cuda")


        out = model(x)
        # _, prob_predict = torch.max(out[:, :1], dim=1)
        prob_predict = (out > 0.5).float()
        # _, prob_true = torch.max(y[:, :1], dim=1)
        accuracy_score += torch.sum(prob_predict == y[:, :1]).item()
        y_true = np.append(y_true, [value for value in y[:, :1].cpu().detach().numpy()])
        y_pred = np.append(y_pred, [value for value in prob_predict.cpu().detach().numpy()])
        loss1 = criterion1(out[:, :1], y[:, :1])
        binary_cross_entropy_val_loss += loss1.detach().item()
    print(model_name + ' infer BCE loss:{:.6f}, Accuracy:{:.6f}'
                    .format(binary_cross_entropy_val_loss, accuracy_score/ num_data))

    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    return binary_cross_entropy_val_loss

def evalute_magnitude_1(dataset_val, features):
    batch_size = cf["training"]["magnitude_1"]["batch_size"]
    model_name = cf["alpha_vantage"]["symbol"] +  "_"  + "magnitude_1"
    # here we re-initialize dataloader so the data doesn't shuffled, so we can plot the values by date
    # load the saved model weights from a file
    model = Magnitude_1(
        input_size = len(features),
        window_size = cf["model"]["magnitude_1"]["window_size"],
        lstm_hidden_layer_size = cf["model"]["magnitude_1"]["lstm_hidden_layer_size"], 
        lstm_num_layers = cf["model"]["magnitude_1"]["lstm_num_layers"], 
        output_steps = cf["model"]["magnitude_1"]["output_steps"],
        kernel_size=4,
        dilation_base=3

    )
    checkpoint = torch.load('./models/' + model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Epoch: ", checkpoint["epoch"], "Valid loss: ", checkpoint["valid_loss"], "Training loss: ", checkpoint["training_loss"])
    model.eval()
    model.to("cuda")
    # create `DataLoader`
    val_dataloader = DataLoader(dataset_val, batch_size=cf["training"]["magnitude_1"]["batch_size"], shuffle=True, drop_last=True)
    num_data = len(val_dataloader) * val_dataloader.batch_size
    # define optimizer, scheduler and loss function
    criterion2 = nn.MSELoss()    

    mean_squared_error_val_loss = 0
    model.eval()

    for idx, (x, y) in enumerate(val_dataloader):
        batchsize = x.shape[0]

        x = x.to("cuda")
        y = y.to("cuda")
        out = model(x)
        loss2 = criterion2(out, y)
        mean_squared_error_val_loss += loss2.detach().item()

    print(model_name + ' infer MSE loss:{:.6f}'
                    .format(mean_squared_error_val_loss))

    return mean_squared_error_val_loss

def evalute_Movement_3(dataset_val, features):
    batch_size = cf["training"]["movement_3"]["batch_size"]
    model_name = cf["alpha_vantage"]["symbol"] +  "_"  + "movement_3"
    # here we re-initialize dataloader so the data doesn't shuffled, so we can plot the values by date
    # load the saved model weights from a file
    model = Movement_3(
        input_size = len(features),
        window_size = cf["model"]["movement_3"]["window_size"],
        lstm_hidden_layer_size = cf["model"]["movement_3"]["lstm_hidden_layer_size"], 
        lstm_num_layers = cf["model"]["movement_3"]["lstm_num_layers"], 
        output_steps = cf["model"]["movement_3"]["output_steps"],
        kernel_size=4,
        dilation_base=3

    )
    checkpoint = torch.load('./models/' + model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Epoch: ", checkpoint["epoch"], "Valid loss: ", checkpoint["valid_loss"], "Training loss: ", checkpoint["training_loss"])
    model.eval()
    model.to("cuda")
    # create `DataLoader`
    val_dataloader = DataLoader(dataset_val, batch_size=cf["training"]["movement_3"]["batch_size"], shuffle=True, drop_last=True)
    num_data = len(val_dataloader) * val_dataloader.batch_size
    # define optimizer, scheduler and loss function
    criterion1 = nn.BCELoss()    
    binary_cross_entropy_val_loss = 0
    accuracy_score = 0
    y_true = np.empty(0)
    y_pred = np.empty(0)
    model.eval()

    for idx, (x, y) in enumerate(val_dataloader):
        batchsize = x.shape[0]

        x = x.to("cuda")
        y = y.to("cuda")


        out = model(x)
        # _, prob_predict = torch.max(out[:, :1], dim=1)
        prob_predict = (out > 0.5).float()
        # _, prob_true = torch.max(y[:, :1], dim=1)
        accuracy_score += torch.sum(prob_predict == y[:, :1]).item()
        y_true = np.append(y_true, [value for value in y[:, :1].cpu().detach().numpy()])
        y_pred = np.append(y_pred, [value for value in prob_predict.cpu().detach().numpy()])
        loss1 = criterion1(out[:, :1], y[:, :1])
        binary_cross_entropy_val_loss += loss1.detach().item()
    print(model_name + ' infer BCE loss:{:.6f}, Accuracy:{:.6f}'
                    .format(binary_cross_entropy_val_loss, accuracy_score/num_data))

    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    return binary_cross_entropy_val_loss

def evalute_magnitude_3(dataset_val, features):
    batch_size = cf["training"]["magnitude_3"]["batch_size"]
    model_name = cf["alpha_vantage"]["symbol"] +  "_"  + "magnitude_3"
    # here we re-initialize dataloader so the data doesn't shuffled, so we can plot the values by date
    # load the saved model weights from a file
    model = Magnitude_3(
        input_size = len(features),
        window_size = cf["model"]["magnitude_3"]["window_size"],
        lstm_hidden_layer_size = cf["model"]["magnitude_3"]["lstm_hidden_layer_size"], 
        lstm_num_layers = cf["model"]["magnitude_3"]["lstm_num_layers"], 
        output_steps = cf["model"]["magnitude_3"]["output_steps"],
        kernel_size=4,
        dilation_base=3

    )
    checkpoint = torch.load('./models/' + model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Epoch: ", checkpoint["epoch"], "Valid loss: ", checkpoint["valid_loss"], "Training loss: ", checkpoint["training_loss"])
    model.eval()
    model.to("cuda")
    # create `DataLoader`
    val_dataloader = DataLoader(dataset_val, batch_size=cf["training"]["magnitude_3"]["batch_size"], shuffle=True, drop_last=True)
    num_data = len(val_dataloader) * val_dataloader.batch_size
    # define optimizer, scheduler and loss function
    criterion2 = nn.MSELoss()    

    mean_squared_error_val_loss = 0
    model.eval()

    for idx, (x, y) in enumerate(val_dataloader):
        batchsize = x.shape[0]

        x = x.to("cuda")
        y = y.to("cuda")
        out = model(x)
        loss2 = criterion2(out, y)
        mean_squared_error_val_loss += loss2.detach().item()

    print(model_name + ' infer MSE loss:{:.6f}'
                    .format(mean_squared_error_val_loss))

    return mean_squared_error_val_loss


def evalute_Movement_7(dataset_val, features):
    batch_size = cf["training"]["movement_7"]["batch_size"]
    model_name = cf["alpha_vantage"]["symbol"] +  "_"  + "movement_7"
    # here we re-initialize dataloader so the data doesn't shuffled, so we can plot the values by date
    # load the saved model weights from a file
    model = Movement_7(
        input_size = len(features),
        window_size = cf["model"]["movement_7"]["window_size"],
        lstm_hidden_layer_size = cf["model"]["movement_7"]["lstm_hidden_layer_size"], 
        lstm_num_layers = cf["model"]["movement_7"]["lstm_num_layers"], 
        output_steps = cf["model"]["movement_7"]["output_steps"],
        kernel_size=4,
        dilation_base=3

    )
    checkpoint = torch.load('./models/' + model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Epoch: ", checkpoint["epoch"], "Valid loss: ", checkpoint["valid_loss"], "Training loss: ", checkpoint["training_loss"])
    model.eval()
    model.to("cuda")
    # create `DataLoader`
    val_dataloader = DataLoader(dataset_val, batch_size=cf["training"]["movement_7"]["batch_size"], shuffle=True, drop_last=True)
    num_data = len(val_dataloader) * val_dataloader.batch_size
    # define optimizer, scheduler and loss function
    criterion1 = nn.BCELoss()    
    binary_cross_entropy_val_loss = 0
    accuracy_score = 0
    y_true = np.empty(0)
    y_pred = np.empty(0)
    model.eval()

    for idx, (x, y) in enumerate(val_dataloader):
        batchsize = x.shape[0]

        x = x.to("cuda")
        y = y.to("cuda")


        out = model(x)
        # _, prob_predict = torch.max(out[:, :1], dim=1)
        prob_predict = (out > 0.5).float()
        # _, prob_true = torch.max(y[:, :1], dim=1)
        accuracy_score += torch.sum(prob_predict == y[:, :1]).item()
        y_true = np.append(y_true, [value for value in y[:, :1].cpu().detach().numpy()])
        y_pred = np.append(y_pred, [value for value in prob_predict.cpu().detach().numpy()])
        loss1 = criterion1(out[:, :1], y[:, :1])
        binary_cross_entropy_val_loss += loss1.detach().item()
    print(model_name + ' infer BCE loss:{:.6f}, Accuracy:{:.6f}'
                    .format(binary_cross_entropy_val_loss, accuracy_score/num_data))

    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    return binary_cross_entropy_val_loss

def evalute_magnitude_7(dataset_val, features):
    batch_size = cf["training"]["magnitude_7"]["batch_size"]
    model_name = cf["alpha_vantage"]["symbol"] +  "_"  + "magnitude_7"
    # here we re-initialize dataloader so the data doesn't shuffled, so we can plot the values by date
    # load the saved model weights from a file
    model = Magnitude_7(
        input_size = len(features),
        window_size = cf["model"]["magnitude_7"]["window_size"],
        lstm_hidden_layer_size = cf["model"]["magnitude_7"]["lstm_hidden_layer_size"], 
        lstm_num_layers = cf["model"]["magnitude_7"]["lstm_num_layers"], 
        output_steps = cf["model"]["magnitude_7"]["output_steps"],
        kernel_size=4,
        dilation_base=3

    )
    checkpoint = torch.load('./models/' + model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Epoch: ", checkpoint["epoch"], "Valid loss: ", checkpoint["valid_loss"], "Training loss: ", checkpoint["training_loss"])
    model.eval()
    model.to("cuda")
    # create `DataLoader`
    val_dataloader = DataLoader(dataset_val, batch_size=cf["training"]["magnitude_7"]["batch_size"], shuffle=True, drop_last=True)
    num_data = len(val_dataloader) * val_dataloader.batch_size
    # define optimizer, scheduler and loss function
    criterion2 = nn.MSELoss()    

    mean_squared_error_val_loss = 0
    model.eval()

    for idx, (x, y) in enumerate(val_dataloader):
        batchsize = x.shape[0]

        x = x.to("cuda")
        y = y.to("cuda")
        out = model(x)
        loss2 = criterion2(out, y)
        mean_squared_error_val_loss += loss2.detach().item()

    print(model_name + ' infer MSE loss:{:.6f}'
                    .format(mean_squared_error_val_loss))

    return mean_squared_error_val_loss


def evalute_Movement_14(dataset_val, features):
    batch_size = cf["training"]["movement_14"]["batch_size"]
    model_name = cf["alpha_vantage"]["symbol"] +  "_"  + "movement_14"
    # here we re-initialize dataloader so the data doesn't shuffled, so we can plot the values by date
    # load the saved model weights from a file
    model = Movement_14(
        input_size = len(features),
        window_size = cf["model"]["movement_14"]["window_size"],
        lstm_hidden_layer_size = cf["model"]["movement_14"]["lstm_hidden_layer_size"], 
        lstm_num_layers = cf["model"]["movement_14"]["lstm_num_layers"], 
        output_steps = cf["model"]["movement_14"]["output_steps"],
        kernel_size=4,
        dilation_base=3

    )
    checkpoint = torch.load('./models/' + model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Epoch: ", checkpoint["epoch"], "Valid loss: ", checkpoint["valid_loss"], "Training loss: ", checkpoint["training_loss"])
    model.eval()
    model.to("cuda")
    # create `DataLoader`
    val_dataloader = DataLoader(dataset_val, batch_size=cf["training"]["movement_14"]["batch_size"], shuffle=True, drop_last=True)
    num_data = len(val_dataloader) * val_dataloader.batch_size
    # define optimizer, scheduler and loss function
    criterion1 = nn.BCELoss()    
    binary_cross_entropy_val_loss = 0
    accuracy_score = 0
    y_true = np.empty(0)
    y_pred = np.empty(0)
    model.eval()

    for idx, (x, y) in enumerate(val_dataloader):
        batchsize = x.shape[0]

        x = x.to("cuda")
        y = y.to("cuda")


        out = model(x)
        # _, prob_predict = torch.max(out[:, :1], dim=1)
        prob_predict = (out > 0.5).float()
        # _, prob_true = torch.max(y[:, :1], dim=1)
        accuracy_score += torch.sum(prob_predict == y[:, :1]).item()
        y_true = np.append(y_true, [value for value in y[:, :1].cpu().detach().numpy()])
        y_pred = np.append(y_pred, [value for value in prob_predict.cpu().detach().numpy()])
        loss1 = criterion1(out[:, :1], y[:, :1])
        binary_cross_entropy_val_loss += loss1.detach().item()
    print(model_name + ' infer BCE loss:{:.6f}, Accuracy:{:.6f}'
                    .format(binary_cross_entropy_val_loss, accuracy_score/num_data))

    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    return binary_cross_entropy_val_loss

def evalute_magnitude_14(dataset_val, features):
    batch_size = cf["training"]["magnitude_14"]["batch_size"]
    model_name = cf["alpha_vantage"]["symbol"] +  "_"  + "magnitude_14"
    # here we re-initialize dataloader so the data doesn't shuffled, so we can plot the values by date
    # load the saved model weights from a file
    model = Magnitude_14(
        input_size = len(features),
        window_size = cf["model"]["magnitude_14"]["window_size"],
        lstm_hidden_layer_size = cf["model"]["magnitude_14"]["lstm_hidden_layer_size"], 
        lstm_num_layers = cf["model"]["magnitude_14"]["lstm_num_layers"], 
        output_steps = cf["model"]["magnitude_14"]["output_steps"],
        kernel_size=4,
        dilation_base=3

    )
    checkpoint = torch.load('./models/' + model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Epoch: ", checkpoint["epoch"], "Valid loss: ", checkpoint["valid_loss"], "Training loss: ", checkpoint["training_loss"])
    model.eval()
    model.to("cuda")
    # create `DataLoader`
    val_dataloader = DataLoader(dataset_val, batch_size=cf["training"]["magnitude_14"]["batch_size"], shuffle=True, drop_last=True)
    num_data = len(val_dataloader) * val_dataloader.batch_size
    # define optimizer, scheduler and loss function
    criterion2 = nn.MSELoss()    

    mean_squared_error_val_loss = 0
    model.eval()

    for idx, (x, y) in enumerate(val_dataloader):
        batchsize = x.shape[0]

        x = x.to("cuda")
        y = y.to("cuda")
        out = model(x)
        loss2 = criterion2(out, y)
        mean_squared_error_val_loss += loss2.detach().item()

    print(model_name + ' infer MSE loss:{:.6f}'
                    .format(mean_squared_error_val_loss))

    return mean_squared_error_val_loss
