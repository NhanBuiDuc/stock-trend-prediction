import torch
from configs import config as cf
import numpy as np
from old import model
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from matplotlib.pyplot import figure
# predict on the unseen data, tomorrow's price 
def to_plot(dataset_test, dataset_val, y_test, y_val, num_data_points, dates, test_dates, val_dates):

    test_model = model.Assemble_1()
    model_name = cf["alpha_vantage"]["symbol"] +  "_"  + "assemble_1"
    checkpoint = torch.load('./models/' + model_name)
    test_model.load_state_dict(checkpoint['model_state_dict'])
    test_model.eval()
    test_model.to("cuda")
    test_scaler = dataset_test.scaler
    val_scaler = dataset_val.scaler
    test_dataloader = DataLoader(dataset_test, batch_size=64, shuffle=False)
    val_dataloader = DataLoader(dataset_val, batch_size=64, shuffle=False)
    
    y_test = np.reshape(y_test, y_test.shape[0])
    y_val = np.reshape(y_val, y_val.shape[0])
    # predict on the training data, to see how well the model managed to learn and memorize

    test_prediction = np.empty(shape=(0))

    for idx, (x, y) in enumerate(test_dataloader):
        x = x.to("cuda")
        out = test_model(x)
        out = out.cpu().detach().numpy()
        out = np.reshape(out, out.shape[0])
        test_prediction = np.concatenate((test_prediction, out))

    # predict on the validation data, to see how the model does

    val_prediction = np.empty(shape=(0))

    for idx, (x, y) in enumerate(val_dataloader):
        x = x.to("cuda")
        out = test_model(x)
        out = out.cpu().detach().numpy()
        out = np.reshape(out, out.shape[0])
        val_prediction = np.concatenate((val_prediction, out))
        
    val_prediction = np.reshape(val_prediction, (val_prediction.shape[0], 1))
    test_prediction = np.reshape(test_prediction, (test_prediction.shape[0],1))
    val_prediction = val_scaler.inverse_transform(val_prediction)
    test_prediction = test_scaler.inverse_transform(test_prediction)
    val_prediction = np.reshape(val_prediction, (val_prediction.shape[0], ))
    test_prediction = np.reshape(test_prediction, (test_prediction.shape[0],))
    if cf["plots"]["show_plots"]:
        # prepare plots
        
        plot_date_test = val_dates + test_dates
        plot_range = 14
        plot_size = len(plot_date_test)
        to_plot_data_y_val = np.zeros(plot_size)
        to_plot_data_y_test = np.zeros(plot_size)
        to_plot_data_y_val_pred = np.zeros(plot_size)
        to_plot_data_y_test_pred = np.zeros(plot_size)

        to_plot_data_y_val[:len(y_val)] = (y_val)
        to_plot_data_y_test[len(y_val):] = (y_test)
        to_plot_data_y_val_pred[:len(val_prediction)] = (val_prediction)
        to_plot_data_y_test_pred[len(val_prediction):] = (test_prediction)

        to_plot_data_y_val = np.where(to_plot_data_y_val == 0, None, to_plot_data_y_val)
        to_plot_data_y_test = np.where(to_plot_data_y_test == 0, None, to_plot_data_y_test)
        to_plot_data_y_val_pred = np.where(to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred)
        to_plot_data_y_test_pred = np.where(to_plot_data_y_test_pred == 0, None, to_plot_data_y_test_pred)

        # plot

        fig = figure(figsize=(50, 50), dpi=80)
        fig.patch.set_facecolor((1.0, 1.0, 1.0))
        plt.plot(plot_date_test[-30+len(val_dates):], to_plot_data_y_val[-30+len(val_dates):], label="Actual prices validation", marker="*", markersize=10, color=cf["plots"]["color_actual_val"])
        plt.plot(plot_date_test[-30+len(val_dates):], to_plot_data_y_val_pred[-30+len(val_dates):], label="Past predicted validation prices", marker="o", markersize=1, color=cf["plots"]["color_pred_val"])
        # plt.plot(plot_date_test, to_plot_data_y_test, label="Actual prices test", marker="*", markersize=10, color=cf["plots"]["color_actual_test"])
        # plt.plot(plot_date_test, to_plot_data_y_test_pred, label="Past predicted validation prices", marker="o", markersize=1, color=cf["plots"]["color_pred_test"])

        xticks = [plot_date_test[i] if ((i%2 == 0 > 2) or i > plot_range)  else None for i in range(plot_range)]
        plt.title("Predicted close price of the next trading day")
        x = np.arange(0,len(xticks))
        plt.xticks(x, xticks, rotation='vertical')

        plt.grid(b=None, which='major', axis='y', linestyle='--')
        plt.legend()
        plt.show()

        fig = figure(figsize=(50, 50), dpi=80)
        fig.patch.set_facecolor((1.0, 1.0, 1.0))
        plt.plot(plot_date_test[-30:], to_plot_data_y_test[-30:], label="Actual prices test", marker="*", markersize=10, color=cf["plots"]["color_actual_test"])
        plt.plot(plot_date_test[-30:], to_plot_data_y_test_pred[-30:], label="Past predicted validation prices", marker="o", markersize=1, color=cf["plots"]["color_pred_test"])
        xticks = [plot_date_test[i] if ((i%2 == 0 > 2) or i > plot_range)  else None for i in range(plot_range)]
        plt.title("Predicted close price of the next trading day")
        x = np.arange(0,len(xticks))
        plt.xticks(x, xticks, rotation='vertical')

        plt.grid(b=None, which='major', axis='y', linestyle='--')
        plt.legend()
        plt.show()
    # print("Predicted close price of the next trading day:", round(test_prediction, 2))

# if __name__ == "__main__":
#     data_df, num_data_points, data_date = utils.download_data_api()

#     to_plot(data_df, num_data_points, data_date)