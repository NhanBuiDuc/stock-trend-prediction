from configs.config import config as cf
import torch
from tqdm import tqdm


# Main Trainer class, used for initialize the type of model with given model name
class Trainer:
    def __init__(self):
        self.cf = cf


def check_best_loss(best_loss, loss):
    if loss < best_loss:
        return True
    return False


# return boolen Stop, patient_count, best_loss, current_loss
def is_early_stop(best_loss, current_loss, patient_count, max_patient):
    stop = False
    if best_loss > current_loss:
        best_loss = current_loss
        patient_count = 0
    else:
        patient_count = patient_count + 1
    if patient_count >= max_patient:
        stop = True
    return stop, patient_count, best_loss, current_loss


def run_epoch(model, dataloader, optimizer, criterion, scheduler, is_training, device):
    epoch_loss = 0

    weight_decay = 0.001
    if is_training:
        model.structure.train()
    else:
        model.structure.eval()

    # create a tqdm progress bar
    dataloader = tqdm(dataloader)
    for idx, (x, y) in enumerate(dataloader):
        if is_training:
            optimizer.zero_grad()
        batchsize = x.shape[0]
        # print(x.shape)
        x = x.to(device)
        y = y.to(device)
        out = model.predict(x)
        loss = criterion(out, y)
        if is_training:
            if loss != torch.nan:
                torch.autograd.set_detect_anomaly(True)
                loss.backward()
                optimizer.step()
            else:
                print("loss = nan")
        batch_loss = (loss.detach().item())
        epoch_loss += batch_loss
        # update the progress bar
        dataloader.set_description(f"At index {idx:.4f}")

    try:
        lr = scheduler.get_last_lr()[0]

    except:
        lr = optimizer.param_groups[0]['lr']
    return epoch_loss, lr


def dataset_check(X, y, window_size, output_size, stride):
    result = []
    batch = X.shape[0]
    for i in range(0, batch - 1, 1):
        if (X[i + 1][(output_size - 1)][0] > X[i][(window_size - 1)][0]) and y[i] == 1:
            result.append(True)
        elif (X[i + 1][(output_size - 1)][0] < X[i][(window_size - 1)][0]) and y[i] == 0:
            result.append(True)
        else:
            result.append(False)

    count_false = 0
    for element in result:
        if not element:
            count_false += 1
    if False in result:
        print("Number of wrong: ", count_false)
        print("Wrong dataset")
    else:
        print("Correct dataset")
