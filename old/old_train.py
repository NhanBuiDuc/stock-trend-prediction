
def train_LSTM_classifier_1(dataset_train, dataset_val, is_training=True):

    binary_model = model.LSTM_Classifier_1(
        input_size = cf["model"]["lstm_classification1"]["input_size"],
        window_size = cf["data"]["window_size"],
        hidden_layer_size = cf["model"]["lstm_classification1"]["lstm_size"], 
        num_layers = cf["model"]["lstm_classification1"]["num_lstm_layers"], 
        output_size = cf["model"]["lstm_classification1"]["output_dates"],
        dropout = cf["model"]["lstm_classification1"]["dropout"]
    )
    binary_model.to("cuda")
    # create `DataLoader`
    train_dataloader = DataLoader(dataset_train, batch_size=cf["training"]["lstm_classification1"]["batch_size"])
    val_dataloader = DataLoader(dataset_val, batch_size=cf["training"]["lstm_classification1"]["batch_size"], shuffle=True)

    # define optimizer, scheduler and loss function
    criterion = nn.BCELoss()

    optimizer = optim.Adam(binary_model.parameters(), lr=cf["training"]["lstm_classification1"]["learning_rate"], betas=(0.9, 0.98), eps=1e-9, weight_decay=0.001)
    # optimizer = optim.SGD(binary_model.parameters(), lr=cf["training"]["lstm_classification1"]["learning_rate"], momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=cf["training"]["lstm_classification1"]["scheduler_step_size"], verbose=True)

    best_loss = sys.float_info.max
    stop = False
    patient = cf["training"]["lstm_classification1"]["patient"]
    patient_count = 0
    stopped_epoch = 0
    # begin training
    for epoch in range(cf["training"]["lstm_classification1"]["num_epoch"]):
        loss_train, lr_train = run_epoch(binary_model,  train_dataloader, optimizer, criterion, scheduler, is_training=True)
        loss_val, lr_val = run_epoch(binary_model, val_dataloader, optimizer, criterion, scheduler, is_training=False)
        scheduler.step(loss_val)
        if(check_best_loss(best_loss=best_loss, loss=loss_val)):
            best_loss = loss_val
            patient_count = 0
            save_best_model(model=binary_model, name="lstm_classification_1", num_epochs=epoch, optimizer=optimizer, val_loss=loss_val, training_loss=loss_train, learning_rate=lr_train)
        else:
            stop, patient_count, best_loss, _ = early_stop(best_loss=best_loss, current_loss=loss_val, patient_count=patient_count, max_patient=patient)

        print('Epoch[{}/{}] | loss train:{:.6f}, valid:{:.6f} | lr:{:.6f}'
                .format(epoch+1, cf["training"]["lstm_classification1"]["num_epoch"], loss_train, loss_val, lr_train))
        
        print("patient", patient_count)
        if(stop == True):
            print("Early Stopped At Epoch: {}", epoch)
            stopped_epoch = patient_count
            break
    return binary_model

def train_LSTM_classifier_7(dataset_train, dataset_val, is_training=True):

    binary_model = model.LSTM_Classifier_7(
        input_size = cf["model"]["lstm_classification7"]["input_size"],
        window_size = cf["data"]["window_size"],
        hidden_layer_size = cf["model"]["lstm_classification7"]["lstm_size"], 
        num_layers = cf["model"]["lstm_classification7"]["num_lstm_layers"], 
        output_size = cf["model"]["lstm_classification7"]["output_dates"],
        dropout = cf["model"]["lstm_classification7"]["dropout"]
    )
    binary_model.to("cuda")
    # create `DataLoader`
    train_dataloader = DataLoader(dataset_train, batch_size=cf["training"]["lstm_classification7"]["batch_size"])
    val_dataloader = DataLoader(dataset_val, batch_size=cf["training"]["lstm_classification7"]["batch_size"], shuffle=True)

    # define optimizer, scheduler and loss function
    criterion = nn.BCELoss()
    # optimizer = optim.SGD(binary_model.parameters(), lr=cf["training"]["lstm_classification7"]["learning_rate"], momentum=0.9)

    optimizer = optim.Adam(binary_model.parameters(), lr=cf["training"]["lstm_classification7"]["learning_rate"], betas=(0.9, 0.98), eps=1e-9, weight_decay=0.001)
    """
    For example, suppose step_size=10 and gamma=0.1.
    This means that the learning rate will be multiplied by 0.1 every 10 epochs.
    If the initial learning rate is 0.1, then the learning rate will be reduced to 0.01 after 10 epochs, 0.001 after 20 epochs, and so on.
    """

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=cf["training"]["lstm_classification7"]["scheduler_step_size"], verbose=True)
    best_loss = sys.float_info.max
    stop = False
    patient = cf["training"]["lstm_classification7"]["patient"]
    patient_count = 0

    # begin training
    for epoch in range(cf["training"]["lstm_classification7"]["num_epoch"]):
        loss_train, lr_train = run_epoch(binary_model,  train_dataloader, optimizer, criterion, scheduler, is_training=True)
        loss_val, lr_val = run_epoch(binary_model, val_dataloader, optimizer, criterion, scheduler, is_training=False)
        scheduler.step(metrics=loss_val)
        if(check_best_loss(best_loss=best_loss, loss=loss_val)):
            best_loss = loss_val
            patient_count = 0
            save_best_model(model=binary_model, name="lstm_classification_7", num_epochs=epoch, optimizer=optimizer, val_loss=loss_val, training_loss=loss_train, learning_rate=lr_train)
        else:
            stop, patient_count, best_loss, _ = early_stop(best_loss=best_loss, current_loss=loss_val, patient_count=patient_count, max_patient=patient)

        print('Epoch[{}/{}] | loss train:{:.6f}, valid:{:.6f} | lr:{:.6f}'
                .format(epoch+1, cf["training"]["lstm_classification7"]["num_epoch"], loss_train, loss_val, lr_train))
        
        print("patient", patient_count)
        if(stop == True):
            print("Early Stopped At Epoch: {}", epoch)
            stopped_epoch = patient_count
            break
    return binary_model


def train_LSTM_classifier_14(dataset_train, dataset_val, is_training=True):

    binary_model = model.LSTM_Classifier_14(
        input_size = cf["model"]["lstm_classification14"]["input_size"],
        window_size = cf["data"]["window_size"],
        hidden_layer_size = cf["model"]["lstm_classification14"]["lstm_size"], 
        num_layers = cf["model"]["lstm_classification14"]["num_lstm_layers"], 
        output_size = cf["model"]["lstm_classification14"]["output_dates"],
        dropout = cf["model"]["lstm_classification14"]["dropout"]
    )
    binary_model.to("cuda")
    # create `DataLoader`
    train_dataloader = DataLoader(dataset_train, batch_size=cf["training"]["lstm_classification14"]["batch_size"])
    val_dataloader = DataLoader(dataset_val, batch_size=cf["training"]["lstm_classification14"]["batch_size"], shuffle=True)

    # define optimizer, scheduler and loss function
    criterion = nn.BCELoss()
    # optimizer = optim.SGD(binary_model.parameters(), lr=cf["training"]["lstm_classification14"]["learning_rate"], momentum=0.9)

    optimizer = optim.Adam(binary_model.parameters(), lr=cf["training"]["lstm_classification14"]["learning_rate"], betas=(0.9, 0.98), eps=1e-9, weight_decay=0.001)
    """
    For example, suppose step_size=10 and gamma=0.1.
    This means that the learning rate will be multiplied by 0.1 every 10 epochs.
    If the initial learning rate is 0.1, then the learning rate will be reduced to 0.01 after 10 epochs, 0.001 after 20 epochs, and so on.
    """

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=cf["training"]["lstm_classification14"]["scheduler_step_size"], verbose=True)
    best_loss = sys.float_info.max
    stop = False
    patient = cf["training"]["lstm_classification14"]["patient"]
    patient_count = 0

    # begin training
    for epoch in range(cf["training"]["lstm_classification14"]["num_epoch"]):
        loss_train, lr_train = run_epoch(binary_model,  train_dataloader, optimizer, criterion, scheduler, is_training=True)
        loss_val, lr_val = run_epoch(binary_model, val_dataloader, optimizer, criterion, scheduler, is_training=False)
        scheduler.step(metrics=loss_val)
        if(check_best_loss(best_loss=best_loss, loss=loss_val)):
            best_loss = loss_val
            patient_count = 0
            save_best_model(model=binary_model, name="lstm_classification_14", num_epochs=epoch, optimizer=optimizer, val_loss=loss_val, training_loss=loss_train, learning_rate=lr_train)
        else:
            stop, patient_count, best_loss, _ = early_stop(best_loss=best_loss, current_loss=loss_val, patient_count=patient_count, max_patient=patient)

        print('Epoch[{}/{}] | loss train:{:.6f}, valid:{:.6f} | lr:{:.6f}'
                .format(epoch+1, cf["training"]["lstm_classification14"]["num_epoch"], loss_train, loss_val, lr_train))
        
        print("patient", patient_count)
        if(stop == True):
            print("Early Stopped At Epoch: {}", epoch)
            stopped_epoch = patient_count
            break
    return binary_model