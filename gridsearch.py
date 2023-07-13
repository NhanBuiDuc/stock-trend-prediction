from trainer.svm_trainer import svm_trainer
from trainer.random_forest_trainer import random_forest_trainer
from trainer.xgboost_trainer import xgboost_trainer as xgboost_tn
import trainer.lstm_trainer as lstm_tn
from trainer.lstm_trainer import lstm_trainer as lstm_tn
from trainer.transformer_trainer import Transformer_trainer as ttn
from model import Model
from configs.config import config

### SVM
# trainer = svm_trainer(new_data=False, full_data=False, mode = "train")
# model = trainer.grid_search()

# ### random_forest
# trainer = random_forest_trainer(new_data=False, full_data=False, mode = "train")
# model = trainer.grid_search()

# # # ### xgboost
# trainer = xgboost_tn(new_data=False, full_data=False, mode = "train")
# model = trainer.grid_search()

##LSTM
# trainer = lstm_tn(new_data=False, full_data=False, mode = "train")
# model = trainer.grid_search()

# Ensembler
trainer = ttn(new_data=False, full_data=False, mode = "train")
model = trainer.grid_search()