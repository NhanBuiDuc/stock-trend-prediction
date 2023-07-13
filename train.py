# from trainer.svm_trainer import svm_trainer
#
# trainer = svm_trainer(new_data=True, full_data=False, mode = "train")
#
# model = trainer.train()
# model = trainer.model
# model = model.load_check_point(model.model_type, trainer.model.name)
# trainer.eval(model)
#
#
# from trainer.random_forest_trainer import random_forest_trainer
#
# trainer = random_forest_trainer(new_data=False, full_data=False, mode = "train")
#
# model = trainer.train()
# model = trainer.model
# model = model.load_check_point(model.model_type, trainer.model.name)
# trainer.eval(model)
#
# import trainer.xgboost_trainer as xgboost_tn
# trainer = xgboost_tn.xgboost_trainer(new_data=False, full_data=False, mode = "train")
#
# trainer.train()
# model = trainer.model
# model = model.load_check_point(model.model_type, trainer.model.name)
# trainer.eval(model)
#
# import trainer.lstm_trainer as lstm_tn
# trainer = lstm_tn.lstm_trainer(new_data=False, full_data=False, mode = "train")
#
# trainer.train()
# model = trainer.model
# model = model.load_check_point(model.model_type, trainer.model.name)
# trainer.eval(model)


import trainer.transformer_trainer as ttn
from model import Model
from configs.config import config

trainer = ttn.Transformer_trainer(new_data=False, full_data=False, mode = "train")

trainer.train()
model = trainer.model
model = model.load_check_point(model.model_type, trainer.model.name)
trainer.eval(model)
