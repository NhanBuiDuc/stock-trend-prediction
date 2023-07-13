
import trainer.transformer_trainer as ttn
from model import Model

trainer = ttn.Transformer_trainer(new_data=False, full_data=False, mode = "eval")

model = trainer.model
model = model.load_check_point(model.model_type, trainer.model.name)
trainer.eval(model)

#
# from trainer.svm_trainer import svm_trainer
#
# trainer = svm_trainer(new_data=False, full_data=False, mode = "train")
#
# # model = trainer.train()
# model = trainer.model
# model = model.load_check_point(trainer.model.name)
# trainer.eval(model)


# import trainer.transformer_trainer as ttn
# from model import Model
# from configs.config import config

# model_type = "transformer"
# symbol = config["alpha_vantage"]["symbol"]
# data_mode = 0
# window_size = 1
# output_size = 1
# model_name = f'{model_type}_{symbol}_w{window_size}_o{output_size}_d{str(data_mode)}'
# trainer = ttn.Transformer_trainer(model_name=model_name, new_data=True, full_data=False, mode = "eval")
# model = Model(name=model_name)
# model = model.load_check_point(model_name)
# trainer.eval(model)