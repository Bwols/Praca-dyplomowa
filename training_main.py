import torch
from train_models import TrainModels


device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
"""
possible models



"""
ta = TrainModels(device,"CondGAN_21.08", model_type="CGAN")
ta.set_loss_criterion("BCE")
ta.train_loop(130)