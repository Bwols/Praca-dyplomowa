import torch
from train_models import TrainModels


device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
"""
possible models



"""
ta = TrainModels(device,visuals_dir="CVAE_17.08_VISUALS",model_dir="CVAE_17.08_MODELS_vanilla", model_type="CVAE")
ta.train_loop(100)