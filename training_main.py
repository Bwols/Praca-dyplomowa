import torch
from train_models import TrainModels
from constants import *

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

print(device)
"""
possible models
prin

testy zrobione na ADAMW to tak naprawdę ADAM


print(device)
def train_GAN():
    ta = TrainModels(device,"EKSPERYMENTY_GAN/ADAMlr=0.0002_05_BCE_16_SUM", model_type="GAN")#trenował 60 minut
    ta.set_loss_criterion("BCE","sum")
    ta.set_batch_size(16)
    ta.set_net_optimizer("generator", ADAM, lr=0.0002, momentum=0.5)
    ta.set_net_optimizer("discriminator", ADAM, lr=0.0002, momentum=0.5)

    ta.train_loop(100)

#train_GAN()

def train_GAN():
    ta = TrainModels(device,"EKSPERYMENTY_GAN/ADAMlr=0.01_05_BCE_16_MEAN", model_type="GAN")
    ta.set_loss_criterion("BCE")
    ta.set_batch_size(16)
    ta.set_net_optimizer("generator", ADAM, lr=0.01, momentum=0.5)
    ta.set_net_optimizer("discriminator", ADAM, lr=0.01, momentum=0.5)

    ta.train_loop(100)

#train_GAN()
"""
def train_GAN():
    ta = TrainModels(device,"EKSPERYMENTY_GAN/ADAMlr=0.01_05_BCE_16_GEN_k=4", model_type="GAN")
    ta.set_loss_criterion("BCE")
    ta.set_batch_size(16)
    ta.set_net_optimizer("generator", ADAM, lr=0.0002, momentum=0.5)
    ta.set_net_optimizer("discriminator", ADAM, lr=0.0002, momentum=0.5)

    ta.train_loop(100)

#train_GAN()



"""
def train_CGAN():
    ta = TrainModels(device,"EKSPERYMENTY_CGAN/ADAM_lr=0.0002_0.5_beta_cond", model_type="CGAN")
    ta.set_loss_criterion("BCE",)
    ta.set_batch_size(16)
    ta.set_net_optimizer("generator", ADAM,lr=0.0002, momentum=0.5)
    ta.set_net_optimizer("discriminator", ADAM, lr=0.0002 , momentum=0.5)

    ta.train_loop(100)
    
    
    
def train_CGAN():
    ta = TrainModels(device,"EKSPERYMENTY_CGAN/ADAM_lr=0.005_0.9_beta_cond", model_type="CGAN")
    ta.set_loss_criterion("BCE","sum")
    ta.set_batch_size(16)
    ta.set_net_optimizer("generator", ADAM,lr=0.01, momentum=0.5)
    ta.set_net_optimizer("discriminator", ADAM, lr=0.01 , momentum=0.5)

    ta.train_loop(100)

train_CGAN()
    """
def train_CGAN():
    ta = TrainModels(device,"EKSPERYMENTY_CGAN/ADAM_lr=0.001_0.9_beta_cond_final_prey", model_type="CGAN")
    ta.set_loss_criterion("BCE")
    ta.set_batch_size(16)
    ta.set_net_optimizer("generator", ADAM,lr=0.001, momentum=0.9)
    ta.set_net_optimizer("discriminator", ADAM, lr=0.001 , momentum=0.5)#35.63145569960276 minutes

    ta.train_loop(100)

#train_CGAN()

def train_VAE():
    ta = TrainModels(device, "EKSPERYMENTY_VAE/ADAM_lr=0.001_0.5_ds_fin_mean", model_type="VAE")
    ta.set_loss_criterion("MSE")
    ta.set_batch_size(64)
    ta.set_net_optimizer("VAE", ADAM, lr=0.001, momentum=0.5)


    ta.train_loop(150)

train_VAE()


def train_CVAE():
    ta = TrainModels(device,"EKSPERYMENTY_CVAE/ADAMlr=0.005_0.9_fixed2", model_type="CVAE") #TODO normailization to 0,1 maska -1,1
    ta.set_loss_criterion("MSE")
    ta.set_batch_size(64)
    ta.set_net_optimizer("VAE", ADAM, lr=0.005, momentum=0.9)


    ta.train_loop(100)



#train_CVAE()
"""
    def choose_gen_dis_optimizer(self):
        beta1 = 0.5
        self.discriminator_optimizer = torch.optim.Adam(self.model.discriminator.parameters(), lr=0.0002, betas=(beta1, 0.999))  # 1 seria 0.001 2 seria 0.0002
        self.generator_optimizer = torch.optim.Adam(self.model.generator.parameters(), lr=0.0002, betas=(beta1, 0.999))

        self.model.set_discriminator_optimizer(self.discriminator_optimizer)
        self.model.set_generator_optimizer(self.generator_optimizer)

    def choose_vae_optimizer(self):
        self.vae_optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
    
        def set_generator_optimizer(self, optimizer_name, lr, momentum):
        if optimizer_name == ADAM:
            optimizer = torch.optim.Adam(self.model.generator.parameters(), lr=lr, betas=(momentum, 0.999))

        elif optimizer_name == ADAMW:
            optimizer = torch.optim.AdamW(self.model.generator.parameters(),lr=lr,betas=(momentum,0.999))

        elif optimizer_name == SGD:
            optmizer = torch.optim.SGD(self.model.generator.parameters(),lr=lr, momentum= momentum)


     def choose_optimizer(self):
        if self.model_type == VANILLA_GAN or self.model_type == CGAN:
            self.choose_gen_dis_optimizer()
        else:
            self.choose_vae_optimizer()



"""