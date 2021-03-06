import torch.nn as nn
from torch import cat, exp, rand_like, sum


class CondVAEEncoderDecoder(nn.Module):# one convolutional layer one fully_connected

    def __init__(self):
        super(CondVAEEncoderDecoder,self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 1, 3, 2, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc_log_var = nn.Linear(256,100)
        self.fc_mu = nn.Linear(256,100)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4196, 64, 6, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 3, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 16, 4, 3, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(16, 3, 4, 2, 1, bias=False),
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )



    def add_latent_mask(self,latent, mask):  # dim(1,100) i 1(64,64)
        mask = mask.view(mask.size()[0], -1)
        x = cat((latent, mask), 1)
        return x

    def add_tensors(self,tensor1,tensor2):
        return cat((tensor1,tensor2),1)

    def encode(self,image, masks):
        x = self.add_tensors(image,masks)

        encoded = self.encoder(x)

        #encoded = torch.flatten(encoded)
        encoded = encoded.view(-1,256)



        mu = self.fc_mu(encoded)
        log_var = self.fc_log_var(encoded)
        return (mu,log_var)

    def decode(self, input):
        z, mask = input

        z = self.add_latent_mask(z,mask)
        z = z.view(-1, 4196, 1, 1)
        decoded = self.decoder(z)
        return decoded

    def get_latent_vector(self, mu ,log_var):#reparametrization trick
        std = exp(0.5*log_var)
        eps = rand_like(std)
        return eps *std + mu

    def final_loss(self,reconstruction_loss,mu,log_var):

        KLD = -0.5 * sum(1 + log_var - mu ** 2 - log_var.exp())

        return reconstruction_loss + KLD




    def forward(self,image, mask):
        mu,log_var = self.encode(image,mask)
        z = self.get_latent_vector(mu,log_var)
        input = (z,mask)
        fake_image = self.decode(input)
        return fake_image,mu,log_var
