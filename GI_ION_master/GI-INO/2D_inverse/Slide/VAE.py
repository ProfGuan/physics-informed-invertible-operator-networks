import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepConvVAE(nn.Module):
    def __init__(self, latent_dim=100):
        super(DeepConvVAE, self).__init__()
        self.latent_dim = latent_dim


        self.enc_conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)    # 112x112 -> 56x56
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)   # 56x56 -> 28x28
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # 28x28 -> 14x14
        self.enc_conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1) # 14x14 -> 7x7

        self.enc_fc1 = nn.Linear(256 * 7 * 7, 1024)
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

  
        self.dec_fc1 = nn.Linear(latent_dim, 1024)
        self.dec_fc2 = nn.Linear(1024, 256 * 7 * 7)

        self.dec_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1) # 7x7 -> 14x14
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 14x14 -> 28x28
        self.dec_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)   # 28x28 -> 56x56
        self.dec_conv4_1 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)  # [32, 56, 56] → [16, 112, 112]
        self.dec_conv5_1 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1)            # [16, 112, 112] → [8, 112, 112]
        self.dec_conv6_1 = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1)             # [8, 112, 112] → [2, 112, 112]
        self.dec_conv4_2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)  # [32, 56, 56] → [16, 112, 112]
        self.dec_conv5_2 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1)            # [16, 112, 112] → [8, 112, 112]
        self.dec_conv6_2 = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1)  
    def encode(self, x):
        h = F.relu(self.enc_conv1(x))  # (batch, 32, 56, 56)
        h = F.relu(self.enc_conv2(h))  # (batch, 64, 28, 28)
        h = F.relu(self.enc_conv3(h))  # (batch, 128, 14, 14)
        h = F.relu(self.enc_conv4(h))  # (batch, 256, 7, 7)
        h = h.view(-1, 256 * 7 * 7)    
        h = F.relu(self.enc_fc1(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def decode(self, z):
        h = F.relu(self.dec_fc1(z))
        h = F.relu(self.dec_fc2(h))
        h = h.view(-1, 256, 7, 7)      

        h = F.relu(self.dec_conv1(h))  # (batch, 128, 14, 14)
        h = F.relu(self.dec_conv2(h))  # (batch, 64, 28, 28)
        h = F.relu(self.dec_conv3(h))  # (batch, 32, 56, 56)
        h1 = F.relu(self.dec_conv4_1(h))  # (batch, 16, 112, 112）
        h1 = F.relu(self.dec_conv5_1(h1))  # (batch, 8, 112, 112)
        h1 = torch.sigmoid(self.dec_conv6_1(h1))  # (batch, 2, 112, 112)
        h2 = F.relu(self.dec_conv4_2(h))  # (batch, 16, 112, 112）
        h2 = F.relu(self.dec_conv5_2(h2))  # (batch, 8, 112, 112)
        h2 = F.softplus(self.dec_conv6_2(h2))  # (batch, 2, 112, 112)
        return h1,h2

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + 1*eps * std

    def forward(self, x):
    
        mu_enc, logvar_enc = self.encode(x)
        z = self.reparameterize(mu_enc, logvar_enc)
        mu_dec, var_dec =self.decode(z)
        return mu_enc, logvar_enc, mu_dec, var_dec
    

device = "cpu"
latent_dim = 20 
model = DeepConvVAE(latent_dim=latent_dim).to(device)

path = './models/vae.pth'

checkpoints = torch.load(path,map_location=device)
model.load_state_dict(checkpoints['model_state_dict'])
