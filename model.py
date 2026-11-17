import numpy as np, torch, torch.nn as nn, constriction
from scipy.special import expit

LATENT_CH = 48
R = 300   

def quantize(y, training):
    if training:
        return y + (torch.rand_like(y) - 0.5)
    return torch.round(y)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 5, 2, 2),   nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, 5, 2, 2), nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, 5, 2, 2), nn.LeakyReLU(0.1),
            nn.Conv2d(128, LATENT_CH, 3, 1, 1),
        )
    def forward(self, x): return self.net(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(LATENT_CH, 128, 3, 1, 1), nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(128, 128, 5, 2, 2, output_padding=1), nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(128, 64, 5, 2, 2, output_padding=1),  nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(64, 3, 5, 2, 2, output_padding=1),
        )
    def forward(self, y): return self.net(y)

class FactorizedPrior(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(1, ch, 1, 1))
        self.log_s = nn.Parameter(torch.zeros(1, ch, 1, 1))
    def forward(self, y_q):
        s = torch.exp(self.log_s).clamp(min=0.05)
        p = (torch.sigmoid((y_q + 0.5 - self.mu) / s) - torch.sigmoid((y_q - 0.5 - self.mu) / s)).clamp(min=1e-9)
        return -torch.log2(p).sum()

class Codec2(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc, self.dec = Encoder(), Decoder()
        self.prior = FactorizedPrior(LATENT_CH)
    def forward(self, x):
        y = self.enc(x)
        y_q = quantize(y, self.training)
        bits = self.prior(y_q)
        return self.dec(y_q), bits / (x.shape[0] * x.shape[2] * x.shape[3]), y_q


def pad8(img):                   
    h, w, _ = img.shape
    ph, pw = (-h) % 8, (-w) % 8
    return np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="edge")

def build_models(m, q):
    """One probability table per channel, computed in float64 on the CPU
    so the encoder and decoder get exactly the same numbers."""
    mu = m.prior.mu.detach().cpu().double().numpy().reshape(-1) / q
    s = np.maximum(np.exp(m.prior.log_s.detach().cpu().double().numpy().reshape(-1)), 0.05) / q
    k = np.arange(-R, R + 1, dtype=np.float64)
    models = []
    for c in range(len(mu)):
        p = expit((k + 0.5 - mu[c]) / s[c]) - expit((k - 0.5 - mu[c]) / s[c])
        p = np.maximum(p, 1e-12)
        models.append(constriction.stream.model.Categorical(p / p.sum(), perfect=False))
    return models

def entropy_encode(y_q, models):     
    coder = constriction.stream.queue.RangeEncoder()
    for c, mdl in enumerate(models):
        coder.encode((y_q[c].reshape(-1) + R).astype(np.int32), mdl)
    return coder.get_compressed()    

def entropy_decode(words, models, shape):   
    coder = constriction.stream.queue.RangeDecoder(words)
    n = shape[1] * shape[2]
    out = np.stack([coder.decode(mdl, n) - R for mdl in models])
    return out.reshape(shape)
