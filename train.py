import os, sys, random, numpy as np, torch, torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from pytorch_msssim import ms_ssim as ms_ssim_torch
from model import Codec2

DATA = sys.argv[1] if len(sys.argv) > 1 else "DIV2K_train_HR"
device = "cuda" if torch.cuda.is_available() else "cpu"

class MultiPatchDataset(Dataset):
    def __init__(self, folder, p=256, k=4):
        self.files = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.endswith(".png")]
        self.p, self.k = p, k
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        a = np.array(Image.open(self.files[i]).convert("RGB"))
        h, w, _ = a.shape
        out = []
        for _ in range(self.k):
            t, l = random.randint(0, h - self.p), random.randint(0, w - self.p)
            c = a[t:t + self.p, l:l + self.p]
            if random.random() < 0.5: c = c[:, ::-1]
            c = np.ascontiguousarray(c).astype(np.float32) / 255.0
            out.append(torch.from_numpy(c).permute(2, 0, 1))
        return torch.stack(out)

loader = DataLoader(MultiPatchDataset(DATA), batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
model = Codec2().to(device)
params = list(model.enc.parameters()) + list(model.dec.parameters())


opt = torch.optim.Adam(params, lr=1e-4)
for epoch in range(30):
    for batch in loader:
        batch = batch.flatten(0, 1).to(device)
        opt.zero_grad()
        rec, _, _ = model(batch)
        loss = nn.functional.mse_loss(rec, batch)
        loss.backward(); opt.step()
    print("stage 1 epoch", epoch + 1, loss.item())


LAMBDA = 100
opt = torch.optim.Adam([{"params": params, "lr": 1e-4},
                        {"params": model.prior.parameters(), "lr": 1e-3}])
for epoch in range(60):
    for batch in loader:
        batch = batch.flatten(0, 1).to(device)
        opt.zero_grad()
        rec, bpp, _ = model(batch)
        ms = ms_ssim_torch(rec.clamp(0, 1), batch, data_range=1.0)
        loss = bpp + LAMBDA * (1 - ms)
        loss.backward(); opt.step()
    print("stage 2 epoch", epoch + 1, "bpp", bpp.item(), "ms-ssim", ms.item())
    torch.save(model.state_dict(), "weights.pth")
