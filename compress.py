import os, sys, struct, numpy as np, torch
from PIL import Image
from model import *
from evaluator import ms_ssim

TARGET = 0.952                                  
QS = [3.0, 2.6, 2.3, 2.0, 1.8, 1.6, 1.4, 1.25, 1.1, 1.0, 0.85, 0.7, 0.55, 0.4]
device = "cuda" if torch.cuda.is_available() else "cpu"

def main(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    m = Codec2().to(device)
    m.load_state_dict(torch.load("weights.pth", map_location=device)); m.eval()
    for f in sorted(os.listdir(in_dir)):
        if not f.lower().endswith(".png"): continue
        orig = np.array(Image.open(os.path.join(in_dir, f)).convert("RGB"), dtype=np.uint8)
        h, w, _ = orig.shape
        x = torch.from_numpy(pad8(orig).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            y = m.enc(x)
            for qi, q in enumerate(QS):
                y_q = torch.round(y / q).clamp(-R, R)
                rec = m.dec(y_q * q).squeeze(0).permute(1, 2, 0).cpu().numpy()
                rec = np.clip(rec * 255, 0, 255).astype(np.uint8)[:h, :w]
                if ms_ssim(orig, rec) >= TARGET:
                    break
        words = entropy_encode(y_q.squeeze(0).cpu().numpy().astype(np.int64), build_models(m, q))
        with open(os.path.join(out_dir, os.path.splitext(f)[0] + ".bin"), "wb") as fh:
            fh.write(struct.pack("<HHB", h, w, qi) + words.tobytes())
        print(f"{f}: q={q} -> {5 + 4 * len(words)} bytes")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
