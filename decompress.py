import os, sys, struct, numpy as np, torch
from PIL import Image
from model import *
from compress import QS

device = "cuda" if torch.cuda.is_available() else "cpu"

def main(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    m = Codec2().to(device)
    m.load_state_dict(torch.load("weights.pth", map_location=device)); m.eval()
    for f in sorted(os.listdir(in_dir)):
        if not f.endswith(".bin"): continue
        data = open(os.path.join(in_dir, f), "rb").read()
        h, w, qi = struct.unpack("<HHB", data[:5])
        q = QS[qi]
        words = np.frombuffer(data[5:], dtype=np.uint32)
        hp, wp = h + (-h) % 8, w + (-w) % 8
        y_q = entropy_decode(words, build_models(m, q), (LATENT_CH, hp // 8, wp // 8))
        with torch.no_grad():
            yq = torch.from_numpy(y_q.astype(np.float32)).unsqueeze(0).to(device)
            rec = m.dec(yq * q).squeeze(0).permute(1, 2, 0).cpu().numpy()
        rec = np.clip(rec * 255, 0, 255).astype(np.uint8)[:h, :w]
        Image.fromarray(rec).save(os.path.join(out_dir, os.path.splitext(f)[0] + ".png"))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
