

Method: convolutional autoencoder (3 stride-2 conv layers) with a learned
per-channel factorized prior, trained with a loss of bit cost + MS-SSIM
on DIV2K only. The encoder rounds the latent with a per-image step size
(chosen so MS-SSIM stays above 0.952) and writes it with a range coder
(constriction).

Install:    pip install -r requirements.txt
Compress:   python compress.py input_dir output_dir
Decompress: python decompress.py compressed_dir output_dir
Train:      python train.py path/to/DIV2K_train_HR
