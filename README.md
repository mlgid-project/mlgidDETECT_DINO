# DINO-GIWAXS
This is a GIWAXS oriented implementation of the paper "[DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection](https://arxiv.org/abs/2203.03605)". 

For more information on how to run the original implementation visit the original [GitHub repo](https://github.com/IDEA-Research/DINO).

Models trained using DINO-GIWAXS can be used in [mlgidDETECT](https://github.com/mlgid-project/mlgidDETECT) to perform inference. 

This repository holds two detectors for GIWAXS polar images (512 x 1024, q across, chi down) that find rings and segments. Pick one:

| | Heatmap detector (**start here**) | DINO detector |
|---|---|---|
| Idea | CNN/ViT backbone + heatmap head, peaks become boxes | DETR with denoising anchor boxes |
| AP, scaled GT (organic / 41) | **0.692 / 0.697** (ConvNeXt-L, mean of 3 seeds) | 0.587 / 0.722 |
| Training time | about 7 h on one 12 GB GPU (peak 11 GB) | about 60 h on an A100 |
| Needs compiled CUDA ops | no | yes (see Installation below) |

Both train on simulated frames only (`simulation.py`) and are scored on two labelled sets of real frames (`organic_labeled.h5`, `41.h5`).
AP is computed against the labels scaled to the box size the model was trained with (`HM_GT_BOX_COEF`), the only comparison between box sizes that is fair.

## Quick start: heatmap detector

```sh
pip install -r requirements.txt
git clone https://github.com/facebookresearch/dinov3        # model code; the weights come separately
```
1. **Weights.** Download `dinov3_convnext_large_pretrain_lvd1689m-*.pth` from [Meta's DINOv3 page](https://github.com/facebookresearch/dinov3#pretrained-backbones) (accept their license; we cannot redistribute it).
2. **Validation data.** Put `organic_labeled.h5` and `41.h5` into `data/` (download: *link to be added with the data release*) or set `HM_DATA_DIR`.
3. **Train** (the defaults are the best recipe, see `python heatmap/train.py --help`):
```sh
export HM_DINOV3_REPO=./dinov3
python heatmap/train.py --out runs/hm --bb_path /path/to/dinov3_convnext_large_pretrain_lvd1689m-*.pth
```
It prints the AP on both sets every 2 epochs (199 s per epoch on an RTX 5070) and can be restarted with the same command (continues from `runs/hm/checkpoint.pth`).
4. **Score** the final checkpoint (scaled-GT AP, the number to report):
```sh
HM_GT_BOX_COEF=auto python heatmap/evaluate.py hm=heatmap:runs/hm/checkpoint.pth
```
5. **Detect peaks in your own data** (pyGID/NeXus h5 files, no labels needed):
```sh
python heatmap/predict.py --ckpt runs/hm/checkpoint.pth --out predictions my_scan.h5 --images
```
writes `boxes.csv` / `boxes.json` (polar-image pixels, score, ring or segment) and a picture per frame.

Options: `--bb dinov3_vith16plus` (ViT-H+, same AP, 2.4x the compute, needs a 40 GB GPU), `--frozen` (backbone not trained: faster, lower AP), `--bb simmim1` (our SimMIM Swin-L, `weights/swin_large_patch4_window12_384_22k.pth`, download: *link to be added*).
Experiments beyond this (other simulators, head sizes, more input channels, DINOv3 continued pretraining, analysis scripts) live on the `dev` branch.

## DINO detector

```sh
python main.py -c config/DINO/DINO_4scale_swin_ssl_nb.py --output_dir runs/dino_nb
HM_GT_BOX_COEF=3.0,3.0 python heatmap/evaluate.py dino=dino:runs/dino_nb/checkpoint.pth
```
Needs the SimMIM Swin-L weights in `weights/` and the two validation files in `data/` (see the config header), and the compiled ops below.
The config is the best DINO recipe so far (lr 4e-5, boxes centre +-1.5 sigma, 380 epochs). The AP printed *during* DINO training uses unscaled labels and is not the number to report; use the second command.
The ONNX export for mlgidDETECT is at the end of this file.

# Installation
 
  We test our models under ```python=3.12.7,pytorch=2.5.1,cuda=12.1```. Other versions might be available as well. Click the `Details` below for more details.
  1. Clone this repo
   ```sh
   git clone git@github.com:mlgid-project/DINO_GIWAXS.git
   cd DINO_GIWAXS
   ```

  2. Install Pytorch and torchvision

   Follow the instruction on https://pytorch.org/get-started/locally/.
   ```sh
   # an example:
   pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
   ```
  3. Install other needed packages
   ```sh
   pip install -r requirements.txt
   ```
  
  4. Compiling CUDA operators
   ```sh
   cd models/dino/ops
   python setup.py build install
   # unit test (should see all checking is True)
   python test.py
   cd ../../..
   ```



  We use the environment same to DAB-DETR and DN-DETR to run DINO. If you have run DN-DETR or DAB-DETR, you can skip this step.

  # Onnx-File Export
  To export an onnx file using the saved ```checkpoint.pth``` file run 

  ``` 
  python export.py --checkpoint /path-to-checkpoint/checkpoint.pth --output /output_dir/name.onnx
  ```
