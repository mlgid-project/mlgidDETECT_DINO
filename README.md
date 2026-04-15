# DINO-GIWAXS
This is a GIWAXS oriented implementation of the paper "[DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection](https://arxiv.org/abs/2203.03605)". 

For more information on how to run the original implementation visit the original [GitHub repo](https://github.com/IDEA-Research/DINO).

Models trained using DINO-GIWAXS can be used in [mlgidDETECT](https://github.com/mlgid-project/mlgidDETECT) to perform inference. 
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
