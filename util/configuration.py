import os
import sys
import logging
import yaml
import cv2

class Config:
    def __init__(self, config_file = None, args = None):
        self.init_default()
        if config_file is not None:
            self.config_file = config_file
            self.load_config()
        if args is not None:
            self.load_args(args)
        self.check_cuda_support()
        self.set_logging_level()
    
    def init_default(self):
        self.GENERAL_DEBUG = False
        self.MODEL_ONNX_PATH = None
        self.MODEL_REDOWNLOAD = False
        self.MODEL_TYPE = 'faster_rcnn'
        self.MODEL_FORCE_CPU: False
        self.INPUT_IMGPATH = None
        self.INPUT_DATASET = None
        self.INPUT_LABELED = True
        self.GEO_PIXELPERANGSTROEM = 500
        self.GEO_RECIPROCAL_SHAPE = [1501,1501]
        self.GEO_QMAX = None
        self.PREPROCESSING_CUDA = False
        self.PREPROCESSING_QUAZIPOLAR = False
        self.PREPROCESSING_FLIPHORIZONTAL = False
        self.PREPROCESSING_POLAR_CONVERSION = True
        self.PREPROCESSING_NO_CONTRASTCORRECTION = False
        self.OUTPUT_FOLDER = './outputs/'
        self.OUTPUT_H5PATH = None


    def load_config(self):
        if not os.path.isfile(self.config_file):
            logging.error(f"Configuration file not found: {self.config_file}")
            sys.exit()

        try:        
            with open(self.config_file, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                logging.debug(f"Configuration successfully loaded from: {self.config_file}")
        except yaml.YAMLError as e:
            logging.error(f"YAML parsing error in config file: {e}")
            sys.exit()
        except Exception as e:
            logging.exception(f"Unexpected error while loading config: {e}")
            sys.exit()

        # Set attributes dynamically
        for section, settings in config.items():
            for key, value in settings.items():
                setattr(self, f"{section}_{key}", value)


    def load_args(self, args):
        if args.onnx_path:
            self.MODEL_ONNX_PATH = args.onnx_path
        if args.epoch:
            self.EVAL_EPOCH = args.epoch
        if args.output_folder:
            self.EVAL_OUTPUT_FOLDER = args.output_folder
        if args.input_dataset:
            self.INPUT_DATASET = args.input_dataset
        if args.image_path:
            self.INPUT_IMGPATH = args.image_path


    def check_cuda_support(self):
        if self.PREPROCESSING_CUDA and (cv2.cuda.getCudaEnabledDeviceCount() > 0):
            try:
                import cupy as cp
                logging.info("Using CUDA for preprocessing!")
                self.PREPROCESSING_CUDA = True
            except ImportError:
                self.PREPROCESSING_CUDA = False
                logging.info("Cupy not installed, fallback to CPU!")
                logging.info("CUDA support for preprocessing not available. Use the script 'setup_cuda.py' to install it.\n The inference might still run on the GPU though.")
        elif self.PREPROCESSING_CUDA:
            self.PREPROCESSING_CUDA = False
            logging.info("CUDA support for preprocessing not available. Use the script 'setup_cuda.py' to install it.\n The inference might still run on the GPU though.")
        else:
            self.PREPROCESSING_CUDA = False

    def set_logging_level(self):
        if self.GENERAL_DEBUG:
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.getLogger().setLevel(logging.INFO)