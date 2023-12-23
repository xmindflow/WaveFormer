# Unlocking Fine-Grained Details with Wavelet-based High-Frequency Enhancement in Transformers <br> <span style="float: right"><sub><sup>MICCAI 2023 MLMI Workshop</sub></sup></span>

[![arXiv](https://img.shields.io/badge/arXiv-2308.13442-b31b1b.svg)](https://arxiv.org/abs/2308.13442)

Medical image segmentation is a critical task that plays a vital role in diagnosis, treatment planning, and disease monitoring. 
Accurate segmentation of anatomical structures and abnormalities from medical images can aid in the early detection and treatment of various diseases. 
In this paper, we address the local feature deficiency of the Transformer model by carefully re-designing the self-attention map to produce accurate dense prediction in medical images. 
To this end, we first apply the wavelet transformation to decompose the input feature map into low-frequency (LF) and high-frequency (HF) subbands. 
The LF segment is associated with coarse-grained features while the HF components preserve fine-grained features such as texture and edge information. 
Next, we reformulate the self-attention operation using the efficient Transformer to perform both spatial and context attention on top of the frequency representation. 
Furthermore, to intensify the importance of the boundary information, we impose an additional attention map by creating a Gaussian pyramid on top of the HF components. 
Moreover, we propose a multi-scale context enhancement block within skip connections to adaptively model inter-scale dependencies to overcome the semantic gap among stages of the encoder and decoder modules. 
Throughout comprehensive experiments, we demonstrate the effectiveness of our strategy on multi-organ and skin lesion segmentation benchmarks.

<p align="center">
  <b>F</b>requency <b>E</b>nhanced <b>T</b>ransformer (<b>FET</b>) model</em>
  <br/>
  <img width="600" alt="image" src="https://github.com/mindflow-institue/WaveFormer/assets/6207884/299ccd26-432d-406d-90d0-46b9255b4693"/>
  <br/>
  <br/>
  <img width="700" alt="image" src="https://github.com/mindflow-institue/WaveFormer/assets/6207884/4a5f688b-9271-40ca-bda1-12ed803965f7"/>
  <br>
  (a) <b>FET</b> block, (b) <b>M</b>ulti-<b>S</b>cale <b>C</b>ontext <b>E</b>nhancement (<b>MSCE</b>) module
</p>



## Citation
```bibtex
@inproceedings{azad2023unlocking,
  title={Unlocking Fine-Grained Details with Wavelet-based High-Frequency Enhancement in Transformers},
  author={Azad, Reza and Kazerouni, Amirhossein and Sulaiman, Alaa and Bozorgpour, Afshin and Aghdam, Ehsan Khodapanah and Jose, Abin, and Merhof, Dorit},
  maintitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  booktitle={Workshop on Machine Learning on Medical Imaging},
  year={2023}.
  organization={Springer}
}
```

## News
- Aug 18, 2023: Accepted in MICCAI 2023 MLMI Workshop! 🥳

## How to use

  ### Requirements
  
  - Ubuntu 16.04 or higher
  - CUDA 11.1 or higher
  - Python v3.7 or higher
  - Pytorch v1.7 or higher
  - Hardware Spec
    - A single GPU with 12GB memory or larger capacity (_we used RTX 3090_)

  ```
einops
h5py
imgaug
matplotlib
MedPy
numpy
opencv_python
pandas
PyWavelets
scipy
SimpleITK
tensorboardX
timm
torch
torchvision
tqdm
  ```
  `pip install -r requirements.txt`

  ### Model weights
  You can download the learned weights in the following.
   Dataset   | Model | download link 
  -----------|-------|----------------
   Synapse   | FET   | [[Download](https://drive.google.com/file/d/1rACU0c0JG9qaP9JjAYBwi5Yn3_7YAvu1/view)] 
  
  ### Training
  For the training, you must run the `train.py` with your desired arguments or you can use the simple written bash script file in `runs/run_tr_n01.sh`.
  You need to change variables and arguments respectively.
  Below, you can find a brief description of the arguments.

  ```conf
usage: train.py [-h] [--root_path ROOT_PATH] [--test_path TEST_PATH] [--dataset DATASET] [--dstr_fast DSTR_FAST] [--en_lnum EN_LNUM] [--br_lnum BR_LNUM] [--de_lnum DE_LNUM]
                [--compact COMPACT] [--continue_tr CONTINUE_TR] [--optimizer OPTIMIZER] [--dice_loss_weight DICE_LOSS_WEIGHT] [--list_dir LIST_DIR] [--num_classes NUM_CLASSES]
                [--output_dir OUTPUT_DIR] [--max_iterations MAX_ITERATIONS] [--max_epochs MAX_EPOCHS] [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS]
                [--eval_interval EVAL_INTERVAL] [--model_name MODEL_NAME] [--n_gpu N_GPU] [--bridge_layers BRIDGE_LAYERS] [--deterministic DETERMINISTIC] [--base_lr BASE_LR]
                [--img_size IMG_SIZE] [--z_spacing Z_SPACING] [--seed SEED] [--opts OPTS [OPTS ...]] [--zip] [--cache-mode {no,full,part}] [--resume RESUME]
                [--accumulation-steps ACCUMULATION_STEPS] [--use-checkpoint] [--amp-opt-level {O0,O1,O2}] [--tag TAG] [--eval] [--throughput]

optional arguments:
  -h, --help            show this help message and exit
  --root_path ROOT_PATH
                        root dir for data
  --test_path TEST_PATH
                        root dir for data
  --dataset DATASET     experiment_name
  --dstr_fast DSTR_FAST
                        SynapseDatasetFast: will load all data into RAM
  --en_lnum EN_LNUM     en_lnum: Laplacian layers (Pyramid) for the encoder
  --br_lnum BR_LNUM     br_lnum: Laplacian layers (Pyramid) for the bridge
  --de_lnum DE_LNUM     de_lnum: Laplacian layers (Pyramid) for the decoder
  --compact COMPACT     compact with 3 blocks instead of 4 blocks
  --continue_tr CONTINUE_TR
                        continue the training from the last saved epoch
  --optimizer OPTIMIZER
                        optimizer: [SGD, AdamW])
  --dice_loss_weight DICE_LOSS_WEIGHT
                        You need to determine <x> (default=0.6): => [loss = (1-x)*ce_loss + x*dice_loss]
  --list_dir LIST_DIR   list dir
  --num_classes NUM_CLASSES
                        output channel of the network
  --output_dir OUTPUT_DIR
                        output dir
  --max_iterations MAX_ITERATIONS
                        maximum epoch number to train
  --max_epochs MAX_EPOCHS
                        maximum epoch number to train
  --batch_size BATCH_SIZE
                        batch_size per GPU
  --num_workers NUM_WORKERS
                        num_workers
  --eval_interval EVAL_INTERVAL
                        eval_interval
  --model_name MODEL_NAME
                        model_name
  --n_gpu N_GPU         total gpu
  --bridge_layers BRIDGE_LAYERS
                        number of bridge layers
  --deterministic DETERMINISTIC
                        whether using deterministic training
  --base_lr BASE_LR     segmentation network learning rate
  --img_size IMG_SIZE   input patch size of network input
  --z_spacing Z_SPACING
                        z_spacing
  --seed SEED           random seed
  --opts OPTS [OPTS ...]
                        Modify config options by adding 'KEY VALUE' pairs.
  --zip                 use zipped dataset instead of folder dataset
  --cache-mode {no, full, part}
                        no: no cache, full: cache all data, part: sharding the dataset into nonoverlapping pieces and only cache one piece
  --resume RESUME       resume from checkpoint
  --accumulation-steps ACCUMULATION_STEPS
                        gradient accumulation steps
  --use-checkpoint      whether to use gradient checkpointing to save memory
  --amp-opt-level {O0,O1,O2}
                        mixed precision opt level, if O0, no amp is used
  --tag TAG             tag of experiment
  --eval                Perform evaluation only
  --throughput          Test throughput only
  ```
  
  ### Inference
  For inference, you need to run the `test.py`. Most of the parameters are like for the `train.py`. You can also use `runs/run_te_n01.sh` for an instance.
  
  To run with arbitrary weights you need to give the argument of `--weights_fpath` with the absolute path of the model weights file.
  
## Experiments

### Synapse Dataset
<p align="center">
  <img width="600" alt="Synapse images" src="https://github.com/mindflow-institue/WaveFormer/assets/6207884/6d2fa946-75ca-4a63-895c-ea2db633ff46">
  <img style="max-width:2020px" alt="Synapse results" src="https://github.com/mindflow-institue/WaveFormer/assets/6207884/6ac06b5e-a3bc-4de3-bd8b-d3ce2d0843f2">
</p>

### ISIC 2018 Dataset
<p align="center">
  <img style="width: 65%; float:left" alt="ISIC images" src="https://github.com/mindflow-institue/WaveFormer/assets/6207884/f0937aa3-3deb-4696-b38e-8501cf097a22">
  <img style="width: 34%;" alt="ISIC results" src="https://github.com/mindflow-institue/WaveFormer/assets/6207884/6d6a6433-af6d-4fab-b992-112b7b8dcf44">
</p>

## References
- DAEFormer [https://github.com/mindflow-institue/DAEFormer]
- ImageNetModel [https://github.com/YehLi/ImageNetModel]
