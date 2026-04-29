# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

"""Configs."""
from fvcore.common.config import CfgNode

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()

# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #
_C.TRAIN = CfgNode()

# If True Train the model, else skip training.
_C.TRAIN.ENABLE = True

# Dataset.
_C.TRAIN.DATASET = "IC_dataset"

# Total mini-batch size.
_C.TRAIN.BATCH_SIZE = 16

# Evaluate model on test data every eval period epochs.
_C.TRAIN.EVAL_PERIOD = 1

# Save model checkpoint every checkpoint period epochs.
_C.TRAIN.CHECKPOINT_PERIOD = 1

# Resume training from the latest checkpoint in the output directory.
_C.TRAIN.AUTO_RESUME = False

# Path to the checkpoint to load the initial weight.
_C.TRAIN.CHECKPOINT_FILE_PATH = "vit_b_16_plus_240-laion400m_e32-699c4b84.pt" #"./configs/MViTv2_T_in1k.pyth" # ./configs/MViTv2_B_in1k.pyth

# If True, reset epochs when loading checkpoint.
_C.TRAIN.CHECKPOINT_EPOCH_RESET = True

# If True, use FP16 for activations
_C.TRAIN.MIXED_PRECISION = False

# If True, show a tqdm progress bar for local/single-process training.
_C.TRAIN.SHOW_PROGRESS = True

# If True, suppress Python warnings during local training runs.
_C.TRAIN.SUPPRESS_WARNINGS = True

# Early stopping patience (0 = disabled). Stop when val AUROC has not improved
# for this many consecutive eval epochs.
_C.TRAIN.EARLY_STOP_PATIENCE = 0

# ---------------------------------------------------------------------------- #
# Augmentation options.
# ---------------------------------------------------------------------------- #
_C.AUG = CfgNode()

# Number of repeated augmentations to used during training.
# If this is greater than 1, then the actual batch size is
# TRAIN.BATCH_SIZE * AUG.NUM_SAMPLE.
_C.AUG.NUM_SAMPLE = 1

# Not used if using randaug.
_C.AUG.COLOR_JITTER = 0.4

# RandAug parameters.
_C.AUG.AA_TYPE = "rand-m9-n6-mstd0.5-inc1"

# Interpolation method.
_C.AUG.INTERPOLATION = "bicubic"

# Probability of random erasing.
_C.AUG.RE_PROB = 0.25

# Random erasing mode.
_C.AUG.RE_MODE = "pixel"

# Random erase count.
_C.AUG.RE_COUNT = 1

# Do not random erase first (clean) augmentation split.
_C.AUG.RE_SPLIT = False

# ---------------------------------------------------------------------------- #
# MipUp options.
# ---------------------------------------------------------------------------- #
_C.MIXUP = CfgNode()

# Whether to use mixup.
_C.MIXUP.ENABLE = False

# Mixup alpha.
_C.MIXUP.ALPHA = 0.8

# Cutmix alpha.
_C.MIXUP.CUTMIX_ALPHA = 1.0

# Probability of performing mixup or cutmix when either/both is enabled.
_C.MIXUP.PROB = 1.0

# Probability of switching to cutmix when both mixup and cutmix enabled.
_C.MIXUP.SWITCH_PROB = 0.5

# Label smoothing.
_C.MIXUP.LABEL_SMOOTH_VALUE = 0.1
# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CfgNode()

# If True test the model, else skip the testing.
_C.TEST.ENABLE = True

# Dataset for testing.
_C.TEST.DATASET = "IC_dataset"

# Total mini-batch size
_C.TEST.BATCH_SIZE = 16

# Path to the checkpoint to load the initial weight.
_C.TEST.CHECKPOINT_FILE_PATH = ""

# If True, convert 3D conv weights to 2D.
_C.TEST.CHECKPOINT_SQUEEZE_TEMPORAL = True

# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()

# Model name
_C.MODEL.MODEL_NAME = "MViT"

# The number of classes to predict for the model.
_C.MODEL.NUM_CLASSES = 1

# Loss function.
_C.MODEL.LOSS_FUNC = "cross_entropy"

# Active InCTRL architecture.
_C.MODEL.ACTIVE_MODEL = "InCTRLAdapt"

# Dropout rate before final projection in the backbone.
_C.MODEL.DROPOUT_RATE = 0.0

# Activation layer for the output head.
_C.MODEL.HEAD_ACT = "softmax"

# Activation checkpointing enabled or not to save GPU memory.
_C.MODEL.ACT_CHECKPOINT = False


# -----------------------------------------------------------------------------
# InCTRL PQA Lite options
# -----------------------------------------------------------------------------
_C.VISUAL_ADAPTER = CfgNode()
_C.VISUAL_ADAPTER.ENABLE = True
_C.VISUAL_ADAPTER.REDUCTION = 4
_C.VISUAL_ADAPTER.ZERO_INIT = True

_C.TEXT_BRANCH = CfgNode()
_C.TEXT_BRANCH.ENABLE = True
_C.TEXT_BRANCH.TYPE = "adaptclip_prompt"
_C.TEXT_BRANCH.TEMPLATES = [
    "a photo of a normal object.",
    "a photo of a damaged object.",
]
_C.TEXT_BRANCH.LOGIT_SCALE = 100.0
_C.TEXT_BRANCH.N_CTX = 12
_C.TEXT_BRANCH.NORMAL_SUFFIX = "normal object."
_C.TEXT_BRANCH.ABNORMAL_SUFFIX = "damaged object."
_C.TEXT_BRANCH.CTX_INIT_STD = 0.02

_C.PQA = CfgNode()
_C.PQA.ENABLE = True
_C.PQA.ENABLE_SEG_HEAD = True
_C.PQA.PATCH_LAYERS = [7, 9, 11]
_C.PQA.CONTEXT_BETA = 1.0
_C.PQA.HIDDEN_DIM = 128
_C.PQA.GLOBAL_TOPK = 10
_C.PQA.NUM_LAYERS = 3  # should match len(PATCH_LAYERS)

_C.FUSION = CfgNode()
_C.FUSION.IMAGE_WEIGHT = 0.35
_C.FUSION.PATCH_WEIGHT = 0.25
_C.FUSION.PQA_WEIGHT = 0.25
_C.FUSION.TEXT_WEIGHT = 0.15
_C.FUSION.MAP_RES_WEIGHT = 0.4
_C.FUSION.MAP_PQA_WEIGHT = 0.4
_C.FUSION.MAP_TEXT_WEIGHT = 0.2
_C.FUSION.PIXEL_FUSION = 'weighted'  # weighted, average_mean, or harmonic_mean
_C.FUSION.ALIGN_FUSION = 'harmonic_mean'  # how to fuse align scores across layers
_C.FUSION.IMAGE_PIXEL_COUPLING = True  # harmonic_mean of image_score + anomaly_map_max
_C.FUSION.VISUAL_WEIGHT = 0.0  # weight for VA visual-text branch in score fusion
_C.FUSION.USE_VISUAL_BRANCH = True  # enable VA visual-text branch (requires VISUAL_ADAPTER.ENABLE)
_C.FUSION.MAP_VISUAL_WEIGHT = 0.1  # weight for visual_map in pixel fusion (only if VISUAL_MASK_WEIGHT > 0)
_C.FUSION.SCORE_OUTPUT = 'auto'  # auto, final_score, or coupled_score

_C.LOSS = CfgNode()
_C.LOSS.IMAGE_WEIGHT = 1.0
_C.LOSS.PQA_WEIGHT = 0.5
_C.LOSS.MASK_WEIGHT = 1.0
_C.LOSS.TEXT_WEIGHT = 0.0
_C.LOSS.TEXT_MASK_WEIGHT = 0.0
_C.LOSS.VISUAL_WEIGHT = 0.0  # CE loss weight for VA visual-text branch
_C.LOSS.VISUAL_MASK_WEIGHT = 0.0  # segmentation loss weight for VA visual map


# -----------------------------------------------------------------------------
# MViT options
# -----------------------------------------------------------------------------
_C.MVIT = CfgNode()

# Options include `conv`, `max`.
_C.MVIT.MODE = "conv"

# If True, perform pool before projection in attention.
_C.MVIT.POOL_FIRST = False

# If True, use cls embed in the network, otherwise don't use cls_embed in transformer.
_C.MVIT.CLS_EMBED_ON = False

# Kernel size for patchtification.
_C.MVIT.PATCH_KERNEL = [7, 7]

# Stride size for patchtification.
_C.MVIT.PATCH_STRIDE = [4, 4]

# Padding size for patchtification.
_C.MVIT.PATCH_PADDING = [3, 3]

# Base embedding dimension for the transformer.
_C.MVIT.EMBED_DIM = 96

# Base num of heads for the transformer.
_C.MVIT.NUM_HEADS = 1

# Dimension reduction ratio for the MLP layers.
_C.MVIT.MLP_RATIO = 4.0

# If use, use bias term in attention fc layers.
_C.MVIT.QKV_BIAS = True

# Drop path rate for the tranfomer.
_C.MVIT.DROPPATH_RATE = 0.1

# Depth of the transformer.
_C.MVIT.DEPTH = 16

# Dimension multiplication at layer i. If 2.0 is used, then the next block will increase
# the dimension by 2 times. Format: [depth_i: mul_dim_ratio]
_C.MVIT.DIM_MUL = []

# Head number multiplication at layer i. If 2.0 is used, then the next block will
# increase the number of heads by 2 times. Format: [depth_i: head_mul_ratio]
_C.MVIT.HEAD_MUL = []

# Stride size for the Pool KV at layer i.
# Format: [[i, stride_t_i, stride_h_i, stride_w_i], ...,]
_C.MVIT.POOL_KV_STRIDE = None

# Initial stride size for KV at layer 1. The stride size will be further reduced with
# the raio of MVIT.DIM_MUL. If will overwrite MVIT.POOL_KV_STRIDE if not None.
_C.MVIT.POOL_KV_STRIDE_ADAPTIVE = None

# Stride size for the Pool Q at layer i.
# Format: [[i, stride_t_i, stride_h_i, stride_w_i], ...,]
_C.MVIT.POOL_Q_STRIDE = []

# Kernel size for Q, K, V pooling.
_C.MVIT.POOL_KVQ_KERNEL = (3, 3)

# If True, perform no decay on positional embedding and cls embedding.
_C.MVIT.ZERO_DECAY_POS_CLS = False

# If True, use absolute positional embedding.
_C.MVIT.USE_ABS_POS = False

# If True, use relative positional embedding for spatial dimentions
_C.MVIT.REL_POS_SPATIAL = True

# If True, init rel with zero
_C.MVIT.REL_POS_ZERO_INIT = False

# If True, using Residual Pooling connection
_C.MVIT.RESIDUAL_POOLING = True

# Dim mul in qkv linear layers of attention block instead of MLP
_C.MVIT.DIM_MUL_IN_ATT = True


# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()

# The path to the data directory.
_C.DATA.PATH_TO_DATA_DIR = ""

# If a imdb have been dumpped to a local file with the following format:
# `{"im_path": im_path, "class": cont_id}`
# then we can skip the construction of imdb and load it from the local file.
_C.DATA.PATH_TO_PRELOAD_IMDB = ""

# The mean value of pixels across the R G B channels.
_C.DATA.MEAN = [0.485, 0.456, 0.406]
# List of input frame channel dimensions.

# The std value of pixels across the R G B channels.
_C.DATA.STD = [0.229, 0.224, 0.225]

# The spatial crop size for training.
_C.DATA.TRAIN_CROP_SIZE = 224

# The spatial crop size for testing.
_C.DATA.TEST_CROP_SIZE = 224

# Crop ratio for for testing. Default is 224/256.
_C.DATA.VAL_CROP_RATIO = 0.875

# If combine train/val split as training for in21k
_C.DATA.IN22K_TRAINVAL = False

# If not None, use IN1k as val split when training in21k
_C.DATA.IN22k_VAL_IN1K = ""

# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()

# Base learning rate.
_C.SOLVER.BASE_LR = 0.00025

# Learning rate policy (see utils/lr_policy.py for options and examples).
_C.SOLVER.LR_POLICY = "cosine"

# Final learning rates for 'cosine' policy.
_C.SOLVER.COSINE_END_LR = 1e-6

# Step size for 'exp' and 'cos' policies (in epochs).
_C.SOLVER.STEP_SIZE = 1

# Steps for 'steps_' policies (in epochs).
_C.SOLVER.STEPS = []

# Learning rates for 'steps_' policies.
_C.SOLVER.LRS = []

# Maximal number of epochs.
_C.SOLVER.MAX_EPOCH = 400

# Momentum.
_C.SOLVER.MOMENTUM = 0.9

# Momentum dampening.
_C.SOLVER.DAMPENING = 0.0

# Nesterov momentum.
_C.SOLVER.NESTEROV = True

# L2 regularization.
_C.SOLVER.WEIGHT_DECAY = 0.05

# Start the warm up from SOLVER.BASE_LR * SOLVER.WARMUP_FACTOR.
_C.SOLVER.WARMUP_FACTOR = 0.1

# Gradually warm up the SOLVER.BASE_LR over this number of epochs.
_C.SOLVER.WARMUP_EPOCHS = 70.0

# The start learning rate of the warm up.
_C.SOLVER.WARMUP_START_LR = 1e-8

# Optimization method.
_C.SOLVER.OPTIMIZING_METHOD = "sgd"

# Base learning rate is linearly scaled with NUM_SHARDS.
_C.SOLVER.BASE_LR_SCALE_NUM_SHARDS = False

# If True, start from the peak cosine learning rate after warm up.
_C.SOLVER.COSINE_AFTER_WARMUP = True

# Minimum learning rate for cosine schedule in local training.
_C.SOLVER.COSINE_MIN_LR = 1e-5

# If True, perform no weight decay on parameter with one dimension (bias term, etc).
_C.SOLVER.ZERO_WD_1D_PARAM = True

# Clip gradient at this value before optimizer update
_C.SOLVER.CLIP_GRAD_VAL = None

# Clip gradient at this norm before optimizer update
_C.SOLVER.CLIP_GRAD_L2NORM = None

# The layer-wise decay of learning rate. Set to 1. to disable.
_C.SOLVER.LAYER_DECAY = 1.0

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# Number of GPUs to use (applies to both training and testing).
_C.NUM_GPUS = 1

# Number of machine to use for the job.
_C.NUM_SHARDS = 1

# The index of the current machine.
_C.SHARD_ID = 0

# Output basedir.
_C.OUTPUT_DIR = "./tmp"

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
_C.RNG_SEED = 10

# Log period in iters.
_C.LOG_PERIOD = 10

# If True, log the model info.
_C.LOG_MODEL_INFO = True

# Distributed backend.
_C.DIST_BACKEND = "nccl"

_C.local_rank = 0

_C.normal_json_path = './datasets/AD_json/hyperkvasir_normal.json'

_C.outlier_json_path = './datasets/AD_json/hyperkvasir_outlier.json'

_C.val_normal_json_path = './datasets/AD_json/elpv_normal.json'

_C.val_outlier_json_path = './datasets/AD_json/elpv_outlier.json'

_C.model = 'ViT-B-16'
_C.pretrained = None

_C.shot = 2
_C.image_size = 240

_C.few_shot_dir = "./visa"


# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CfgNode()

_C.DATA_LOADER.data_path = "./data"

# Number of data loader workers per training process.
_C.DATA_LOADER.NUM_WORKERS = 2

# Load data to pinned host memory.
_C.DATA_LOADER.PIN_MEMORY = True



def assert_and_infer_cfg(cfg):
    # TRAIN assertions.
    assert cfg.NUM_GPUS == 0 or cfg.TRAIN.BATCH_SIZE % cfg.NUM_GPUS == 0

    # TEST assertions.
    assert cfg.NUM_GPUS == 0 or cfg.TEST.BATCH_SIZE % cfg.NUM_GPUS == 0

    # Execute LR scaling by num_shards.
    if cfg.SOLVER.BASE_LR_SCALE_NUM_SHARDS:
        cfg.SOLVER.BASE_LR *= cfg.NUM_SHARDS
        cfg.SOLVER.WARMUP_START_LR *= cfg.NUM_SHARDS
        cfg.SOLVER.COSINE_END_LR *= cfg.NUM_SHARDS

    # General assertions.
    assert cfg.SHARD_ID < cfg.NUM_SHARDS
    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()
