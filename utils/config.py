import torch.nn as nn
import ml_collections



def get_sweep():
    sweep_configuration = {
        "method": "bayes",
        "name": "beta_gamma_sweep",
        "metric": {"goal": "maximize", "name": "val/target_accuracy"},
        "parameters": {

            "eval_batch_size": {"values": [5]},
            "shift_size": {"values": [0]},
            "weight_decay": {"values": [0.9]},
            "decay_type": {"values": ["cosine"]},
            "warmup_steps": {"values": [500]},
            "max_grad_norm": {"values": [1.0]},
            "train_batch_size": {"values": [10]},
            "learning_rate": {"values": [3e-3]},
            "momentum": {"values": [0.9]},
            "beta": {"values": [0.02]},
            "gamma": {"values": [0.0002]},
            "pretrained_dir": {"values":['/AI/UVT/checkpoints/swin_base_patch4_window7_224.pth']},
            "num_steps": {"values":[6000]},
            "dataset": {"values":['office']},
            "source_list": {"values":['data/office/amazon_list.txt']},
            "target_list": {"values":['data/office/webcam_list.txt']},
            "test_list_source": {"values":['data/office/amazon_list.txt']},
            "test_list_target": {"values":['data/office/webcam_list.txt']},

            "img_size": {"values": [224]},
            "patch_size": {"values": [4]},
            "in_chans": {"values":[3]},
            "num_classes": {"values":[31]},
            "embed_dim": {"values": [128]},
            "depths": {"values":[[2, 2, 18, 2]]},
            "num_heads": {"values":[[4, 8, 16, 32]]},
            "window_size": {"values":[7]},
            "mlp_ratio": {"values":[4.0]},
            "qkv_bias": {"values":[True]},
            "qk_scale": {"values":[None]},
            "drop_rate": {"values":[0.0]},
            "attn_drop_rate": {"values":[0.0]},
            "drop_path_rate": {"values":[0.5]},
            "ape": {"values":[False]},
            "patch_norm": {"values":[True]},
            "use_checkpoint": {"values":[False]},
            "fused_window_process": {"values":[False]},

        },
    }

    return sweep_configuration


def get_swin_tiny():
    config = ml_collections.ConfigDict()
    config.IMG_SIZE=224
    config.PATCH_SIZE=4
    config.SHIFT_SIZE=0
    config.IN_CHANS=3
    config.NUM_CLASSES=31
    config.EMBED_DIM=96
    config.DEPTHS=[2, 2, 6, 2]
    config.NUM_HEADS=[3, 6, 12, 24]
    config.WINDOW_SIZE=7
    config.MLP_RATIO=4.
    config.QKV_BIAS=True
    config.QK_SCALE=None
    config.DROP_RATE=0.
    config.ATTN_DROP_RATE=0.
    config.DROP_PATH_RATE=0.2
    config.NORM_LAYER=nn.LayerNorm
    config.APE=False
    config.PATCH_NORM=True
    config.USE_CHECKPOINT=False
    config.FUSED_WINDOW_PROCESS=False
    return config


def get_swin_small():
    config = ml_collections.ConfigDict()
    config.IMG_SIZE=224
    config.SHIFT_SIZE=0
    config.PATCH_SIZE=4
    config.IN_CHANS=3
    config.NUM_CLASSES=31
    config.EMBED_DIM=96
    config.DEPTHS=[2, 2, 18, 2]
    config.NUM_HEADS=[3, 6, 12, 24]
    config.WINDOW_SIZE=7
    config.MLP_RATIO=4.
    config.QKV_BIAS=True
    config.QK_SCALE=None
    config.DROP_RATE=0.
    config.ATTN_DROP_RATE=0.
    config.DROP_PATH_RATE=0.3
    config.NORM_LAYER=nn.LayerNorm
    config.APE=False
    config.PATCH_NORM=True
    config.USE_CHECKPOINT=False
    config.FUSED_WINDOW_PROCESS=False
    return config


def get_swin_base():
    config = ml_collections.ConfigDict()
    config.IMG_SIZE=224
    config.SHIFT_SIZE=0
    config.PATCH_SIZE=4
    config.IN_CHANS=3
    config.NUM_CLASSES=31
    config.EMBED_DIM=128
    config.DEPTHS=[2, 2, 18, 2]
    config.NUM_HEADS=[4, 8, 16, 32]
    config.WINDOW_SIZE=7
    config.MLP_RATIO=4.
    config.QKV_BIAS=True
    config.QK_SCALE=None
    config.DROP_RATE=0.
    config.ATTN_DROP_RATE=0.
    config.DROP_PATH_RATE=0.5
    config.NORM_LAYER=nn.LayerNorm
    config.APE=False
    config.PATCH_NORM=True
    config.USE_CHECKPOINT=False
    config.FUSED_WINDOW_PROCESS=False
    return config


def get_swin_large():
    config = ml_collections.ConfigDict()
    config.IMG_SIZE=224
    config.PATCH_SIZE=4
    config.SHIFT_SIZE=0
    config.IN_CHANS=3
    config.NUM_CLASSES=31
    config.EMBED_DIM=192
    config.DEPTHS=[2, 2, 18, 2]
    config.NUM_HEADS=[6, 12, 24, 48]
    config.WINDOW_SIZE=7
    config.MLP_RATIO=4.
    config.QKV_BIAS=True
    config.QK_SCALE=None
    config.DROP_RATE=0.
    config.ATTN_DROP_RATE=0.
    config.DROP_PATH_RATE=0.2
    config.NORM_LAYER=nn.LayerNorm
    config.APE=False
    config.PATCH_NORM=True
    config.USE_CHECKPOINT=False
    config.FUSED_WINDOW_PROCESS=False
    return config



def get_transadapter_base():
    config = ml_collections.ConfigDict()
    config.IMG_SIZE=224
    config.PATCH_SIZE=4
    config.SHIFT_SIZE=0
    config.IN_CHANS=3
    config.NUM_CLASSES=31
    config.EMBED_DIM=128
    config.DEPTHS=[2, 2, 18, 2]
    config.NUM_HEADS=[4, 8, 16, 32]
    config.WINDOW_SIZE=7
    config.MLP_RATIO=4.
    config.QKV_BIAS=True
    config.QK_SCALE=None
    config.DROP_RATE=0.
    config.ATTN_DROP_RATE=0.
    config.DROP_PATH_RATE=0.5
    config.NORM_LAYER=nn.LayerNorm
    config.APE=False
    config.PATCH_NORM=True
    config.USE_CHECKPOINT=False
    config.FUSED_WINDOW_PROCESS=True
    return config