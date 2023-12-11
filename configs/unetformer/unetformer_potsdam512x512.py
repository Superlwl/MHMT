_base_ = [
    '../_base_/models/unetformer.py', '../_base_/datasets/potsdam.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k_unetformer_loveda.py'
]
work_dir = '/home/lyu4/lwl_wsp/mmsegmentation/lwl_work_dirs/unetformer_potsdam_512x512_80k'
norm_cfg = dict(type='BN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4),
    decode_head=dict(
        type='UNetFormerHead',
        in_channels=(64, 128, 256, 512),
        in_index=[0, 1, 2, 3],
        channels=64,
        dropout=0.1,
        window_size=8,
        num_classes=6,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
