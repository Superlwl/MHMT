_base_ = [
    '../_base_/models/mobilevitv2_mhmt.py', '../_base_/datasets/loveda_1024x1024.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
# crop_size=(140, 140)
work_dir = ''
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='MobileViT',
        image_size=(1024, 1024),
        input_channel=3,
        dims=[144, 192, 240],
        channels=[16, 32, 64, 64, 96, 128, 160, 640],
        num_classes=7),
    decode_head=dict(
        type='CSAHead',
        in_channels=(64, 96, 128, 640),
        in_index=[0, 1, 2, 3],
        channels=64,
        dropout=0.1,
        window_size=7,
        num_classes=7,
        resolutions=[(256, 256), (128, 128), (64, 64), (32, 32)],
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)))