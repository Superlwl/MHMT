_base_ = [
    '../_base_/models/unetformer.py', '../_base_/datasets/loveda_512x512.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k_unetformer_loveda.py'
]
work_dir = '/home/lyu4/lwl_wsp/mmsegmentation/lwl_work_dirs/unetformer_loveda_512x512_80k'