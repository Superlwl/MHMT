import mmcv
import os.path as osp
import torch

# `PixelData` 是 MMEngine 中用于定义像素级标注或预测的数据结构。
# 请参考下面的MMEngine数据结构教程文件：
# https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/data_element.html#pixeldata

from mmengine.structures import PixelData

# `SegDataSample` 是在 MMSegmentation 中定义的不同组件之间的数据结构接口，
# 它包括 ground truth、语义分割的预测结果和预测逻辑。
# 详情请参考下面的 `SegDataSample` 教程文件：
# https://github.com/open-mmlab/mmsegmentation/blob/1.x/docs/en/advanced_guides/structures.md

from mmseg.structures import SegDataSample
from mmseg.visualization import SegLocalVisualizer

out_file = 'out_file_loveda'
save_dir = './lwl_work_dirs'

image = mmcv.imread(
    osp.join(
        osp.dirname(__file__),
        '/home/lyu4/lwl_wsp/mmsegmentation/data/LoveDA/img_dir/val/2525.png'
    ),
    'color')
sem_seg = mmcv.imread(
    osp.join(
        osp.dirname(__file__),
        '/home/lyu4/lwl_wsp/mmsegmentation/data/LoveDA/ann_dir/val/2525.png'  # noqa
    ),
    'unchanged')
sem_seg = torch.from_numpy(sem_seg)
gt_sem_seg_data = dict(data=sem_seg)
gt_sem_seg = PixelData(**gt_sem_seg_data)
data_sample = SegDataSample()
data_sample.gt_sem_seg = gt_sem_seg

seg_local_visualizer = SegLocalVisualizer(
    vis_backends=[dict(type='LocalVisBackend')],
    save_dir=save_dir)

# 数据集的元信息通常包括类名的 `classes` 和
# 用于可视化每个前景颜色的 `palette` 。
# 所有类名和调色板都在此文件中定义：
# https://github.com/open-mmlab/mmsegmentation/blob/1.x/mmseg/utils/class_names.py

seg_local_visualizer.dataset_meta = dict(
    classes=('background', 'building', 'road', 'water', 'barren', 'forest',
                 'agricultural'),
    palette=[[255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 0, 255],
                 [159, 129, 183], [0, 255, 0], [255, 195, 128]])

# 当`show=True`时，直接显示结果，
# 当 `show=False`时，结果将保存在本地文件夹中。

seg_local_visualizer.add_datasample(out_file, image,
                                    data_sample, show=False)