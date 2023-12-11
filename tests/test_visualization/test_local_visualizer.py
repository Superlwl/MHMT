# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile
from unittest import TestCase

import cv2
import mmcv
import numpy as np
import torch
from mmengine.structures import PixelData

from mmseg.structures import SegDataSample
from mmseg.visualization import SegLocalVisualizer


class TestSegLocalVisualizer(TestCase):

    def test_add_datasample(self):
        h = 1024
        w = 1024
        num_class = 7
        out_file = 'out_file'

        image = np.random.randint(0, 256, size=(h, w, 3)).astype('uint8')

        # test gt_sem_seg
        gt_sem_seg_data = dict(data=torch.randint(0, num_class, (1, h, w)))
        gt_sem_seg = PixelData(**gt_sem_seg_data)

        def test_add_datasample_forward(gt_sem_seg):
            data_sample = SegDataSample()
            data_sample.gt_sem_seg = gt_sem_seg

            with tempfile.TemporaryDirectory() as tmp_dir:
                seg_local_visualizer = SegLocalVisualizer(
                    vis_backends=[dict(type='LocalVisBackend')],
                    save_dir=tmp_dir)
                seg_local_visualizer.dataset_meta = dict(
                    classes=('background', 'foreground'),
                    palette=[[120, 120, 120], [6, 230, 230]])

                # test out_file
                seg_local_visualizer.add_datasample(out_file, image,
                                                    data_sample)

                assert os.path.exists(
                    osp.join(tmp_dir, 'vis_data', 'vis_image',
                             out_file + '_0.png'))
                drawn_img = cv2.imread(
                    osp.join(tmp_dir, 'vis_data', 'vis_image',
                             out_file + '_0.png'))
                assert drawn_img.shape == (h, w, 3)

                # test gt_instances and pred_instances
                pred_sem_seg_data = dict(
                    data=torch.randint(0, num_class, (1, h, w)))
                pred_sem_seg = PixelData(**pred_sem_seg_data)

                data_sample.pred_sem_seg = pred_sem_seg

                seg_local_visualizer.add_datasample(out_file, image,
                                                    data_sample)
                self._assert_image_and_shape(
                    osp.join(tmp_dir, 'vis_data', 'vis_image',
                             out_file + '_0.png'), (h, w * 2, 3))

                seg_local_visualizer.add_datasample(
                    out_file, image, data_sample, draw_gt=False)
                self._assert_image_and_shape(
                    osp.join(tmp_dir, 'vis_data', 'vis_image',
                             out_file + '_0.png'), (h, w, 3))

        if torch.cuda.is_available():
            test_add_datasample_forward(gt_sem_seg.cuda())
        test_add_datasample_forward(gt_sem_seg)

    def test_cityscapes_add_datasample(self):
        h = 1024
        w = 1024
        num_class = 7
        out_file = 'out_file_loveda'

        image = mmcv.imread(
            osp.join(
                osp.dirname(__file__),
                '../data/LoveDA/img_dir/val/2522.png'  # noqa
            ),
            'color')
        sem_seg = mmcv.imread(
            osp.join(
                osp.dirname(__file__),
                '../data/LoveDA/ann_dir/val/2522.png'  # noqa
            ),
            'unchanged')
        sem_seg = torch.unsqueeze(torch.from_numpy(sem_seg), 0)
        gt_sem_seg_data = dict(data=sem_seg)
        gt_sem_seg = PixelData(**gt_sem_seg_data)

        def test_cityscapes_add_datasample_forward(gt_sem_seg):
            data_sample = SegDataSample()
            data_sample.gt_sem_seg = gt_sem_seg

            with tempfile.TemporaryDirectory() as tmp_dir:
                seg_local_visualizer = SegLocalVisualizer(
                    vis_backends=[dict(type='LocalVisBackend')],
                    save_dir=tmp_dir)
                seg_local_visualizer.dataset_meta = dict(
                    classes=('background', 'building', 'road', 'water', 'barren', 'forest',
                 'agricultural'),
                    palette=[[255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 0, 255],
                 [159, 129, 183], [0, 255, 0], [255, 195, 128]])
                # test out_file
                seg_local_visualizer.add_datasample(
                    out_file,
                    image,
                    data_sample,
                    out_file=osp.join(tmp_dir, 'test.png'))
                self._assert_image_and_shape(
                    osp.join(tmp_dir, 'test.png'), (h, w, 3))

                # test gt_instances and pred_instances
                pred_sem_seg_data = dict(
                    data=torch.randint(0, num_class, (1, h, w)))
                pred_sem_seg = PixelData(**pred_sem_seg_data)

                data_sample.pred_sem_seg = pred_sem_seg

                # test draw prediction with gt
                seg_local_visualizer.add_datasample(out_file, image,
                                                    data_sample)
                self._assert_image_and_shape(
                    osp.join(tmp_dir, 'vis_data', 'vis_image',
                             out_file + '_0.png'), (h, w * 2, 3))
                # test draw prediction without gt
                seg_local_visualizer.add_datasample(
                    out_file, image, data_sample, draw_gt=False)
                self._assert_image_and_shape(
                    osp.join(tmp_dir, 'vis_data', 'vis_image',
                             out_file + '_0.png'), (h, w, 3))

        if torch.cuda.is_available():
            test_cityscapes_add_datasample_forward(gt_sem_seg.cuda())
        test_cityscapes_add_datasample_forward(gt_sem_seg)

    def _assert_image_and_shape(self, out_file, out_shape):
        assert os.path.exists(out_file)
        drawn_img = cv2.imread(out_file)
        assert drawn_img.shape == out_shape
