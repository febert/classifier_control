import os
import torchvision
import torch
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np

from classifier_control.classifier.utils.vis_utils import plot_graph

class Logger:
    def __init__(self, log_dir, n_logged_samples=10, summary_writer=None):
        self._log_dir = log_dir
        self._n_logged_samples = n_logged_samples
        if summary_writer is not None:
            self._summ_writer = summary_writer
        else:
            self._summ_writer = SummaryWriter(log_dir)

    def _loop_batch(self, fn, name, val, *argv, **kwargs):
        """Loops the logging function n times."""
        for log_idx in range(min(self._n_logged_samples, len(val))):
            name_i = os.path.join(name, "_%d" % log_idx)
            fn(name_i, val[log_idx], *argv, **kwargs)

    @staticmethod
    def _check_size(val, size):
        if isinstance(val, torch.Tensor) or isinstance(val, np.ndarray):
            assert len(val.shape) == size, "Size of tensor does not fit required size, {} vs {}".format(len(val.shape),
                                                                                                        size)
        elif isinstance(val, list):
            assert len(val[0].shape) == size - 1, "Size of list element does not fit required size, {} vs {}".format(
                len(val[0].shape), size - 1)
        else:
            raise NotImplementedError("Input type {} not supported for dimensionality check!".format(type(val)))
        if (val[0].shape[1] > 10000) or (val[0].shape[2] > 10000):
            raise ValueError("This might be a bit too much")

    def log_scalar(self, scalar, name, step, phase):
        self._summ_writer.add_scalar('{}_{}'.format(name, phase), scalar, step)

    def log_scalars(self, scalar_dict, group_name, step, phase):
        """Will log all scalars in the same plot."""
        self._summ_writer.add_scalars('{}_{}'.format(group_name, phase), scalar_dict, step)

    def log_images(self, image, name, step, phase):
        self._check_size(image, 4)  # [N, C, H, W]
        self._loop_batch(self._summ_writer.add_image, '{}_{}'.format(name, phase), image, step)

    def log_video(self, video_frames, name, step, phase):
        assert len(video_frames.shape) == 4, "Need [T, C, H, W] input tensor for single video logging!"
        if not isinstance(video_frames, torch.Tensor): video_frames = torch.tensor(video_frames)
        video_frames = torch.transpose(video_frames, 0, 1)  # tbX requires [C, T, H, W]
        video_frames = video_frames.unsqueeze(0)  # add an extra dimension to get grid of size 1
        self._summ_writer.add_video('{}_{}'.format(name, phase), video_frames, step)

    def log_videos(self, video_frames, name, step, phase, fps=3):
        assert len(video_frames.shape) == 5, "Need [N, T, C, H, W] input tensor for video logging!"
        video_frames = video_frames.unsqueeze(1)  # add an extra dimension after batch to get grid of size 1
        self._loop_batch(self._summ_writer.add_video, '{}_{}'.format(name, phase), video_frames, step, fps=fps)

    def log_image_grid(self, images, name, step, phase, nrow=8):
        assert len(images.shape) == 4, "Image grid logging requires input shape [batch, C, H, W]!"
        img_grid = torchvision.utils.make_grid(images, nrow=nrow)
        self.log_images(img_grid, '{}_{}'.format(name, phase), step)

    def log_video_grid(self, video_frames, name, step, phase, fps=3):
        assert len(video_frames.shape) == 5, "Need [N, T, C, H, W] input tensor for video logging!"
        self._summ_writer.add_video('{}_{}'.format(name, phase), video_frames, step, fps=fps)

    def log_figures(self, figure, name, step, phase):
        """figure: matplotlib.pyplot figure handle"""
        assert figure.shape[0] > 0, "Figure logging requires input shape [batch x figures]!"
        self._loop_batch(self._summ_writer.add_figure, '{}_{}'.format(name, phase), figure, step)

    def log_figure(self, figure, name, step, phase):
        """figure: matplotlib.pyplot figure handle"""
        self._summ_writer.add_figure('{}_{}'.format(name, phase), figure, step)

    def log_graph(self, array, name, step, phase):
        """figure: matplotlib.pyplot figure handle"""
        im = plot_graph(array)
        self._summ_writer.add_image('{}_{}'.format(name, phase), im, step)

    def dump_scalars(self, log_path=None):
        log_path = os.path.join(self._log_dir, "scalar_data.json") if log_path is None else log_path
        self._summ_writer.export_scalars_to_json(log_path)


# class TdistClassifierLogger(Logger):
#     def log_single_tdist_classifier_image(self, model_output, inputs, name, step, phase):
#
#         model_output.pos_pair






