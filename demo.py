# Copyright (c) Liuhao Ge. All Rights Reserved.
r"""
Basic evaluation script for PyTorch
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os.path as osp
import torch
import cv2

from hand_shape_pose.config import cfg
from hand_shape_pose.model.shape_pose_network import ShapePoseNetwork
from hand_shape_pose.data.build import build_dataset

from hand_shape_pose.util.logger import setup_logger, get_logger_filename
from hand_shape_pose.util.miscellaneous import mkdir
from hand_shape_pose.util.vis import save_batch_image_with_mesh_joints, draw_2d_skeleton
from hand_shape_pose.util import renderer


def main():
    parser = argparse.ArgumentParser(description="3D Hand Shape and Pose Inference")
    parser.add_argument(
        "--config-file",
        default="configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = osp.join(cfg.EVAL.SAVE_DIR, args.config_file)
    mkdir(output_dir)
    logger = setup_logger("hand_shape_pose_inference", output_dir, filename='eval-' + get_logger_filename())
    logger.info(cfg)

    # 1. Load network model
    model = ShapePoseNetwork(cfg, output_dir)
    device = cfg.MODEL.DEVICE
    model.to(device)
    model.load_model(cfg)

    mesh_renderer = renderer.MeshRenderer(model.hand_tri.astype('uint32'))

    # 3. Inference
    model.eval()
    results_pose_cam_xyz = {}
    cpu_device = torch.device("cpu")
    cam_params = torch.Tensor([[923.6480, 923.7590, 640.0470, 362.3360]])
    pose_scales = torch.Tensor([5])
    pose_roots = torch.Tensor([[50.0, 50.0, 50.0]])

    cap = cv2.VideoCapture(0)
    while True:
        # img = cv2.imread("temp.jpg")
        _, img = cap.read()
        # images, cam_params, bboxes, pose_roots, pose_scales, image_ids = batch
        img = cv2.resize(img[:, 80: 560, :], (256, 256))
        # print(img.shape)
        # cv2.rectangle(img, (10, 22), (237, 253), (255, 0, 0))
        # # for 480x480: x, y, h, w = 2, 45, 420, 430
        # # for 256x256: x, y, h, w = 10, 22, 227, 231
        # cv2.imshow("", img)
        # cv2.waitKey()
        bboxes = torch.Tensor([[10, 22, 227, 231]])
        images = torch.Tensor([img,])
        images, cam_params, bboxes, pose_roots, pose_scales = \
            images.to(device), cam_params.to(device), bboxes.to(device), pose_roots.to(device), pose_scales.to(device)
        # print(cam_params, bboxes, pose_roots, pose_scales, "#"*20 ,sep="\n********\n")
        # img = images[0].cpu().numpy()

        with torch.no_grad():
            est_mesh_cam_xyz, est_pose_uv, est_pose_cam_xyz = \
                model(images, cam_params, bboxes, pose_roots, pose_scales)

            est_mesh_cam_xyz = [o.to(cpu_device) for o in est_mesh_cam_xyz]
            est_pose_uv = [o.to(cpu_device) for o in est_pose_uv]
            est_pose_cam_xyz = [o.to(cpu_device) for o in est_pose_cam_xyz]
        cv2.imshow("", draw_2d_skeleton(img, est_pose_uv[0].numpy()))
        cv2.waitKey(10)
        # [print(x) for x in est_pose_cam_xyz[0]]
        # print(pose_roots[0], len(est_pose_cam_xyz[0]))
        # import sys
        # sys.exit()

        # results_pose_cam_xyz.update({img_id.item(): result for img_id, result in zip(image_ids, est_pose_cam_xyz)})

        # if True:# i % cfg.EVAL.PRINT_FREQ == 0:
        #     # 4. evaluate pose estimation
        #     # avg_est_error = dataset_val.evaluate_pose(results_pose_cam_xyz, save_results=False)  # cm
        #     # msg = 'Evaluate: [{0}/{1}]\t' 'Average pose estimation error: {2:.2f} (mm)'.format(
        #     #     len(results_pose_cam_xyz), len(dataset_val), avg_est_error * 10.0)
        #     # logger.info(msg)

        #     # 5. visualize mesh and pose estimation
        #     if cfg.EVAL.SAVE_BATCH_IMAGES_PRED:
        #         file_name = '{}_{}.jpg'.format(osp.join(output_dir, 'pred'), "rara")
        #         logger.info("Saving image: {}".format(file_name))
        #         save_batch_image_with_mesh_joints(mesh_renderer, images.to(cpu_device), cam_params.to(cpu_device),
        #                                           bboxes.to(cpu_device), est_mesh_cam_xyz, est_pose_uv,
        #                                           est_pose_cam_xyz, file_name)

    # overall evaluate pose estimation
    assert len(results_pose_cam_xyz) == len(dataset_val), \
        "The number of estimation results (%d) is inconsistent with that of the ground truth (%d)." % \
        (len(results_pose_cam_xyz), len(dataset_val))

    avg_est_error = dataset_val.evaluate_pose(results_pose_cam_xyz, cfg.EVAL.SAVE_POSE_ESTIMATION, output_dir)  # cm
    logger.info("Overall:\tAverage pose estimation error: {0:.2f} (mm)".format(avg_est_error * 10.0))


if __name__ == "__main__":
    main()
