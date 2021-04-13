"""
Demo the feature-metric registration algorithm
Creator: Xiaoshui Huang
Date: 2021-04-13
"""
import os
import sys
import copy
import open3d
import torch
import torch.utils.data
import logging
import numpy as np
from model import PointNet, Decoder, SolveRegistration
import se_math.transforms as transforms

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

# visualize the point clouds
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    open3d.io.write_point_cloud("source_pre.ply", source_temp)
    open3d.visualization.draw_geometries([source_temp, target_temp])
    source_temp.transform(transformation)
    open3d.io.write_point_cloud("source.ply", source_temp)
    open3d.io.write_point_cloud("target.ply", target_temp)
    open3d.visualization.draw_geometries([source_temp, target_temp])


class Demo:
    def __init__(self):
        self.dim_k = 1024
        self.max_iter = 10  # max iteration time for IC algorithm
        self._loss_type = 1  # see. self.compute_loss()

    def create_model(self):
        # Encoder network: extract feature for every point. Nx1024
        ptnet = PointNet(dim_k=self.dim_k)
        # Decoder network: decode the feature into points, not used during the evaluation
        decoder = Decoder()
        # feature-metric ergistration (fmr) algorithm: estimate the transformation T
        fmr_solver = SolveRegistration(ptnet, decoder, isTest=True)
        return fmr_solver

    def evaluate(self, solver, p0, p1, device):
        solver.eval()
        with torch.no_grad():
            p0 = torch.tensor(p0,dtype=torch.float).to(device)  # template (1, N, 3)
            p1 = torch.tensor(p1,dtype=torch.float).to(device)  # source (1, M, 3)
            solver.estimate_t(p0, p1, self.max_iter)

            est_g = solver.g  # (1, 4, 4)
            g_hat = est_g.cpu().contiguous().view(4, 4)  # --> [1, 4, 4]

            return g_hat

def main(p0, p1, p0_pcd, p1_pcd):
    fmr = Demo()
    model = fmr.create_model()
    pretrained_path = "./result/fmr_model_7scene.pth"
    model.load_state_dict(torch.load(pretrained_path, map_location='cpu'))

    device = "cpu"
    model.to(device)
    T_est = fmr.evaluate(model, p0, p1, device)

    draw_registration_result(p1_pcd, p0_pcd, T_est)

if __name__ == '__main__':
    path0 = "./data/src0.ply"
    path1 = "./data/src1.ply"

    p0_src = open3d.io.read_point_cloud(path0)
    downpcd0 = p0_src.voxel_down_sample(voxel_size=0.05)
    p0 = np.asarray(downpcd0.points)
    p0 = np.expand_dims(p0,0)

    '''
    # generate random rotation sample
    trans = transforms.RandomTransformSE3(0.8, True)
    p0_src_tensor = torch.tensor((np.asarray(p0_src.points)),dtype=torch.float)
    p0_tensor_transformed = trans(p0_src_tensor)
    p1_src = p0_tensor_transformed.cpu().numpy()
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(p1_src)
    open3d.io.write_point_cloud(path1, pcd)
    '''

    p1 = open3d.io.read_point_cloud(path1)
    downpcd1 = p1.voxel_down_sample(voxel_size=0.05)
    p1 = np.asarray(downpcd1.points)
    p1 = np.expand_dims(p1, 0)
    main(p0, p1, downpcd0, downpcd1)
