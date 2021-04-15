"""
This file is used to load 3D point cloud for network training
Creator: Xiaoshui Huang
Date: 2020-06-19
"""
import numpy
import torch.utils.data
import os
import glob
import copy
import six
import numpy as np
import torch
import torch.utils.data
import torchvision

import se_math.se3 as se3
import se_math.so3 as so3
import se_math.mesh as mesh
import se_math.transforms as transforms

"""
The following three functions are defined for getting data from specific database 
"""


# find the total class names and its corresponding index from a folder
# (see the data storage structure of modelnet40)
def find_classes(root):
    """ find ${root}/${class}/* """
    classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


# get the indexes from given class names
def classes_to_cinfo(classes):
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


# get the whole 3D point cloud paths for a given class
def glob_dataset(root, class_to_idx, ptns):
    """ glob ${root}/${class}/${ptns[i]} """
    root = os.path.expanduser(root)
    samples = []

    # loop all the folderName (class name) to find the class in class_to_idx
    for target in sorted(os.listdir(root)):
        d = os.path.join(root, target)
        if not os.path.isdir(d):
            continue
        # check if it is the class we want
        target_idx = class_to_idx.get(target)
        if target_idx is None:
            continue
        # to find the all point cloud paths in the class folder
        for i, ptn in enumerate(ptns):
            gptn = os.path.join(d, ptn)
            names = glob.glob(gptn)
            for path in sorted(names):
                item = (path, target_idx)
                samples.append(item)
    return samples


# a general class for obtaining the 3D point cloud data from a database
class PointCloudDataset(torch.utils.data.Dataset):
    """ glob ${rootdir}/${classes}/${pattern}
    """

    def __init__(self, rootdir, pattern, fileloader, transform=None, classinfo=None):
        super().__init__()

        if isinstance(pattern, six.string_types):
            pattern = [pattern]

        # find all the class names
        if classinfo is not None:
            classes, class_to_idx = classinfo
        else:
            classes, class_to_idx = find_classes(rootdir)

        # get all the 3D point cloud paths for the class of class_to_idx
        samples = glob_dataset(rootdir, class_to_idx, pattern)
        if not samples:
            raise RuntimeError("Empty: rootdir={}, pattern(s)={}".format(rootdir, pattern))

        self.fileloader = fileloader
        self.transform = transform

        self.classes = classes
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        define the getitem function for Dataloader of torch
        load a 3D point cloud by using a path index
        :param index:
        :return:
        """
        path, target = self.samples[index]
        sample = self.fileloader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def split(self, rate):
        """ dateset -> dataset1, dataset2. s.t.
            len(dataset1) = rate * len(dataset),
            len(dataset2) = (1-rate) * len(dataset)
        """
        orig_size = len(self)
        select = np.zeros(orig_size, dtype=int)
        csize = np.zeros(len(self.classes), dtype=int)

        for i in range(orig_size):
            _, target = self.samples[i]
            csize[target] += 1
        dsize = (csize * rate).astype(int)
        for i in range(orig_size):
            _, target = self.samples[i]
            if dsize[target] > 0:
                select[i] = 1
                dsize[target] -= 1

        dataset1 = copy.deepcopy(self)
        dataset2 = copy.deepcopy(self)

        samples1 = list(map(lambda i: dataset1.samples[i], np.where(select == 1)[0]))
        samples2 = list(map(lambda i: dataset2.samples[i], np.where(select == 0)[0]))

        dataset1.samples = samples1
        dataset2.samples = samples2
        return dataset1, dataset2


class ModelNet(PointCloudDataset):
    """ [Princeton ModelNet](http://modelnet.cs.princeton.edu/) """

    def __init__(self, dataset_path, train=1, transform=None, classinfo=None, is_uniform_sampling=False):
        # if you would like to uniformly sampled points from mesh, use this function below
        if is_uniform_sampling:
            loader = mesh.offread_uniformed # used uniformly sampled points.
        else:
            loader = mesh.offread # use the original vertex in the mesh file
        if train > 0:
            pattern = 'train/*.off'
        elif train == 0:
            pattern = 'test/*.off'
        else:
            pattern = ['train/*.off', 'test/*.off']
        super().__init__(dataset_path, pattern, loader, transform, classinfo)


class ShapeNet2(PointCloudDataset):
    """ [ShapeNet](https://www.shapenet.org/) v2 """

    def __init__(self, dataset_path, transform=None, classinfo=None):
        loader = mesh.objread
        pattern = '*/models/model_normalized.obj'
        super().__init__(dataset_path, pattern, loader, transform, classinfo)


class Scene7(PointCloudDataset):
    """ [Scene7 PointCloud](https://github.com/XiaoshuiHuang/fmr) """

    def __init__(self, dataset_path, train=1, transform=None, classinfo=None):
        loader = mesh.plyread
        if train > 0:
            pattern = '*.ply'
        elif train == 0:
            pattern = '*.ply'
        else:
            pattern = ['*.ply', '*.ply']
        super().__init__(dataset_path, pattern, loader, transform, classinfo)


class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, rigid_transform, source_modifier=None, template_modifier=None):
        self.dataset = dataset
        self.rigid_transform = rigid_transform
        self.source_modifier = source_modifier
        self.template_modifier = template_modifier

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pm, _ = self.dataset[index]
        if self.source_modifier is not None:
            p_ = self.source_modifier(pm)
            p1 = self.rigid_transform(p_)
        else:
            p1 = self.rigid_transform(pm)
        igt = self.rigid_transform.igt

        if self.template_modifier is not None:
            p0 = self.template_modifier(pm)
        else:
            p0 = pm

        # p0: template, p1: source, igt: transform matrix from p0 to p1
        return p0, p1, igt


class TransformedFixedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, perturbation):
        self.dataset = dataset
        self.perturbation = numpy.array(perturbation)  # twist (len(dataset), 6)

    def do_transform(self, p0, x):
        # p0: [N, 3]
        # x: [1, 6]
        # x: rotation and translation
        w = x[:, 0:3]
        q = x[:, 3:6]
        R = so3.exp(w).to(p0)  # [1, 3, 3]
        g = torch.zeros(1, 4, 4)
        g[:, 3, 3] = 1
        g[:, 0:3, 0:3] = R  # rotation
        g[:, 0:3, 3] = q  # translation
        p1 = se3.transform(g, p0)
        igt = g.squeeze(0)  # igt: p0 -> p1
        return p1, igt

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        twist = torch.from_numpy(numpy.array(self.perturbation[index])).contiguous().view(1, 6)
        pm, _ = self.dataset[index]
        x = twist.to(pm)
        p1, igt = self.do_transform(pm, x)
        p0 = pm
        # p0: template, p1: source, igt: transform matrix from p0 to p1
        return p0, p1, igt


def get_categories(args):
    cinfo = None
    if args.categoryfile:
        categories = [line.rstrip('\n') for line in open(args.categoryfile)]
        categories.sort()
        c_to_idx = {categories[i]: i for i in range(len(categories))}
        cinfo = (categories, c_to_idx)
    return cinfo


# global dataset function, could call to get dataset
def get_datasets(args):
    if args.dataset_type == 'modelnet':
        # download modelnet40 for training
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = os.path.join(BASE_DIR, 'ModelNet40')
        if not os.path.exists(DATA_DIR):
            www = 'http://modelnet.cs.princeton.edu/ModelNet40.zip'
            zipfile = os.path.basename(www)
            os.system('wget %s; unzip %s' % (www, zipfile))
            os.system('mv %s %s' % (zipfile[:-4], BASE_DIR))
            os.system('rm %s' % (zipfile))
        if not os.path.exists(DATA_DIR):
            exit(
                "Please download ModelNET40 and put it in the data folder, the download link is http://modelnet.cs.princeton.edu/ModelNet40.zip")

        if args.mode == 'train':
            # set path and category file for training
            args.dataset_path = './data/ModelNet40'
            args.categoryfile = './data/categories/modelnet40_half1.txt'
            cinfo = get_categories(args)
            transform = torchvision.transforms.Compose([ \
                transforms.Mesh2Points(), \
                transforms.OnUnitCube(), \
                transforms.Resampler(args.num_points), \
                ])
            traindata = ModelNet(args.dataset_path, train=1, transform=transform, classinfo=cinfo, is_uniform_sampling=args.uniformsampling)
            testdata = ModelNet(args.dataset_path, train=0, transform=transform, classinfo=cinfo, is_uniform_sampling=args.uniformsampling)
            mag_randomly = True
            trainset = TransformedDataset(traindata, transforms.RandomTransformSE3(args.mag, mag_randomly))
            testset = TransformedDataset(testdata, transforms.RandomTransformSE3(args.mag, mag_randomly))
            return trainset, testset
        else:
            # set path and category file for test
            args.dataset_path = './data/ModelNet40'
            args.categoryfile = './data/categories/modelnet40_half1.txt'
            cinfo = get_categories(args)

            # get the ground truth perturbation
            perturbations = None
            if args.perturbations:
                perturbations = numpy.loadtxt(args.perturbations, delimiter=',')

            transform = torchvision.transforms.Compose([transforms.Mesh2Points(), transforms.OnUnitCube()])

            testdata = ModelNet(args.dataset_path, train=0, transform=transform, classinfo=cinfo, is_uniform_sampling=args.uniformsampling)
            testset = TransformedFixedDataset(testdata, perturbations)
            return testset

    elif args.dataset_type == '7scene':
        if args.mode == 'train':
            # set path and category file for training
            args.dataset_path = './data/7scene'
            args.categoryfile = './data/categories/7scene_train.txt'
            cinfo = get_categories(args)

            transform = torchvision.transforms.Compose([ \
                transforms.Mesh2Points(), \
                transforms.OnUnitCube(), \
                transforms.Resampler(args.num_points)])

            dataset = Scene7(args.dataset_path, transform=transform, classinfo=cinfo)
            traindata, testdata = dataset.split(0.8)

            mag_randomly = True
            trainset = TransformedDataset(traindata, transforms.RandomTransformSE3(args.mag, mag_randomly))
            testset = TransformedDataset(testdata, transforms.RandomTransformSE3(args.mag, mag_randomly))
            return trainset, testset
        else:
            # set path and category file for testing
            args.dataset_path = './data/7scene'
            args.categoryfile = './data/categories/7scene_test.txt'
            cinfo = get_categories(args)

            transform = torchvision.transforms.Compose([ \
                transforms.Mesh2Points(), \
                transforms.OnUnitCube(), \
                transforms.Resampler(10000), \
                ])

            testdata = Scene7(args.dataset_path, transform=transform, classinfo=cinfo)

            # randomly generate transformation matrix
            mag_randomly = True
            testset = TransformedDataset(testdata, transforms.RandomTransformSE3(0.8, mag_randomly))
            return testset
