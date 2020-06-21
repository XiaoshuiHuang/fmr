# Feature-metric registration
This repository is the implementation of our CVPR 2020 work: "Feature-metric Registration: A Fast Semi-supervised Approach for Robust Point Cloud Registration without Correspondences"

There are several lights of this work:

1. 💡 This work solves the point cloud registration using feature-metric projection error. 

2. 💡 This work can be trained with unsupervised or semi-supervised manner. 

3. 💡 This work can handle both high noise and density variations. 

4. 💡 This work is potential to handle cross-source point cloud registration. 


To run the code, please follow the below steps:

### 1. Install dependencies:

    pip install torch===1.5.1 torchvision===0.6.1 -f https://download.pytorch.org/whl/torch_stable.html argparse logging numpy glob   matplotlib six

### 2. Train the model

   2.1. Train on dataset ModelNet40:  
   
    python train.py -data modelnet
   
   2.2. Train on dataset 7scene:  
   
    python train.py -data 7scene
   
### 3. Evalute the model

   3.1. Evaluate on dataset ModelNet40: 
   
    python evalute.py -data modelnet
   
   3.1. Evaluate on dataset 7scene: 
   
    python evalute.py -data 7scene
 
### 4. pre-trained models

will update soon

### 5. Test on new point clouds

will update soon
 
### 6. Citation

```
@InProceedings{Huang_2020_CVPR,
    author = {Huang, Xiaoshui and Mei, Guofeng and Zhang, Jian},
    title = {Feature-Metric Registration: A Fast Semi-Supervised Approach for Robust Point Cloud Registration Without Correspondences},
    booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020}
}
```

### Acknowledgement

We would like to thank the open-source code of [AtlasNet](https://github.com/ThibaultGROUEIX/AtlasNet) and [pointnetlk](https://github.com/hmgoforth/PointNetLK)
