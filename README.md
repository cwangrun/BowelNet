# BowelNet


Pytorch implementation of the paper "BowelNet: Joint Semantic-Geometric Ensemble Learning for Bowel Segmentation from Both Partially and Fully Labeled CT Images". Email: chong.wang@adelaide.edu.au

<img width="515" height="180" src="https://github.com/runningcw/BowelNet/blob/master/bowel_fineseg/arch/pipeline.png"/></dev>
<img width="285" height="185" src="https://github.com/runningcw/BowelNet/blob/master/bowel_fineseg/arch/segmentors.png"/></dev>


## Introduction:

The algorithm is a two-stage coarse-to-fine framework for the sgmentation of entire bowel (including duodenum, jejunum-ileum, colon, sigmoid, and rectum) in abdominal CT images. 

(1) [Joint localzation of the five bowel parts using both partially- and fully-labeled images](https://github.com/runningcw/BowelNet/tree/master/bowel_coarseseg)

(2) [Fine segmentation of each part using geometric (i.e., boundary and skeleton) guidance](https://github.com/runningcw/BowelNet/tree/master/bowel_fineseg)


## Dataset:

We use a private large abdominal CT dataset with partially and fully-labeled segmentation masks. Dataset structure:

```


	│	│   ├── <patient_1>_<slice_1>.png
	│	│  ...
	│	│  ...
	│	│  ...
	│	│   └── <patient_m>_<slice_n>.png
	│	├── GT
	│	│   ├── <patient_1>_<slice_1>.png
	│	│  ...
	│	│  ...
	│	│  ...
	│	│   └── <patient_m>_<slice_n>.png
	│	└── T2
	│	│   ├── <patient_1>_<slice_1>.png
	│	│  ...
	│	│  ...
	│	│  ...
	│	│   └── <patient_m>_<slice_n>.p
		
1. 这是一级有序列表，第一层级还是数字1
  1. 这是二级有序列表，数字在这里会转义成罗马数字
    1. 这是三级有序列表，数字在这里会被转义成英文字母
      1. 这是四级有序列表，显示效果就不会再变化了，依旧是英文字母
		
```

## Citation:
__If you are interested in this work or use the software, please consider citing the paper__:

C. Wang, Z. Cui, J. Yang, M. Han, G. Carneiro and D. Shen, "BowelNet: Joint Semantic-Geometric Ensemble Learning for Bowel Segmentation from Both Partially and Fully Labeled CT Images," in IEEE Transactions on Medical Imaging, doi: 10.1109/TMI.2022.3225667
