# BowelNet


Pytorch implementation of the paper "BowelNet: Joint Semantic-Geometric Ensemble Learning for Bowel Segmentation from Both Partially and Fully Labeled CT Images". Email: chong.wang@adelaide.edu.au

<img width="300" height="183" src="https://github.com/runningcw/BowelNet/blob/master/bowel_fineseg/arch/pipeline.png"/></dev>
<img width="285" height="183" src="https://github.com/runningcw/BowelNet/blob/master/bowel_fineseg/arch/segmentors.png"/></dev>


## Introduction:

The algorithm is a two-stage coarse-to-fine framework for the sgmentation of entire bowel (including duodenum, jejunum-ileum, colon, sigmoid, and rectum) from abdominal CT images. The first stage jointly localizes all types of the bowel, trained robustly on both partially and fully labeled samples. The second stage finely segments each type of localized the bowels using geometric bowel representations and hybrid psuedo labels.

[(1) Joint localzation of the five bowel parts using both partially- and fully-labeled images](https://github.com/runningcw/BowelNet/tree/master/bowel_coarseseg)

[(2) Fine segmentation of each part using geometric (i.e., boundary and skeleton) guidance](https://github.com/runningcw/BowelNet/tree/master/bowel_fineseg)


## Dataset:

We use a private large abdominal CT dataset with partially and fully-labeled segmentation masks. Our dataset structure is as follows:

```
BowelSegData
	├── Fully_labeled_5C
	│	├── abdomen
	│	│   ├── <patient_1>.nii.gz
	│	│   ...
	│	├── male
	│	│   ├── <patient_1>.nii.gz
	│	│   ...
	│	└── female
	│	    ├── <patient_1>.nii.gz
	│	    ...
	├── Colon_Sigmoid
        │	├── abdomen
	│	│   ├── <patient_1>.nii.gz
	│	│   ...
	│	├── male
	│	│   ├── <patient_1>.nii.gz
	│	│   ...
	│	└── female
	│	    ├── <patient_1>.nii.gz
	│	    ...
	└── Smallbowel
	 	├── abdomen
	 	│   ├── <patient_1>.nii.gz
	 	│   ...
	 	├── male
	 	│   ├── <patient_1>.nii.gz
	  	│   ...
	 	└── female
	 	    ├── <patient_1>.nii.gz
	 	    ...
```

## Data Preprocessing:

[Preprocessing](https://github.com/runningcw/BowelNet/blob/master/bowel_coarseseg/preprocessing.py) includes cropping abdominal body region. We average all 2D CT slices to form a mean image and then apply a thresholding on it to obtain the abdominal body region (excluding CT bed).


## Citation:
__If you are interested in this work or use the software, please consider citing the paper__:

[C. Wang, Z. Cui, J. Yang, M. Han, G. Carneiro and D. Shen, "BowelNet: Joint Semantic-Geometric Ensemble Learning for Bowel Segmentation from Both Partially and Fully Labeled CT Images," in IEEE Transactions on Medical Imaging, doi: 10.1109/TMI.2022.3225667](https://ieeexplore.ieee.org/document/9966840)
