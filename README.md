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

prostate_data_1
	├── Fully_labeled_5C
	│	├── abdomen
	│	│   ├── <patient_1>.nii.gz
	│	│  ...
	│	│  ...
	│	│   └── <patient_m>.nii.gz
	│	├── male
	│	│   ├── <patient_1>.nii.gz
	│	│  ...
	│	│  ...
	│	│   └── <patient_m>.nii.gz
	│	└── female
	│	│   ├── <patient_1>.nii.gz
	│	│  ...
	│	│  ...
	│	│   └── <patient_m>.nii.gz
	└── Colon_Sigmoid
        │	├── abdomen
	│	│   ├── <patient_1>.nii.gz
	│	│  ...
	│	│  ...
	│	│   └── <patient_m>.nii.gz
	│	├── male
	│	│   ├── <patient_1>.nii.gz
	│	│  ...
	│	│  ...
	│	│   └── <patient_m>.nii.gz
	│	└── female
	│	│   ├── <patient_1>.nii.gz
	│	│  ...
	│	│  ...
	│	│   └── <patient_m>.nii.gz
	└── Smallbowel
	│	├── abdomen
	│	│   ├── <patient_1>.nii.gz
	│	│  ...
	│	│  ...
	│	│   └── <patient_m>.nii.gz
	│	├── male
	│	│   ├── <patient_1>.nii.gz
	│	│  ...
	│	│  ...
	│	│   └── <patient_m>.nii.gz
	│	└── female
	│	│   ├── <patient_1>.nii.gz
	│	│  ...
	│	│  ...
	│	│   └── <patient_m>.nii.gz
		
```

## Citation:
__If you are interested in this work or use the software, please consider citing the paper__:

C. Wang, Z. Cui, J. Yang, M. Han, G. Carneiro and D. Shen, "BowelNet: Joint Semantic-Geometric Ensemble Learning for Bowel Segmentation from Both Partially and Fully Labeled CT Images," in IEEE Transactions on Medical Imaging, doi: 10.1109/TMI.2022.3225667
