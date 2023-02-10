# BowelNet




<img width="500" height="300" src="https://github.com/runningcw/BowelNet/blob/master/bowel_fineseg/arch/segmentors.png"/></dev>



A Pytorch implementation of the paper "BowelNet: Joint Semantic-Geometric Ensemble Learning for Bowel Segmentation from Both Partially and Fully Labeled CT Images"


Introduction:

The algorithm is a two-stage coarse-to-fine segmentation framework for the sgmentation of entire bowel (including duodenum, jejunum-ileum, colon, sigmoid, and rectum) in abdominal CT images. 

(1) [Joint localzation of the 5 bowel parts using both partially- and fully-labeled images](https://github.com/runningcw/BowelNet/tree/master/bowel_coarseseg)

(2) [Fine segmentation of each part using geometric (i.e., boundary and skeleton) guidance](https://github.com/runningcw/BowelNet/tree/master/bowel_fineseg)


Some example images and segmentation results will come soon...

   
If you use the code, please consider to cite the following paper:

C. Wang, Z. Cui, J. Yang, M. Han, G. Carneiro and D. Shen, "BowelNet: Joint Semantic-Geometric Ensemble Learning for Bowel Segmentation from Both Partially and Fully Labeled CT Images," in IEEE Transactions on Medical Imaging, doi: 10.1109/TMI.2022.3225667.
