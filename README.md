# Training scheme

This section describes the planned training experiments for OCT image classification via contrastive self-supervised learning approaches. In total, the plan includes (48) training experiments which are divided as follow: (6) experiments for SSL pretraining spanned over two methods which are BYOL and SimSiam. In addition, each SSL model has three positive pairs selection strategies in which all positive pairs are considered to be extracted from the same patient as a basis for the three strategies which can be briefly described as follow:

1. Unrestricted: Single image with two different augmented views without any extra condition.
2. XY OCT scans: both horizontal and vertical retina scans for the same patient, the same eye direction and the same diagnosis session
3. Consecutiove: Two scans from consecutive diagnosis sessions for the same patient, the same eye direction and the same OCT scan (X-X) (Y-Y).

Having both models trained over the three strategies, we can proceed toward the down-stream tasks training which account for (42) training experiments that can be described as follow:

1. Linear classification: Where the backbone model is being frozen and just the classification layer is trained. This scheme is mainly intended for evaluating the quality of the learned features from SSL models (18 experiments).
2. Fine tuning: Where both convolutional layers as well as the classification layers are trained simultaneously based with the learned SSL weights as initializations (18 experiments).
3. From scratch : random weights initialization (3 experiments) 
4. Transfer learning: using weights learned from ImageNet dataset.

The following shows in details the training plan fore each experiments Type:

## Pretext task 

### BYOL training

- [ ] unrestricted
- [ ] xy-scans
- [ ] consecutive

### SimSiam training

- [x] unrestricted
- [ ] xy-scans
- [ ] consecutive



## Down-Stream tasks

#### 1-Linear

##### BYOL-unrestricted

- [ ] binary
- [ ] multi-class
- [ ] grading

##### BYOL-xy-scans

- [ ] binary
- [ ] multi-class
- [ ] grading

##### BYOL-consecutive

- [ ] binary
- [ ] multi-class
- [ ] grading

##### SimSiam-unrestricted

- [ ] binary
- [ ] multi-class
- [ ] grading

##### Sim-Siam-xy-scans

- [ ] binary
- [ ] multi-class
- [ ] grading

##### SimSiam-consecutive

- [ ] binary
- [ ] multi-class
- [ ] grading

------------------------------------------------

#### 2-Fine tuning

##### BYOL-unrestricted

- [ ] binary
- [ ] multi-class
- [ ] grading

##### BYOL-xy-scans

- [ ] binary
- [ ] multi-class
- [ ] grading

##### BYOL-consecutive

- [ ] binary
- [ ] multi-class
- [ ] grading

-----------------------------------------------

##### SimSiam-unrestricted

- [ ] binary
- [ ] multi-class
- [ ] grading

##### Sim-Siam-xy-scans

- [ ] binary
- [ ] multi-class
- [ ] grading

##### SimSiam-consecutive

- [ ] binary
- [ ] multi-class
- [ ] grading

--------------------------------------------------

#### 3- From scratch

- [x] binary
- [x] multi-class
- [x] grading

-------------------------------------------------------------

#### 4- Transfer learning

- [x] binary
- [x] multi-class
- [x] grading





