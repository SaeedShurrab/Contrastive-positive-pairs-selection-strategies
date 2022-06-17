# Project title:

> **New Positive Pairs Selection Strategies in Self-Supervised Contrastive Learning Applied to Medical Images.**



## Project group members:

| Name                | Role       | Email                                                        |
| ------------------- | ---------- | ------------------------------------------------------------ |
| Saeed A. Shurrab    | Student    | [sashurrab18@cit.just.edu.jo](mailto:sashurrab18@cit.just.edu.jo) |
| Prof. Rehab Duwairi | Supervisor | [amerb@just.edu.jo](mailto:amerb@just.edu.jo)                |



# Abstract:

> Annotated medical data scarcity is major problem confronted by the researchers in the field of machine/deep learning. This is because machine/deep learning algorithms require large amounts of high quality annotated medical data to build accurate model. However, building such datasets is expensive and time-consuming operation that requires specialized medical expertise to accomplish this task which is not always available. Several approaches have been developed to mitigate the effect of this problem such transfer learning, data augmentation and semi-supervised learning. self-supervised learning approach is another possible solution that emerged recently in the field of machine learning. Self-supervised learning aims at learning useful representation by generating supervisory signals from unlabeled data without the need for human annotation. Contrastive learning is a branch of self-supervised learning that aims at learning representation by maximizing similarity metric between two augmented views for the same image (positive pairs), while minimizing the similarity with different images (negative examples). 
>
> This research aims at developing new positive pairs selection strategies for contrastive learning when applied on optical coherence tomography scans (OCT). Two factors are considered when developing new strategies including the patient meta-data as well as the necessity to generate each positive pair from the same patient rather than different patients. As a result, two strategies have been developed in this research which are **X-Y OCT scans** and **consecutive diagnosis visits**. In addition, these strategies are compared with the regular positive pairs generation in computer vision tasks which accomplished through random augmentation operations, ImageNet pretrained models and random Initialization. Lastly, three different classification task were employed to evaluate the effectiveness of the proposed strategies which are binary OCT scans classification as normal and abnormal, multiple disease classification and macular retinal oedema severity grading.
>
> The result showed that the proposed strategies improve the classification accuracy, precision, recall and the area under operating receiver curve. More clearly, X-Y scans strategy achieved the best overall accuracy on both binary classification and severity grading tasks with 87.52\% and 80.53\% respectively. Further, Consecutive diagnosis scan achieved the best overall accuracy on the the multiple disease classification task with 80.94\%. Moreover, the proposed strategies showed better performance when compared to training from scratch and using ImageNet pretrained models. Further, the feasibility of the proposed strategies has been validated on Zhang Lab public OCT dataset which showed better performance as compared to the regular augmentation.



## Citation:

Please consider citing our work when using this repo:

```
to be included later on
```



# How to run the code:

## General requirements

### Python version

To avoid dependency problems, please make sure that your python version is: 

```
python 3.8.3+
```

### Environment setup

Follow the following steps:

1. Clone this repo

   ```shell
   git clone https://github.com/SaeedShurrab/thesis-code.git
   ```

2. Navigate to the cloned repository directory:

   ```shell
   cd thesis-code
   ```

3.  Install the environment packages 

   ```
   pip install -r requirements.txt
   ```

   



## Data setup

The research project contains two data sets which are:

1. pretext dataset
2. down-stream dataset

In addition, the file ``data-structuring.py`` is responsible for building both datasets in a **training** ready format as follow:

Create data directory inside the project repo:

```shell
mkdir data
```

 

### Pretext data setup

To prepare the pretext dataset, run the following command

```shell
python data-structuring.py --dataset pretext
```

The file will automatically extract the dataset from the archive file and prepare it. In addition, the dataset is available in this format:

```
./pretext/p#/session-date/eye-direction/xxx-p#-secession-date-eye-direction-orientation.bmp
```

**explanation**

1. **pretext**: pretext dataset directory.
2. **p#**: each patient data are in one folder, e.g., ``p0``: patient 0, ``p1``: pateint 1.....etc.
3. **session-date** for each patient, there may be a single or multiple diagnosis sessions, each session data in one directory: e.g, ``20200831``
4. **eye-direction**: for each patient, the data of every eye is is placed in a separate directory which may be ``L`` or ``R`
5. **xxx-p0-secession-date-eye-direction-orientation.bmp**: each eye-direction folders contains two image where one is horizontal ``x-axis`` and the second is vertical ``y-axis``. 
6. **naming convention**: ``xxx: serial number`` | ``p#: patient number`` | ``session-date: session-date`` | ``eye-direction: eye-direction`` | ``orientation:  x or y`` 



### down-stream data setup

To prepare the down-stream data, run the following command

```shell
python data-structuring.py --dataset down-stream
```

the program will prompt you to select the suitable classification data form which may be ``binary``, ``multi-class`` or `` grading``

```shell
select downstream data form which can be (binary, multi-class, grading): binary
```

For **grading** task, the program will prompt you to select certain disease which may be ``CSR``, ``MRO``,  ``GA``, ``CNV``, ``FMH``, ``PMH``, or `` VMT`` 

```shell
Please select the disease at which grading data will be prepared for (CSR, MRO, GA, CNV, FMH, PMH, VMT): MRO
```

Upon preparation completion, each data form will be available in a directory indicated by its form and split into ``train``, ``val``, and ``test`` according to the following proportions ``80%``, ``15%`` and ``5%`` respectively.



## Training setup

The research project contains two training tasks which are:

1. pretext task training
2. down-stream task training

for each task, there is a specific training file which are : ``simsiam-trainer.py``  for training the pretext dataset and ``downs-stream-trainier.py`` for downs-stream training task.



### pretext training setup

To train the pretext training task, run the following command:

```shell
python simsiam-trainer.py
```

This command will run the pretraining task with default arguments.



To customize the training arguments, run the following command and adjust the required training arguments:

```shell
python simsiam-trainer.py --strategy consecutive --data-dir /datastores/pretext/ --batch-size 128 --num-workers 8 --pin-memory True --backbone resnet18 --optimizer sgd --learning-rate 0.005 --weight-decay 0.0001 --scheduler cosine --ngpus -1 --epochs 100 --precision 16 --es-delta 0.001 --es-patience 10
```



to gain extra information about the each argument, run the following command

```shell
python simsiam-trainer.py --help
```



### down-stream training setup

To train the down-stream training task, run the following command:

```shell
python downs-stream-trainier.py
```

This command will run the pretraining task with default arguments.



To customize the training arguments, run the following command and adjust the required training arguments:

```shell
python downs-stream-trainier.py --training-scheme fine-tune --ssl-model SimSiam --strategy consecutive --weights-path ./epoch=93-step=25285.ckpt --classification-problem grading --data-dir ./data/down-stream --batch-size 16 --pin-memory True --backbone resnet18 --optimizer adam --learning-rate 0.000002 --weight-decay 0.001 --scheduler cosine --ngpus -1 --epochs 100 --precision 16 --es-delta 0.001 --es-patience 5
```



to gain extra information about the each argument, run the following command

```shell
python downs-stream-trainier.py --help
```



