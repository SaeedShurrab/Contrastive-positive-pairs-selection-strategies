# Project title:

> **Retina’s Disorders Classification and Severity Grading via Optical Coherence Tomography Images and Self Supervised Learning Approaches**



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
