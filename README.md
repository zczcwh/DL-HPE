# Deep Learning-Based Human Pose Estimation: A Survey

## Deep Learning-Based Human Pose Estimation: A Survey  [[Paper](https://arxiv.org/abs/2012.13392)]

## Authors 

[Ce Zheng<sup>∗</sup>](https://zczcwh.github.io/), 
[Wenhan Wu<sup>∗</sup>](https://sites.google.com/view/wenhanwu/%E9%A6%96%E9%A1%B5), 
[Taojiannan Yang](https://sites.google.com/view/taojiannanyang/home), 
[Sijie Zhu](https://sites.google.com/uncc.edu/sijiezhu/home),
[Chen Chen](https://webpages.uncc.edu/cchen62/),
[Ruixu Liu](https://udayton.edu/directory/engineering/electrical_and_computer/liu-ruixu.php),
[Ju Shen](https://udayton.edu/directory/artssciences/computerscience/shen_ju.php),
[Nasser Kehtarnavaz](https://personal.utdallas.edu/~nxk019000/index.html),
[Mubarak Shah](https://www.crcv.ucf.edu/person/mubarak-shah/).

## Overview
This is the official repository of **Deep Learning-Based Human Pose Estimation:A Survey**, a comprehensive and systematic review of recent deep learning-based solutions for 2D and 3D human pose estimation(HPE). It also presents comparison results of different 2D and 3D HPE methods on several publicly available datasets. Additionally, more than 240 research papers since 2014 are covered and **we will update this page on a regular basis. Please feel free to contact <a href="czheng6@uncc.edu">Ce Zheng</a> or <a href="wwu25@uncc.edu"> Wenhan Wu</a> if you have any suggestions!**

## Introduction
Human pose estimation aims to locate the human body parts and build human body representation (e.g., body skeleton) from
input data such as images and videos. It has drawn increasing attention during the past decade and has been utilized in a wide range of
applications including human-computer interaction, motion analysis, augmented reality, and virtual reality. Although the recently
developed deep learning-based solutions have achieved high performance in human pose estimation, there still remain challenges due to
insufficient training data, depth ambiguities, and occlusions. The goal of this survey paper is to provide a comprehensive review of recent
deep learning-based solutions for both 2D and 3D pose estimation via a systematic analysis and comparison of these solutions based on
their input data and inference procedures. More than 240 research papers since 2014 are covered in this survey. Furthermore, 2D and 3D
human pose estimation datasets and evaluation metrics are included. Quantitative performance comparisons of the reviewed methods on
popular datasets are summarized and discussed. Finally, the challenges involved, applications, and future research directions are
concluded.

### Taxonomy
<p align="center"> <img src="./taxonomy.png" width="105%"> </p>

## 2D HPE datasets
### Summary
<p align="center"> <img src="./2D datasets.png" width="70%"> </p>

### Datasets
- Frames Labeled In Cinema (FLIC) [[paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Sapp_MODEC_Multimodal_Decomposable_2013_CVPR_paper.pdf) [[dataset page]](https://bensapp.github.io/flic-dataset.html)
- Max Planck Institute for Informatics (MPII) Human Pose [[paper]](https://openaccess.thecvf.com/content_cvpr_2014/papers/Andriluka_2D_Human_Pose_2014_CVPR_paper.pdf) [[dataset page]](http://human-pose.mpi-inf.mpg.de/#)
- Leeds Sports Pose (LSP) [[paper]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.395.8452&rep=rep1&type=pdf) [[dataset page]](https://sam.johnson.io/research/lsp.html)
- Microsoft  Common  Objects  in  Context  (COCO) [[paper]](https://link.springer.com/chapter/10.1007/978-3-319-10602-1_48) [[dataset page]](https://cocodataset.org/#home)
- AI Challenger Human Keypoint Detection (AIC-HKD) [[paper]](https://arxiv.org/pdf/1711.06475.pdf) [[dataset page]](https://challenger.ai/)
- Penn Action Dataset [[paper]](https://openaccess.thecvf.com/content_iccv_2013/papers/Zhang_From_Actemes_to_2013_ICCV_paper.pdf) [[dataset page]](http://dreamdragon.github.io/PennAction/)
- Joint-annotated Human Motion Database (J-HMDB) [[paper]](https://www.cv-foundation.org/openaccess/content_iccv_2013/papers/Jhuang_Towards_Understanding_Action_2013_ICCV_paper.pdf) [[dataset page]](http://jhmdb.is.tue.mpg.de/)
- PoseTrack dataset [[paper]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Andriluka_PoseTrack_A_Benchmark_CVPR_2018_paper.pdf) [[dataset page]](https://posetrack.net/)


## 2D HPE results on polupar datasets
<p align="center"> <img src="./2D LSP datasets.png" width="85%"> </p>

<p align="center"> <img src="./2D MPII dataset.png" width="85%"> </p>

<p align="center"> <img src="./2D MPII dataset2.png" width="85%"> </p>

<p align="center"> <img src="./2D COCO dataset.png" width="85%"> </p>

## 3D HPE datasets

## 3D HPE results on popular datasets

## Citation
If you find our work useful in your research, please consider citing:

     @misc{zheng2020deep,
      title={Deep Learning-Based Human Pose Estimation: A Survey}, 
      author={Ce Zheng and Wenhan Wu and Taojiannan Yang and Sijie Zhu and Chen Chen and Ruixu Liu and Ju Shen and Nasser Kehtarnavaz and Mubarak Shah},
      year={2020},
      eprint={2012.13392},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
     }

## Updates
