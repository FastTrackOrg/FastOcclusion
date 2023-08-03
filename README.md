# FastOcclusion
# FastOcclusion

## Introduction
Welcome to the FastOcclusion project! This repository contains the code for a proof-of-concept (POC) designed to tackle occlusion challenges within self-similar objects using deep learning models. The project's primary objective is to address the issue of occlusion by employing a two-step approach. Initially, the project involves generating simulated occluded images utilizing non-occluded objects. Subsequently, a YOLO segmentation model is trained on these synthetic images. The ultimate ambition is to deploy this trained model to videos where self-similar objects are susceptible to occlusion.

## Generate Dataset
To begin, execute the following command in your terminal:
```bash
python generate_dataset.py
```

The dataset is derived from the `test_*.png` images, each image featuring a single object. These objects undergo rotations and translations to replicate occlusion scenarios. The pixel-wise annotations are generated based on the object's region within the occluded image. Notably, the occlusions are limited to instances where an object occludes itself, as the detector's purpose is to address occlusions rather than functioning as a conventional object detector.

![labeling example](./labeling.jpg "Training data")

The dataset is divided into three main partitions:
- **Train and Validate**: This segment is utilized for the training process. Constructed using fish images.
- **Unseen Test Images**: These images comprise objects and scenarios that were not encountered during the model's training phase, ensuring a robust assessment of its capabilities. Constructed using rat images

## Training
Initiate the training process by executing the following command:
```bash
python train.py
```

The training procedure utilizes the train and validate dataset in conjunction with the YOLOv8 nano model. Once training concludes, the model's performance is evaluated on the test dataset, gauging its efficacy on entirely new and unseen objects.
