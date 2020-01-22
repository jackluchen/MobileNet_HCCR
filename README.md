# Using MobileNet toTackle Handwritten Chinese Character Recognition Task

## Abstract
We trained a light Convolutional Neural Network to tackle the Handwritten Chinese Character Recognition task under complex conditions. After studying and comparing different popular CNN architectures, we chose MobileNet as our network for its relatively small training cost and real-time application value. In order to solve the problem of insufficient data in the training set, we used Data Augmentation method to generate various writing forms of Chinese characters through simple processing of the input images. As a result, the model after training could be robust to any image transformations such as rotation and flip. This paper introduces the history and current progress of HCCR problem and basic elements of Convolutional Neural Network, emphatically, it describes the architecture of MobileNet and its reduction theory on computational cost. Furthermore, it presents the details of our Data Augmentation and hyperparameters adjustment method. In the end, the analysis and discussion of our final experiment results are included as well.

## Acknowledgement
This is the designated project of the Machine Learning And Visual Computing course in SDU. The authors would like to thank Prof. Chandrajit Bajaj from University of Texas at Austin for his excellent teaching and patient explaining, and Prof. Jianlong Wu from Shandong University for all the suggestions and guidance he offered in this project. Without their supervision and dedication, we would not have done such amazing work as beginners of Deep Learning.