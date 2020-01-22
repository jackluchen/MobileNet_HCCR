## Result
In the ﬁnal test, our model predicted on a dataset which contains 9000 images in total, 
with 3000 images testing on robustness on rotation, ﬂip and resize respectively. 
(The rotation angle is between -25 degrees and 25 degrees,the scale of resizing width and height is between 0.5 to 2.5.) 
The accuracy of our model on the whole dataset is 80.57%.

82.13% on rotation, 

93.60% on ﬂip,

only 65.97% on the resized image set.

## Analysis
The main reason of the low accuracy on resized data set is lack of training data, which trace back to the data augmentation process. 
We thought by resizing all input images to one uniformed scale (64x64) should train our model robust to all resized images. 
However, it turns out that for those resized images which the inside character is seriously distorted, which even very hard to recognize for a highly educated human being,
our model performed really bad and fail to recognize almost half of the testing images. 

We think the accuracy could be improved by adding a resize operation into our augmentation pipeline, 
which will randomly shrink or enlarge the width and length of the image to 0.5 to 2.5 times. 
We presume it will make our model capable of recognizing different input images with various sizes with a relatively high accuracy. 
