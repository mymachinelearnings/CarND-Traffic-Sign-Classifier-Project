# Udacity Self-Driving Car Engineer Nanodegree


## Term 1 Project 1 : Traffic Sign Classifier
### Project Writeup

#### Project Writeup

---
 
Create a Traffic Sign Classifier that can analyze the input images and predict the result via a Neural Network

---
The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report
---
## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
You're reading it! and here is a link to my [project code](https://github.com/mymachinelearnings/CarND-Traffic-Sign-Classifier-Project/blob/attempt2/TrafficSignClassifier.ipynb)


## Data Set Summary & Exploration

### 1. Basic summary of the data set.

German Traffic Sign dataset is used for training, validation & testing the model. This is a freely available dataset that can be used for traffic sign classification use cases

Udacity has provided this data set as a pickled dataset. I've used Pickle library to import the dataset.
The data set consisted of training, validation and test data sets seperately.
I assigned each of the data sets to different python variables

Here's some stats on the data set

| Criteria		                    | Results	        					        | 
|:---------------------------------:|:---------------------------------------------:| 
| Size of Training set      		| 34799   									    | 
| Size of Validation set     		| 4410 										    |
| Size of Test set					| 12630											|
| Shape of the Image	      		| 32 * 32 Colored Images					 	|
| Number of Unique Classes			| 43      							            |


### 2. Visualizing the dataset
Using histogram representation, here is the distribution of the training images aginst its classes

![](data/WriteupImages/HistogramTrainingBefore.png?raw=true)

Based on this visualization, it is clear that some of the classes have very low number of images. With such a small data set on certain classes, the model cannot predict accurately. To avoid this, the data is augmented which is explained in the next section

Here's a sample set from the training data for each of the first 10 classes
![](data/WriteupImages/TrainingSetImages10Classes.png?raw=true)

## Design and Test a Model Architecture

### 1. Image Pre-processing
Images from the training data are 32 * 32 Color Images.

**Gray Scale Conversion**
Color Images have 3 channels, one each for Red, Green, Blue. We can use color images as is for the model, but the computational power required is higher than if we are working on 1 channel. In order to do this, we convert the data into Gray Scale.

**CLAHE (Contrast Limited Adaptive Histogram Equalization)**
Histogram equalization is required as a pre-processing step to enhance the contrast of dull images. Images with lesser light are very hard to distinguish, and if you observe the histogram of image pixels, it will be more concentrated in one area. Histogram equalization is a technique to spread the histogram across the graph which makes a better contrast image. But this poses a problem sometimes where a bright part of the image becomes more brighter. In such cases, a technique called adaptive histogram equalization is used where histogram equalization will be done block wise in the image. This technique is otherwise called CLAHE

**Normalization**
Images are nothing but a set of numbers in digital representation. Each of these numbers represent the pixel color value that ranges from [0-255]. By this, it is clear that this is not zero-mean. Making an image zero-mean is important for backprop to work in a much faster and efficient way. It is done by subtracting each pixel 127.5 and dividing by 127.5

Here's a representation of the three variations of an image compared to the original image

![](data/WriteupImages/ImageConversion.png?raw=true)

**Data Augmentation**
As mentioned in the previous section, the number of training images per class should be decent enough to train the model accurately. Since some classes have very less number of images, i've augmented the dataset to make sure each class has atleast 800 images.
In this case, i've copied the images of the class and virtually duplicated them so that the number goes high.

Post augmentation, here's the histogram representation of the distribution
![](data/WriteupImages/HistogramTrainingAfter.png?raw=true)

### 2. Image Classifier Design
Image Classifier (a.k.a. model from now onwards) is implemented based on the CNN architecture by Yann LeCun. Though this architecture is proved and yeilds ~89% accuracy against validation set for the German Traffic Sign Classifier, I've decided to tweak the architecture so that it achieves higher accuracy

Here's my Model design

Yann LeCun's architecture consisted of the following architecture

- Conv --> MaxPool --> Conv --> MaxPool --> FC1 --> FC2 --> o/p

This is a pretty good architecture, but for the German Traffic Sign dataset, this is giving an accuracy less than 90%
In order to achieve better accuracy, I've tweaked the model a bit and this resulted in almost 96% accuracy
Here's my model's architecture
- Conv --> Relu --> MaxPool --> Conv --> Relu --> MaxPool -->  FC1 --> Dropout --> FC2 --> Dropout --> FC3 --> Dropout --> Logits

My model consisted of the following layers, and corresponding output dimensions of the activation layers. The third argument represents the number of filters in each layer

| Layer                             | Output Dimensions                            | Stride & Padding             |
|------------------------------------|---------------------------------------|------------------------------------|
|Input                               |     n Images of 32 x 32 Gray scale    |                                    |
|Convolution 3x3, 6 filters          |     [30, 30, 6]                       |      1x1 Stride, VALID padding     |
|RELU                                |                                       |                                    |
|MaxPool                             |     [15, 15, 6]                       |      2x2 Stride, 2x2 Kernel        |
|Convolution 4x4, 32 filters         |     [12, 12, 32]                      |      1x1 Stride, VALID padding     |
|RELU                                |                                       |                                    |
|MaxPool                             |     [6, 6, 32]                        |      2x2 Stride, 2x2 Kernel        |
|Convolution 3x3, 32 filters         |     [4, 4, 32]                        |      1x1 Stride, VALID padding     |
|RELU                                |                                       |                                    |
|Flatten                             |     4*4*32 = 512                      |                                    |
|Fully Connected                     |     [512, 256]                        |                                    |
|Dropout                             |     0.5                               |                                    |
|Fully Connected                     |     [256, 120]                        |                                    |
|Dropout                             |     0.5                               |                                    |
|Fully Connected                     |     [120, 84]                         |                                    |
|Dropout                             |     0.5                               |                                    |
|Fully Connected                     |     43 Classes                        |                                    |

### 3. Training the Model
The model is trained with the following hyper parameters
- LEARN_RATE = 0.0005
- BATCH_SIZE = 128
- EPOCHS = 60

For each epoch, the model is given a learning rate of 0.005 with a batch size of 128. The training program loops through the entire training set, picks 128 images **at random** and proceeds with the training. Picking the images at random is important so that the model is not overfitted.

In the training pipeline, we train the model by calculating the Logits using the Neural Network Cross Entropy loss, which is the difference between the logits vector and ground truth(in the form of one hot encoded vector).

This loss is averaged over all the training samples(batches) and is minimized using Adam Optimizer. Adam Optimizer is similar to Stochastic Gradient Descent but with a much better algorith. Based on  the loss determined, backprop is performed and weights, biases are adjusted in the negative direction of the gradient.

### 4. Arriving at the architecture
I started with the LeNet Implementation exercise provided by Udacity Self Driving Car Nanodegree Program, made some adjustments to the hyper parameters, but I could not acheive an accuracy on the validation set beyond 89%

This made me think to pre-process the data and augment the training set to contain atleast 800 images per class. This gave me an accuracy of ~93%. Though this is enough for the project, I want to check if I can achieve better results.

This took me to modifying the architecture. I've extended Yann LeCun's architecture to have one additional Convolution Layer and 2 additional fully connected layers. This gave me an accuracy of ~95%

I further used dropout mechanism to finally achieve an accuracy of ~96% on the validation set.

I think this can further be increased by better augmenting the data using various mechanisms like rotation, adjusting brightness, etc. but I think this number is decent enough to have a sense of satisfaction with the project :)

Notes :
- Having learning rate as low as possible is better so that the weight adjustments happen properly
- I didn't observe much difference with having batch size 128 or 256, hence i've settled for 128
- Having more epochs is always a good idea. I had chosen 60 so that it doens't take too much during training and at the same time not too less
- observed that having dropout layers only on fully connected layers yeilds more accuracy than to have the dropouts on Convolution layers. The differece is about 1% as observed in about 10 different training cycles

### 5. Validation & Test Results
The whole process took me about 2-3 days just for training and tuning hyper parameters. Having GPU instances really helped me to fasten the training process

Here are my results

| Data Set| Accuracy|
|------------------------|--------------|
|Training Set            |     99.1%    |
|Validation Set          |     95.7%    |
|Test Set                |      94.2%   |

### 6. Testing the Model with new Images
I've tested the model against 5 images corresponding to German Traffic Signs downloaded from internet.
Testing was done by manually creating the label array to the images and fed to the evaluate function of the model

Here are the images
![](data/WriteupImages/ImagesFromNet.png?raw=true)

To my surprise, the model predicted the test set with an **accuracy of 100%** - meaning it predicted accurately for all the 5 images
Here's a snippet of the result
![](data/WriteupImages/ImagesFromNetResultSnippet.png?raw=true)

Looking at the SoftMax probabilities of the results on these images, here's a prediction of top 3 classes pictorially
![](data/WriteupImages/ImageFromNetPredictions.png?raw=true)

If you observe the % predictions at the top of the image(set as title), you will observe that all the images were accurately predicted with great probability except for the 2nd image which is a bumpy road sign. You see the probability is ~45% for this. This is conflicting with the 'Road Work'. This may have hapenned since the model was not able to predict this class accurately with high results - the possible reason could be due to the the lack of variation in the training data. Augmenting this set to increase the quality of the training data would increase the % accuracy of this prediction

Here's a bar chart representation of the softmax probabilities of the test images against all the classes
![](data/WriteupImages/ImageFromNetBars.png?raw=true)

If you observe, the prediction on 2nd will be more visible and the notes mentioned above applies here.

#### Possible Improvements
Though the model works well on the valdiation and test sets, the model can definitely be improved by doing the following
- Better Augmentation : The training set can be augmented by various techniques to arrive at a better training set
- Possible modifications to the network design : Neural Net can have any design, and what best or not can be arrived by doing a trial and error. With the time I have, I think i've achieved at a decent design


### This ends my project writeup on the Traffic Sign Classifer

---
### Installation & Setup
---

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset and Repository

1. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.
2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```

---

### **Authors** <br/>
* Ravi Kiran Savirigana

### **Acknowledgements** <br/>
Thanks to Udacity for providing the startup code to start with. And a great community help from stackoverflow.com & github.com


```python

```
