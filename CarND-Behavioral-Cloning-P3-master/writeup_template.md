#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with  

1. original input image crop center_image[60:140,:](model.py lines 77)
   







2.  crop image resize 64x64 (model.py lines 77)
               








3  input image normalization 127.5 - 1.0(model.py lines 129-142)
3 .3x3 filter sizes and depths 24 , 
4 .3x3 filter sizes and depths 32 
5 .3x3 filter sizes and depths 48
6 .3x3 filter sizes and depths 64
 

The model includes RELU layers to introduce nonlinearity (code line 132,134,136), and the data is normalized in the model using a Keras lambda layer (code line 127). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 139 and 141).   lines 139 is 0.5 and lines 141 is 0.3 and Train Set 0.9 ratio and 0.1 validation set divide code line 117.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 151-157). I missed save but log file.The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.





















1 epoch : train cost: 0.0094 validation cost:0.0106
2 epoch : train cost: 0.0069 validation cost:0.0113
3 epoch : train cost: 0.0066 validation cost:0.0110

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate 
was not tuned manually. I set 0.00001 (model.py line 33).

####4. Appropriate training data

Training data was chosen to udacity data set. I used a combination of center lane driving, recovering from the left and right sides of the road. 

left angle is csv +0.25 and right angle is csv-0.25(line59-71). And augumentation flip fuction(line 87-88) and for road color distinguish RGB color space.(model.py line 78) and I have short of memory so I used python generator(model.py 45-96) My keras is not compatible tensorflow version. So I backpropation model is theano. Command method is THEANO_FLAGS=floatX=float32,device=gpu,nvcc.flags=-D_FORCE_INLINES python3 model.py 


For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the Nvidia model I thought this model might be appropriate because already verification.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that I used low drop out layer add. I change learning rate 0.0001->0.00001  And I use left Image and right image then correction angle term right is -0.25 left is +0.25

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I change dropout ratio. (0.8-0.8) →(0.5-0.3)

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 124-142) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

Layer (type)                 Output Shape              Param #   
=================================================================
lambda_6 (Lambda)            (None, 64, 64, 3)         0         
_________________________________________________________________
conv2d_21 (Conv2D)           (None, 32, 32, 24)        672       
_________________________________________________________________
max_pooling2d_6 (MaxPooling2 (None, 31, 31, 24)        0         
_________________________________________________________________
conv2d_22 (Conv2D)           (None, 15, 15, 32)        6944      
_________________________________________________________________
activation_16 (Activation)   (None, 15, 15, 32)        0         
_________________________________________________________________
conv2d_23 (Conv2D)           (None, 13, 13, 48)        13872     
_________________________________________________________________
activation_17 (Activation)   (None, 13, 13, 48)        0         
_________________________________________________________________
conv2d_24 (Conv2D)           (None, 11, 11, 64)        27712     
_________________________________________________________________
activation_18 (Activation)   (None, 11, 11, 64)        0         
_________________________________________________________________
flatten_6 (Flatten)          (None, 7744)              0         
_________________________________________________________________
dense_16 (Dense)             (None, 100)               774500    
_________________________________________________________________
dropout_11 (Dropout)         (None, 100)               0         
_________________________________________________________________
dense_17 (Dense)             (None, 10)                1010      
_________________________________________________________________
dropout_12 (Dropout)         (None, 10)                0         
_________________________________________________________________
dense_18 (Dense)             (None, 1)                 11        
=================================================================
Total params: 824,721
Trainable params: 824,721
Non-trainable params: 0

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving: this result is crop and resize.








I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
