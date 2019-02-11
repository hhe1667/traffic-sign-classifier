# **Traffic Sign Recognition** 

## Writeup
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points.

You're reading it! and here is a link to my [project code](https://github.com/hhe1667/traffic-sign-classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

I used Pandas to calculate summary statistics of the traffic signs data set:

* The sizes of training set, validation set, test set
* The shape of a traffic sign image
* The traffic sign class_id, label, and number of images for each class

#### Exploratory visualization of the dataset.

I then plotted example images for each of the traffic signs. The example images are randomly picked from the training set. The traffic sign id and name are plotted along with the images. See the plot in the notebook [output](https://github.com/hhe1667/traffic-sign-classifier/blob/master/Traffic_Sign_Classifier.html).


### Design and Test a Model Architecture

#### 1. Pre-processing

I did a minimal pre-processing by normalizing the images. I decided not to convert the images to grayscale because I believe the color carries important information.


#### 2. Model Architecture

My model is based on the LeNet. The final model consisted of the following layers:

| Layer         		|     Description	        							| 
|:---------------------:|:-----------------------------------------------------:| 
| Input         		| 32x32x3 RGB image   									| 
| Convolution 3x3     	| kernel=5, 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|														|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 							|
| Dropout	        	|  														|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16			|
| RELU					|														|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 							|
| Dropout	        	|  														|
| Flatten	        	| outputs 400 											|
| Fully connected		| outputs 120        									|
| Fully connected		| outputs 84        									|
| Softmax				| outputs n_classes     								|
|						|														|


#### 3. Model training

To train the model, I used softmax cross-entropy as the loss and the
AdamOptimizer to minimize the loss. The loss converged after around 20 epochs.
I tuned the dropout rate as 0.2 which maximized the validation set accuracy.

#### 4. Getting the validation set accuracy to be at least 0.93.

I started with the LeNet model. I examined the accuracy of the training set
and validation set. The training set accuracy reached ~99.5% while the
validation set accuracy stayed around 91%. The gap indicated the model was
overfitting. Thus, I added dropout to each convolution layer. This improved
the validation set accuracy to ~95%.

The test set accuracy 94.3% is close to that of the validation set. This gives
evidence that the validation set is representative of the test set.

My final model results were:
* training set accuracy of 99.6%
* validation set accuracy of 95.8%
* test set accuracy of 94.3%


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 6 traffic signs that I found on the web by Google image search:

![no-entry.jpg](data/no-entry.jpg) ![pedestrians.jpg](data/pedestrians.jpg) ![speed-limit-30.jpeg](data/speed-limit-30.jpeg)
![stop_sign.jpg](data/stop_sign.jpg) ![warning.png](data/warning.png) ![yield.jpg](data/yield.jpg)


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| no-entry.jpg     		| No entry   									| 
| pedestrians.jpg    	| General Caution (Wrong prediction)			|
| speed-limit-30		| Speed limit (30km/h)							|
| stop_sign.jpg	   		| Stop 							 				|
| warning.png			| General caution      							|
| yield.jpg				| Yield             							|

The model was able to correctly predict 5 out of 6 traffic signs, which gives an accuracy of 83%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The top-5 softmax probabilities and class names for each image are printed in the Jupyter notebook.

The pedestrians.jpg was mis-predicted as "General Caution". Comparing images of
the two signs, it appeared that the model was confused pedestrian with the exclamation mark.

For "speed-limit-30.jpeg", the top probability is 94.5% and the second top
probability is 5.52% with class "Speed limit (80km/h)". This implies that the
model is not that certain in predicting the actual speed limit.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


