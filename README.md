# CLASSIFICATION OF RICE VARIETIES

## DESCRIPTION:
This project is about classifying the rice variety by using Deep Convolutional neural network model.
## DATASET:
Arborio, Basmati, Ipsala, Jasmine and Karacadag rice varieties were used with a total of 75k images including 15k pieces from each extracted from https://www.muratkoklu.com/datasets/ 
## Analysis

### Partitioning and Visualizing Data
The dataset needs to be split into two parts: one for training and one for validation. As each epoch passes, the model gets trained on the training subset. Then, it assesses its performance and accuracy on the validation subset simultaneously

### Importing the Pre-trained Model
We selected the ResNet-50 model from TensorFlow’s Keras Library to use transfer learning and create a classifier model from keras library.
# Network Architecture 

```bash
ResNet50 CNN model, a variant of ResNet which has 48 Convolutional layers along with 1 MaxPool and 1 Average Pool layer.
* A convolution with a kernel size of 7 * 7 and 64 different kernels all with a stride of size 2 giving us 1 layer.
* Next, we see max pooling with also a stride size of 2.
* In the next convolution there is a 1 * 1,64 kernel following this a 3 * 3,64 kernel and at last a 1 * 1,256 kernel, these three layers are     repeated in total 3 time so giving us 9 layers in this step.
* Next, we see kernel of 1 * 1,128 after that a kernel of 3 * 3,128 and at last a kernel of 1 * 1,512 this step was repeated 4 time so giving     us 12 layers in this step.
* After that there is a kernel of 1 * 1,256 and two more kernels with 3 * 3,256 and 1 * 1,1024 and this is repeated 6 time giving us a total of 18 layers.
* Then again, a 1 * 1,512 kernel with two more of 3 * 3,512 and 1 * 1,2048 and this was repeated 3 times giving us a total of 9 layers.
* After that we do an average pool and end it with a fully connected layer containing 1000 nodes and at the end a SoftMax function so this       gives us 1 layer.
* We don't actually count the activation functions and the max/ average pooling layers.
so, totaling this it gives us a 1 + 9 + 12 + 18 + 9 + 1 = 50 layers Deep Convolutional network.
```

### Training and Finding Accuracy
We did split the dataset into two sets out of which 75 percent is considered for training and rest for testing. Meanwhile we train the data we validate the results escaping the wrong predictions. After training and Evaluating model, we check the accuracy and loss over the given dataset.


 ## GROUP – 24 Members:
 BT20CSE095 - Tejas Peshwe <br>
 BT20CSE096 - Vishal Pilli <br>
 BT20CSE097 - Prajval Tushar Pase <br>
 BT20CSE098 - Pranjal Chouhan 

