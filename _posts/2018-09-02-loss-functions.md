---

layout: post

title:  "Deep Dive: Loss Functions"
date:   2018-08-03 20:00:00 +0200
categories: loss functions machine learning

---

When building a new deep learning model, there are four fundamental things that must be chosen:

 	1. **Data**
     * What the model will be trained on?
 	2. **Architecture**
     * What is the underlying strucutre of the model?
	3. **Loss Function**
    * How well is the model doing?
	4. **Optimizer**
    * What changes should we make to make our model better?

As you may have guessed, this post is going to focus on loss functions.

## What is a loss function?

Let's think about a simple classifier. We have a big pile of pictures that are either a picture of a panda or an armadillo and we want our network to be able to sort them into two piles:

![Pandas_Armadillos](/Users/sam/blog/samringer.github.io/_site/assets/Pandas_Armadillos.png)

​										*Pandas & Armadillos*

Let's say we show the neural network 100 of these pictures and it makes predictions about the content of each one. We want a way of knowing how good these predictions are.

**Idea 1: Counting**

The easiest way of assesing performance it to simple count how many correct predictions the network made. For example "Of the 100 pictures, our network correctly classified 62 of them." This way of scoring is called [precision & recall](https://en.wikipedia.org/wiki/Precision_and_recall).

Precision and recall has one big problem in the context of deep learning. It is *non-differntiable*. This means that, although precision and recall can tell us how good the predictions are at the moment, they can't be used to train the network to produce better predictions in the future.

**Idea 2: Change To Probability**

We can do slightly better by having the network output how confident it is for its predictions for each individual image. For example, the network might say the following:

*"I am 88% sure that image 6 is a panda"* 

or 

"*I am 3% sure Image 2 is a panda*"

and in terms of training, we may respond:

*"Well done! Image 6 was a panda, but update your parameters so next time you are closer to 100% sure, rather than 88%."* 

or in the case of the second example:

"*Well done! Image 2 was an armadillo. Next time try and aim for a 0% confidence instead of 3%*"

(If the second example doesn't make sense remember that an output of "*0% panda*" is equivalent to "*100% armadillo*".)

This is the first example we have come across of a **loss function**. A loss function lets us combine two numbers (the models prediction and the actual label) into **one number.** The simple loss function above finds the difference between the prediction and the label (*100% - 88% = 12%* for example 1 and *3% - 0% = 3%* for example 2.)

The errors calculated by the loss function are known as the **loss**. We want to minimise the error and so a loss closer to 0 is better.

Unlike precision & recall, loss functions are *differentiable* and so our model can be trained!

## Loss functions in action









that, given a picture of either a picture of a panda or an armadillo





How close network is to ideal. Based on observated error. Loss function lets us quantify error for many different points. Explain that this is why loss function is useful. Loss function lets us boil down a vector of errors into a single number.

Loss function is only a function of model weights and biases.

Talk about the mountain scape. Changing model parameters is like walking. Optimizers are which direction and how far. Different points on landscape are different parameters.

Different loss functions define different landscapes.

Quantify how well the prediction made by the network agree with the actual labels.

Mean squared error loss (Euclidean distance)

Infinity Norm

### Regression

MSE (quite sensitive to outliers.) (When might this sensitivty to outliers be useful.)

Mean absolute error.

Mean squared log error loss

Mean absolute percentage error loss (Maybe don't include this and others that ern't as useful) "So is the MAE. The MSLE and the MAPE are worth taking into consideration if our network is predicting outputs that vary largely in range. Suppose that a network is to predict two output variables: one in the range of [0, 10] and the other in the range of [0, 100]. In this case, the MAE and the MSE will penal‐ ize the error in the second output more significantly than the first. The MAPE makes it a relative error and therefore doesn’t discriminate based on the range. The MSLE squishes the range of all the outputs down, simply how 10 and 100 translate to 1 and 2 (in log base 10). " This is normally dealt with with standard normalisation.



### Classification

Hinge loss (equivalent of MSE in classification)

Logistic loss (when probabilities of being different classes are more interesting than classification)

Negative log likelyhood

DEEP DIVE INTO CROSS ENTROPY



### Reconstruction

Used for autoencoders (Maybe skim over this and go over in more depth another time.)



