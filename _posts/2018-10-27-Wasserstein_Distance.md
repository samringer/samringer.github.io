---
layout: post
title:  "Training GANs Using Wasserstein Distance"
date:   2018-08-03 20:00:00 +0200
permalink: /wasserstein_distance/
---

### KL Divergence

A Generative Adversarial Network (or GAN) aims to generate fake data by learning to mimic the probability distribution of some real data, making it an example of *unsupervised learning*. As the GAN aims to copy a real probability distribution, we need a loss function that can tell us how different two probability distributions are. In this case, we want the generated probability distribution to be as close to the real data probability distribution as possible, so we want to minimise said loss function.

Compairing the similarity of two probability distributions isn't straightforward. The simplest way is to use a metric called the [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence). 

![KL_Divergence](assests/images/WGAN/KL_Divergence.jpg)

KL EQUATION ANNOTED IMAGE HERE!

KL divergence quantifies the difference in entropy between two distributions, but it is flawed. Note that the KL divergence between $P(x)$ and $Q(x)$ is different from the KL divergence between $Q(x)$ and $P(x)$. In other words, the distance between $P(x)$ and $Q(x)$ is different than the distance between $Q(x)$ and $P(x)$. This is like saying the distance from London to Manchester is different from the distance from Manchester to London.

*Note: To all intents and purposes, when we say the <u>distance</u> between two probability distribution, we mean the dissimilarity between them.*

The fault is known as the *asymmetry* of KL divergence. There are other issues with KL divergence. Most notably, when our generated probability distribution is undefined at a point (*i.e $Q(x) = 0*) then the KL divergence shoots off to infinity.

### Jensen-Shannon Divergence

[Jensen-Shannon divergence](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence) aims to address this asymmetry. Like KL divergence, it is a measure of similarity between two probability distribution. However, unlike KL divergence, you get the same result by swapping $P(x)$ and $Q(x)$. The JS divergence is therefore *symmetric*. Most traditional GANs aim to minimise a loss function that is equivalent to the JS divergence.

JS EQUATION ANNOTATED HERE!

Sadly, the JS divergence is not the final stop on our journey to find a suitable loss function for our GAN. The nature of how probability distributions like $P(x)$ and $Q(x)$ manifest in high dimensions means that there will be many scenarios during training where either $P(x)$ or $Q(x)$ is 0 (we will explore this is more detail later). In these cases, the JS divergence experiences a sharp change and becomes non-differentiable. What is really needed is a loss function that is symmetric and smooth for all possible $P(x)$ and $Q(x)$.

### Wasserstein Distance

The Wasserstein distance is such a loss function:

W DIST ANNOTED IMAGE HERE!

The Wasserstein distance is symmetric, but to understand its main advantage we have to take a closer look at how both the real and generated data are distrubted in a multidimensional space. (Apologies if the following gets quite abstract/dense...)

Consider the following analogy. The universe we live in is a rather big 3-dimensional space. Humans, however, are only found in a very small part of that space, *i.e Earth*. Not only that but we are only found on the *surface* of the Earth, which appears as 2-dimensional to us humans. So humans only inhabit a small 2-dimensional space inside a much larger 3-dimensional space. In more mathematical terms, you can say that humans live on a *2-manifold* inside a 3-dimensional space.

MAYBE CHANGE DIMENSIONS

Now, in a slight change of direction, think about a stack of 128x128 pictures of Academy Award nominee Stanley Tucci.

<img src="assets/images/WGAN/Tucc_Collage.png" width="700px" />

<html><center><i>Tucc</i></center></html>  

Each picture of Tucci has 128x128 = 16,384 individual pixels. (Let's keep things simply and ignore RGB). By abstracting a bit, try and imagine a 16,384 dimensional space. Each point in the space core correspondse to a different 128x128 picture. Moving along a single dimension in this space is equivalent to one pixel in our image changing value. Moreover, *every* possible 128x128 picture you can think of correspondse to a different point in this space, whether it be a *funny examples here*

FUNNY ANIMAL PICTURES HERE

In our 16,384 dimensional space, our real pictures of Tucci are going to only inhabit a very small space, most likely existsing on lower dimensional manifolds. This because there are far fewer 128x128 pictures of Tucci than there are 128x128 picures. 

If we are training a generator to produce pictures of Stanley then the generator's output images are likely to exist on a seperate set of manifolds in the 16,384 dimensional space. It is very, very unlikely that any of the real data's manifolds and the generator's manifolds are going to overlap in the multi-dimensional space. If this is the case then the probability distributions of the real and generated images are said to be *disjoint.* 

Know we can finally get to why the Wasserstein distance is a better loss function for training GANs than the KL or JS divergences. In the likely situation where $P(x)$ and $Q(x)$ are disjoint, the KL and JS divergences often produce bad gradients, making training both the discriminator and generator very difficult. However, when we use the Wasserstein distance as a loss function, we get good gradients *everywhere*, even if the two probability distributions are disjoint.

### Learning The Wasserstein Distance

The above equation showing the Wasserstein distance is quite complicated. Annoyingly, it can't be hand-written into the code like a more simple loss function like *MSE*. If coding the loss function isn't possible then how can we possibly use it? This being machine learning, the answer should be obvious. We will use a neural network to learn the Wasserstein distance for itself!

This learning is possible using the *Kantorovich-Rubinstein duality*, which is a result that expresses the Wasserstein distance in another form:

KANTOROVICH RUBINSTEIN EQUATION HERE!!!!

Take $f(x)$ to be the discriminator function (the discriminator takes in input $x$ and has an output $f(x)$). Consider the difference between what the generator outputs when it sees some real data and what it outputs when it sees some fake data. The Kantorovich-Rubenstein duality tells us that the upperbound of this difference is equal to the Wasserstein distance. This is a fairly remarkable result. However, it comes with a catch.

### Lipschitz Functions

The catch is that the function our discriminator learns must belong to a specific family of functions, known as *1-Lipschitz functions*. Although it sounds complicated, Lipschitz functions can be understood on a fairly intuitive level. If we have two points, $x_1$ and $x_2$,  we can run them both through a function to get two new points, $y_1$ and $y_2$. If, for *any* $x_1$ and $x_2$ we select, the distance between $y_1$ and $y_2$ is less than or equal to the distance between $x_1$ and $x_2$ then our function is 1-Lipschitz.

The last piece in the jigsaw is making sure the function the discriminator learns is 1-Lipschitz. This is still an ongoing area of research. So far the different approaches taken have been fairly hacky. They include:

* Clipping the weights of all the weight matricies inside the discriminator so they stay within a fixed value. This is what was done in the [original WGAN paper](https://arxiv.org/abs/1701.07875).
* Adding an extra term to our learnt Wasserstein distance loss function. This term penalises the discriminator if the norm of the gradients differ from 1. The resulting architecture is called [WGAN-GP](https://arxiv.org/abs/1704.00028), the GP standing for *Gradient-Penalty*.
* [Spectral normalisation](https://arxiv.org/abs/1802.05957). This idea is more theoretically sound than the methods above. Each weight matrix of the discriminator is normalised in proportion to its largest eigenvalue. Doing so limits the 'stretchiness' of each matrix and ensures they are 1-Lipschitz.

With any luck, you should now have enough knowledge of the Wasserstein distance to cringe everytime you read a GAN paper where an outdated loss function is used!