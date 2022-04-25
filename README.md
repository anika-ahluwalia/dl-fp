# dl-fp

## Introduction

[This paper](https://arxiv.org/pdf/2101.04061v2.pdf) proposes the use of Generative Adversarial Networks (GANs) to aid in facial recognition and restoration in partially corrupted or incomplete images. The paper attempts to look at images with low resolution, blur, or other image degradations and use reinforcement learning to predict how the degraded portions could be restored. We chose this paper because we wanted to work with images and image manipulation in a potentially useful way. We also are interested to examine the ways in which the dataset (and the potential biases present in the dataset) could impact the results that are produced. 

We will utilize Generative Adversarial Networks (GAN), a form of reinforcement learning in which two neural networks compete, one trying to solve the task while the other tries to come up with more challenging examples.


## Related Work
[Sharma 2021](https://link.springer.com/article/10.1007/s11831-021-09705-4) examines five techniques for 3D facial reconstruction: deep learning, epipolar geometry, one-shot learning, 3D morphable models, and shape-from-shading methods. They discuss current datasets for facial reconstruction as well as  several applications of the technology, such as facial puppetry, video dubbing, virtual makeup, and speech-driven animation. The paper finds that the biggest challenge in 3D facial reconstruction is finding datasets, since current public datasets are not sufficiently large enough. Additionally, deep learning models require extensive computational resources, which is another hardware challenge. Because the paper focuses on the 3D face in a broad sense, they present specific facial features – such as the lips, eyelids, and hair – as possible areas of future exploration. While our paper deep-dives into 2D facial restoration via images, it is interesting examining facial restoration in a 3D context.

Our paper’s implementation: ​​https://arxiv.org/pdf/2101.04061v2.pdf 

Another implementation: https://www.hindawi.com/journals/mpe/2021/2384435/ 

## Data

We will be working with the Flickr-Faces HQ dataset. Because all of the images in the dataset are clean, we will have to introduce noise. We can introduce noise by randomly computing it and assigning it to a variable, and adding that noise to the dataset. Then, we can partition the dataset into 3 parts (https://www.researchgate.net/post/How-to-add-some-noise-data-to-my-classification-datasets). Our data consists of 70,000 images, and the github repo to this data consists of a script that enables us to download the 1024x1024 images. As a result, we will not need to do significant preprocessing beyond accessing those images and storing them. 


### Methodology
The model will use a GAN to examine images that have been partially degraded (we will introduce noise to the images) and restore them. We will compare this to the original images (with no degradation) to improve the performance of the model as it continues to train. The GANs will primarily be learning the convolutions of faces and learning how to recognize facial details throughout the training process.
The most difficult part of implementing the model will be understanding how to successfully utilize and implement GANs to store the data about what is known about facial features. Storing this information about facial priors is more complex and intricate than anything else we have done so far, and the GANs are a fairly new topic that we have only recently been introduced to in class. 

### Metrics
We plan to separate out the dataset that we found into a few sections, one of which will be used for testing (the first 60,000 for training, the latter 10,000 for testing). Once our model is trained and tested, we would like to actually visualize results for ourselves to see how the image restoration looks. At the end, we hope to be able to input our own degraded images of our faces to see how the restoration process performs on those. 

For training and testing, because we will have access to both the original images and the degraded images, we should be able to compare pixel and feature similarity of the restored image that our model produced to the original (undegraded) image to compute a similarity score that should give us a notion of accuracy of our model’s predictions. 

The authors of our current paper proposed a Generative Facial Prior Generative Adversarial Network (GFP-GAN) in order to reconstruct 2D facial images with a single pass. While typical GAN inversion methods necessitate image-specific optimization, their model augments facial details and colors with just one forward pass; specifically, it utilizes a degradation removal module and a pre-trained face GAN. They quantify their results by comparing their GFP-GAN loss metrics (FID, NIQE, PSNR, SSIM, LPIPS) with the loss values of several other face restoration methods. Additionally, they find several different losses for each component of the model, such as facial component loss and feature style loss.

## Goals
Base: Successfully create a GAN that stores information about facial priors.
Target: Achieve 50% accuracy on our testing set with facial restoration.
Stretch: Create a model with which we successfully can restore our own images.

## Ethics

### What broader societal issues are relevant to your chosen problem space?
Our chosen problem space involves several broader societal issues, including deep fakes and bias in facial recognition. Because we’re training our model to generate realistic images of faces, one could conceivably use it to create deep fakes of real or fake individuals. Furthermore, because the model will be trained on a real-world dataset of images, problems with representation in the dataset could contribute to biased outcomes in facial regeneration. For instance, lack of representation of racial minorities might lead to regeneration that lightens skin tones.

### What is your dataset? Are there any concerns about how it was collected, or labeled? Is it representative? What kind of underlying historical or societal biases might it contain?
Our dataset is a collection of 70,000 images found from the website Flickr. While the dataset contains images of people of varying ages, races, and ethnicities, the dataset inherits all of the bias from the Flickr website. In addition, automatic filters were used to clean the dataset. While the github repo does not mention any additional information regarding this filter, some filters have historically been known to make darker skin tones lighter. As a result, our dataset likely reflects racial bias. 


## Division of labor

Preprocessing the data & Degrading original images – Naomi + Galen

Writing a GFP-GAN model — Galen + Anika + Naomi + Lauren

Training and Testing  & Computing accuracy scores (pixel similarity or feature matching) – Anika + Lauren

