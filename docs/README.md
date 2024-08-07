# GAIL Model with Sharpened Cosign Similarity as an alternative for CNN

To solve the technical test, I used the Generative Adversarial Imitation Learning model better known by the acronym, GAIL. The choice behind this was simple: AI experts in the video game space have used GAIL frequently to solve problems of this kind. 

While conducting research on approaches to similar problems in AI, I came across, "Mastering the Game of Go Without Human Knowledge" by David Silver et al at DeepMind, and this inspired me to take the same Policy and Value approach, but use a Discriminator later at the end a la GAN effectively making my model a flavor of GAIL. Though I understand that GAIL is not traditionally understood as an Inverse Reinforcement Learning technique due to its dependence on learning a policy as opposed to learning a reward, I utilized the reward paradigm to add robustness to the model overall. This is not a new technique, but one that I applied in a new way. 

In what originally started as a Twitter conversation, Brandon Rohrer, formally of Facebook and now a Data Scientist at LinkedIn, proved an exciting discovery- that using a sliding window implementation of cosign similarity is an improvement over using traditional neural networks for feature detection. The reason for this is that convolution depends on the sliding dot product interaction between the kernel and the signal without normalizing the corresponding vectors. Normalizing both vectors to the magnitude of 1 and addint two paramaters, p and q, with p raised to the power of some exponent, p, for peak similarity and q serving as a floor reduces the amount of noise in the model and increases the effectiveness of the model's feature detection. I took this logic and incorporated it into a custom sequential model in keras for this test. 

The logic of sharpened cosign similarity and the Inverse Reinforcement Learning algorithm are included in the images folder. 

**Credit to: Brandon Rohrer, Raphael Pisoni, Jonathan Ho and Stefano Erhman.**
