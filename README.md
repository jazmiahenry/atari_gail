# Using Gail To Win a Game of Atari
An application of General Adversarial Imitation Learning using Sharpened Cosign Similarity instead of Convoluted Neural Networks.

To solve the technical test, I used the Generative Adversarial Imitation Learning model better known by the acronym, GAIL. The choice behind this was simple: AI experts in the video game space have used GAIL frequently to solve problems that require specialized knowledge to complete a task without nedding to possess that specialized knowledge themselves. GAIL is an algorithm that is placed squarely between the principles of Machine Learning and Control Theory. Proposed by Jonathan Ho and Stefano Erhman in 2016, this algorithm takes the approach of extracting an expert's cost function using Inverse Reinforcement Learning and extracting the policy from that cost function using reinforcement learning and improves upon it. The paper describing their work is here: https://arxiv.org/pdf/1606.03476.pdf.

The mathematical formula behind Inverse Reinforcement Learning is here:

<img width="680" alt="IRL_formula" src="https://user-images.githubusercontent.com/48301423/174195818-3f88564c-7405-49be-84c1-af59e923208c.png">


While conducting research on approaches to similar problems in AI, I came across, "Mastering the Game of Go Without Human Knowledge" by David Silver et al at DeepMind, and this inspired me to take the same Policy and Value approach, but use a Discriminator later at the end a la GAN effectively making my model a flavor of GAIL. Though I understand that GAIL is not traditionally understood as an Inverse Reinforcement Learning technique due to its dependence on learning a policy as opposed to learning a reward, I utilized the reward paradigm to add robustness to the model overall. This is not a new technique, but one that I applied in a new way. 

In what originally started as a Twitter conversation, Brandon Rohrer, formally of Facebook and now a Data Scientist at LinkedIn, proved an exciting discovery- that using a sliding window implementation of cosign similarity is an improvement over using traditional neural networks for feature detection. The reason for this is that convolution depends on the sliding dot product interaction between the kernel and the signal without normalizing the corresponding vectors. Normalizing both vectors to the magnitude of 1 and addint two paramaters, p and q, with p raised to the power of some exponent, p, for peak similarity and q serving as a floor reduces the amount of noise in the model and increases the effectiveness of the model's feature detection. I took this logic and incorporated it into a custom sequential model in keras for this test. 

The logic of sharpened cosign similarity and the Inverse Reinforcement Learning algorithm is here:

<img width="406" alt="sharp_cos_sim" src="https://user-images.githubusercontent.com/48301423/174194952-941942fe-a237-40e8-a30c-6e3a4366cbde.png">

A link to more about Sharpened Cosign Similarity is here: https://e2eml.school/scs.html


