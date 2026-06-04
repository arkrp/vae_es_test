# vae_es_test

This is a quick test I am doing to see if it is conceptually sound to utilize a similar strategy to EGGROLL to train variational autoencoder models. I will train a very simple vae model on the MNIST dataset using both traditional gradient decent and the gradient subsitution method described in the EGGROLL paper. I have no expectation that the EGGROLL will perform better. This scenario clearly favors gradient decent. However, I seek to establish that the EGGROLL method can be used, in hopes that its unique qualities may help to better train models which gradient decent may struggle with. (VQ-VAE)

Gonna write the equations here. The basics behind ES and EGGROLL are as follows

$$
\frac{\partial}{\partial\theta}
\mathbb{E}_{\phi|\theta}
\left[
    \mathcal{L(\phi)}
\right]
=\mathbb{E}_{\phi|\theta}
\left[
\mathcal{L(\phi)}
\frac{\partial}{\partial\theta}
\log p(\phi|\theta)
\right]
$$

This identity is the core part of ES, the algorithm that eggrol is built upon. We can see that we see that we can get the gradient of a proxy of our loss function without ever computing the gradient of our loss!

EGGROLL is a method to use an approximate ES in order to update matrix parameters. First of all, it samples from low rank. When sampling, we first sample $A \in \mathbb{R}_{m \times r}$, $B \in \mathbb{R}_{n \times r}$. Both $A$ and $B$ are sampled from iid normal distributions. We then combine A and B together into a matrix perterbation! $E = \frac{1}{\sqrt{r}}AB^{T}$.

Using these random perterbations we can sample our $\phi$ distribution!

$$
\phi_{t,i} = \theta_t + \sigma E_{t,i}
$$


The eggroll paper notes that the sampled $E$ matricies are approximately standard gaussian. As such, we can approximate the score function of our $\phi$ distribution by treating it as such! so basically we say that:

$$
\frac{\partial}{\partial \theta} \log p(\phi_{t,i}|\theta_t) \approx \frac{E_{i,t}}{\sigma}
$$

We can put this all together into a clean little gradient estimator

$$
\frac{\partial}{\partial \theta} \mathcal{L}
\approx
\frac{1}{\sigma}\mathbb{E}
\left[
    E \cdot \mathcal{L}(\theta + \sigma E)
\right]
$$

We replace $\mathcal{L}$ with an unbiased estimator of course. After that we just feed this puppy right into the gears of the SGD machine!

Of course there are more implementation details. $A$ and $B$ exist so that we don't have to materialize the E matricies until the very end of the update step to be reported as the final gradient to be given to SGD or adamw. But honestly thats the gist of it. Thats the whole algo.

Well. Explanation done. Now I actually need to code something. Lamentations.
