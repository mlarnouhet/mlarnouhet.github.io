---
title: "Notes on diffusion"
description: "From score matching to DDPM, NCSN, SDEs, and EDM: working toward a unified understanding of diffusion models."
date: 2026-10-16
---

## Introduction

Recently, I have been diving into some of the main diffusion-related papers out there, such as DDPM [\[1\]](#ref-ddpm "Denoising Diffusion Probabilistic Models"), DDIM [\[2\]](#ref-ddim "Denoising Diffusion Implicit Models"), NCSN [\[3\]](#ref-ncsn "Generative Modeling by Estimating Gradients of the Data Distribution"), SDE [\[4\]](#ref-sde "Score-Based Generative Modeling through Stochastic Differential Equations"), and EDM [\[5\]](#ref-edm "Elucidating the Design Space of Diffusion-Based Generative Models"), trying to make sense of them and understand how diffusion methods and frameworks have evolved over time.

A lot of effort has been invested over the years to unify different diffusion methods into a single framework, such as SDE and then EDM. In spite of that, and after going through these papers, I still found it challenging to fully make sense of this unified framework. That is why I decided to dive deeper into the topic and summarize my findings in this blog.

I will start by presenting the two successful classes of diffusion-based generative models that were initially introduced: DDPM and NCSN. Both of these models revolve around the idea of adding slowly increasing noise to data points and then learning how to denoise these data points to form a generative model of the data. After presenting these methods, I will explain how the SDE framework unifies them and how the EDM framework improves on that.

## 1. Score Matching

### 1.1. Learning a density gradient

Both DDPM and NCSN, as well as the SDE and EDM frameworks, make use of score matching [\[6\]](#ref-score-matching "Aapo Hyvärinen (2005), Estimation of Non-Normalized Statistical Models by Score Matching") to learn a time-dependent score of the perturbed data distribution.

Given an unknown data distribution $p_{\mathrm{data}}(x)$, the score function is:

$$
S(x)=\nabla_x \log p_{\mathrm{data}}(x)
\tag{1}
$$

It is a vector pointing in the direction where the probability density increases. In the case of diffusion models, it can help us move from a noisy image toward the distribution of real images.

Score matching trains a network to estimate the score function $S_\theta$ by minimizing the objective:

$$
\mathcal{L}(\theta)
=\frac{1}{2}\mathbb{E}_{x\sim p_{\mathrm{data}}}
\Bigl[\bigl\lVert S_\theta(x)-\nabla_x\log p_{\mathrm{data}}(x)\bigr\rVert_2^2\Bigr]
\tag{2}
$$

This can be rewritten (up to a constant) as:

$$
\mathcal{L}(\theta)
=\mathbb{E}_{x\sim p_{\mathrm{data}}}
\left[\operatorname{tr}\bigl(\nabla_x S_\theta(x)\bigr)
+\frac{1}{2}\lVert S_\theta(x)\rVert_2^2\right]
\tag{3}
$$

Here $\nabla_x S_\theta(x)$ is the Jacobian of $S_\theta(x)$, and its trace (the divergence) is:

$$
\operatorname{tr}(\nabla_x S_\theta(x))
=\sum_{i=1}^{d}\frac{\partial S_{\theta,i}(x)}{\partial x_i}
$$

where $d$ is the dimension of the data.

This initial formulation of score matching is unfortunately not scalable because computing the divergence becomes an important computational bottleneck as the dimensionality of the data becomes large, as is the case with images.

This is why the diffusion methods we are going to discuss instead focus on another approach, denoising score matching [\[7\]](#ref-denoising-score-matching "Pascal Vincent (2011), A Connection Between Score Matching and Denoising Autoencoders").

### 1.2. Denoising score matching

Instead of directly learning the score of the data distribution, we perturb the initial data distribution with noise and then attempt to learn the score of that perturbed distribution.

Let $x\sim p_{\mathrm{data}}(x)$, $\tilde x$ be the perturbed data point, $\sigma>0$, and define the perturbation kernel:

$$
q_\sigma(\tilde x\mid x)=\mathcal{N}(\tilde x;x,\sigma^2 I)
\tag{4}
$$

with corresponding marginal:

$$
q_\sigma(\tilde x)
=\int q_\sigma(\tilde x\mid x)\,p_{\mathrm{data}}(x)\,dx
$$

The objective becomes:

$$
\mathcal{L}(\theta)=\frac{1}{2}
\mathbb{E}_{x\sim p_{\mathrm{data}},\,\tilde{x}\sim q_\sigma(\cdot\mid x)}
\Bigl[
\bigl\lVert S_\theta(\tilde{x})-\nabla_{\tilde{x}}\log q_\sigma(\tilde{x}\mid x)\bigr\rVert_2^2
\Bigr]
\tag{5}
$$

This is much easier to compute, given that:

$$
\tilde x=x+\sigma\varepsilon,
\qquad \varepsilon\sim\mathcal{N}(0,I)
$$

so:

$$
\nabla_{\tilde x}\log q_\sigma(\tilde x\mid x)
=-\frac{\tilde x-x}{\sigma^2}
=-\frac{\varepsilon}{\sigma}
$$

We can also show that the optimal network minimizing this objective satisfies the following equality almost surely:

$$
S_{\theta^*}(\tilde x)=\nabla_{\tilde x}\log q_\sigma(\tilde x)
\tag{6}
$$


We first establish the following property of squared-error regression: Let $X$ and $Y$ be random variables. Minimizing $\mathbb{E}[\|f(X)-Y\|_2^2]$ yields the optimal predictor $f^*(X)=\mathbb{E}[Y\mid X]$.

Let $m(X)=\mathbb{E}[Y\mid X]$. Expanding the squared L2 norm,

$$
\begin{aligned}
\mathbb{E}[\|f(X)-Y\|_2^2]
&=\mathbb{E}[\|f(X)-m(X)\|_2^2]+\mathbb{E}[\|Y-m(X)\|_2^2]\\
&\quad-2\mathbb{E}[(f(X)-m(X))^\top(Y-m(X))]
\end{aligned}
$$

Since $\mathbb{E}[Y-m(X)\mid X]=\mathbb{E}[Y\mid X]-m(X)=0$, the cross term vanishes by using the law of total expectation:

$$
\mathbb{E}[(f(X)-m(X))^\top(Y-m(X))]=\mathbb{E}[(f(X)-m(X))^\top\mathbb{E}[Y-m(X)\mid X]]=0
$$

We end up with:

$$
\mathbb{E}[\|f(X)-Y\|_2^2]
=\mathbb{E}[\|f(X)-m(X)\|_2^2]+C
$$

where $C=\mathbb{E}[\|Y-m(X)\|_2^2]$ is independent of $f$. Hence the minimizer is $f^*(X)=m(X)=\mathbb{E}[Y\mid X]$.

Applying this result to our setting, the loss can be rewritten as

$$
\mathcal{L}(\theta)=\frac12\mathbb{E}_{x\sim p_{\mathrm{data}},\,\tilde{x}\sim q_\sigma(\cdot\mid x)}\left[
\left\|S_\theta(\tilde x)-\mathbb{E}[\nabla_{\tilde x}\log q_\sigma(\tilde x\mid x)\mid\tilde x]\right\|_2^2
\right]+C_\sigma
$$

Consequently, assuming the model can represent the optimum,

$$
S_{\theta^*}(\tilde x)=\mathbb{E}[\nabla_{\tilde x}\log q_\sigma(\tilde x\mid x)\mid\tilde x]
$$

We now show that this conditional expectation equals the score of the perturbed data distribution.

$$
\begin{aligned}
S_{\theta^*}(\tilde x)
&=\int \nabla_{\tilde x}\log q_\sigma(\tilde x\mid x)\,p(x\mid\tilde x)\,dx\\
&=\int\frac{\nabla_{\tilde x}q_\sigma(\tilde x\mid x)}{q_\sigma(\tilde x\mid x)}\,p(x\mid\tilde x)\,dx\\
&=\frac{1}{q_\sigma(\tilde x)}
\int p_{\mathrm{data}}(x)\nabla_{\tilde x}q_\sigma(\tilde x\mid x)\,dx\\
&=\frac{\nabla_{\tilde x}q_\sigma(\tilde x)}{q_\sigma(\tilde x)}
=\nabla_{\tilde x}\log q_\sigma(\tilde x)
\end{aligned}
$$

where we used Bayes' rule, $p(x\mid\tilde x)=p_{\mathrm{data}}(x)q_\sigma(\tilde x\mid x)/q_\sigma(\tilde x)$.

The main intuition behind this result is that different clean images can produce the same noisy observation. The network does not know which clean image produced a given noisy observation, so it ends up learning the average of all possible corrections, weighted by how plausible each clean image is given the observation.

## 2. DDPM and DDIM

### 2.1 DDPM

The DDPM method [\[1\]](#ref-ddpm) generates images by learning to reverse a diffusion process, which is a Markov chain that gradually adds noise to the data until the signal is destroyed.

More precisely, consider a sequence of positive noise scales $0<\beta_1,\ldots,\beta_T<1$.
Given an image $x_0\sim p_{\mathrm{data}}(x)$, the Markov chain computes a sequence of latents $x_1,\ldots,x_T$ defined by:

$$
p(x_i\mid x_{i-1})
=\mathcal{N}\bigl(x_i;\sqrt{1-\beta_i}\,x_{i-1},\beta_i I\bigr)
\tag{7}
$$

This is the forward process, or diffusion process. The reverse process starts by sampling $x_T\sim\mathcal N(0,I)$ and then denoises it step by step by sampling:

$$
\begin{aligned}
x_{i-1}&\sim p_\theta(x_{i-1}\mid x_i),\\[0.4em]
p_\theta(x_{i-1}\mid x_i)
&=\mathcal{N}\!\left(
x_{i-1};
\frac{x_i+\beta_iS_\theta(x_i,i)}{\sqrt{1-\beta_i}},
\beta_iI
\right)
\end{aligned}
\tag{8}
$$

Here $S_\theta$ is a neural network modeling the time-dependent score of the perturbed distributions.

A useful property of the forward process is that we can sample $x_i$ directly from $x_0$:

$$
p_{0:i}(x_i\mid x_0)
=\mathcal{N}\bigl(x_i;\sqrt{\bar\alpha_i}\,x_0,(1-\bar\alpha_i)I\bigr)
\tag{9}
$$

where:

$$
\bar\alpha_i=\prod_{j=1}^{i}(1-\beta_j)
$$

This allows us to train our network using the following score-matching objective:

$$
\theta^*=\arg\min_\theta\sum_{i=1}^{T}(1-\bar\alpha_i)\mathbb{E}_{x_0\sim p_{\mathrm{data}},\,x_i\sim p_{0:i}(\cdot\mid x_0)}\Bigl[\bigl\lVert S_\theta(x_i,i)-\nabla_{x_i}\log p_{0:i}(x_i\mid x_0)\bigr\rVert_2^2\Bigr]\tag{10}
$$

### 2.2 DDIM

Denoising Diffusion Implicit Models [\[2\]](#ref-ddim) reuse the DDPM training objective and network but change the reverse sampling rule. We consider the deterministic version ($\eta=0$), which doesn't introduce fresh Gaussian noise during the reverse steps.

Starting from $x_T\sim\mathcal N(0,I)$, at noise level $i+1$ we use the score network to predict the noise and estimate the clean image:

$$
\hat\varepsilon_{i+1}=-\sqrt{1-\bar\alpha_{i+1}}\,S_\theta(x_{i+1},i+1) \quad \text{and} \quad \hat x_0=\frac{x_{i+1}-\sqrt{1-\bar\alpha_{i+1}}\,\hat\varepsilon_{i+1}}{\sqrt{\bar\alpha_{i+1}}}
$$

To move to the lower noise level $i$, we combine these estimates using the corresponding signal and noise scales:

$$
x_i=\sqrt{\bar\alpha_i}\,\hat x_0+\sqrt{1-\bar\alpha_i}\,\hat\varepsilon_{i+1}
\tag{11}
$$

## 3. NCSN

The NCSN method [\[3\]](#ref-ncsn) generates samples via Langevin dynamics using gradients of the log density of the data distribution estimated with score matching. Because these gradients can be hard to estimate when the data resides on low-dimensional manifolds, they are estimated from perturbed data distributions instead.

Let $\sigma_{\min}=\sigma_1<\cdots<\sigma_N=\sigma_{\max}$ be a sequence of positive noise scales and let 
$p_{\sigma_i}(\tilde x\mid x)=\mathcal{N}(\tilde x;x,\sigma_i^2 I)$ be their corresponding perturbation kernels.



The scale $\sigma_{\min}$ is small enough that $p_{\sigma_{\min}}(x)\approx p_{\mathrm{data}}(x)$, while $\sigma_{\max}$ is large enough that $p_{\sigma_{\max}}(x)\approx\mathcal{N}(x;0,\sigma_{\max}^2I)$.

We train a neural network to estimate the scores of the perturbed distributions $p_{\sigma_i}(\tilde x)
=\int p_{\mathrm{data}}(x)p_{\sigma_i}(\tilde x\mid x)\,dx$ by using the denoising score-matching objective:

$$
\theta^*=\arg\min_\theta\sum_{i=1}^{N}\sigma_i^2\mathbb{E}_{x\sim p_{\mathrm{data}},\,\tilde x\sim p_{\sigma_i}(\cdot\mid x)}\Bigl[\bigl\lVert S_\theta(\tilde x,\sigma_i)-\nabla_{\tilde x}\log p_{\sigma_i}(\tilde x\mid x)\bigr\rVert_2^2\Bigr]\tag{12}
$$

Sampling is then done by running $M$ steps of Langevin dynamics:

$$
x_i^m
=x_i^{m-1}+\varepsilon_i S_\theta(x_i^{m-1},\sigma_i)
+\sqrt{2\varepsilon_i}\,z_i^m
\tag{13}
$$

where $m=1,\ldots,M$, $\varepsilon_i>0$ is the step size, and $z_i^m\sim\mathcal N(0,I)$.
This is done for $i=N,\ldots,1$, starting at $x_N^0\sim\mathcal{N}(0,\sigma_{\max}^2I)$ and assuming $x_i^0\approx x_{i+1}^M$.

## 4. SDE Framework

### 4.1 Presentation of the SDE framework

The framework presented by Song et al. in *Score-Based Generative Modeling through Stochastic Differential Equations* [\[4\]](#ref-sde) uses stochastic differential equations to transform a data distribution into a simpler distribution by slowly adding noise. The SDE can also be reversed to transform the simple prior distribution back into the original data distribution by slowly removing the noise.

The reverse SDE uses a time-dependent score, which can be learned to then numerically integrate the reverse equation and sample from the data distribution (see [Figure 1](#fig-sde-framework)).

Song et al. show that both DDPM and NCSN can be expressed inside this framework. We will first present the framework in more detail and then explain how DDPM and NCSN fit within it.

<figure id="fig-sde-framework">
  <img src="/images/SDE_1.png" alt="Forward and reverse diffusion processes, with colored stochastic SDE paths and white probability flow ODE trajectories." loading="lazy" />
  <figcaption><strong><a href="#fig-sde-framework">Figure 1.</a></strong> The forward SDE gradually transforms the data distribution into a simple prior by adding noise; the reverse SDE uses the score to recover the data distribution. Colored curves show stochastic SDE paths, while white curves show deterministic probability flow ODE trajectories. Source: <a href="#ref-sde" title="Song et al., Score-Based Generative Modeling through Stochastic Differential Equations">[4]</a>.</figcaption>
</figure>

The SDE framework consists of constructing a diffusion process $\{x(t)\}_{t=0}^T$ indexed by a continuous time variable t such that $x(0)\sim p_{\mathrm{data}}$, and $ x(T)\sim p_T$, where $p_T$ is a prior distribution that is easy to sample from (usually Gaussian).

We have the following SDE:

$$
dx=f(x,t)\,dt+g(t)\,dw_t
\tag{14}
$$

Here $f(x,t)$ is a vector-valued drift coefficient, $g(t)$ is a scalar-valued diffusion coefficient, and $w_t$ corresponds to a standard Wiener process, a continuous-time stochastic process describing random movement over time.

In all the cases we will be dealing with, $f(x,t)=f(t)x$. This SDE indicates that the evolution of the distribution of $x$ is governed by a drift term, $f(x,t)\,dt$, which is a predictable push that changes $x$, and a diffusion term, $g(t)\,dw_t$, which is a random push caused by Gaussian noise.

This SDE has a corresponding reverse-time SDE:

$$
dx=\bigl[f(x,t)-g(t)^2\nabla_x\log p_t(x)\bigr]\,d\bar t+g(t)\,d\bar w_t\tag{15}
$$

Here $d\bar w_t$ denotes the Wiener process when time flows backward and $d\bar t$ is an infinitesimal negative timestep. We can also see that this reverse SDE uses the score of the time-dependant perturbed distribution $p_t(x)$.

We can use the reverse SDE to sample from $p_{\mathrm{data}}$ by solving it numerically. First we need to train a neural network to estimate the time-dependent score $\nabla_x\log p_t(x)$. We can do this using denoising score matching:

$$
\theta^*=\arg\min_\theta\mathbb{E}_{t\sim\mathcal{U}(0,T),\,x(0)\sim p_{\mathrm{data}},\,x(t)\sim p_{0t}(\cdot\mid x(0))}\Bigl[\lambda(t)\,\bigl\lVert S_\theta(x(t),t)-\nabla_{x(t)}\log p_{0t}(x(t)\mid x(0))\bigr\rVert_2^2\Bigr]\tag{16}
$$

with $\lambda(t)$ proportional to the inverse of the expected squared norm of the conditional score:

$$
\lambda(t)\propto
\frac{1}{\mathbb{E}\left[\left\lVert\nabla_{x(t)}\log p_{0t}(x(t)\mid x(0))\right\rVert_2^2\right]}
$$

The transition kernel $p_{0t}(x(t)\mid x(0))$ can be computed as follows:

Start with the linear SDE:

$$
dx=f(t)x\,dt+g(t)\,dw_t
$$

Let:

$$
s(t)=\exp\!\left(\int_0^t f(u)\,du\right) \quad \text{and} \quad y(t)=\frac{x(t)}{s(t)}
$$

We have:

$$
dy=\frac{dx}{s(t)}+x\,d\left(\frac{1}{s(t)}\right)
$$

Since:

$$
d\left(\frac{1}{s(t)}\right)
=-\frac{s'(t)}{s(t)^2}\,dt
$$

That gives us:

$$
dy=\frac{dx}{s(t)}-\frac{s'(t)x}{s(t)^2}\,dt
$$

Replacing $dx$ by its expression:

$$
dy=\frac{f(t)x\,dt+g(t)\,dw_t}{s(t)}
-\frac{s'(t)x}{s(t)^2}\,dt
$$

We also have $s'(t)=f(t)s(t)$, so:

$$
dy=\frac{g(t)}{s(t)}\,dw_t
$$

We then integrate the equation:

$$
y(t)=y(0)+\int_0^t\frac{g(\xi)}{s(\xi)}\,dw_\xi
$$

Therefore:

$$
x(t)=s(t)x_0+s(t)\int_0^t\frac{g(\xi)}{s(\xi)}\,dw_\xi
$$

Let:

$$
I_t=\int_0^t\frac{g(\xi)}{s(\xi)}\,dw_\xi
$$

We can show that $I_t$ is Gaussian with mean zero and covariance matrix $\sigma(t)^2 I_d$ with $\sigma(t)^2 = \int_0^t\frac{g(\xi)^2}{s(\xi)^2}\,d\xi$.

We can start by partitioning the $[0,t]$ interval with $0=t_0<\cdots<t_N=t$ and approximating $I_t$ by:

$$
I_t\approx\sum_{i=0}^{N-1}h(t_i)(w_{t_{i+1}}-w_{t_i})=\sum_{i=0}^{N-1}h(t_i)\Delta w_i
$$

with $h(t)=g(t)/s(t)$ and $\Delta w_i = (w_{t_{i+1}}-w_{t_i})$ the d-dimensional Brownian increments.

The Brownian increments $\{\Delta w_i\}_{i=0}^{N-1}$ are independent Gaussian random variables with:

$$
\mathbb{E}[\Delta w_i]=0 \quad \text{and} \quad \operatorname{Cov}(\Delta w_i)=\mathbb{E}[(\Delta w_i)^2]=\Delta t_i I_d, \quad \text{where} \quad \Delta t_i=t_{i+1}-t_i.
$$

These properties let us derive the mean and variance of $I_t$:

$$
\mathbb{E}[I_t]
=\sum_{i=0}^{N-1}h(t_i)\mathbb{E}[\Delta w_i]=0
$$

Furthermore:

$$
\operatorname{Cov}(I_t)
=\mathbb{E}[I_t^2]-\mathbb{E}[I_t]^2
=\mathbb{E}[I_t^2]
$$

and:

$$
I_t^2
=\left(\sum_{i=0}^{N-1}h(t_i)\Delta w_i\right)^2
=\sum_{i=0}^{N-1}h(t_i)^2(\Delta w_i)^2+\sum_{i\ne j}h(t_i)h(t_j)\Delta w_i\Delta w_j
$$

Since $\Delta w_i$ and $\Delta w_j$ are independent for $i\ne j$:

$$
\mathbb{E}[\Delta w_i\Delta w_j]
=\mathbb{E}[\Delta w_i]\,\mathbb{E}[\Delta w_j]=0
$$

Thus:

$$
\operatorname{Cov}(I_t)=
\mathbb{E}[I_t^2]
=\sum_{i=0}^{N-1}h(t_i)^2\mathbb{E}[(\Delta w_i)^2]
=\sum_{i=0}^{N-1}h(t_i)^2\Delta t_i I_d
$$

Since $I_t$ is a sum of independent Gaussians, it is also Gaussian. When $N\to\infty$, it converges to:

$$
I_t\sim\mathcal{N}\!\left(
0,\int_0^t\frac{g(\xi)^2}{s(\xi)^2}\,d\xi\,I_d\right)
=\mathcal{N}(0,\sigma(t)^2I_d)
$$

The law of $x(t)$ follows:

$$
x(t)\mid x(0)\sim
 \mathcal{N}\left(s(t)x(0),s(t)^2\sigma(t)^2I_d\right)
\tag{17}
$$

Once we have a trained neural network, sampling can be done by discretizing the reverse SDE

$$
x(t_i)=x(t_{i+1})-f(t_{i+1})x(t_{i+1})+g(t_{i+1})^2S_\theta(x(t_{i+1}),t_{i+1})+g(t_{i+1})z_{i+1}
$$

where $z_{i+1}\sim\mathcal N(0,I)$ are independent standard Gaussian vectors. For this discrete update, we absorb the step size $\Delta t_i=t_{i+1}-t_i>0$ into the coefficients using $f(t_{i+1})\leftarrow f(t_{i+1})\Delta t_i$ and $g(t_{i+1})\leftarrow g(t_{i+1})\sqrt{\Delta t_i}$.

The numerical solver used to solve the SDE is what Song et al. call the predictor part of their sampling algorithm. This predictor is combined with a corrector, in the form of annealed Langevin dynamics, which helps improve the sample distribution that was just updated by the predictor. See [Figure 2](#fig-predictor-corrector) for the full PC sampling schemes of both VE-SDE and VP-SDE.

<figure id="fig-predictor-corrector">
  <img src="/images/pc.png" alt="VE-SDE and VP-SDE predictor–corrector sampling algorithms, with predictor steps highlighted in blue and Langevin corrector steps in orange." loading="lazy" />
  <figcaption><strong><a href="#fig-predictor-corrector">Figure 2.</a></strong> Predictor–corrector sampling for the VE-SDE (left) and VP-SDE (right). Source: Song et al., <a href="#ref-sde" title="Song et al., Score-Based Generative Modeling through Stochastic Differential Equations">[4]</a>.</figcaption>
</figure>

### 4.2 Probability-flow ODE

In addition to the reverse SDE, we can derive the following probability-flow ODE from the SDE:

$$
dx=\left[f(t)x-\frac{1}{2}g(t)^2\nabla_x\log p_t(x)\right]dt
\tag{18}
$$

This ODE induces the same time-dependent marginals as the SDE and constitutes another, deterministic, way of generating samples.

### 4.3 VP-SDE and VE-SDE

We will now look at how DDPM and NCSN fit within the SDE framework.

#### 4.3.1 NCSN and the VE-SDE

Each perturbation kernel can be derived from the following Markov chain:

$$
\begin{equation}
x_i=x_{i-1}+\sqrt{\sigma_i^2-\sigma_{i-1}^2}\,z_{i-1},\quad i=1,\ldots,N,\quad z_{i-1}\sim\mathcal{N}(0,I),\quad x_0\sim p_{\mathrm{data}}
\end{equation}
$$

For $i=1,\ldots,N$, let:
$$
\begin{equation}
x_i=x\!\left(\frac{i}{N}\right),\quad
\sigma_i=\sigma\!\left(\frac{i}{N}\right),\quad
z_i=z\!\left(\frac{i}{N}\right)
\end{equation}
$$

When $N\to\infty$, these quantities become continuous functions of $t\in[0,1]$. Let $\Delta t=1/N$. For $t\in\{0,1/N,\ldots,(N-1)/N\}$:

$$
x(t+\Delta t)
=x(t)+\sqrt{\sigma(t+\Delta t)^2-\sigma(t)^2}\,z(t)
$$

Therefore:

$$
dx=\sqrt{d[\sigma(t)^2]}\,z(t)
=\sqrt{\frac{d[\sigma(t)^2]}{dt}}\,\sqrt{dt}\,z(t)
$$

Brownian motion satisfies $dw_t\sim\mathcal N(0,dt\,I)$. Since $\sqrt{dt}\,z(t)\sim\mathcal N(0,dt\,I)$, we obtain the following SDE:

$$
dx=\sqrt{\frac{d[\sigma(t)^2]}{dt}}\,dw_t
\tag{19}
$$

The process defined by this SDE has unbounded variance, which is why it is called the variance-exploding SDE (VE-SDE).

NCSN defines a geometric noise schedule:

$$
\sigma_i=\sigma_{\min}
\left(\frac{\sigma_{\max}}{\sigma_{\min}}\right)^{\frac{i-1}{N-1}}
$$

which, in the continuous-time limit, corresponds to:

$$
\sigma(t)=\sigma_{\min}
\left(\frac{\sigma_{\max}}{\sigma_{\min}}\right)^t
$$

This yields the following SDE:

$$
dx=\sigma_{\min}
 \left(\frac{\sigma_{\max}}{\sigma_{\min}}\right)^t
 \sqrt{2\log\left(\frac{\sigma_{\max}}{\sigma_{\min}}\right)}\,dw_t
$$

This SDE induces the perturbation kernel:

$$
p_{0t}(x(t)\mid x(0))
\quad=\mathcal{N}\!\left(
x(t);x(0),
\sigma_{\min}^2
\left(\frac{\sigma_{\max}}{\sigma_{\min}}\right)^{2t}I
\right)
\tag{20}
$$

This is the kernel used in NCSN.

The associated objective function is also the same:

$$\theta^*=\operatorname*{arg\,min}\limits_{\theta} \mathbb{E}_{\substack{t\sim\mathcal{U}(0,1),\,x(0)\sim p_{\mathrm{data}},\,x(t)\sim p_{0t}(\cdot\mid x(0))}}\;\Bigl[\lambda(t)\,\bigl\lVert S_\theta(x(t),t)-\nabla_{x(t)}\log p_{0t}(x(t)\mid x(0))\bigr\rVert_2^2\Bigr]$$

with:

$$
\lambda(t)\propto \frac{1}{\mathbb{E}\left[\left\lVert\nabla_{x(t)}\log p_{0t}(x(t)\mid x(0))\right\rVert_2^2\right]}
= \sigma(t)^2
$$

When it comes to sampling, NCSN's sampling scheme corresponds to the predictor–corrector framework, where the predictor is the identity function and the corrector is annealed Langevin dynamics.

#### 4.3.2 DDPM and the VP-SDE

DDPM uses the following Markov chain:

$$
x_i=\sqrt{1-\beta_i}\,x_{i-1}+\sqrt{\beta_i}\,z_{i-1},
\qquad i=1,\ldots,N
$$

We can write $\beta_i=\bar\beta_i/N$, where $\bar\beta_i$ is interpreted as a noise rate. When $N\to\infty$, $\bar\beta_i$ becomes a continuous function $\beta(t)$, with $\beta(i/N)=\bar\beta_i$, and:

$$
\begin{aligned}
x(t+\Delta t)
&=\sqrt{1-\beta(t+\Delta t)\Delta t}\,x(t)
+\sqrt{\beta(t+\Delta t)\Delta t}\,z(t)\\
&\simeq\left(1-\frac{1}{2}\beta(t)\Delta t\right)x(t)
+\sqrt{\beta(t)\Delta t}\,z(t)
\end{aligned}
$$

So:

$$
dx=-\frac{1}{2}\beta(t)x(t)\,dt+\sqrt{\beta(t)}\,dw_t
\tag{21}
$$

The process defined by this SDE has bounded variance, which is why it is called the variance-preserving SDE (VP-SDE).

DDPM defines:

$$
\beta_i=\frac{\bar\beta_{\min}}{N}
+\frac{i-1}{N(N-1)}(\bar\beta_{\max}-\bar\beta_{\min}),
\qquad i=1,\ldots,N
$$

Here $\bar\beta_{\min}$ and $\bar\beta_{\max}$ are rates. The continuous-time noise rate is:

$$
\beta(t)=\bar\beta_{\min}
+t(\bar\beta_{\max}-\bar\beta_{\min}),\qquad t\in[0,1]
$$

This gives the SDE:

$$
dx={}-\frac{1}{2}\bigl[
\bar\beta_{\min}+t(\bar\beta_{\max}-\bar\beta_{\min})
\bigr]x\,dt
+\sqrt{
\bar\beta_{\min}+t(\bar\beta_{\max}-\bar\beta_{\min})
}\,dw_t
$$

The corresponding perturbation kernel is:

$$
p_{0t}(x(t)\mid x(0))
=\mathcal{N}\bigl(
x(t);\sqrt{\alpha(t)}x(0),(1-\alpha(t))I
\bigr)
\tag{22}
$$

with $t\in[0,1]$ and:

$$
\alpha(t)
=\exp\left(-\frac{1}{2}t^2(\bar\beta_{\max}-\bar\beta_{\min})
-t\bar\beta_{\min}\right)
$$

We can show that $\alpha(t)$ corresponds to DDPM's:

$$
\bar\alpha_i=\prod_{j=1}^{i}(1-\beta_j)
$$

Since $\beta_i$ is on the order of $10^{-4}$ to $10^{-1}$, we can write, approximately:

$$
\begin{aligned}
\prod_{j=1}^{i}(1-\beta_j)
&\approx \prod_{j=1}^{i}e^{-\beta_j}\\
&=e^{-\sum_{j=1}^{i}\beta_j}\\
&=\exp\!\Bigl(-\frac{i}{N}\bar\beta_{\min}-\frac{i(i-1)}{2N(N-1)}(\bar\beta_{\max}-\bar\beta_{\min})\Bigr)
\end{aligned}
$$

We can reparameterize $t=i/N\in[0,1]$. Since $(i-1)/(N-1)\approx i/N\approx t$:

$$
\bar\alpha_i
\approx\exp\!\left(
-t\bar\beta_{\min}
-\frac{t^2}{2}(\bar\beta_{\max}-\bar\beta_{\min})
\right)
=\alpha(t)
$$

This means that the DDPM perturbation kernel matches the one induced by the SDE framework.

The objective function is the same as previously, with $\lambda(t)\propto 1-\alpha(t)$.

Finally, we can show that DDPM's sampling scheme is equivalent to the reverse diffusion predictor induced by the VP-SDE:

$$
\begin{aligned}
x_i
&=\frac{1}{\sqrt{1-\beta_{i+1}}}
\bigl(x_{i+1}+\beta_{i+1}S_{\theta^*}(x_{i+1},i+1)\bigr)+\sqrt{\beta_{i+1}}\,z_{i+1}\\[0.4em]
&=\left(1+\frac{1}{2}\beta_{i+1}+o(\beta_{i+1})\right)
\bigl(x_{i+1}+\beta_{i+1}S_{\theta^*}(x_{i+1},i+1)\bigr)+\sqrt{\beta_{i+1}}\,z_{i+1}\\[0.4em]
&\simeq\left(1+\frac{1}{2}\beta_{i+1}\right)
\bigl(x_{i+1}+\beta_{i+1}S_{\theta^*}(x_{i+1},i+1)\bigr)+\sqrt{\beta_{i+1}}\,z_{i+1}\\[0.4em]
&=\left(1+\frac{1}{2}\beta_{i+1}\right)x_{i+1}
+\beta_{i+1}S_{\theta^*}(x_{i+1},i+1)
+\frac{1}{2}\beta_{i+1}^2S_{\theta^*}(x_{i+1},i+1)+\sqrt{\beta_{i+1}}\,z_{i+1}\\[0.4em]
&\simeq\left(1+\frac{1}{2}\beta_{i+1}\right)x_{i+1}
+\beta_{i+1}S_{\theta^*}(x_{i+1},i+1)+\sqrt{\beta_{i+1}}\,z_{i+1}\\[0.4em]
&=\left[2-\left(1-\frac{1}{2}\beta_{i+1}\right)\right]x_{i+1}
+\beta_{i+1}S_{\theta^*}(x_{i+1},i+1)+\sqrt{\beta_{i+1}}\,z_{i+1}\\[0.4em]
&\simeq\left[2-\left(1-\frac{1}{2}\beta_{i+1}\right)+o(\beta_{i+1})\right]x_{i+1}
+\beta_{i+1}S_{\theta^*}(x_{i+1},i+1)+\sqrt{\beta_{i+1}}\,z_{i+1}\\[0.4em]
&=\left(2-\sqrt{1-\beta_{i+1}}\right)x_{i+1}
+\beta_{i+1}S_{\theta^*}(x_{i+1},i+1)+\sqrt{\beta_{i+1}}\,z_{i+1}
\end{aligned}
$$

This is the SDE-induced sampler. Overall, DDPM's sampling corresponds to the predictor–corrector framework with ancestral sampling as the predictor and the identity as the corrector.

#### 4.3.3 DDIM and the VE-ODE

The deterministic DDIM iterate can be written as:

$$
\frac{x_{t-dt}}{\sqrt{\bar\alpha_{t-dt}}}
=\frac{x_t}{\sqrt{\bar\alpha_t}}
+\left(
\sqrt{\frac{1-\bar\alpha_{t-dt}}{\bar\alpha_{t-dt}}}
-\sqrt{\frac{1-\bar\alpha_t}{\bar\alpha_t}}
\right)\varepsilon_\theta(x_t,t)
$$

Where $\bar\alpha_t = \prod_{j=1}^{t}(1-\beta_j)$, and $\varepsilon_\theta(x_t,t)=-\sqrt{1-\bar\alpha_t}\,S_\theta(x_t,t)$ is the predicted noise. We can then reparameterize the equation with:

$$
\bar{x}_t=\frac{x_t}{\sqrt{\bar\alpha_t}}
\quad \text{and} \quad
\sigma_t=\sqrt{\frac{1-\bar\alpha_t}{\bar\alpha_t}}
$$

In the continuous case, $\bar{x}$ and $\sigma$ become functions of $t$, with $\sigma$ increasing and $\sigma(0)=0$. We obtain the following ODE:

$$
d\bar{x}(t)
=\varepsilon_\theta\!\left(
\frac{\bar{x}(t)}{\sqrt{1+\sigma(t)^2}},t
\right)d\sigma(t)
$$

To establish the connection to the VE-ODE, we can first consider the fact that $\bar{x}(t)\mid\bar{x}(0) = \bar{x}(0)+\sigma(t)\varepsilon\sim\mathcal N(\bar{x}(0),\sigma(t)^2I)$. This transition kernel corresponds to the variance-exploding schedule, for which we know the ODE. Let $q_t$ be the marginal probability density of $\bar{x}(t)$, the VE probability-flow ODE is:

$$
d\bar{x}(t)
=-\frac12\frac{d[\sigma(t)^2]}{dt}\,\nabla_{\bar{x}}\log q_t(\bar{x})\,dt
=-\sigma(t)\,\nabla_{\bar{x}}\log q_t(\bar{x})\,d\sigma(t)
$$

Since the score satisfies:

$$
\nabla_{\bar{x}}\log q_t(\bar{x})
=-\frac{1}{\sigma(t)}\,\varepsilon_\theta\!\left(
\frac{\bar{x}(t)}{\sqrt{1+\sigma(t)^2}},t
\right)
$$

Substituting this relation into the VE-ODE gives exactly the DDIM ODE above.

## 5. EDM

### 5.1 Studying the ODE

The EDM paper [\[5\]](#ref-edm) reformulates the SDE framework to make it easier to use in practice by clearly separating the concrete design choices. In particular, training and sampling can be designed separately: training learns a denoiser across noise levels, while sampling determines how to use it to generate samples. A fixed trained network can therefore be used with different numerical solvers, time-step schedules, and amounts of stochasticity. We start by analyzing the ODE because its deterministic properties make it easier to understand the impact of each design choice, with the gained insights carrying over to the SDE.

The probability-flow ODE is:

$$
dx=\left[f(t)x-\frac{1}{2}g(t)^2\nabla_x\log p_t(x)\right]dt
$$

The authors argue that the initial ODE defined by Song et al. in the SDE paper using the functions $f$ and $g$ is not convenient to use in practice. Indeed, $f$ and $g$ are used to infer:

$$
s(t)=\exp\left(\int_0^t f(u)\,du\right)

\quad \text{and} \quad

\sigma(t)=\sqrt{\int_0^t\left(\frac{g(u)}{s(u)}\right)^2du}
$$

which are then used to define the transition kernels:

$$
p_{0t}(x(t)\mid x(0))
=\mathcal{N}\bigl(x(t);s(t)x(0),s(t)^2\sigma(t)^2I\bigr)
$$

and the marginals:

$$
p_t(x)=\int p_{0t}(x(t)\mid x(0))p_{\mathrm{data}}(x(0))\,dx(0)
$$

But in practice, we need these marginals for training and sampling, not the original functions $f$ and $g$. It would therefore be more appropriate to define the ODE directly using $s$ and $\sigma$, which determine the transition kernels and marginals.

We can start by expressing $p_t(x)$ in closed form:

$$
\begin{aligned}
p_t(x)
&=\int_{\mathbb{R}^d}p_{0t}(x\mid x_0)p_{\mathrm{data}}(x_0)\,dx_0\\
&=\int_{\mathbb{R}^d}p_{\mathrm{data}}(x_0)
\mathcal{N}\bigl(x;s(t)x_0,s(t)^2\sigma(t)^2I\bigr)\,dx_0\\
&=s(t)^{-d}\int_{\mathbb{R}^d}p_{\mathrm{data}}(x_0)
\mathcal{N}\left(\frac{x}{s(t)};x_0,\sigma(t)^2I\right)\,dx_0\\
&=s(t)^{-d}p\left(\frac{x}{s(t)},\sigma(t)\right)
\end{aligned}
$$

Here the data distribution convolved with Gaussian noise is:

$$
p(x,\sigma)=p_{\mathrm{data}}*\mathcal{N}(0,\sigma^2I)
\tag{23}
$$

We can then express $f$ and $g$ as functions of $s$ and $\sigma$:

$$
f(t)=\frac{\dot s(t)}{s(t)} \quad \text{and} \quad
g(t)=s(t)\sqrt{2\dot\sigma(t)\sigma(t)}
$$

Substituting into the ODE, we obtain:

$$
dx={}\frac{\dot s(t)}{s(t)}x\,dt
-s(t)^2\dot\sigma(t)\sigma(t)
\nabla_x\log p\!\left(\frac{x}{s(t)},\sigma(t)\right)dt
\tag{24}
$$

Parameterizing the ODE with $s(t)$ and $\sigma(t)$ directly informs us about the diffusion process: $\sigma(t)$ changes the signal-to-noise ratio while $s(t)$ changes the overall magnitude of the image.

We can also rewrite the ODE as a function of a denoiser $D_\theta$. Using:

$$
\nabla_x\log p\!\left(\frac{x}{s(t)},\sigma(t)\right)
=\frac{1}{s(t)}\nabla_{\tilde x}\log p(\tilde x,\sigma(t)) \quad \text{with} \quad
\tilde x=\frac{x}{s(t)}
$$

and the denoising identity:

$$
\nabla_{\tilde x}\log p(\tilde x,\sigma(t))
=\frac{D_\theta(\tilde x,\sigma(t))-\tilde x}{\sigma(t)^2}
\tag{25}
$$

we obtain:

$$
\frac{dx}{dt}
=\left(\frac{\dot\sigma(t)}{\sigma(t)}
+\frac{\dot s(t)}{s(t)}\right)x
-\frac{\dot\sigma(t)s(t)}{\sigma(t)}
D_\theta\!\left(\frac{x}{s(t)},\sigma(t)\right)
\tag{26}
$$

The ODE can then be solved by numerical integration. This requires us to choose a numerical solver and the time-step schedule.

Since discretization errors tend to be larger at low noise levels, the authors recommend designing the time-step schedule so that it decreases monotonically with decreasing $\sigma$. These errors can be further minimized using a second-order solver like Runge–Kutta, at the cost of more neural function evaluations.

Another way to reduce truncation errors is to design the ODE so that curvature is limited toward the origin. In particular, EDM recommends setting:

$$
(\sigma(t),s(t))=(t,1)
$$

which approximately corresponds to straight lines.

Taken together, these changes yield significant improvements in the FID score of generated images.

### 5.2 Role of stochasticity

Deterministic sampling using the ODE offers benefits such as faster sampling, but it can also affect the quality of generated images. We can address this by introducing stochasticity through the reverse SDE:

$$
dx_\pm={}\Bigl[
-\dot\sigma(t)\sigma(t)\nabla_x\log p(x,\sigma(t))
\pm\beta(t)\sigma(t)^2\nabla_x\log p(x,\sigma(t))
\Bigr]dt
+\sqrt{2\beta(t)}\,\sigma(t)\,dw_t
\tag{27}
$$

Here $s(t)$ is set to $1$, and $dx_+$ and $dx_-$ represent the SDEs moving forward and backward in time respectively. The reverse SDE adds a Langevin diffusion term which, through noise injection, helps correct errors made in earlier steps. This helps to randomly explore the distribution $p_t$ at time $t$ and pulls the samples towards it. [Figure 3](#fig-stochastic-healing) illustrates this healing property by considering the case where the data points are incorrectly distributed at time t. Following the ODE leads us to completely miss a mode of the initial distribution whereas injecting stochasticity through use of the SDE allows us to recover the full distribution.

<figure id="fig-stochastic-healing">
  <div class="figure-pair">
    <img src="/images/healing_1.png" alt="Deterministic ODE trajectories from incorrectly distributed intermediate samples converge to only one of the two data modes." width="862" height="676" loading="lazy" />
    <img src="/images/healing_2.png" alt="Stochastic SDE trajectories explore the distribution through Langevin dynamics and recover both data modes." width="860" height="686" loading="lazy" />
  </div>
  <figcaption><strong><a href="#fig-stochastic-healing">Figure 3.</a></strong> The data distribution <var>p</var><sub>data</sub> is one-dimensional with two modes. Starting from incorrectly distributed intermediate samples, the deterministic ODE (left) misses one mode. Stochasticity in the SDE (right) helps recover the full original distribution. Source: Miika Aittala's EDM video, <a href="#ref-edm-video" title="Miika Aittala, Elucidating the Design Space of Diffusion-Based Generative Models">[8]</a>.</figcaption>
</figure>

The stochastic sampler introduced by EDM ([Figure 4](#fig-edm-sampler)) is based on this idea of adding noise to correct the errors introduced by the deterministic ODE solver. Starting at step $i$ with noise level $t_i$, it begins by injecting noise to move the sample to a higher noise level $\hat t_i$. It then solves the ODE backward from $\hat t_i$ to $t_{i+1}$ in a single step.

However, injecting too much noise can also degrade the quality of the final image. That is why the EDM sampler controls when to add noise by checking whether the current time step belongs to the interval delimited by the hyperparameters $[S_{\mathrm{tmin}},S_{\mathrm{tmax}}]$. If it does, noise is injected. The parameter $S_{\mathrm{churn}}$ controls the quantity of noise to inject.

<figure id="fig-edm-sampler">
  <img src="/images/edm_sampler.png" alt="EDM stochastic sampling algorithm, showing controlled noise injection followed by an Euler step and a second-order correction." width="1017" height="410" loading="lazy" />
  <figcaption><strong><a href="#fig-edm-sampler">Figure 4.</a></strong> EDM's stochastic sampler. Source: Karras et al., Algorithm 2, <a href="#ref-edm" title="Karras et al., Elucidating the Design Space of Diffusion-Based Generative Models">[5]</a>.</figcaption>
</figure>

### 5.3 Preconditioning and training

The EDM framework proposes wrapping the neural network in a signal-management / preconditioning layer that makes it easier to train. In particular, it is good practice to keep the network's input and target signal magnitudes fixed (e.g. unit variance) to avoid large variations in gradient magnitudes.

In our case, the input is $x=y+n$, where $y$ is a clean image and $n\sim\mathcal N(0,\sigma^2I)$ is noise. EDM chooses the following formulation for the denoiser:

$$
D_\theta(x,\sigma)
=c_{\mathrm{skip}}(\sigma)x
+c_{\mathrm{out}}(\sigma)
F_\theta\bigl(c_{\mathrm{in}}(\sigma)x,c_{\mathrm{noise}}(\sigma)\bigr)
\tag{28}
$$

Here $c_{\mathrm{skip}}(\sigma)x$ is a skip connection and $F_\theta$ is the neural network. It takes as input $x$ multiplied by the scale $c_{\mathrm{in}}(\sigma)$, which sets its variance to $1$. $c_{\mathrm{out}}(\sigma)$ and $c_{\mathrm{skip}}(\sigma)$ are chosen so that the training target has unit variance and $c_{\mathrm{out}}(\sigma)$ is minimized, so that errors are amplified as little as possible. 

The skip connection behaves in different ways depending on the input noise level.

At low noise levels, it is a good idea to predict the noise added to the input (or rather its opposite). That predicted noise is then scaled back down by $c_{\mathrm{out}}(\sigma)$, reducing errors made by the network, and added to the noised image coming from the skip connection, for which $c_{\mathrm{skip}}(\sigma)\approx1$, to yield the final predicted image.

At high noise levels, predicting the noise is a bad idea because whatever error the network makes would be amplified by $c_{\mathrm{out}}(\sigma)$. Instead, we directly predict the final image and deactivate the skip connection by setting $c_{\mathrm{skip}}(\sigma)\approx0$.

The skip connection acts as a continuous switch between low and high noise levels. When dealing with intermediate noise levels, the network predicts a mix of noise and image.

Using this preconditioning, the training loss becomes:

$$
\mathcal{L}(\theta)=\mathbb{E}_{\sigma\sim p_{\mathrm{train}},\,y\sim p_{\mathrm{data}},\,n\sim\mathcal N(0,\sigma^2I)}\left[\lambda(\sigma)c_{\mathrm{out}}(\sigma)^2\left\lVert F_\theta\bigl(c_{\mathrm{in}}(\sigma)(y+n),c_{\mathrm{noise}}(\sigma)\bigr)-\frac{y-c_{\mathrm{skip}}(\sigma)(y+n)}{c_{\mathrm{out}}(\sigma)}\right\rVert_2^2\right]
\tag{29}
$$

Where $p_{\mathrm{train}}$ is log-normal. The weight $\lambda(\sigma)$ is chosen to give unit effective weight at every noise level:

$$
\lambda(\sigma)c_{\mathrm{out}}(\sigma)^2=1.
\tag{30}
$$

## 6. Closing thoughts

I did my best to summarize how diffusion frameworks have evolved over time without leaving out too many important details. I omitted some topics, such as deriving the SDE from the heat equation PDE and the Fokker-Planck equation as I deemed this part too theoretical and wanted to stay close to the practical application of diffusion models.

I also intend to write a follow-up post on flow matching techniques, which are closely related to diffusion models, particularly their ODE formulation.

## References

<p id="ref-ddpm">[1] Jonathan Ho, Ajay Jain, and Pieter Abbeel (2020). <a href="https://arxiv.org/abs/2006.11239"><cite>Denoising Diffusion Probabilistic Models</cite></a> (DDPM).</p>

<p id="ref-ddim">[2] Jiaming Song, Chenlin Meng, and Stefano Ermon (2020). <a href="https://arxiv.org/abs/2010.02502"><cite>Denoising Diffusion Implicit Models</cite></a> (DDIM).</p>

<p id="ref-ncsn">[3] Yang Song and Stefano Ermon (2019). <a href="https://arxiv.org/abs/1907.05600"><cite>Generative Modeling by Estimating Gradients of the Data Distribution</cite></a> (NCSN).</p>

<p id="ref-sde">[4] Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole (2021). <a href="https://arxiv.org/abs/2011.13456"><cite>Score-Based Generative Modeling through Stochastic Differential Equations</cite></a> (SDE).</p>

<p id="ref-edm">[5] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine (2022). <a href="https://arxiv.org/abs/2206.00364"><cite>Elucidating the Design Space of Diffusion-Based Generative Models</cite></a> (EDM).</p>

<p id="ref-score-matching">[6] Aapo Hyvärinen (2005). <a href="https://jmlr.org/papers/v6/hyvarinen05a.html"><cite>Estimation of Non-Normalized Statistical Models by Score Matching</cite></a>. Journal of Machine Learning Research, 6:695–709.</p>

<p id="ref-denoising-score-matching">[7] Pascal Vincent (2011). <a href="https://doi.org/10.1162/NECO_a_00142"><cite>A Connection Between Score Matching and Denoising Autoencoders</cite></a>. Neural Computation, 23(7):1661–1674.</p>

<p id="ref-edm-video">[8] Miika Aittala. <a href="https://www.youtube.com/watch?v=T0Qxzf0eaio"><cite>Elucidating the Design Space of Diffusion-Based Generative Models</cite></a> [Video]. Finnish Center for Artificial Intelligence FCAI, YouTube.</p>
