---
title: "Diffusion models — transcribed notes"
description: "From score matching to DDPM, NCSN, SDEs, and EDM: working toward a unified understanding of diffusion models."
date: 2026-10-16
tags: [Diffusion models, Generative modeling, Mathematics]
---

> Transcribed from 21 photographed pages of handwritten notes. Equations are rendered in LaTeX, prose is lightly punctuated, and notation is standardized for readability. A few apparent slips in the handwriting have been interpreted in context; doubtful passages are noted where appropriate. The page numbers below follow the order of the uploaded photographs.

## Introduction — Notes on Diffusion

<!-- Introductory page: 483.jpg -->

Recently, I have been diving into some of the main diffusion-related papers out there, such as DDPM, DDIM, NCSN, SDE, and EDM, trying to make sense of them and understand how diffusion methods and frameworks have evolved over time.

A lot of effort has been invested over the years to unify different diffusion methods into a single framework, such as SDE and then EDM. In spite of that, and after going through these papers, I still found it challenging to fully make sense of this unified framework. That is why I decided to dive deeper into the topic and summarize my findings in this blog.

I will start by presenting the two successful classes of diffusion-based generative models that were initially introduced: DDPM and NCSN. Both of these models revolve around the idea of adding slowly increasing noise to data points and then learning how to denoise these data points to form a generative model of the data. After presenting these methods, I will explain how the SDE framework unifies them and how the EDM framework improves on that.

## 1. Score Matching

<!-- Page 1: 484.jpg -->

Both DDPM and NCSN, as well as the SDE and EDM frameworks, make use of **score matching** to learn a time-dependent score of the perturbed data distribution.

Given an unknown data distribution $p_{\mathrm{data}}(x)$, the score function

$$
S(x)=\nabla_x \log p_{\mathrm{data}}(x)
$$

is a vector pointing in the direction where the probability density increases. In the case of diffusion models, it can help us move from a noisy image toward the distribution of real images.

Score matching trains a network to estimate the score by minimizing the objective

$$
\mathcal L(\theta)
=\frac12\mathbb E_{p_{\mathrm{data}}}
\left[\left\|S_\theta(x)-\nabla_x\log p_{\mathrm{data}}(x)\right\|_2^2\right].
$$

This can be rewritten (up to a constant independent of $\theta$) as

$$
\mathcal L(\theta)
=\mathbb E_{p_{\mathrm{data}}}
\left[\operatorname{tr}\bigl(\nabla_x S_\theta(x)\bigr)
+\frac12\|S_\theta(x)\|_2^2\right].
$$

Here $\nabla_x S_\theta(x)$ is the Jacobian of $S_\theta(x)$, and its trace (the divergence) is

$$
\operatorname{tr}(\nabla_x S_\theta(x))
=\sum_{i=1}^{d}\frac{\partial S_{\theta,i}(x)}{\partial x_i},
$$

where $d$ is the dimension of the data.

This initial formulation of score matching is unfortunately not scalable because computing the Jacobian becomes an important computational bottleneck as the dimensionality of the data becomes large, as is the case with images.

<!-- Page 2: 485.jpg -->

This is why the diffusion methods we are going to discuss instead focus on another approach, **denoising score matching**.

Instead of directly learning the score of the data distribution, we perturb the initial data distribution with noise and then attempt to learn the score of that perturbed distribution.

Let $x\sim p_{\mathrm{data}}(x)$, $\sigma>0$, and define the perturbation kernel

$$
q_\sigma(\tilde x\mid x)=\mathcal N(\tilde x;x,\sigma^2 I),
$$

with corresponding marginal

$$
q_\sigma(\tilde x)
=\int q_\sigma(\tilde x\mid x)\,p_{\mathrm{data}}(x)\,dx.
$$

The objective becomes

$$
\mathcal L(\theta)
=\frac12\mathbb E_{x\sim p_{\mathrm{data}},\,\tilde x\sim q_\sigma(\cdot\mid x)}
\left[\left\|S_\theta(\tilde x)
-\nabla_{\tilde x}\log q_\sigma(\tilde x\mid x)\right\|_2^2\right].
$$

This is much easier to compute, given that

$$
\tilde x=x+\sigma\varepsilon,
\qquad \varepsilon\sim\mathcal N(0,I),
$$

so

$$
\nabla_{\tilde x}\log q_\sigma(\tilde x\mid x)
=-\frac{\tilde x-x}{\sigma^2}
=-\frac{\varepsilon}{\sigma}.
$$

We can also show that the optimal network minimizing this objective satisfies

$$
S_{\theta^*}(\tilde x)=\nabla_{\tilde x}\log q_\sigma(\tilde x)
$$

almost surely.

<!-- Page 3: 486.jpg -->

We can start by proving a property of squared-error regression: if we minimize

$$
\mathbb E[\|f(X)-Y\|^2],
$$

then the optimal prediction is

$$
f^*(X)=\mathbb E[Y\mid X].
$$

We have

$$
\begin{aligned}
\mathbb E[\|f(X)-Y\|^2]
&=\mathbb E\left[\left\|(f(X)-\mathbb E[Y\mid X])-(Y-\mathbb E[Y\mid X])\right\|^2\right]\\
&=\mathbb E[\|f(X)-\mathbb E[Y\mid X]\|^2]
 +\mathbb E[\|Y-\mathbb E[Y\mid X]\|^2]\\
&\quad-2\mathbb E\left[(f(X)-\mathbb E[Y\mid X])^\top(Y-\mathbb E[Y\mid X])\right].
\end{aligned}
$$

We can show that the cross term is zero using the law of total expectation:

$$
\begin{aligned}
&\mathbb E\left[(f(X)-\mathbb E[Y\mid X])^\top(Y-\mathbb E[Y\mid X])\right]\\
&=\mathbb E\left[\mathbb E\left[(f(X)-\mathbb E[Y\mid X])^\top(Y-\mathbb E[Y\mid X])\mid X\right]\right]\\
&=\mathbb E\left[(f(X)-\mathbb E[Y\mid X])^\top
\mathbb E[Y-\mathbb E[Y\mid X]\mid X]\right].
\end{aligned}
$$

But

$$
\begin{aligned}
\mathbb E[Y-\mathbb E[Y\mid X]\mid X]
&=\mathbb E[Y\mid X]-\mathbb E[\mathbb E[Y\mid X]\mid X]\\
&=\mathbb E[Y\mid X]-\mathbb E[Y\mid X]=0.
\end{aligned}
$$

Thus the cross term is zero, and

$$
\mathbb E[\|f(X)-Y\|^2]
=\mathbb E[\|f(X)-\mathbb E[Y\mid X]\|^2]
+\mathbb E[\|Y-\mathbb E[Y\mid X]\|^2].
$$

The second term does not depend on $f$, so the minimizer is the conditional expectation.


<!-- Page 4: 487.jpg -->

Then the loss to be minimized is a conditional mean-squared error,

$$
\mathbb E\left[\|f(X)-\mathbb E[Y\mid X]\|^2\right].
$$

In our case, this gives

$$
\mathcal L(\theta)
=\frac12\mathbb E_{x\sim p_{\mathrm{data}},\,\tilde x\sim q_\sigma(\cdot\mid x)}
\left[\left\|S_\theta(\tilde x)
-\mathbb E\bigl[\nabla_{\tilde x}\log q_\sigma(\tilde x\mid x)\mid\tilde x\bigr]\right\|_2^2\right].
$$

Therefore

$$
S_{\theta^*}(\tilde x)
=\mathbb E\bigl[\nabla_{\tilde x}\log q_\sigma(\tilde x\mid x)\mid\tilde x\bigr].
$$

We can then show that

$$
S_{\theta^*}(\tilde x)=\nabla_{\tilde x}\log q_\sigma(\tilde x).
$$

Indeed,

$$
\begin{aligned}
S_{\theta^*}(\tilde x)
&=\mathbb E\bigl[\nabla_{\tilde x}\log q_\sigma(\tilde x\mid x)\mid\tilde x\bigr]\\
&=\int \nabla_{\tilde x}\log q_\sigma(\tilde x\mid x)\,p(x\mid\tilde x)\,dx\\
&=\int \frac{\nabla_{\tilde x}q_\sigma(\tilde x\mid x)}{q_\sigma(\tilde x\mid x)}\,p(x\mid\tilde x)\,dx.
\end{aligned}
$$

From Bayes' rule,

$$
q_\sigma(\tilde x\mid x)
=\frac{p(x\mid\tilde x)q_\sigma(\tilde x)}{p_{\mathrm{data}}(x)}.
$$

Hence

$$
\begin{aligned}
S_{\theta^*}(\tilde x)
&=\int\frac{p_{\mathrm{data}}(x)}{q_\sigma(\tilde x)}
\nabla_{\tilde x}q_\sigma(\tilde x\mid x)\,dx\\
&=\frac{\nabla_{\tilde x}q_\sigma(\tilde x)}{q_\sigma(\tilde x)}\\
&=\nabla_{\tilde x}\log q_\sigma(\tilde x).
\end{aligned}
$$

The main intuition behind this result is that different clean images can produce the same noisy observation. The network does not know which clean image produced a given noisy observation, so it ends up learning the **average of all possible corrections**.


## 2. DDPM

<!-- Page 5: 488.jpg -->

The **DDPM** method generates images by learning to reverse a diffusion process, which is a Markov chain that gradually adds noise to the data until the signal is destroyed.

More precisely, given a sequence of positive noise scales

$$
0<\beta_1,\ldots,\beta_T<1
$$

and an image $x_0\sim p_{\mathrm{data}}(x)$, the Markov chain computes a sequence of latents $x_1,\ldots,x_T$ defined by

$$
p(x_i\mid x_{i-1})
=\mathcal N\bigl(x_i;\sqrt{1-\beta_i}\,x_{i-1},\beta_i I\bigr).
$$

This is the **forward process**, or diffusion process. The reverse process starts by sampling $x_T\sim\mathcal N(0,I)$ and then denoises it step by step by sampling

$$
x_{i-1}\sim p_\theta(x_{i-1}\mid x_i)
=\mathcal N\left(
 x_{i-1};
 \frac{x_i+\beta_i S_\theta(x_i,i)}{\sqrt{1-\beta_i}},
 \beta_i I
\right).
$$

Here $S_\theta$ is a neural network modeling the time-dependent score of the perturbed distributions.

A useful property of the forward process is that we can sample $x_i$ directly from

$$
p_{0:i}(x_i\mid x_0)
=\mathcal N\bigl(x_i;\sqrt{\bar\alpha_i}\,x_0,(1-\bar\alpha_i)I\bigr),
$$

where

$$
\bar\alpha_i=\prod_{j=1}^{i}(1-\beta_j).
$$

This allows us to train our network using the following ELBO-related score-matching objective (as written schematically in the notes):

$$
\theta^*=\arg\min_\theta
\sum_{i=1}^{T}(1-\alpha_i)
\mathbb E_{x_0\sim p_{\mathrm{data}}}
\mathbb E_{x_i\sim p_{0:i}(\cdot\mid x_0)}
\left[\left\|S_\theta(x_i,i)-\nabla_{x_i}\log p_{0:i}(x_i\mid x_0)\right\|_2^2\right].
$$

*Note: the handwritten weighting in this last formula is difficult to distinguish from the product notation; the precise ELBO weights depend on the chosen DDPM parameterization.*

## 3. NCSN

<!-- Page 6: 489.jpg -->

The **NCSN** method generates samples via Langevin dynamics using gradients of the data distribution estimated with score matching. Because these gradients can be hard to estimate when the data resides on low-dimensional manifolds, they are estimated from perturbed data distributions instead.

Let

$$
\sigma_{\min}=\sigma_1<\cdots<\sigma_N=\sigma_{\max}
$$

be a sequence of positive noise scales. We define perturbation kernels

$$
p_{\sigma_i}(\tilde x\mid x)
=\mathcal N(\tilde x;x,\sigma_i^2 I).
$$

The scale $\sigma_{\min}$ is small enough that $p_{\sigma_{\min}}(x)\approx p_{\mathrm{data}}(x)$, while $\sigma_{\max}$ is large enough that

$$
p_{\sigma_{\max}}(x)\approx\mathcal N(x;0,\sigma_{\max}^2I).
$$

We train a neural network to estimate the scores of the noised distributions

$$
p_{\sigma_i}(\tilde x)
=\int p_{\mathrm{data}}(x)p_{\sigma_i}(\tilde x\mid x)\,dx
$$

using the denoising score-matching objective

$$
\theta^*=
\arg\min_\theta
\sum_{i=1}^{N}\sigma_i^2
\mathbb E_{x\sim p_{\mathrm{data}}}
\mathbb E_{\tilde x\sim p_{\sigma_i}(\cdot\mid x)}
\left[\left\|S_\theta(\tilde x,\sigma_i)
-\nabla_{\tilde x}\log p_{\sigma_i}(\tilde x\mid x)\right\|_2^2\right].
$$

Sampling is then done by running $M$ steps of **Langevin dynamics**:

$$
x_i^m
=x_i^{m-1}+\varepsilon_i S_\theta(x_i^{m-1},\sigma_i)
+\sqrt{2\varepsilon_i}\,z_i^m,
$$

where $m=1,\ldots,M$, $\varepsilon_i>0$ is the step size, and $z_i^m\sim\mathcal N(0,I)$.

This is done for $i=N,\ldots,1$, starting at

$$
x_N^0\sim\mathcal N(0,\sigma_{\max}^2I),
\qquad x_i^0\approx x_{i+1}^M.
$$

## 4. SDE Framework

<!-- Page 7: 490.jpg -->

The framework presented by Song et al. in *Score-Based Generative Modeling through Stochastic Differential Equations* uses stochastic differential equations to transform a data distribution into a simpler distribution by slowly adding noise. The SDE can also be reversed to transform the simple prior distribution back into the original data distribution by slowly removing the noise.

The reverse SDE uses a time-dependent score, which can be learned to then numerically integrate the reverse equation and sample from the data distribution.

Song et al. show that both DDPM and NCSN can be expressed inside this framework. We will first present the framework in more detail and then explain how DDPM and NCSN fit within it.

The SDE framework consists of constructing a diffusion process $\{x(t)\}_{t=0}^T$ indexed by a continuous time variable such that

$$
x(0)\sim p_{\mathrm{data}},\qquad x(T)\sim p_T,
$$

where $p_T$ is a prior distribution that is easy to sample from (usually Gaussian).

<!-- Page 8: 491.jpg -->

We have the following SDE:

$$
\boxed{dx=f(x,t)\,dt+g(t)\,dw_t.}
$$

Here $f(x,t)$ is a vector-valued drift coefficient, $g(t)$ is a scalar-valued diffusion coefficient, and $dw_t$ corresponds to a standard Wiener process, a continuous-time stochastic process describing random movement over time.

In all the cases we will deal with, $f(x,t)=f(t)x$. This SDE indicates that the evolution of the distribution of $x$ is governed by a drift term, $f(x,t)\,dt$, which is a predictable push that changes $x$, and a diffusion term, $g(t)\,dw_t$, which is a random push caused by Gaussian noise.

This SDE has a corresponding **reverse-time SDE**:

$$
\boxed{
 dx=\left[f(x,t)-g(t)^2\nabla_x\log p_t(x)\right]d\bar t
 +g(t)\,d\bar w_t.
}
$$

Here $d\bar w_t$ denotes the Wiener process for backward-time evolution (with the equation integrated from $T$ down to $0$).

We can use the reverse SDE to sample from $p_{\mathrm{data}}$ by solving it numerically. First we need to train a neural network to estimate the time-dependent score $\nabla_x\log p_t(x)$. We can do this using denoising score matching:

$$
\theta^*=\arg\min_\theta
\mathbb E_{t\sim\mathcal U(0,T)}\left[
\lambda(t)\mathbb E_{x(0)\sim p_{\mathrm{data}}}
\mathbb E_{x(t)\sim p_{0t}(\cdot\mid x(0))}
\left\|S_\theta(x(t),t)
-\nabla_{x(t)}\log p_{0t}(x(t)\mid x(0))\right\|_2^2
\right],
$$

with, for example, a weighting proportional to the inverse of the expected squared conditional-score norm:

$$
\lambda(t)\propto
\frac{1}{\mathbb E\left[\left\|\nabla_{x(t)}\log p_{0t}(x(t)\mid x(0))\right\|_2^2\right]}.
$$

The transition kernel $p_{0t}(x(t)\mid x(0))$ can be computed as follows.

<!-- Page 9: 492.jpg -->

Start with the linear SDE

$$
dx=f(t)x\,dt+g(t)\,dw_t.
$$

Let

$$
s(t)=\exp\left(\int_0^t f(u)\,du\right),
\qquad y(t)=\frac{x(t)}{s(t)}.
$$

By the product rule,

$$
dy=\frac{dx}{s(t)}+x\,d\left(\frac1{s(t)}\right).
$$

Since

$$
d\left(\frac1{s(t)}\right)
=-\frac{s'(t)}{s(t)^2}\,dt,
$$

we have

$$
dy=\frac{dx}{s(t)}-\frac{s'(t)x}{s(t)^2}\,dt.
$$

Replacing $dx$ by its expression,

$$
dy=\frac{f(t)x\,dt+g(t)\,dw_t}{s(t)}
-\frac{s'(t)x}{s(t)^2}\,dt.
$$

We also have $s'(t)=f(t)s(t)$, so

$$
dy=\frac{g(t)}{s(t)}\,dw_t.
$$

We then integrate the equation:

$$
y(t)=y(0)+\int_0^t\frac{g(\xi)}{s(\xi)}\,dw_\xi.
$$

Therefore

$$
x(t)=s(t)x_0+s(t)\int_0^t\frac{g(\xi)}{s(\xi)}\,dw_\xi.
$$

Let

$$
I_t=\int_0^t\frac{g(\xi)}{s(\xi)}\,dw_\xi.
$$

We can show that $I_t$ is Gaussian with mean zero and variance

$$
\sigma(t)^2=\int_0^t\frac{g(\xi)^2}{s(\xi)^2}\,d\xi.
$$

We can start by partitioning $[0,t]$ with $0=t_0<\cdots<t_N=t$ and approximating $I_t$ by

$$
I_t\approx\sum_{i=0}^{N-1}h(t_i)(w_{t_{i+1}}-w_{t_i})
=\sum_{i=0}^{N-1}h(t_i)\Delta w_i,
$$

with $h(t)=g(t)/s(t)$ and $\Delta w_i$ the Brownian increments.

<!-- Page 10: 493.jpg -->

The Brownian increments $\{\Delta w_i\}_{i=0}^{N-1}$ are independent Gaussian random variables with

$$
\mathbb E[\Delta w_i]=0,
\qquad\operatorname{Var}(\Delta w_i)
=\mathbb E[(\Delta w_i)^2]=\Delta t_i,
$$

where $\Delta t_i=t_{i+1}-t_i$.

These properties let us derive the mean and variance of $I_t$:

$$
\mathbb E[I_t]
=\sum_{i=0}^{N-1}h(t_i)\mathbb E[\Delta w_i]=0.
$$

Furthermore,

$$
\operatorname{Var}(I_t)
=\mathbb E[I_t^2]-\mathbb E[I_t]^2
=\mathbb E[I_t^2],
$$

and

$$
I_t^2
=\left(\sum_{i=0}^{N-1}h(t_i)\Delta w_i\right)^2
=\sum_{i=0}^{N-1}h(t_i)^2(\Delta w_i)^2
+\sum_{i\ne j}h(t_i)h(t_j)\Delta w_i\Delta w_j.
$$

Since $\Delta w_i$ and $\Delta w_j$ are independent for $i\ne j$,

$$
\mathbb E[\Delta w_i\Delta w_j]
=\mathbb E[\Delta w_i]\,\mathbb E[\Delta w_j]=0.
$$

Thus

$$
\begin{aligned}
\mathbb E[I_t^2]
&=\sum_{i=0}^{N-1}h(t_i)^2\mathbb E[(\Delta w_i)^2]\\
&=\sum_{i=0}^{N-1}h(t_i)^2\Delta t_i.
\end{aligned}
$$

Hence $\mathbb E[I_t]=0$ and

$$
\operatorname{Var}(I_t)
=\sum_{i=0}^{N-1}h(t_i)^2\Delta t_i.
$$

Since $I_t$ is a sum of independent Gaussians, it is also Gaussian. When $N\to\infty$, it converges to

$$
I_t\sim\mathcal N\left(0,\int_0^t\frac{g(\xi)^2}{s(\xi)^2}\,d\xi\,I\right)
=\mathcal N(0,\sigma(t)^2I).
$$

The law of $x(t)$ follows:

$$
\boxed{
 x(t)\mid x(0)\sim
 \mathcal N\left(s(t)x(0),s(t)^2\sigma(t)^2I\right).
}
$$

<!-- Page 11: 494.jpg -->

Once we have a trained neural network, sampling can be done by discretizing the reverse SDE. A schematic discrete reverse step, as recorded in the notes, is

$$
\begin{aligned}
x(t_i)= {}&x(t_{i+1})-f(t_{i+1})x(t_{i+1})\\
&-g(t_{i+1})^2 S_\theta(x(t_{i+1}),t_{i+1})
+g(t_{i+1})z_{i+1},
\end{aligned}
$$

where $z_{i+1}\sim\mathcal N(0,I)$. *The handwritten expression suppresses the time-step factors; an actual Euler–Maruyama implementation must account for the step size and the direction of time.*

The numerical solver used to solve the SDE is what Song et al. call the **predictor** part of their sampling algorithm. This predictor is combined with a **corrector**, in the form of annealed Langevin dynamics, which helps improve the sample distribution that was just updated by the predictor.

### Probability-flow ODE

In addition to the reverse SDE, we can derive the following probability-flow ODE from the SDE:

$$
\boxed{
 dx=\left[f(t)x-\frac12g(t)^2\nabla_x\log p_t(x)\right]dt.
}
$$

This ODE induces the same time-dependent marginals as the SDE and constitutes another, deterministic, way of generating samples.

### VP-SDE and VE-SDE

We will now look at how DDPM and NCSN fit within the SDE framework.

#### NCSN and the VE-SDE

Each perturbation kernel can be derived from the following Markov chain:

$$
x_i=x_{i-1}+\sqrt{\sigma_i^2-\sigma_{i-1}^2}\,z_{i-1},
\qquad i=1,\ldots,N,
$$

where $z_{i-1}\sim\mathcal N(0,I)$ and $x_0\sim p_{\mathrm{data}}$.

<!-- Page 12: 495.jpg -->

For $i=1,\ldots,N$, set

$$
x_i=x\left(\frac{i}{N}\right),\qquad
\sigma_i=\sigma\left(\frac{i}{N}\right),\qquad
z_i=z\left(\frac{i}{N}\right).
$$

When $N\to\infty$, these quantities become continuous functions of $t\in[0,1]$. Let $\Delta t=1/N$. For $t\in\{0,1/N,\ldots,(N-1)/N\}$,

$$
x(t+\Delta t)
=x(t)+\sqrt{\sigma(t+\Delta t)^2-\sigma(t)^2}\,z(t).
$$

Therefore

$$
dx=\sqrt{d[\sigma(t)^2]}\,z(t)
=\sqrt{\frac{d[\sigma(t)^2]}{dt}}\,\sqrt{dt}\,z(t).
$$

Brownian motion satisfies $dw_t\sim\mathcal N(0,dt\,I)$, so $\sqrt{dt}\,z(t)\sim\mathcal N(0,dt\,I)$. We obtain the SDE

$$
\boxed{dx=\sqrt{\frac{d[\sigma(t)^2]}{dt}}\,dw_t.}
$$

The process defined by this SDE has unbounded variance, which is why it is called the **variance-exploding SDE (VE-SDE)**.

NCSN defines a geometric noise schedule

$$
\sigma_i=\sigma_{\min}
\left(\frac{\sigma_{\max}}{\sigma_{\min}}\right)^{\frac{i-1}{N-1}},
$$

which, in the continuous-time limit, corresponds to

$$
\sigma(t)=\sigma_{\min}
\left(\frac{\sigma_{\max}}{\sigma_{\min}}\right)^t.
$$

This yields the following SDE:

$$
\boxed{
 dx=\sigma_{\min}
 \left(\frac{\sigma_{\max}}{\sigma_{\min}}\right)^t
 \sqrt{2\log\left(\frac{\sigma_{\max}}{\sigma_{\min}}\right)}\,dw_t.
}
$$

<!-- Page 13: 496.jpg -->

This SDE induces the perturbation kernel

$$
p_{0t}(x(t)\mid x(0))
=\mathcal N\left(x(t);x(0),
\sigma_{\min}^2\left(\frac{\sigma_{\max}}{\sigma_{\min}}\right)^{2t}I\right)
$$

as used in NCSN (with the usual choice or approximation for the initial noise variance).

The associated objective function is also the same:

$$
\theta^*=\arg\min_\theta
\mathbb E_{t\sim\mathcal U(0,1)}\left[
\lambda(t)\mathbb E_{x(0)\sim p_{\mathrm{data}}}
\mathbb E_{x(t)\sim p_{0t}(\cdot\mid x(0))}
\left\|S_\theta(x(t),t)
-\nabla_{x(t)}\log p_{0t}(x(t)\mid x(0))\right\|_2^2
\right],
$$

with

$$
\lambda(t)\sim\frac1{\mathbb E\left[\left\|\nabla_{x(t)}\log p_{0t}(x(t)\mid x(0))\right\|_2^2\right]}
\propto\sigma(t)^2.
$$

When it comes to sampling, NCSN sampling corresponds to the predictor–corrector framework, where the predictor is the identity function and the corrector is annealed Langevin dynamics.

#### DDPM and the VP-SDE

DDPM uses the following Markov chain:

$$
x_i=\sqrt{1-\beta_i}\,x_{i-1}+\sqrt{\beta_i}\,z_{i-1},
\qquad i=1,\ldots,N.
$$

We can write $\beta_i=\bar\beta_i/N$, where $\bar\beta_i$ is interpreted as a noise rate. When $N\to\infty$, $\bar\beta_i$ becomes a continuous function $\beta(t)$, with $\beta(i/N)=\bar\beta_i$, and

$$
x(t+\Delta t)
=\sqrt{1-\beta(t+\Delta t)\Delta t}\,x(t)
+\sqrt{\beta(t+\Delta t)\Delta t}\,z(t).
$$

Using a first-order approximation,

$$
\begin{aligned}
x(t+\Delta t)
&\simeq\left(1-\frac12\beta(t)\Delta t\right)x(t)
+\sqrt{\beta(t)\Delta t}\,z(t).
\end{aligned}
$$

So

$$
\boxed{dx=-\frac12\beta(t)x(t)\,dt+\sqrt{\beta(t)}\,dw_t.}
$$

<!-- Page 14: 497.jpg -->

The process defined by this SDE has bounded variance, which is why it is called the **variance-preserving SDE (VP-SDE)**.

DDPM defines

$$
\beta_i=\frac{\bar\beta_{\min}}{N}
+\frac{i-1}{N(N-1)}(\bar\beta_{\max}-\bar\beta_{\min}),
\qquad i=1,\ldots,N.
$$

Here $\bar\beta_{\min}$ and $\bar\beta_{\max}$ are rates. Therefore,

$$
\beta(t)=\bar\beta_{\min}
+t(\bar\beta_{\max}-\bar\beta_{\min}),\qquad t\in[0,1],
$$

is the continuous-time noise rate used in our SDE:

$$
\boxed{
 dx=-\frac12\left[\bar\beta_{\min}
+t(\bar\beta_{\max}-\bar\beta_{\min})\right]x\,dt
+\sqrt{\bar\beta_{\min}
+t(\bar\beta_{\max}-\bar\beta_{\min})}\,dw_t.
}
$$

The corresponding perturbation kernel is

$$
p_{0t}(x(t)\mid x(0))
=\mathcal N\left(x(t);\sqrt{\alpha(t)}x(0),(1-\alpha(t))I\right),
$$

with $t\in[0,1]$ and

$$
\alpha(t)
=\exp\left[-\frac12t^2(\bar\beta_{\max}-\bar\beta_{\min})
-t\bar\beta_{\min}\right].
$$

We can show that $\alpha(t)$ corresponds to DDPM's

$$
\bar\alpha_i=\prod_{j=1}^{i}(1-\beta_j).
$$

Since $\beta_i$ is of order $10^{-4}$ to $10^{-1}$, we can write, approximately,

$$
\begin{aligned}
\prod_{i=1}^{r}(1-\beta_i)
&\approx\prod_{i=1}^{r}e^{-\beta_i}
=e^{-\sum_{i=1}^{r}\beta_i}\\
&=\exp\left[-\frac rN\bar\beta_{\min}
-\frac{r(r-1)}{2N(N-1)}
(\bar\beta_{\max}-\bar\beta_{\min})\right].
\end{aligned}
$$

We can reparameterize $t=r/N\in[0,1]$. Since $(r-1)/(N-1)\approx r/N\approx t$,

$$
\bar\alpha_r\approx
\exp\left[-t\bar\beta_{\min}
-\frac{t^2}{2}(\bar\beta_{\max}-\bar\beta_{\min})\right]
=\alpha(t).
$$

This means DDPM's perturbation kernel matches the one induced by the SDE framework.

<!-- Page 15: 498.jpg -->

The objective function is the same as previously, with

$$
\lambda(t)\propto 1-\alpha(t).
$$

Finally, we can show that DDPM's sampling scheme is equivalent, to first order, to the predictor induced by the VP-SDE:

$$
\begin{aligned}
x_i
&=\frac{1}{\sqrt{1-\beta_{i+1}}}
\left(x_{i+1}+\beta_{i+1}S_{\theta^*}(x_{i+1},i+1)\right)
+\sqrt{\beta_{i+1}}\,z_{i+1}\\
&\simeq\left(1+\frac12\beta_{i+1}\right)
\left(x_{i+1}+\beta_{i+1}S_{\theta^*}(x_{i+1},i+1)\right)
+\sqrt{\beta_{i+1}}\,z_{i+1}\\
&\simeq\left(1+\frac12\beta_{i+1}\right)x_{i+1}
+\beta_{i+1}S_{\theta^*}(x_{i+1},i+1)
+\sqrt{\beta_{i+1}}\,z_{i+1}\\
&\simeq\left(2-\sqrt{1-\beta_{i+1}}\right)x_{i+1}
+\beta_{i+1}S_{\theta^*}(x_{i+1},i+1)
+\sqrt{\beta_{i+1}}\,z_{i+1}.
\end{aligned}
$$

This is the SDE-induced sampler. Overall, DDPM's sampling corresponds to the predictor–corrector framework with ancestral sampling as the predictor and the identity as the corrector.

## 5. EDM

<!-- Page 16: 499.jpg -->

The **EDM** paper reformulates the SDE framework to make it easier to use in practice by clearly separating the concrete design choices. It starts by analyzing the ODE because its deterministic properties make it easier to understand the impact of each design choice, with the gained insights carrying over to the SDE.

The probability-flow ODE is

$$
 dx=\left[f(t)x-\frac12g(t)^2\nabla_x\log p_t(x)\right]dt.
$$

The authors argue that the initial ODE defined by Song et al. in the SDE paper using the functions $f$ and $g$ is not convenient to use in practice. Indeed, $f$ and $g$ are used to infer

$$
s(t)=\exp\left(\int_0^t f(u)\,du\right),
$$

and

$$
\sigma(t)=\sqrt{\int_0^t\left(\frac{g(u)}{s(u)}\right)^2du},
$$

which are then used to define the transition kernels

$$
p_{0t}(x(t)\mid x(0))
=\mathcal N\bigl(x(t);s(t)x(0),s(t)^2\sigma(t)^2I\bigr),
$$

and the marginals

$$
p_t(x)=\int p_{0t}(x(t)\mid x(0))p_{\mathrm{data}}(x(0))\,dx(0).
$$

But in practice, we need these marginals for training and sampling, not the original functions $f$ and $g$. It would therefore be more appropriate to define the ODE directly using $s$ and $\sigma$, which determine the transition kernels and marginals.

<!-- Page 17: 500.jpg -->

We can start by expressing $p_t(x)$ in closed form:

$$
\begin{aligned}
p_t(x)
&=\int_{\mathbb R^d}p_{0t}(x\mid x_0)p_{\mathrm{data}}(x_0)\,dx_0\\
&=\int_{\mathbb R^d}p_{\mathrm{data}}(x_0)
\mathcal N\bigl(x;s(t)x_0,s(t)^2\sigma(t)^2I\bigr)\,dx_0\\
&=s(t)^{-d}\int_{\mathbb R^d}p_{\mathrm{data}}(x_0)
\mathcal N\left(\frac{x}{s(t)};x_0,\sigma(t)^2I\right)\,dx_0\\
&=s(t)^{-d}p\left(\frac{x}{s(t)},\sigma(t)\right),
\end{aligned}
$$

where

$$
p(x,\sigma)=p_{\mathrm{data}}*\mathcal N(0,\sigma^2I)
$$

is the data distribution convolved with Gaussian noise.

We can then express $f$ and $g$ as functions of $s$ and $\sigma$:

$$
f(t)=\frac{\dot s(t)}{s(t)},
\qquad
g(t)=s(t)\sqrt{2\dot\sigma(t)\sigma(t)}.
$$

Substituting into the ODE, we obtain

$$
\boxed{
 dx=\left[
 \frac{\dot s(t)}{s(t)}x
 -s(t)^2\dot\sigma(t)\sigma(t)
 \nabla_x\log p\left(\frac{x}{s(t)},\sigma(t)\right)
 \right]dt.
}
$$

Parameterizing the ODE with $s(t)$ and $\sigma(t)$ directly informs us about the diffusion process: $\sigma(t)$ changes the signal-to-noise ratio, while $s(t)$ changes the overall magnitude of the image.

We can also rewrite the ODE as a function of a denoiser $D_\theta$. Using

$$
\nabla_x\log p\left(\frac{x}{s(t)},\sigma(t)\right)
=\frac1{s(t)}\nabla_{\tilde x}\log p(\tilde x,\sigma(t)),
\qquad \tilde x=\frac{x}{s(t)},
$$

and the denoising identity

$$
\nabla_{\tilde x}\log p(\tilde x,\sigma(t))
=\frac{D_\theta(\tilde x,\sigma(t))-\tilde x}{\sigma(t)^2},
$$

we obtain

$$
\boxed{
\frac{dx}{dt}
=\left(\frac{\dot\sigma(t)}{\sigma(t)}
+\frac{\dot s(t)}{s(t)}\right)x
-\frac{\dot\sigma(t)s(t)}{\sigma(t)}
D_\theta\left(\frac{x}{s(t)},\sigma(t)\right).
}
$$

<!-- Page 18: 501.jpg -->

The ODE can then be solved by numerical integration. This requires us to choose a numerical solver and the time-step schedule.

Since errors tend to be larger at low noise levels, the authors recommend designing the time-step schedule so that it decreases monotonically with decreasing $\sigma$. These errors can be further minimized using a second-order solver like **Runge–Kutta**, at the cost of more neural function evaluations.

Another way to reduce truncation errors is to design the ODE so that curvature is limited toward the origin. In particular, EDM recommends setting

$$
(\sigma(t),s(t))=(t,1),
$$

which approximately corresponds to straight lines.

Taken together, these changes yield significant improvements in the FID score of generated images.

Deterministic sampling using the ODE offers benefits such as faster sampling, but it can also affect the quality of generated images. We can address this by introducing stochasticity through the reverse SDE:

$$
\boxed{
 dx_\pm=
 \left[-\dot\sigma(t)\sigma(t)\nabla_x\log p(x,\sigma(t))
 \;\pm\;\beta(t)\sigma(t)^2\nabla_x\log p(x,\sigma(t))\right]dt
 +\sqrt{2\beta(t)}\,\sigma(t)\,dw_t.
}
$$

Here $s(t)$ is set to $1$, and $dx_+$ and $dx_-$ represent the SDEs moving forward and backward in time, respectively. The reverse SDE adds a Langevin term, which—through noise injection—helps correct errors made in earlier steps.

<!-- Page 19: 502.jpg -->

The stochastic sampler introduced by EDM is based on this idea of adding noise to correct the errors introduced by the deterministic ODE solver. Starting at step $i$ with noise level $t_i$, it begins by injecting noise to move the sample to a higher noise level $\hat t_i$. It then solves the ODE backward from $\hat t_i$ to $t_{i+1}$ in a single step.

However, injecting too much noise can also degrade the quality of the final image. That is why the EDM sampler controls when to add noise by checking whether the current time step belongs to the interval delimited by the hyperparameters $[S_{\mathrm{tmin}},S_{\mathrm{tmax}}]$. If it does, noise is injected. The parameter $S_{\mathrm{churn}}$ controls the quantity of noise injected.

### Preconditioning and training

The EDM framework proposes wrapping the neural network in a **signal-management / preconditioning layer** that makes it easier to train. In particular, it is good practice to keep the network's input and output signal magnitudes fixed (e.g., unit variance) to avoid large variations in gradient magnitudes.

In our case, the input is $x=y+n$, where $y$ is a clean image and $n\sim\mathcal N(0,\sigma^2I)$ is noise. EDM chooses the following formulation for the denoiser:

$$
\boxed{
D_\theta(x,\sigma)
=c_{\mathrm{skip}}(\sigma)x
+c_{\mathrm{out}}(\sigma)
F_\theta\bigl(c_{\mathrm{in}}(\sigma)x,c_{\mathrm{noise}}(\sigma)\bigr).
}
$$

Here $c_{\mathrm{skip}}(\sigma)x$ is a skip connection and $F_\theta$ is the neural network. It takes as input $x$ multiplied by the scale $c_{\mathrm{in}}(\sigma)$, which sets its variance to $1$. The network output is then scaled back to its original scale with $c_{\mathrm{out}}(\sigma)$. In this way, the network predicts a unit-variance output, which is easier to learn.

<!-- Page 20: 503.jpg -->

The skip connection behaves in different ways depending on the input noise level.

At **low noise levels**, it is a good idea to predict the noise added to the input (or rather its opposite). That predicted noise is then scaled back down by $c_{\mathrm{out}}(\sigma)$, reducing errors made by the network, and subtracted from the noised image coming from the skip connection, for which $c_{\mathrm{skip}}(\sigma)\approx1$, to yield the final predicted image.

At **high noise levels**, predicting the noise is a bad idea because whatever error the network makes would be amplified by $c_{\mathrm{out}}(\sigma)$. Instead, we directly predict the final image and deactivate the skip connection by setting $c_{\mathrm{skip}}(\sigma)\approx0$.

The skip connection acts as a continuous switch between low and high noise levels. When dealing with intermediate noise levels, the network predicts a mix of noise and image.

Using this preconditioning, the training loss is

$$
\mathbb E_{\sigma,y,n}\left[
\lambda(\sigma)c_{\mathrm{out}}(\sigma)^2
\left\|
F_\theta\bigl(c_{\mathrm{in}}(\sigma)(y+n),c_{\mathrm{noise}}(\sigma)\bigr)
-\frac{y-c_{\mathrm{skip}}(\sigma)(y+n)}{c_{\mathrm{out}}(\sigma)}
\right\|_2^2
\right],
$$

with $\lambda(\sigma)$ set so that the effective weight $\lambda(\sigma)c_{\mathrm{out}}(\sigma)^2$ is uniform:

$$
\lambda(\sigma)c_{\mathrm{out}}(\sigma)^2=1
\qquad\text{for all noise levels.}
$$

---

*End of the photographed notes.*
