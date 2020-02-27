####  Adaptive Filtering

Here we present an approach to signal filtering using an *adaptive filter* that is in some sense self-designing (really self-optimizing). The adaptive filter described here bases its own "design" (its internal adjustment settings) upon *estimated* (measured) statistical characteristics of input and output signals. The statistics are not measured explicitly and then used to design the filter; rather, the filter design is accomplished in a single process by a recursive algorithm that automatically updates the system adjustments with the arrival of each new data sample. How do we build such system?

A set of stationary input signals is weighted and summed to form an output signal. The input signals in the set are assumed to occur *simultaneously* (this is just for the general framework, we'll see then how to apply this method in order to model an autoregressive model!) and discretely in time. The $j_{th}$ set of input signals is designated by the vector $\mathbf{X}^T(j) = [x_1(j),x_2(j),....x_n(j)]$ , the set of weights is designed by the vector $\mathbf{W}^T(j) = [w_1(j),w_2(j),...,x_n(j)]$, the $j_{th}$ output signal is:
$$
y(j) = \sum_{l=1}^{n}w_l(j)x_l(j)
$$
This can be written in matrix form as:
$$
  y(j) = \bold{W}^T(j)\bold{X}(j) = \bold{X}^T(j)\bold{W}(j)
$$
 Denoting the desired response for the $j_{th}$ set of input signals as $ d(j) $, the error at the $j_{th}$ time is:
$$
  \bold{\epsilon}(j) = d(j) - \bold{y}(j) = d(j) - \bold{W}^T(j)\bold{X}(j)
$$
 The square of this error is:
$$
  \bold{\epsilon^{2}}(j) = d^{2}(j) -2d(j)\bold{X}^T(j)\bold{W}(j) + \bold{W}^T(j)\bold{X}(j)\bold{X}^T(j)\bold{W}(j)
$$
 Which can be rewritten as
$$
E[\bold{\epsilon^{2}}(j)] = d^{2}(j) -2\bold{\Phi}^T(x,d)\bold{W}(j) + \bold{W}^T(j)\bold{\Phi}(x,x)\bold{W}(j)
$$
  where the vector of cross-correlation between the input signals and the desired response is defined as
$$
  \bold{\Phi}(x,d) =
    	E\left[ {\begin{array}{cc}
    	x_1(j)d(j) \\
    	x_2(j)d(j) \\
    	.\\
    	.\\
    	.\\
    	x_n(j)d(j)
    	\end{array}} \right] 
$$
  and where the correlation matrix of the input signals is defined as
$$
   E[\bold{X}(j)\bold{X}^T(j)] =
    	E\left[ {\begin{array}{cc}
    	x_1(j)x_1(j) & x_1(j)x_2(j) & ... &  \\
    	x_2(j)x_1(j) & x_1(j)x_2(j) & ... &  \\
    	.\\
    	.\\
    	.\\
    	& & x_n(j)x_n(j)
    	\end{array}} \right] = \bold{\Phi}(x,x)
$$
It may be observed that for stationary input signals, the mean-square error is precisely a second-order function of the weights. The mean-square-error performance function may be visualized as a *bowl* shaped surface, a parabolic function of the weight variables. The adaptive process has the job of continually seeking the "bottom of the bowl". A means of accomplishing this by the well-known method of steepest descent is discussed below.

In the nonstationary case, the bottom of the bowl *may be moving*, while the orientation and curvature of the bowl may be changing. The adaptive process has to track the bottom of the bowl when inputs are non-stationary. It will be assumed that the input and desired-response signals are stationary. Here we are concerned with transient phenomena that take place when a system is adapting to an unknown stationary input process, and in addition, it is concerned with steady-state behaviour after the adaptive transients die out.

The method of steepest descent uses gradients of the performance surface in seeking its minimum. The gradient at any point on the performance surface may be obtained by differentiating the mean-square-error function with respect to the weight vector.

The gradient is
$$
\frac{\part}{\part \mathbf{W}(j)}\left(d^{2}(j) -2\bold{\Phi}^T(x,d)\bold{W}(j) + \bold{W}^T(j)\bold{\Phi}(x,x)\bold{W}(j)\right)=\\
\underset{N\times 1}{\underbrace{\nabla[\overline{\epsilon}^{2}(j)]}} = -2\underset{N\times 1}{\underbrace{\bold{\Phi}(x,d)}} + 2\underset{N\times 1}{\underbrace{\bold{\Phi}(x,x)\bold{W}(j)}}
$$
Note that
$$
  \frac{\part}{\mathbf{W}(j)}\left(-2\bold{\Phi}^T(x,d)\bold{W}(j)\right) = -2\underset{N\times 1}{\underbrace{\bold{\Phi}(x,d)}}\\
  
  \frac{\part}{\mathbf{W}(j)}\left(\bold{W}^T(j)\bold{\Phi}(x,x)\bold{W}(j)\right) = 2\underset{N\times 1}{\underbrace{\bold{\Phi}(x,x)\bold{W}(j)}}
$$
To find the "optimal" weight vector $\bold{W}_{LMS}$ that yields the last mean-square error, set the gradient to zero. Accordingly:
$$
  \underset{N\times 1}{\underbrace{\bold{\Phi}(x,d)}} = \underset{N\times N}{\underbrace{\bold{\Phi}(x,x)}}\underset{N\times 1}{\underbrace{\bold{W}_{LMS}}} \\
  
  \underset{N\times 1}{\underbrace{\bold{W}_{LMS}}} = \underset{N\times N}{\underbrace{\bold{\Phi^{-1}}(x,x)}}\underset{N\times 1}{\underbrace{\bold{\Phi}(x,d)}}
$$
The equation above is the *Wiener-Hopf equation* in matrix form.

However it is also possible to evaluate the optimal vector *iteratively*, where in each step we change the vector *proportionally to the negative of the gradient vector*.
$$
\bold{W}_{j+1} = \bold{W}_j - \mu\nabla_j
$$
where $\mu$ is a scalar that controls the stability and rate of convergence of the algorithm. It is easy to demonstrate that
$$
  \nabla[\overline{\epsilon}^{2}(j)] = -2\bold{\Phi}(x,d) + 2\bold{\Phi}(x,x)\bold{W}(j) = -2\left( d(j) - \bold{W}^T(j)\bold{X}(j)\right)\bold{X}_j=-2\bold{\epsilon}_j\bold{X}_j
$$
  And the optimal weight vector is estimated as
$$
  \bold{\overline{W}}_{j+1} = \bold{\overline{W}}_j + 2\mu\bold{\epsilon}_j\bold{X}_j
$$
It's important to observe that in this case we don't have to compute neither the *autocorrelation* neither the *cross-correlation* matrices, which is really computationally convenient!

Morevover the equation above can be visualized in a block diagram as you can see in this figure:

  ![](/Users/z051m4/Desktop/University/Cerutti/images/ADA1.png)

A necessary and sufficient condition for convergence is:
$$
0 < \mu < \lambda_{max}^{-1}
$$
where $\lambda_{max}$ is the largest eigenvalue of the correlation matrix $\bold{\Phi}(x,x)$.

But more importantly we can use the *adaptive filter* in order to model an *Autoregressive model* !  Our target (which whas $d_j$) becomes $x_j$ and our inputs become $x_{j-1}$ , $x_{j-2}$ , $\dots$ , $x_{j-n+1}$.

  ![](/Users/z051m4/Desktop/University/Cerutti/images/ADA2.png)

  It is common practice to add a new input $x_0=1$ in order to associate it with a learnable bias $w_0$ .

  Let's talk now about another important topic: *Adaptive Noise Cancelling* .

  ![](/Users/z051m4/Desktop/University/Cerutti/images/ANC1.PNG)

Consider the following problem. A signal $s(t)$ is contaminated with an additive noise $n_0(t)$ , and with another noise, $\eta(t)$ ; we assume that $s$ , $n_0$ and $\eta$ are uncorrelated. The noise $n_0$ is generated by a white noise process $n(t)$ that has passed an unknown linear filter , $H_1$ . The additive noise $n_0(t)$ is therefore a colored noise. Assume also that we have a reference signal $x(t)$ consisting of a white noise $\xi(t)$ and another noise $n_r(t)$ . The second noise is the result of the noise process $n(t)$ , contributing to the primary noise, but after another unknown linear filter, $H_2$ . Note that here we have:
$$
N(z) = H_1(z)^{-1}N_0(z) = H_2(z)^{-1}N_r(z)
$$
So
$$
N_0(z) = H_2(z)^{-1}H_1(z)N_r(z)
$$
Where $N$, $N_0$, $N_r$ , $H_1(z)$ and $H_2(z)$ are the $z$ transforms of $n(t)$ , $n_0(t)$ , $n_r(t)$ , $h_1(t)$ and $h_2(t)$ respectively. We assume that the auxiliary noises $\eta(t)$ and $\xi(t)$ are white and uncorrelated with one another, with $n(t)$, and with the signal $s(t)$.

The concept of the adaptive noise canceller is as follows. An adaptive estimate $n_0(j)$ , denoted by $\hat{n}_0(j)$ , is calculated by the adaptive LMS filter. As shown before this filter is an adaptive AR filter estimating the unknown filter $H_2^{-1}(z^{-1})H_1(z^{-1})$ by means of the reference input $n_r(j)$ and the error. Adaptive noise cancelling filters have been extensively used in biomedical signal processing and many other applications.

For further details check the file `cohen-biomedical-signal-processing.pdf` at page `151`