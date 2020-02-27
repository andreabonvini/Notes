### Wiener Filter

Typical deterministic filters are designed for a desired frequency response. However, the design of the Wiener Filter takes a different approach. While using deterministic filters we assume to have knowledge of the spectral properties of the original signal and the noise, the Wiener Filter instead seeks the linear time-invariant filter whose output would come as close to the original signal as possible.

The *Wiener Filter* is a *non-recursive* filter used to produce an estimate of a desired or target random process by linear time-invariant filtering an observed noisy process, assuming known *stationary* signal and noise spectra, and additive noise. The Wiener filter minimizes the mean square error between the estimated random process and the desired process.

The hypothesis behind the Wiener Filter are:

- $y(k) = x + v(k) \space$

	Where $x$ is the signal we are interested in and $v$ is a random noise. 
	$x$  and  $v$ are not necessarily linked by an additive relationship.

- $x$ and $v$ are stationary stochastic processes.

- $M$ (number of samples) must be sufficiently large ( $M\to \infty $).

Given these hypothesis *Wiener* designed a LTI filter able to minimize the quadratic error.

The filter is non-recursive and $h(i)$ will be the coefficients of the Wiener Filter.
Since we have to "clean" the $y$ signal we must choose the right values of $h(i)$ in order to reduce the effect of the noise. To do so we compute the derivative of the error function *w.r.t.* the $h(i)$ coefficients and put it to $0$ to find the minimum.
$$
\hat{x} = \sum_{i= 1}^{M}h(i)\cdot y(i) \\

p_{e} = E[e^{2}] = E[(x-\hat{x})^{2}] = E[(x - \sum_{i= 1}^{M}h(i)\cdot y(i))^{2}] \\

\frac{\partial{p_{e}}}{\partial{h(j)}} = -2E[(x - \sum_{i= 1}^{M}h(i)\cdot y(i))]\cdot y(j) = 0  \\

\text{   $j =
1,2,$..$,M$} \\
$$
We drop $(-2)$ and obtain
$$
E[(x-\sum_{i=1}^{M}h(i) \cdot y(i)) ]\cdot y(j) = 0 \\
\sum_{i}^{M}h(i) \cdot E[y(i) \cdot y(j)] = E[x \cdot y(j)] \\
$$
We define
$$
E[y(i) \cdot y(j)] = p_{y}(i,j) \text{   (autocorrelation)} \\
E[x \cdot y(j)] = p_{xy}(j)  \text{   (correlation between x and y)}
$$
And reach the *Wiener-Hopf* equation:
$$
\color{blue}{\sum_{i}^{M}h(i) \cdot p_y(i,j) = p_{xy}(j)} \ \ \ \ \ \ \ \forall j\in [1,M]\\
h(i) = Unknown \\
p_y(i,j) = Known \\
p_{xy}(j) = Known \\\ \\
$$
In *matricial form*:
$$
\begin{cases}
\overline{h} = p_{y}^{-1}\overline{p_{xy}} \\
\hat{x} = \overline{h}^{T}\overline{y} = \overline{p_{xy}}^{T}p_{y}^{-1}\overline{y} \\
p_{e} = E[x^{2}] - \overline{p_{xy}}^{T}p_{y}^{-1}\overline{p_{xy}}
\end{cases}
$$
The *Wiener Filter* is *optimal* among the time-invariant linear filters but if the hypothesis are not fulfilled is *sub-optimal*.

If the spectra of the signal and of the noise are both *rational* the *wiener filter* in the frequency domain corresponds to:
$$
H(\omega)=\frac{\Phi_{XX}(\omega)}{\Phi_{XX}(\omega)+\Phi_{NN}(\omega)}
$$

