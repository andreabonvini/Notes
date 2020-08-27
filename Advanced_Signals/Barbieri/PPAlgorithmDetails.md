### How we model the Inter Event Probability distribution

We define $p(t|H_t)$ as an *Inverse Gaussian*
$$
p(t|H_t,\theta) = \left[\frac{\theta_{p+1}}{2\pi(t-u_k)^3}\right]^{1/2}e^{-\frac{1}{2}\frac{\theta_{p+1}\left(t-u_k-\mu(H_{u_k},\theta)\right)^2}{\mu(H_{u_k},\theta)^2(t-u_k)}}
$$
Where $H_{u_k} = (\color{green}{u_k},w_k,w_{k-1},\dots,w_{k-p+1})$ , $w_k = u_k-u_{k-1}$ is the $k_{\text{th}}$ inter-event interval (note that we need $\color{green}{u_k}$ just to "reset" our distribution to the last observed event ), $\mu(H_{u_k},\theta)=\theta_0+\sum_{j=1}^{p}\theta_jw_{k-j+1} > 0$ is the mean, $\theta_{p+1}$ is the scale parameter and $\theta = \left(\theta_0,\theta_1,\dots,\theta_{p+1}\right)$.

The useful features that we can extract once estimted this probability distribution are the *mean* and the *std​*
$$
\mu =\mu(H_{u_k},\theta)
$$

$$
\sigma = \sqrt{\frac{\mu(H_{u_k},\theta)^3}{\theta_{p+1}}}
$$

### How we perform Local maximum-likelihood estimation of $\mu$ and $\sigma$

We use a local maximum-likelihood procedure to estimate the time-varying parameter $\theta$. Suppose we have a recording period $(0,T]$, let $l$ be the length of the local likelihood observation interval for $[l,T]$ (since we can just considers the intervals from $(l-l, l]$ to $(T-l,T]$) , and let $\Delta$ define how much the local likelihood time interval is shifted to compute the next parameter update. Let $t^l$ be the local likelihood interval $(t-l,t]$ , and assume that, within $t^l$, we observe $n_t$ event times $t-l<u_1<u_2<\dots <u_{n_t} \le t$. We let $u_{t-l:t} = (u_1,\dots,u_{n_t})$. If $\theta$ is time varying, then at time $t$ we define the local maximum-likelihood estimate $\hat{\theta}_t$ to be the the maximum-likelihood estimate of $\theta$ on $t^l$.

To compute the local maximum-likelihood estimate of $\theta$, we first define the local joint probability density of $u_{t-l:t}$. If we condition on the $p$ inter event intervals preceding each event in $u_{t-l:t}$, then the local observation interval $t^l$ induces right censoring of the inter event interval measurements, because the $n_t+1_{st}$ interval is not completely observed. If we take into account the right censoring the local log likelihood is 
$$
\log p(u_{t-l:t}|\theta_t) = \sum_{i=2}^{n_t}w(t-u_i)\log p(u_i-u_{i-1}|H_{u_{i-1}},\theta_t)\\

+ w(t-u_{n_t})\log\int_{t-u_{n_t}}^{\infty}p(v|H_{u_{n_t}},\theta_t)dv
$$
Where $w(t)$ is a weighting function for the local likelihood estimation. We chose $w(t-u)= e^{-\alpha(t-u)}$, where $\alpha$ is a weighting time constant that governs the degree of influence of a previous observation $u$ on the local likelihood at time $t$. Increasing $\alpha$ decreases the influence of a previous observation on the local likelihood and vice versa.

We use a Newton-Raphson procedure to maximize the local maximum-likelihood estimate of $\theta_t$. Because there is significant overlap between adjacent local likelihood intervals, we start the Newton-Raphson procedure at $t$ with the previous local maximum likelihood estimate at time $t-\Delta$. Once $\hat{\theta}_t$ is computed, the interval $t^l$ is shifted to $(t-l+\Delta, t+ \Delta]$, and the local maximum-likelihood estimation is repeated. The procedure is continued until $t=T$. A key feature of our analysis framework is that we can  estimate the $\mu$ and $\sigma$ parameters in continuous time. This is because the HDIG model.