### Inhomogenous Poisson Process in the Time Domain

In an Inhomogeneous Poisson Process we define the Poisson Rate as
$$
\Lambda(t_0,t) = \int_{t_0}^t\lambda(u)du
$$
where $\lambda(u)$ is a locally integrable positive function.

We define $N(t)$ as the number of events in $(\ 0,t\ ]$, the Poisson *probability mass function* extended to the inhomogeneous case states that the probability of observing exactly $k$ event in the time interval $(t_0,t]$ is equal to:
$$
Pr(N(t)-N(t_0) =  k) \\
= \frac{\Lambda(t_0,t)^k}{k!}e^{-\Lambda(t_0,t)}\\
= \frac{\left[\int_{t_0}^{t}\lambda(u)du\right]^k}{k!}e^{-\int_{t_0}^{t}\lambda(u)du}
$$
In order to build the inter-arrival intervals *pdf* we want to define the probability that an event occur in a given time interval $(t_0,t]$.

We firstly define the probability that the next event do *not* occur in the time interval $(t_0,t]$, i.e. the next event arrival time $t_{\text{next}}$ is greater than $t$ :
$$
Pr(t_{\text{next}}> t)\\
=Pr(\ (N(t)-N(t_0)\ ) =  0)\\
= \frac{\left[{\int_{t_0}^{t}}\lambda(u|H_u)du\right]^0}{0!}e^{-\int_{t_0}^{t}\lambda(u)du}\\
= e^{-\int_{t_0}^{t}\lambda(u)du}
$$

From here we know that 
$$
Pr(t_0 < t_{\text{next}} < t) = 1-e^{-\int_{t_0}^{t}\lambda(u)du}
$$
which represents the *cumulative distribution function* (CDF) up to time $t$, the *probability density function* $p(t)$ of the next event time is given by the derivative of CDF
$$
p(t) = \frac{\part}{\part t}\left(1-e^{-\int_{t_0}^{t}\lambda(u)du}\right)
$$

$$
p(t) = \lambda(t) e^{-\int_{t_0}^{t}\lambda(u)du}
$$