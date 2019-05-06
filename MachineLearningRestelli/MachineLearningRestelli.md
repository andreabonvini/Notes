# Machine Learning

*A series of notes on the "Machine Learning" course as taught by Marcello Restelli and Francesco Trovò during the second semester of the academic year 2018-2019 at Politecnico di Milano.*

### Theory Questions

*Here are listed all the theory questions since 06/07/2017*

- ***Describe the supervised learning technique denominated Support Vector Machines for classiﬁcation problems.***

- ***Deﬁne the VC dimension and the describe the importance and usefulness of VC dimension in machine learning.***

- ***Describe the diﬀerences existing between the Q-learning and SARSA algorithms***

- ***Describe the supervised learning technique called ridge regression for regression problems.***

  (*William Bonvini*)  
  Ridge Regression is a regularization technique that aims to reduce model complexity and prevent over-fitting which may result from simple linear regression.

  In ridge regression, the cost function is altered by adding a penalty equivalent to the square of the magnitude of the coefficients.
  $$
  cost \space function=\sum_{i=1}^M(y_i-\sum_{j=0}^p(w_j\times x_{ij})\space)^2+\lambda\sum_{j=0}^pw_j^2
  $$
  where ${M}$ is the number of samples and ${p}$ is the number of features.

  The penalty term ${\lambda}$ regularizes the coefficients of the features such that if they take large values the optimization function is penalized. 

  When ${\lambda \to 0}$, the cost function becomes similar to the linear regression cost function. So lowering ${\lambda}$, the model will resemble the linear regression model.

  It is always principled to standardize the features before applying the ridge regression algorithm. Why is this? The coefficients that are produced by the standard least squares method are scale equivariant, i.e. if we multiply each input by ${c}$ then the corresponding coefficients are scaled by a factor of ${\frac{1}{c}}$. Therefore, regardless of how the predictor is scaled, the multiplication of the coefficient and the predictor ${(w_jx_j)}$ remains the same. However, this is not the case with ridge regression, and therefore, we need to standardize the predictors or bring the predictors to the same scale before performing ridge regression. the formula used to do this is given below.
  $$
  \hat{x}_{ij}=\frac{x_{ij}}{\sqrt{\frac{1}{n}\sum^n_{i=1}(x_{ij}-\bar{x}_j)^2}}
  $$
   

  (Source: [towardsdatascience 1](https://towardsdatascience.com/ridge-and-lasso-regression-a-complete-guide-with-python-scikit-learn-e20e34bcbf0b ) & [towardsdatascience 2](https://towardsdatascience.com/regularization-in-machine-learning-76441ddcf99a) )

  

  Since ${\lambda}$ is not defined a priori, we need a method to select a good value for it. We use Cross-Validation for solving this problem: we choose a grid of ${\lambda}$ values, and compute the cross-validation error rate for each value of ${\lambda}$. We then select the value for ${\lambda}$ for which the cross-validation error is the smallest. Finally, the model is re-fit using all of the available observations and the selected value of ${\lambda}$.

  Restelli offers the following cost function notation:

  ${L(w)=L_D(\mathbf{w})+\lambda L_W(\mathbf{w}) }$

  where ${L_D(\mathbf{w})}$ is the error on data terms (e.g. RSS) and ${L_W(\mathbf{w})}$ is the model complexity term.

  By taking ${L(\mathbf{w})=\frac{1}{2} \mathbf{w}^T\mathbf{w}=\frac{1}{2}||\mathbf{w}||^2_2}$

  we obtain:
  $$
  L(\mathbf{w})=\frac{1}{2}\sum_{i=1}^N(t_i-\mathbf{w}^T\phi(\mathbf{x}_i))^2+\frac{\lambda}{2}||\mathbf{w}||^2_2
  $$
  We observe that the loss function is still quadratic in **w**:
  $$
  \hat{\mathbf{w}}_{ridge}=(\lambda \mathbf{I} + \mathbf{\Phi}^T\mathbf{\Phi})^{-1}\mathbf{\Phi}^T\mathbf{t}
  $$
  (Source: Restelli's Slides)

  

  Ridge Regression is, for example, used when the number of samples is relatively small wrt the number of features. Ridge Regression can improve predictions made from new data (i.e. reducing variance) by making predictions less sensitive to the Training Data.

  (Source: [statquests explanation](https://www.youtube.com/watch?v=Q81RR3yKn30))

- ***Describe the diﬀerences existing between the Montecarlo and the Temporal Diﬀerence methods in the model-free estimation of a value function for a given policy.***

- ***Describe the diﬀerence between on-policy and oﬀ-policy reinforcement learning techniques. Make an example of an on-policy algorithm and an example of an oﬀ-policy algorithm.***

- ***Describe the Gaussian Processes model for regression problems***

- ***Describe the value iteration algorithm. Does the algorithm always return the optimal policy?***

- ***Describe the UCB1 algorithm. Is it a deterministic or a stochastic algorithm?***

- ***Describe the logistic regression algorithm and compare it with the perceptron algorithm.***

- ***Describe the SVM algorithm for classiﬁcation problems. Which algorithm can we use to train an SVM? Provide an upper bound to the generalization error of an SVM.***

- ***Describe what are eligibility traces and how they are used in the TD(λ) algorithm. Explain what happens when λ = 0 and when λ = 1.***

- ***Describe and compare the ridge regression and the LASSO algorithms.***

- ***Describe the Principal Component Analysis technique***

- ***Describe and compare Value Iteration and Policy Iteration algorithms.***

- ***Which criteria would you consider for model selection in each one of the following settings:***

  - *A small dataset and a space of simple models;*

  - *A small dataset and a space of complex models;*

  - *A large dataset and a space of simple models;*
  - *A large dataset and a trainer with parallel computing abilities.*

- ***Categorize the following ML problems. For each one of them suggest a set of features which might be useful to solve the problem and a method to solve it.***

  - *Recognize handwritten characters;*

  - *Identify interesting features from an image;*

  - *Teach a robot how to play bowling;*

  - *Predicting car values for second-hand retailers.*

- ***State if the following are applicable to generic RL problems or MAB problems. Motivate your answers.***

  - *We should take into account state-action pairs one by one to estimate the value function;*

  - *Past actions you perform on the MDP might inﬂuence the future rewards you will gain;*

  - *The Markov property holds;*

  - *The time horizon of the episode is ﬁnite.*

- ***After training a perceptron classiﬁer on a given dataset, you ﬁnd that it does not achieve the desired performance on the training set, nor the validation one. Which of the following might be a promising step to take? Motivate your answers.*** 

  -  Use an SVM with a Gaussian Kernel;

  - *Add features by basing on the problem characteristics;*

  - *Use linear regression with a linear kernel, without introducing new features;*

  - *Introduce a regularization term.*

- ***The generic deﬁnition of a policy is a stochastic function $\pi(h_i) = P(a_i|h_i)$ that, given a history $h_i = \{o_1,a_1,s_1,\dots,o_i,a_i,s_i\}$, provides a distribution over the possible actions $\{a_i\}_i$. Formulate the speciﬁc deﬁnition of a policy if the considered problem is:*** 

  - *Markovian, Stochastic, Non-stationary;*

  - *History based, Deterministic, Stationary;*

  - *Markovian, Deterministic, Stationary;*

  - *History based, Stochastic, Non-stationary.*

- ***Describe the ridge regression algorithm and compare it with the Bayesian linear regression approach.***

- ***Deﬁne the VC dimension of a hypothesis space. What is the VC dimension of linear classiﬁers?***

- ***Describe which methods can be used to compute the value function $V^{\pi}$ of a policy $\pi$ in a discounted Markov Decision Process.***

- ***Describe the supervised learning technique denominated Support Vector Machines for classiﬁcation problems.***

- ***Describe the supervised learning technique denominated logistic regression for classiﬁcation problems.***

- ***Describe the policy iteration technique for control problems on Markov Decision Processes***

- ***Describe the two problems tackled by Reinforcement Learning (RL): prediction and control. Describe how the Monte Carlo RL technique can be used to solve these two problems.***

- ***Describe the purpose of using kernels in Machine Learning techniques. How can you construct a valid Kernel? Provide an example of a ML method using kernels and describe the speciﬁc advantage of using them for this method.***

  *(Andrea Bonvini)*

  *Sources:* [Kernel Methods - Bloomberg](https://www.youtube.com/watch?v=m1otj-SdwYw&feature=youtu.be)

  - Kernel methods allow for HUGE feature spaces (even INFINITE dimensional feature spaces) without suffering from computational cost.

  - A method is *kernalized* in inputs only appear inside inner products: $\langle \psi(x),\psi(y)\rangle​$ for $x,y \space \in X​$ 

  - The kernel function corresponding to $\psi$ and inner product $\langle\cdot,\cdot\rangle$ is:

  - $$
    k(x,y) = \langle \psi(x),\psi(y)\rangle
    $$

  $XX^T =n\times d \cdot d\times n=n\times n$ !!!!!! we don't care about the dimension spaceeeee

  ss

  x

  x

  x

  x

  x

  x

  x

  x

  x

  x

  x

  x

  

- ***Tell if the following statements are true or false. Provide adequate motivations to your answer.***

  - *Reinforcement Learning (RL) techniques use a tabular representation of MDPs to handle continuous state and/or action spaces*

  - *We can use data coming from sub-optimal policies to learn the optimal one.*

  - *In RL we always estimate the model of the environment.*

  - *In RL we require to have the model of the environment.*

- ***Consider separately the following characteristics for an ML problem:***

  - *Small dataset*.

  - *Limited computational resources for training.*

  - *Limited computational resources for prediction.*

  - *Prior information on data distribution.*

  *Provide motivations for the use of either a parametric or non-parametric method.*

- ***Categorize the following ML problems:***

  - *Pricing goods for an e-commerce website.*

  - *Teaching a robot to play table tennis.*

  - *Predicting housing prices for real estate.*

  - *Identifying counterfeit notes and coins.*

- ***Consider the Thompson Sampling algorithm. Assume to have the following posterior distributions $Beta_i(αt,βt)$ for arms $A = {a_1,...,a_5}$ rewards, which are distributed as Bernoulli random variables with mean $\mu_i$, and you extracted from them the samples $\hat{r}(a_i)$:***
  $$
  a_1:\space\alpha_t = 1\space\space\space\beta_t=5\space\space\space\hat{r}(a_1)=0.63\space\space\space\mu_1=0.1\\a_2:\space\alpha_t = 6\space\space\space\beta_t=4\space\space\space\hat{r}(a_2)=0.35\space\space\space\mu_2=0.5\\a_3:\space\alpha_t = 11\space\space\space\beta_t=23\space\space\space\hat{r}(a_3)=0.16\space\space\space\mu_3=0.3\\a_4:\space\alpha_t = 12\space\space\space\beta_t=25\space\space\space\hat{r}(a_4)=0.22\space\space\space\mu_4=0.2\\a_5:\space\alpha_t = 38\space\space\space\beta_t=21\space\space\space\hat{r}(a_5)=0.7\space\space\space\mu_5=0.6
  $$


  - *How much pseudo-regret the $TS$ algorithm accumulated so far, assuming we started from uniform $Beta(1,1)$ priors?*

  - *Which one of the previous posteriors is the most peaked one?*

  - *What would $UCB1$ have chosen for the next round? Assume $Bernoulli$ rewards and that in the Bayesian setting we started from uniform ​$Beta(1,1)$ priors?*