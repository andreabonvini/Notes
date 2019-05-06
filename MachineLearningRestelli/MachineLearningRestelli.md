# Machine Learning

*A series of notes on the "Machine Learning" course as taught by Marcello Restelli and Francesco Trovò during the second semester of the academic year 2018-2019 at Politecnico di Milano.*

### Theory Questions

*Here are listed all the theory questions since 06/07/2017*

- ***Describe the supervised learning technique denominated Support Vector Machines for classiﬁcation problems.***

- ***Deﬁne the VC dimension and the describe the importance and usefulness of VC dimension in machine learning.***

- ***Describe the diﬀerences existing between the Q-learning and SARSA algorithms***

- ***Describe the supervised learning technique called ridge regression for regression problems.***

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

  One of the main advantages kernel methods allow for HUGE feature spaces (even INFINITE dimensional feature spaces) without suffering from computational cost.

  - A method is *kernalized* in inputs only appear inside inner products: $\langle \psi(x),\psi(y)\rangle​$ for $x,y \space \in X​$ 

  - The kernel function corresponding to $\psi$ and inner product $\langle\cdot,\cdot\rangle$ is:

  - $$
    k(x,y) = \langle \psi(x),\psi(y)\rangle
    $$

  $XX^T =n\times d \cdot d\times n=n\times n​$ !!!!!! we don't care about the dimension spaceeeee

  

  

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