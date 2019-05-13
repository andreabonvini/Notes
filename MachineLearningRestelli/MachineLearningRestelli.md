# Machine Learning

*A series of notes on the "Machine Learning" course as taught by Marcello Restelli and Francesco Trovò during the second semester of the academic year 2018-2019 at Politecnico di Milano.*

## Theory Questions

*Here are listed all the theory questions since 06/07/2017*

- ***Describe the supervised learning technique denominated Support Vector Machines for classiﬁcation problems.***

  (*Andrea Bonvini*) 

  { DISCLAIMER : this is a full derivation of the *SVMs* equations, for a more concise overview of *SVMs* you can refer to the [PoliMi Data Science community Notes](<https://polimidatascientists.it/notes.html>) }

  { TO DO : adjust part on *KKT* conditions, add *kernels* and *slack variables* }

  Our goal is to build a binary classifier by finding an hyperplane which is able to separate the data with the biggest *margin* possible. 

  <img src="images/svm1.png" style="zoom:40%"/>

  With SVMs we force our *margin* to be at least *something* in order to accept it. by doing that we restrict the number of possible dichotomies, and therefore if we're able to separate the points with a fat dichotomy (*margin*) then that fat dichotomy will have a smaller *VC* dimension then we'd have without any restriction. Let's do that.

  Let be $\mathbf{x}_n​$ the nearest data point to the *hyperplane* $\mathbf{w}^T\mathbf{x} = 0​$ (just image a *line* in a $2​$-D space for simplicity), before finding the distance we just have state two observations:

  - There's a minor technicality about the *hyperplane* $\mathbf{w}^T\mathbf{x} = 0$ which is annoying , let's say I multiply the vector $\mathbf{w}$ by $1000000$ , I get the *same* hyperplane! So any formula that takes $\mathbf{w}$ and produces the margin will have to have built-in *scale-invariance*, we do that by normalizing $\mathbf{w}$ , requiring that for the nearest data point $\mathbf{x}_n$:
    $$
    |\mathbf{w}^T\mathbf{x}_n|=1
    $$
    ( So I just scale $\mathbf{w}$ up and down in order to fulfill the condition stated above, we just do it because it's *mathematically convenient*! By the way remember that $1$ does *not* represent the Euclidean distance)

  - When you solve for the margin, the $w_1$ to $w_d$ will play a completely different role from the role $w_0$ , so it is no longer convenient to have them on the same vector. We  pull out $w_0$ from $\mathbf{w}$ and rename $w_0$ with $b$ (for *bias*).
    $$
    \mathbf{w} = (w_1,\dots,w_d)\\w_0=b
    $$

  So now our notation is changed:

  The *hyperplane* is represented by
  $$
  \mathbf{w}^T\mathbf{x} +b= 0
  $$
  and our constraint becomes
  $$
  |\mathbf{w}^T\mathbf{x}_n+b|=1
  $$
  It's trivial to demonstrate that the vector $\mathbf{w}$ is orthogonal to the *hyperplane*, just suppose to have two point $\mathbf{x}'$ and $\mathbf{x''}$ belonging to the *hyperplane* , then $\mathbf{w}^T\mathbf{x}' +b= 0$ and $\mathbf{w}^T\mathbf{x}'' +b= 0$.

  And of course $\mathbf{w}^T\mathbf{x}'' +b - (\mathbf{w}^T\mathbf{x}' +b)=\mathbf{w}^T(\mathbf{x}''-\mathbf{x}') = 0 $ 

  Since $\mathbf{x}''-\mathbf{x}'$ is a vector which lays on the *hyperplane* , we deduce that $\mathbf{w}$ is orthogonal to the *hyperplane*.

  <img src="images/svm2.png" style="zoom:60%"/>

  Then the distance from $\mathbf{x}_n$ to the *hyperplane* can be expressed as a dot product between $\mathbf{x}_n-\mathbf{x}$ (where $\mathbf{x}$ is any point belonging to the plane) and the unit vector $\hat{\mathbf{w}}$ , where $\hat{\mathbf{w}} = \frac{\mathbf{w}}{||\mathbf{w}||}$ ( the distance is just the projection of $\mathbf{x}_n-\mathbf{x}$ in the direction of $\hat{\mathbf{w}}​$ ! )
  $$
  distance = |\;\hat{\mathbf{w}}^T(\mathbf{x}_n-\mathbf{x})\;|
  $$
  (We take the absolute value since we don't know if $\mathbf{w}$ is facing $\mathbf{x}_n$ or is facing the other direction )

  <img src="images/svm3.PNG" style="zoom:70%"/>

  We'll now try to simplify our notion of *distance*.
  $$
  distance = |\;\hat{\mathbf{w}}^T(\mathbf{x}_n-\mathbf{x})\;| = \frac{1}{||\mathbf{w}||}|\;\mathbf{w}^T\mathbf{x}_n-\mathbf{w}^T\mathbf{x}\;|
  $$
  This can be simplified if we add and subtract the missing term $b$.
  $$
  distance = \frac{1}{||\mathbf{w}||}|\;\mathbf{w}^T\mathbf{x}_n+b-\mathbf{w}^T\mathbf{x}-b\;| = \frac{1}{||\mathbf{w}||}|\;\mathbf{w}^T\mathbf{x}_n+b-(\mathbf{w}^T\mathbf{x}+b)\;|
  $$
  Well, $\mathbf{w}^T\mathbf{x}+b$ is just the value of the equation of the plane...for a point *on* the plane. So without any doubt $\mathbf{w}^T\mathbf{x}+b= 0$ , our notion of *distance* becomes
  $$
  distance = \frac{1}{||\mathbf{w}||}|\;\mathbf{w}^T\mathbf{x}_n+b\;|
  $$
  But wait...what is $|\;\mathbf{w}^T\mathbf{x}_n+b\;|$ ? It is the constraint the we defined at the beginning of our derivation!
  $$
  |\mathbf{w}^T\mathbf{x}_n+b|=1
  $$
  So we end up with the formula for the distance being just
  $$
  distance = \frac{1}{||\mathbf{w}||}
  $$
  *Which is sick AF*.

  Let's now formulate the optimization problem: 
  $$
  \underset{w}{\operatorname{argmax}}\frac{1}{||\mathbf{w}||}\\\text{subject to}\;\underset{n=1,2,\dots,N}{\operatorname{min}}|\mathbf{w}^T\mathbf{x}_n+b|=1
  $$
  Since this is not a *friendly* optimization problem (the constraint have a minimum and an absolute value in them, which are annoying) we are going to find an equivalent problem which is easier to solve. Our optimization problem can be rewritten as
  $$
  \underset{w}{\operatorname{argmin}}\frac{1}{2}\mathbf{w}^T\mathbf{w}\\y_n(\mathbf{w}^T\mathbf{x}_n+b)\ge1 \;\;\;\;\text{for $n = 1,2,\dots,N$}
  $$
  where $y_n$ is a variable that we introduce that will be equal to either $+1$ or $-1$ accordingly to the sign of our prediction $(\mathbf{w}^T\mathbf{x}_n+b)$ . One could argue that the new constraint is actually different from the former one, since maybe the $\mathbf{w}$ that we'll find will allow the constraint to be *strictly* greater than $1$ for every possible point in our dataset [ $y_n(\mathbf{w}^T\mathbf{x}_n+b)> 1 \;\;\forall{n}$ ] while we'd like it to be *exactly* equal to $1$ for *at least* one value of $n$. But that's actually not true! Since we're trying to minimize $\frac{1}{2}\mathbf{w}^T\mathbf{w}$ our algorithm will try to scale down the right hyperplane $\mathbf{w}^T\mathbf{x}_n+b$  (by "*scaling down*" I simply mean multiplying it by a constant factor e.g. $\gamma < 1$ ) until it touches $1$ for some specific point $n$ of the dataset.

  So how can we solve this? This is a constraint optimization problem with inequality constraints, we have to derive the *Lagrangian* and apply the [*KKT*](<http://www.svms.org/kkt/>) (Karush–Kuhn–Tucker) conditions.

  *Objective Function:*

  We have to minimize
  $$
  \mathcal{L}(\mathbf{w},b,\mathbf{\alpha}) = \frac{1}{2}\mathbf{w}^T\mathbf{w}-\sum_{n=1}^{N}\alpha_n(y_n(\mathbf{w}^T\mathbf{x}_n+b)-1)\\
  $$
  *w.r.t.* to $\mathbf{w}$ and $b$ and maximize it *w.r.t.* the *Lagrange Multipliers* $\alpha_n\ge 0 $ (which becomes our only constraint)

  We can easily get the two conditions for the unconstrained part:
  $$
  \nabla_{\mathbf{w}}\mathcal{L}=\mathbf{w}-\sum_{n=1}^{N}\alpha_n y_n\mathbf{x}_n = 0 \;\;\;\;\;\;\;\; \mathbf{w}=\sum_{n=1}^{N}\alpha_n y_n\mathbf{x}_n\\
  \frac{\part\mathcal{L}}{\part b} = -\sum_{n=1}^{N}\alpha_n y_n = 0\;\;\;\;\;\;\;\;\;\;\;\sum_{n=1}^{N}\alpha_n y_n=0
  $$
  Now we can reformulate the *Lagrangian* by applying some substitutions 
  $$
  \mathcal{L}(\mathbf{w},b,\mathbf{\alpha}) = \frac{1}{2}\mathbf{w}^T\mathbf{w}-\sum_{n=1}^{N}\alpha_n(y_n(\mathbf{w}^T\mathbf{x}_n+b)-1)\\
  \mathcal{L}(\mathbf{\alpha}) =\sum_{n=1}^{N}\alpha_n-\frac{1}{2}\sum_{n=1}^{N}\sum_{m=1}^{M}y_n y_m\alpha_n\alpha_m\mathbf{x}_n^T\mathbf{x}_m
  $$
  (if you have doubts just go to minute 36.50 of [this](https://www.youtube.com/watch?v=eHsErlPJWUU) lecture by professor Yaser Abu-Mostafa at *Caltech* )

  We end up with the *dual* formulation of the problem
  $$
  \underset{\alpha}{\operatorname{argmax}}\sum_{n=1}^{N}\alpha_n-\frac{1}{2}\sum_{n=1}^{N}\sum_{m=1}^{M}y_n y_m\alpha_n\alpha_m\mathbf{x}_n^T\mathbf{x}_m\\
  \;\\
  s.t. \;\;\;\;\;\;\;\;\alpha_n\ge0\;\;\;\forall{n}\\
  \;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\sum_{n=1}^{N}\alpha_n y_n=0
  $$
  We can notice that the old constraint $\mathbf{w}=\sum_{n=1}^{N}\alpha_n y_n\mathbf{x}_n$ doesn't appear in the new formulation since it is *not* a constraint on $\alpha$ , it was a constraint on $\mathbf{w}$ which is not part of our formulation anymore.

  How do we find the solution? we throw this objective (which btw happens to be a *convex* function) to a *quadratic programming* package.

  Once the *quadratic programming* package gives you back the solution you find out that a whole bunch of $\alpha$ are just $0$ !  All the $\alpha$ which are not $0$ are the *support vectors* ! (i.e. the vectors that determines the width of the *margin*) , this can be noted by observing the last *KKT* condition, in fact either a constraint is active ( $g_i(w^{*}) = 0$ ) , and hence the point is a support vector, or its multiplier is zero. 

- ***Deﬁne the VC dimension and describe the importance and usefulness of VC dimension in machine learning.***

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

  It is always principled to standardize the features before applying the ridge regression algorithm. Why is this? The coefficients that are produced by the standard least squares method are scale equivariant, i.e. if we multiply each input by ${c}​$ then the corresponding coefficients are scaled by a factor of ${\frac{1}{c}}​$. Therefore, regardless of how the predictor is scaled, the multiplication of the coefficient and the predictor ${(w_jx_j)}​$ remains the same. However, this is not the case with ridge regression, and therefore, we need to standardize the predictors or bring the predictors to the same scale before performing ridge regression. the formula used to do this is given below.
  $$
  \hat{x}_{ij}=\frac{x_{ij}}{\sqrt{\frac{1}{n}\sum^n_{i=1}(x_{ij}-\bar{x}_j)^2}}
  $$


  (Source: [towardsdatascience 1](https://towardsdatascience.com/ridge-and-lasso-regression-a-complete-guide-with-python-scikit-learn-e20e34bcbf0b ) & [towardsdatascience](https://towardsdatascience.com/regularization-in-machine-learning-76441ddcf99a) )

  

  Since ${\lambda}​$ is not defined a priori, we need a method to select a good value for it. We use Cross-Validation for solving this problem: we choose a grid of ${\lambda}​$ values, and compute the cross-validation error rate for each value of ${\lambda}​$. We then select the value for ${\lambda}​$ for which the cross-validation error is the smallest. Finally, the model is re-fit using all of the available observations and the selected value of ${\lambda}​$.

  Restelli offers the following cost function notation:

  ${L(w)=L_D(\mathbf{w})+\lambda L_W(\mathbf{w}) }$

  where ${L_D(\mathbf{w})}​$ is the error on data terms (e.g. RSS) and ${L_W(\mathbf{w})}​$ is the model complexity term.

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
  (William Bonvini)

  Before diving into the definitions, let's define what is Regularization: it's a technique which makes slight modifications to the learning algorithm such that the model avoids overfitting, so performing better on unseen data.  

  Ridge Regression is a Regularization Technique which consists in adding to the Linear Regression Loss Function a penalty term called L2 regularization element:  

  ${L(\mathbf{w})=\frac{1}{2}\sum_{i=1}^N(t_i-\mathbf{w}^T\phi(\mathbf{x}_i))^2+\frac{\lambda}{2}||\mathbf{w}||^2_2} $ 

  Lasso Regression is a Regularization Technique very similar to Ridge Regression, but instead of adding a L2 regularization element, it adds the so called L1 regularization element:  

  ${{L(\mathbf{w})=\frac{1}{2}\sum_{i=1}^N(t_i-\mathbf{w}^T\phi(\mathbf{x}_i))^2+\frac{\lambda}{2}||\mathbf{w}||_1}} $ 

  The main difference between Ridge and Lasso Regression is that Ridge Regression can only shrink to weights of the features close to 0 while Lasso Regression can shrink them all the way to 0.   
  This is due to the fact that Ridge Regression squares the features weights, while Lasso Regression considers the absolute value of such weights.  

  This means that Lasso Regression can exclude useless features from the lost function, so being better than Ridge Regression at reducing the variance in models that contain a lot of useless features. In contrast, Ridge Regression tends to do a little better when most features are useful.  

  You may ask yourself why Lasso is able to shrink some weights exactly to zero, while Ridge doesn't.  
  The following example may be explanatory:  

  Consider a model with only one feature ${x_1}$. This model learns the following output: ${\hat{f}(x_1)=4x_1}$. 
  Now let's add a new feature to the model: ${x_2}$.  Suppose that such second feature does not tell anything new to the model, which means that it depends linearly from ${x_1}$. Actually, ${x_2 = x_1}$.  
  This means that any of the following weights will do the job:  
  ${\hat{f}(x_1,x_2)=4x_1}$

  ${\hat{f}(x_1,x_2)=2x_1+2x_2}$

  ${\hat{f}(x_1,x_2)=x_1+3x_2}$

  ${\hat{f}(x_1,x_2)=4x_2}$

  We can generalize saying that ${\hat{f}(x_1,x_2)=w_1x_1+w_2x_2 \space \space\space with \space\space w_1+w_2=4}$.    

  Now consider the ${l_1}​$ and ${l_2}​$ norms of various solutions, remembering that ${l_1=|w_1+w_2|}​$ and that  ${l_2=(w_1^2+w_2^2)}​$ .  

  | ${w_1}$ | ${w_2}$ | ${l_1}$ | ${l_2}$ |
  | ------- | ------- | ------- | ------- |
  | 4       | 0       | 4       | 16      |
  | 2       | 2       | 4       | 8       |
  | 1       | 3       | 4       | 10      |
  | -1      | 5       | 6       | 26      |

  we can see that minimizing ${l_2}​$ we obtain ${w_1=w_2=2}​$, which means that it, in this case, tends to spread equally the weights.  
  While ${l_1}​$ can choose arbitrarily between the first three options, as long as the weights have the same sign it's ok. 

  Now suppose ${x_2=2x_1}$, which means that ${x_2}$ does not add new information to the model, but such features have different scale now. We can say that all functions with ${w_1+2w_2=k}$ (in the example above ${k=4}$ ) give the same predictions and have same empirical risk.  

  For clarity I will show you some of the possible values we can assign to the weights.  

  | ${w_1}$ | ${w_2}$ | ${l_1}$ | ${l_2}$ |
  | ------- | ------- | ------- | ------- |
  | 4       | 0       | 4       | 16      |
  | 3       | 0.5     | 3.5     | 9.25    |
  | 2       | 1       | 3       | 5       |
  | 1       | 1.5     | 2.5     | 3.25    |
  | 0       | 2       | 2       | 4       |

  ${l_1}$ (which translates into Lasso Regression) chooses ${w_1=0 \space ; \space w_2=2}$  
  ${l_2}$ (which translates into Ridge Regression) chooses ${w_1=1 \space ; \space w_2=1.5}$  
  Obviously I'm oversimplifying,  these won't be the actual chosen values for ${l_2}$. Ridge will choose similar values that will better minimize ${l_2}$, I just didn't list all the possible combinations for ${w_1}$ and ${w_2}$, but the important thing is that Lasso will actually go for ${w_1=0;w_2=2}$.  

  What have we noticed then?  

  - For Identical Features
    - ${l_1}$ regularization spreads weight arbitrarily (all weights same sign)
    - ${l_2}$ regularization spreads weight evenly
  - For Linearly Related Features
    - ${l_1}$ regularization chooses the variable with the largest scale, 0 weight to the others
    - ${l_2}$ prefers variables with larger scale, it spreads the weight proportional to scale

  

  (Sources: [PoliMi Data Scientists Notes - Machine Learning](https://polimidatascientists.it/notes.html)   ;   [Bloomberg - Lasso, Ridge, and Elastic Net](https://www.youtube.com/watch?v=KIoz_aa1ed4&t=934s)	)  

  

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
  (William Bonvini) 
  I've already described Ridge Regression previously.

  *Comparison:*

  Ridge Regression is a frequentist approach:  
  the model assumes that the response variable (y) is a linear combination of weights multiplied by a set of predictor variables (x). The full formula also includes an error term to account for random sampling noise. 

  What we obtain from frequentist linear regression is a single estimate for the model parameters based only on the training data. Our model is completely informed by the data: in this view, everything that we need to know for our model is encoded in the training data we have available.

  Ridge Regression gives us a single point estimate for the output. However, if we have a small dataset we might like to express our estimate as a distribution of possible values. This is where Bayesian Linear Regression comes in.

  The aim of Bayesian Linear Regression is not to find the single “best” value of the model parameters, but rather to determine the *posterior distribution* (*a probability distribution that represents your updated beliefs about the parameter after having seen the data*) for the model parameters.  
  Not only is the response generated from a probability distribution, but the model parameters are assumed to come from a distribution as well. The posterior probability of the model parameters is conditional upon the training inputs and outputs:
  $$
  P(\beta|y,X)=\frac{P(y|B,X)P(\beta|X)}{P(y|X)}
  $$
  Here, ${P(\beta |y,X)}$ is the posterior probability distribution of the model parameters given the inputs and outputs. This is equal to the likelihood of the data, ${P(y|\beta,X)}$, multiplied by the prior probability of the parameters and divided by a normalization constant. This is a simple expression of Bayes Theorem, the fundamental underpinning of Bayesian Inference:
  $$
  Posterior = \frac{Likelihood*Prior}{Normalization}
  $$


  Let's stop and think about what this means. In contrast to Ridge Regression , or Linear Regression in general, we have a *posterior* distribution for the model parameters that is proportional to 

  - the likelihood of the data

  - the *prior* probability of the parameters. 

  Here we can observe the two primary benefits of Bayesian Linear Regression:

  1. **Priors**:   
     if we have domain knowledge, or a guess for what the model parameters should be, we can include them in our model, unlike in the frequentist apporach which assumes everything there is to know about the parameters comes from the data. If we don't have any estimates ahead of time, we can use <u>non-informative priors</u> for the parameters such as a normal distribution.

  2. **Posterior**:  
     The result of performing Bayesian Linear Regression is a distribution of possible model parameters based on the data and the prior.  
     This allows us to quantify our uncertainty about the model: if we have fewer data points, the posterior distribution will be more spread out.

  

  The formulation of model parameters as distributions encapsulates the Bayesian worldview: we start out with an initial estimate, our prior, and as we gather more evidence, **our model becomes less wrong**. Bayesian reasoning is a natural extension of our intuition. Often, we have an initial hypothesis, and as we collect data that either supports or disproves our ideas, we change our model of the world (ideally this is how we would reason)!

  Source:

  [towardsdatascience - Introduction to Bayesian Linear Regression](https://towardsdatascience.com/introduction-to-bayesian-linear-regression-e66e60791ea7)

  

- ***Deﬁne the VC dimension of a hypothesis space. What is the VC dimension of linear classiﬁers?***

- ***Describe which methods can be used to compute the value function $V^{\pi}$ of a policy $\pi$ in a discounted Markov Decision Process.***

- ***Describe the supervised learning technique denominated Support Vector Machines for classiﬁcation problems.***

- ***Describe the supervised learning technique denominated logistic regression for classiﬁcation problems.***

- ***Describe the policy iteration technique for control problems on Markov Decision Processes***

- ***Describe the two problems tackled by Reinforcement Learning (RL): prediction and control. Describe how the Monte Carlo RL technique can be used to solve these two problems.***

- ***Describe the purpose of using kernels in Machine Learning techniques. How can you construct a valid Kernel? Provide an example of a ML method using kernels and describe the speciﬁc advantage of using them for this method.***

  *(Andrea Bonvini)*

  Traditionally, theory and algorithms of machine learning and statistics has been very well developed for the linear case. Real world data analysis problems, on the other hand, often require nonlinear methods to detect the kind of dependencies that allow successful prediction of properties of interest. By using a positive definite kernel, one can sometimes have the best of both worlds. The kernel corresponds to a dot product in a (*usually high-dimensional, possibly infinite*) feature space. In this space, our estimation methods are linear, but as long as we can formulate everything in terms of kernel evaluations, we never explicitly have to compute in the high dimensional feature space! (This is called the *Kernel Trick*)

  Suppose we have a mapping $\varphi : \R^d \to \R^m$ that brings our vectors in to some feature space $\R^m$. Then the dot product of $\textbf{x}$ and $\textbf{y}$ in this space is $\varphi (\textbf{x})^T\varphi (\textbf{y})$. A kernel is a function $k$ that corresponds to this dot product, i.e. $k(\textbf{x},\textbf{y})=\varphi (\textbf{x})^T\varphi (\textbf{y}) $ . Why is this useful? *Kernels* give a way to compute dot products in some feature space without even knowing what this space is and what is $\varphi$ . For example, consider a simple polynomial kernel $k(\textbf{x},\textbf{y})=(1+\textbf{x}^T\textbf{y})^2$ with $\textbf{x},\textbf{y} \in \R^2$. This doesn't seem to correspond to any mapping function $\varphi$ ,  it's just a function that returns a real number. Assuming that $\textbf{x} = (x_1,x_2)$ and $\textbf{y} = (y_1,y_2)$, let's expand this expression:
  $$
  k(\textbf{x},\textbf{y})=(1+\textbf{x}^T\textbf{y})^2 = (1+x_1y_1 + x_2y_2)^2=\\1+x_1^2y_1^2+x_2^2y_2^2+2x_1y_1+2x_2y_2+2x_1x_2y_1y_2
  $$
  Note that this is nothing else but a dot product between two vectors$(1, x_1^2, x_2^2, \sqrt{2} x_1, \sqrt{2} x_2, \sqrt{2} x_1 x_2)$ and $(1, y_1^2, y_2^2, \sqrt{2} y_1, \sqrt{2} y_2, \sqrt{2} y_1 y_2)$ and $\varphi(\mathbf x) = \varphi(x_1, x_2) = (1, x_1^2, x_2^2, \sqrt{2} x_1, \sqrt{2} x_2, \sqrt{2} x_1 x_2)​$.

  So the kernel $k(\mathbf x, \mathbf y) = (1 + \mathbf x^T \mathbf y)^2 = \varphi(\mathbf x)^T \varphi(\mathbf y)$ computes a dot product in 6-dimensional space without explicitly visiting this space.

  Another example is Gaussian kernel $k(\mathbf x, \mathbf y) = \exp\big(- \gamma \, \|\mathbf x - \mathbf y\|^2 \big)$. If we Taylor-expand this function, we'll see that it corresponds to an infinite-dimensional codomain of $\varphi$.

  Instead, the simplest *kernel* is the *linear kernel* which corresponds to an *identity mapping* in the feature space: $k(\mathbf{x},\mathbf{x'}) = \varphi(\mathbf{x})^T\varphi(\mathbf{x'}) = \mathbf{x}^T\mathbf{x}$ 

  Moreover, the *kernel* is a *symmetric* function of its arguments: $k(\mathbf{x},\mathbf{x'}) = k(\mathbf{x'},\mathbf{x})$

  ------------------

  Many linear models for regression and classiﬁcation can be reformulated in terms of *dual representation* in which the *kernel function arises naturally* ! For example if we consider a linear regression model we know that we obtain the best parameters by minimizing the *regularized sum of squares* error function (*ridge*):
  $$
  L_{\mathbf{w}} = \frac{1}{2}\sum_{n=1}^{N}(\mathbf{w}^T\varphi(\mathbf{x_n})-t_n)^2+\frac{\lambda}{2}\mathbf{w}^T\mathbf{w}
  $$
  Setting the gradient of $L_{\mathbf{w}}$ w.r.t $\mathbf{w}$ equal to $0$ we obtain the following:
  $$
  \mathbf{w} = -\frac{1}{\lambda}\sum_{n=1}^{N}(\mathbf{w}^T\varphi(\mathbf{x_n})-t_n)\varphi(\mathbf{x_n}) = \sum_{n=1}^Na_n\varphi(\mathbf{x}_n)=\Phi^T\mathbf{a}
  $$
  Where $\Phi$ is the design matrix whose $n^{th}$ row is $\varphi(\mathbf{x}_n)^T$ (remember that in $L_{\mathbf{w}}$ all the vectors are *column* vectors!) and the coefficients $a_n$ are functions of $\mathbf{w}$. So our definition of $\mathbf{w}$ is function of $\mathbf{w}$ itself...which is surely weird, just *wait for it...*

  We now define the *Gram Matrix* $\mathbf{K} = \Phi \times \Phi^T$, an $N \times N$ matrix, with elements:
  $$
  K_{nm} = \varphi(\mathbf{x_n})^T\varphi(\mathbf{x_m})=k(\mathbf{x}_n,\mathbf{x}_m)
  $$
  So, given $N​$ vectors, the *Gram Matrix* is the matrix of all *inner products* 

  ![](images/GramMatrix.PNG)

  If we substitute $\mathbf{w} = \Phi^T\mathbf{a}$ into $L_{\mathbf{w}}$ we get
  $$
  L_{\mathbf{a}} = \frac{1}{2}\mathbf{a}^T\Phi\Phi^T\Phi\Phi^T\mathbf{a}-\mathbf{a}^T\Phi\Phi^T\mathbf{t}+\frac{1}{2}\mathbf{t}^T\mathbf{t}+\frac{\lambda}{2}\mathbf{a}^T\Phi\Phi^T\mathbf{a}
  $$
  where $\mathbf{t} = (t_1,...,t_N)^T$. Guess what? we can rewrite the Loss function in terms of *Gram Matrix* !
  $$
  L_{\mathbf{a}} = \frac{1}{2}\mathbf{a}^TKK\mathbf{a}-\mathbf{a}^TK\mathbf{t}+\frac{1}{2}\mathbf{t}^T\mathbf{t}+\frac{\lambda}{2}\mathbf{a}^TK\mathbf{a}
  $$
   Solving for $\mathbf{a}$ by combining $\mathbf{w} = \Phi^T\mathbf{a}$ and $a_n = -\frac{1}{\lambda}(\mathbf{w}^T\varphi(\mathbf{x}_n)-t_n)$ (setting the gradient to $0$ etc...)
  $$
  \mathbf{a}=(K+\lambda\mathbf{I}_N)^{-1}\mathbf{t}
  $$
  Consider that $K = N\times N$ and $\mathbf{t} = N\times 1$, so $\mathbf{a} = N \times 1$.

  So we can make our prediction for a new input $\mathbf{x}​$ (which has dimension $D\times 1​$ obviously, $\varphi(\mathbf{x})​$ will have dimension $M\times 1​$ instead, where $M>D​$) by substituting back into our linear regression model:
  $$
  y(\mathbf{x}) = \mathbf{w}^T\varphi(\mathbf{x}) = (\Phi^T\mathbf{a})^T\varphi(\mathbf{x}) = \mathbf{a}^T\Phi\varphi(\mathbf{x})= \mathbf{k}(\mathbf{x})^T(K+\lambda\mathbf{I}_N)^{-1}\mathbf{t}
  $$
  where $\mathbf{k}(\mathbf{x})$ has elements $k_n(\mathbf{x}) = k(\mathbf{x}_n,\mathbf{x})$ . Prediction is just a linear combination of the *target values* from the *training set* . (If you make a dimensionality check you will see that $y(\mathbf{x})$ will be just a number)

  The good thing is that instead of inverting an $M\times M​$ matrix, we are inverting an $N\times N​$ matrix! (as we already said different times,  this allows us to work with *very high or infinite dimensionality* of $\mathbf{x}​$).

  -----------------------------

   But *how* can we build a valid *kernel*?

   We have manly two ways to do it:

  - *By construction*: we choose a feature space mapping $\varphi (\mathbf{x})$ and use it to ﬁnd the corresponding kernel. (I'd call this method *by hand*)

  - It is possible to test whether a function is a valid kernel without having to construct the basis function explicitly. The necessary and suﬃcient condition for a function $k(\mathbf{x},\mathbf{x}')$ to be a kernel is that the Gram matrix $K$ is positive semi-deﬁnite for all possible choices of the set $\{x_n\}$. It means that $ \mathbf{x}^TK\mathbf{x}\ge 0$ for non-zero vectors $\mathbf{x}$ with real entries, i.e.$\sum_n\sum_m K_{n,m}x_nx_m \ge 0$ for any real number $x_n,x_m$. 

    *Mercer's Theorem :* Any continuous, symmetric, positive semi-deﬁnite kernel function $k(\mathbf{x},\mathbf{y})$ can be expressed as a dot product in a high-dimensional space.

    New kernels can be constructed from simpler kernels as *building blocks*:

    ![](images/Kernels.PNG)

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


  - *How much pseudo-regret the $TS​$ algorithm accumulated so far, assuming we started from uniform $Beta(1,1)​$ priors?*

  - *Which one of the previous posteriors is the most peaked one?*

  - *What would $UCB1$ have chosen for the next round? Assume $Bernoulli$ rewards and that in the Bayesian setting we started from uniform ​$Beta(1,1)$ priors?*

## Interesting Articles

- [Polynomial Regression](https://towardsdatascience.com/polynomial-regression-bbe8b9d97491)

- [Difference between Frequentist and Bayesian Approach](https://towardsdatascience.com/introduction-to-bayesian-linear-regression-e66e60791ea7) 

  