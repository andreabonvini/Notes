# Machine Learning 

`or: How I Learned to Stop Worrying and Love Restelli`

*A series of extremely verbose and surely overkill notes on the "Machine Learning" course as taught by Marcello Restelli and Francesco Trovò during the second semester of the academic year 2018-2019 at Politecnico di Milano.*

`Feel free to modify the document to correct any mistakes / add additional material in order to favour a better understanding of the concepts`

# Theory Questions

*Here are listed all the theory questions since 06/07/2017, divided in three chapters: Supervised Learning, Reinforcement Learning, and Exercises*

[TOC]

<div style="page-break-after: always;"></div> 

# Supervised Learning

### Support Vector Machines

***Describe the supervised learning technique denominated Support Vector Machines for classiﬁcation problems. Which algorithm can we use to train an SVM? Provide an upper bound to the generalization error of an SVM.***

Our goal is to build a binary classifier by finding an hyperplane which is able to separate the data with the biggest *margin* possible. 

<img src="images/svm1.png" style="zoom:40%"/>

With SVMs we force our *margin* to be at least *something* in order to accept it, by doing that we restrict the number of possible dichotomies, and therefore if we're able to separate the points with a fat dichotomy (*margin*) then that fat dichotomy will have a smaller *VC* dimension then we'd have without any restriction. Let's do that.

Let be $\mathbf{x}_n$ the nearest data point to the *hyperplane* $\mathbf{w}^T\mathbf{x} = 0$ (just imagine a *line* in a $2$-D space for simplicity), before finding the distance we just have to state two observations:

- There's a minor technicality about the *hyperplane* $\mathbf{w}^T\mathbf{x} = 0$ which is annoying , let's say I multiply the vector $\mathbf{w}$ by $1000000$ , I get the *same* hyperplane! So any formula that takes $\mathbf{w}$ and produces the margin will have to have built-in *scale-invariance*, we do that by normalizing $\mathbf{w}$ , requiring that for the nearest data point $\mathbf{x}_n$:
  $$
  |\mathbf{w}^T\mathbf{x}_n|=1
  $$
  ( So I just scale $\mathbf{w}$ up and down in order to fulfill the condition stated above, we just do it because it's *mathematically convenient*! By the way remember that $1$ does *not* represent the Euclidean distance)

- When you solve for the margin, the $w_1$ to $w_d$ will play a completely different role from the role of $w_0$ , so it is no longer convenient to have them on the same vector. We  pull out $w_0$ from $\mathbf{w}$ and rename $w_0$ with $b$ (for *bias*).
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

Then the distance from $\mathbf{x}_n$ to the *hyperplane* can be expressed as a dot product between $\mathbf{x}_n-\mathbf{x}$ (where $\mathbf{x}$ is any point belonging to the plane) and the unit vector $\hat{\mathbf{w}}$ , where $\hat{\mathbf{w}} = \frac{\mathbf{w}}{||\mathbf{w}||}$ ( the distance is just the projection of $\mathbf{x}_n-\mathbf{x}$ in the direction of $\hat{\mathbf{w}}$ ! )
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
Since this is not a *friendly* optimization problem (the constraint is characterized by a minimum and an absolute, which are annoying) we are going to find an equivalent problem which is easier to solve. Our optimization problem can be rewritten as
$$
\underset{w}{\operatorname{argmin}}\frac{1}{2}\mathbf{w}^T\mathbf{w}\\y_n(\mathbf{w}^T\mathbf{x}_n+b)\ge1 \;\;\;\;\text{for $n = 1,2,\dots,N$}
$$
where $y_n$ is a variable that we introduce that will be equal to either $+1$ or $-1$ accordingly to its real target value ( remember that this is a *supervised learning* technique and we know the real target value of each sample) . One could argue that the new constraint is actually different from the former one, since maybe the $\mathbf{w}$ that we'll find will allow the constraint to be *strictly* greater than $1$ for every possible point in our dataset [ $y_n(\mathbf{w}^T\mathbf{x}_n+b)> 1 \;\;\forall{n}$ ] while we'd like it to be *exactly* equal to $1$ for *at least* one value of $n$. But that's actually not true! Since we're trying to minimize $\frac{1}{2}\mathbf{w}^T\mathbf{w}$ our algorithm will try to scale down $\mathbf{w}$ until $\mathbf{w}^T\mathbf{x}_n+b$ will touch $1$ for some specific point $n$ of the dataset. 

So how can we solve this? This is a constraint optimization problem with inequality constraints, we have to derive the *Lagrangian* and apply the [*KKT*](<http://www.svms.org/kkt/>) (Karush–Kuhn–Tucker) conditions.

*Objective Function:*

We have to minimize
$$
\mathcal{L}(\mathbf{w},b,\mathbf{\alpha}) = \frac{1}{2}\mathbf{w}^T\mathbf{w}-\sum_{n=1}^{N}\alpha_n(y_n(\mathbf{w}^T\mathbf{x}_n+b)-1)\\
$$
*w.r.t.* to $\mathbf{w}$ and $b$ and maximize it *w.r.t.* the *Lagrange Multipliers* $\alpha_n$ 

We can easily get the two conditions for the unconstrained part:
$$
\nabla_{\mathbf{w}}\mathcal{L}=\mathbf{w}-\sum_{n=1}^{N}\alpha_n y_n\mathbf{x}_n = 0 \;\;\;\;\;\;\;\; \mathbf{w}=\sum_{n=1}^{N}\alpha_n y_n\mathbf{x}_n\\
\frac{\part\mathcal{L}}{\part b} = -\sum_{n=1}^{N}\alpha_n y_n = 0\;\;\;\;\;\;\;\;\;\;\;\sum_{n=1}^{N}\alpha_n y_n=0
$$
And list the other *KKT* conditions:
$$
y_i(\mathbf{w}^T\mathbf{x}_i+b)-1\ge0\;\;\;\;\;\;\forall{i}\\
\alpha_i\ge0\;\;\;\;\;\;\;\forall{i}\\
\alpha_i(y_i(\mathbf{w}^T\mathbf{x}_i+b)-1)=0\;\;\;\;\;\;\forall{i}
$$
*Alert* :  the last condition is called the KKT *dual complementary condition* and will be key for showing that the SVM has only a small number of "support vectors", and will also give us our convergence test when we'll talk about the *SMO* algorithm. 

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

Once the *quadratic programming* package gives us back the solution we find out that a whole bunch of $\alpha$ are just $0$ !  All the $\alpha$ which are not $0$ are the *support vectors* ! (i.e. the vectors that determines the width of the *margin*) , this can be noted by observing the last *KKT* condition, in fact either a constraint is active , and hence the point is a support vector, or its multiplier is zero. 

Now that we solved the problem we can get both $\mathbf{w}$  and $b$.
$$
\mathbf{w} = \sum_{\mathbf{x}_n \in \text{ SV}}\alpha_ny_n\mathbf{x}_n\\
y_n(\mathbf{w}^T\mathbf{x}_{n\in\text{SV}}+b)=1
$$
where $\mathbf{x}_{n\in\text{SV}}$ is any *support vector*. (you'd find the *same* $b$ for every support vector)

But the coolest thing about *SVMs* is that we can rewrite our *objective functions* as follows:
$$
\mathcal{L}(\mathbf{\alpha}) =\sum_{n=1}^{N}\alpha_n-\frac{1}{2}\sum_{n=1}^{N}\sum_{m=1}^{M}y_n y_m\alpha_n\alpha_mk(\mathbf{x}_n\mathbf{x}_m)
$$
We can use *kernels* !! (if you don't know what I'm talking about read the *kernel* related question present somewhere in this document)

Finally we end up with the following equation for classifying *new points*:
$$
y(\mathbf{x}) = sign\left(\sum_{n=1}^{N}\alpha_n t_n k(\mathbf{x},\mathbf{x}_n)+b\right)
$$
The method described until here is called *hard-margin SVM* since the margin has to be satisfied strictly, it can happens that the points are not *linearly separable* in *any* way, or we just want to handle *noisy data* to avoid overfitting, so now we're going to briefly define another version of it, which is called *soft-margin SVM* that allows for few errors and penalizes for them.

We introduce *slack variables* $\xi_i$ , in this way we allow to *violate* the margin constraint but we add a *penalty*.

We now have to 
$$
\text{Minimize}\ \ ||\mathbf{w}||_2^2+C\sum_i \xi_i \\
\text{s.t.}\\ \ y_i(\mathbf{w}^Tx_i+b)\ge1-\xi_i\ ,\ \ \ \forall{i}\\
\xi_i\ge0\ ,\ \ \ \forall{i}
$$
$C$ is a coefficient that allows to treadeoff bias-variance and is chosen by *cross-validation*.

And obtain the *Dual Representation*

$$
  \text{Maximize}\ \ \ \mathcal{L}(\mathbf{\alpha}) =\sum_{n=1}^{N}\alpha_n-\frac{1}{2}\sum_{n=1}^{N}\sum_{m=1}^{M}y_n y_m\alpha_n\alpha_mk(\mathbf{x}_n\mathbf{x}_m)\\
  \text{s.t.}\\
  0\le\alpha_n\le C\ \ \ \ \ \forall{i}\\
  \sum_{n=1}^N\alpha_n t_n = 0
$$
  Support vectors are points associated with $\alpha_n > 0$

  if $\alpha_n<C$ the points lies *on the margin*

  if $\alpha_n = C$ the point lies *inside the margin*, and it can be either *correctly classified* ($\xi_i \le 1$) or *misclassified* ($\xi_i>1$)  

  Fun fact: When $C$ is large, larger slacks penalize the objective function of SVM’s more than when $C$ is small. As $C$ approaches infinity, this means that having any slack variable set to non-zero would have infinite penalty. Consequently, as $C$ approaches infinity, all slack variables are set to $0$ and we end up with a hard-margin SVM classifier.

And what about generalization? Can we compute an *Error* bound in order to see if our model is overfitting? 

As *Vapnik* said: "In the support-vectors learning algorithm the complexity of the construction does not depend on the dimensionality of the feature space, but on the number of support vectors." So it's reasonable to define an upper bound of the error as:
$$
  L_h\le\frac{\mathbb{E}[\text{number of support vectors}]}{N}
$$
This is called *Leave-One-Out Bound* (I don't know why, maybe it's written [here ](<https://ocw.mit.edu/courses/mathematics/18-465-topics-in-statistics-statistical-learning-theory-spring-2007/lecture-notes/l4.pdf> )). The good thing is that it can be easily computed and we don't need to run SVM multiple times.

The other kind of bound is called *Margin bound*: a bound on the VC dimension which decreases with the margin. The larger the margin, the less the variance and so, the less the VC dimension. Unfortunately the bound is quite pessimistic 

Sometimes for computational reasons, when we solve a problem characterized by a huge dataset, it is not possible to compute *all* the support vectors with generic quadratic programming solvers (the number of constraints depends on the number of samples), hence, specialized optimization algorithms are often used. One example is *Sequential Minimal Optimization (SMO)*:

Remember our formulation for the *soft-margin SVM*:
$$
\mathcal{L}(\mathbf{\alpha}) =\sum_{n=1}^{N}\alpha_n-\frac{1}{2}\sum_{n=1}^{N}\sum_{m=1}^{M}y_n y_m\alpha_n\alpha_mk(\mathbf{x}_n\mathbf{x}_m)\\
  s.t.\\
  0\le\alpha_i\le C\ \ \ \ \text{for}\ i =1,2,\dots,n\\
  \sum_{i=1}^ny_i\alpha_i=0
$$
*SMO* breaks this problem into a series of smallest possible sub-problems, which are then solved analytically. Because of the linear equality constraint involving the Lagrange multipliers $\alpha _{i}$ , the smallest possible problem involves two such multipliers. Then, for any two multipliers $\alpha_1$ and $\alpha_2$ the constraints are reduced to:
$$
0\le\alpha_1,\alpha_2\le C\\
  y_1\alpha_1+y_2\alpha_2=k
$$
and this reduced problem can be solved analytically: one needs to find a minimum of a one-dimensional quadratic function. $k$ is the negative of the sum over the rest of terms in the equality constraint, which is fixed in each iteration ( we do this because we want that $\sum_{i=1}^ny_i\alpha_i=0$ ).

The algorithm proceeds as follows:

  - Find a Lagrange multiplier $\alpha_1$ that violates the *KKT* conditions for the optimization problem.
  - Pick a second multiplier $\alpha_2$ and optimize the pair ($\alpha_1$,$\alpha_2$).
  - Repeat steps $1$ and $2$ until convergence.

When all the Lagrange multipliers satisfy the KKT conditions (within a user-defined tolerance), the problem has been solved. Although this algorithm is guaranteed to converge, heuristics are used to choose the pair of multipliers so as to accelerate the rate of convergence. This is critical for large data sets since there are $\frac{n(n-1)}{2}$  possible choices for $\alpha_i$ and $\alpha_j$ .

<div style="page-break-after: always;"></div> 

### PAC Learning & VC Dimension

***What do we mean as PAC-Learning and Agnostic-Learning?***

First, some concepts you need to know:

- We are talking about *Classification*.

- Overfitting happens:

  - Because with a large hypothesis space the training error is a bad estimate of the prediction error, hence we would like to infer something about the generalization error from the training samples. 
  - When the learner doesn’t have access to enough samples, hence we would like to estimate how many samples are enough.

  This cannot be performed by measuring the bias and the variance, but we can bound them.

Given:

- Set of instances $\mathcal{X}$
- Set of hypotheses $\mathcal{H}$ (finite)
- Set of possible target concepts $C$. Each concept $c$ corresponds to a boolean function $c:\mathcal{X} \to\{0,1\}$ which can be viewed as belonging to a certain class or not
- Training instances generated by a fixed, unknown probability distribution $P$ over $X$. 

The learner observes a sequence $D$ of training examples $\langle x,c(x) \rangle$, for some target concept $c \in C$ and it must output a hypothesis $h$ estimating $c$.

$h$ is evaluated by its performance on subsequent instances drawn according to $P$
$$
L_{true} = Pr_{x \in P}[c(x) \neq h(x)]
$$
We want to bound $L_{true}$ given $L_{train}$, which is the percentage of misclassiﬁed training instances.

Let's talk now about *Version Spaces* : The version space $VS_{\mathcal{H},\mathcal{D}}$ is the subset of hypothesis in $H$ consistent with the training data $D$ (in other words is the subset of $H$ where $L_{train} = 0$).

![](images/VS1.PNG)

How likely is the learner to pick a *bad hypothesis* ?

![](images/th1.PNG)

If you're interested in the proof:

------

![](images/PROOF.PNG)

where $k$ is (probably) the number of hypothesis $h \in VS_{\mathcal{H},\mathcal{D}}$  

------

Now, we use a *Probably Approximately Correct (PAC) bound*:

If we want this probability to be at most $\delta$ we can write
$$
|H|e^{-\epsilon N}\le \delta
$$
which means
$$
N \ge \frac{1}{\epsilon}\left(\ln|H|+\ln\left(\frac{1}{\delta}\right)\right)
$$
and
$$
\epsilon \ge \frac{1}{N}\left(\ln|H|+\ln\left(\frac{1}{\delta}\right)\right)
$$
Note that if, *for example*, we consider $M$ boolean features, there are $|C| = 2^M$ distinct concepts and hence $|H| = 2^{2^M}$ (which is huuuge)

If you wonder why let's suppose we have just $2$ boolean features ($A$ and $B$ ) , then we have $|H| = 2^{2^2} = 16$ distinct boolean functions :

```
A   B|  F0  F1  F2  F3  F4  F5  F6  F7
0   0|  0   0   0   0   0   0   0   0
0   1|  0   0   0   0   1   1   1   1
1   0|  0   0   1   1   0   0   1   1
1   1|  0   1   0   1   0   1   0   1

A   B|  F8  F9  F10 F11 F12 F13 F14 F15
0   0|  1   1   1   1   1   1   1   1
0   1|  0   0   0   0   1   1   1   1
1   0|  0   0   1   1   0   0   1   1
1   1|  0   1   0   1   0   1   0   1

function            symbol          name
F0                  0               FALSE
F1                  A ^ B           AND
F2                  A ^ !B          A AND NOT B
F3                  A               A
F4                  !A ^ B          NOT A AND B
F5                  B               B
F6                  A xor B         XOR
F7                  A v B           OR
F8                  A nor B         NOR
F9                  A XNOR B        XNOR
F10                 !B              NOT B
F11                 A v !B          A OR NOT B
F12                 !A              NOT A
F13                 !A v B          NOT A OR B
F14                 A nand B        NAND
F15                 1               TRUE
```

and so the bounds would have an *exponential* dependency on the number of features M !
$$
N \ge \frac{1}{\epsilon}\left(\ln|H|+\ln\left(\frac{1}{\delta}\right)\right)\\
N \ge \frac{1}{\epsilon}\left(\ln2^{2^M}+\ln\left(\frac{1}{\delta}\right)\right)\\
N \ge \frac{1}{\epsilon}\left(\underline{\underline{2^M}}\ln2+\ln\left(\frac{1}{\delta}\right)\right)\\
\epsilon \ge \frac{1}{N}\left(\ln|H|+\ln\left(\frac{1}{\delta}\right)\right)\\
\epsilon \ge \frac{1}{N}\left(\ln2^{2^M}+\ln\left(\frac{1}{\delta}\right)\right)\\
\epsilon \ge \frac{1}{N}\left(\underline{\underline{2^M}}\ln2+\ln\left(\frac{1}{\delta}\right)\right)
$$
which is bad news.

Instead of having an *exponential* dependency on $M$ we'd like to have a, *guess what?* , *polynomial* dependency!

Now, look at the bounds we defined earlier:
$$
N \ge \frac{1}{\epsilon}\left(\ln|H|+\ln\left(\frac{1}{\delta}\right)\right)\\
\epsilon \ge \frac{1}{N}\left(\ln|H|+\ln\left(\frac{1}{\delta}\right)\right)
$$
Consider a class $C$ of possible target concepts deﬁned over a set of instances $X$ and a learner $L$ using hypothesis space $H$.

*Definition :*

$C$ is ***PAC-learnable*** it there exists an algorithm $L$ such that for every $c \in C$ , for any distribution $P$ , for any $\epsilon$ such that $0\le\epsilon\le\frac{1}{2}$ and $\delta$ such that $0\le\delta\le 1$, with probability at least $1-\delta$, outputs an hypothesis $h\in H$, such that $L_{true}(h) \le \epsilon$, using a number of samples that is polynomial of $\frac{1}{\epsilon}$ and $\frac{1}{\delta}$ 

$C$ is ***efficiently PAC-learnable*** by a learner $L$ using $H$ if and only if every $c \in C$ , for any distribution $P$ , for any $\epsilon$ such that $0\le\epsilon\le\frac{1}{2}$ and $\delta$ such that $0\le\delta\le \frac{1}{2}$, with probability at least $1-\delta$, outputs an hypothesis $h\in H$, such that $L_{true}(h) \le \epsilon$, using a number of samples that is polynomial of $\frac{1}{\epsilon}$ , $\frac{1}{\delta}$, $M$ and $size(c)$.

Let's talk now about ***Agnostic Learning***...

Usually the *train* error is not equal to zero, so the $VS$ is empty. In this case there is the need of bounding the gap between train and true errors.
$$
L_{true}(h) \le L_{train}(h) + \epsilon\\
L_{true}(h) - L_{train}(h)\le \epsilon
$$
Firstly, some background:

In probability theory , the *Hoeffding's inequality* provides an upper bound on the probability that the sum of bounded random variables deviates from its expected value by more than a certain amount. Formally speaking, for $N$ *i.i.d* coin flips $X_1,\dots,X_N$ where $X_i \in \{0,1\}$ and $0<\epsilon<1$ , we define the empirical mean
$$
\overline{X}=\frac{1}{N}(X_1+\dots+X_N)
$$
obtaining the following bound:
$$
Pr(\mathbb{E}[\overline{X}]-\overline{X}>\epsilon)<e^{-2N\epsilon^2}
$$
*Theorem:*

![](images/AGN1.PNG)

*PAC bound and Bias-Variance Tradeoff*

![](images/AGN2.PNG)

- For large $|H|$ 

  - Low bias (assuming we can find a good $h$ )
  - High variance (because bound is loser)

- For small $|H|$

  - High bias (is there a good $h$ ? )
  - Low variance (tighter bound)

- Given $\delta$, $\epsilon$ how large should $N$ be?
  $$
  N\ge\frac{1}{2\epsilon^2}\left(\ln |H|+\ln\frac{1}{\delta}\right)
  $$

`manca la parte che trovi su PMDS` 

***Deﬁne the VC dimension and describe the importance and usefulness of VC dimension in machine learning. Deﬁne the VC dimension of a hypothesis space. What is the VC dimension of a linear classiﬁer?***

- We are always talking about *Classification*.

- When counting the number of hypotheses, the entire input space is taken into consideration. In the case of a perceptron, each perceptron differs from another if they differ in at least one input point, and since the input is continuous, there are an infinite number of different perceptrons. (e.g. in a $2-D$ space you can draw an infinite number of different lines)

  Instead of counting the number of hypotheses in the entire input space, we are going to restrict the count only to the samples: a *finite* set of input points. Then, simply count the number of the possible *dichotomies*. A dichotomy is like a mini-hypothesis, it’s a *configuration of labels* on the sample’s input points.

  A *hypothesis* is a function that maps an input from the entire *input space* to a result:
  $$
  h:\mathcal{X}\to\{-1,+1\}
  $$
  The number of hypotheses $|\mathcal{H}|$ can be infinite.

  A *dichotomy* is a hypothesis that maps from an input from the *sample size* to a result:
  $$
  h:\{\mathbf{x}_1,\mathbf{x}_2,\dots,\mathbf{x}_N\}\to\{-1,+1\}
  $$
  The number of *dichotomies* $|\mathcal{H}\{\mathbf{x}_1,\mathbf{x}_2,\dots,\mathbf{x}_N\}|$ is at most $2^N$, where $N$ is the sample size.

  e.g. for a sample size $N = 3$ we have at most $8$ possible dichotomies:

  ```
  	x1 x2 x3
  1	-1 -1 -1
  2	-1 -1 +1
  3	-1 +1 -1
  4	-1 +1 +1
  5	+1 -1 -1 
  6	+1 -1 +1
  7	+1 +1 -1
  8	+1 +1 +1
  
  ```

- The *growth function* is a function that counts the *most* dichotomies on any $N$ points.
  $$
  m_{\mathcal{H}}(N)=\underset{\mathbf{x}_1,\dots,\mathbf{x}_N\in\mathcal{X}}{max}|\mathcal{H}(\mathbf{x}_1,\dots,\mathbf{x}_N)|
  $$
  This translates to choosing any $N$ points and laying them out in *any* fashion in the input space. Determining $m$ is equivalent to looking for such a layout of the $N$ points that yields the *most* dichotomies. 

  The growth function satisfies:
  $$
  m_{\mathcal{H}}(N)\le 2^N
  $$
  This can be applied to the perceptron. For example, when $N=4$, we can lay out the points so that they are easily separated. However, given a layout, we must then consider all possible configurations of labels on the points, one of which is the following:

  <img src="images/perc.PNG" style="zoom:75%"/>

  This is where the perceptron breaks down because it *cannot* separate that configuration, and so $m_{\mathcal{H}}(4)=14$ because two configurations—this one and the one in which the left/right points are blue and top/bottom are red—cannot be represented. For this reason, we have to expect that that for perceptrons, $m$ can’t be the maximum possible because it would imply that perceptrons are as strong as can possibly be.

The *VC* ( *Vapnik-Chervonenkis ) dimension* of a hypothesis set $\mathcal{H}$ , denoted by $d_{VC}(\mathcal{H})$ is the largest value of $N$ for which $m_{\mathcal{H}}(N)=2^N$  , in other words is "*the most points $\mathcal{H}$ can shatter* " 

We can say that the *VC* dimension is one of many measures that characterize the expressive power, or capacity, of a hypothesis class. 

You can think of the VC dimension as "how many points can this model class memorize?" (a ton? $\to$ BAD! not so many? $\to$ GOOD!)

----

With respect to learning, the effect of the VC dimension is that if the VC dimension is finite, then the hypothesis will generalize:
$$
d_{vc}(\mathcal H)\ \Longrightarrow\ g \in \mathcal H \text { will generalize }
$$
The key observation here is that this statement is independent of:

- The learning algorithm
- The input distribution
- The target function

The only things that factor into this are the training examples, the hypothesis set, and the final hypothesis.

The VC dimension for a linear classifier (i.e. a *line* in 2D, a *plane* in 3D etc...) is $d+1$ (a line can shatter at most $2+1=3$ points, a plane can shatter at most $3+1=4$ points etc...)

Proof: [here](<http://wittawat.com/posts/vc_dimension_linear_classifier.html>)

`READ THE SECTION ON PMDS TOO! `

<div style="page-break-after: always;"></div> 

### Ridge Regression

**Describe the supervised learning technique called ridge regression for regression problems.**  
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
(Sources: [tds - Ridge And Lasso Regression](https://towardsdatascience.com/ridge-and-lasso-regression-a-complete-guide-with-python-scikit-learn-e20e34bcbf0b ) -  [tds - Regularization in ML](https://towardsdatascience.com/regularization-in-machine-learning-76441ddcf99a) )

Since ${\lambda}$ is not defined a priori, we need a method to select a good value for it.  
We use Cross-Validation for solving this problem: we choose a grid of ${\lambda}$ values, and compute   
the cross-validation error rate for each value of ${\lambda}$.  
​We then select the value for ${\lambda}$ for which the cross-validation error is the smallest.  
​Finally, the model is re-fit using all of the available observations and the selected value of ${\lambda}$.  
​Restelli offers the following cost function notation:

${L(\mathbf{w})=L_D(\mathbf{w})+\lambda L_W(\mathbf{w}) }$

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

Ridge Regression is, for example, used when the number of samples is relatively small wrt the  
number of features.  
Ridge Regression can improve predictions made from new data (i.e. reducing variance) by  
making predictions less sensitive to the Training Data.

(Source: [statquests explanation](https://www.youtube.com/watch?v=Q81RR3yKn30))

<div style="page-break-after: always;"></div> 

### Ridge vs Lasso

***Describe and compare the ridge regression and the LASSO algorithms.***    
*(William Bonvini)*

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

Now consider the ${l_1}$ and ${l_2}$ norms of various solutions, remembering that ${l_1=|w_1+w_2|}$ and that  ${l_2=(w_1^2+w_2^2)}$ .  

| ${w_1}$ | ${w_2}$ | ${l_1}$ | ${l_2}$ |
| ------- | ------- | ------- | ------- |
| 4       | 0       | 4       | 16      |
| 2       | 2       | 4       | 8       |
| 1       | 3       | 4       | 10      |
| -1      | 5       | 6       | 26      |

we can see that minimizing ${l_2}$ we obtain ${w_1=w_2=2}$, which means that it, in this case, tends to spread equally the weights.  
While ${l_1}$ can choose arbitrarily between the first three options, as long as the weights have the same sign it's ok. 

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

<div style="page-break-after: always;"></div> 

### Ridge Regression vs Bayesian Linear Regression

***Describe the ridge regression algorithm and compare it with the Bayesian linear regression approach.***
*(William Bonvini)* 

`INCOMPLETE: THIS JUST EXPLAIN THE BAYESIAN APPROACH (ALREADY SEEN IN SOFT COMPUTING -> MAXIMUM A POSTERIORI ESTIMATION)`

I've already described Ridge Regression previously.

*Comparison:*

Ridge Regression is a frequentist approach:  
the model assumes that the response variable (y) is a linear combination of weights multiplied by a set of predictor variables (x). The full formula also includes an error term to account for random sampling noise. 

What we obtain from frequentist linear regression is a single estimate for the model parameters based only on the training data. Our model is completely informed by the data: in this view, everything that we need to know for our model is encoded in the training data we have available.

Ridge Regression gives us a single point estimate for the output. However, if we have a small dataset we might like to express our estimate as a distribution of possible values. This is where Bayesian Linear Regression comes in.

The aim of Bayesian Linear Regression is not to find the single “best” value of the model parameters, but rather to determine the *posterior distribution* (*a probability distribution that represents your updated beliefs about the parameter after having seen the data*) for the model parameters.  
Not only is the response generated from a probability distribution, but the model parameters are assumed to come from a distribution as well. The posterior probability of the model parameters is conditional upon the training inputs and outputs:
$$
P(\beta|y,X)=\frac{P(y|\beta,X)P(\beta|X)}{P(y|X)}
$$
Here, ${P(\beta |y,X)}$ is the posterior probability distribution of the model parameters given the inputs and outputs. This is equal to the likelihood of the data, ${P(y|\beta,X)}$, multiplied by the prior probability of the parameters and divided by a normalization constant. This is a simple expression of Bayes Theorem, the fundamental underpinning of Bayesian Inference:
$$
Posterior = \frac{Likelihood*Prior}{Normalization}
$$
Let's stop and think about what this means. In contrast to Ridge Regression , or Linear Regression in general, we have a *posterior* distribution for the model parameters that is proportional to  

- The likelihood of the data
- The prior probability of the parameters

Here we can observe the two primary benefits of Bayesian Linear Regression:

1. **Priors**:   
   if we have domain knowledge, or a guess for what the model parameters should be, we can include them in our model, unlike in the frequentist approach which assumes everything there is to know about the parameters comes from the data. If we don't have any estimates ahead of time, we can use <u>non-informative priors</u> for the parameters such as a normal distribution.
2. **Posterior**:  
   The result of performing Bayesian Linear Regression is a distribution of possible model parameters based on the data and the prior.  
   This allows us to quantify our uncertainty about the model: if we have fewer data points, the posterior distribution will be more spread out.

The formulation of model parameters as distributions encapsulates the Bayesian worldview: we start out with an initial estimate, our prior, and as we gather more evidence, **our model becomes less wrong**. Bayesian reasoning is a natural extension of our intuition. Often, we have an initial hypothesis, and as we collect data that either supports or disproves our ideas, we change our model of the world (ideally this is how we would reason)!

(Sources: [towardsdatascience - Introduction to Bayesian Linear Regression](https://towardsdatascience.com/introduction-to-bayesian-linear-regression-e66e60791ea7) )  

FORSE W_0 CHE DICE NELLE SLIDE SI RIFERISCE ALLA MEDIA E S_0 ALLA MATRICE DI COVARIANZA BOH

AAAAA GUARDA LA SLIDE CHE FA IL PARAGONE CON RIDGE!

***We can derive Ridge Regression from Bayesian Linear Regression!***
If we choose a prior distribution as follows:

![](images/br1.png)

![](images/br2.png)



<div style="page-break-after: always;"></div> 

### Logistic Regression

***Describe the logistic regression algorithm and compare it with the perceptron algorithm.***

Although the name might confuse, please note that it is a *classiﬁcation* algorithm.

Considering a problem of two-class classiﬁcation, in logistic regression the posterior probability of class $C_1$ can be written as a logistic sigmoid function:
$$
p(C_1|\phi) = \frac{1}{1+e^{-\mathbf{w}^T\phi}}=\sigma(\mathbf{w}^T\phi)
$$
![](images/sigmoid.png)

and $p(C_2|\phi) = 1 - p(C_1|\phi)$ 

Applying the *Maximum Likelihood* approach...

Given a dataset $\mathcal{D} = \{\mathbf{x}_n,t_n\}$, $t_n \in \{0,1\}$, we have to maximize the probability of getting the right label:
$$
P(\mathbf{t}|\mathbf{X},\mathbf{w}) = \prod_{n=1}^{N}y_n^{t_n}(1-y_n)^{1-t_n},\ \ y_n = \sigma(\mathbf{w}^T\phi_n)
$$
Taking the negative log of the likelihood, the *cross-entropy* error function can be deﬁned and it has to be minimized:
$$
L(\mathbf{w}) = -\ln P(\mathbf{t}|\mathbf{X},\mathbf{w}) = -\sum_{n=1}^{N}(t_n\ln y_n+(1-t_n)\ln(1-y_n))=\sum_{n}^NL_n
$$
Differentiating and using the chain rule:
$$
\frac{\part L_n}{\part y_n}= \frac{y_n-t_n}{y_n(1-y_n)},\ \ \ \ \frac{\part y_n}{\part\mathbf{w}}=y_n(1-y_n)\phi_n\\
\frac{\part L_n}{\part \mathbf{w}}= \frac{\part L_n}{\part y_n}\frac{\part y_n}{\part\mathbf{w}}=(y_n-t_n)\phi
$$
The gradient of the loss function is
$$
\nabla L(\mathbf{w}) = \sum_{n=1}^{N}(y_n-t_n)\phi_n
$$
It has the same form as the gradient of the sum-of-squares error function for linear regression. But in this case $y$ is not a linear function of $\mathbf{w}$ and so, there is no closed form solution. The error function is *convex* (only one optimum) and can be optimized by standard *gradient-based* optimization techniques. It is, hence, easy to adapt to the online learning setting.

Talking about *Multiclass Logistic Regression*...

For the multiclass case, the posterior probabilities can be represented by a *softmax* transformation of linear functions of feature variables:
$$
p(C_k|\phi)=y_k(\phi)=\frac{e^{\mathbf{w}_k^T\phi}}{\sum_j e^{\mathbf{w}_j^T\phi}}
$$
$\phi(\mathbf{x})$ has been abbreviated with $\phi$ for simplicity.

*Maximum Likelihood* is used to directly determine the parameters
$$
p(\mathbf{T}|\Phi,\mathbf{w}_1,\dots,\mathbf{w}_K)=\prod_{n=1}^{N}{\underset{\text{Term for correct class$\;\;\;\;\;\;\;\;\;\;\;\;\;\;\,\,\;\;\;\;\;\;\;\;\;\;\;$}}{\underbrace{\left(\prod_{k=1}^{K}p(C_k|\phi_n)^{t_{nk}}\right)}=\prod_{n=1}^{N}\left(\prod_{k=1}^{K}y_{nk}^{t_{nk}}\right)}}\\
$$
where $y_{nk}=p(C_k|\phi_n)=\frac{e^{\mathbf{w}_k^T\phi_n}}{\sum_j e^{\mathbf{w}_j^T\phi_n}}$

The *cross-entropy* function is:
$$
L(\mathbf{w}_1,\dots,\mathbf{w}_K)=-\ln p(\mathbf{T}|\Phi,\mathbf{w}_1,\dots,\mathbf{w}_K)=-\sum_{n=1}^{N}\left(\sum_{k=1}^{K}t_{nk}\ln y_{nk}\right)
$$
Taking the gradient
$$
\nabla L_{\mathbf{w}_j}(\mathbf{w}_1,\dots,\mathbf{w}_K) =\sum_{n=1}^{N}(y_{nj}-t_{nj})\phi_n
$$
The *perceptron* is an example of *linear discriminant models*, it is an *online* linear classification algorithm.
$$
y(\mathbf{x})=f(\mathbf{w}^T\phi(\mathbf{x}))\\
\text{where}\\

f(a)=
\begin{cases}
+1,\ \ a\ge0\\
-1,\ \ a < 0
\end{cases}
$$
Target values are $+1$ for $C_1$ and $-1$ for $C_2$.

The algorithms finds the *separating hyperplane* by minimizing the distance of *misclassified points* to the *decision boundary*.

Using the number of misclassified points as loss function is not effective since it is a *piecewise constant function*.

We are seeking a vector $\mathbf{w}$ such that $\mathbf{w}^T\phi(\mathbf{x}_n)>0$ when $\mathbf{x}_n \in C_1$ and $\mathbf{w}^T\phi(\mathbf{x}_n)<0$ otherwise.

The *perceptron criterion* assigns

- zero error to correct classification
- $\mathbf{w}^T\phi(\mathbf{x}_n)t_n$ to misclassified patterns $\mathbf{x}_n$ (it is proportional to the distance to the decision boundary)

The *loss* function to be minimized is
$$
L_P(\mathbf{w}) =-\sum_{n\in \mathcal{M}}\mathbf{w}^T\phi(\mathbf{x}_n)t_n
$$
Minimization is performed using *stochastic gradient descent* :
$$
\mathbf{w}^{k+1}=\mathbf{w}^k-\alpha\nabla L_P(\mathbf{w})=\mathbf{w}^k+\alpha\phi(\mathbf{x}_n)t_n
$$
Since the perceptron function does not change if $\mathbf{w}$ is multiplied by a constant, the *learning rate $\alpha$ can be set to $1$*

The effect of a single update is to *reduce the error* due to the *misclassified pattern*, this *does not imply* that the *loss* is reduced at each stage.

*Theorem* (***Perceptron Convergence Theorem***)

"If the training data set is *linearly separable* in the feature space $\phi$, then the perceptron learning algorithm is guaranteed to ﬁnd an exact solution in a ﬁnite number of steps."

*Problems*:

- The number of steps before convergence may be substantial.
- We are not able to distinguish between *non-separable* problems and *slowly converging* ones.
- If multiple solutions exist, the one found depends by the *initialization* of the parameters and the *order of presentation* of the data points.

![](images/LRvsPer.PNG)

*Please Note* : Here we used the *step function* instead of the *sign function* for the Perceptron!


<div style="page-break-after: always;"></div> 

### PCA

***Describe the Principal Component Analysis technique***

*PCA* is an unsupervised learning method which aims to *reduce* the dimensionality of an input space $\mathcal{X}$ .

Formally, principal component analysis (PCA) is a statistical procedure that uses an *orthogonal transformation* to convert a set of observations of possibly correlated variables into a set of values of *linearly uncorrelated* variables called *principal components*.

To have a graphical intuition:

<img src="images/PCA.png" style="zoom:60%"/>

It is based on the principle of projecting the data onto the input subspace which accounts for most of the variance: 

- Find a line such that when the data is projected onto that line, it has the maximum variance. 
- Find a new line, orthogonal to the ﬁrst one, that has maximum projected variance. 
- Repeat until $m$ lines have been identiﬁed and project the points in the data set on these lines. 

The precise steps of *PCA* are the following (remember that $\mathbf{X}$ is an $n\times d$ matrix where $n$ denotes the number of samples and $d$ is the dimensionality) : 

- Compute the mean of the data
  $$
  \overline{\mathbf{x}} = \frac{1}{N}\sum_{n=1}^N\mathbf{x}_n
  $$

- Bring the data to zero-mean (by subtracting $\overline{\mathbf{x}}$)

- Compute the covariance matrix $\mathbf{S} = \mathbf{X}^T\mathbf{X} = \frac{1}{N-1}\sum_{n=1}^{N}(\mathbf{x}_n-\overline{\mathbf{x}})^T(\mathbf{x}_n-\overline{\mathbf{x}})$

  - Eigenvector $\mathbf{e}_1$ with largest eigenvalue $\lambda_1$ is the *first principal component* 
  - Eigenvector $\mathbf{e}_k$ with $k^{th}$ largest eigenvalue $\lambda_k$ is the *$k^{th}$ principal component*
  - $\frac{\lambda_k}{\sum_i\lambda_i}$ is the proportion of variance captured by the $k^{th}$ principal component.

Transforming the reduced dimensionality projection back into the original spaces gives a reduced dimensionality reconstruction of the data, that will have some error. This error can be small and often acceptable given the other beneﬁts of dimensionality reduction. PCA has multiple beneﬁts:

- Helps to reduce the computational complexity 
- Can help supervised learning, because reduced dimensions allow simpler hypothesis spaces and less risk of overﬁtting 
- Can be used for noise reduction 

But also some drawbacks:

- Fails when data consists of multiple clusters
- The directions of greatest variance may not be the most informative
- Computational problems with many dimensions 
- PCA computes linear combination of features, but data often lies on a nonlinear manifold. Suppose that the data is distributed on two dimensions as a circumference: it can be actually represented by one dimension, but PCA is not able to capture it.

<div style="page-break-after: always;"></div> 

### Gaussian Processes

***Describe the Gaussian Processes model for regression problems*** (*kriging*)

In probability theory and statistics, a **Gaussian process** is a stochastic process (a collection of random variables indexed by time or space), such that every finite collection of those random variables has a multivariate normal distribution, i.e. every finite linear combination of them is normally distributed. The distribution of a Gaussian process is the joint distribution of all those (infinitely many) random variables, and as such, it is a distribution over functions with a continuous domain, e.g. time or space.

A machine-learning algorithm that involves a Gaussian process uses `lazy learning` and a measure of the similarity between points (the *kernel function*) to predict the value for an unseen point from training data. The prediction is not just an estimate for that point, but also has uncertainty information—it is a one-dimensional Gaussian distribution (which is the marginal distribution at that point).

`*lazy learning* is a learning method in which generalization of the training data is, in theory, delayed until a query is made to the system, as opposed to in *eager learning* , where the system tries to generalize the training data before receiving queries. The main advantage gained in employing a lazy learning method is that the target function will be approximated locally, such as in the k-nearest neighbor algorithm. Because the target function is approximated locally for each query to the system, lazy learning systems can simultaneously solve multiple problems and deal successfully with changes in the problem domain.`

*Slides Definition*:

A *Gaussian Process* is defined as a probability distribution over functions $y(\mathbf{x})$ such that the set of values of $y(\mathbf{x})$ evaluated at an arbitrary set of points $\mathbf{x}_1,\dots,\mathbf{x}_n$ *jointly have a Gaussian distribution*. 

The distribution is completely specified by the second-order statistics, the mean and the covariance:

- Usually we do not have any prior information about the mean of $y(x)$, so we'll take it to be $0$.

- The covariance is given by the *kernel function*
  $$
  \mathbb{E}[y(\mathbf{x}_n)y(\mathbf{x}_m)] = k(\mathbf{x}_n,\mathbf{x}_m)
  $$

Let's talk about *Gaussian Process Regression* (aka *Kriging*)

Take into account the noise on the target
$$
t_n = y(\mathbf{x_n})+\epsilon_n
$$
Random noise under a *Gaussian distribution*
$$
p(t_n|y(\mathbf{x}_n))=\mathcal{N}(t_n|y(\mathbf{x}_n),\sigma^2)
$$
Because the noise is *independent* on each data point, the joint distribution is still *Gaussian*:
$$
p(\mathbf{t}|\mathbf{y})=\mathcal{N}(\mathbf{t}|\mathbf{y},\sigma^2\mathbf{I}_N)
$$
Since $p(\mathbf{y}) = \mathcal{N}(\mathbf{y}|\mathbf{0},\mathbf{K})$ , we can compute the marginal distribution:
$$
p(\mathbf{t})=\int p(\mathbf{t}|\mathbf{y})p(\mathbf{y})d\mathbf{y}=\mathcal{N}(\mathbf{t}|\mathbf{0},\mathbf{C})
$$
where $C(\mathbf{x}_n,\mathbf{x}_m) = k(\mathbf{x}_n,\mathbf{x}_m)+\sigma^2\delta_{nm}$.

Since the two *Gaussians* are *independent* their covariances simply *add*.

<div style="page-break-after: always;"></div> 

### Kernels

***Describe the purpose of using kernels in Machine Learning techniques. How can you construct a valid Kernel? Provide an example of a ML method using kernels and describe the speciﬁc advantage of using them for this method.***

Traditionally, theory and algorithms of machine learning and statistics have been very well developed for the linear case. Real world data analysis problems, on the other hand, often require nonlinear methods to detect the kind of dependencies that allow successful prediction of properties of interest. By using a positive definite kernel, one can sometimes have the best of both worlds. The kernel corresponds to a dot product in a (*usually high-dimensional, possibly infinite*) feature space. In this space, our estimation methods are linear, but as long as we can formulate everything in terms of kernel evaluations, we never explicitly have to compute in the high dimensional feature space! (This is called the *Kernel Trick*)

Suppose we have a mapping $\varphi : \R^d \to \R^m$ that brings our vectors in to some feature space $\R^m$. Then the dot product of $\textbf{x}$ and $\textbf{y}$ in this space is $\varphi (\textbf{x})^T\varphi (\textbf{y})$. A kernel is a function $k$ that corresponds to this dot product, i.e. $k(\textbf{x},\textbf{y})=\varphi (\textbf{x})^T\varphi (\textbf{y}) $ . Why is this useful? *Kernels* give a way to compute dot products in some feature space without even knowing what this space is and what is $\varphi$ . For example, consider a simple polynomial kernel $k(\textbf{x},\textbf{y})=(1+\textbf{x}^T\textbf{y})^2$ with $\textbf{x},\textbf{y} \in \R^2$. This doesn't seem to correspond to any mapping function $\varphi$ ,  it's just a function that returns a real number. Assuming that $\textbf{x} = (x_1,x_2)$ and $\textbf{y} = (y_1,y_2)$, let's expand this expression:
$$
k(\textbf{x},\textbf{y})=(1+\textbf{x}^T\textbf{y})^2 = (1+x_1y_1 + x_2y_2)^2=\\1+x_1^2y_1^2+x_2^2y_2^2+2x_1y_1+2x_2y_2+2x_1x_2y_1y_2
$$
Note that this is nothing else but a dot product between two vectors$(1, x_1^2, x_2^2, \sqrt{2} x_1, \sqrt{2} x_2, \sqrt{2} x_1 x_2)$ and $(1, y_1^2, y_2^2, \sqrt{2} y_1, \sqrt{2} y_2, \sqrt{2} y_1 y_2)$ and $\varphi(\mathbf x) = \varphi(x_1, x_2) = (1, x_1^2, x_2^2, \sqrt{2} x_1, \sqrt{2} x_2, \sqrt{2} x_1 x_2)$.

So the kernel $k(\mathbf x, \mathbf y) = (1 + \mathbf x^T \mathbf y)^2 = \varphi(\mathbf x)^T \varphi(\mathbf y)$ computes a dot product in 6-dimensional space without explicitly visiting this space.

Another example is Gaussian kernel $k(\mathbf x, \mathbf y) = \exp\big(- \gamma \, \|\mathbf x - \mathbf y\|^2 \big)$. If we Taylor-expand this function, we'll see that it corresponds to an infinite-dimensional codomain of $\varphi$.

Instead, the simplest *kernel* is the *linear kernel* which corresponds to an *identity mapping* in the feature space: $k(\mathbf{x},\mathbf{x'}) = \varphi(\mathbf{x})^T\varphi(\mathbf{x'}) = \mathbf{x}^T\mathbf{x}$ 

Moreover, the *kernel* is a *symmetric* function of its arguments: $k(\mathbf{x},\mathbf{x'}) = k(\mathbf{x'},\mathbf{x})$

------

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
So, given $N$ vectors, the *Gram Matrix* is the matrix of all *inner products* 

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

So we can make our prediction for a new input $\mathbf{x}$ (which has dimension $D\times 1$ obviously, $\varphi(\mathbf{x})$ will have dimension $M\times 1$ instead, where $M>D$) by substituting back into our linear regression model:
$$
y(\mathbf{x}) = \mathbf{w}^T\varphi(\mathbf{x}) = (\Phi^T\mathbf{a})^T\varphi(\mathbf{x}) = \mathbf{a}^T\Phi\varphi(\mathbf{x})= \mathbf{k}(\mathbf{x})^T(K+\lambda\mathbf{I}_N)^{-1}\mathbf{t}
$$
where $\mathbf{k}(\mathbf{x})$ has elements $k_n(\mathbf{x}) = k(\mathbf{x}_n,\mathbf{x})$ . Prediction is just a linear combination of the *target values* from the *training set* . (If you make a dimensionality check you will see that $y(\mathbf{x})$ will be just a number)

The good thing is that instead of inverting an $M\times M$ matrix, we are inverting an $N\times N$ matrix! (as we already said different times,  this allows us to work with *very high or infinite dimensionality* of $\mathbf{x}$).

------

 But *how* can we build a valid *kernel*?

 We have manly two ways to do it:

- *By construction*: we choose a feature space mapping $ \varphi (\mathbf{x})$ and use it to ﬁnd the corresponding kernel. (I'd call this method *by hand*)

- It is possible to test whether a function is a valid kernel without having to construct the basis function explicitly. The necessary and suﬃcient condition for a function $k(\mathbf{x},\mathbf{x}')$ to be a kernel is that the Gram matrix $K$ is positive semi-deﬁnite for all possible choices of the set $\{x_n\}$. It means that $\mathbf{x}^TK\mathbf{x}\ge 0$ for non-zero vectors $\mathbf{x}$ with real entries, i.e.$\sum_n\sum_m K_{n,m}x_nx_m \ge 0$ for any real number $x_n,x_m$. 

  *Mercer's Theorem :* Any continuous, symmetric, positive semi-deﬁnite kernel function $k(\mathbf{x},\mathbf{y})$ can be expressed as a dot product in a high-dimensional space.

  New kernels can be constructed from simpler kernels as *building blocks*:

  ![](images/Kernels.PNG)

***What is Kernel Regression?***

The *radial basis function* kernel is a popular kernel function used in various kernelized learning algorithms. In particular, it is commonly used in support vector machine classification. The RBF kernel on two samples $\mathbf{x}$ and $\mathbf{x}'$, represented as feature vectors in some *input space*, is defined as:
$$
K(\mathbf{x},\mathbf{x}')=e^{-\frac{||\mathbf{x}-\mathbf{x}'||^2}{2\sigma^2}}
$$
where $||\mathbf{x}-\mathbf{x}'||^2$ may be recognized as the squared Euclidean distance between two feature vectors, $\sigma$ is a free parameter. Since the value of the RBF kernel decreases with distance and ranges between $0$ (in the limit) and $1$ (when $\mathbf{x} =\mathbf{x}'$) , it has a ready interpretation as a similarity measure. It can be seen (*by expansion)* that the feature space of the kernel has an infinite number of dimensions.

But how is this framework related to *regression*? $\to$ *Kernel Regression*!

Before we dive into the actual regression algorithm, let’s look at the approach from a high level. Let’s say you have the following scatter plot, and you want to approximate the $y$ value at $x = 60$. We’ll call this our "query point".

<img src="C:/Users/andre/Desktop/Github/Notes/MachineLearningRestelli/images/KR1.png" style="zoom:70%"/>

How would you go about it? One way would be to look at the data points near $x = 60$, say from $x = 58$ to $x = 62$, and average their $y$ values. Even better would be to somehow weight the values based on their distance from our query point, so that points closer to $x = 60$ got more weight than points farther away.

This is precisely what *Gaussian Kernel Regression* does, it takes a weighted average of the surrounding points. Say we want to take the weighted average of three values: $3$, $4$ and $5$. To do this, we multiply each value by its weight (I've chosen some arbitrary weights: $0.2$,$0.4$ and $0.6$) , take the sum, then divide by the sum of the weights:
$$
\frac{0.2\cdot3+0.4\cdot4+0.6\cdot5}{0.2+0.4+0.6}=\frac{5.2}{1.2}=4.33
$$
More generally, the weighted average is found as:
$$
\overline{y}=\frac{\sum_{i=1}^m(w_iy_i)}{\sum_{i=1}^mw_i}
$$
where $w_i$ is the weight to assign to value $y_i$ and $m$ is the number of values in the set.

In *Kernel Regression* in order to compute the weight values to use in our regression problem, we're going to use the *Gaussian Function*, which has the perfect behavior for computing our weight values! The function will produces its highest value when the distance between the data point and the query point is zero. For data points farther from the query, the weight value will fall off exponentially. 

To arrive at the final equation for Gaussian Kernel Regression, we’ll start with the equation for taking a weighted average and replace the weight values with our *Gaussian* kernel function.
$$
y^*=\frac{\sum_{i=1}^m(K(x^*,x_i)y_i)}{\sum_{i=1}^mK(x^*,x_i)}
$$
It is interesting to note that Gaussian Kernel Regression is equivalent to creating an RBF Network with the following properties:

- Every training example is stored as an RBF neuron center
- The $\beta$ coefficient ( the *first* set of weights) for every neuron is set to the same value.
- There is one output node.
- The output weight for each RBF neuron is equal to the output value ( $y_i$ ) of its data point.
- The output of the RBFN must be normalized by dividing it by the sum of all of the RBF neuron activations.



![](C:/Users/andre/Desktop/Github/Notes/MachineLearningRestelli/images/RBFN.png)

The input can be modeled as a vector of real numbers $\mathbf{x}\in \mathbb{R}^n$. The output of the network is then a scalar function of the input vector, $\varphi:\R^n\to\R$ and is given by
$$
\varphi(\mathbf{x})=\sum_{i=1}^Ny_i\rho(||\mathbf{x}-\mathbf{x}_i||)
$$

$$
\rho(||\mathbf{x}-\mathbf{x}_i||)=e^{-\frac{||\mathbf{x}-\mathbf{x}_i||^2}{2\sigma_i^2}}=e^{-\beta_i||\mathbf{x}-\mathbf{x}_i||^2}
$$

Most of the times it is convenient to use *normalized* radial function as basis. Normalization is used in practice as it avoids having regions of input space where all basis functions take *small values*, which would necessarily lead to predictions in such regions that are either *small* or controlled purely by the *bias parameter*. In this case we have
$$
\varphi(\mathbf{x})=\sum_{i=1}^{N}y_i u(||\mathbf{x}-\mathbf{x}_i||) \\
u||\mathbf{x}-\mathbf{x}_i|| = \frac{\rho||\mathbf{x}-\mathbf{x}_i||}{\sum_{j=1}^N\rho||\mathbf{x}-\mathbf{x}_j||}
$$
Here is a $1$-D example, just to give you an idea:

`Here we use` $c_1$ `and` $c_2$ `as *centroids*, it makes sense that we don't want always to average over *all* the samples of our dataset, instead we can choose some *relevant* points (that I call *centroids* ) in our formulation by performing, for example, some local averaging...That was kinda what we did in fuzzy systems![Soft-Computing](apart from the fact that we didn't use gaussians as Membership Functions). According to David Salsbrug: "Coming up with almost exactly the same computer algorithm, fuzzy systems and kernel density-based regressions appear to have been developed completely independently of one another.`

<img src="C:/Users/andre/Desktop/Github/Notes/MachineLearningRestelli/images/URB1.png" style="zoom:70%"/>

Two unnormalized radial basis functions in one input dimension. The basis function centers are located at $c_1=0.75$ and $c_2=3.25$.

<img src="C:/Users/andre/Desktop/Github/Notes/MachineLearningRestelli/images/URB2.png" style="zoom:70%"/>

Two normalized radial basis functions in one input dimension. The basis function centers are the same as before, in this specific case the activation functions become *sigmoids*!

Let's try to derive the *kernel regression* formulation more formally:

*Kernel Regression* is a non-parametric technique in statistics to estimate the *conditional expectation* of a *random variable*. The objective is to find a non-linear relation between a pair of random variables $\mathbf{X}$ and $\mathbf{Y}$. In any nonparametric regression, the conditional expectation of a variable $\mathbf{Y}$ relative to a variable $\mathbf{X}$ may be written:
$$
\mathbb{E}(Y|X) = m(X)
$$
where $m$ is an unknown function.

*Nadaraya* and *Watson*, both in 1964, proposed to estimate $m$ as a locally weighted average, using a kernel as a weighting function. The Nadaraya-Watson estimator is:
$$
\hat{m_h}(x) =\frac{\sum_{i=1}^nK_h(x-x_i)y_i}{\sum_{j=1}^nK_h(x-x_j)}
$$
where $K_h$ is a kernel with a bandwidth $h$ (which is related to the variance). The denominator is a weighting term with sum $1$.

*Derivation*:
$$
\mathbb{E}(Y|X=x) = \int{yf(y|x)dy}=\int y\frac{f(x,y)}{f(x)}dy
$$
Using the *kernel density estimation* (also termed the *Parzen–Rosenblatt* window method, it is just a non parametric way to estimate the *pdf* of a random variable) for both the joint distribution $f(x,y)$ and $f(x)$ with a kernel $K$
$$
\hat{f}(x,y) = \frac{1}{n}\sum_{i=1}^{n}K_h(x-x_i)K_h(y-y_i)\\
\hat{f}(x) = \frac{1}{n}\sum_{i=1}^{n}K_h(x-x_i)
$$
we get
$$
\hat{\mathbb{E}}(Y|X=x)=\int \frac{y\sum_{i=1}^{n}K_h(x-x_i)K_h(y-y_i)}{\sum_{j=1}^{n}K_h(x-x_j)}dy\\
=\frac{\sum_{i=1}^{n}K_h(x-x_i)\int yK_h(y-y_i)dy}{\sum_{j=1}^{n}K_h(x-x_j)}\\
=\frac{\sum_{i=1}^{n}K_h(x-x_i)y_i}{\sum_{j=1}^{n}K_h(x-x_j)}\\
$$


<div style="page-break-after: always;"></div> 

# Reinforcement Learning

### Q-Learning vs SARSA

***Describe the diﬀerences existing between the Q-learning and SARSA algorithms***  
*(William Bonvini)*  
First of all, let's say what they are used for:  
*SARSA* and *Q-Learning* are two algorithms used to do control using the model free method called *Temporal Difference*.   
If you don't remember what a control task is, here you are:   
*Control is the task of obtaining an improved policy ${\pi'}$ starting from a policy ${\pi}$.*  
Now let's jump into the differences:  
Q-Learning is an example of off-policy learning, while SARSA is an example of on-policy learning.   
It implies that    

- Q-learning uses a target policy (greedy) to choose the best next action ${a'}$ while following the behavior policy (${\epsilon}$-greedy)  (off-policy).  =={TODO: clearify}==
  ${Q(S_t,A_t)\leftarrow Q(S_t,A_t)+ \alpha \big( \color{red} R_{t+1}+\gamma \max_{a' \in A}  Q(S_{t+1},a') \color{black} - Q(S_t,A_t)\big)}$   

  

- SARSA learns the Q-value based on the action performed following its own policy (on-policy)  
  ${Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha (\color{red} R+\gamma Q(S_{t+1},A_{t+1}) \color{black} -Q(S_t,A_t))}$   


If you want to get a full understanding of both algorithms, here you are:

***SARSA Algorithm***  

  <img src="images/sarsa1.jpg" style="zoom:45%"/>

It's called SARSA because the agent starts in ${S}$, performs action ${A}$ following its own policy. 

Afterwards, we are going to randomly sample from our environment to see what reward ${R}$ we receive and that state ${S'}$ we end up in.  
then we are going to sample again from our policy to generate ${A'}$.  
So basically, SARSA, indicates a particular update pattern we can use.  
*Updating ${Q}$ functions with SARSA*:  
Now let's study out update function:
${Q(S,A)\leftarrow Q(S,A)+\alpha (\color{red} R+\gamma Q(S',A') \color{black} -Q(S,A))}$

We move our ${Q}$ value a little bit in the direction of our TD target (the red colored part) minus the ${Q}$ value of where we started.  
This update is done after every transition from a nonterminal state ${s}$. If ${s'}$ is terminal, then ${Q(s',a')}$ is zero.

*Policy Improvement/ Control with SARSA*:  
Ok, so far we did prediction: we updated our ${Q}$ function using the formula above. Implicitly we did Policy Evaluation. How do we do Policy Improvement when we apply SARSA?

we simply use an ${\epsilon}$- greedy policy improvement:  

  - All ${m}$ actions are tried with non-zero probability.
  - With probability ${1-\epsilon}$ we choose the greedy action
  - With probability ${\epsilon}$ we choose an action at random (possibly we select the greedy one!)

$$
  \pi(s,a)=\begin{cases}\frac{\epsilon}{m}+1-\epsilon  \ \ \ \ if \ \    a^*=arg\max_{a\in A} Q(s,a) \\
  \frac{\epsilon}{m} \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ otherwise\end{cases} 
$$

***Q-Learning***  

Consider being in state ${S_t}$ and selecting our next action following our behavioral policy: ${A_{t+1}\sim \bar{\pi}(\cdot |S_t)}$.  
In the meanwhile consider some alternative successor action that we might have taken following our target policy ${A'\sim\pi(\cdot|S_t)}$. 
Finally we update our ${Q}$-value for the state we started in and the action that we actually took (${Q(S_t,A_t)}$) towards the value of our alternative action ${A'}$. 

${Q(S_t,A_t)\leftarrow Q(S_t,A_t)+ \alpha( \color{red} R_{t+1}+\gamma Q(S_{t+1},A') \color{black} - Q(S_t,A_t))}$

${S_t}$ = actual state  
${A_t}$= actual action taken following behavioral policy ${\pi}$.   
${\alpha}$ = learning rate  
${R_{t+1}}$ = actual reward taken by performing ${A_t}.$    

${\gamma}$ = discounting factor.  
${S_{t+1}}$= successor state.

${A'}$ = action sampled from our target policy in state ${S_t}$. 

A special case of this updating process is the Q-Learning algorithm.  
In this case, the target policy ${\pi}$ is a greedy policy wrt ${Q(s,a)}$ and the behavior policy ${\bar{\pi}}$ is ${\epsilon}$-greedy wrt ${Q(s,a)}$.
$$
\pi(S_{t+1})=\arg\max_{a'}Q(S_{t+1},a')
$$

Let's update the new estimation of the final return:
$$
R_{t+1} +\gamma Q(S_{t+1},A')=          \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \
$$

$$
R_{t+1}+\gamma Q(S_{t+1},\arg\max_{a'}Q(S_{t+1},a') )=
$$

$$
R_{t+1}+\max_{a'} \gamma Q(S_{t+1},a') \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \
$$

If we plug this estimation in the general Q update equation I described earlier, just by replacing the  
old red colored component with the new one, we obtain the Q-update equation for Q-Learning:     
$$
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+ \alpha \big( \color{red} R_{t+1}+\gamma \max_{a' \in A}  Q(S_{t+1},a') \color{black} - Q(S_t,A_t)\big)
$$
(Sources: [Model Free Algorithms](https://medium.com/deep-math-machine-learning-ai/ch-12-1-model-free-reinforcement-learning-algorithms-monte-carlo-sarsa-q-learning-65267cb8d1b4) - [Deep Mind Model Free Control](https://www.youtube.com/watch?v=0g4j2k_Ggc4&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ&index=5) )

<div style="page-break-after: always;"></div> 

### Monte Carlo vs Temporal Difference

***Describe the diﬀerences existing between the Montecarlo and the Temporal Diﬀerence methods in the model-free estimation of a value function for a given policy.***  
*(William Bonvini)*  
If you are looking for a concise answer just go to the end.  
Monte Carlo and Temporal Difference are two different algorithms to solve Model Free Reinforcement Learning planning problems.

The question asks us to find the differences between MC-Prediction and TD-Prediction, so let's first write down both update equations:  

*Monte-Carlo Update:*  
The following updates are used *each time* an episode ends.  
For each state ${s_t}$ with return ${v_t}$: 

Stationary Case:
$$
N(s_t)\leftarrow N(s_t)+1
$$

$$
V(s_t)\leftarrow V(s_t)+ \frac{1}{N(s_t)}\bigg(v_t-V(s_t)\bigg)
$$

Non Stationary Case (we use a running mean: we forget old episodes thanks to ${\alpha}$):
$$
\color{blue}V(s_t)\leftarrow V(s_t)+ \alpha\bigg(\color{red}v_t\color{blue}-V(s_t)\bigg)
$$
${v_t}$: the return we observed from ${s_t}$ during the current episode:  
${v_t=G_t=R_{t+1}+\gamma R_{t+2}+...+\gamma^{T-1}R_T}$

${N(s_t)}$: the total number of times I visited state ${s_t}$ along all episodes I've ever run.   
Usually, in real world scenarios, we deal with non-stationary setups: we don't want to remember everything about the past. This is the case in RL as well, because during the control task we keep on updating our policy and get better and better evaluations of our states, so we want to get rid of the contribute of old evaluations  ${\to}$ we always go with the non-stationary update function.  
(if you haven't understood this last part don't worry, it will be clearer once you study control techniques).

${}$

*TD-Update*  
$$
\color{blue} V(s_t)\leftarrow V(s_t)+\alpha \bigg(\color{red}r_{t+1}+\gamma V(s_{t+1})\color{blue}-V(s_t)\bigg)
$$
The one above is the simplest temporal-difference learning algorithm, called ${TD(0)}$. We'll use it as reference. The red colored part is called TD-target.

In this case we are updating our value function towards the estimated return after one step:  
Specifically, the estimate consists in two parts: the immediate reward ${r_{t+1}}$ plus the discounted value of the next step ${\gamma V(S_{t+1})}$. 

**Gimme the differences!**

Ok, here you are:  
Let's start from a concrete example:  
Imagine you are driving your car and suddenly see another car moving towards you. You think you are going to crash, but in the end the car swerves out of the way, and you don't actually crash. In MC you wouldn't get any crash in the end so you wouldn't update your values, even if you almost died.  
In TD learning, you start thinking that everything's fine, but ,when you see the car coming towards you, you immediately update you estimate of the return ("damn it, If I don't do something I'm gonna die") and choose to slow down the car (which translates into choosing the action of decelerating because it gives a better estimate of the final reward)

You asked for a second example? no? sorry:  

Another driving example.  
We need to predict how much time we need to reach home from our office.  
"Elapsed time" is the time passed, "Predicted Time to Go" is the time we are predicting to need in order to get home, "Predicted Total Time" is the total time, starting from our office, we predict to need to get home.  
So: the "Elapsed Time" from a state ${i}$ to a state ${j}$ is our reward: from the office to the car we get 5 minutes of reward, from the car to the exit of the highway we get 15 minutes of reward. "Predicted time to go" is our value function. The sum of the total elapsed time and the "Predicted Time to Go" give us the total predicted time.  
(But wait, why is the reward positive? shouldn't it be negative? yes, but it's just an example, easier to deal with positive values).

![](images\driving.png)

Each row corresponds to a state: the starting state is the office, then we get to the parking lot, highway, etc.  
Now, consider ${\alpha=1}$ for both MC and TD (just to make things easier). what does this mean? that we completely forget about the past (just look at the equations above and you'll find out why).  
The following charts show you how both algorithms update the value function of each state (for visualization purposes Sutton (the book) plots the predicted total time instead of the Predicted Time to Go):

![](images\driving_mc_td.png)

MC updates every state to have a "predicted total time" equal to the end estimate, so to 43:  
${V(lo)\leftarrow 30+1\bigg(43-(30)\bigg)=43}$  	(Elapsed Time = 0)  

${V(rc)\leftarrow 35+1\bigg(38-(35)\bigg)=38}$ 		(Elapsed Time = 5)    

${V(eh) \leftarrow 15+1\bigg(23-(15)\bigg)=23}$			(Elapsed Time = 20)  
${...}$  
The sum of each ${V}$ with the relative elapsed time makes always 43.

While TD updates every state to the estimate of the successor state.:   
(we are considering ${\gamma=1}$ as well)

${V(lo)\leftarrow V(lo)+1 \bigg(r_{t+1}+\gamma V(rc)-V(lo)\bigg)}$

${V(lo)\leftarrow 30 + 1\bigg(5+1\cdot35-(30)\bigg)=40}$		(Elapsed Time = 0)  

${V(rc) \leftarrow 35 + 1\bigg( 15+1\cdot 15-(35)\bigg)=30}$		(Elapsed Time = 5)  

${V(eh)\leftarrow 15 + \bigg(10+1\cdot 10-(15)\bigg)=20}$		(Elapsed Time = 20)

The sum of each ${V(S_t)}$ with the relative elapsed time makes always the total predicted time in state ${S_{t+1}}$.

**Complete/Incomplete - Episodic/Continuing environments**

- TD can learn *before* knowing the final outcome, it  learns online, step by step. (online means that it learns on the go, doesn't wait the episode to be over)
- MC *must* wait until the end of the episode before the return is known.

But what if there is no episode? what if there is no final outcome?

- TD can learn from incomplete sequences
- MC can only learn from complete sequences
- TD works in continuing (non-terminating) environments
- MC only works for episodic (terminating) environments

**Bias & Variance differences**

But the major difference between this two algorithms translates into a Bias/Variance Trade-Off:  
Monte-Carlo's estimate of the value function ${v_\pi(s_t)}$ is *unbiased*:   
${G_t= R_{t+1}+\gamma R_{t+2} +...+ \gamma^{T-1}R_{T}}$  
${V}$ is an actual sample of the expected return, we are not introducing any bias.  

Temporal-Difference's estimate of the value function ${v_\pi(s_t)}$ is *biased*   

${TD}$ - ${target}$ ${= R_{t+1}+\gamma V(S_{t+1})}$

because ${V(S_{t+1})}$ is just an estimate of the value function of ${S_{t+1}}$.  
That said, TD is biased but it's much more lower variance in the return wrt MC, why?

- MC return ${G_t}$ depends on many random actions, transitions, rewards
- TD-target depends on one random action, transition, reward



Moreover TD is more efficient, since it is much more lower variance!  
Let's see an example of this:

*Random Walk Example*:

![](images\rw1.png)

consider the following MDP where you get zero reward for any transition but moving right from E. the episode ends only in two situations:  
When you reach the leftmost square, or the rightmost one. let's adopt a policy that makes you move to the left and to the right with 50% probability. Once we run any algorithm we should obtain something like this:
$$
V(A)=\frac{1}{6} \\
V(B)=\frac{2}{6} \\
V(C)=\frac{3}{6} \\
V(D)=\frac{4}{6} \\
V(E)=\frac{5}{6}
$$
The chart above shows you how the value estimates of each state change based on how many episodes we run. (this chart was plotted applying ${TD(0)}$).  
We can see that by running roughly 100 episode we get a good approximation of the true value function.

Now let's compare TD and MC for this example:

![](images\rw2.png)

In the ${y}$ axis we have the total error over all states, which means "how much the function we estimated is wrong wrt to the true one".  
We can notice that we plot lines for both MC an TD and for different step sizes ${\alpha}$.  
We notice that TD does better than MC, doesn't matter how many episodes we run.  
We even notice that MC is noisier (more variance).

**Markov / non-Markov Environments**  

Last but non least: which of them takes advantage of the Markov property? (*the future is independent of the past given the present*). the short answer is: TD does, MC does not.

The long answer is: follow the example (we are almost done).

So far we have dealt with convergence given that we run a huge number of episodes:

${MC}$ and ${TD}$ converge: ${V(s) \to v_\pi(s)}$ as experience ${\to \infty}$.

But what if we can't run a huge number of episodes? what if we are provided with 8 episodes and we should learn only from them? In this cases we would get a *batch* solution: we would repeatedly sample from our ${k}$ episodes to learn everything we can.

To get an intuition consider the following problem:

<img src="images/ab1.png"  style="zoom:50%">

Each line corresponds to an episode, and each episode correspond to a sequence of states and rewards.  
if we run Monte-Carlo we would get ${V(A)=0}$ and ${V(B)=\frac{6}{8}}$. Why? In every episode we saw ${A}$ we got a total reward of ${0}$ (we visit it only in the first episode), so it makes sense that, even if we run 1000 times the first episode we would still get ${V(A)=0}$.  
This is a completely legit value for ${A}$.  
${V(B)=\frac{6}{8}}$ because, if we run ${k}$ times our 8 episodes, we get that ${6k}$ times out of the ${8k}$ we encountered ${B}$, we got a total reward of 1.

Now let's run ${TD(0)}$:  
${V(A)=\frac{6}{8}}$ and ${V(B)=\frac{6}{8}}$.  
Why? the return of ${A}$ has changed? you could see it like this:  
In ${TD}$ we have that
$$
V(A)\leftarrow V(A)+\alpha \bigg(r_{t+1}+\gamma V(B)-V(A)\bigg) \\
  V(A)\leftarrow V(A)+\alpha \bigg(0+\gamma V(B)-V(A)\bigg)
$$
the expected return of ${B}$ will become ${\frac{6}{8}}$ so ${V(A)}$ gets updated toward that value.  
${TD}$ is able to capture, *even in finite experience situations*, the Markov property.  
${TD}$, *even in finite experience situations*, is able to build an MDP model, ${MC}$ can't! 

This is what ${TD}$ builds:  
<img src="images\ab2.png" style="zoom:50%">

if you want to have a look to the math behind all of this here you are:  
<img src="images/ab3.png"  style="zoom:50%">



MC would still capture partially the Markov property if it was given a higher number of episodes to deal with, but, since he can deal with just those 8 ones (and those 8 ones are not very representative of the model) it can't capture the structure of the MDP. 



**Function Approximation**

What is function approximation? well, in most cases we have tons and tons of states, and it's not very efficient to compute the value function of each single state, so with function approximation we mean that we compute an approximate value for some states' value function.

it's bad to use function approximation in TD because, once you update the value of a certain ${s}$, you need to update the linear equation that approximates the behavior of each value of ${V}$ wrt to the states ${s' \neq s}$.

Ok, we are done, what follows is a concise summary of the differences between the two algorithms:

**Concise Summary**

if you want to briefly answer the question you could probably just say the following:

- MC
  - high variance, zero bias
  - good convergence properties
  - converges even with function approximation
  - Not very sensitive to initial value
  - Very simple to understand and use
  - learns only from complete sequences
  - works only for episodic environments
  - Usually more effective in non-Markov environments
- TD
  - low variance, some bias
  - ${TD(0)}$ converges to ${V_\pi(s)}$
  - doesn't always converge with function appriximation
  - more sensitive to the initial value
  - learns even from incomplete sequences
  - works for both episodic and continuing environments
  - Usually more effective in Markov environments



<img src="images\mcbackup.png" style="zoom:50%">

<img src="images/tdbackup.png" style="zoom:50%" >





(  Sources: David Silver's Slides;  [David Silver's RL Lecture 04](https://www.youtube.com/watch?v=PnHCvfgC_ZA&t=1702s)  ) 



<div style="page-break-after: always;"></div> 

### On-Policy vs Off-Policy

***Describe the diﬀerence between on-policy and oﬀ-policy reinforcement learning techniques. Make an example of an on-policy algorithm and an example of an oﬀ-policy algorithm.***   
*(William Bonvini)*  
Let's first revise some concepts:  

- a  **probability distribution** is a mathematical function that provides the probabilities of occurrence of different possible outcomes in an experiment
- A **policy** ${\pi}$ is a distribution, a mapping, at any given point in time, from states to probabilities of selecting each possible action. It decides which action the agents selects, defining its behavior.    
  A more concise definition is the following:  
  A policy ${\pi}$ is a distribution over actions given the state:  
  ${\pi(a|s)= \mathbb{P} [a|s]}$   

The difference between Off and On policy techniques is the following:  
**On-policy learning** "learns on the job". The policy that I'm following is the policy that I'm learning about.   
It learns about policy ${\pi}$ from experience sampled from ${\pi}$ itself.   
An example of on-policy technique is  *SARSA Algorithm*.   

SARSA update function (on-policy):  
${Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha (\color{red} R_{t+1}+\gamma Q(S_{t+1},A_{t+1}) \color{black} -Q(S_t,A_t))}$  

**Off-policy learning** "learns over someone's shoulders". It learns about the **target policy** ${\pi(a|s)}$ while following a **behavior policy** ${\bar{\pi}(a|s)}$.  
Off policies learn from observing humans or other agents.  
They re-use experience generated from old policies ${\pi_1,\pi_2,...,\pi_{t-1}}$ in order to generate the new target policy ${\pi}$.

the best known example of why off-policy learning is used is the one regarding the exploration-exploitation tradeoff. We can follow and exploratory policy  and at the same time learn about the optimal policy.  
Another interesting use of off-policy learning is wanting to learn about multiple policies while following one: there might be many different behaviors we want to figure out.

An example of off-policy technique is *Q-Learning*.  

*Q-Learning* update function (off-policy)  :  
${Q(S_t,A_t)\leftarrow Q(S_t,A_t)+ \alpha( \color{red} R_{t+1}+\gamma \max_{a' \in A}  Q(S_{t+1},a') \color{black} - Q(S_t,A_t))}$  

(Sources: PMDS Notes and [Model Free Algorithms](https://medium.com/deep-math-machine-learning-ai/ch-12-1-model-free-reinforcement-learning-algorithms-monte-carlo-sarsa-q-learning-65267cb8d1b4) )



<div style="page-break-after: always;"></div> 

### Value Iteration

***Describe the value iteration algorithm. Does the algorithm always return the optimal policy?***   
*(William Bonvini)*

If you are looking for a concise answer go to the end.

Value iteration is the most popular dynamic programming algorithm applied to RL. Since we are talking about Dynamic Programming, It's Model Based.

Value iteration is based on the Principle of Optimality:

If the first action I take is optimal and then I follow an optimal policy from whichever state I end up, the overall behavior is optimal.

***Principle of Optimality***  
A policy ${\pi(a|s)}$ achieves the optimal value from state ${s}$ , ${v_\pi (s)=v_* (s)}$, if and only if, for any state ${s'}$ reachable from ${s}$,
${\pi}$ achieves the optimal value from state ${s'}$, ${v_\pi (s')=v_*(s')}$.

Ok, how to exploit this?  
If we know the solution to the subproblems ${v_* (s')}$, we can find ${v_* (s)}$ just by applying a one-step lookahead:
$$
v_* (s) \leftarrow \max _{a \in A}\bigg\{{R_s^a+\gamma \sum_{s' \in S}P_{ss'}^a v_*(s')\bigg\} }
$$
The idea of value iteration is to apply these update iteratively: we plug in to the right member of the equation the current value function (so, it's not optimal at first!), obtain a new value function, plug such new value function to the right member, obtain a new value function, and so on until we find the optimal value function.

Intuition: start with the final rewards and work backwards.  
Shortest Path Example:    
<img src="images/shortestpath.png" style="zoom:60%"/> 

This problem consists in finding the optimal value function for each cell. the goal of the game is to reach the terminal state (top-left corner), the possible actions are move left, up,right, down. Each actions' reward is ${-1}$. 

With value iteration we are able to find the optimal value function just by iterating on the Bellman's Optimality Equation.

We initialize all the values to ${0}$.

In ${ V_2}$  we have that 

${[0,0]}$:
$$
V_* ([0,0]) \leftarrow \max _{a \in A}\bigg\{R_s^a+\gamma \sum_{s' \in S}P_{ss'}^a V_*(s') \bigg\}
$$

$$
V_* ([0,0]) \leftarrow 0+1 \sum_{s' \in S}P_{ss'}^a V_*(s')
$$

$$
V_* ([0,0]) \leftarrow 0+0
$$

${[0,1]}$:

$$
  V_* ([0,1]) \leftarrow \max _{a \in A}\bigg\{R_s^a+\gamma \sum_{s' \in S}P_{ss'}^a V_*(s') \bigg\} 
  \\
  V_* ([0,1]) \leftarrow -1+1 \bigg(1\cdot0+0\cdot(-1)+0\cdot(-1)+0\cdot(-1)\bigg) 
  \\
  V_*([0,1])\leftarrow-1
$$


  ${[2,2]}$:
$$
  V_*([2,2])\leftarrow -1+1\cdot\bigg(1\cdot(-1)+0\cdot(-1)+0\cdot(-1)+0\cdot(-1)\bigg)
  \\
  V_*([2,2])\leftarrow-2
$$
  (In ${[2,2]}$ I chose randomly to perform one of the actions, they all give the same result. In ${[0,1]}$ I considered moving west since it's the most convenient choice).

  I did a couple of examples for ${V_2}$, hopefully you can get the sense of the algorithm.

  Let's take stock of the situation: value iteration is a method for solving MDPs, how do we do it? by applying iteratively the Bellman Optimality Equation, doing so we find the optimal value function.
$$
  v_1 \to v_2 \to v_3 \to... \to v_*
$$
  *What are the differences between policy iteration and value iteration?*

  - in V.I. we are not building a policy at each step, we are working directly in value space. in P.I. there is an alternation between value and policy space.
  - Intermediate value functions of V.I. may not correspond to any policy, while intermediate value functions of P.I. do. What does this mean? It means that in VI, during the iteration, we could get an intermediate ${v}$,  which does not correspond to any ${v_\pi}$ for any ${\pi}$.
  - We can say that Value iteration is equivalent to do Modified Policy Iteration with ${k=1}$. 

  *One last image to sum up:*  
  <img src="images/value_iteration.png" style="zoom:70%"/>

  *In a nutshell:* 

  1. Take the current value function and plug it in the leaves. 
  2. for each state (consider it the root of the tree):  
     Such root
     1.  Does a lookahead
     2. Maximizes over all the things it might do
     3. Takes an expectation over all the things the environment might do. 
     4. Backs it up to get its new value.
  3. Back to step 1 until you find the optimal policy

  Important: Value Iteration assures to find the optimal value function, consequently it assures to find the optimal policy. 

  How come?

  Define the max-norm: ${||V||_\infty}=\max_s|V(s)|$

  *Theorem*:

  Value Iteration converges to the optimal state-value function ${\lim_{k\to\infty}V_k=V^*}$

  *Proof*:
$$
  ||V_{k+1}-V^*||_\infty =||T^*V_k-T^*V^*||_\infty\le \gamma||V_k-V^*||_\infty \\
  \le  \ ... \ \le \gamma^{k+1}||V_0-V^*||_\infty \to \infty
$$


  *Theorem*  
$$
||V_{i+1}-V_i||_\infty < \epsilon \implies ||V_{i+1}-V^*||_\infty < \frac{2\epsilon\gamma}{1-\gamma}
$$

***Concise Answer***  

Value iteration is the most popular dynamic programming algorithm applied to RL. Since we are talking about Dynamic Programming, It's Model Based.

The problem of ﬁnding the optimal policy ${\pi_*}$ is solved by iteratively applying the Bellman Optimality equation, without any explicit policy.  
In fact, intermediate value functions may not correspond to any policy.  

  *Bellman's Optimality Equation*:
$$
  v_* (s) \leftarrow \max _{a \in A}\bigg\{{R_s^a+\gamma \sum_{s' \in S}P_{ss'}^a v_*(s')\bigg\} }
$$
  Value Iteration always return the optimal policy, as shown by the following theorem.

  Define the max-norm: ${||V||_\infty}=\max_s|V(s)|$

  *Theorem*:

  Value Iteration converges to the optimal state-value function ${\lim_{k\to\infty}V_k=V^*}$

  *Proof*:
$$
  ||V_{k+1}-V^*||_\infty =||T^*V_k-T^*V^*||_\infty\le  \\ \gamma||V_k-V^*||_\infty 
  \le  \ ... \ \le \gamma^{k+1}||V_0-V^*||_\infty \to \infty
$$
  *Theorem*  
$$
  ||V_{i+1}-V_i||_\infty < \epsilon \implies ||V_{i+1}-V^*||_\infty < \frac{2\epsilon\gamma}{1-\gamma}
$$


  ( Sources: PMDS Notes - [Deep Mind Dynamic Programming](https://www.youtube.com/watch?v=Nd1-UUMVfz4&t=142s) )

  

<div style="page-break-after: always;"></div> 



### Eligibility Traces

***Describe what are eligibility traces and how they are used in the TD(λ) algorithm. Explain what happens when λ = 0 and when λ = 1.***   
*(William Bonvini)*  
If you are interested in the concise answer read from the section *Eligibility Traces* on.  
First of all let's give some contest. 
***TD(${ \lambda}$)***  
It's a Temporal Difference algorithm that consists in averaging over ${n}$-steps returns over different ${n}$. This is done in a weighted way, using the value ${ \lambda}$.

***Forward-View ${TD(\lambda) }$***

Forward-View is an offline algorithm ${\to}$ needs to run the entire episode.  
$$
V(s_t) \leftarrow V(s_t)+\alpha (v_t^\lambda -V(s_t))  
\\
v_t^\lambda=(1-\lambda)\sum_{n=1}^\infty \lambda^{n-1}v_t^{(n)}
$$
where
$$
v_t^{(\color{red}n\color{black})}=r_{t+1}+\gamma r_{t+2}+...+\gamma^{\color{red}n\color{black}-1}r_{t+\color{red}n\color{black}} +\gamma^{\color{red}n\color{black}}V(s_{t+\color{red}n}\color{black})
$$
  <img src="images/fwtd1.png" zoom=70%>



- Updates the value function towards the ${ \lambda}$ return 
- Forward-View looks into the future to compute the target ${v_t^\lambda}$
- Like MC, can only be compute from complete sequences

***Backward-View ${TD(\lambda)}$***

This is an online algorithm, so there is no need for the episodes to be complete.  

- It keeps an eligibility trace for every state ${s}$ (we'll explain them soon)

- It updates value ${V(s)}$ for every state ${s}$ in proportion to the ${TD}$-error ${\delta_t}$ (one-step error!) and the eligibility trace ${e_t(s)}$  
  
  $$
  \\
  V(s) \leftarrow V(s)+\alpha \delta_te_t(s)
  \\
  \ 
  \\
  \delta_t=R_{t+1}+ \gamma V(S_{t+1})-V(S_t)
  $$
  

<img src="images/et2.png" zoom=70>

Intuition: imagine to be a rat, and you notice that usually, when a bell rings three times and the light of the room gets turned on, you get electrocuted.  

It has many advantages:

- It updates the weight vector on every step of an episode rather than only at the end, and thus its estimates may be better sooner.
-  Its computations are equally distributed in time rather than all at the end of the episode. 
-  It can be applied to continuing or incomplete episodes problems rather than just complete episodes problems.

How to do it?

Thanks to Eligibility Traces. 

***Eligibility Traces***

Eligibility Traces are used in Backward-View ${TD(\lambda)}$ algorithms, both for prediction and control.   
An *Eligibility Trace* is a short-term memory vector${\mathbf{e}_t \in \R^d}$ .  
When a state is visited, the corresponding component of ${\mathbf{e}_t}$ is bumped up and than begins to fade away. So, for a given state ${s}$ , the update of ${V(s)}$ will occur in proportion to the ${TD}$-error ${\delta_t}$ and to the eligibility trace ${e_t(s)}$.  
The *trace-decay* parameter ${\lambda}$ determines the rate at which the trace falls.  

$$
e_0(s)=0 
\\
e_t(s)=\gamma\lambda e_{t-1}(s)+ \mathbf{1}(s=s_t)
\\
\ 
\\
\delta_t=R_{t+1}+ \gamma V(S_{t+1})-V(S_t)
\\
\
\\
\color{blue} 
V(s) \leftarrow V(s)+\alpha \delta_te_t(s)
$$

The pseudo-code for ${e_t(s)}$ says that ${e(t)}$ decays of a factor ${\gamma \lambda}$ when it's not visited, but, when it's visited, it does decay of that same factor and at the same time gets incremented by ${1}$.

The eligibility trace keeps track of which states have contributed, positively or negatively, to recent state valuations, by combining both *frequency heuristics* (assign credit to the most frequent states) and *recency heuristics* (assign credit to the most frequent states).  
<img src="images/et1.png" style="zoom:70%"/>

***What happens for ${\lambda=0}$ and ${\lambda=1}$?***

*${\lambda=0}$*:  
When ${ \lambda=0}$, only the current state gets updated.  
$$
e_t(s\neq s_t)= \gamma \cdot0\cdot e_{t-1}(s)+0=0 
\\
e_t(s=s_t)=\gamma\cdot 0 \cdot e_{t+1}(s)+1=1
\\
V(s) \leftarrow V(s)+\alpha \delta_te_t{(s)} \ \ \ \ \ \ \ \ \ \ \ \  \ \forall s
$$
Which is exactly equivalent to ${TD(0)}$'s  update:  
$$
V(s) \leftarrow V(s)+\alpha \delta_t \ \ \  \ \ \ (s=s_t)​
$$


*${\lambda=1}$*:    

It's roughly equivalent to Every-Visit Monte-Carlo, we'll see why.  
Consider an episode where any state ${s}$ is visited at time-step ${k}$.  
${TD(1)}$ eligibility trace for ${s}$ gets discounted from the moment it's visited ${(t=k)}$:
$$
e_t(s)=\gamma e_{t-1}(s)+\mathbf{1}(s_t=s) 
\\
= \begin{cases}
0 \ \ \ \ \ \  \ \ \ if \ \  \ t<k
\\
1 \ \ \ \ \ \ \ \  \  if \ \ \ t=k
\\
\gamma^{t-k} \ \ \ \  if \ \ \ t>k

\end{cases}
$$
The second and third case can be merged:  ${e_t(s)=\gamma^{t-k} \ \ \ \ if \ \ \ t \ge k}$, I wrote it like that just because I think it's more intuitive. 

So ${TD(1)}$ updates accumulate errors *online*
$$
\sum_{t=1}^{T-1}\alpha \delta_t e_t(s)=\alpha \sum_{t=k}^{T-1}\gamma^{t-k}\delta_t=\alpha (G_k-V(s_k))
$$
How did we get to the last equation? let's see:  
By the end of the episode it accumulates the total error:
$$
\delta_k+\gamma\delta_{k+1}+\gamma^2\delta_{k+2}+...+\gamma^{T-1-k}\delta_{T-1}
$$

Just by rearranging such total error  we understand that, for  ${\lambda=1}$, the sum of TD errors telescopes into ${MC}$ error.
$$
\delta_k+\gamma\delta_{k+1}+\gamma^2\delta_{k+2}+...+\gamma^{T-1-k}\delta_{T-1} \\
=R_{t+1}+\gamma V(s_{t+1})-V(s_t) 
\\
+\gamma R_{t+2}+\gamma^2 V(s_{t+2})-\gamma V(s_{t+1})
\\
+ \gamma ^2 R_{t+3} + \gamma^3 V(s_{t+3})-\gamma^2 V(s_{t+2})
\\
+
\\
.
\\
.
\\
.
\\
\gamma^{T-1-t}R_T
+\gamma^{T-t}V(s_t)-\gamma^{T-1-t}V(s_{t-1})
\\
=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+...+\gamma^{T-1-t}R_t-V(s_t)
\\
=G_t-V(s_t)
$$


So ${TD(1)}$ is roughly equivalent to every-visit Monte-Carlo. Why roughly? because the total error is accumulated online, step-by-step.  
If the value function was updated offline, at the end of the episode, then the total update would have been exactly the same as MC.  

Important observation (not needed for answering the question though):  
*The sum of offline updates is identical for forward-view and backward-view ${TD(\lambda)}$*  
$$
\sum_{t=1}^T\alpha \delta_te_t(s)=\sum_{t=1}^{T}\alpha \bigg(v_t^\lambda -V(s_t)\bigg)\mathbf{1} (s_t=s)
$$
(Sources: David Silver's Slides on Prediction - Restelli's Slides on Finite MDPs)





<div style="page-break-after: always;"></div> 

### Value Iteration vs Policy Iteration

***Describe and compare Value Iteration and Policy Iteration algorithms.***   
*(WIlliam Bonvini)*

Value iteration and Policy Iteration are two algorithms used to do control in Model-Based environments.

Value Iteration can be considered a particular case of Modified Policy Iteration.

***Policy Iteration***  
I'm talking about PI for Model Based Problems, in fact the improvement step is completely deterministic.

It's divided in two steps:

- Policy Evaluation 
- Policy Improvement

*Policy Evaluation* consists in evaluating a certain policy ${\pi}$ by iteratively applying the *Bellman Expectation Equation*
$$
V_{k+1}(s)\leftarrow \sum_{a\in A}\pi (a|s)\Bigg[R(s,a)+\gamma \sum_{s'\in S}P(s'|s,a)V_k(s') \Bigg]
$$
It means that the value function at the iteration ${k+1}$ is given by the immediate reward obtained following policy ${\pi}$ plus the discounted average total reward obtained from the successor state ${s'}$.  
The evaluation is completed (the value function converges to the true value function for that policy) when ${k \to \infty}$. 

*Policy Improvement* consists in coming up with a better policy ${\pi'}$ starting from a policy ${ \pi}$. This is achieved by acting greedily wrt to the value function evaluated in the first step of policy iteration.

${\pi'(s)=\arg \max_{a \in A}}{Q^\pi(s,a)}$

By repeating evaluation and improvement we are certain of obtaining in the end the optimal policy ${\pi^*}$

***Value Iteration***  
*Value Iteration* consists in applying iteratively the *Bellman Optimality Equation*
$$
V^*(s)=\max_{a \in A}{\bigg\{R(s,a)+\gamma \sum_{s' \in S}P(s'|s,a)V^*(s') \bigg\}}
$$
until the actual optimal value function is found.

The optimal value function is found when the old value function ${V_k}$ and the new one ${V_{k+1}}$ differ less than a small number ${\epsilon}$.  
Value iteration  is based on the principle of Optimality:  
A policy ${\pi(a|s)}$ achieves the optimal value from state ${s}$, ${v_\pi (s)=v_* (s)}$, if and only if, for any state ${s'}$ reachable from ${s}$,  
${\pi}$ achieves the optimal value from state ${s'}$, ${v_\pi (s')=v_*(s')}$.

This algorithm assures convergence to the optimal value function, and consequently to the optimal policy.

***Differences***

- in Value Iteration we are not building a policy at each step, we are working directly in value space. in Policy Iteration there is an alternation between value and policy space.
- Intermediate value functions of Value Iteration may not correspond to any policy, while intermediate value functions of Policy Iteration do. What does this mean? It means that in VI, during the iteration, we could get an intermediate ${v}$,  which does not correspond to any ${v_\pi}$ for any ${\pi}$.
- We can say that Value iteration is equivalent to do Modified Policy Iteration with ${k=1}$.  Modified Policy Iteration is just Policy Iteration, but we don't wait the value function to converge to the true one, we stop the evaluation at ${k=const}$. 

(Sources: this document)

<div style="page-break-after: always;"></div> 

### Methods to compute V function in DMDP

***Describe which methods can be used to compute the value function $V^{\pi}$ of a policy $\pi$ in a discounted Markov Decision Process.***   
*(William Bonvini)*

The question is asking what are the methods o do prediction in Reinforcement Learning.  
We have to distinguished between Model-Based environments and Model-Free environments.

***Model-Based***  
Model-Based means that you are given all the dynamics of the system: the transition matrix, and the rewards for each state action pair.

The methods to do prediction in model-based problems are dynamic programming algorithms, and these are:

- Policy Evaluation
- Modified Policy Evaluation

*<u>Policy Evaluation</u>*

It consists in iteratively applying the *Bellman Expectation Equation* to a value function following policy ${\pi}$
$$
V_{k+1}(s)\leftarrow \sum_{a\in A}\pi (a|s)\Bigg[R(s,a)+\gamma \sum_{s'\in S}P(s'|s,a)V_k(s') \Bigg]
$$
until ${k \to \infty}$.

<u>*Modified Policy Evaluation*</u>

It's a modified version of Policy Evaluation which returns an approximation of the value function ${V^\pi}$ for of the policy ${\pi}$.

It differs from Policy Evaluation just by the fact that we stop the evaluation after a certain number of steps ${\to}$ we don't wait for the full convergence.

***Model-Free***  
Model-Free means that you are not given the transition matrix and the rewards for each state action pair.

There are mainly two algorithms to do prediction in Model-Free problems:

- Monte Carlo
- Temporal Difference

<u>*Monte Carlo*</u>

The way Monte Carlo estimates the state-value function for a given policy from experience is simply by averaging the returns observed after visits to that state. As more returns are observed, the average should converge to the expected value. 

So Monte Carlo policy evaluations uses empirical mean return instead of the expected return and it can be computed with two different approaches:

- First-Visit MC  
  Average returns only for the first time ${s}$ is visited (**unbiased** estimator) in an episode
- Every-Visit MC  
  Average returns for every time ${s}$ is visited (**biased** but **consistent** estimator)

Once an episode is over, we proceed with updating the two values

- ${V(s)}$ : the state-value function
- ${N(s)}$: the total number of times ${s}$ has been visited

for each state ${s}$ that has been visited during the last episode.

Stationary Case:

${N(s_t)\leftarrow N(s_t)+1}$

${V(s_t)\leftarrow V(s_t)+ \frac{1}{N(s_t)}(v_t-V(s_t) )}$

Non-Stationary Case:
$$
V(s_t)\leftarrow V(s_t)+ \alpha\bigg(v_t-V(s_t)\bigg)
$$
with ${{v_t=G_t=R_{t+1}+\gamma R_{t+2}+...+\gamma^{T-1}R_T}}$

<u>*Temporal Difference*</u>  
Temporal Difference prediction consists in updating our value function towards the estimated return after one step:  
Specifically, the estimate consists in two parts: the immediate reward ${r_{t+1}}$ plus the discounted value of the next step ${\gamma V(S_{t+1})}$. 
$$
V(s_t)\leftarrow V(s_t)+\alpha \bigg(r_{t+1}+\gamma V(s_{t+1})-V(s_t)\bigg)
$$
The one above is the simplest temporal-difference learning algorithm, called ${TD(0)}$, which means Temporal Difference with ${\lambda=0}$   
The general algorithm for Temporal Difference depends on such value ${\lambda}$ and  ${0\le\lambda \le 1}$.

in ${TD(0)}$ we estimate the new return by doing just a one-step lookahead, but we could even do a two-steps lookahead or in general a ${k}$-step lookahead.

if ${\lambda = 1}$ we obtain Monte Carlo learning.

(Sources: this document)

<div style="page-break-after: always;"></div> 

### Policy Iteration

***Describe the policy iteration technique for control problems on Markov Decision Processes***  
*(William Bonvini*)   
If you want a concise answer just go to the end.  
premise: what is a control problem? is the task of finding the optimal value function, which translates into finding the optimal policy.  
Policy Iteration is a dynamic programming policy optimization technique that can be decoupled in two phases:

- Policy Evaluation
- Policy Improvement

***Policy Evaluation*** 
Consists in computing the state-value function ${V^\pi}$ for a given policy ${\pi}$.   
It is done by iteratively applying the Bellman expectation backup.  
$$
V_{k+1}(s)\leftarrow \sum_{a\in A}\pi (a|s)\Bigg[R(s,a)+\gamma \sum_{s'\in S}P(s'|s,a)V_k(s') \Bigg]
$$
Applying a backup operation to each state is called **sweep**.  

What we will actually do is to start off with an arbitrary initial value function, let's call it ${V_1}$. So, this ${V_1}$ tells us what is the value of all states in the MDP (the canonical case is to start with ${V(\forall s)=0}$).  
Afterwards we are going to plug in one step of our Bellman equation: we do a one step lookahead and doing so we figure out a new value function that we call ${V_2}$.  
If we iterate this process many times we end up with the true value function ${V_\pi}$.  
$$
V_1\to V_2 \to... \to V_\pi
$$
Ok, but how do we actually pass from ${V_k}$ to ${V_{k+1}}$?  

We do it by using *synchronous backups*: 

- at each iteration ${k+1}$
- for all states ${s \in S}$
- update ${V_{k+1}(s)}$ from ${V_k(s')}$
- where ${s'}$ is a successor state of ${s}$



So let's understand exactly how to do such update:  
<img src="images/policy_evaluation1.png" style="zoom:50%"/>



The picture above shows our one-step lookahead tree and the *Bellman Expectation Equation*, what do they tell us?  
They tell us that the value of the root is given by a one-step lookahead: we consider all the actions we might take and all the states we might go in.  
We know the value of each state we might go in, so we back them all up (summing them together weighted by the probabilities involved) and we get the new value of the root.  
Why do we know the value of such successor states? Because in our example they are all leaves, obviously if they weren't leaves we should have iterated the process over such states.

So, imagine just to start always from the root of your tree, go down the tree until you find the leaves, and then come back up by doing what I just described.  
Ok but by doing so we just "changed our skin" once, we went from ${V_1}$ to ${V_2}$ for example. Well, in order to go from ${V_2}$ to ${V_3}$ and in the end find ${V_\pi}$ we just iterate this process of updating ${V}$.

So, let's consider we just computed ${V_2}$, now we simply plug ${V_2}$ into the leaves, back them up, and get ${V_3}$ at the root.

This process is guaranteed to converge to the true value function ${V_\pi}$ (pay attention, it's not that it converges to the optimal value function, it might, but what I'm saying is that it converges to the value function of the policy you are considering!).

Let's make and example:   
<img src="images/sgw1.png" style="zoom:70%"/>

This is called the Small Gridworld example.  
There are two terminal states (the top left and bottom right squares). there are 14 non-terminal states. The reward is -1 until the terminal state is reached. 
For now consider only the left column of the following images:   
<img src="images/sgw2.PNG"   style="zoom:70%"/>



<img src="images/sgw3.png" style="zoom:70%"/>



Our initial estimate for each state is a trivial 0.  
Now we apply one step of iterative policy evaluation and we'll end up with a new value for each of these states.

For example, let's look at the update from ${k=0}$ to ${k=1}$.  We updated the value of almost each state to -1.0 because whichever step I take is going to give me -1 unless I'm already at the goal (in fact the two terminal states keep on having 0).  
Now let's consider the grid in ${k=2}$ and in particular third cell in the second row.  
In the grid with ${k=1}$ it contained a ${-1.0}$, while now it contains ${-2.0}$. This is due to the fact that wherever I go from that cell (left,up,right,down), I have an immediate cost of ${-1}$ and I find myself in a state with value ${-1}$, in fact all the states around such cell contain a ${-1}$. So the value gets updated to the average of all of these returns, and it's ${-2}$.

Always considering the transition from ${k=1}$ to ${k=2}$, look at the second cell of the first row. It gets updated to ${1.75}$ (in the image you'll see ${1.7}$ just because it was truncated). Why ${1.75}$? because if I go up, I actually stick where I am, so I pay ${-1}$ for the action and ${-1}$ because I end up in a state whose value is ${-1}$. Same thing for moving on the right and on the bottom. If instead I move to the left, I get rewarded ${-1}$ for the action but I end up in a state whose value is ${0}$, so let's average: ${\frac{(-1-1)+(-1-1)+(-1-1)+(-1)}{4}=-1.75}$.

Ok, enough examples.

If we keep on iterating, we end up with the true value function ${(k=\infty)}$ and this value function tells us how many steps on average I'd need to take from each state in order to reach a terminal state following a uniform policy (0.25 possibility of going in each direction).  
Ok, policy evaluation is done.

***Policy Improvement***

Now let's talk about the column on the right:  
While evaluating we can improve our policy by acting greedily wrt to the value function!  
Already in the second grid ${(k=1)}$ we can see that if we find ourselves in the second cell of the first row, the only reasonable thing to do is to go left, because the value of all the other neighbors is worse.

So one interesting thing about Policy Evaluation is that, by evaluating our policy, we could infer a new policy!  

And there is more: after a few iterations of policy evaluation (in our example, when ${k=3}$) , even if the true value function has not been determined yet, the new policy has usually already converged, because it depends on the shape of ${V}$, not on its absolute value. So, instead of using the closed form solution, which is expensive, applying the iteration for a few steps allows to have a bad approximation of the value function, but a good estimation of the policy (this concept is the key for the *modified policy iteration* algorithm, we'll see it soon).   

Very important: In the Small Gridworld example we are lucky because the first value function we evaluate (approximate evaluation for ${k=3}$ or full evaluation for ${k=\infty}$) gives us right away the optimal policy, but this is not always the case!  

*Policy improvement* consists in changing the policy according to the newly estimated values.  
For a given state ${s}$, would it be better to do an action  ${a \neq \pi(s)}$?  
We can improve the policy by acting greedily:  
$$
\pi'(s)=arg\  \max_{a \in A}Q^\pi(s,a)
$$
This improves the value from any state ${s}$ over one step  
$$
Q^\pi(s,\pi'(s))=\max_{a \in A}Q^\pi(s,a)\ge Q^\pi(s,\pi(s))=V^\pi(s)
$$
<u>*Theorem: Policy Improvement theorem*</u>

let ${\pi}$ and ${\pi '}$ be any pair of deterministic policies such that
$$
Q^\pi(s,\pi '(s))\ge V^\pi(s) \ \ \ , \ \ \ \forall s \in S
$$
Then, the policy ${\pi '}$ must be as good as, or better than ${\pi}$
$$
V^{\pi'}(s)\ge V^\pi (s) \ \ \ , \ \ \ \forall s \in S
$$
***Let's put it all together and optimize***

<u>*policy iteration*</u>

1. You start from any policy you want
2. fully evaluate that policy by iterating policy evaluation until ${k\to\infty}$ 
3. come up with a better policy by acting greedily wrt the old policy.
4. if you haven't found the optimal policy yet consider the new policy and go back to step 2, otherwise terminate.

Optimizations:  

*<u>modified policy iteration</u>* (the step 2 is explained at the beginning of the *Policy Improvement* section):

1. You start from any policy you want
2. approximately evaluate that policy by iterating policy evaluation until ${k\to const}$ 
3. come up with a better policy by acting greedily wrt the old policy.
4. if you haven't found the optimal policy yet consider the new policy and go back to step 2, otherwise terminate.

<u>*value iteration*</u>

1. You start from any policy you want
2. very approximately evaluate that policy by doing policy evaluation just once ${(k=1) }$ 
3. come up with a better policy by acting greedily wrt the old policy.
4. if you haven't found the optimal policy yet consider the new policy and go back to step 2, otherwise terminate.

**Concise Answer**

Policy Iteration is a dynamic programming policy optimization technique that can be decoupled in two phases:

- Policy Evaluation
- Policy Improvement

<u>*Policy Evaluation*</u>

An iterative policy evaluation is performed by applying the Bellman expectation backup an inﬁnite number of times. A full policy-evaluation backup
$$
V_{k+1}(s)\leftarrow \sum_{a\in A}\pi (a|s)\Bigg[R(s,a)+\gamma \sum_{s'\in S}P(s'|s,a)V_k(s') \Bigg]
$$
Applying a backup operation to each state is called *sweep*.  
We use synchronous backups: 

- at each iteration ${k+1}$
- for all states ${s \in S}$
- update ${V_{k+1}(s)}$ from ${V_k (s)}$1(s)  

After few iterations even if the true value function is not determined, the policy has usually already converged, because it depends on the shape of V, not on its absolute value. So, instead of using the closed form solution, which is expensive, applying the iteration for a few steps allows to have a bad approximation of the value function, but a good estimation of the policy. The usage of this approximated evaluation combined with policy improvement is called *modified policy evaluation*.

*<u>Policy improvement</u>*  
It consists in changing the policy according to the newly estimated values.  
For a given state ${s}$, would it be better to do an action  ${a \neq \pi(s)}$?  
We can improve the policy by acting greedily:  
$$
\pi'(s)=arg\  \max_{a \in A}Q^\pi(s,a)
$$
This improves the value from any state ${s}$ over one step  
$$
Q^\pi(s,\pi'(s))=\max_{a \in A}Q^\pi(s,a)\ge Q^\pi(s,\pi(s))=V^\pi(s)
$$
<u>*Theorem: Policy Improvement theorem*</u>

let ${\pi}$ and ${\pi '}$ be any pair of deterministic policies such that
$$
Q^\pi(s,\pi '(s))\ge V^\pi(s) \ \ \ , \ \ \ \forall s \in S
$$
Then, the policy ${\pi '}$ must be as good as, or better than ${\pi}$
$$
V^{\pi'}(s)\ge V^\pi (s) \ \ \ , \ \ \ \forall s \in S
$$
  ( Sources: PMDS Notes ${\leftarrow}$ there are some mistakes though - [Deep Mind Dynamic Programming](https://www.youtube.com/watch?v=Nd1-UUMVfz4&t=142s) )



<div style="page-break-after: always;"></div> 



### Prediction vs Control

***Describe the two problems tackled by Reinforcement Learning (RL): prediction and control. Describe how the Monte Carlo RL technique can be used to solve these two problems.***   
*(William Bonvini)*

- Prediction:  
  this type of task consists in predicting the expected total reward from any given state assuming the function ${\pi(a|s)}$ is given. 
- Control:  
  This type of task consists in finding the policy ${\pi(a|s)}$ that maximizes the expected total reward from any given state. In other words, some policy ${\pi}$ is given, and it finds the optimal policy ${\pi^*}$. 

Monte Carlo is Model Free RL technique.  
Model Free means that we don't have complete knowledge of the environment, for example, namely, we don't know the transition matrix ${P}$ and the rewards ${r}$ associated with each state-action pair.  

***Prediction In Monte Carlo***  
The way Monte Carlo estimates the state-value function for a given policy from experience is simply by averaging the returns observed after visits to that state. As more returns are observed, the average should converge to the expected value. 

So Monte Carlo policy evaluations uses empirical mean return instead of the expected return and it can be computed with two different approaches:

- First-Visit MC  
  Average returns only for the first time ${s}$ is visited (**unbiased** estimator) in an episode
- Every-Visit MC  
  Average returns for every time ${s}$ is visited (**biased** but **consistent** estimator)



Once an episode is over, we proceed with updating the two values

- ${V(s)}$ : the state-value function
- ${N(s)}$: the total number of times ${s}$ has been visited

for each state ${s}$ that has been visited during the last episode.

${N(s_t)\leftarrow N(s_t)+1}$

${V(s_t)\leftarrow V(s_t)+ \frac{1}{N(s_t)}(\color{red}v_t-V(s_t)\color{black} )}$

So what have we done? 

1. We incremented the number of occurrences ${N}$ of the visited state ${s_t}$.
2. We updated our estimate of ${V}$ a little bit in the direction of the error (red colored) between the old expected value of ${V}$ and the return we actually observed during this episode ${v_t}$.  
   Just think of it as saying that ${V}$ is exactly the same ${V}$ as before but incremented/decremented of a small quantity weighted by ${\frac{1}{N(s_t)}}$.  Or google *incremental mean* to get such equation starting from the generic equation of the *arithmetic mean*. 

***Control in Monte Carlo***

{DISCLAIMER: I derived MC Control step by step, if you want you can jump to *GLIE Monte Carlo Control* for the short answer}  
We use Policy Iteration for the control tasks.

It is composed of Policy Evaluation and Policy Improvement.

- Policy Evaluation:  
  Estimate ${v_\pi}$ 
- Policy Improvement  
  Generate ${\pi ' \ge \pi}$ 

 So let's try to plug in this iteration process in Monte Carlo.  
The first thing that comes up in our mind is to do the following:  

<img src="images/policy_it_1.png" style="zoom:55%"/>

Which means, estimate the ${V}$ values of the visited states ${s}$ using policy ${\pi}$ and then act greedily wrt to ${V}$ to improve our policy.   
(The diagram above shows the iteration process: while the arrow goes up we do evaluation, while it goes down we do improvement. These arrows become smaller and smaller because after some iterations ${V}$ and ${\pi}$ converge.)

There are actually two problems with this  approach: we can't use ${V}$ and we can't be so greedy.

***We can't use ${V}$***:

we want to be model free, but how can we me model free if we are using the state-value function ${V}$?   
We'd still need a model of the MDP to figure out how to act greedily wrt to ${V}$. We'd only know the values of the estimated return of each state, and we want to know what action is the best to be taken, so we would have to imagine in which state ${s'}$ we would go for each possible action. But wait, we don't know the transition matrix ${P}$!  
The solution consists in using ${Q}$. action-value functions ${(Q)}$ allow us to do control in a model free setting.  
sum up:  
Greedy Policy Improvement over ${V(s)}$ *requires model* of MDP (can't be used in model free setting)
$$
\pi'(s)=arg \max_{a \in A}\bigg(R(s,a)+P(s'|s,a)V(s')\bigg)
$$
Greedy Policy Improvement over ${Q(s,a)}$ is *model-free*
$$
\pi'(s)=arg \max_{a \in A}Q(s,a) So here we are, this is our new approach:
$$

So here we are, this is our new approach:

  <img src="images/policy_it_2.png" style="zoom:55%"/>

The same as before, but instead of ${V}$ we use ${Q}$

  

***We can't be so greedy***

If we act greedily all the time we don't do exploration, we just do exploitation: we just exploit what looks to be the best path, but maybe there is another path that for the first steps does not give us a good reward, but later on gives us a better reward than the path we are exploiting.  
The simplest idea for ensuring continual exploitation is to use an ${\epsilon}$-greedy approach:
$$
  \pi(s,a)=
  \begin{cases}
  \frac{\epsilon}{m}+1-\epsilon \ \ \ \ \ if \ a^*=arg \max_{a \in A}Q(s,a) \\
  \frac{\epsilon}{m} \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \  otherwise
  \end{cases}
$$
This means that all ${m}$ actions are tried with a non-zero probability. With probability ${\frac{\epsilon}{m}+1-\epsilon}$ we choose the greedy action, and with probability ${\frac{\epsilon}{m}}$ we choose any of the remaining ${m-1}$ actions. Pay attention: we could see it as choosing the greedy action with probability ${1-\epsilon}$ and with probability ${\frac{\epsilon}{m}}$ any of the actions, including the greedy one.  
This approach works well, and this is shown by the Policy Improvement Theorem:  
For any ${\epsilon}$–greedy policy ${\pi}$, the ${\epsilon}$–greedy policy ${\pi '}$ wrt ${Q^\pi}$ is an improvement.  
Therefore ${V^{\pi'}(s)\ge V^{\pi}(s)}$ 
(The demonstration of the theorem is easily findable on the internet or on page 56 of Restelli's slides of lecture 10 ).  
Moreover, when the number of episodes ${k\to \infty}$, the policy converges on a greedy policy

Here we are with our new approach:

  <img src="images/policy_it_3.png" style="zoom:40%"/>

We just got rid of the greedy improvement, and went for an ${\epsilon}$-greedy one. (yes, there is a mistake in the slide, in the diagram there should be written ${\pi=\epsilon}$-greedy${(Q)}$) )

***But there is more!***

Let's make this a little more efficient:  
In this kind of policy iteration frameworks, it's not necessary to fully evaluate our policy (run many episodes and get the mean return for each ${Q}$): we can just run **one** episode, update only the ${Q}$ values of the state-action pairs we visited during such episode (evaluation), improve our policy based on the new ${Q}$s we obtained, and repeat. Why should we wait to get more episodes of information when we could already improve the policy?   
So once again, here we are with our new approach:

  <img src="images/policy_it_4.png" style="zoom:45%"/>

**No, we are not done yet, there's still one little problem to solve**    We almost have the full picture. One last step:    
How can we really guarantee that we find the best possible policy? what we really want to find is ${\pi^*}$, the optimal policy. Such policy is greedy by definition, it tells us what action to take (with no doubts) in state ${s}$.  
So what we want to do is to explore in order to  make sure we are not missing out on anything, but at a certain point, asymptotically, we want to stop exploring, and just exploiting.  
This translates into the GLIE property (Greedy in the Limit with Infinite Exploration):

  - All state-action pairs are explored **infinitely** many tymes:  
    $$
    \lim_{k\to\infty}N_k(s,a)=\infty
    $$

  - The policy **converges** on a **greedy** policy:  
    $$
    \lim_{k\to\infty}\pi_k(a|s)=
    \begin{cases} 
    \mathbf{1} \ \ \ \ if \ a=\arg \max_{a'\in A}Q_k(s,a') \\
    \mathbf{0} \ \ \ \ otherwise
    \end{cases}
    $$
    
    (if you haven't read PoliMi slides don't bother reading: on Restelli's slides there is a mistake: there's written ${Q_k(s',a')}$ but that ${s'}$ should be ${s}$. )

  Now we are ready to actually answer the question: what is used in Monte Carlo Control?
  Here it is:  
  **GLIE Monte Carlo Control**  

  1. Sample ${k}$-th episode using ${\pi}$: ${\{S_1,A_1,R_2,...,S_T\} \sim \pi}$  

  2. For each state ${S_t}$ and action ${A_t}$ visited in the episode:  
     $$
     N(S_t,A_t) \leftarrow N(S_t,A_t)+1
     $$

     $$
     Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\frac{1}{N(S_t,A_t)}(G_t-Q(S_t,A_t))
     $$

       

  3.  Improve the policy based on the new action-value function, considering:  
     $$
     \epsilon \leftarrow\frac{1}{k}
     $$
     ${ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \  \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \pi \leftarrow \epsilon}$-greedy${(Q)}$

So what has changed from before? well, we are updating ${\epsilon}$ every time we run a new episode! Now it will become smaller and smaller each time we generate a new episode.  
Theorem:  
*GLIE Monte Carlo Control converges to the optimal action-value function:*  
$$
  Q(s,a)\to q^*(s,a)
$$
Well, GLIE MC is our first full solution. We can throw this into any MDP and it will find the right solution!   
So, let's sum up the solutions we adopted for MC control:

  1. Use ${Q}$, not ${V}$
  2. evaluate and improve your policy *every time you run an episode*
        3. use an ${\epsilon}$-greedy policy
  4. the value of ${\epsilon}$ should decay at every iteration in order to guarantee to find the optimal policy

*I'll see you in another life when we are both cats*

(Sources:  [David Silver's Lesson 5 on RL ](https://www.youtube.com/watch?v=0g4j2k_Ggc4&t=630s) -  Restelli's Slides  -  [Model Free Algorithms](https://medium.com/deep-math-machine-learning-ai/ch-12-1-model-free-reinforcement-learning-algorithms-monte-carlo-sarsa-q-learning-65267cb8d1b4)  )



  

<div style="page-break-after: always;"></div> 

### UCB1 Algorithm {TODO}

***Describe the UCB1 algorithm. Is it a deterministic or a stochastic algorithm?***





<div style="page-break-after: always;"></div> 

# Exercises {TODO}

- ***Which criteria would you consider for model selection in each one of the following settings:***
  - *A small dataset and a space of simple models;*
  - *A small dataset and a space of complex models;*
  - *A large dataset and a space of simple models;*
  - *A large dataset and a trainer with parallel computing abilities.*

<div style="page-break-after: always;"></div> 

- ***Categorize the following ML problems. For each one of them suggest a set of features which might be useful to solve the problem and a method to solve it.***
  - *Recognize handwritten characters;*
  - *Identify interesting features from an image;*
  - *Teach a robot how to play bowling;*
  - *Predicting car values for second-hand retailers.*



<div style="page-break-after: always;"></div> 

- ***State if the following are applicable to generic RL problems or MAB problems. Motivate your answers.***
  - *We should take into account state-action pairs one by one to estimate the value function;*
  - *Past actions you perform on the MDP might inﬂuence the future rewards you will gain;*
  - *The Markov property holds;*
  - *The time horizon of the episode is ﬁnite.*
- ***After training a perceptron classiﬁer on a given dataset, you ﬁnd that it does not achieve the desired performance on the training set, nor the validation one. Which of the following might be a promising step to take? Motivate your answers.*** 
  - Use an SVM with a Gaussian Kernel;
  - *Add features by basing on the problem characteristics;*
  - *Use linear regression with a linear kernel, without introducing new features;*
  - *Introduce a regularization term.*

<div style="page-break-after: always;"></div> 

- ***The generic deﬁnition of a policy is a stochastic function $\pi(h_i) = P(a_i|h_i)$ that, given a history $h_i = \{o_1,a_1,s_1,\dots,o_i,a_i,s_i\}$, provides a distribution over the possible actions $\{a_i\}_i$. Formulate the speciﬁc deﬁnition of a policy if the considered problem is:*** 
  - *Markovian, Stochastic, Non-stationary;*
  - *History based, Deterministic, Stationary;*
  - *Markovian, Deterministic, Stationary;*
  - *History based, Stochastic, Non-stationary.*



<div style="page-break-after: always;"></div> 



- ***Tell if the following statements are true or false. Provide adequate motivations to your answer.***
  - [ ] *Reinforcement Learning (RL) techniques use a tabular representation of MDPs to handle continuous state and/or action spaces*
  - [ ] *We can use data coming from sub-optimal policies to learn the optimal one.*
  - [ ] *In RL we always estimate the model of the environment.*
  - [ ] *In RL we require to have the model of the environment.*



<div style="page-break-after: always;"></div> 

- ***Consider separately the following characteristics for an ML problem:***

  - *Small dataset*.
  - *Limited computational resources for training.*
  - *Limited computational resources for prediction.*
  - *Prior information on data distribution.*

  *Provide motivations for the use of either a parametric or non-parametric method.*



<div style="page-break-after: always;"></div> 

- ***Categorize the following ML problems:***
  - *Pricing goods for an e-commerce website.*
  - *Teaching a robot to play table tennis.*
  - *Predicting housing prices for real estate.*
  - *Identifying counterfeit notes and coins.*



<div style="page-break-after: always;"></div> 

- ***Consider the Thompson Sampling algorithm. Assume to have the following posterior distributions $Beta_i(αt,βt)$ for arms $A = {a_1,...,a_5}$ rewards, which are distributed as Bernoulli random variables with mean $\mu_i$, and you extracted from them the samples $\hat{r}(a_i)$:***
  $$
  a_1:\space\alpha_t = 1\space\space\space\beta_t=5\space\space\space\hat{r}(a_1)=0.63\space\space\space\mu_1=0.1\\a_2:\space\alpha_t = 6\space\space\space\beta_t=4\space\space\space\hat{r}(a_2)=0.35\space\space\space\mu_2=0.5\\a_3:\space\alpha_t = 11\space\space\space\beta_t=23\space\space\space\hat{r}(a_3)=0.16\space\space\space\mu_3=0.3\\a_4:\space\alpha_t = 12\space\space\space\beta_t=25\space\space\space\hat{r}(a_4)=0.22\space\space\space\mu_4=0.2\\a_5:\space\alpha_t = 38\space\space\space\beta_t=21\space\space\space\hat{r}(a_5)=0.7\space\space\space\mu_5=0.6
  $$

  - *How much pseudo-regret the $TS$ algorithm accumulated so far, assuming we started from uniform $Beta(1,1)$ priors?*
  - *Which one of the previous posteriors is the most peaked one?*
  - *What would $UCB1$ have chosen for the next round? Assume $Bernoulli$ rewards and that in the Bayesian setting we started from uniform $Beta(1,1)$ priors?*



<div style="page-break-after: always;"></div> 



# Interesting Articles

- [Polynomial Regression](https://towardsdatascience.com/polynomial-regression-bbe8b9d97491)
- [Difference between Frequentist and Bayesian Approach](https://towardsdatascience.com/introduction-to-bayesian-linear-regression-e66e60791ea7) 
- [Model Based Algorithms](https://medium.com/deep-math-machine-learning-ai/ch-12-reinforcement-learning-complete-guide-towardsagi-ceea325c5d53)
- [Model Free Algorithms](https://medium.com/deep-math-machine-learning-ai/ch-12-1-model-free-reinforcement-learning-algorithms-monte-carlo-sarsa-q-learning-65267cb8d1b4)
- [Q-Learning (and Policy Gradients)](https://medium.com/deep-math-machine-learning-ai/ch-13-deep-reinforcement-learning-deep-q-learning-and-policy-gradients-towards-agi-a2a0b611617e)

