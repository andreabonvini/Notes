# Linear Discriminant Analysis

*LDA* is a supervised technique of feature extraction used to find a linear combination of the available features which separate the classes.

The main objective is to reduce the dimension of data, in order to reduce the computational cost of classification. After applying the LDA to a dataset, the "new features" minimizes the dispersion between samples of the same class, and maximizes the dispersion between samples of different classes.

LDA is a very powerful technique, but it has some limitations. The LDA can generate only C. or less, number of features, where C is the number of classes.

Sometimes, you are dealing with datasets of 10 features, for example, and there is only 2 classes, so LDA have to reduce to 2 axis. Such reductions, usually leads to loss of information, which causes low classification accuracy.

If you already know *Principal Component Analysis*, imagine a PCA where the principal component is not the axis of maximum variance, but the axis of maximum variance "of a class", and the next axis for the next class, and so on.


$$
\color{black}P(Y=k|X=x)=\frac{\color{blue}P(X=x|Y=k)\color{green}P(Y=k)}{\color{purple}P(X=x)}\\
\color{black}P(Y=k|X=x)=\frac{\color{green}\pi_k\color{blue}f_k(x)}{\color{purple}\sum_{l=1}^{K}\pi_lf_l(x)}
$$
$f_k(x)$ is the *density* for $X$ in class $k$ . We usually consider it *normal*.

$\pi_k$ is the marginal *prior* probability for class $k$.

We classify a new point according to which density is highest.

<img src="/Users/z051m4/Desktop/Projects/Github/Notes/Advanced_Signals/Barbieri/images/LDA.PNG" style="zoom:60%"/>

When the classes are well-separated, the parameter estimates for the *logistic regression* model are surprisingly unstable. *Linear discriminant analysis* does not suffer from this problem. 

The Gaussian Density has the form
$$
f_k(x)=\frac{1}{\sqrt{2\pi}\sigma_k}e^{-\frac{1}{2}\left(\frac{x-\mu_k}{\sigma_k}\right)^2}
$$
here $\mu_k$ is the mean and $\sigma_k^2$ is the variance for class $k$. Plugging this into Bayes formula we get
$$
p_k(x) = P(Y=k|X=x)=\frac{\pi_k\frac{1}{\sqrt{2\pi}\sigma_k}e^{-\frac{1}{2}\left(\frac{x-\mu_k}{\sigma_k}\right)^2}}{\sum_{l=1}^{K}\pi_l\frac{1}{\sqrt{2\pi}\sigma_l}e^{-\frac{1}{2}\left(\frac{x-\mu_l}{\sigma_l}\right)^2}}
$$
To classify at the value $X=x$ we need to see which of the $p_k(x)$ is largest. 

If we assume all the $\sigma_k=\sigma$ are the same, taking logs and discarding terms that do not depend on $k$ , we see that this is equivalent to assigning $x$ to the class with the largest *discriminant score*:
$$
\delta_k(x) = x\cdot\frac{\mu_k}{\sigma^2}-\frac{\mu_k^2}{2\sigma^2}+\log(\pi_k)
$$
Note that $\delta_k(x)$ is a ***linear*** function of $x$.

*Estimating the parameters*:

We just estimate the parameters using the training data.
$$
\hat{\pi}_k=\frac{n_k}{n}\\
\hat{\mu}_k=\frac{1}{n_k}\sum_{i:y_i=k}x_i\\
\hat{\sigma}^2=\frac{1}{n-K}\sum_{k=1}^K\sum_{i:y_i=k}(x_i-\hat{\mu}_k)^2
$$
Once we have estimated $\hat{\delta}_k(x)$ we can turn these into estimates for class probabilities:
$$
\hat{P}(Y=k|X=x)=\frac{e^{\hat{\delta}_k(x)}}{\sum_{l=1}^{K}e^{\hat{\delta}_l(x)}}
$$
When 𝐾=2, we classify to class $2$ if $\hat{P}(Y=k|X=x) \ge0.5$ or otherwise to class $1$.

Obviously we can change the threshold $0.5$ and build the ROC curve to choose the model we prefer.

`to do: naive nayes and qda`

naive bayes assumes features are independent in each class. 

Logistic Regression is very popular for classification, especially when K=2. 

Linear Discrimination Analysis is useful when n is small, or the classes are well separated, and Gaussian assumptions are reasonable. Also when K>2.

Naive Bayes is useful when $p$ is very large.