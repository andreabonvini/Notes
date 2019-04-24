# MIT 18.06 Linear Algebra Course - Notes 

### Lecture 1: The Geometry of  Linear Equations

The fundamental problem of Linear Algebra is to solve a system of linear equations.
$$
\begin{bmatrix}
    2 & -1\\
    -1 & 2
  \end{bmatrix}
  \begin{bmatrix}
  	\bold{x}\\
  	\bold{y}
  \end{bmatrix}
= 
\begin{bmatrix}
  	0\\
  	3
  \end{bmatrix}
$$
​										A**x** = b
$$
\bold{x}
\begin{bmatrix}
2\\
-1
\end{bmatrix}
+
\bold{y}
\begin{bmatrix}
-1\\
2
\end{bmatrix}

= 


\begin{bmatrix}
0\\
3
\end{bmatrix}
$$
What is the equation asking for? It's asking us to find somehow to combine these two vectors in the right amount in order to obtain the result *b*. It's asking us to find the right linear combination of the columns of A (i.e. the two vectors).

Let's do now a 3x3 example:

$$
2x-y = 0\\
-x+2y-z = 0\\
-3y + 4z=4\\
A = \begin{bmatrix}
2 & -1 & 0\\
-1 & 2 & -1\\
0 & -3 & 4
\end{bmatrix}\\
b = 
\begin{bmatrix}
0\\
-1\\
4
\end{bmatrix}\\
\begin{bmatrix}
2\\
-1\\
0
\end{bmatrix}\bold{x}+
\begin{bmatrix}
-1\\
2\\
-3
\end{bmatrix}\bold{y}+
\begin{bmatrix}
0\\
-1\\
4
\end{bmatrix}\bold{z}=
\begin{bmatrix}
0\\
-1\\
4
\end{bmatrix}\\
$$


In this simple example it's trivial to see that the right answer is **x** = **y** = 0, **z** = 1.

Now we ask ourselves: Can I solve Ax = b for every b? i.e. Do the linear combination of the columns vectors fill the whole 3D space? 

For this matrix the answer is YES. When would the answer be NO? e.g. where the tree vectors happen to lie in the same plane... we'll come back to this concept in the following lectures.

## Lecture 2: Elimination with Matrices

in order to operate the classical Gauss elimination we have to know how to perform some basic matrix manipulation.

Remember that *I* is the identity matrix.
$$
M = \begin{bmatrix}
a & b & c\\
d & e & f\\
g & h & i\\
\end{bmatrix}
\
I = \begin{bmatrix}
1 & 0 & 0\\
0 & 1 & 0\\
0 & 0 & 1\\
\end{bmatrix}
$$

* Subtraction (*e.g. subtract 3\*row1 from row2*)
  $$
  \begin{bmatrix}
  1 & 0 & 0\\
  -3 & 1 & 0\\
  0 & 0 & 1
  \end{bmatrix}
  \begin{bmatrix}
  a & b & c\\
  d & e & f\\
  g & h & i\\
  \end{bmatrix}=
  \begin{bmatrix}
  a & b & c\\
  d-3a & e-3b & f-3c\\
  g & h & i\\
  \end{bmatrix}
  $$

* Row Permutation (*e.g. switch row1 with row2*)
  $$
  \begin{bmatrix}
  0 & 1 & 0\\
  1 & 0 & 0\\
  0 & 0 & 1
  \end{bmatrix}
  \begin{bmatrix}
  a & b & c\\
  d & e & f\\
  g & h & i\\
  \end{bmatrix}=
  \begin{bmatrix}
  d & e & f\\
  a & b & c\\
  g & h & i\\
  \end{bmatrix}
  $$

* Column Permutation (*e.g. switch col1 with col2*)
  $$
  \begin{bmatrix}
  a & b & c\\
  d & e & f\\
  g & h & i\\
  \end{bmatrix}
  \begin{bmatrix}
  0 & 1 & 0\\
  1 & 0 & 0\\
  0 & 0 & 1
  \end{bmatrix}=
  \begin{bmatrix}
  b & a & f\\
  e & d & c\\
  h & g & i\\
  \end{bmatrix}
  $$
  

We can start to build an intuition about what's an Inverse Matrix. Let's consider the matrix defined in order to perform subtraction:
$$
\begin{bmatrix}
1 & 0 & 0\\
-3 & 1 & 0\\
0 & 0 & 1
\end{bmatrix}
$$
it's inverse matrix will be the matrix that will "un-do" the effects of the former matrix:
$$
\begin{bmatrix}
1 & 0 & 0\\
+3 & 1 & 0\\
0 & 0 & 1
\end{bmatrix}
$$
(SUBTRACT 3\*ROW1 FROM ROW2) ----> (ADD 3\*ROW1 FROM ROW2)



## Lecture 3 - Multiplication and Inverse matrices

If I have a matrix **A** of dimension **m\*n** and a matrix **B** of dimension **n\*p** I'll obtain a matrix **P** of dimensions **m\*p**. (See that the number of columns of A *MUST BE* equal to the number of rows of B)

For SQUARE matrixes, if an Inverse matrix exists, it is both a Left and a Right inverse matrix:
$$
A^{-1}A = I = AA^{-1}
$$
