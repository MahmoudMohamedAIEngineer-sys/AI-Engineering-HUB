Figure 4.13 A functional phylogeny of matrices encountered in machine learning. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/7914ad191a8440e30240576fb1809599ddf8c1a8e90c8958b1718eb2c613ea21.jpg)


## 4.7 Matrix Phylogeny

In Chapters 2 and 3, we covered the basics of linear algebra and analytic geometry. In this chapter, we looked at fundamental characteristics of matrices and linear mappings. Figure 4.13 depicts the phylogenetic tree of relationships between different types of matrices (black arrows indicating “is a subset of”) and the covered operations we can perform on them (in blue). We consider all real matrices $\pmb { A } \in \mathbb { R } ^ { n \times m }$ . For non-square matrices (where $n \neq m )$ , the SVD always exists, as we saw in this chapter. Focusing on square matrices $A \in \mathbb { R } ^ { n \times n }$ , the determinant informs us whether a square matrix possesses an inverse matrix, i.e., whether it belongs to the class of regular, invertible matrices. If the square $n \times n$ matrix possesses n linearly independent eigenvectors, then the matrix is non-defective and an eigendecomposition exists (Theorem 4.12). We know that repeated eigenvalues may result in defective matrices, which cannot be diagonalized. 

The word “phylogenetic” describes how we capture the relationships among individuals or groups and derived from the Greek words for “tribe” and “source”. 

Non-singular and non-defective matrices are not the same. For example, a rotation matrix will be invertible (determinant is nonzero) but not diagonalizable in the real numbers (eigenvalues are not guaranteed to be real numbers). 

We dive further into the branch of non-defective square $n \times n$ matrices. A is normal if the condition $\pmb { A } ^ { \top } \pmb { A } = \pmb { A } \pmb { A } ^ { \top }$ holds. Moreover, if the more restrictive condition holds that $\pmb { A } ^ { \top } \pmb { A } = \pmb { A } \pmb { A } ^ { \top } = \pmb { I } ,$ , then A is called orthogonal (see Definition 3.8). The set of orthogonal matrices is a subset of the regular (invertible) matrices and satisfies $\mathbf { \bar { A } } ^ { \top } = A ^ { - 1 }$ 

Normal matrices have a frequently encountered subset, the symmetric matrices $\ b { S } \in \mathbb { R } ^ { n \times n }$ , which satisfy $S \stackrel { \cdot } { = } S ^ { \top }$ . Symmetric matrices have only real eigenvalues. A subset of the symmetric matrices consists of the positive definite matrices P that satisfy the condition of ${ \pmb x } ^ { \top } { \pmb P } { \pmb x } > 0$ for all $\pmb { x } \in \mathbb { R } ^ { n } \backslash \{ \mathbf { 0 } \}$ . In this case, a unique Cholesky decomposition exists (Theorem 4.18). Positive definite matrices have only positive eigenvalues and are always invertible $( \mathrm { i . e . }$ , have a nonzero determinant). 

Another subset of symmetric matrices consists of the diagonal matrices D. Diagonal matrices are closed under multiplication and addition, but do not necessarily form a group (this is only the case if all diagonal entries are nonzero so that the matrix is invertible). A special diagonal matrix is the identity matrix I. 

## 4.8 Further Reading

Most of the content in this chapter establishes underlying mathematics and connects them to methods for studying mappings, many of which are at the heart of machine learning at the level of underpinning software solutions and building blocks for almost all machine learning theory. Matrix characterization using determinants, eigenspectra, and eigenspaces provides fundamental features and conditions for categorizing and analyzing matrices. This extends to all forms of representations of data and map pings involving data, as well as judging the numerical stability of computational operations on such matrices (Press et al., 2007). 

Determinants are fundamental tools in order to invert matrices and compute eigenvalues “by hand”. However, for almost all but the smallest instances, numerical computation by Gaussian elimination outperforms determinants (Press et al., 2007). Determinants remain nevertheless a powerful theoretical concept, e.g., to gain intuition about the orientation of a basis based on the sign of the determinant. Eigenvectors can be used to perform basis changes to transform data into the coordinates of meaningful orthogonal, feature vectors. Similarly, matrix decomposition methods, such as the Cholesky decomposition, reappear often when we compute or simulate random events (Rubinstein and Kroese, 2016). Therefore, the Cholesky decomposition enables us to compute the reparametrization trick where we want to perform continuous differentiation over random variables, e.g., in variational autoencoders (Jimenez Rezende et al., 2014; Kingma and Welling, 2014). 

Eigendecomposition is fundamental in enabling us to extract meaningful and interpretable information that characterizes linear mappings. 

Therefore, the eigendecomposition underlies a general class of machine learning algorithms called spectral methods that perform eigendecomposition of a positive-definite kernel. These spectral decomposition methods encompass classical approaches to statistical data analysis, such as the following: 

Principal component analysis (PCA (Pearson, 1901), see also Chapter 10), in which a low-dimensional subspace, which explains most of the variability in the data, is sought. 

Fisher discriminant analysis, which aims to determine a separating hyperplane for data classification (Mika et al., 1999). 

Multidimensional scaling (MDS) (Carroll and Chang, 1970). 

The computational efficiency of these methods typically comes from finding the best rank-k approximation to a symmetric, positive semidefinite matrix. More contemporary examples of spectral methods have different origins, but each of them requires the computation of the eigenvectors and eigenvalues of a positive-definite kernel, such as Isomap (Tenenbaum et al., 2000), Laplacian eigenmaps (Belkin and Niyogi, 2003), Hessian eigenmaps (Donoho and Grimes, 2003), and spectral clustering (Shi and Malik, 2000). The core computations of these are generally underpinned by low-rank matrix approximation techniques (Belabbas and Wolfe, 2009) as we encountered here via the SVD. 

The SVD allows us to discover some of the same kind of information as the eigendecomposition. However, the SVD is more generally applicable to non-square matrices and data tables. These matrix factorization methods become relevant whenever we want to identify heterogeneity in data when we want to perform data compression by approximation, e.g., instead of storing n m values just storing (n+m)k values, or when we want to perform data pre-processing, e.g., to decorrelate predictor variables of a design matrix (Ormoneit et al., 2001). The SVD operates on matrices, which we can interpret as rectangular arrays with two indices (rows and columns). The extension of matrix-like structure to higher-dimensional arrays are called tensors. It turns out that the SVD is the special case of a more general family of decompositions that operate on such tensors (Kolda and Bader, 2009). SVD-like operations and low-rank approximations on tensors are, for example, the Tucker decomposition (Tucker, 1966) or the CP decomposition (Carroll and Chang, 1970). 

The SVD low-rank approximation is frequently used in machine learning for computational efficiency reasons. This is because it reduces the amount of memory and operations with nonzero multiplications we need to perform on potentially very large matrices of data (Trefethen and Bau III, 1997). Moreover, low-rank approximations are used to operate on matrices that may contain missing values as well as for purposes of lossy compression and dimensionality reduction (Moonen and De Moor, 1995; Markovsky, 2011). 

Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 

## Exercises

4.1 Compute the determinant using the Laplace expansion (using the first row) and the Sarrus rule for 

$$
\boldsymbol {A} = \left[ \begin{array}{c c c} 1 & 3 & 5 \\ 2 & 4 & 6 \\ 0 & 2 & 4 \end{array} \right].
$$

4.2 Compute the following determinant efficiently: 

$$
\left[ \begin{array}{c c c c c} 2 & 0 & 1 & 2 & 0 \\ 2 & - 1 & 0 & 1 & 1 \\ 0 & 1 & 2 & 1 & 2 \\ - 2 & 0 & 2 & - 1 & 2 \\ 2 & 0 & 0 & 1 & 1 \end{array} \right].
$$

4.3 Compute the eigenspaces of 

a. 

$$
\boldsymbol {A} := \left[ \begin{array}{c c} 1 & 0 \\ 1 & 1 \end{array} \right]
$$

b. 

$$
\boldsymbol {B} := \left[ \begin{array}{c c} - 2 & 2 \\ 2 & 1 \end{array} \right]
$$

4.4 Compute all eigenspaces of 

$$
\boldsymbol {A} = \left[ \begin{array}{c c c c} 0 & - 1 & 1 & 1 \\ - 1 & 1 & - 2 & 3 \\ 2 & - 1 & 0 & 0 \\ 1 & - 1 & 1 & 0 \end{array} \right].
$$

4.5 Diagonalizability of a matrix is unrelated to its invertibility. Determine for the following four matrices whether they are diagonalizable and/or invertible 

$$
\left[ \begin{array}{c c} 1 & 0 \\ 0 & 1 \end{array} \right], \quad \left[ \begin{array}{c c} 1 & 0 \\ 0 & 0 \end{array} \right], \quad \left[ \begin{array}{c c} 1 & 1 \\ 0 & 1 \end{array} \right], \quad \left[ \begin{array}{c c} 0 & 1 \\ 0 & 0 \end{array} \right].
$$

4.6 Compute the eigenspaces of the following transformation matrices. Are they diagonalizable? 

a. For 

$$
\boldsymbol {A} = \left[ \begin{array}{c c c} 2 & 3 & 0 \\ 1 & 4 & 3 \\ 0 & 0 & 1 \end{array} \right]
$$

b. For 

$$
\boldsymbol {A} = \left[ \begin{array}{c c c c} 1 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{array} \right]
$$

©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 

4.7 Are the following matrices diagonalizable? If yes, determine their diagonal form and a basis with respect to which the transformation matrices are diagonal. If no, give reasons why they are not diagonalizable. 

a. 

$$
\mathbf {A} = \left[ \begin{array}{l l} 0 & 1 \\ - 8 & 4 \end{array} \right]
$$

b. 

$$
\boldsymbol {A} = \left[ \begin{array}{c c c} 1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{array} \right]
$$

c. 

$$
\boldsymbol {A} = \left[ \begin{array}{c c c c} 5 & 4 & 2 & 1 \\ 0 & 1 & - 1 & - 1 \\ - 1 & - 1 & 3 & 0 \\ 1 & 1 & - 1 & 2 \end{array} \right]
$$

d. 

$$
\boldsymbol {A} = \left[ \begin{array}{c c c} 5 & - 6 & - 6 \\ - 1 & 4 & 2 \\ 3 & - 6 & - 4 \end{array} \right]
$$

4.8 Find the SVD of the matrix 

$$
\boldsymbol {A} = \left[ \begin{array}{c c c} 3 & 2 & 2 \\ 2 & 3 & - 2 \end{array} \right].
$$

4.9 Find the singular value decomposition of 

$$
\boldsymbol {A} = \left[ \begin{array}{c c} 2 & 2 \\ - 1 & 1 \end{array} \right].
$$

4.10 Find the rank-1 approximation of 

$$
\boldsymbol {A} = \left[ \begin{array}{c c c} 3 & 2 & 2 \\ 2 & 3 & - 2 \end{array} \right]
$$

4.11 Show that for any $\pmb { A } \in \mathbb { R } ^ { m \times n }$ the matrices $A ^ { \top } A$ and $A A ^ { \top }$ possess the same nonzero eigenvalues. 

4.12 Show that for x $\neq \mathbf { 0 }$ Theorem 4.24 holds, i.e., show that 

$$
\max _ {\boldsymbol {x}} \frac {\| \boldsymbol {A} \boldsymbol {x} \| _ {2}}{\| \boldsymbol {x} \| _ {2}} = \sigma_ {1},
$$

where $\sigma _ { 1 }$ is the largest singular value of $\pmb { A } \in \mathbb { R } ^ { m \times n }$ 

# Vector Calculus

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/a10be2042dd272dcecfe3de08e5f0264f8e600826bc8bec4f8b78b28eef6e504.jpg)


Many algorithms in machine learning optimize an objective function with respect to a set of desired model parameters that control how well a model explains the data: Finding good parameters can be phrased as an optimization problem (see Sections 8.2 and 8.3). Examples include: (i) lin ear regression (see Chapter 9), where we look at curve-fitting problems and optimize linear weight parameters to maximize the likelihood; (ii) neural-network auto-encoders for dimensionality reduction and data com pression, where the parameters are the weights and biases of each layer, and where we minimize a reconstruction error by repeated application of the chain rule; and (iii) Gaussian mixture models (see Chapter 11) for modeling data distributions, where we optimize the location and shape parameters of each mixture component to maximize the likelihood of the model. Figure 5.1 illustrates some of these problems, which we typically solve by using optimization algorithms that exploit gradient information (Section 7.1). Figure 5.2 gives an overview of how concepts in this chapter are related and how they are connected to other chapters of the book. 

Central to this chapter is the concept of a function. A function $f$ is a quantity that relates two quantities to each other. In this book, these quantities are typically inputs $\pmb { x } \in \mathbb { R } ^ { D }$ and targets (function values) $f ( { \pmb x } )$ which we assume are real-valued if not stated otherwise. Here $\mathbb { R } ^ { D }$ is the domain of $f ,$ and the function values $f ( { \pmb x } )$ are the image/codomain of $f .$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/e36734b02e62fe71f52d1fbad8b08cfe7bd24d0f387e2c854d005c4d659b9615.jpg)



(a) Regression problem: Find parameters, such that the curve explains the observations (crosses) well.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/4f19cfe1f8e4d982330ead12c09c1acc0bbd1b5dbb4a52dd6c5ff60ecb9682c3.jpg)



domain image/codomain Figure 5.1 Vector calculus plays a central role in (a) regression (curve fitting) and (b) density estimation, $\mathrm { i . e . , }$ modeling data distributions.



(b) Density estimation with a Gaussian mixture model: Find means and covariances, such that the data (dots) can be explained well.


139 


Vector Calculus



Figure 5.2 A mind map of the concepts introduced in this chapter, along with when they are used in other parts of the book.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/857bc05e188f2d1196b4b76c67b9c06a10a9a9b360f0b8594097e85b4c057357.jpg)


Section 2.7.3 provides much more detailed discussion in the context of linear functions. We often write 

$$
f: \mathbb {R} ^ {D} \to \mathbb {R}\tag{5.1a}
$$

$$
\boldsymbol {x} \mapsto f (\boldsymbol {x})\tag{5.1b}
$$

to specify a function, where (5.1a) specifies that $f$ is a mapping from $\mathbb { R } ^ { D }$ to <sup>R</sup> and (5.1b) specifies the explicit assignment of an input x to a function value $f ( { \pmb x } )$ . A function $f$ assigns every input x exactly one function value f(x). 

## Example 5.1

Recall the dot product as a special case of an inner product (Section 3.2). In the previous notation, the function $f ( \pmb { x } ) = \pmb { x } ^ { \top } \pmb { x } , \ \pmb { x } \in \mathbb { R } ^ { 2 }$ , would be specified as 

$$
f: \mathbb {R} ^ {2} \to \mathbb {R}\tag{5.2a}
$$

$$
\pmb {x} \mapsto x _ {1} ^ {2} + x _ {2} ^ {2}.\tag{5.2b}
$$

In this chapter, we will discuss how to compute gradients of functions, which is often essential to facilitate learning in machine learning models since the gradient points in the direction of steepest ascent. Therefore, 

5.1 Differentiation of Univariate Functions 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/36e26a5e6885b839de1352ba362e7bd7580f559a9f8132dc12c0d83eef81b615.jpg)


Figure 5.3 The average incline of a function $f$ between $x _ { 0 }$ and $x _ { 0 } + \delta x$ is the incline of the secant (blue) through $f ( x _ { 0 } )$ and f(x<sub>0</sub> + δx) and given by $\delta y / \delta x$ 

vector calculus is one of the fundamental mathematical tools we need in machine learning. Throughout this book, we assume that functions are differentiable. With some additional technical definitions, which we do not cover here, many of the approaches presented can be extended to sub-differentials (functions that are continuous but not differentiable at certain points). We will look at an extension to the case of functions with constraints in Chapter 7. 

## 5.1 Differentiation of Univariate Functions

In the following, we briefly revisit differentiation of a univariate function, which may be familiar from high school mathematics. We start with the difference quotient of a univariate function $y = f ( x ) , x , y \in \mathbb { R }$ , which we will subsequently use to define derivatives. 

Definition 5.1 (Difference Quotient). The difference quotient 

$$
\frac {\delta y}{\delta x} := \frac {f (x + \delta x) - f (x)}{\delta x}\tag{5.3}
$$

computes the slope of the secant line through two points on the graph of $f .$ In Figure 5.3, these are the points with x-coordinates $x _ { 0 }$ and $x _ { 0 } + \delta x$ 

The difference quotient can also be considered the average slope of $f$ between x and $x + \delta x$ if we assume $f$ to be a linear function. In the limit for $\delta x  0$ , we obtain the tangent of $f$ at $x ,$ , if $f$ is differentiable. The tangent is then the derivative of $f$ at x. 

Definition 5.2 (Derivative). More formally, for $h > 0$ the derivative of $f$ derivative at x is defined as the limit 

$$
\frac {\mathrm{d} f}{\mathrm{d} x} := \lim _ {h \rightarrow 0} \frac {f (x + h) - f (x)}{h},\tag{5.4}
$$

and the secant in Figure 5.3 becomes a tangent. 

The derivative of $f$ points in the direction of steepest ascent of $f .$ 

## Example 5.2 (Derivative of a Polynomial)

We want to compute the derivative of $f ( x ) = x ^ { n } , n \in \mathbb { N }$ . We may already know that the answer will be $n x ^ { n - 1 }$ , but we want to derive this result using the definition of the derivative as the limit of the difference quotient. 

Using the definition of the derivative in (5.4), we obtain 

$$
\frac {\mathrm{d} f}{\mathrm{d} x} = \lim _ {h \rightarrow 0} \frac {f (x + h) - f (x)}{h}\tag{5.5a}
$$

$$
= \lim _ {h \to 0} \frac {(x + h) ^ {n} - x ^ {n}}{h}\tag{5.5b}
$$

$$
= \lim _ {h \to 0} \frac {\sum_ {i = 0} ^ {n} \binom {n} {i} x ^ {n - i} h ^ {i} - x ^ {n}}{h}.\tag{5.5c}
$$

We see that $x ^ { n } = { \binom { n } { 0 } } x ^ { n - 0 } h ^ { 0 }$ . By starting the sum at 1, the $x ^ { n }$ -term cancels, and we obtain 

$$
\frac {\mathrm{d} f}{\mathrm{d} x} = \lim _ {h \rightarrow 0} \frac {\sum_ {i = 1} ^ {n} \binom {n} {i} x ^ {n - i} h ^ {i}}{h}\tag{5.6a}
$$

$$
= \lim _ {h \to 0} \sum_ {i = 1} ^ {n} {\binom {n} {i}} x ^ {n - i} h ^ {i - 1}\tag{5.6b}
$$

$$
= \lim _ {h \rightarrow 0} \binom {n} {1} x ^ {n - 1} + \underbrace {\sum_ {i = 2} ^ {n} \binom {n} {i}} _ {\text {   }} x ^ {n - i} h ^ {i - 1}\tag{5.6c}
$$

$$
\rightarrow 0 \mathrm{as} h \rightarrow 0
$$

$$
= \frac {n !}{1 ! (n - 1) !} x ^ {n - 1} = n x ^ {n - 1}.\tag{5.6d}
$$

## 5.1.1 Taylor Series

The Taylor series is a representation of a function $f$ as an infinite sum of terms. These terms are determined using derivatives of $f$ evaluated at $x _ { 0 }$ . 

Taylor polynomial We define $t ^ { 0 } : = 1$ for all $t \in \mathbb { R }$ 

Definition 5.3 (Taylor Polynomial). The Taylor polynomial of degree n of $f : \mathbb { R } \to \mathbb { R }$ at $x _ { 0 }$ is defined as 

$$
T _ {n} (x) := \sum_ {k = 0} ^ {n} \frac {f ^ {(k)} \left(x _ {0}\right)}{k !} \left(x - x _ {0}\right) ^ {k},\tag{5.7}
$$

where $f ^ { ( k ) } ( x _ { 0 } )$ is the kth derivative of $f$ at $x _ { 0 }$ (which we assume exists) and $\frac { f ^ { ( k ) } ( x _ { 0 } ) } { k ! }$ are the coefficients of the polynomial. 

Definition 5.4 (Taylor Series). For a smooth function $f \in \mathcal { C } ^ { \infty } , f : \mathbb { R } \to \mathbb { R }$ , the Taylor series of $f$ at $x _ { 0 }$ is defined as 

Taylor series 

5.1 Differentiation of Univariate Functions 

$$
T _ {\infty} (x) = \sum_ {k = 0} ^ {\infty} \frac {f ^ {(k)} \left(x _ {0}\right)}{k !} \left(x - x _ {0}\right) ^ {k}.\tag{5.8}
$$

For $x _ { 0 } = 0$ , we obtain the Maclaurin series as a special instance of the Taylor series. If $f ( x ) = T _ { \infty } ( x )$ , then f is called analytic. 

Remark. In general, a Taylor polynomial of degree n is an approximation of a function, which does not need to be a polynomial. The Taylor polynomial is similar to $f$ in a neighborhood around $x _ { 0 }$ . However, a Taylor polynomial of degree n is an exact representation of a polynomial $f$ of degree k $\leqslant n$ since all derivatives $f ^ { ( i ) } , i > k$ vanish. 

## Example 5.3 (Taylor Polynomial)

We consider the polynomial 

$$
f (x) = x ^ {4}\tag{5.9}
$$

and seek the Taylor polynomial $T _ { 6 }$ , evaluated at $x _ { 0 } = 1$ . We start by computing the coefficients $f ^ { ( k ) } ( 1 )$ for $k = 0 , \ldots , 6 \colon$ 

$$
f (1) = 1\tag{5.10}
$$

$$
f ^ {\prime} (1) = 4\tag{5.11}
$$

$$
f ^ {\prime \prime} (1) = 1 2\tag{5.12}
$$

$$
f ^ {(3)} (1) = 2 4\tag{5.13}
$$

$$
f ^ {(4)} (1) = 2 4\tag{5.14}
$$

$$
f ^ {(5)} (1) = 0\tag{5.15}
$$

$$
f ^ {(6)} (1) = 0\tag{5.16}
$$

Therefore, the desired Taylor polynomial is 

$$
T _ {6} (x) = \sum_ {k = 0} ^ {6} \frac {f ^ {(k)} (x _ {0})}{k !} (x - x _ {0}) ^ {k}\tag{5.17a}
$$

$$
= 1 + 4 (x - 1) + 6 (x - 1) ^ {2} + 4 (x - 1) ^ {3} + (x - 1) ^ {4} + 0.\tag{5.17b}
$$

Multiplying out and re-arranging yields 

$$
T _ {6} (x) = (1 - 4 + 6 - 4 + 1) + x (4 - 1 2 + 1 2 - 4)
$$

$$
+ x ^ {2} (6 - 1 2 + 6) + x ^ {3} (4 - 4) + x ^ {4}\tag{5.18a}
$$

$$
= x ^ {4} = f (x),\tag{5.18b}
$$

i.e., we obtain an exact representation of the original function. 



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



Figure 5.4 Taylor polynomials. The original function $f ( x ) =$ 

$$
(x) + \cos (x)
$$

(black, solid) is approximated by Taylor polynomials (dashed) around $x _ { 0 } = 0 .$ 

Higher-order Taylor polynomials approximate the function $f$ better and more globally. $T _ { 1 0 }$ is already similar to $f$ in [−4, 4]. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/6295d3a5d3f058376008883c7ca72d0fc733f8266285326751579c51e3a58cc9.jpg)


## Example 5.4 (Taylor Series)

Consider the function in Figure 5.4 given by 

$$
f (x) = \sin (x) + \cos (x) \in \mathcal {C} ^ {\infty}.\tag{5.19}
$$

We seek a Taylor series expansion of $f$ at $x _ { 0 } = 0$ , which is the Maclaurin series expansion of $f$ . We obtain the following derivatives: 

$$
f (0) = \sin (0) + \cos (0) = 1\tag{5.20}
$$

$$
f ^ {\prime} (0) = \cos (0) - \sin (0) = 1\tag{5.21}
$$

$$
f ^ {\prime \prime} (0) = - \sin (0) - \cos (0) = - 1\tag{5.22}
$$

$$
f ^ {(3)} (0) = - \cos (0) + \sin (0) = - 1\tag{5.23}
$$

$$
f ^ {(4)} (0) = \sin (0) + \cos (0) = f (0) = 1\tag{5.24}
$$

We can see a pattern here: The coefficients in our Taylor series are only $\pm 1$ (since sin $( 0 ) = 0 )$ , each of which occurs twice before switching to the other one. Furthermore, $f ^ { ( k + 4 ) } ( 0 ) = f ^ { ( k ) } ( 0 )$ 

Therefore, the full Taylor series expansion of $f$ at $x _ { 0 } = 0$ is given by 

$$
T _ {\infty} (x) = \sum_ {k = 0} ^ {\infty} \frac {f ^ {(k)} (x _ {0})}{k !} (x - x _ {0}) ^ {k}\tag{5.25a}
$$

$$
= 1 + x - \frac {1}{2 !} x ^ {2} - \frac {1}{3 !} x ^ {3} + \frac {1}{4 !} x ^ {4} + \frac {1}{5 !} x ^ {5} - \dots\tag{5.25b}
$$

$$
= 1 - \frac {1}{2 !} x ^ {2} + \frac {1}{4 !} x ^ {4} \mp \dots + x - \frac {1}{3 !} x ^ {3} + \frac {1}{5 !} x ^ {5} \mp \dots\tag{5.25c}
$$

$$
= \sum_ {k = 0} ^ {\infty} (- 1) ^ {k} \frac {1}{(2 k) !} x ^ {2 k} + \sum_ {k = 0} ^ {\infty} (- 1) ^ {k} \frac {1}{(2 k + 1) !} x ^ {2 k + 1}
$$

$$
= \cos (x) + \sin (x),\tag{5.25d}
$$

(5.25e) 

Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 

where we used the power series representations 

$$
\cos (x) = \sum_ {k = 0} ^ {\infty} (- 1) ^ {k} \frac {1}{(2 k) !} x ^ {2 k},\tag{5.26}
$$

$$
\sin (x) = \sum_ {k = 0} ^ {\infty} (- 1) ^ {k} \frac {1}{(2 k + 1) !} x ^ {2 k + 1}.\tag{5.27}
$$

Figure 5.4 shows the corresponding first Taylor polynomials $T _ { n }$ for $n =$ 0, 1, 5, 10. 

Remark. A Taylor series is a special case of a power series 

$$
f (x) = \sum_ {k = 0} ^ {\infty} a _ {k} (x - c) ^ {k}\tag{5.28}
$$

where $a _ { k }$ are coefficients and c is a constant, which has the special form in Definition 5.4. 

## 5.1.2 Differentiation Rules

In the following, we briefly state basic differentiation rules, where we denote the derivative of f by $f ^ { \prime }$ . 

$$
\text { Product   rule: } \quad (f (x) g (x)) ^ {\prime} = f ^ {\prime} (x) g (x) + f (x) g ^ {\prime} (x)\tag{5.29}
$$

$$
\text { Quotient   rule: } \quad \left(\frac {f (x)}{g (x)}\right) ^ {\prime} = \frac {f ^ {\prime} (x) g (x) - f (x) g ^ {\prime} (x)}{(g (x)) ^ {2}}\tag{5.30}
$$

$$
\text {   Sum   rule:   } \quad (f (x) + g (x)) ^ {\prime} = f ^ {\prime} (x) + g ^ {\prime} (x)\tag{5.31}
$$

$$
\text { Chain   rule: } \quad (g (f (x))) ^ {\prime} = (g \circ f) ^ {\prime} (x) = g ^ {\prime} (f (x)) f ^ {\prime} (x)\tag{5.32}
$$

Here, $g \circ f$ denotes function composition $x \mapsto f ( x ) \mapsto g ( f ( x ) )$ 

## Example 5.5 (Chain Rule)

Let us compute the derivative of the function $h ( x ) = ( 2 x + 1 ) ^ { 4 }$ using the chain rule. With 

$$
h (x) = (2 x + 1) ^ {4} = g (f (x)),\tag{5.33}
$$

$$
f (x) = 2 x + 1,\tag{5.34}
$$

$$
g (f) = f ^ {4},\tag{5.35}
$$

we obtain the derivatives of $f$ and g as 

$$
f ^ {\prime} (x) = 2,
$$

$$
g ^ {\prime} (f) = 4 f ^ {3},\tag{5.36}
$$

(5.37) 



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



such that the derivative of h is given as 

$$
h ^ {\prime} (x) = g ^ {\prime} (f) f ^ {\prime} (x) = \left(4 f ^ {3}\right) \cdot 2 \stackrel {(5. 3 4)} {=} 4 (2 x + 1) ^ {3} \cdot 2 = 8 (2 x + 1) ^ {3},\tag{5.38}
$$

where we used the chain rule (5.32) and substituted the definition of $f$ in (5.34) in $g ^ { \prime } ( f )$ 

## 5.2 Partial Differentiation and Gradients

Differentiation as discussed in Section 5.1 applies to functions f of a scalar variable $x \in \ \mathbb { R }$ . In the following, we consider the general case where the function $f$ depends on one or more variables $\pmb { x } \in \mathbb { R } ^ { n } , \mathbf { e . g . }$ $f ( \pmb { x } ) = f ( x _ { 1 } , x _ { 2 } )$ . The generalization of the derivative to functions of several variables is the gradient. 

We find the gradient of the function f with respect to x by varying one variable at a time and keeping the others constant. The gradient is then the collection of these partial derivatives. 

Definition 5.5 (Partial Derivative). For a function $f : \mathbb { R } ^ { n }  \mathbb { R } .$ x ↓ $f ( \pmb { x } ) , \pmb { x } \in \mathbb { R } ^ { n }$ of n variables $x _ { 1 } , \ldots , x _ { n }$ we define the partial derivatives as 

$$
\frac {\partial f}{\partial x _ {1}} = \lim _ {h \rightarrow 0} \frac {f (x _ {1} + h , x _ {2} , \dots , x _ {n}) - f (\boldsymbol {x})}{h}\tag{5.39}
$$

$$
\frac {\partial f}{\partial x _ {n}} = \lim _ {h \rightarrow 0} \frac {f (x _ {1} , \dots , x _ {n - 1} , x _ {n} + h) - f (\boldsymbol {x})}{h}
$$

and collect them in the row vector 

$$
\nabla_ {\boldsymbol {x}} f = \operatorname{grad} f = \frac {\mathrm{d} f}{\mathrm{d} \boldsymbol {x}} = \left[ \begin{array}{c c c c} \frac {\partial f (\boldsymbol {x})}{\partial x _ {1}} & \frac {\partial f (\boldsymbol {x})}{\partial x _ {2}} & \dots & \frac {\partial f (\boldsymbol {x})}{\partial x _ {n}} \end{array} \right] \in \mathbb {R} ^ {1 \times n},\tag{5.40}
$$

where n is the number of variables and 1 is the dimension of the image/ range/codomain of $f .$ . Here, we defined the column vector $\pmb { x } = [ x _ { 1 } , \ldots , x _ { n } ] ^ { \top }$ $\in \mathbb { R } ^ { n }$ . The row vector in (5.40) is called the gradient of f or the Jacobian and is the generalization of the derivative from Section 5.1. 

Remark. This definition of the Jacobian is a special case of the general definition of the Jacobian for vector-valued functions as the collection of partial derivatives. We will get back to this in Section 5.3. ◇ 

Example 5.6 (Partial Derivatives Using the Chain Rule) For $f ( x , y ) = ( x + 2 y ^ { 3 } ) ^ { 2 }$ , we obtain the partial derivatives 

$$
\frac {\partial f (x , y)}{\partial x} = 2 (x + 2 y ^ {3}) \frac {\partial}{\partial x} (x + 2 y ^ {3}) = 2 (x + 2 y ^ {3}),\tag{5.41}
$$

$$
\frac {\partial f (x , y)}{\partial y} = 2 \left(x + 2 y ^ {3}\right) \frac {\partial}{\partial y} \left(x + 2 y ^ {3}\right) = 1 2 \left(x + 2 y ^ {3}\right) y ^ {2}.\tag{5.42}
$$

where we used the chain rule (5.32) to compute the partial derivatives. 

Remark (Gradient as a Row Vector). It is not uncommon in the literature to define the gradient vector as a column vector, following the convention that vectors are generally column vectors. The reason why we define the gradient vector as a row vector is twofold: First, we can consistently generalize the gradient to vector-valued functions $f : \mathbb { R } ^ { n }  \mathbb { R } ^ { m }$ (then the gradient becomes a matrix). Second, we can immediately apply the multi-variate chain rule without paying attention to the dimension of the gradient. We will discuss both points in Section 5.3. 

Example 5.7 (Gradient) 

For $f ( x _ { 1 } , x _ { 2 } ) = x _ { 1 } ^ { 2 } x _ { 2 } + x _ { 1 } x _ { 2 } ^ { 3 } \in \mathbb { R } ,$ the partial derivatives $( \mathrm { i . e . }$ , the derivatives of $f$ with respect to x<sub>1</sub> and $x _ { 2 } )$ are 

$$
\frac {\partial f (x _ {1} , x _ {2})}{\partial x _ {1}} = 2 x _ {1} x _ {2} + x _ {2} ^ {3}\tag{5.43}
$$

$$
\frac {\partial f (x _ {1} , x _ {2})}{\partial x _ {2}} = x _ {1} ^ {2} + 3 x _ {1} x _ {2} ^ {2}\tag{5.44}
$$

and the gradient is then 

$$
\frac {\mathrm{d} f}{\mathrm{d} \boldsymbol {x}} = \left[ \begin{array}{c c} \frac {\partial f (x _ {1} , x _ {2})}{\partial x _ {1}} & \frac {\partial f (x _ {1} , x _ {2})}{\partial x _ {2}} \end{array} \right] = \left[ \begin{array}{c c} 2 x _ {1} x _ {2} + x _ {2} ^ {3} & x _ {1} ^ {2} + 3 x _ {1} x _ {2} ^ {2} \end{array} \right] \in \mathbb {R} ^ {1 \times 2}.\tag{5.45}
$$

## 5.2.1 Basic Rules of Partial Differentiation

Product rule: 

In the multivariate case, where $\mathbf { x } \in \mathbb { R } ^ { n }$ , the basic differentiation rules that we know from school (e.g., sum rule, product rule, chain rule; see also Section 5.1.2) still apply. However, when we compute derivatives with respect to vectors $\pmb { x } \in \mathbb { R } ^ { n }$ we need to pay attention: Our gradients now involve vectors and matrices, and matrix multiplication is not commutative (Section 2.2.1), i.e., the order matters. 

$$
(f g) ^ {\prime} = f ^ {\prime} g + f g ^ {\prime},
$$

Sum rule: 

$$
(f + g) ^ {\prime} = f ^ {\prime} + g ^ {\prime},
$$

Chain rule: 

$$
(g (f)) ^ {\prime} = g ^ {\prime} (f) f ^ {\prime}
$$

Here are the general product rule, sum rule, and chain rule: 

$$
\text { Product   rule: } \quad \frac {\partial}{\partial \boldsymbol {x}} (f (\boldsymbol {x}) g (\boldsymbol {x})) = \frac {\partial f}{\partial \boldsymbol {x}} g (\boldsymbol {x}) + f (\boldsymbol {x}) \frac {\partial g}{\partial \boldsymbol {x}}\tag{5.46}
$$

$$
\text {   Sum   rule:   } \quad \frac {\partial}{\partial \boldsymbol {x}} (f (\boldsymbol {x}) + g (\boldsymbol {x})) = \frac {\partial f}{\partial \boldsymbol {x}} + \frac {\partial g}{\partial \boldsymbol {x}}\tag{5.47}
$$

©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 

Chain rule: 

$$
\frac {\partial}{\partial \boldsymbol {x}} (g \circ f) (\boldsymbol {x}) = \frac {\partial}{\partial \boldsymbol {x}} \bigl (g (f (\boldsymbol {x})) \bigr) = \frac {\partial g}{\partial f} \frac {\partial f}{\partial \boldsymbol {x}}\tag{5.48}
$$

This is only an intuition, but not mathematically correct since the partial derivative is not a fraction. 

Let us have a closer look at the chain rule. The chain rule (5.48) resembles to some degree the rules for matrix multiplication where we said that neighboring dimensions have to match for matrix multiplication to be defined; see Section 2.2.1. If we go from left to right, the chain rule exhibits similar properties: $\partial f$ shows up in the “denominator” of the first factor and in the “numerator” of the second factor. If we multiply the factors together, multiplication is defined, i.e., the dimensions of $\partial f$ match, and $\partial f$ “cancels”, such that $\partial g / \partial x$ remains. 

## 5.2.2 Chain Rule

Consider a function $f : \mathbb { R } ^ { 2 } $ <sup>R</sup> of two variables $x _ { 1 } , x _ { 2 }$ . Furthermore, $x _ { 1 } ( t )$ and $x _ { 2 } ( t )$ are themselves functions of t. To compute the gradient of $f$ with respect to $t ,$ we need to apply the chain rule (5.48) for multivariate functions as 

$$
\frac {\mathrm{d} f}{\mathrm{d} t} = \left[ \begin{array}{c c} \frac {\partial f}{\partial x _ {1}} & \frac {\partial f}{\partial x _ {2}} \end{array} \right] \left[ \begin{array}{c} \frac {\partial x _ {1} (t)}{\partial t} \\ \frac {\partial x _ {2} (t)}{\partial t} \end{array} \right] = \frac {\partial f}{\partial x _ {1}} \frac {\partial x _ {1}}{\partial t} + \frac {\partial f}{\partial x _ {2}} \frac {\partial x _ {2}}{\partial t},\tag{5.49}
$$

where d denotes the gradient and $\partial$ partial derivatives. 

## Example 5.8

Consider $f ( x _ { 1 } , x _ { 2 } ) = x _ { 1 } ^ { 2 } + 2 x _ { 2 }$ , where x = sin t and $x _ { 2 } = \cos t ,$ , then 

$$
\frac {\mathrm{d} f}{f} = \frac {\partial f}{f} \frac {\partial x _ {1}}{\partial x _ {1}} + \frac {\partial f}{f} \frac {\partial x _ {2}}{\partial x _ {2}}
$$

$$
\overline {{\mathrm{d} t}} - \overline {{\partial x _ {1}}} \overline {{\partial t}} + \overline {{\partial x _ {2}}} \overline {{\partial t}}\tag{5.50a}
$$

$$
= 2 \sin t \frac {\partial \sin t}{\partial t} + 2 \frac {\partial \cos t}{\partial t}\tag{5.50b}
$$

$$
= 2 \sin t \cos t - 2 \sin t = 2 \sin t (\cos t - 1)\tag{5.50c}
$$

is the corresponding derivative of $f$ with respect to t. 

If $f ( x _ { 1 } , x _ { 2 } )$ is a function of $x _ { 1 }$ and $x _ { 2 } ,$ , where $x _ { 1 } ( s , t )$ and $x _ { 2 } ( s , t )$ are themselves functions of two variables s and $t ,$ the chain rule yields the partial derivatives 

$$
\frac {\partial f}{\partial s} = \frac {\partial f}{\partial x _ {1}} \frac {\partial x _ {1}}{\partial s} + \frac {\partial f}{\partial x _ {2}} \frac {\partial x _ {2}}{\partial s},\tag{5.51}
$$

$$
\frac {\partial f}{\partial t} = \frac {\partial f}{\partial x _ {1}} \frac {\partial x _ {1}}{\partial t} + \frac {\partial f}{\partial x _ {2}} \frac {\partial x _ {2}}{\partial t},\tag{5.52}
$$

Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 

## 5.3 Gradients of Vector-Valued Functions

and the gradient is obtained by the matrix multiplication 

$$
\frac {\mathrm{d} f}{\mathrm{d} (s , t)} = \frac {\partial f}{\partial \boldsymbol {x}} \frac {\partial \boldsymbol {x}}{\partial (s , t)} = \underbrace {\left[ \begin{array}{c c} \frac {\partial f}{\partial x _ {1}} & \frac {\partial f}{\partial x _ {2}} \end{array} \right]} _ {= \frac {\partial f}{\partial \boldsymbol {x}}} \underbrace {\left[ \begin{array}{c c} \frac {\partial x _ {1}}{\partial s} & \frac {\partial x _ {1}}{\partial t} \\ \frac {\partial x _ {2}}{\partial s} & \frac {\partial x _ {2}}{\partial t} \end{array} \right]} _ {= \frac {\partial \boldsymbol {x}}{\partial (s, t)}}.\tag{5.53}
$$

This compact way of writing the chain rule as a matrix multiplication only makes sense if the gradient is defined as a row vector. Otherwise, we will need to start transposing gradients for the matrix dimensions to match. This may still be straightforward as long as the gradient is a vector or a matrix; however, when the gradient becomes a tensor (we will discuss this in the following), the transpose is no longer a triviality. 

Remark (Verifying the Correctness of a Gradient Implementation). The definition of the partial derivatives as the limit of the corresponding difference quotient (see (5.39)) can be exploited when numerically checking the correctness of gradients in computer programs: When we compute gradients and implement them, we can use finite differences to numerically test our computation and implementation: We choose the value h to be small $( \mathsf { e } . \mathsf { g } . , h = 1 0 ^ { - 4 } )$ and compare the finite-difference approximation from (5.39) with our (analytic) implementation of the gradient. If the error is small, our gradient implementation is probably correct. “Small” could mean that $\sqrt { \frac { \sum _ { i } ( d h _ { i } - d f _ { i } ) ^ { 2 } } { \sum _ { i } ( d h _ { i } + d f _ { i } ) ^ { 2 } } } < 1 0 ^ { - 6 }$ , where $d h _ { i }$ is the finite-difference approximation and $d f _ { i }$ is the analytic gradient of $f$ with respect to the ith variable $x _ { i }$ $\diamondsuit$ 

## 5.3 Gradients of Vector-Valued Functions

Thus far, we discussed partial derivatives and gradients of functions $f ~ ;$ $\mathbb { R } ^ { n } \to$ <sup>R</sup> mapping to the real numbers. In the following, we will generalize the concept of the gradient to vector-valued functions (vector fields) $f$ : $\mathbb { R } ^ { n } \to \mathbb { R } ^ { m }$ , where $n \geqslant 1$ and $m > 1$ 

For a function $f : \mathbb { R } ^ { n }  \mathbb { R } ^ { m }$ and a vector $\pmb { x } = [ x _ { 1 } , \ldots , x _ { n } ] ^ { \top } \in \mathbb { R } ^ { n }$ , the corresponding vector of function values is given as 

$$
\boldsymbol {f} (\boldsymbol {x}) = \left[ \begin{array}{c} f _ {1} (\boldsymbol {x}) \\ \vdots \\ f _ {m} (\boldsymbol {x}) \end{array} \right] \in \mathbb {R} ^ {m}.\tag{5.54}
$$

Writing the vector-valued function in this way allows us to view a vectorvalued function $f : \mathbb { R } ^ { n }  \mathbb { R } ^ { m }$ as a vector of functions $[ f _ { 1 } , \ldots , f _ { m } ] ^ { \intercal }$ , $f _ { i } : \mathbb { R } ^ { n }  \mathbb { R }$ that map onto <sup>R</sup>. The differentiation rules for every $f _ { i }$ are exactly the ones we discussed in Section 5.2. 



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



Therefore, the partial derivative of a vector-valued function $f : \mathbb { R } ^ { n } $ $\mathbb { R } ^ { m }$ with respect to $x _ { i } \in \mathbb { R } , i = 1 , \dots n ,$ , is given as the vector 

$$
\frac {\partial \boldsymbol {f}}{\partial x _ {i}} = \left[ \begin{array}{c} \frac {\partial f _ {1}}{\partial x _ {i}} \\ \vdots \\ \frac {\partial f _ {m}}{\partial x _ {i}} \end{array} \right] = \left[ \begin{array}{c} \lim _ {h \to 0} \frac {f _ {1} (x _ {1} , \ldots , x _ {i - 1} , x _ {i} + h , x _ {i + 1} , \ldots x _ {n}) - f _ {1} (\boldsymbol {x})}{h} \\ \vdots \\ \lim _ {h \to 0} \frac {f _ {m} (x _ {1} , \ldots , x _ {i - 1} , x _ {i} + h , x _ {i + 1} , \ldots x _ {n}) - f _ {m} (\boldsymbol {x})}{h} \end{array} \right] \in \mathbb {R} ^ {m}.\tag{5.55}
$$

From (5.40), we know that the gradient of f with respect to a vector is the row vector of the partial derivatives. In (5.55), every partial derivative $\partial f / \partial x _ { i }$ is itself a column vector. Therefore, we obtain the gradient of $f$ : $\mathbb { R } ^ { n } \to \mathbb { R } ^ { m }$ with respect to x $\mathbf { \Psi } \in \mathbb { R } ^ { n }$ by collecting these partial derivatives: 

(5.56a) 

(5.56b) 

Definition 5.6 (Jacobian). The collection of all first-order partial derivatives of a vector-valued function $f : \mathbb { R } ^ { n }  \mathbb { R } ^ { m }$ is called the Jacobian. The Jacobian J is an $m \times n$ matrix, which we define and arrange as follows: 

(5.57) 

$$
\begin{array}{l} \boldsymbol {J} = \nabla_ {\boldsymbol {x}} \boldsymbol {f} = \frac {\mathrm{d} \boldsymbol {f} (\boldsymbol {x})}{\mathrm{d} \boldsymbol {x}} = \left[ \begin{array}{c c c} \frac {\partial \boldsymbol {f} (\boldsymbol {x})}{\partial x _ {1}} & \dots & \frac {\partial \boldsymbol {f} (\boldsymbol {x})}{\partial x _ {n}} \end{array} \right] \\ = \left[ \begin{array}{c c c} \frac {\partial f _ {1} (\boldsymbol {x})}{\partial x _ {1}} & \dots & \frac {\partial f _ {1} (\boldsymbol {x})}{\partial x _ {n}} \\ \vdots & & \vdots \\ \frac {\partial f _ {m} (\boldsymbol {x})}{\partial x _ {1}} & \dots & \frac {\partial f _ {m} (\boldsymbol {x})}{\partial x _ {n}} \end{array} \right], \\ \boldsymbol {x} = \left[ \begin{array}{c} x _ {1} \\ \vdots \\ x _ {n} \end{array} \right], J (i, j) = \frac {\partial f _ {i}}{\partial x _ {j}}. \end{array}\tag{5.58}
$$

(5.59) 

As a special case of (5.58), a function $f : \mathbb { R } ^ { n } \to \mathbb { R } ^ { 1 }$ , which maps a vector $\pmb { x } \in \mathbb { R } ^ { n }$ onto a scalar $\begin{array} { r } { ( \mathbf { e } . \mathbf { g } . , f ( \pmb { x } ) = \sum _ { i = 1 } ^ { n } x _ { i } ) } \end{array}$ , possesses a Jacobian that is a row vector (matrix of dimension $1 \times n )$ ; see (5.40). 

Remark. In this book, we use the numerator layout of the derivative, i.e., the derivative $\mathrm { d } f / \mathrm { d } x$ of $\pmb { f } \in \mathbb { R } ^ { m }$ with respect to $\pmb { x } \in \mathbb { R } ^ { n }$ is an $m \ \times$ n matrix, where the elements of $f$ define the rows and the elements of x define the columns of the corresponding Jacobian; see (5.58). There exists also the denominator layout, which is the transpose of the numerator layout. In this book, we will use the numerator layout. ◇ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/f74b3de8c535dc9490389c10f9fb8d11bee42a37263be8067aba6ce533157717.jpg)



Figure 5.5 The determinant of the Jacobian of f can be used to compute the magnifier between the blue and orange area.


denominator layout 

We will see how the Jacobian is used in the change-of-variable method for probability distributions in Section 6.7. The amount of scaling due to the transformation of a variable is provided by the determinant. 

In Section 4.1, we saw that the determinant can be used to compute the area of a parallelogram. If we are given two vectors $ { b _ { 1 } } ~ = ~ [ 1 , 0 ] ^ { \top }$ ， $\pmb { b } _ { 2 } = [ 0 , 1 ] ^ { \top }$ as the sides of the unit square (blue; see Figure 5.5), the area of this square is 

$$
\left| \det \left(\left[ \begin{array}{c c} 1 & 0 \\ 0 & 1 \end{array} \right]\right) \right| = 1.\tag{5.60}
$$

If we take a parallelogram with the sides $\mathbf { { c } } _ { 1 } = [ - 2 , 1 ] ^ { \top } , \mathbf { { c } } _ { 2 } = [ 1 , 1 ] ^ { \top }$ (orange in Figure 5.5), its area is given as the absolute value of the determinant (see Section 4.1) 

$$
\left| \det \left(\left[ \begin{array}{c c} - 2 & 1 \\ 1 & 1 \end{array} \right]\right) \right| = | - 3 | = 3  ,\tag{5.61}
$$

i.e., the area of this is exactly three times the area of the unit square. We can find this scaling factor by finding a mapping that transforms the unit square into the other square. In linear algebra terms, we effectively perform a variable transformation from $( b _ { 1 } , b _ { 2 } )$ to $( c _ { 1 } , c _ { 2 } )$ . In our case, the mapping is linear and the absolute value of the determinant of this mapping gives us exactly the scaling factor we are looking for. 

We will describe two approaches to identify this mapping. First, we exploit that the mapping is linear so that we can use the tools from Chapter 2 to identify this mapping. Second, we will find the mapping using partial derivatives using the tools we have been discussing in this chapter. 

Approach 1 To get started with the linear algebra approach, we identify both $\{ b _ { 1 } , b _ { 2 } \}$ and $\{ c _ { 1 } , c _ { 2 } \}$ as bases of $\mathbb { R } ^ { 2 }$ (see Section 2.6.1 for a recap). What we effectively perform is a change of basis from $( b _ { 1 } , b _ { 2 } )$ to $( c _ { 1 } , c _ { 2 } )$ , and we are looking for the transformation matrix that implements the basis change. Using results from Section 2.7.2, we identify the desired basis change matrix as 

$$
\boldsymbol {J} = \left[ \begin{array}{c c} - 2 & 1 \\ 1 & 1 \end{array} \right],\tag{5.62}
$$

such that $J b _ { 1 } = c _ { 1 }$ and $J b _ { 2 } = c _ { 2 }$ . The absolute value of the determinant of $J ,$ , which yields the scaling factor we are looking for, is given as $| \mathrm { d e t } ( J ) | = 3 .$ , i.e., the area of the square spanned by $( c _ { 1 } , c _ { 2 } )$ is three times greater than the area spanned by $( b _ { 1 } , b _ { 2 } )$ 

Approach 2 The linear algebra approach works for linear transformations; for nonlinear transformations (which become relevant in Section 6.7), we follow a more general approach using partial derivatives. 

For this approach, we consider a function $f : \mathbb { R } ^ { 2 } \to \mathbb { R } ^ { 2 }$ that performs a variable transformation. In our example, f maps the coordinate representation of any vector $\pmb { x } \in \mathbb { R } ^ { 2 }$ with respect to $( b _ { 1 } , b _ { 2 } )$ onto the coordinate representation $\boldsymbol { y } \in \mathbb { R } ^ { 2 }$ with respect to $( c _ { 1 } , c _ { 2 } )$ . We want to identify the mapping so that we can compute how an area (or volume) changes when it is being transformed by $f .$ . For this, we need to find out how ${ \pmb f } ( { \pmb x } )$ changes if we modify x a bit. This question is exactly answered by the Jacobian matrix $\frac { \mathrm { d } f } { \mathrm { d } x } \in \mathbb { R } ^ { 2 \times 2 }$ . Since we can write 

$$
y _ {1} = - 2 x _ {1} + x _ {2}\tag{5.63}
$$

$$
y _ {2} = x _ {1} + x _ {2}\tag{5.64}
$$

we obtain the functional relationship between x and $^ { y , }$ which allows us to get the partial derivatives 

$$
\frac {\partial y _ {1}}{\partial x _ {1}} = - 2, \quad \frac {\partial y _ {1}}{\partial x _ {2}} = 1, \quad \frac {\partial y _ {2}}{\partial x _ {1}} = 1, \quad \frac {\partial y _ {2}}{\partial x _ {2}} = 1\tag{5.65}
$$

and compose the Jacobian as 

$$
\boldsymbol {J} = \left[ \begin{array}{c c} \frac {\partial y _ {1}}{\partial x _ {1}} & \frac {\partial y _ {1}}{\partial x _ {2}} \\ \frac {\partial y _ {2}}{\partial x _ {1}} & \frac {\partial y _ {2}}{\partial x _ {2}} \end{array} \right] = \left[ \begin{array}{c c} - 2 & 1 \\ 1 & 1 \end{array} \right].\tag{5.66}
$$

The Jacobian represents the coordinate transformation we are looking for. It is exact if the coordinate transformation is linear (as in our case), and (5.66) recovers exactly the basis change matrix in (5.62). If the coordinate transformation is nonlinear, the Jacobian approximates this nonlinear transformation locally with a linear one. The absolute value of the Jacobian determinant $| \mathrm { d e t } ( J ) |$ is the factor by which areas or volumes are scaled when coordinates are transformed. Our case yields $| \mathrm { d e t } ( J ) | = 3$ 

Figure 5.6 Dimensionality of (partial) derivatives. 

The Jacobian determinant and variable transformations will become relevant in Section 6.7 when we transform random variables and probability distributions. These transformations are extremely relevant in machine learning in the context of training deep neural networks using the reparametrization trick, also called infinite perturbation analysis. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/56728b1d455f9b883d7ae87bd0e7ff983a7f74463eef5a89e559f5253e8bce68.jpg)


In this chapter, we encountered derivatives of functions. Figure 5.6 summarizes the dimensions of those derivatives. If $f : \mathbb { R } \to$ <sup>R</sup> the gradient is simply a scalar (top-left entry). For $f : \mathbb { R } ^ { D } $ <sup>R</sup> the gradient is a $1 \times D$ row vector (top-right entry). For $f : \mathbb { R } \to \mathbb { R } ^ { E }$ , the gradient is an $E \times 1$ column vector, and for $f : \mathbb { R } ^ { D }  \mathbb { R } ^ { E }$ the gradient is an $E \times D$ matrix. 

## Example 5.9 (Gradient of a Vector-Valued Function)

We are given 

$$
\boldsymbol {f} (\boldsymbol {x}) = \boldsymbol {A} \boldsymbol {x}, \quad \boldsymbol {f} (\boldsymbol {x}) \in \mathbb {R} ^ {M}, \quad \boldsymbol {A} \in \mathbb {R} ^ {M \times N}, \quad \boldsymbol {x} \in \mathbb {R} ^ {N}.
$$

To compute the gradient $\mathrm { d } f / \mathrm { d } x$ we first determine the dimension of $\mathrm { d } f / \mathrm { d } x \mathrm { : }$ Since $f : \mathbb { R } ^ { N } \to \mathbb { R } ^ { M }$ , it follows that $\mathrm { d } f / \mathrm { d } x \in \mathbb { R } ^ { M \times N }$ . Second, to compute the gradient we determine the partial derivatives of $f$ with respect to every $x _ { j }$ 

$$
f _ {i} (\boldsymbol {x}) = \sum_ {j = 1} ^ {N} A _ {i j} x _ {j} \implies \frac {\partial f _ {i}}{\partial x _ {j}} = A _ {i j}\tag{5.67}
$$

We collect the partial derivatives in the Jacobian and obtain the gradient 

$$
\frac {\mathrm{d} \boldsymbol {f}}{\mathrm{d} \boldsymbol {x}} = \left[ \begin{array}{c c c} \frac {\partial f _ {1}}{\partial x _ {1}} & \dots & \frac {\partial f _ {1}}{\partial x _ {N}} \\ \vdots & & \vdots \\ \frac {\partial f _ {M}}{\partial x _ {1}} & \dots & \frac {\partial f _ {M}}{\partial x _ {N}} \end{array} \right] = \left[ \begin{array}{c c c} A _ {1 1} & \dots & A _ {1 N} \\ \vdots & & \vdots \\ A _ {M 1} & \dots & A _ {M N} \end{array} \right] = \boldsymbol {A} \in \mathbb {R} ^ {M \times N}\tag{5.68}
$$

Example 5.10 (Chain Rule) 

Consider the function $h : \mathbb { R } \to \mathbb { R } , h ( t ) = ( f \circ g ) ( t )$ with 

$$
f: \mathbb {R} ^ {2} \to \mathbb {R}\tag{5.69}
$$

$$
g: \mathbb {R} \to \mathbb {R} ^ {2}\tag{5.70}
$$

$$
f (\pmb {x}) = \exp (x _ {1} x _ {2} ^ {2}),\tag{5.71}
$$

$$
\boldsymbol {x} = \left[ \begin{array}{c} x _ {1} \\ x _ {2} \end{array} \right] = g (t) = \left[ \begin{array}{c} t \cos t \\ t \sin t \end{array} \right]\tag{5.72}
$$

and compute the gradient of $h$ with respect to t. Since $f : \mathbb { R } ^ { 2 } \to \mathbb { F }$ and $g : \mathbb { R } \to \mathbb { R } ^ { 2 }$ we note that 

$$
\frac {\partial f}{\partial \boldsymbol {x}} \in \mathbb {R} ^ {1 \times 2}, \quad \frac {\partial g}{\partial t} \in \mathbb {R} ^ {2 \times 1}.\tag{5.73}
$$

The desired gradient is computed by applying the chain rule: 

$$
\frac {\mathrm{d} h}{\mathrm{d} t} = \frac {\partial f}{\partial \boldsymbol {x}} \frac {\partial \boldsymbol {x}}{\partial t} = \left[ \begin{array}{c c} \frac {\partial f}{\partial x _ {1}} & \frac {\partial f}{\partial x _ {2}} \end{array} \right] \left[ \begin{array}{c} \frac {\partial x _ {1}}{\partial t} \\ \frac {\partial x _ {2}}{\partial t} \end{array} \right]\tag{5.74a}
$$

$$
= \left[ \begin{array}{c c} \exp (x _ {1} x _ {2} ^ {2}) x _ {2} ^ {2} & 2 \exp (x _ {1} x _ {2} ^ {2}) x _ {1} x _ {2} \end{array} \right] \left[ \begin{array}{c} \cos t - t \sin t \\ \sin t + t \cos t \end{array} \right]\tag{5.74b}
$$

$$
= \exp (x _ {1} x _ {2} ^ {2}) \big (x _ {2} ^ {2} (\cos t - t \sin t) + 2 x _ {1} x _ {2} (\sin t + t \cos t) \big),\tag{5.74c}
$$

where $x _ { 1 } = t \cos t$ and $x _ { 2 } = t$ sin $t ;$ see (5.72). 



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



We will discuss this model in much more detail in Chapter 9 in the context of linear regression, where we need derivatives of the least-squares loss L with respect to the parameters θ. 

Example 5.11 (Gradient of a Least-Squares Loss in a Linear Model) Let us consider the linear model 

$$
\boldsymbol {y} = \boldsymbol {\Phi} \boldsymbol {\theta},\tag{5.75}
$$

where $\pmb { \theta } \in \mathbb { R } ^ { D }$ is a parameter vector, $\Phi \in \mathbb { R } ^ { N \times D }$ are input features and $\boldsymbol { y } \in \mathbb { R } ^ { N }$ are the corresponding observations. We define the functions 

$$
L (\boldsymbol {e}) := \left\| \boldsymbol {e} \right\| ^ {2},\tag{5.76}
$$

$$
e (\boldsymbol {\theta}) := \boldsymbol {y} - \Phi \boldsymbol {\theta}.
$$

least-squares loss 

(5.77) 

We seek $\frac { \partial L } { \partial \pmb { \theta } }$ , and we will use the chain rule for this purpose. L is called a least-squares loss function. 

Before we start our calculation, we determine the dimensionality of the gradient as 

$$
\frac {\partial L}{\partial \boldsymbol {\theta}} \in \mathbb {R} ^ {1 \times D}.\tag{5.78}
$$

The chain rule allows us to compute the gradient as 

$$
\frac {\partial L}{\partial \pmb {\theta}} = \frac {\partial L}{\partial \pmb {e}} \frac {\partial \pmb {e}}{\partial \pmb {\theta}},\tag{5.79}
$$

dLdtheta = np.einsum( ’n,nd’, dLde,dedtheta) 

where the dth element is given by 

$$
\frac {\partial L}{\partial \boldsymbol {\theta}} [ 1, d ] = \sum_ {n = 1} ^ {N} \frac {\partial L}{\partial \boldsymbol {e}} [ n ] \frac {\partial \boldsymbol {e}}{\partial \boldsymbol {\theta}} [ n, d ].\tag{5.80}
$$

We know that $\lVert e \rVert ^ { 2 } = e ^ { \top } e$ (see Section 3.2) and determine 

$$
\frac {\partial L}{\partial \boldsymbol {e}} = 2 \boldsymbol {e} ^ {\top} \in \mathbb {R} ^ {1 \times N}.\tag{5.81}
$$

Furthermore, we obtain 

$$
\frac {\partial e}{\partial \theta} = - \Phi \in \mathbb {R} ^ {N \times D},\tag{5.82}
$$

such that our desired derivative is 

$$
\frac {\partial L}{\partial \boldsymbol {\theta}} = - 2 \boldsymbol {e} ^ {\top} \boldsymbol {\Phi} \stackrel {(5. 7 7)} {=} - \underbrace {2 (\boldsymbol {y} ^ {\top} - \boldsymbol {\theta} ^ {\top} \boldsymbol {\Phi} ^ {\top})} _ {1 \times N} \underbrace {\boldsymbol {\Phi}} _ {N \times D} \in \mathbb {R} ^ {1 \times D}.\tag{5.83}
$$

Remark. We would have obtained the same result without using the chain rule by immediately looking at the function 

$$
L _ {2} (\boldsymbol {\theta}) := \left\| \boldsymbol {y} - \boldsymbol {\Phi} \boldsymbol {\theta} \right\| ^ {2} = \left(\boldsymbol {y} - \boldsymbol {\Phi} \boldsymbol {\theta}\right) ^ {\top} \left(\boldsymbol {y} - \boldsymbol {\Phi} \boldsymbol {\theta}\right).\tag{5.84}
$$

This approach is still practical for simple functions like $L _ { 2 }$ but becomes impractical for deep function compositions. ◇ 

## 5.4 Gradients of Matrices

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/f84ae6a4b80a28cc0c9923cf7a73cdc7ab697e19a91bd2e0d49d642e4cd18448.jpg)



Partial derivatives:


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/bd83721ceed4427351aea4ae715ecfba0a3b092c8269641bbe25eca99febaf1c.jpg)



(a) Approach 1: We compute the partial derivative $\frac { \partial A } { \partial x _ { 1 } } , \frac { \partial A } { \partial x _ { 2 } } , \frac { \partial A } { \partial x _ { 3 } }$ , each of which is $\mathrm { ~ a ~ 4 ~ } \times \mathrm { ~ 2 ~ }$ matrix, and col late them in $\mathbb { 1 } \mathbb { a } \mathbb { 4 } \times 2 \mathbb { \times } 3$ tensor.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/58d7c92ed98cc84112acd8aad806fdcd375299d254a715b4224cf39a6e733ba0.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/53e823247cfc403fd2da5b6da40310b15addcfc1475e026168c6d1c0a360cca7.jpg)



(b) Approach 2: We re-shape (flatten) $\pmb { A } \in \mathbb { R } ^ { 4 \times 2 }$ into a vector $\tilde { \boldsymbol { A } } \in \mathbb { R } ^ { 8 }$ . Then, we compute the gradient $\frac { \mathrm { d } \tilde { A } } { \mathrm { d } x } \in \mathbb { R } ^ { 8 \times 3 }$ We obtain the gradient tensor by re-shaping this gradient as illustrated above.


## 5.4 Gradients of Matrices

We will encounter situations where we need to take gradients of matrices with respect to vectors (or other matrices), which results in a multidimensional tensor. We can think of this tensor as a multidimensional array that 

Figure 5.7 Visualization of gradient computation of a matrix with respect to a vector. We are interested in computing the gradient of $\pmb { A } \in \mathbb { R } ^ { 4 \times 2 }$ with respect to a vector x $\in \mathbb { R } ^ { 3 }$ . We know that gradient $\frac { \mathrm { d } \boldsymbol { A } } { \mathrm { d } \boldsymbol { r } } \bar { \in } \mathrm { I R } ^ { 4 \times 2 \times 3 } .$ . We follow two equivalent approaches to arrive there: (a) collating partial derivatives into a Jacobian tensor; (b) flattening of the matrix into a vector, computing the Jacobian matrix, re-shaping into a Jacobian tensor. 

We can think of a tensor as a multidimensional array. 

collects partial derivatives. For example, if we compute the gradient of an $m \times n$ matrix A with respect to a $p \times q$ matrix $B ,$ , the resulting Jacobian would be $( m \times n ) \times ( p \times q )$ , i.e., a four-dimensional tensor $J ,$ whose entries are given as $J _ { i j k l } = \partial A _ { i j } / \partial B _ { k l }$ 

Since matrices represent linear mappings, we can exploit the fact that there is a vector-space isomorphism (linear, invertible mapping) between the space $\mathbb { R } ^ { m \times n }$ of $m \times n$ matrices and the space $\mathbb { R } ^ { m n }$ of mn vectors. Therefore, we can re-shape our matrices into vectors of lengths mn and $p q .$ , respectively. The gradient using these mn vectors results in a Jacobian of size mn $\times \ p q .$ . Figure 5.7 visualizes both approaches. In practical applications, it is often desirable to re-shape the matrix into a vector and continue working with this Jacobian matrix: The chain rule $_ { ( 5 . 4 8 ) }$ boils down to simple matrix multiplication, whereas in the case of a Jacobian tensor, we will need to pay more attention to what dimensions we need to sum out. 

Example 5.12 (Gradient of Vectors with Respect to Matrices) Let us consider the following example, where 

$$
\boldsymbol {f} = \boldsymbol {A} \boldsymbol {x}, \quad \boldsymbol {f} \in \mathbb {R} ^ {M}, \quad \boldsymbol {A} \in \mathbb {R} ^ {M \times N}, \quad \boldsymbol {x} \in \mathbb {R} ^ {N}\tag{5.85}
$$

and where we seek the gradient d $1 f / \mathrm { d } A$ . Let us start again by determining the dimension of the gradient as 

$$
\frac {\mathrm{d} \boldsymbol {f}}{\mathrm{d} \boldsymbol {A}} \in \mathbb {R} ^ {M \times (M \times N)}.\tag{5.86}
$$

By definition, the gradient is the collection of the partial derivatives: 

$$
\frac {\mathrm{d} \boldsymbol {f}}{\mathrm{d} \boldsymbol {A}} = \left[ \begin{array}{c} \frac {\partial f _ {1}}{\partial \boldsymbol {A}} \\ \vdots \\ \frac {\partial f _ {M}}{\partial \boldsymbol {A}} \end{array} \right], \quad \frac {\partial f _ {i}}{\partial \boldsymbol {A}} \in \mathbb {R} ^ {1 \times (M \times N)}.\tag{5.87}
$$

To compute the partial derivatives, it will be helpful to explicitly write out the matrix vector multiplication: 

$$
f _ {i} = \sum_ {j = 1} ^ {N} A _ {i j} x _ {j}, \quad i = 1, \dots , M,\tag{5.88}
$$

and the partial derivatives are then given as 

$$
\frac {\partial f _ {i}}{\partial A _ {i q}} = x _ {q}.\tag{5.89}
$$

This allows us to compute the partial derivatives of $f _ { i }$ with respect to a row of A, which is given as 

$$
\frac {\partial f _ {i}}{\partial A _ {i , :}} = \boldsymbol {x} ^ {\top} \in \mathbb {R} ^ {1 \times 1 \times N},\tag{5.90}
$$



Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 



$$
\frac {\partial f _ {i}}{\partial A _ {k \neq i , :}} = \mathbf {0} ^ {\top} \in \mathbb {R} ^ {1 \times 1 \times N}\tag{5.91}
$$

where we have to pay attention to the correct dimensionality. Since $f _ { i }$ maps onto <sup>R</sup> and each row of A is of size $1 \times N$ , we obtain a $1 \times 1 \times N .$ sized tensor as the partial derivative of $f _ { i }$ with respect to a row of A. 

We stack the partial derivatives (5.91) and get the desired gradient in (5.87) via 

$$
\frac {\partial f _ {i}}{\partial \boldsymbol {A}} = \left[ \begin{array}{c} \boldsymbol {0} ^ {\top} \\ \vdots \\ \boldsymbol {0} ^ {\top} \\ \boldsymbol {x} ^ {\top} \\ \boldsymbol {0} ^ {\top} \\ \vdots \\ \boldsymbol {0} ^ {\top} \end{array} \right] \in \mathbb {R} ^ {1 \times (M \times N)}.\tag{5.92}
$$

Example 5.13 (Gradient of Matrices with Respect to Matrices) Consider a matrix $\pmb { R } \in \mathbb { R } ^ { M \times N }$ and $\pmb { f } : \mathbb { R } ^ { M \times N } \overset { \textstyle ^ { \cdot } } { \to } \mathbb { R } ^ { N \times N }$ with 

$$
\boldsymbol {f} (\boldsymbol {R}) = \boldsymbol {R} ^ {\top} \boldsymbol {R} =: \boldsymbol {K} \in \mathbb {R} ^ {N \times N},\tag{5.93}
$$

where we seek the gradient d $\scriptstyle K / \mathrm { d } R$ 

To solve this hard problem, let us first write down what we already know: The gradient has the dimensions 

$$
\frac {\mathrm{d} \boldsymbol {K}}{\mathrm{d} \boldsymbol {R}} \in \mathbb {R} ^ {(N \times N) \times (M \times N)},\tag{5.94}
$$

which is a tensor. Moreover, 

$$
\frac {\mathrm{d} K _ {p q}}{\mathrm{d} \boldsymbol {R}} \in \mathbb {R} ^ {1 \times M \times N}\tag{5.95}
$$

for $p , q = 1 , \ldots , N$ , where $K _ { p q }$ is the $( p , q )$ th entry of $K = f ( R )$ . Denoting the ith column of R by $\mathbf { \nabla } r _ { i } .$ every entry of $\kappa$ is given by the dot product of two columns of R, i.e., 

$$
K _ {p q} = \boldsymbol {r} _ {p} ^ {\top} \boldsymbol {r} _ {q} = \sum_ {m = 1} ^ {M} R _ {m p} R _ {m q}.\tag{5.96}
$$

When we now compute the partial derivative $\frac { \partial K _ { p q } } { \partial R _ { i j } }$ we obtain 

$$
\frac {\partial K _ {p q}}{\partial R _ {i j}} = \sum_ {m = 1} ^ {M} \frac {\partial}{\partial R _ {i j}} R _ {m p} R _ {m q} = \partial_ {p q i j},\tag{5.97}
$$



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



$$
\partial_ {p q i j} = \left\{ \begin{array}{l l} R _ {i q} & \text {if} j = p, p \neq q \\ R _ {i p} & \text {if} j = q, p \neq q \\ 2 R _ {i q} & \text {if} j = p, p = q \\ 0 & \text {otherwise} \end{array} \right..\tag{5.98}
$$

From (5.94), we know that the desired gradient has the dimension $( N \times$ $N ) \times ( M \times N )$ , and every single entry of this tensor is given by $\partial _ { p q i j }$ in (5.98), where $p , q , j = 1 , \ldots , N$ and $i = 1 , \ldots , M$ 

## 5.5 Useful Identities for Computing Gradients

In the following, we list some useful gradients that are frequently required in a machine learning context (Petersen and Pedersen, 2012). Here, we use $\operatorname { t r } ( \cdot )$ as the trace (see Definition 4.4), det( ) as the determinant (see Section 4.1) and $f ( \pmb { X } ) ^ { - 1 }$ as the inverse of $f ( X )$ , assuming it exists. 

$$
\frac {\partial}{\partial \boldsymbol {X}} \boldsymbol {f} (\boldsymbol {X}) ^ {\top} = \left(\frac {\partial \boldsymbol {f} (\boldsymbol {X})}{\partial \boldsymbol {X}}\right) ^ {\top}\tag{5.99}
$$

$$
\frac {\partial}{\partial \boldsymbol {X}} \operatorname{tr} (\boldsymbol {f} (\boldsymbol {X})) = \operatorname{tr} \left(\frac {\partial \boldsymbol {f} (\boldsymbol {X})}{\partial \boldsymbol {X}}\right)\tag{5.100}
$$

$$
\frac {\partial}{\partial \boldsymbol {X}} \det (\boldsymbol {f} (\boldsymbol {X})) = \det (\boldsymbol {f} (\boldsymbol {X})) \operatorname{tr} \left(\boldsymbol {f} (\boldsymbol {X}) ^ {- 1} \frac {\partial \boldsymbol {f} (\boldsymbol {X})}{\partial \boldsymbol {X}}\right)\tag{5.101}
$$

$$
\frac {\partial}{\partial \boldsymbol {X}} \boldsymbol {f} (\boldsymbol {X}) ^ {- 1} = - \boldsymbol {f} (\boldsymbol {X}) ^ {- 1} \frac {\partial \boldsymbol {f} (\boldsymbol {X})}{\partial \boldsymbol {X}} \boldsymbol {f} (\boldsymbol {X}) ^ {- 1}\tag{5.102}
$$

$$
\frac {\partial \boldsymbol {a} ^ {\top} \boldsymbol {X} ^ {- 1} \boldsymbol {b}}{\partial \boldsymbol {X}} = - (\boldsymbol {X} ^ {- 1}) ^ {\top} \boldsymbol {a b} ^ {\top} (\boldsymbol {X} ^ {- 1}) ^ {\top}\tag{5.103}
$$

$$
\frac {\partial \boldsymbol {x} ^ {\top} \boldsymbol {a}}{\partial \boldsymbol {x}} = \boldsymbol {a} ^ {\top}\tag{5.104}
$$

$$
\frac {\partial \boldsymbol {a} ^ {\top} \boldsymbol {x}}{\partial \boldsymbol {x}} = \boldsymbol {a} ^ {\top}\tag{5.105}
$$

$$
\frac {\partial \boldsymbol {a} ^ {\top} \boldsymbol {X} \boldsymbol {b}}{\partial \boldsymbol {X}} = \boldsymbol {a} \boldsymbol {b} ^ {\top}\tag{5.106}
$$

$$
\frac {\partial \boldsymbol {x} ^ {\top} \boldsymbol {B} \boldsymbol {x}}{\partial \boldsymbol {x}} = \boldsymbol {x} ^ {\top} (\boldsymbol {B} + \boldsymbol {B} ^ {\top})\tag{5.107}
$$

$$
\frac {\partial}{\partial \boldsymbol {s}} (\boldsymbol {x} - \boldsymbol {A} \boldsymbol {s}) ^ {\top} \boldsymbol {W} (\boldsymbol {x} - \boldsymbol {A} \boldsymbol {s}) = - 2 (\boldsymbol {x} - \boldsymbol {A} \boldsymbol {s}) ^ {\top} \boldsymbol {W} \boldsymbol {A}\tag{5.108}
$$

Remark. In this book, we only cover traces and transposes of matrices. However, we have seen that derivatives can be higher-dimensional tensors, in which case the usual trace and transpose are not defined. In these cases, the trace of a $D \times D \times E \times F$ tensor would be an $E \times F _ { - }$ dimensional matrix. This is a special case of a tensor contraction. Similarly, when we “transpose” a tensor, we mean swapping the first two dimensions. Specifically, in (5.99) through (5.102), we require tensor-related computations when we work with multivariate functions $f ( \cdot )$ and compute derivatives with respect to matrices (and choose not to vectorize them as discussed in Section 5.4). 

## 5.6 Backpropagation and Automatic Differentiation

In many machine learning applications, we find good model parameters by performing gradient descent (Section 7.1), which relies on the fact that we can compute the gradient of a learning objective with respect to the parameters of the model. For a given objective function, we can obtain the gradient with respect to the model parameters using calculus and applying the chain rule; see Section 5.2.2. We already had a taste in Section 5.3 when we looked at the gradient of a squared loss with respect to the parameters of a linear regression model. 

Consider the function 

$$
f (x) = \sqrt {x ^ {2} + \exp (x ^ {2})} + \cos \left(x ^ {2} + \exp (x ^ {2})\right).\tag{5.109}
$$

By application of the chain rule, and noting that differentiation is linear, we compute the gradient 

$$
\begin{array}{l} \frac {\mathrm{d} f}{\mathrm{d} x} = \frac {2 x + 2 x \exp (x ^ {2})}{2 \sqrt {x ^ {2} + \exp (x ^ {2})}} - \sin \left(x ^ {2} + \exp (x ^ {2})\right) \left(2 x + 2 x \exp (x ^ {2})\right) \\ = 2 x \left(\frac {1}{2 \sqrt {x ^ {2} + \exp (x ^ {2})}} - \sin \left(x ^ {2} + \exp (x ^ {2})\right)\right) \left(1 + \exp (x ^ {2})\right). \end{array}\tag{5.110}
$$

Writing out the gradient in this explicit way is often impractical since it often results in a very lengthy expression for a derivative. In practice, it means that, if we are not careful, the implementation of the gradient could be significantly more expensive than computing the function, which imposes unnecessary overhead. For training deep neural network models, the backpropagation algorithm (Kelley, 1960; Bryson, 1961; Dreyfus, 1962; Rumelhart et al., 1986) is an efficient way to compute the gradient of an error function with respect to the parameters of the model. 

## 5.6.1 Gradients in a Deep Network

An area where the chain rule is used to an extreme is deep learning, where the function value y is computed as a many-level function composition 

$$
\boldsymbol {y} = \left(f _ {K} \circ f _ {K - 1} \circ \dots \circ f _ {1}\right) (\boldsymbol {x}) = f _ {K} \left(f _ {K - 1} \left(\dots \left(f _ {1} (\boldsymbol {x})\right) \dots\right)\right),\tag{5.111}
$$

where x are the inputs $( \boldsymbol { \mathrm { e . g . } }$ , images), y are the observations (e.g., class labels), and every function $f _ { i } , i = 1 , \dots , K$ , possesses its own parameters. 

Figure 5.8 Forward pass in a multi-layer neural network to compute the loss L as a function of the inputs x and the parameters $A _ { i } , ~ b _ { i } .$ 

We discuss the case, where the activation functions are identical in each layer to unclutter notation. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/2745e5c35ce59fea0fbebc04cb48461aa690a192aafb7f29aaa276cd5a3da7fe.jpg)


In neural networks with multiple layers, we have functions $f _ { i } ( { \pmb x } _ { i - 1 } ) =$ $\sigma ( A _ { i - 1 } { \pmb x } _ { i - 1 } + { \pmb b } _ { i - 1 } )$ in the ith layer. Here $\mathbf { \delta _ { x - 1 } }$ is the output of layer $i - 1$ and σ an activation function, such as the logistic sigmoid $\frac { 1 } { 1 + e ^ { - x } } .$ , tanh or a rectified linear unit (ReLU). In order to train these models, we require the gradient of a loss function L with respect to all model parameters $A _ { j } , b _ { j }$ for $j = 1 , \ldots , K$ . This also requires us to compute the gradient of L with respect to the inputs of each layer. For example, if we have inputs x and observations y and a network structure defined by 

$$
\boldsymbol {f} _ {i} := \sigma_ {i} (\boldsymbol {A} _ {i - 1} \boldsymbol {f} _ {i - 1} + \boldsymbol {b} _ {i - 1}), \quad i = 1, \ldots , K,\tag{5.112}
$$

(5.113) 

see also Figure 5.8 for a visualization, we may be interested in finding $A _ { j } , b _ { j }$ for $j = 0 , \ldots , K - 1$ , such that the squared loss 

$$
L (\pmb {\theta}) = \| \pmb {y} - \pmb {f} _ {K} (\pmb {\theta}, \pmb {x}) \| ^ {2}\tag{5.114}
$$

is minimized, where $\pmb \theta = \{ A _ { 0 } , \pmb b _ { 0 } , \dots , \pmb A _ { K - 1 } , \pmb b _ { K - 1 } \} ,$ 

To obtain the gradients with respect to the parameter set $\theta ,$ , we require the partial derivatives of L with respect to the parameters $\pmb { \theta } _ { j } = \{ { \pmb { A } } _ { j } , { \pmb { b } } _ { j } \}$ of each layer $j = 0 , \ldots , K - 1$ . The chain rule allows us to determine the partial derivatives as 

A more in-depth discussion about gradients of neural networks can be found in Justin Domke’s lecture notes 

$$
\frac {\partial L}{\partial \pmb {\theta} _ {K - 1}} = \frac {\partial L}{\partial \pmb {f} _ {K}} \frac {\partial \pmb {f} _ {K}}{\partial \pmb {\theta} _ {K - 1}}\tag{5.115}
$$

https://tinyurl. com/yalcxgtv. 

$$
\frac {\partial L}{\partial \boldsymbol {\theta} _ {K - 2}} = \frac {\partial L}{\partial \boldsymbol {f} _ {K}} \boxed {\frac {\partial \boldsymbol {f} _ {K}}{\partial \boldsymbol {f} _ {K - 1}} \frac {\partial \boldsymbol {f} _ {K - 1}}{\partial \boldsymbol {\theta} _ {K - 2}}}\tag{5.116}
$$

$$
\frac {\partial L}{\partial \boldsymbol {\theta} _ {K - 3}} = \frac {\partial L}{\partial \boldsymbol {f} _ {K}} \frac {\partial \boldsymbol {f} _ {K}}{\partial \boldsymbol {f} _ {K - 1}} \boxed {\frac {\partial \boldsymbol {f} _ {K - 1}}{\partial \boldsymbol {f} _ {K - 2}} \frac {\partial \boldsymbol {f} _ {K - 2}}{\partial \boldsymbol {\theta} _ {K - 3}}}\tag{5.117}
$$

$$
\frac {\partial L}{\partial \boldsymbol {\theta} _ {i}} = \frac {\partial L}{\partial \boldsymbol {f} _ {K}} \frac {\partial \boldsymbol {f} _ {K}}{\partial \boldsymbol {f} _ {K - 1}} \dots \boxed {\frac {\partial \boldsymbol {f} _ {i + 2}}{\partial \boldsymbol {f} _ {i + 1}} \frac {\partial \boldsymbol {f} _ {i + 1}}{\partial \boldsymbol {\theta} _ {i}}}\tag{5.118}
$$

The orange terms are partial derivatives of the output of a layer with respect to its inputs, whereas the blue terms are partial derivatives of the output of a layer with respect to its parameters. Assuming, we have already computed the partial derivatives $\partial L / \partial \pmb { \theta } _ { i + 1 }$ , then most of the computation can be reused to compute $\partial L / \partial \pmb \theta _ { i }$ . The additional terms that we 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/79c7b060945eb8e611c6b5ca1a49b7ce89757094fb669ca01be22b04012dbe81.jpg)



need to compute are indicated by the boxes. Figure 5.9 visualizes that the gradients are passed backward through the network.



Figure 5.9 Backward pass in a multi-layer neural network to compute the gradients of the loss function.



Figure 5.10 Simple graph illustrating the flow of data from x to y via some intermediate variables a, b.


## 5.6.2 Automatic Differentiation

It turns out that backpropagation is a special case of a general technique in numerical analysis called automatic differentiation. We can think of automatic differentation as a set of techniques to numerically (in contrast to symbolically) evaluate the exact (up to machine precision) gradient of a function by working with intermediate variables and applying the chain rule. Automatic differentiation applies a series of elementary arithmetic operations, e.g., addition and multiplication and elementary functions, e.g., sin, cos, exp, log. By applying the chain rule to these operations, the gradient of quite complicated functions can be computed automatically. Automatic differentiation applies to general computer programs and has forward and reverse modes. Baydin et al. (2018) give a great overview of automatic differentiation in machine learning. 

Figure 5.10 shows a simple graph representing the data flow from inputs x to outputs $y$ via some intermediate variables $a ,$ , b. If we were to compute the derivative dy/dx, we would apply the chain rule and obtain 

$$
{\frac {\mathrm{d} y}{\mathrm{d} x}} = {\frac {\mathrm{d} y}{\mathrm{d} b}} {\frac {\mathrm{d} b}{\mathrm{d} a}} {\frac {\mathrm{d} a}{\mathrm{d} x}}.\tag{5.119}
$$

Intuitively, the forward and reverse mode differ in the order of multiplication. Due to the associativity of matrix multiplication, we can choose between 

$$
{\frac {\mathrm{d} y}{\mathrm{d} x}} = \left({\frac {\mathrm{d} y}{\mathrm{d} b}} {\frac {\mathrm{d} b}{\mathrm{d} a}}\right) {\frac {\mathrm{d} a}{\mathrm{d} x}},\tag{5.120}
$$

$$
\frac {\mathrm{d} y}{\mathrm{d} x} = \frac {\mathrm{d} y}{\mathrm{d} b} \left(\frac {\mathrm{d} b}{\mathrm{d} a} \frac {\mathrm{d} a}{\mathrm{d} x}\right).\tag{5.121}
$$

Equation (5.120) would be the reverse mode because gradients are propagated backward through the graph, i.e., reverse to the data flow. Equation (5.121) would be the forward mode, where the gradients flow with the data from left to right through the graph. 

Automatic differentiation is different from symbolic differentiation and numerical approximations of the gradient, e.g., by using finite differences. 

In the following, we will focus on reverse mode automatic differentiation, which is backpropagation. In the context of neural networks, where the input dimensionality is often much higher than the dimensionality of the labels, the reverse mode is computationally significantly cheaper than the forward mode. Let us start with an instructive example. 

## Example 5.14

Consider the function 

$$
f (x) = \sqrt {x ^ {2} + \exp (x ^ {2})} + \cos \left(x ^ {2} + \exp (x ^ {2})\right)\tag{5.122}
$$

from (5.109). If we were to implement a function f on a computer, we would be able to save some computation by using intermediate variables: 

$$
a = x ^ {2},\tag{5.123}
$$

$$
b = \exp (a),\tag{5.124}
$$

$$
c = a + b,\tag{5.125}
$$

$$
d = \sqrt {c},\tag{5.126}
$$

$$
e = \cos (c),\tag{5.127}
$$

$$
f = d + e.\tag{5.128}
$$

Figure 5.11 Computation graph with inputs x, function values $f ,$ and intermediate variables a, b, c, d, e. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/2fdb07ebba7df0a4d5878fe61bfe3dedc9f1cf86a1c7e346dad1575efce2c42a.jpg)


This is the same kind of thinking process that occurs when applying the chain rule. Note that the preceding set of equations requires fewer operations than a direct implementation of the function $f ( x )$ as defined in (5.109). The corresponding computation graph in Figure 5.11 shows the flow of data and computations required to obtain the function value $f .$ 

The set of equations that include intermediate variables can be thought of as a computation graph, a representation that is widely used in implementations of neural network software libraries. We can directly compute the derivatives of the intermediate variables with respect to their corresponding inputs by recalling the definition of the derivative of elementary functions. We obtain the following: 

$$
\frac {\partial a}{\partial x} = 2 x\tag{5.129}
$$

$$
\frac {\partial b}{\partial a} = \exp (a)\tag{5.130}
$$



Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 



$$
{\frac {\partial c}{\partial a}} = 1 = {\frac {\partial c}{\partial b}}\tag{5.131}
$$

$$
{\frac {\partial d}{\partial c}} = {\frac {1}{2 \sqrt {c}}}\tag{5.132}
$$

$$
\frac {\partial e}{\partial c} = - \sin (c)\tag{5.133}
$$

$$
{\frac {\partial f}{\partial d}} = 1 = {\frac {\partial f}{\partial e}}.\tag{5.134}
$$

By looking at the computation graph in Figure 5.11, we can compute $\partial f / \partial x$ by working backward from the output and obtain 

$$
{\frac {\partial f}{\partial c}} = {\frac {\partial f}{\partial d}} {\frac {\partial d}{\partial c}} + {\frac {\partial f}{\partial e}} {\frac {\partial e}{\partial c}}\tag{5.135}
$$

$$
\partial f \quad \partial f \partial c
$$

$$
\overline {{\partial b}} = \overline {{\partial c}} \overline {{\partial b}}\tag{5.136}
$$

$$
\frac {\partial f}{\partial a} = \frac {\partial f}{\partial b} \frac {\partial b}{\partial a} + \frac {\partial f}{\partial c} \frac {\partial c}{\partial a}\tag{5.137}
$$

$$
{\frac {\partial f}{\partial x}} = {\frac {\partial f}{\partial a}} {\frac {\partial a}{\partial x}}.\tag{5.138}
$$

Note that we implicitly applied the chain rule to obtain $\partial f / \partial x$ . By substituting the results of the derivatives of the elementary functions, we get 

$$
{\frac {\partial f}{\partial c}} = 1 \cdot {\frac {1}{2 {\sqrt {c}}}} + 1 \cdot (- \sin (c))\tag{5.139}
$$

$$
\frac {\partial f}{\partial b} = \frac {\partial f}{\partial c} \cdot 1\tag{5.140}
$$

$$
{\frac {\partial f}{\partial a}} = {\frac {\partial f}{\partial b}} \exp (a) + {\frac {\partial f}{\partial c}} \cdot 1\tag{5.141}
$$

$$
{\frac {\partial f}{\partial x}} = {\frac {\partial f}{\partial a}} \cdot 2 x.\tag{5.142}
$$

By thinking of each of the derivatives above as a variable, we observe that the computation required for calculating the derivative is of similar complexity as the computation of the function itself. This is quite counterintuitive since the mathematical expression for the derivative $\frac { \partial f } { \partial x } ~ ( 5 . 1 1 0 )$ is significantly more complicated than the mathematical expression of the function $f ( x )$ in (5.109). 

Automatic differentiation is a formalization of Example 5.14. Let $x _ { 1 } , \ldots , x _ { d }$ be the input variables to the function, $x _ { d + 1 } , \dots , x _ { D - 1 }$ be the intermediate variables, and $x _ { D }$ the output variable. Then the computation graph can be expressed as follows: 

$$
\text { For   } i = d + 1, \dots , D: \quad x _ {i} = g _ {i} (x _ {\mathrm{Pa} (x _ {i})}),\tag{5.143}
$$



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



where the $g _ { i } ( \cdot )$ are elementary functions and $x _ { \operatorname* { P a } ( x _ { i } ) }$ are the parent nodes of the variable $x _ { i }$ in the graph. Given a function defined in this way, we can use the chain rule to compute the derivative of the function in a stepby-step fashion. Recall that by definition $f = x _ { D }$ and hence 

$$
\frac {\partial f}{\partial x _ {D}} = 1.\tag{5.144}
$$

For other variables $x _ { i } ,$ we apply the chain rule 

$$
\frac {\partial f}{\partial x _ {i}} = \sum_ {x _ {j}: x _ {i} \in \mathrm{Pa} (x _ {j})} \frac {\partial f}{\partial x _ {j}} \frac {\partial x _ {j}}{\partial x _ {i}} = \sum_ {x _ {j}: x _ {i} \in \mathrm{Pa} (x _ {j})} \frac {\partial f}{\partial x _ {j}} \frac {\partial g _ {j}}{\partial x _ {i}},\tag{5.145}
$$

where $\mathrm { P a } ( x _ { j } )$ is the set of parent nodes of $x _ { j }$ in the computation graph. Equation (5.143) is the forward propagation of a function, whereas (5.145) is the backpropagation of the gradient through the computation graph. For neural network training, we backpropagate the error of the prediction with respect to the label. 

The automatic differentiation approach above works whenever we have a function that can be expressed as a computation graph, where the elementary functions are differentiable. In fact, the function may not even be a mathematical function but a computer program. However, not all computer programs can be automatically differentiated, e.g., if we cannot find differential elementary functions. Programming structures, such as for loops and if statements, require more care as well. 

## 5.7 Higher-Order Derivatives

So far, we have discussed gradients, i.e., first-order derivatives. Sometimes, we are interested in derivatives of higher order, e.g., when we want to use Newton’s Method for optimization, which requires second-order derivatives (Nocedal and Wright, 2006). In Section 5.1.1, we discussed the Taylor series to approximate functions using polynomials. In the multivariate case, we can do exactly the same. In the following, we will do exactly this. But let us start with some notation. 

Consider a function $f : \mathbb { R } ^ { 2 }  \mathbb { R }$ of two variables x, y. We use the following notation for higher-order partial derivatives (and for gradients): 

1 $\frac { \partial ^ { 2 } f } { \partial x ^ { 2 } }$ is the second partial derivative of f with respect to x. 

$\frac { \partial ^ { n } f } { \partial x ^ { n } }$ is the nth partial derivative of $f$ with respect to x. 

1 $\begin{array} { r } { \frac { \partial ^ { 2 } f } { \partial y \partial x } = \frac { \partial } { \partial y } \left( \frac { \partial f } { \partial x } \right) } \end{array}$ is the partial derivative obtained by first partial differentiating with respect to x and then with respect to $y .$ 

$\frac { \partial ^ { 2 } f } { \partial x \partial y }$ is the partial derivative obtained by first partial differentiating by y and then x. 

The Hessian is the collection of all second-order partial derivatives. 

Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/6ea7da9249141fa031b262595a36e03e1fcfd7a7cdfc47204bada2e87f3ecf9d.jpg)



Figure 5.12 Linear approximation of a function. The original function $f$ is linearized at $x _ { 0 } = - 2$ using a first-order Taylor series expansion.



If $f ( x , y )$ is a twice (continuously) differentiable function, then


$$
\frac {\partial^ {2} f}{\partial x \partial y} = \frac {\partial^ {2} f}{\partial y \partial x},\tag{5.146}
$$

i. $\mathbf { . e . }$ , the order of differentiation does not matter, and the corresponding Hessian matrix 

$$
\boldsymbol {H} = \left[ \begin{array}{c c} \frac {\partial^ {2} f}{\partial x ^ {2}} & \frac {\partial^ {2} f}{\partial x \partial y} \\ \frac {\partial^ {2} f}{\partial x \partial y} & \frac {\partial^ {2} f}{\partial y ^ {2}} \end{array} \right]\tag{5.147}
$$

is symmetric. The Hessian is denoted as $\nabla _ { x , y } ^ { 2 } f ( x , y )$ . Generally, for $\mathbf { x } \in \mathbb { R } ^ { n }$ and $f : \mathbb { R } ^ { n }  \mathbb { R }$ , the Hessian is an $n \times n$ matrix. The Hessian measures the curvature of the function locally around $( x , y )$ 

Remark (Hessian of a Vector Field). If $f : \mathbb { R } ^ { n }  \mathbb { R } ^ { m }$ is a vector field, the Hessian is an $( m \times n \times n )$ )-tensor. $\diamondsuit$ 

## 5.8 Linearization and Multivariate Taylor Series

The gradient $\nabla f$ of a function f is often used for a locally linear approximation of $f$ around $\scriptstyle { \pmb { x } } _ { 0 } { \mathrm { : } }$ : 

$$
f (\boldsymbol {x}) \approx f (\boldsymbol {x} _ {0}) + (\nabla_ {\boldsymbol {x}} f) (\boldsymbol {x} _ {0}) (\boldsymbol {x} - \boldsymbol {x} _ {0}).\tag{5.148}
$$

Here $( \nabla _ { \pmb { x } } f ) ( \pmb { x } _ { 0 } )$ is the gradient of $f$ with respect to $^ { x , }$ evaluated at $\scriptstyle { \mathbf { \mathscr { x } } } _ { \mathrm { 0 } }$ Figure 5.12 illustrates the linear approximation of a function $f$ at an input $x _ { 0 }$ . The original function is approximated by a straight line. This approximation is locally accurate, but the farther we move away from $x _ { 0 }$ the worse the approximation gets. Equation (5.148) is a special case of a mul tivariate Taylor series expansion of $f$ at $\scriptstyle { \mathbf { x } } _ { 0 } .$ , where we consider only the first two terms. We discuss the more general case in the following, which will allow for better approximations. 

©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 

Figure 5.13 Visualizing outer products. Outer products of vectors increase the dimensionality of the array by 1 per term. (a) The outer product of two vectors results in a matrix; (b) the outer product of three vectors yields a third-order tensor. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/104b2216c0c601bba180ad7915d4623a6fe442d4e79e8fe6d459cef583978622.jpg)



(a) Given a vector $\delta \in \mathbb { R } ^ { 4 }$ , we obtain the outer product $\delta ^ { 2 } : = \delta \otimes \delta = \delta \delta ^ { \top } \in$ $\mathbb { R } ^ { 4 \times 4 }$ as a matrix.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/e44289435151cd1b85e90b55bd916a144dd1e8b5cf8452f36c88077e07fb0354.jpg)



(b) An outer product $\delta ^ { 3 } : = \delta \otimes \delta \otimes \delta \in \mathbb { R } ^ { 4 \times 4 \times 4 }$ results in a third-order tensor (“three dimensional matrix”), i.e., an array with three indexes.


Definition 5.7 (Multivariate Taylor Series). We consider a function 

$$
f: \mathbb {R} ^ {D} \to \mathbb {R}\tag{5.149}
$$

$$
\boldsymbol {x} \mapsto f (\boldsymbol {x}), \quad \boldsymbol {x} \in \mathbb {R} ^ {D},\tag{5.150}
$$

that is smooth at $\scriptstyle { \pmb x } _ { 0 }$ . When we define the difference vector $\pmb { \delta } : = \pmb { x } - \pmb { x } _ { 0 }$ 44 the multivariate Taylor series of $f$ at $\scriptstyle \left( { \pmb x } _ { 0 } \right)$ is defined as 

$$
f (\boldsymbol {x}) = \sum_ {k = 0} ^ {\infty} \frac {D _ {\boldsymbol {x}} ^ {k} f (\boldsymbol {x} _ {0})}{k !} \boldsymbol {\delta} ^ {k},\tag{5.151}
$$

where $D _ { x } ^ { k } f ( { \pmb x } _ { 0 } )$ is the k-th (total) derivative of f with respect to ${ \mathbf { } } ^ { \mathbf { } } \mathbf { { \mathbf { { x } } } } ,$ evaluated at $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ 

Definition 5.8 (Taylor Polynomial). The Taylor polynomial of degree n of $f$ at $\scriptstyle { \pmb x } _ { 0 }$ contains the first $n + 1$ components of the series in (5.151) and is defined as 

$$
T _ {n} (\boldsymbol {x}) = \sum_ {k = 0} ^ {n} \frac {D _ {\boldsymbol {x}} ^ {k} f (\boldsymbol {x} _ {0})}{k !} \delta^ {k}.\tag{5.152}
$$

In (5.151) and (5.152), we used the slightly sloppy notation of $\delta ^ { k }$ which is not defined for vectors $\pmb { x } \in \mathbb { R } ^ { D } , D > 1$ , and $k > 1$ . Note that both $D _ { x } ^ { k } f$ and $\delta ^ { k }$ are k-th order tensors, i.e., k-dimensional arrays. The k times 

kth-order tensor $\delta ^ { k } \in \mathbb { R } ^ { \widehat { D \times D \times \ldots \times D } }$ is obtained as a k-fold outer product, denoted by , of the vector $\pmb { \delta } \in \mathbb { R } ^ { D }$ . For example, 

$$
\boldsymbol {\delta} ^ {2} := \boldsymbol {\delta} \otimes \boldsymbol {\delta} = \boldsymbol {\delta} \boldsymbol {\delta} ^ {\top}, \quad \boldsymbol {\delta} ^ {2} [ i, j ] = \delta [ i ] \delta [ j ]\tag{5.153}
$$

Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 

5.8 Linearization and Multivariate Taylor Series 

$$
\boldsymbol {\delta} ^ {3} := \boldsymbol {\delta} \otimes \boldsymbol {\delta} \otimes \boldsymbol {\delta}, \quad \boldsymbol {\delta} ^ {3} [ i, j, k ] = \delta [ i ] \delta [ j ] \delta [ k ].\tag{5.154}
$$

Figure 5.13 visualizes two such outer products. In general, we obtain the terms 

$$
D _ {\boldsymbol {x}} ^ {k} f (\boldsymbol {x} _ {0}) \boldsymbol {\delta} ^ {k} = \sum_ {i _ {1} = 1} ^ {D} \dots \sum_ {i _ {k} = 1} ^ {D} D _ {\boldsymbol {x}} ^ {k} f (\boldsymbol {x} _ {0}) [ i _ {1}, \dots , i _ {k} ] \delta [ i _ {1} ] \dots \delta [ i _ {k} ]\tag{5.155}
$$

in the Taylor series, where $D _ { x } ^ { k } f ( x _ { 0 } ) \delta ^ { k }$ contains k-th order polynomials. Now that we defined the Taylor series for vector fields, let us explicitly write down the first terms $D _ { x } ^ { k } f ( x _ { 0 } ) \delta ^ { k }$ of the Taylor series expansion for $k = 0 , \ldots , 3$ and $\pmb { \delta } : = \pmb { x } - \pmb { x } _ { 0 } ;$ 

$$
k = 0: D _ {\boldsymbol {x}} ^ {0} f (\boldsymbol {x} _ {0}) \boldsymbol {\delta} ^ {0} = f (\boldsymbol {x} _ {0}) \in \mathbb {R}\tag{5.156}
$$

$$
k = 1: D _ {\boldsymbol {x}} ^ {1} f (\boldsymbol {x} _ {0}) \boldsymbol {\delta} ^ {1} = \underbrace {\nabla_ {\boldsymbol {x}} f (\boldsymbol {x} _ {0})} _ {1 \times D} \underbrace {\boldsymbol {\delta}} _ {D \times 1} = \sum_ {i = 1} ^ {D} \nabla_ {\boldsymbol {x}} f (\boldsymbol {x} _ {0}) [ i ] \delta [ i ] \in \mathbb {R}\tag{5.157}
$$

$$
k = 2: D _ {\boldsymbol {x}} ^ {2} f (\boldsymbol {x} _ {0}) \boldsymbol {\delta} ^ {2} = \operatorname{tr} \bigl (\underbrace {\boldsymbol {H} (\boldsymbol {x} _ {0})} _ {D \times D} \underbrace {\boldsymbol {\delta}} _ {D \times 1} \underbrace {\boldsymbol {\delta} ^ {\top}} _ {1 \times D} \bigr) = \boldsymbol {\delta} ^ {\top} \boldsymbol {H} (\boldsymbol {x} _ {0}) \boldsymbol {\delta}\tag{5.158}
$$

$$
= \sum_ {i = 1} ^ {D} \sum_ {j = 1} ^ {D} H [ i, j ] \delta [ i ] \delta [ j ] \in \mathbb {R}\tag{5.159}
$$

$$
k = 3: D _ {\boldsymbol {x}} ^ {3} f (\boldsymbol {x} _ {0}) \boldsymbol {\delta} ^ {3} = \sum_ {i = 1} ^ {D} \sum_ {j = 1} ^ {D} \sum_ {k = 1} ^ {D} D _ {x} ^ {3} f (\boldsymbol {x} _ {0}) [ i, j, k ] \delta [ i ] \delta [ j ] \delta [ k ] \in \mathbb {R}\tag{5.160}
$$

Here, ${ \pmb H } ( { \pmb x } _ { 0 } )$ is the Hessian of f evaluated at $\scriptstyle { \pmb x } _ { 0 }$ . 

Example 5.15 (Taylor Series Expansion of a Function with Two Variables) 

Consider the function 

$$
f (x, y) = x ^ {2} + 2 x y + y ^ {3}.\tag{5.161}
$$

We want to compute the Taylor series expansion of f at $( x _ { 0 } , y _ { 0 } ) = ( 1 , 2 )$ Before we start, let us discuss what to expect: The function in (5.161) is a polynomial of degree 3. We are looking for a Taylor series expansion, which itself is a linear combination of polynomials. Therefore, we do not expect the Taylor series expansion to contain terms of fourth or higher order to express a third-order polynomial. This means that it should be sufficient to determine the first four terms of (5.151) for an exact alternative representation of (5.161). 

To determine the Taylor series expansion, we start with the constant term and the first-order derivatives, which are given by 

$$
f (1, 2) = 1 3\tag{5.162}
$$



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 





Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 



$$
\frac {\partial f}{\partial x} = 2 x + 2 y \implies \frac {\partial f}{\partial x} (1, 2) = 6\tag{5.163}
$$

$$
\frac {\partial f}{\partial y} = 2 x + 3 y ^ {2} \implies \frac {\partial f}{\partial y} (1, 2) = 1 4.\tag{5.164}
$$

Therefore, we obtain 

$$
D _ {x, y} ^ {1} f (1, 2) = \nabla_ {x, y} f (1, 2) = \left[ \begin{array}{c c} \frac {\partial f}{\partial x} (1, 2) & \frac {\partial f}{\partial y} (1, 2) \end{array} \right] = \left[ \begin{array}{c c} 6 & 1 4 \end{array} \right] \in \mathbb {R} ^ {1 \times 2}\tag{5.165}
$$

such that 

$$
\frac {D _ {x , y} ^ {1} f (1 , 2)}{1 !} \boldsymbol {\delta} = \left[ \begin{array}{c c} 6 & 1 4 \end{array} \right] \left[ \begin{array}{c} x - 1 \\ y - 2 \end{array} \right] = 6 (x - 1) + 1 4 (y - 2).\tag{5.166}
$$

Note that $D _ { x , y } ^ { 1 } f ( 1 , 2 ) \delta$ contains only linear terms, i.e., first-order polynomials. 

The second-order partial derivatives are given by 

$$
\frac {\partial^ {2} f}{\partial x ^ {2}} = 2 \implies \frac {\partial^ {2} f}{\partial x ^ {2}} (1, 2) = 2\tag{5.167}
$$

$$
\frac {\partial^ {2} f}{\partial y ^ {2}} = 6 y \implies \frac {\partial^ {2} f}{\partial y ^ {2}} (1, 2) = 1 2\tag{5.168}
$$

$$
\frac {\partial^ {2} f}{\partial y \partial x} = 2 \implies \frac {\partial^ {2} f}{\partial y \partial x} (1, 2) = 2\tag{5.169}
$$

$$
\frac {\partial^ {2} f}{\partial x \partial y} = 2 \implies \frac {\partial^ {2} f}{\partial x \partial y} (1, 2) = 2.\tag{5.170}
$$

When we collect the second-order partial derivatives, we obtain the Hessian 

$$
\boldsymbol {H} = \left[ \begin{array}{c c} \frac {\partial^ {2} f}{\partial x ^ {2}} & \frac {\partial^ {2} f}{\partial x \partial y} \\ \frac {\partial^ {2} f}{\partial y \partial x} & \frac {\partial^ {2} f}{\partial y ^ {2}} \end{array} \right] = \left[ \begin{array}{c c} 2 & 2 \\ 2 & 6 y \end{array} \right],\tag{5.171}
$$

such that 

$$
\boldsymbol {H} (1, 2) = \left[ \begin{array}{c c} 2 & 2 \\ 2 & 1 2 \end{array} \right] \in \mathbb {R} ^ {2 \times 2}.\tag{5.172}
$$

Therefore, the next term of the Taylor-series expansion is given by 

$$
\frac {D _ {x , y} ^ {2} f (1 , 2)}{2 !} \boldsymbol {\delta} ^ {2} = \frac {1}{2} \boldsymbol {\delta} ^ {\top} \boldsymbol {H} (1, 2) \boldsymbol {\delta}\tag{5.173a}
$$

$$
= \frac {1}{2} \left[ \begin{array}{c c} x - 1 & y - 2 \end{array} \right] \left[ \begin{array}{c c} 2 & 2 \\ 2 & 1 2 \end{array} \right] \left[ \begin{array}{c} x - 1 \\ y - 2 \end{array} \right]\tag{5.173b}
$$

$$
= (x - 1) ^ {2} + 2 (x - 1) (y - 2) + 6 (y - 2) ^ {2}.\tag{5.173c}
$$

Here, $D _ { x , y } ^ { 2 } f ( 1 , 2 ) \delta ^ { 2 }$ contains only quadratic terms, i.e., second-order polynomials. 

The third-order derivatives are obtained as 

$$
D _ {x, y} ^ {3} f = \left[ \begin{array}{c c} \frac {\partial \boldsymbol {H}}{\partial x} & \frac {\partial \boldsymbol {H}}{\partial y} \end{array} \right] \in \mathbb {R} ^ {2 \times 2 \times 2},\tag{5.174}
$$

$$
D _ {x, y} ^ {3} f [:,:, 1 ] = \frac {\partial \boldsymbol {H}}{\partial x} = \left[ \begin{array}{c c} \frac {\partial^ {3} f}{\partial x ^ {3}} & \frac {\partial^ {3} f}{\partial x ^ {2} \partial y} \\ \frac {\partial^ {3} f}{\partial x \partial y \partial x} & \frac {\partial^ {3} f}{\partial x \partial y ^ {2}} \end{array} \right],\tag{5.175}
$$

$$
D _ {x, y} ^ {3} f [:,:, 2 ] = \frac {\partial \pmb {H}}{\partial y} = \left[ \begin{array}{c c} \frac {\partial^ {3} f}{\partial y \partial x ^ {2}} & \frac {\partial^ {3} f}{\partial y \partial x \partial y} \\ \frac {\partial^ {3} f}{\partial y ^ {2} \partial x} & \frac {\partial^ {3} f}{\partial y ^ {3}} \end{array} \right].\tag{5.176}
$$

Since most second-order partial derivatives in the Hessian in (5.171) are constant, the only nonzero third-order partial derivative is 

$$
\frac {\partial^ {3} f}{\partial y ^ {3}} = 6 \implies \frac {\partial^ {3} f}{\partial y ^ {3}} (1, 2) = 6.\tag{5.177}
$$

Higher-order derivatives and the mixed derivatives of degree 3 (e.g., $\textstyle { \frac { \partial f ^ { 3 } } { \partial x ^ { 2 } \partial y } } )$ vanish, such that 

$$
D _ {x, y} ^ {3} f [:,:, 1 ] = \left[ \begin{array}{c c} 0 & 0 \\ 0 & 0 \end{array} \right], \quad D _ {x, y} ^ {3} f [:,:, 2 ] = \left[ \begin{array}{c c} 0 & 0 \\ 0 & 6 \end{array} \right]\tag{5.178}
$$

and 

$$
\frac {D _ {x , y} ^ {3} f (1 , 2)}{3 !} \delta^ {3} = (y - 2) ^ {3},\tag{5.179}
$$

which collects all cubic terms of the Taylor series. Overall, the (exact) Taylor series expansion of $f$ at $( x _ { 0 } , y _ { 0 } ) = ( 1 , 2 )$ is 

$$
f (x) = f (1, 2) + D _ {x, y} ^ {1} f (1, 2) \pmb {\delta} + \frac {D _ {x , y} ^ {2} f (1 , 2)}{2 !} \pmb {\delta} ^ {2} + \frac {D _ {x , y} ^ {3} f (1 , 2)}{3 !} \pmb {\delta} ^ {3}\tag{5.180a}
$$

$$
\begin{array}{l} = f (1, 2) + \frac {\partial f (1 , 2)}{\partial x} (x - 1) + \frac {\partial f (1 , 2)}{\partial y} (y - 2) \\ \quad + \frac {1}{2 !} \left(\frac {\partial^ {2} f (1 , 2)}{\partial x ^ {2}} (x - 1) ^ {2} + \frac {\partial^ {2} f (1 , 2)}{\partial y ^ {2}} (y - 2) ^ {2} \right. \\ \quad \left. + 2 \frac {\partial^ {2} f (1 , 2)}{\partial x \partial y} (x - 1) (y - 2)\right) + \frac {1}{6} \frac {\partial^ {3} f (1 , 2)}{\partial y ^ {3}} (y - 2) ^ {3} \\ = 1 3 + 6 (x - 1) + 1 4 (y - 2) \\ \quad + (x - 1) ^ {2} + 6 (y - 2) ^ {2} + 2 (x - 1) (y - 2) + (y - 2) ^ {3} \end{array}\tag{5.180b}
$$

(5.180c) 

In this case, we obtained an exact Taylor series expansion of the polynomial in (5.161), i.e., the polynomial in (5.180c) is identical to the original polynomial in (5.161). In this particular example, this result is not surprising since the original function was a third-order polynomial, which we expressed through a linear combination of constant terms, first-order, second-order, and third-order polynomials in (5.180c). 



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



## 5.9 Further Reading

Further details of matrix differentials, along with a short review of the required linear algebra, can be found in Magnus and Neudecker (2007). Automatic differentiation has had a long history, and we refer to Griewank and Walther (2003), Griewank and Walther (2008), and Elliott (2009) and the references therein. 

In machine learning (and other disciplines), we often need to compute expectations, i.e., we need to solve integrals of the form 

$$
\mathbb {E} _ {\boldsymbol {x}} [ f (\boldsymbol {x}) ] = \int f (\boldsymbol {x}) p (\boldsymbol {x}) d \boldsymbol {x}.\tag{5.181}
$$

Even if $p ( { \pmb x } )$ is in a convenient form (e.g., Gaussian), this integral generally cannot be solved analytically. The Taylor series expansion of f is one way of finding an approximate solution: Assuming $p ( { \pmb x } ) = \mathcal { N } ( { \pmb \mu } , { \pmb \Sigma } )$ is Gaussian, then the first-order Taylor series expansion around $\pmb { \mu }$ locally linearizes the nonlinear function $f .$ For linear functions, we can compute the mean (and the covariance) exactly if $p ( { \pmb x } )$ is Gaussian distributed (see Section 6.5). This property is heavily exploited by the extended Kalman filter (Maybeck, 1979) for online state estimation in nonlinear dynamical systems (also called “state-space models”). Other deterministic ways to approximate the integral in (5.181) are the unscented transform (Julier and Uhlmann, 1997), which does not require any gradients, or the Laplace approximation (MacKay, 2003; Bishop, 2006; Murphy, 2012), which uses a second-order Taylor series expansion (requiring the Hessian) for a local Gaussian approximation of $p ( { \pmb x } )$ around its mode. 

## Exercises

5.1 Compute the derivative $f ^ { \prime } ( x )$ for 

$$
f (x) = \log (x ^ {4}) \sin (x ^ {3}).
$$

5.2 Compute the derivative $f ^ { \prime } ( x )$ of the logistic sigmoid 

$$
f (x) = \frac {1}{1 + \exp (- x)}.
$$

5.3 Compute the derivative $f ^ { \prime } ( x )$ of the function 

$$
f (x) = \exp (- \frac {1}{2 \sigma^ {2}} (x - \mu) ^ {2}),
$$

where $\mu , \sigma \in \mathbb { R }$ are constants. 

5.4 Compute the Taylor polynomials $T _ { n } , n = 0 , \ldots , 5$ of $f ( x ) = \sin ( x ) + \cos ( x )$ at $x _ { 0 } = 0 .$ 

5.5 Consider the following functions: 

$$
f _ {1} (\boldsymbol {x}) = \sin (x _ {1}) \cos (x _ {2}), \quad \boldsymbol {x} \in \mathbb {R} ^ {2}
$$

$$
f _ {2} (\pmb {x}, \pmb {y}) = \pmb {x} ^ {\top} \pmb {y}, \quad \pmb {x}, \pmb {y} \in \mathbb {R} ^ {n}
$$

$$
f _ {3} (\pmb {x}) = \pmb {x x} ^ {\top}, \qquad \pmb {x} \in \mathbb {R} ^ {n}
$$

Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 

a. What are the dimensions of $\begin{array} { r } { \frac { \partial f _ { i } } { \partial { \bf x } } ~ ? } \end{array}$ 

b. Compute the Jacobians. 

5.6 Differentiate f with respect to t and g with respect to X, where 

$$
\begin{array}{l l} f (\boldsymbol {t}) = \sin (\log (\boldsymbol {t} ^ {\top} \boldsymbol {t}))  , & t \in \mathbb {R} ^ {D} \\ g (\boldsymbol {X}) = \operatorname{tr} (\boldsymbol {A X B})  , & \boldsymbol {A} \in \mathbb {R} ^ {D \times E}, \boldsymbol {X} \in \mathbb {R} ^ {E \times F}, \boldsymbol {B} \in \mathbb {R} ^ {F \times D}  , \end{array}
$$

where $\operatorname { t r } ( \cdot )$ denotes the trace. 

Compute the derivatives $\mathrm { d } f / \mathrm { d } x$ of the following functions by using the chain rule. Provide the dimensions of every single partial derivative. Describe your steps in detail. 

a. 

$$
f (z) = \log (1 + z), \quad z = \boldsymbol {x} ^ {\top} \boldsymbol {x}, \quad \boldsymbol {x} \in \mathbb {R} ^ {D}
$$

b. 

$$
f (\boldsymbol {z}) = \sin (\boldsymbol {z}), \quad \boldsymbol {z} = \boldsymbol {A} \boldsymbol {x} + \boldsymbol {b}, \quad \boldsymbol {A} \in \mathbb {R} ^ {E \times D}, \boldsymbol {x} \in \mathbb {R} ^ {D}, \boldsymbol {b} \in \mathbb {R} ^ {E}
$$

where sin(·) is applied to every element of $_ { z , }$ 

5.8 Compute the derivatives $\mathrm { d } f / \mathrm { d } x$ of the following functions. Describe your steps in detail. 

a. Use the chain rule. Provide the dimensions of every single partial derivative. 

$$
\begin{array}{r} f (z) = \exp (- \frac {1}{2} z) \\ z = g (\pmb {y}) = \pmb {y} ^ {\top} \pmb {S} ^ {- 1} \pmb {y} \\ \pmb {y} = h (\pmb {x}) = \pmb {x} - \pmb {\mu} \end{array}
$$

where $\mathbf { \Psi } : \mu \in \mathbb { R } ^ { D } , S \in \mathbb { R } ^ { D \times D }$ 

b. 

$$
f (\boldsymbol {x}) = \operatorname{tr} \left(\boldsymbol {x x} ^ {\top} + \sigma^ {2} \boldsymbol {I}\right), \quad \boldsymbol {x} \in \mathbb {R} ^ {D}
$$

Here $\operatorname { t r } ( A )$ is the trace of $A , \mathrm { i . e . }$ , the sum of the diagonal elements $A _ { i i }$ Hint: Explicitly write out the outer product. 

c. Use the chain rule. Provide the dimensions of every single partial derivative. You do not need to compute the product of the partial derivatives explicitly. 

$$
\begin{array}{l} \boldsymbol {f} = \tanh (\boldsymbol {z}) \in \mathbb {R} ^ {M} \\ \boldsymbol {z} = \boldsymbol {A x} + \boldsymbol {b}, \quad \boldsymbol {x} \in \mathbb {R} ^ {N}, \boldsymbol {A} \in \mathbb {R} ^ {M \times N}, \boldsymbol {b} \in \mathbb {R} ^ {M}. \end{array}
$$

Here, tanh is applied to every component of z. 

5.9 We define 

$$
\begin{array}{c} g (\boldsymbol {x}, \boldsymbol {z}, \boldsymbol {\nu}) := \log p (\boldsymbol {x}, \boldsymbol {z}) - \log q (\boldsymbol {z}, \boldsymbol {\nu}) \\ \boldsymbol {z} := t (\boldsymbol {\epsilon}, \boldsymbol {\nu}) \end{array}
$$

for differentiable functions $p , q , t$ and x $\in \mathbb { R } ^ { D } , z \in \mathbb { R } ^ { E } , \pmb { \nu } \in \mathbb { R } ^ { F } , \pmb { \epsilon } \in \mathbb { R } ^ { G }$ . By using the chain rule, compute the gradient 

$$
\frac {\mathrm{d}}{\mathrm{d} \pmb {\nu}} g (\pmb {x}, \pmb {z}, \pmb {\nu}).
$$



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/3c7e9ff90ce8de793778e3781c1daa8bc33ec9663aa10f18b7bf26f61334c0e0.jpg)


# Probability and Distributions

Probability, loosely speaking, concerns the study of uncertainty. Probability can be thought of as the fraction of times an event occurs, or as a degree of belief about an event. We then would like to use this probability to measure the chance of something occurring in an experiment. As mentioned in Chapter 1, we often quantify uncertainty in the data, uncertainty in the machine learning model, and uncertainty in the predictions produced by the model. Quantifying uncertainty requires the idea of a random variable, which is a function that maps outcomes of random experiments to a set of properties that we are interested in. Associated with the random variable is a function that measures the probability that a particular outcome (or set of outcomes) will occur; this is called the probability distribution. 

Probability distributions are used as a building block for other concepts, such as probabilistic modeling (Section 8.4), graphical models (Section 8.5), and model selection (Section 8.6). In the next section, we present the three concepts that define a probability space (the sample space, the events, and the probability of an event) and how they are related to a fourth concept called the random variable. The presentation is deliberately slightly hand wavy since a rigorous presentation may occlude the intuition behind the concepts. An outline of the concepts presented in this chapter are shown in Figure 6.1. 

## 6.1 Construction of a Probability Space

The theory of probability aims at defining a mathematical structure to describe random outcomes of experiments. For example, when tossing a single coin, we cannot determine the outcome, but by doing a large number of coin tosses, we can observe a regularity in the average outcome. Using this mathematical structure of probability, the goal is to perform automated reasoning, and in this sense, probability generalizes logical reasoning (Jaynes, 2003). 

## 6.1.1 Philosophical Issues

When constructing automated reasoning systems, classical Boolean logic does not allow us to express certain forms of plausible reasoning. Consider 

## 172

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/0bd491ae27c85c679c52266be660a2800c05b158dee5729c5cbace81d1fe2854.jpg)



Figure 6.1 A mind map of the concepts related to random variables and probability distributions, as described in this chapter.


the following scenario: We observe that A is false. We find B becomes less plausible, although no conclusion can be drawn from classical logic. We observe that B is true. It seems A becomes more plausible. We use this form of reasoning daily. We are waiting for a friend, and consider three possibilities: H1, she is on time; H2, she has been delayed by traffic; and H3, she has been abducted by aliens. When we observe our friend is late, we must logically rule out H1. We also tend to consider H2 to be more likely, though we are not logically required to do so. Finally, we may consider H3 to be possible, but we continue to consider it quite unlikely. How do we conclude H2 is the most plausible answer? Seen in this way, probability theory can be considered a generalization of Boolean logic. In the context of machine learning, it is often applied in this way to formalize the design of automated reasoning systems. Further arguments about how probability theory is the foundation of reasoning systems can be found in Pearl (1988). 

The philosophical basis of probability and how it should be somehow related to what we think should be true (in the logical sense) was studied by Cox (Jaynes, 2003). Another way to think about it is that if we are precise about our common sense we end up constructing probabilities. E. T. Jaynes (1922–1998) identified three mathematical criteria, which must apply to all plausibilities: 

1. The degrees of plausibility are represented by real numbers. 

2. These numbers must be based on the rules of common sense. 

“For plausible reasoning it is necessary to extend the discrete true and false values of truth to continuous plausibilities” (Jaynes, 2003). 

3. The resulting reasoning must be consistent, with the three following meanings of the word “consistent”: 

(a) Consistency or non-contradiction: When the same result can be reached through different means, the same plausibility value must be found in all cases. 

(b) Honesty: All available data must be taken into account. 

(c) Reproducibility: If our state of knowledge about two problems are the same, then we must assign the same degree of plausibility to both of them. 

The Cox–Jaynes theorem proves these plausibilities to be sufficient to define the universal mathematical rules that apply to plausibility p, up to transformation by an arbitrary monotonic function. Crucially, these rules are the rules of probability. 

Remark. In machine learning and statistics, there are two major interpretations of probability: the Bayesian and frequentist interpretations (Bishop, 2006; Efron and Hastie, 2016). The Bayesian interpretation uses probability to specify the degree of uncertainty that the user has about an event. It is sometimes referred to as “subjective probability” or “degree of belief”. The frequentist interpretation considers the relative frequencies of events of interest to the total number of events that occurred. The probability of an event is defined as the relative frequency of the event in the limit when one has infinite data. ◇ 

Some machine learning texts on probabilistic models use lazy notation and jargon, which is confusing. This text is no exception. Multiple distinct concepts are all referred to as “probability distribution”, and the reader has to often disentangle the meaning from the context. One trick to help make sense of probability distributions is to check whether we are trying to model something categorical (a discrete random variable) or something continuous (a continuous random variable). The kinds of questions we tackle in machine learning are closely related to whether we are considering categorical or continuous models. 

## 6.1.2 Probability and Random Variables

There are three distinct ideas that are often confused when discussing probabilities. First is the idea of a probability space, which allows us to quantify the idea of a probability. However, we mostly do not work directly with this basic probability space. Instead, we work with random variables (the second idea), which transfers the probability to a more convenient (often numerical) space. The third idea is the idea of a distribution or law associated with a random variable. We will introduce the first two ideas in this section and expand on the third idea in Section 6.2. 

Modern probability is based on a set of axioms proposed by Kolmogorov 

## 6.1 Construction of a Probability Space

(Grinstead and Snell, 1997; Jaynes, 2003) that introduce the three concepts of sample space, event space, and probability measure. The probability space models a real-world process (referred to as an experiment) with random outcomes. 

## The sample space Ω

The sample space is the set of all possible outcomes of the experiment, usually denoted by Ω. For example, two successive coin tosses have a sample space of hh, tt, ht, th , where “h” denotes “heads” and “t” denotes “tails”. 

## The event space

The event space is the space of potential results of the experiment. A subset A of the sample space Ω is in the event space if at the end of the experiment we can observe whether a particular outcome $\omega \in \Omega$ is in A. The event space is obtained by considering the collection of subsets of Ω, and for discrete probability distributions (Section 6.2.1) is often the power set of Ω. 

## The probability P

With each event $A \in { \mathcal { A } }$ , we associate a number $P ( A )$ that measures the probability or degree of belief that the event will occur. P (A) is called the probability of A. 

The probability of a single event must lie in the interval [0, 1], and the total probability over all outcomes in the sample space Ω must be 1, i.e., $P ( \Omega ) = 1$ . Given a probability space $( \Omega , A , P )$ , we want to use it to model some real-world phenomenon. In machine learning, we often avoid explicitly referring to the probability space, but instead refer to probabilities on quantities of interest, which we denote by . In this book, we refer to $\tau$ as the target space and refer to elements of $\tau$ as states. We introduce a function $\dot { X } : \bar { \Omega } \stackrel { \bar { } } {  } \mathcal { T }$ that takes an element of Ω (an outcome) and returns a particular quantity of interest x, a value in . This association/mapping from Ω to is called a random variable. For example, in the case of tossing two coins and counting the number of heads, a random variable X maps to the three possible outcomes: $X ( \mathrm { h h } ) = 2 , X ( \mathrm { h t } ) = 1 , X ( \mathrm { t h } ) = 1$ , and $X ( \mathrm { t t } ) = 0$ . In this particular case, $\mathcal { T } = \{ 0 , 1 , 2 \}$ , and it is the probabilities on elements of that we are interested in. For a finite sample space Ω and finite , the function corresponding to a random variable is essentially a lookup table. For any subset $S \subseteq \tau$ , we associate $P _ { X } ( S ) \in [ 0 , 1 ]$ (the probability) to a particular event occurring corresponding to the random variable X. Example 6.1 provides a concrete illustration of the terminology. 

Remark. The aforementioned sample space Ω unfortunately is referred to by different names in different books. Another common name for Ω is “state space” (Jacod and Protter, 2004), but state space is sometimes reserved for referring to states in a dynamical system (Hasselblatt and 

Katok, 2003). Other names sometimes used to describe Ω are: “sample description space”, “possibility space,” and “event space”. 

## Example 6.1

We assume that the reader is already familiar with computing probabilities of intersections and unions of sets of events. A gentler introduction to probability with many examples can be found in chapter 2 of Walpole et al. (2011). 

Consider a statistical experiment where we model a funfair game consisting of drawing two coins from a bag (with replacement). There are coins from USA (denoted as $) and UK (denoted as £) in the bag, and since we draw two coins from the bag, there are four outcomes in total. The state space or sample space Ω of this experiment is then ($, $), ($, £), (£, $), (£, £). Let us assume that the composition of the bag of coins is such that a draw returns at random a $ with probability 0.3. 

The event we are interested in is the total number of times the repeated draw returns $. Let us define a random variable X that maps the sample space Ω to , which denotes the number of times we draw $ out of the bag. We can see from the preceding sample space we can get zero $, one $, or two $s, and therefore $\bar { \mathcal { T } } = \{ 0 , 1 , 2 \}$ . The random variable X (a function or lookup table) can be represented as a table like the following: 

$$
X ((\mathbb {S}, \mathbb {S})) = 2\tag{6.1}
$$

$$
X ((\mathbb {S}, \mathcal {L})) = 1\tag{6.2}
$$

$$
X ((£, \mathbb {S})) = 1\tag{6.3}
$$

$$
X ((£, £)) = 0.\tag{6.4}
$$

Since we return the first coin we draw before drawing the second, this implies that the two draws are independent of each other, which we will discuss in Section 6.4.5. Note that there are two experimental outcomes, which map to the same event, where only one of the draws returns $. Therefore, the probability mass function (Section 6.2.1) of X is given by 

$$
\begin{array}{r l} P (X = 2) & = P ((,)) \\ & = P () \cdot P () \\ & = 0. 3 \cdot 0. 3 = 0. 0 9 \end{array}\tag{6.5}
$$

$$
\begin{array}{r l} P (X = 1) & = P ((, £) \cup (£,)) \\ & = P ((, £)) + P ((£,)) \\ & = 0. 3 \cdot (1 - 0. 3) + (1 - 0. 3) \cdot 0. 3 = 0. 4 2 \end{array}\tag{6.6}
$$

$$
\begin{array}{r l} P (X = 0) & = P ((£, £)) \\ & = P (£) \cdot P (£) \\ & = (1 - 0. 3) \cdot (1 - 0. 3) = 0. 4 9. \end{array}
$$

(6.7) 

## 6.1 Construction of a Probability Space

In the calculation, we equated two different concepts, the probability of the output of X and the probability of the samples in Ω. For example, in (6.7) we say $P ( X = 0 ) = P ( ( \mathcal { L } , \mathcal { L } ) )$ . Consider the random variable $X : \Omega  \mathcal { T }$ and a subset $S \subseteq \tau$ (for example, a single element of $\tau$ such as the outcome that one head is obtained when tossing two coins). Let $X ^ { - 1 } ( S )$ be the pre-image of S by X, i.e., the set of elements of Ω that map to S under $X ; \{ \omega \in \Omega : X ( \omega ) \in S \}$ . One way to understand the transformation of probability from events in Ω via the random variable X is to associate it with the probability of the pre-image of S (Jacod and Protter, 2004). For $S \subseteq { \mathcal { T } }$ , we have the notation 

$$
P _ {X} (S) = P (X \in S) = P (X ^ {- 1} (S)) = P (\{\omega \in \Omega : X (\omega) \in S \}).\tag{6.8}
$$

The left-hand side of (6.8) is the probability of the set of possible outcomes $( \boldsymbol { \mathrm { e . g . } }$ , number of $\$ 12$ that we are interested in. Via the random variable $X ,$ , which maps states to outcomes, we see in the right-hand side of (6.8) that this is the probability of the set of states (in Ω) that have the property $( \boldsymbol { \mathrm { e } } . \boldsymbol { \mathrm { g } } . , \boldsymbol { \Phi } \mathscr { L } , \mathscr { L } \boldsymbol { \Phi } )$ . We say that a random variable X is distributed according to a particular probability distribution $P _ { X }$ , which defines the probability mapping between the event and the probability of the outcome of the random variable. In other words, the function $P _ { X }$ or equivalently $P \circ X ^ { - 1 }$ is the law or distribution of random variable X. 

Remark. The target space, that is, the range of the random variable X, is used to indicate the kind of probability space, i.e., a random variable. When $\tau$ is finite or countably infinite, this is called a discrete random variable (Section 6.2.1). For continuous random variables (Section 6.2.2), we only consider $\mathcal { T } = \mathbb { R }$ or $\mathcal { T } = \mathbb { R } ^ { D }$ 

## 6.1.3 Statistics

Probability theory and statistics are often presented together, but they concern different aspects of uncertainty. One way of contrasting them is by the kinds of problems that are considered. Using probability, we can consider a model of some process, where the underlying uncertainty is captured by random variables, and we use the rules of probability to derive what happens. In statistics, we observe that something has happened and try to figure out the underlying process that explains the observations. In this sense, machine learning is close to statistics in its goals to construct a model that adequately represents the process that generated the data. We can use the rules of probability to obtain a “best-fitting” model for some data. 

Another aspect of machine learning systems is that we are interested in generalization error (see Chapter 8). This means that we are actually interested in the performance of our system on instances that we will observe in future, which are not identical to the instances that we have seen so far. This analysis of future performance relies on probability and statistics, most of which is beyond what will be presented in this chapter. The interested reader is encouraged to look at the books by Boucheron et al. (2013) and Shalev-Shwartz and Ben-David (2014). We will see more about statistics in Chapter 8. 

## 6.2 Discrete and Continuous Probabilities

Let us focus our attention on ways to describe the probability of an event as introduced in Section 6.1. Depending on whether the target space is discrete or continuous, the natural way to refer to distributions is different. When the target space $\tau$ is discrete, we can specify the probability that a random variable X takes a particular value $x \in \tau$ , denoted as $P ( X = x )$ The expression $P ( X = x )$ for a discrete random variable X is known as the probability mass function. When the target space $\tau$ is continuous, e.g., the real line <sup>R</sup>, it is more natural to specify the probability that a random variable X is in an interval, denoted by $P ( a \leqslant X \leqslant b )$ for $a < b .$ . By convention, we specify the probability that a random variable X is less than a particular value $x ,$ denoted by $P ( X \leqslant x )$ . The expression $P ( X \leqslant x )$ for a continuous random variable X is known as the cumulative distribution function. We will discuss continuous random variables in Section 6.2.2. We will revisit the nomenclature and contrast discrete and continuous random variables in Section 6.2.3. 

Remark. We will use the phrase univariate distribution to refer to distributions of a single random variable (whose states are denoted by non-bold x). We will refer to distributions of more than one random variable as multivariate distributions, and will usually consider a vector of random variables (whose states are denoted by bold x). ◇ 

## 6.2.1 Discrete Probabilities

When the target space is discrete, we can imagine the probability distribution of multiple random variables as filling out a (multidimensional) array of numbers. Figure 6.2 shows an example. The target space of the joint probability is the Cartesian product of the target spaces of each of the random variables. We define the joint probability as the entry of both values jointly 

$$
P (X = x _ {i}, Y = y _ {j}) = \frac {n _ {i j}}{N},\tag{6.9}
$$

where $n _ { i j }$ is the number of events with state $x _ { i }$ and $y _ { j }$ and N the total number of events. The joint probability is the probability of the intersection of both events, that is, $P ( X = x _ { i } , Y = y _ { j } ) = P ( X = x _ { i } \cap Y = y _ { j } )$ Figure 6.2 illustrates the probability mass function (pmf) of a discrete probability distribution. For two random variables $X$ and $Y _ { i }$ , the probability that $X = x$ and $Y = y$ is (lazily) written as $p ( x , y )$ and is called the joint probability. One can think of a probability as a function that takes state x and $y$ and returns a real number, which is the reason we write $p ( x , y )$ The marginal probability that X takes the value x irrespective of the value of random variable $Y$ is (lazily) written as $p ( x )$ . We write $X \sim p ( x )$ to denote that the random variable X is distributed according to $p ( x )$ . If we consider only the instances where $X = x ,$ , then the fraction of instances (the conditional probability) for which $Y = y$ is written (lazily) as $p ( y \mid x )$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/bd53c09c198338af1de1c2a1ebce99ec0e333c55cc3f692424572204f826a49c.jpg)



Figure 6.2 Visualization of a discrete bivariate probability mass function, with random variables X and Y . This diagram is adapted from Bishop (2006).


## Example 6.2

Consider two random variables X and $Y ,$ , where X has five possible states and $Y$ has three possible states, as shown in Figure 6.2. We denote by $n _ { i j }$ the number of events with state $X ~ = ~ x _ { i }$ and $Y ~ = ~ y _ { j }$ , and denote by $N$ the total number of events. The value $c _ { i }$ is the sum of the individual frequencies for the ith column, that is, $\begin{array} { r } { c _ { i } = \sum _ { j = 1 } ^ { 3 } n _ { i j } } \end{array}$ . Similarly, the value $r _ { j }$ is the row sum, that is, $\textstyle r _ { j } = \sum _ { i = 1 } ^ { 5 } n _ { i j }$ . Using these definitions, we can compactly express the distribution of X and $Y$ 

The probability distribution of each random variable, the marginal probability, can be seen as the sum over a row or column 

$$
P (X = x _ {i}) = \frac {c _ {i}}{N} = \frac {\sum_ {j = 1} ^ {3} n _ {i j}}{N}\tag{6.10}
$$

and 

$$
P (Y = y _ {j}) = \frac {r _ {j}}{N} = \frac {\sum_ {i = 1} ^ {5} n _ {i j}}{N},\tag{6.11}
$$

where $c _ { i }$ and $r _ { j }$ are the ith column and jth row of the probability table, respectively. By convention, for discrete random variables with a finite number of events, we assume that probabilties sum up to one, that is, 

$$
\sum_ {i = 1} ^ {5} P (X = x _ {i}) = 1 \quad \text { and } \quad \sum_ {j = 1} ^ {3} P (Y = y _ {j}) = 1.\tag{6.12}
$$

The conditional probability is the fraction of a row or column in a par-



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



ticular cell. For example, the conditional probability of Y given X is 

$$
P (Y = y _ {j} \mid X = x _ {i}) = \frac {n _ {i j}}{c _ {i}},\tag{6.13}
$$

and the conditional probability of X given Y is 

$$
P (X = x _ {i} \mid Y = y _ {j}) = \frac {n _ {i j}}{r _ {j}}.\tag{6.14}
$$

In machine learning, we use discrete probability distributions to model categorical variables, i.e., variables that take a finite set of unordered values. They could be categorical features, such as the degree taken at university when used for predicting the salary of a person, or categorical labels, such as letters of the alphabet when doing handwriting recognition. Discrete distributions are also often used to construct probabilistic models that combine a finite number of continuous distributions (Chapter 11). 

## 6.2.2 Continuous Probabilities

We consider real-valued random variables in this section, i.e., we consider target spaces that are intervals of the real line <sup>R</sup>. In this book, we pretend that we can perform operations on real random variables as if we have discrete probability spaces with finite states. However, this simplification is not precise for two situations: when we repeat something infinitely often, and when we want to draw a point from an interval. The first situation arises when we discuss generalization errors in machine learning (Chapter 8). The second situation arises when we want to discuss continuous distributions, such as the Gaussian (Section 6.5). For our purposes, the lack of precision allows for a briefer introduction to probability. 

Remark. In continuous spaces, there are two additional technicalities, which are counterintuitive. First, the set of all subsets (used to define the event space in Section 6.1) is not well behaved enough. needs to be restricted to behave well under set complements, set intersections, and set unions. Second, the size of a set (which in discrete spaces can be obtained by counting the elements) turns out to be tricky. The size of a set is called its measure. For example, the cardinality of discrete sets, the length of an interval in <sup>R</sup>, and the volume of a region in $\mathbb { R } ^ { d }$ are all measures. Sets that behave well under set operations and additionally have a topology are called a Borel σ-algebra. Betancourt details a careful construction of probability spaces from set theory without being bogged down in technicalities; see https://tinyurl.com/yb3t6mfd. For a more precise construction, we refer to Billingsley (1995) and Jacod and Protter (2004). 

In this book, we consider real-valued random variables with their corresponding Borel σ-algebra. We consider random variables with values in $\mathbb { R } ^ { \bar { D } }$ to be a vector of real-valued random variables. 

Definition 6.1 (Probability Density Function). A function $f : \mathbb { R } ^ { D } $ <sup>R</sup> is called a probability density function (pdf ) if 

1. $\forall \pmb { x } \in \mathbb { R } ^ { D } : f ( \pmb { x } ) \geqslant 0$ 

2. Its integral exists and 

$$
\int_ {\mathbb {R} ^ {D}} f (\boldsymbol {x}) \mathrm{d} \boldsymbol {x} = 1.\tag{6.15}
$$

For probability mass functions (pmf) of discrete random variables, the integral in (6.15) is replaced with a sum (6.12). 

Observe that the probability density function is any function f that is non-negative and integrates to one. We associate a random variable X with this function f by 

$$
P (a \leqslant X \leqslant b) = \int_ {a} ^ {b} f (x) \mathrm{d} x,\tag{6.16}
$$

where a, $b \in \mathbb { R }$ and $x \in \mathbb { R }$ are outcomes of the continuous random variable X. States x $\in \mathbb { R } ^ { D }$ are defined analogously by considering a vector of $x \in \mathbb { R }$ . This association (6.16) is called the law or distribution of the random variable X. 

Remark. In contrast to discrete random variables, the probability of a continuous random variable X taking a particular value $P ( X = x )$ is zero. This is like trying to specify an interval in (6.16) where $a = b .$ ◇ 

Definition 6.2 (Cumulative Distribution Function). A cumulative distribution function (cdf) of a multivariate real-valued random variable X with states $\pmb { x } \in \mathbb { R } ^ { D }$ is given by 

$$
F _ {X} (\boldsymbol {x}) = P (X _ {1} \leqslant x _ {1}, \dots , X _ {D} \leqslant x _ {D}),\tag{6.17}
$$

where $X = [ X _ { 1 } , \ldots , X _ { D } ] ^ { \top } , \pmb { x } = [ x _ { 1 } , \ldots , x _ { D } ] ^ { \top } ,$ , and the right-hand side represents the probability that random variable $X _ { i }$ takes the value smaller than or equal to $x _ { i }$ 

The cdf can be expressed also as the integral of the probability density function $f ( { \pmb x } )$ so that 

$$
F _ {X} (\boldsymbol {x}) = \int_ {- \infty} ^ {x _ {1}} \dots \int_ {- \infty} ^ {x _ {D}} f (z _ {1}, \ldots , z _ {D}) \mathrm{d} z _ {1} \dots \mathrm{d} z _ {D}.\tag{6.18}
$$

Remark. We reiterate that there are in fact two distinct concepts when talking about distributions. First is the idea of a pdf (denoted by $f ( x ) )$ , which is a nonnegative function that sums to one. Second is the law of a random variable $X$ , that is, the association of a random variable X with the pdf $f ( x )$ 



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



Figure 6.3 Examples of (a) discrete and (b) continuous uniform distributions. See Example 6.3 for details of the distributions. 

The actual values of these states are not meaningful here, and we deliberately chose numbers to drive home the point that we do not want to use (and should ignore) the ordering of the states. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/1d5ccffbbb54f8c77ae9a5e82167a062fd65a0e88662c6926b77924fe6b1af56.jpg)



(a) Discrete distribution


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/7019ed1778eb269d421a68765765d8e4b786c61535e3116d64e81ad16cf13cae.jpg)



(b) Continuous distribution


For most of this book, we will not use the notation $f ( x )$ and $F _ { X } ( x )$ as we mostly do not need to distinguish between the pdf and cdf. However, we will need to be careful about pdfs and cdfs in Section 6.7. 

## 6.2.3 Contrasting Discrete and Continuous Distributions

Recall from Section 6.1.2 that probabilities are positive and the total probability sums up to one. For discrete random variables (see (6.12)), this implies that the probability of each state must lie in the interval [0, 1]. However, for continuous random variables the normalization (see (6.15)) does not imply that the value of the density is less than or equal to 1 for all values. We illustrate this in Figure 6.3 using the uniform distribution for both discrete and continuous random variables. 

## Example 6.3

We consider two examples of the uniform distribution, where each state is equally likely to occur. This example illustrates some differences between discrete and continuous probability distributions. 

Let Z be a discrete uniform random variable with three states $\{ z =$ $- 1 . 1 , z = 0 . 3 , z = 1 . 5 \}$ . The probability mass function can be represented as a table of probability values: 

$$
\begin{array}{c c c c} z & - 1. 1 & 0. 3 & 1. 5 \\ P (Z = z) \framebox {\frac {1}{3}} & \framebox {\frac {1}{3}} & \framebox {\frac {1}{3}} \end{array}
$$

Alternatively, we can think of this as a graph (Figure 6.3(a)), where we use the fact that the states can be located on the x-axis, and the y-axis represents the probability of a particular state. The y-axis in Figure 6.3(a) is deliberately extended so that is it the same as in Figure 6.3(b). 

Let X be a continuous random variable taking values in the range 0.9 <sup>⩽</sup> $X \leqslant 1 . 6 _ { : }$ , as represented by Figure 6.3(b). Observe that the height of the density can be greater than 1. However, it needs to hold that 


6.3 Sum Rule, Product Rule, and Bayes’ Theorem


<table><tr><td>Type</td><td>“Point probability”</td><td>“Interval probability”</td></tr><tr><td>Discrete</td><td><eq>P(X = x)</eq>Probability mass function</td><td>Not applicable</td></tr><tr><td>Continuous</td><td><eq>p(x)</eq>Probability density function</td><td><eq>P(X \leqslant x)</eq>Cumulative distribution function</td></tr></table>

$$
\int_ {0. 9} ^ {1. 6} p (x) \mathrm{d} x = 1.\tag{6.19}
$$

Remark. There is an additional subtlety with regards to discrete probability distributions. The states $z _ { 1 } , \ldots , z _ { d }$ do not in principle have any structure, i.e., there is usually no way to compare them, for example $z _ { 1 } = \mathrm { r e d } , z _ { 2 } = \mathrm { g r e e n } , z _ { 3 } = \mathrm { b l u e }$ . However, in many machine learning applications discrete states take numerical values, $\mathrm { e . g . , } z _ { 1 } = - 1 . 1 , z _ { 2 } =$ $0 . 3 , z _ { 3 } = 1 . 5$ , where we could say $z _ { 1 } < z _ { 2 } < z _ { 3 }$ . Discrete states that assume numerical values are particularly useful because we often consider expected values (Section 6.4.1) of random variables. 

Unfortunately, machine learning literature uses notation and nomenclature that hides the distinction between the sample space $\Omega ,$ the target space $\tau ,$ and the random variable X. For a value x of the set of possible outcomes of the random variable X, i.e., $x \in \mathcal { T } , p ( x )$ denotes the probability that random variable X has the outcome x. For discrete random variables, this is written as $P ( X = x )$ , which is known as the probability mass function. The pmf is often referred to as the “distribution”. For continuous variables, $p ( x )$ is called the probability density function (often referred to as a density). To muddy things even further, the cumulative distribution function $P ( X \leqslant x )$ is often also referred to as the “distribution”. In this chapter, we will use the notation X to refer to both univariate and multivariate random variables, and denote the states by x and x respectively. We summarize the nomenclature in Table 6.1. 

Remark. We will be using the expression “probability distribution” not only for discrete probability mass functions but also for continuous probability density functions, although this is technically incorrect. In line with most machine learning literature, we also rely on context to distinguish the different uses of the phrase probability distribution. 

## 6.3 Sum Rule, Product Rule, and Bayes’ Theorem

We think of probability theory as an extension to logical reasoning. As we discussed in Section 6.1.1, the rules of probability presented here follow 

Table 6.1 Nomenclature for probability distributions. 

We think of the outcome x as the argument that results in the probability p(x). 

naturally from fulfilling the desiderata (Jaynes, 2003, chapter 2). Probabilistic modeling (Section 8.4) provides a principled foundation for designing machine learning methods. Once we have defined probability distributions (Section 6.2) corresponding to the uncertainties of the data and our problem, it turns out that there are only two fundamental rules, the sum rule and the product rule. 

Recall from (6.9) that $p ( { \pmb x } , { \pmb y } )$ is the joint distribution of the two random variables $\mathbf { \nabla } _ { \mathbf { x } , \mathbf { y } }$ . The distributions $p ( { \pmb x } )$ and $p ( \pmb { y } )$ are the corresponding marginal distributions, and $p ( \pmb { y } \vert \pmb { x } )$ is the conditional distribution of y given x. Given the definitions of the marginal and conditional probability for discrete and continuous random variables in Section 6.2, we can now present the two fundamental rules in probability theory. 

The first rule, the sum rule, states that 

$$
p (\boldsymbol {x}) = \left\{ \begin{array}{l l} \sum_ {\boldsymbol {y} \in \mathcal {Y}} p (\boldsymbol {x}, \boldsymbol {y}) & \text { if } \boldsymbol {y} \text { is   discrete } \\ \int_ {\mathcal {Y}} p (\boldsymbol {x}, \boldsymbol {y}) \mathrm{d} \boldsymbol {y} & \text { if } \boldsymbol {y} \text { is   continuous } \end{array} \right.,\tag{6.20}
$$

where $\mathcal { V }$ are the states of the target space of random variable Y . This means that we sum out (or integrate out) the set of states y of the random variable Y . The sum rule is also known as the marginalization property. The sum rule relates the joint distribution to a marginal distribution. In general, when the joint distribution contains more than two random variables, the sum rule can be applied to any subset of the random variables, resulting in a marginal distribution of potentially more than one random variable. More concretely, if $\pmb { x } = [ x _ { 1 } , \ldots , x _ { D } ] ^ { \top }$ , we obtain the marginal 

$$
p (x _ {i}) = \int p (x _ {1}, \dots , x _ {D}) \mathrm{d} \boldsymbol {x} _ {\backslash i}\tag{6.21}
$$

by repeated application of the sum rule where we integrate/sum out all random variables except $x _ { i } ,$ which is indicated by $\backslash i ,$ , which reads $^ { \mathfrak { s } } \mathrm { a l l }$ except i.” 

Remark. Many of the computational challenges of probabilistic modeling are due to the application of the sum rule. When there are many variables or discrete variables with many states, the sum rule boils down to performing a high-dimensional sum or integral. Performing high-dimensional sums or integrals is generally computationally hard, in the sense that there is no known polynomial-time algorithm to calculate them exactly. ◇ 

The second rule, known as the product rule, relates the joint distribution to the conditional distribution via 

$$
p (\boldsymbol {x}, \boldsymbol {y}) = p (\boldsymbol {y} \mid \boldsymbol {x}) p (\boldsymbol {x}).\tag{6.22}
$$

The product rule can be interpreted as the fact that every joint distribution of two random variables can be factorized (written as a product) 

of two other distributions. The two factors are the marginal distribution of the first random variable $p ( { \pmb x } )$ , and the conditional distribution of the second random variable given the first $p ( \pmb { y } \vert \pmb { x } )$ . Since the ordering of random variables is arbitrary in $p ( { \pmb x } , { \pmb y } )$ , the product rule also implies $p ( { \pmb x } , { \pmb y } ) = p ( { \pmb x } \mid { \pmb y } ) p ( { \pmb y } )$ . To be precise, (6.22) is expressed in terms of the probability mass functions for discrete random variables. For continuous random variables, the product rule is expressed in terms of the probability density functions (Section 6.2.3). 

In machine learning and Bayesian statistics, we are often interested in making inferences of unobserved (latent) random variables given that we have observed other random variables. Let us assume we have some prior knowledge $p ( { \pmb x } )$ about an unobserved random variable $_ { \pmb { x } }$ and some relationship $p ( \pmb { y } \vert \pmb { x } )$ between x and a second random variable $^ { y , }$ which we can observe. If we observe $^ { y , }$ we can use Bayes’ theorem to draw some conclusions about x given the observed values of $\mathbf { \pmb { y } } .$ Bayes’ theorem (also Bayes’ rule or Bayes’ law) 

$$
\underbrace {p (\boldsymbol {x} \mid \boldsymbol {y})} _ {\text { posterior }} = \frac {\overbrace {p (\boldsymbol {y} \mid \boldsymbol {x})} ^ {\text { likelihood   prior }} \overbrace {p (\boldsymbol {x})} ^ {\text { evidence }}}{\underbrace {p (\boldsymbol {y})} _ {\text { evidence }}}\tag{6.23}
$$

is a direct consequence of the product rule in (6.22) since 

$$
p (\boldsymbol {x}, \boldsymbol {y}) = p (\boldsymbol {x} \mid \boldsymbol {y}) p (\boldsymbol {y})\tag{6.24}
$$

and 

$$
p (\pmb {x}, \pmb {y}) = p (\pmb {y} | \pmb {x}) p (\pmb {x})\tag{6.25}
$$

so that 

$$
p (\boldsymbol {x} \mid \boldsymbol {y}) p (\boldsymbol {y}) = p (\boldsymbol {y} \mid \boldsymbol {x}) p (\boldsymbol {x}) \iff p (\boldsymbol {x} \mid \boldsymbol {y}) = \frac {p (\boldsymbol {y} \mid \boldsymbol {x}) p (\boldsymbol {x})}{p (\boldsymbol {y})}.\tag{6.26}
$$

In (6.23), $p ( { \pmb x } )$ is the $p r i o r _ { \mathrm { { i } } }$ , which encapsulates our subjective prior knowledge of the unobserved (latent) variable x before observing any data. We can choose any prior that makes sense to us, but it is critical to ensure that the prior has a nonzero pdf (or pmf) on all plausible x, even if they are very rare. 

The likelihood $p ( \pmb { y } \vert \pmb { x } )$ describes how x and $\textbf {  { y } }$ are related, and in the case of discrete probability distributions, it is the probability of the data y if we were to know the latent variable x. Note that the likelihood is not a distribution in ${ \mathbf { } } ^ { \mathbf { } } \mathbf { { \mathbf { { x } } } } ,$ but only in $\mathbf { \pmb { y } } .$ . We call $p ( \pmb { y } \vert \pmb { x } )$ either the “likelihood of x (given $y ) ^ { \dag }$ or the “probability of $\textbf {  { y } }$ given $\pmb { x } ^ { \prime \prime }$ but never the likelihood of $\textbf {  { y } }$ (MacKay, 2003). 

The posterior $p ( { \pmb x } \mid { \pmb y } )$ is the quantity of interest in Bayesian statistics because it expresses exactly what we are interested in, i.e., what we know about x after having observed $\pmb { y }$ 

The quantity 

$$
p (\boldsymbol {y}) := \int p (\boldsymbol {y} \mid \boldsymbol {x}) p (\boldsymbol {x}) \mathrm{d} \boldsymbol {x} = \mathbb {E} _ {X} [ p (\boldsymbol {y} \mid \boldsymbol {x}) ]\tag{6.27}
$$

is the marginal likelihood/evidence. The right-hand side of (6.27) uses the expectation operator which we define in Section 6.4.1. By definition, the marginal likelihood integrates the numerator of (6.23) with respect to the latent variable x. Therefore, the marginal likelihood is independent of x, and it ensures that the posterior $p ( { \pmb x } \mid { \pmb y } )$ is normalized. The marginal likelihood can also be interpreted as the expected likelihood where we take the expectation with respect to the prior $p ( { \pmb x } )$ ). Beyond normalization of the posterior, the marginal likelihood also plays an important role in Bayesian model selection, as we will discuss in Section 8.6. Due to the integration in (8.44), the evidence is often hard to compute. 

Bayes’ theorem (6.23) allows us to invert the relationship between x and y given by the likelihood. Therefore, Bayes’ theorem is sometimes called the probabilistic inverse. We will discuss Bayes’ theorem further in Section 8.4. 

Remark. In Bayesian statistics, the posterior distribution is the quantity of interest as it encapsulates all available information from the prior and the data. Instead of carrying the posterior around, it is possible to focus on some statistic of the posterior, such as the maximum of the posterior, which we will discuss in Section 8.3. However, focusing on some statistic of the posterior leads to loss of information. If we think in a bigger context, then the posterior can be used within a decision-making system, and having the full posterior can be extremely useful and lead to decisions that are robust to disturbances. For example, in the context of model-based reinforcement learning, Deisenroth et al. (2015) show that using the ful posterior distribution of plausible transition functions leads to very fast (data/sample efficient) learning, whereas focusing on the maximum of the posterior leads to consistent failures. Therefore, having the full posterior can be very useful for a downstream task. In Chapter 9, we will continue this discussion in the context of linear regression. ◇ 

## 6.4 Summary Statistics and Independence

We are often interested in summarizing sets of random variables and comparing pairs of random variables. A statistic of a random variable is a deterministic function of that random variable. The summary statistics of a distribution provide one useful view of how a random variable behaves, and as the name suggests, provide numbers that summarize and characterize the distribution. We describe the mean and the variance, two wellknown summary statistics. Then we discuss two ways to compare a pair of random variables: first, how to say that two random variables are independent; and second, how to compute an inner product between them. 

## 6.4.1 Means and Covariances

Mean and (co)variance are often useful to describe properties of probability distributions (expected values and spread). We will see in Section 6.6 that there is a useful family of distributions (called the exponential family), where the statistics of the random variable capture all possible information. 

The concept of the expected value is central to machine learning, and the foundational concepts of probability itself can be derived from the expected value (Whittle, 2000). 

Definition 6.3 (Expected Value). The expected value of a function $g : \mathbb { R } $ <sup>R</sup> of a univariate continuous random variable $X \sim p ( x )$ is given by 

$$
\mathbb {E} _ {X} [ g (x) ] = \int_ {\mathcal {X}} g (x) p (x) \mathrm{d} x.\tag{6.28}
$$

Correspondingly, the expected value of a function $g$ of a discrete random variable $X \sim p ( x )$ is given by 

$$
\mathbb {E} _ {X} [ g (x) ] = \sum_ {x \in \mathcal {X}} g (x) p (x),\tag{6.29}
$$

where is the set of possible outcomes (the target space) of the random variable X. 

In this section, we consider discrete random variables to have numerical outcomes. This can be seen by observing that the function g takes real numbers as inputs. 

Remark. We consider multivariate random variables X as a finite vector of univariate random variables $[ X _ { 1 } , \ldots , X _ { D } ] ^ { \top }$ . For multivariate random variables, we define the expected value element wise 

$$
\mathbb {E} _ {X} [ g (\boldsymbol {x}) ] = \left[ \begin{array}{c} \mathbb {E} _ {X _ {1}} [ g (x _ {1}) ] \\ \vdots \\ \mathbb {E} _ {X _ {D}} [ g (x _ {D}) ] \end{array} \right] \in \mathbb {R} ^ {D},
$$

(6.30) 

where the subscript $\mathbb { E } _ { X _ { d } }$ indicates that we are taking the expected value with respect to the dth element of the vector x. $\diamondsuit$ 

Definition 6.3 defines the meaning of the notation $\mathbb { E } _ { X }$ as the operator indicating that we should take the integral with respect to the probabil ity density (for continuous distributions) or the sum over all states (for discrete distributions). The definition of the mean (Definition 6.4), is a special case of the expected value, obtained by choosing g to be the identity function. 

Definition 6.4 (Mean). The mean of a random variable X with states 

x $\in \mathbb { R } ^ { D }$ is an average and is defined as 

$$
\mathbb {E} _ {X} [ \boldsymbol {x} ] = \left[ \begin{array}{c} \mathbb {E} _ {X _ {1}} [ x _ {1} ] \\ \vdots \\ \mathbb {E} _ {X _ {D}} [ x _ {D} ] \end{array} \right] \in \mathbb {R} ^ {D},\tag{6.31}
$$

where 

$$
\mathbb {E} _ {X _ {d}} [ x _ {d} ] := \left\{ \begin{array}{l l} \int_ {\mathcal {X}} x _ {d} p (x _ {d}) \mathrm{d} x _ {d} & \text { if   X   is   a   continuous   random   variable } \\ \sum_ {x _ {i} \in \mathcal {X}} x _ {i} p (x _ {d} = x _ {i}) & \text { if   X   is   a   discrete   random   variable } \end{array} \right.\tag{6.32}
$$

for $d = 1 , \ldots , D$ , where the subscript d indicates the corresponding dimension of x. The integral and sum are over the states $\mathcal { X }$ of the target space of the random variable X. 

In one dimension, there are two other intuitive notions of “average”, which are the median and the mode. The median is the “middle” value if we sort the values, i.e., 50% of the values are greater than the median and 50% are smaller than the median. This idea can be generalized to continuous values by considering the value where the cdf (Definition 6.2) is 0.5. For distributions, which are asymmetric or have long tails, the median provides an estimate of a typical value that is closer to human intuition than the mean value. Furthermore, the median is more robust to outliers than the mean. The generalization of the median to higher dimensions is non-trivial as there is no obvious way to $\mathbf { \Delta } ^ { \mathfrak { s o r t } ^ { \mathfrak { n } } }$ in more than one dimension (Hallin et al., 2010; Kong and Mizera, 2012). The mode is the most frequently occurring value. For a discrete random variable, the mode is defined as the value of $x$ having the highest frequency of occurrence. For a continuous random variable, the mode is defined as a peak in the density $p ( { \pmb x } )$ . A particular density $p ( { \pmb x } )$ may have more than one mode, and furthermore there may be a very large number of modes in high-dimensional distributions. Therefore, finding all the modes of a distribution can be computationally challenging. 

## Example 6.4

Consider the two-dimensional distribution illustrated in Figure 6.4: 

$$
p (x) = 0. 4 \mathcal {N} \left(\boldsymbol {x} \mid \left[ \begin{array}{c} 1 0 \\ 2 \end{array} \right], \left[ \begin{array}{c c} 1 & 0 \\ 0 & 1 \end{array} \right]\right) + 0. 6 \mathcal {N} \left(\boldsymbol {x} \mid \left[ \begin{array}{c} 0 \\ 0 \end{array} \right], \left[ \begin{array}{c c} 8. 4 & 2. 0 \\ 2. 0 & 1. 7 \end{array} \right]\right).\tag{6.33}
$$

We will define the Gaussian distribution ${ \mathcal { N } } ( \mu , \sigma ^ { 2 } )$ in Section 6.5. Also shown is its corresponding marginal distribution in each dimension. Observe that the distribution is bimodal (has two modes), but one of the marginal distributions is unimodal (has one mode). The horizontal bimodal univariate distribution illustrates that the mean and median can be different from each other. While it is tempting to define the twodimensional median to be the concatenation of the medians in each dimension, the fact that we cannot define an ordering of two-dimensional points makes it difficult. When we say “cannot define an ordering”, we mean that there is more than one way to define the relation $< s 0$ that 

$$
\left[ \begin{array}{c} 3 \\ 0 \end{array} \right] <   \left[ \begin{array}{c} 2 \\ 3 \end{array} \right].
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/534ed302f489a8c5028a0da3331e3537fefb839fff9306a674a6c10eaeb63d8b.jpg)



Figure 6.4 Illustration of the mean, mode, and median for a two-dimensional dataset, as well as its marginal densities.


Remark. The expected value (Definition 6.3) is a linear operator. For example, given a real-valued function $f ( { \pmb x } ) = a g ( { \pmb x } ) + b h ( { \pmb x } )$ where $a , b \in \mathbb { R }$ and $\pmb { x } \in \mathbb { R } ^ { D }$ , we obtain 

$$
\mathbb {E} _ {X} [ f (\boldsymbol {x}) ] = \int f (\boldsymbol {x}) p (\boldsymbol {x}) \mathrm{d} \boldsymbol {x}\tag{6.34a}
$$

$$
= \int [ a g (\pmb {x}) + b h (\pmb {x}) ] p (\pmb {x}) \mathrm{d} \pmb {x}\tag{6.34b}
$$

$$
= a \int g (\boldsymbol {x}) p (\boldsymbol {x}) \mathrm{d} x + b \int h (\boldsymbol {x}) p (\boldsymbol {x}) \mathrm{d} \boldsymbol {x}\tag{6.34c}
$$

$$
= a \mathbb {E} _ {X} [ g (\boldsymbol {x}) ] + b \mathbb {E} _ {X} [ h (\boldsymbol {x}) ].\tag{6.34d}
$$

For two random variables, we may wish to characterize their correspon-

©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 

dence to each other. The covariance intuitively represents the notion of how dependent random variables are to one another. 

Definition 6.5 (Covariance (Univariate)). The covariance between two univariate random variables $X , Y \in \mathbb { R }$ is given by the expected product of their deviations from their respective means, i.e., 

$$
\operatorname{Cov} _ {X, Y} [ x, y ] := \mathbb {E} _ {X, Y} \bigl [ (x - \mathbb {E} _ {X} [ x ]) (y - \mathbb {E} _ {Y} [ y ]) \bigr ].\tag{6.35}
$$

Remark. When the random variable associated with the expectation or covariance is clear by its arguments, the subscript is often suppressed (for example, $\mathbb { E } _ { X } [ x ]$ is often written as <sup>E</sup>[x]). ◇ 

By using the linearity of expectations, the expression in Definition 6.5 can be rewritten as the expected value of the product minus the product of the expected values, i.e., 

$$
\operatorname{Cov} [ x, y ] = \mathbb {E} [ x y ] - \mathbb {E} [ x ] \mathbb {E} [ y ].\tag{6.36}
$$

The covariance of a variable with itself Cov[x, x] is called the variance and is denoted by $\mathbb { V } _ { X } [ x ]$ . The square root of the variance is called the standard deviation and is often denoted by $\sigma ( x )$ . The notion of covariance can be generalized to multivariate random variables. 

Definition 6.6 (Covariance (Multivariate)). If we consider two multivariate random variables X and Y with states $\pmb { x } \in \mathbb { R } ^ { D }$ and $\pmb { y } \in \mathbb { R } ^ { E }$ respectively, the covariance between X and Y is defined as 

$$
\operatorname{Cov} [ \boldsymbol {x}, \boldsymbol {y} ] = \mathbb {E} [ \boldsymbol {x} \boldsymbol {y} ^ {\top} ] - \mathbb {E} [ \boldsymbol {x} ] \mathbb {E} [ \boldsymbol {y} ] ^ {\top} = \operatorname{Cov} [ \boldsymbol {y}, \boldsymbol {x} ] ^ {\top} \in \mathbb {R} ^ {D \times E}.\tag{6.37}
$$

Definition 6.6 can be applied with the same multivariate random variable in both arguments, which results in a useful concept that intuitively captures the “spread” of a random variable. For a multivariate random variable, the variance describes the relation between individual dimensions of the random variable. 

Definition 6.7 (Variance). The variance of a random variable X with states $\pmb { x } \in \mathbb { R } ^ { D }$ and a mean vector $\pmb { \mu } \in \mathbb { R } ^ { D }$ is defined as 

$$
\mathbb {V} _ {X} [ \pmb {x} ] = \mathrm{Cov} _ {X} [ \pmb {x}, \pmb {x} ]\tag{6.38a}
$$

$$
= \mathbb {E} _ {X} [ (\pmb {x} - \pmb {\mu}) (\pmb {x} - \pmb {\mu}) ^ {\top} ] = \mathbb {E} _ {X} [ \pmb {x} \pmb {x} ^ {\top} ] - \mathbb {E} _ {X} [ \pmb {x} ] \mathbb {E} _ {X} [ \pmb {x} ] ^ {\top}\tag{6.38b}
$$

$$
= \left[ \begin{array}{c c c c} \operatorname{Cov} [ x _ {1}, x _ {1} ] & \operatorname{Cov} [ x _ {1}, x _ {2} ] & \ldots & \operatorname{Cov} [ x _ {1}, x _ {D} ] \\ \operatorname{Cov} [ x _ {2}, x _ {1} ] & \operatorname{Cov} [ x _ {2}, x _ {2} ] & \ldots & \operatorname{Cov} [ x _ {2}, x _ {D} ] \\ \vdots & \vdots & \ddots & \vdots \\ \operatorname{Cov} [ x _ {D}, x _ {1} ] & \ldots & \ldots & \operatorname{Cov} [ x _ {D}, x _ {D} ] \end{array} \right].\tag{6.38c}
$$

The $D \times D$ matrix in (6.38c) is called the covariance matrix of the multivariate random variable X. The covariance matrix is symmetric and positive semidefinite and tells us something about the spread of the data. On its diagonal, the covariance matrix contains the variances of the marginals 


6.4 Summary Statistics and Independence


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/e3b259cdf12848d29847351e7a58024fc32b6e2fa8d7dceb03b2ba23d80f18f9.jpg)



(a) x and y are negatively correlated.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/81742ee9594f1a146b35adcfae7b493796505df6bf368c79552488807bac7d4c.jpg)



(b) x and y are positively correlated.


$$
p (x _ {i}) = \int p (x _ {1}, \dots , x _ {D}) \mathrm{d} x _ {\backslash i},\tag{6.39}
$$

where $^ { \mathfrak { a } } \backslash i ^ { \mathfrak { n } }$ denotes “all variables but $i ^ { \mathfrak { n } } .$ . The off-diagonal entries are the cross-covariance terms $\operatorname { C o v } [ x _ { i } , x _ { j } ]$ for $i , j = 1 , \ldots , D , \ i \neq j$ 

Remark. In this book, we generally assume that covariance matrices are positive definite to enable better intuition. We therefore do not discuss corner cases that result in positive semidefinite (low-rank) covariance matrices. 

When we want to compare the covariances between different pairs of random variables, it turns out that the variance of each random variable affects the value of the covariance. The normalized version of covariance is called the correlation. 

Definition 6.8 (Correlation). The correlation between two random variables $X , Y$ is given by 

$$
\operatorname{corr} [ x, y ] = \frac {\operatorname{Cov} [ x , y ]}{\sqrt {\mathbb {V} [ x ] \mathbb {V} [ y ]}} \in [ - 1, 1 ].\tag{6.40}
$$

The correlation matrix is the covariance matrix of standardized random variables, $x / \sigma ( x )$ . In other words, each random variable is divided by its standard deviation (the square root of the variance) in the correlation matrix. 

The covariance (and correlation) indicate how two random variables are related; see Figure 6.5. Positive correlation corr[x, y] means that when x grows, then $y$ is also expected to grow. Negative correlation means that as x increases, then y decreases. 

## 6.4.2 Empirical Means and Covariances

The definitions in Section 6.4.1 are often also called the population mean and covariance, as it refers to the true statistics for the population. In machine learning, we need to learn from empirical observations of data. Consider a random variable X. There are two conceptual steps to go from population statistics to the realization of empirical statistics. First, we use the fact that we have a finite dataset (of size N ) to construct an empirical statistic that is a function of a finite number of identical random variables, $X _ { 1 } , \ldots , X _ { N }$ . Second, we observe the data, that is, we look at the realization $x _ { 1 } , \ldots , x _ { N }$ of each of the random variables and apply the empirical statistic. 

Specifically, for the mean (Definition 6.4), given a particular dataset we can obtain an estimate of the mean, which is called the empirical mean or sample mean. The same holds for the empirical covariance. 

Definition 6.9 (Empirical Mean and Covariance). The empirical mean vector is the arithmetic average of the observations for each variable, and it is defined as 

$$
\bar {\boldsymbol {x}} := \frac {1}{N} \sum_ {n = 1} ^ {N} \boldsymbol {x} _ {n},\tag{6.41}
$$

where ${ \pmb x } _ { n } \in \mathbb { R } ^ { D }$ 

Similar to the empirical mean, the empirical covariance matrix is a $D \times D$ matrix 

$$
\boldsymbol {\Sigma} := \frac {1}{N} \sum_ {n = 1} ^ {N} (\boldsymbol {x} _ {n} - \bar {\boldsymbol {x}}) (\boldsymbol {x} _ {n} - \bar {\boldsymbol {x}}) ^ {\top}.\tag{6.42}
$$

To compute the statistics for a particular dataset, we would use the realizations (observations) $\pmb { x } _ { 1 } , \ldots , \pmb { x } _ { N }$ and use (6.41) and (6.42). Empirical covariance matrices are symmetric, positive semidefinite (see Section 3.2.3). 

## 6.4.3 Three Expressions for the Variance

We now focus on a single random variable X and use the preceding empirical formulas to derive three possible expressions for the variance. The following derivation is the same for the population variance, except that we need to take care of integrals. The standard definition of variance, corresponding to the definition of covariance (Definition 6.5), is the expectation of the squared deviation of a random variable X from its expected value $\mu , \mathrm { i . e . }$ , 

$$
\mathbb {V} _ {X} [ x ] := \mathbb {E} _ {X} [ (x - \mu) ^ {2} ].\tag{6.43}
$$

The expectation in (6.43) and the mean $\mu = \operatorname { \mathbb { E } } _ { X } ( x )$ are computed using (6.32), depending on whether X is a discrete or continuous random variable. The variance as expressed in (6.43) is the mean of a new random variable $Z : = ( X - \mu ) ^ { 2 }$ 

When estimating the variance in (6.43) empirically, we need to resort to a two-pass algorithm: one pass through the data to calculate the mean $\mu$ using (6.41), and then a second pass using this estimate $\hat { \mu }$ calculate the variance. It turns out that we can avoid two passes by rearranging the terms. The formula in (6.43) can be converted to the so-called raw-score formula for variance: 

$$
\mathbb {V} _ {X} [ x ] = \mathbb {E} _ {X} [ x ^ {2} ] - \left(\mathbb {E} _ {X} [ x ]\right) ^ {2}.\tag{6.44}
$$

The expression in (6.44) can be remembered as “the mean of the square minus the square of the mean”. It can be calculated empirically in one pass through data since we can accumulate $x _ { i }$ (to calculate the mean) and $x _ { i } ^ { 2 }$ simultaneously, where $x _ { i }$ is the ith observation. Unfortunately, if implemented in this way, it can be numerically unstable. The raw-score version of the variance can be useful in machine learning, e.g., when deriving the bias–variance decomposition (Bishop, 2006). 

A third way to understand the variance is that it is a sum of pairwise differences between all pairs of observations. Consider a sample $x _ { 1 } , \ldots , x _ { N }$ of realizations of random variable X, and we compute the squared difference between pairs of $x _ { i }$ and $x _ { j }$ . By expanding the square, we can show that the sum of $N ^ { 2 }$ pairwise differences is the empirical variance of the observations: 

$$
\frac {1}{N ^ {2}} \sum_ {i, j = 1} ^ {N} (x _ {i} - x _ {j}) ^ {2} = 2 \left[ \frac {1}{N} \sum_ {i = 1} ^ {N} x _ {i} ^ {2} - \left(\frac {1}{N} \sum_ {i = 1} ^ {N} x _ {i}\right) ^ {2} \right].\tag{6.45}
$$

We see that (6.45) is twice the raw-score expression (6.44). This means that we can express the sum of pairwise distances (of which there are $N ^ { 2 }$ of them) as a sum of deviations from the mean (of which there are N ). Geometrically, this means that there is an equivalence between the pairwise distances and the distances from the center of the set of points. From a computational perspective, this means that by computing the mean (N terms in the summation), and then computing the variance (again N terms in the summation), we can obtain an expression (left-hand side of (6.45)) that has $N ^ { 2 }$ terms. 

## 6.4.4 Sums and Transformations of Random Variables

We may want to model a phenomenon that cannot be well explained by textbook distributions (we introduce some in Sections 6.5 and 6.6), and hence may perform simple manipulations of random variables (such as adding two random variables). 

Consider two random variables X, Y with states x, $\pmb { y } \in \mathbb { R } ^ { D }$ . Then: 

$$
\mathbb {E} [ \boldsymbol {x} + \boldsymbol {y} ] = \mathbb {E} [ \boldsymbol {x} ] + \mathbb {E} [ \boldsymbol {y} ]\tag{6.46}
$$

$$
\mathbb {E} [ \boldsymbol {x} - \boldsymbol {y} ] = \mathbb {E} [ \boldsymbol {x} ] - \mathbb {E} [ \boldsymbol {y} ]\tag{6.47}
$$

$$
\mathbb {V} [ \boldsymbol {x} + \boldsymbol {y} ] = \mathbb {V} [ \boldsymbol {x} ] + \mathbb {V} [ \boldsymbol {y} ] + \operatorname{Cov} [ \boldsymbol {x}, \boldsymbol {y} ] + \operatorname{Cov} [ \boldsymbol {y}, \boldsymbol {x} ]\tag{6.48}
$$

$$
\mathbb {V} [ \boldsymbol {x} - \boldsymbol {y} ] = \mathbb {V} [ \boldsymbol {x} ] + \mathbb {V} [ \boldsymbol {y} ] - \operatorname{Cov} [ \boldsymbol {x}, \boldsymbol {y} ] - \operatorname{Cov} [ \boldsymbol {y}, \boldsymbol {x} ].\tag{6.49}
$$



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



Mean and (co)variance exhibit some useful properties when it comes to affine transformation of random variables. Consider a random variable $X$ with mean $\pmb { \mu }$ and covariance matrix $\pmb { \Sigma }$ and a (deterministic) affine transformation $\pmb { y } = \pmb { A x } + \pmb { b }$ of x. Then $\textbf {  { y } }$ is itself a random variable whose mean vector and covariance matrix are given by 

$$
\mathbb {E} _ {Y} [ \boldsymbol {y} ] = \mathbb {E} _ {X} [ \boldsymbol {A} \boldsymbol {x} + \boldsymbol {b} ] = \boldsymbol {A} \mathbb {E} _ {X} [ \boldsymbol {x} ] + \boldsymbol {b} = \boldsymbol {A} \boldsymbol {\mu} + \boldsymbol {b},\tag{6.50}
$$

$$
\mathbb {V} _ {Y} [ \boldsymbol {y} ] = \mathbb {V} _ {X} [ \boldsymbol {A x} + \boldsymbol {b} ] = \mathbb {V} _ {X} [ \boldsymbol {A x} ] = \boldsymbol {A} \mathbb {V} _ {X} [ \boldsymbol {x} ] \boldsymbol {A} ^ {\top} = \boldsymbol {A} \boldsymbol {\Sigma} \boldsymbol {A} ^ {\top},\tag{6.51}
$$

This can be shown directly by using the definition of the mean and covariance. 

respectively. Furthermore, 

$$
\operatorname{Cov} [ \boldsymbol {x}, \boldsymbol {y} ] = \mathbb {E} [ \boldsymbol {x} (\boldsymbol {A x} + \boldsymbol {b}) ^ {\top} ] - \mathbb {E} [ \boldsymbol {x} ] \mathbb {E} [ \boldsymbol {A x} + \boldsymbol {b} ] ^ {\top}
$$

$$
= \mathbb {E} [ \boldsymbol {x} ] \boldsymbol {b} ^ {\top} + \mathbb {E} [ \boldsymbol {x x} ^ {\top} ] \boldsymbol {A} ^ {\top} - \boldsymbol {\mu b} ^ {\top} - \boldsymbol {\mu \mu} ^ {\top} \boldsymbol {A} ^ {\top}\tag{6.52a}
$$

(6.52b) 

$$
= \boldsymbol {\mu} \boldsymbol {b} ^ {\top} - \boldsymbol {\mu} \boldsymbol {b} ^ {\top} + \left(\mathbb {E} [ \boldsymbol {x x} ^ {\top} ] - \boldsymbol {\mu} \boldsymbol {\mu} ^ {\top}\right) \boldsymbol {A} ^ {\top}\tag{6.52c}
$$

$$
\stackrel {(6. 3 8 \mathrm{b})} {=} \boldsymbol {\Sigma} \boldsymbol {A} ^ {\top},\tag{6.52d}
$$

where $\pmb { \Sigma } = \mathbb { E } [ \pmb { x } \pmb { x } ^ { \top } ] - \pmb { \mu } \pmb { \mu } ^ { \top }$ is the covariance of X. 

## 6.4.5 Statistical Independence

Definition 6.10 (Independence). Two random variables $X , Y$ are statistically independent if and only if 

$$
p (\boldsymbol {x}, \boldsymbol {y}) = p (\boldsymbol {x}) p (\boldsymbol {y}).\tag{6.53}
$$

Intuitively, two random variables X and $Y$ are independent if the value of $\textbf {  { y } }$ (once known) does not add any additional information about x (and vice versa). If $X , Y$ are (statistically) independent, then 

p(y <sub>|</sub> x) = p(y) 

$p ( { \pmb x } | { \pmb y } ) = p ( { \pmb x } )$ 

$$
\mathbb {V} _ {X, Y} [ \boldsymbol {x} + \boldsymbol {y} ] = \mathbb {V} _ {X} [ \boldsymbol {x} ] + \mathbb {V} _ {Y} [ \boldsymbol {y} ]
$$

$$
\mathbf {\Phi} \operatorname{Cov} _ {X, Y} [ \boldsymbol {x}, \boldsymbol {y} ] = \mathbf {0}
$$

The last point may not hold in converse, i.e., two random variables can have covariance zero but are not statistically independent. To understand why, recall that covariance measures only linear dependence. Therefore, random variables that are nonlinearly dependent could have covariance zero. 

## Example 6.5

Consider a random variable X with zero mean $( \mathbb { E } _ { X } [ x ] = 0 )$ and also $\mathbb { E } _ { X } [ x ^ { 3 } ] = 0$ . Let $y = x ^ { 2 }$ (hence, $Y$ is dependent on $X )$ and consider the covariance (6.36) between X and Y . But this gives 

$$
\operatorname{Cov} [ x, y ] = \mathbb {E} [ x y ] - \mathbb {E} [ x ] \mathbb {E} [ y ] = \mathbb {E} [ x ^ {3} ] = 0.\tag{6.54}
$$



Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 



In machine learning, we often consider problems that can be modeled as independent and identically distributed $( i . i . d . )$ random variables, $X _ { 1 } , \ldots , X _ { N }$ . For more than two random variables, the word “independent” (Definition 6.10) usually refers to mutually independent random variables, where all subsets are independent (see Pollard (2002, chapter 4) and Jacod and Protter (2004, chapter 3)). The phrase “identically distributed” means that all the random variables are from the same distribution. 

Another concept that is important in machine learning is conditional independence. 

Definition 6.11 (Conditional Independence). Two random variables X and $Y$ are conditionally independent given $Z$ if and only if 

$$
p (\boldsymbol {x}, \boldsymbol {y} \mid \boldsymbol {z}) = p (\boldsymbol {x} \mid \boldsymbol {z}) p (\boldsymbol {y} \mid \boldsymbol {z}) \quad \text { for   all } \quad \boldsymbol {z} \in \mathcal {Z}  ,\tag{6.55}
$$

where $\mathcal { Z }$ is the set of states of random variable Z. We write $X \perp \perp Y \mid Z$ to denote that X is conditionally independent of Y given $Z$ 

Definition 6.11 requires that the relation in (6.55) must hold true for every value of $_ { z . }$ . The interpretation of (6.55) can be understood as “given knowledge about $z ,$ the distribution of x and $\textbf {  { y } }$ factorizes”. Independence can be cast as a special case of conditional independence if we write X $Y \mid \emptyset .$ . By using the product rule of probability (6.22), we can expand the left-hand side of (6.55) to obtain 

$$
p (\boldsymbol {x}, \boldsymbol {y} \mid \boldsymbol {z}) = p (\boldsymbol {x} \mid \boldsymbol {y}, \boldsymbol {z}) p (\boldsymbol {y} \mid \boldsymbol {z}).\tag{6.56}
$$

By comparing the right-hand side of (6.55) with (6.56), we see that $p ( \pmb { y } \vert z )$ appears in both of them so that 

$$
p (\boldsymbol {x} \mid \boldsymbol {y}, \boldsymbol {z}) = p (\boldsymbol {x} \mid \boldsymbol {z}).\tag{6.57}
$$

Equation (6.57) provides an alternative definition of conditional independence, i.e., X $Y \mid Z$ . This alternative presentation provides the interpretation “given that we know $z ,$ , knowledge about $\textbf {  { y } }$ does not change our knowledge of $\scriptstyle { \pmb x } ^ { \prime \prime }$ 

## 6.4.6 Inner Products of Random Variables

Recall the definition of inner products from Section 3.2. We can define an inner product between random variables, which we briefly describe in this section. If we have two uncorrelated random variables $X , Y$ , then 

$$
\mathbb {V} [ x + y ] = \mathbb {V} [ x ] + \mathbb {V} [ y ].\tag{6.58}
$$

Since variances are measured in squared units, this looks very much like the Pythagorean theorem for right triangles $c ^ { 2 } = a ^ { 2 } + b ^ { 2 }$ 

In the following, we see whether we can find a geometric interpretation of the variance relation of uncorrelated random variables in (6.58). 

Figure 6.6 Geometry of random variables. If random variables X and Y are uncorrelated, they are orthogonal vectors in a corresponding vector space, and the Pythagorean theorem applies. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/074ffc1d1b8d87c4fe582b4351a6966ef2fdcbbab5eb2b71eb0eb6cb0918d64f.jpg)


Random variables can be considered vectors in a vector space, and we can define inner products to obtain geometric properties of random variables (Eaton, 2007). If we define 

$$
\langle X, Y \rangle := \operatorname{Cov} [ x, y ]\tag{6.59}
$$

for zero mean random variables X and $Y ,$ , we obtain an inner product. We see that the covariance is symmetric, positive definite, and linear in either argument. The length of a random variable is 

$$
\| X \| = \sqrt {\operatorname{Cov} [ x , x ]} = \sqrt {\mathbb {V} [ x ]} = \sigma [ x ],\tag{6.60}
$$

i.e., its standard deviation. The “longer” the random variable, the more uncertain it is; and a random variable with length 0 is deterministic. 

If we look at the angle θ between two random variables X, Y , we get 

$$
\cos \theta = \frac {\langle X , Y \rangle}{\| X \| \| Y \|} = \frac {\operatorname{Cov} [ x , y ]}{\sqrt {\mathbb {V} [ x ] \mathbb {V} [ y ]}},\tag{6.61}
$$

which is the correlation (Definition 6.8) between the two random variables. This means that we can think of correlation as the cosine of the angle between two random variables when we consider them geometrically. We know from Definition 3.7 that $X \perp Y \iff \langle X , Y \rangle = 0$ . In our case, this means that X and Y are orthogonal if and only if Cov[x, y] = 0, i.e., they are uncorrelated. Figure 6.6 illustrates this relationship. 

Remark. While it is tempting to use the Euclidean distance (constructed 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/b43118e0c84d31fa2813b4e7b763ef6ff0bb35362c5c389fead983bff93aead7.jpg)



Figure 6.7 Gaussian distribution of two random variables x<sub>1</sub> and x<sub>2</sub>.



from the preceding definition of inner products) to compare probability distributions, it is unfortunately not the best way to obtain distances between distributions. Recall that the probability mass (or density) is posi tive and needs to add up to 1. These constraints mean that distributions live on something called a statistical manifold. The study of this space of probability distributions is called information geometry. Computing distances between distributions are often done using Kullback-Leibler divergence, which is a generalization of distances that account for properties of the statistical manifold. Just like the Euclidean distance is a special case of a metric (Section 3.3), the Kullback-Leibler divergence is a special case of two more general classes of divergences called Bregman divergences and f-divergences. The study of divergences is beyond the scope of this book, and we refer for more details to the recent book by Amari (2016), one of the founders of the field of information geometry.


## 6.5 Gaussian Distribution

The Gaussian distribution is the most well-studied probability distribution for continuous-valued random variables. It is also referred to as the normal distribution. Its importance originates from the fact that it has many computationally convenient properties, which we will be discussing in the following. In particular, we will use it to define the likelihood and prior for linear regression (Chapter 9), and consider a mixture of Gaussians for density estimation (Chapter 11). 

There are many other areas of machine learning that also benefit from using a Gaussian distribution, for example Gaussian processes, variational inference, and reinforcement learning. It is also widely used in other application areas such as signal processing $( \boldsymbol { \mathrm { e . g . } }$ , Kalman filter), control $( \boldsymbol { \mathrm { e . g . , } }$ linear quadratic regulator), and statistics $( \boldsymbol { \mathrm { e . g . } }$ , hypothesis testing). 

normal distribution The Gaussian distribution arises naturally when we consider sums of independent and identically distributed random variables. This is known as the central limit theorem (Grinstead and Snell, 1997). 


Figure 6.8 Gaussian distributions overlaid with 100 samples. (a) Onedimensional case; (b) two-dimensional case.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/1aacfd8d516dbb43a9690544410857fa1818647ed28cb3098030722deabe9933.jpg)



(a) Univariate (one-dimensional) Gaussian; The red cross shows the mean and the red line shows the extent of the variance.


## Probability and Distributions

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/28cc4519303b288f90417e76d280abb7d9cd6a6d572f44d7c4b02b2884619cf0.jpg)



(b) Multivariate (two-dimensional) Gaussian, viewed from top. The red cross shows the mean and the colored lines show the contour lines of the density.


For a univariate random variable, the Gaussian distribution has a density that is given by 

$$
p (x \mid \mu , \sigma^ {2}) = \frac {1}{\sqrt {2 \pi \sigma^ {2}}} \exp \left(- \frac {(x - \mu) ^ {2}}{2 \sigma^ {2}}\right).\tag{6.62}
$$

The multivariate Gaussian distribution is fully characterized by a mean vector $\pmb { \mu }$ and a covariance matrix Σ and defined as 

$$
p (\boldsymbol {x} \mid \boldsymbol {\mu}, \boldsymbol {\Sigma}) = (2 \pi) ^ {- \frac {D}{2}} | \boldsymbol {\Sigma} | ^ {- \frac {1}{2}} \exp \left(- \frac {1}{2} (\boldsymbol {x} - \boldsymbol {\mu}) ^ {\top} \boldsymbol {\Sigma} ^ {- 1} (\boldsymbol {x} - \boldsymbol {\mu})\right),\tag{6.63}
$$

where $\pmb { x } \in \mathbb { R } ^ { D }$ . We write $p ( { \pmb x } ) = \mathcal { N } ( { \pmb x } \mid { \pmb \mu } , { \pmb \Sigma } )$ or $X \sim { \mathcal { N } } ( \mu , \Sigma )$ . Figure 6.7 shows a bivariate Gaussian (mesh), with the corresponding contour plot. Figure 6.8 shows a univariate Gaussian and a bivariate Gaussian with corresponding samples. The special case of the Gaussian with zero mean and identity covariance, that is, ${ \pmb \mu } = { \bf 0 }$ and $\Sigma = I ,$ , is referred to as the standard normal distribution. 

Gaussians are widely used in statistical estimation and machine learning as they have closed-form expressions for marginal and conditional distributions. In Chapter 9, we use these closed-form expressions extensively for linear regression. A major advantage of modeling with Gaussian random variables is that variable transformations (Section 6.7) are often not needed. Since the Gaussian distribution is fully specified by its mean and covariance, we often can obtain the transformed distribution by applying the transformation to the mean and covariance of the random variable. 

## 6.5.1 Marginals and Conditionals of Gaussians are Gaussians

In the following, we present marginalization and conditioning in the general case of multivariate random variables. If this is confusing at first reading, the reader is advised to consider two univariate random variables instead. Let X and $Y$ be two multivariate random variables, that may have different dimensions. To consider the effect of applying the sum rule of probability and the effect of conditioning, we explicitly write the Gaussian distribution in terms of the concatenated states $[ { \pmb x } ^ { \top } { \pmb y } ^ { \top } ] ^ { \top }$ so that 

$$
p (\boldsymbol {x}, \boldsymbol {y}) = \mathcal {N} \left(\left[ \begin{array}{c} \boldsymbol {\mu} _ {x} \\ \boldsymbol {\mu} _ {y} \end{array} \right], \left[ \begin{array}{c c} \boldsymbol {\Sigma} _ {x x} & \boldsymbol {\Sigma} _ {x y} \\ \boldsymbol {\Sigma} _ {y x} & \boldsymbol {\Sigma} _ {y y} \end{array} \right]\right)  ,\tag{6.64}
$$

where $\Sigma _ { x x } = \mathrm { C o v } [ { \pmb x } , { \pmb x } ]$ and $\begin{array} { r } { \Sigma _ { y y } = \mathrm { C o v } [ { \pmb y } , { \pmb y } ] } \end{array}$ are the marginal covariance matrices of x and y, respectively, and $\Sigma _ { x y } = \mathrm { C o v } [ { \pmb x } , { \pmb y } ]$ is the crosscovariance matrix between x and y. 

The conditional distribution $p ( { \pmb x } \mid { \pmb y } )$ is also Gaussian (illustrated in Figure 6.9(c)) and given by (derived in Section 2.3 of Bishop, 2006) 

$$
p (\boldsymbol {x} \mid \boldsymbol {y}) = \mathcal {N} \big (\boldsymbol {\mu} _ {x \mid y}, \boldsymbol {\Sigma} _ {x \mid y} \big)\tag{6.65}
$$

$$
\pmb {\mu} _ {x \mid y} = \pmb {\mu} _ {x} + \pmb {\Sigma} _ {x y} \pmb {\Sigma} _ {y y} ^ {- 1} (\pmb {y} - \pmb {\mu} _ {y})\tag{6.66}
$$

$$
\pmb {\Sigma} _ {x \mid y} = \pmb {\Sigma} _ {x x} - \pmb {\Sigma} _ {x y} \pmb {\Sigma} _ {y y} ^ {- 1} \pmb {\Sigma} _ {y x}.\tag{6.67}
$$

Note that in the computation of the mean in (6.66), the y-value is an observation and no longer random. 

Remark. The conditional Gaussian distribution shows up in many places, where we are interested in posterior distributions: 

The Kalman filter (Kalman, 1960), one of the most central algorithms for state estimation in signal processing, does nothing but computing Gaussian conditionals of joint distributions (Deisenroth and Ohlsson, 2011; Sarkk¨ a¨, 2013). 

Gaussian processes (Rasmussen and Williams, 2006), which are a practical implementation of a distribution over functions. In a Gaussian process, we make assumptions of joint Gaussianity of random variables. By (Gaussian) conditioning on observed data, we can determine a posterior distribution over functions. 

Latent linear Gaussian models (Roweis and Ghahramani, 1999; Murphy, 2012), which include probabilistic principal component analysis (PPCA) (Tipping and Bishop, 1999). We will look at PPCA in more detail in Section 10.7. 

The marginal distribution $p ( { \pmb x } )$ of a joint Gaussian distribution $p ( { \pmb x } , { \pmb y } )$ (see (6.64)) is itself Gaussian and computed by applying the sum rule (6.20) and given by 

$$
p (\boldsymbol {x}) = \int p (\boldsymbol {x}, \boldsymbol {y}) \mathrm{d} \boldsymbol {y} = \mathcal {N} \left(\boldsymbol {x} \mid \boldsymbol {\mu} _ {x}, \boldsymbol {\Sigma} _ {x x}\right).\tag{6.68}
$$

The corresponding result holds for $p ( \pmb { y } )$ , which is obtained by marginalizing with respect to x. Intuitively, looking at the joint distribution in (6.64), we ignore (i.e., integrate out) everything we are not interested in. This is illustrated in Figure 6.9(b). 



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



## Example 6.6

Figure 6.9 (a) Bivariate Gaussian; (b) marginal of a joint Gaussian distribution is Gaussian; (c) the conditional distribution of a Gaussian is also Gaussian. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/a65d4ab5c2eabb16528094a40320c0d10a16533ec8d67fb6414700b5ded197f0.jpg)



(a) Bivariate Gaussian.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/bcdc7b88722f3a6cef59c8c8fbeb19f36812074c0660eb54b37e061af136605a.jpg)



(b) Marginal distribution.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/289f6080efc196b2da69e63a6655a3e90789e737432ce2fb55b566fde6459dea.jpg)



(c) Conditional distribution.



Consider the bivariate Gaussian distribution (illustrated in Figure 6.9):


$$
p (x _ {1}, x _ {2}) = \mathcal {N} \left(\left[ \begin{array}{c} 0 \\ 2 \end{array} \right], \left[ \begin{array}{c c} 0. 3 & - 1 \\ - 1 & 5 \end{array} \right]\right).\tag{6.69}
$$

We can compute the parameters of the univariate Gaussian, conditioned on $x _ { 2 } = - 1$ , by applying (6.66) and (6.67) to obtain the mean and variance respectively. Numerically, this is 

$$
\mu_ {x _ {1} \mid x _ {2} = - 1} = 0 + (- 1) \cdot 0. 2 \cdot (- 1 - 2) = 0. 6\tag{6.70}
$$

and 

$$
\sigma_ {x _ {1} \mid x _ {2} = - 1} ^ {2} = 0. 3 - (- 1) \cdot 0. 2 \cdot (- 1) = 0. 1.\tag{6.71}
$$

Therefore, the conditional Gaussian is given by 

$$
p (x _ {1} \mid x _ {2} = - 1) = \mathcal {N} (0. 6, 0. 1).\tag{6.72}
$$

The marginal distribution $p ( x _ { 1 } )$ , in contrast, can be obtained by applying (6.68), which is essentially using the mean and variance of the random variable $x _ { 1 }$ , giving us 

$$
p (x _ {1}) = \mathcal {N} (0, 0. 3).\tag{6.73}
$$



Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 



## 6.5.2 Product of Gaussian Densities

For linear regression (Chapter 9), we need to compute a Gaussian likelihood. Furthermore, we may wish to assume a Gaussian prior (Section 9.3). We apply Bayes’ Theorem to compute the posterior, which results in a multiplication of the likelihood and the prior, that is, the multiplication of two Gaussian densities. The product of two Gaussians $\mathcal { N } ( \pmb { x } | \pmb { a } , \mathbf { \tilde { A } } ) \mathcal { N } ( \pmb { x } | b , B )$ is a Gaussian distribution scaled by ${ \tt a c } \in \mathbb { R }$ , given by $\displaystyle c \mathcal { N } ( \pmb { x } | \pmb { c } , \pmb { C } )$ with 

$$
\boldsymbol {C} = \left(\boldsymbol {A} ^ {- 1} + \boldsymbol {B} ^ {- 1}\right) ^ {- 1}\tag{6.74}
$$

$$
\boldsymbol {c} = \boldsymbol {C} (\boldsymbol {A} ^ {- 1} \boldsymbol {a} + \boldsymbol {B} ^ {- 1} \boldsymbol {b})\tag{6.75}
$$

$$
c = (2 \pi) ^ {- \frac {D}{2}} | \boldsymbol {A} + \boldsymbol {B} | ^ {- \frac {1}{2}} \exp \left(- \frac {1}{2} (\boldsymbol {a} - \boldsymbol {b}) ^ {\top} (\boldsymbol {A} + \boldsymbol {B}) ^ {- 1} (\boldsymbol {a} - \boldsymbol {b})\right)\tag{6.76}
$$

The scaling constant c itself can be written in the form of a Gaussian density either in a or in b with an “inflated” covariance matrix $\mathbf { A } + \mathbf { B }$ $\mathrm { i . e . , } c = { \mathcal { N } } { \big ( } a \mid b , A + B { \big ) } = { \mathcal { N } } { \big ( } b \mid a , A + B { \big ) }$ 

Remark. For notation convenience, we will sometimes use $\mathcal { N } ( \pmb { x } | \pmb { m } , S )$ to describe the functional form of a Gaussian density even if x is not a random variable. We have just done this in the preceding demonstration when we wrote 

$$
c = \mathcal {N} (\boldsymbol {a} \mid \boldsymbol {b}, \boldsymbol {A} + \boldsymbol {B}) = \mathcal {N} (\boldsymbol {b} \mid \boldsymbol {a}, \boldsymbol {A} + \boldsymbol {B}).\tag{6.77}
$$

Here, neither a nor b are random variables. However, writing c in this way is more compact than (6.76). 

## 6.5.3 Sums and Linear Transformations

If $X , Y$ are independent Gaussian random variables $( \mathrm { i . e . } ,$ , the joint distribution is given as $p ( { \pmb x } , { \pmb y } ) = p ( { \pmb x } ) p ( { \pmb y } ) )$ with $p ( { \pmb x } ) = \mathcal { N } \big ( { \pmb x } \mid { \pmb \mu } _ { x } , { \pmb \Sigma } _ { x } \big )$ and $p ( \pmb { y } ) = \mathcal { N } \big ( \pmb { y } \mid \pmb { \mu } _ { y } , \pmb { \Sigma } _ { y } \big )$ , then ${ \pmb x } + { \pmb y }$ is also Gaussian distributed and given by 

$$
p (\boldsymbol {x} + \boldsymbol {y}) = \mathcal {N} \left(\boldsymbol {\mu} _ {x} + \boldsymbol {\mu} _ {y}, \boldsymbol {\Sigma} _ {x} + \boldsymbol {\Sigma} _ {y}\right).\tag{6.78}
$$

Knowing that $p ( { \pmb x } + { \pmb y } )$ is Gaussian, the mean and covariance matrix can be determined immediately using the results from (6.46) through (6.49). This property will be important when we consider i.i.d. Gaussian noise acting on random variables, as is the case for linear regression (Chapter 9). 

## Example 6.7

Since expectations are linear operations, we can obtain the weighted sum of independent Gaussian random variables 

$$
p (a \boldsymbol {x} + b \boldsymbol {y}) = \mathcal {N} \left(a \boldsymbol {\mu} _ {x} + b \boldsymbol {\mu} _ {y}, a ^ {2} \boldsymbol {\Sigma} _ {x} + b ^ {2} \boldsymbol {\Sigma} _ {y}\right).\tag{6.79}
$$



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



Remark. A case that will be useful in Chapter 11 is the weighted sum of Gaussian densities. This is different from the weighted sum of Gaussian random variables. 

In Theorem 6.12, the random variable x is from a density that is a mixture of two densities $p _ { 1 } ( x )$ and $p _ { 2 } ( x )$ , weighted by α. The theorem can be generalized to the multivariate random variable case, since linearity of expectations holds also for multivariate random variables. However, the idea of a squared random variable needs to be replaced by $\pmb { x } \pmb { x } ^ { \top }$ 

Theorem 6.12. Consider a mixture of two univariate Gaussian densities 

$$
p (x) = \alpha p _ {1} (x) + (1 - \alpha) p _ {2} (x),\tag{6.80}
$$

where the scalar $0 < \alpha < 1$ is the mixture weight, and $p _ { 1 } ( x )$ and $p _ { 2 } ( x )$ are univariate Gaussian densities (Equation (6.62)) with different parameters, i.e., $( \mu _ { 1 } , \sigma _ { 1 } ^ { 2 } ) \neq ( \mu _ { 2 } , \sigma _ { 2 } ^ { 2 } )$ 

Then the mean of the mixture density $p ( x )$ is given by the weighted sum of the means of each random variable: 

$$
\mathbb {E} [ x ] = \alpha \mu_ {1} + (1 - \alpha) \mu_ {2}.\tag{6.81}
$$

The variance of the mixture density $p ( x )$ is given by 

$$
\mathbb {V} [ x ] = \left[ \alpha \sigma_ {1} ^ {2} + (1 - \alpha) \sigma_ {2} ^ {2} \right] + \left(\left[ \alpha \mu_ {1} ^ {2} + (1 - \alpha) \mu_ {2} ^ {2} \right] - \left[ \alpha \mu_ {1} + (1 - \alpha) \mu_ {2} \right] ^ {2}\right)\tag{6.82}
$$

Proof The mean of the mixture density $p ( x )$ is given by the weighted sum of the means of each random variable. We apply the definition of the mean (Definition 6.4), and plug in our mixture (6.80), which yields 

$$
\mathbb {E} [ x ] = \int_ {- \infty} ^ {\infty} x p (x) \mathrm{d} x\tag{6.83a}
$$

$$
= \int_ {- \infty} ^ {\infty} (\alpha x p _ {1} (x) + (1 - \alpha) x p _ {2} (x)) d x\tag{6.83b}
$$

$$
= \alpha \int_ {- \infty} ^ {\infty} x p _ {1} (x) \mathrm{d} x + (1 - \alpha) \int_ {- \infty} ^ {\infty} x p _ {2} (x) \mathrm{d} x\tag{6.83c}
$$

$$
= \alpha \mu_ {1} + (1 - \alpha) \mu_ {2}.\tag{6.83d}
$$

To compute the variance, we can use the raw-score version of the variance from (6.44), which requires an expression of the expectation of the squared random variable. Here we use the definition of an expectation of a function (the square) of a random variable (Definition 6.3), 

$$
\mathbb {E} [ x ^ {2} ] = \int_ {- \infty} ^ {\infty} x ^ {2} p (x) \mathrm{d} x\tag{6.84a}
$$

$$
= \int_ {- \infty} ^ {\infty} \left(\alpha x ^ {2} p _ {1} (x) + (1 - \alpha) x ^ {2} p _ {2} (x)\right) \mathrm{d} x\tag{6.84b}
$$

Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 

## 6.5 Gaussian Distribution

$$
= \alpha \int_ {- \infty} ^ {\infty} x ^ {2} p _ {1} (x) \mathrm{d} x + (1 - \alpha) \int_ {- \infty} ^ {\infty} x ^ {2} p _ {2} (x) \mathrm{d} x\tag{6.84c}
$$

$$
= \alpha (\mu_ {1} ^ {2} + \sigma_ {1} ^ {2}) + (1 - \alpha) (\mu_ {2} ^ {2} + \sigma_ {2} ^ {2}),\tag{6.84d}
$$

where in the last equality, we again used the raw-score version of the variance (6.44) giving $\sigma ^ { 2 } = \mathbb { E } [ x ^ { 2 } ] - \mu ^ { 2 }$ . This is rearranged such that the expectation of a squared random variable is the sum of the squared mean and the variance. 

Therefore, the variance is given by subtracting (6.83d) from (6.84d), 

$$
\mathbb {V} [ x ] = \mathbb {E} [ x ^ {2} ] - (\mathbb {E} [ x ]) ^ {2}\tag{6.85a}
$$

$$
= \alpha (\mu_ {1} ^ {2} + \sigma_ {1} ^ {2}) + (1 - \alpha) (\mu_ {2} ^ {2} + \sigma_ {2} ^ {2}) - (\alpha \mu_ {1} + (1 - \alpha) \mu_ {2}) ^ {2}\tag{6.85b}
$$

$$
= \left[ \alpha \sigma_ {1} ^ {2} + (1 - \alpha) \sigma_ {2} ^ {2} \right]
$$

$$
+ \left(\left[ \alpha \mu_ {1} ^ {2} + (1 - \alpha) \mu_ {2} ^ {2} \right] - \left[ \alpha \mu_ {1} + (1 - \alpha) \mu_ {2} \right] ^ {2}\right).\tag{6.85c}
$$

Remark. The preceding derivation holds for any density, but since the Gaussian is fully determined by the mean and variance, the mixture density can be determined in closed form. $\diamondsuit$ 

For a mixture density, the individual components can be considered to be conditional distributions (conditioned on the component identity). Equation (6.85c) is an example of the conditional variance formula, also known as the law of total variance, which generally states that for two random variables X and Y it holds that $\mathbb { V } _ { X } [ x ] = \mathbb { E } _ { Y } [ \mathbb { V } _ { X } [ x | y ] ] + \mathbb { V } _ { Y } [ \mathbb { E } _ { X } [ x | y ] ]$ 2 i.e., the (total) variance of X is the expected conditional variance plus the variance of a conditional mean. 

We consider in Example 6.17 a bivariate standard Gaussian random variable X and performed a linear transformation Ax on it. The outcome is a Gaussian random variable with mean zero and covariance $A A ^ { \top }$ . Observe that adding a constant vector will change the mean of the distribution, without affecting its variance, that is, the random variable ${ \pmb x } + { \pmb \mu }$ is Gaussian with mean µ and identity covariance. Hence, any linear/affine transformation of a Gaussian random variable is Gaussian distributed. 

Consider a Gaussian distributed random variable $X \sim { \mathcal { N } } ( \mu , \Sigma )$ . For a given matrix A of appropriate shape, let $Y$ be a random variable such that $y = A x$ is a transformed version of ${ \bf { x } } .$ . We can compute the mean of $\textbf {  { y } }$ by exploiting that the expectation is a linear operator (6.50) as follows: 

$$
\mathbb {E} [ \boldsymbol {y} ] = \mathbb {E} [ \boldsymbol {A x} ] = \boldsymbol {A} \mathbb {E} [ \boldsymbol {x} ] = \boldsymbol {A} \boldsymbol {\mu}.\tag{6.86}
$$

Similarly the variance of $\textbf {  { y } }$ can be found by using (6.51): 

$$
\mathbb {V} [ \boldsymbol {y} ] = \mathbb {V} [ \boldsymbol {A x} ] = \boldsymbol {A V} [ \boldsymbol {x} ] \boldsymbol {A} ^ {\top} = \boldsymbol {A} \boldsymbol {\Sigma} \boldsymbol {A} ^ {\top}.\tag{6.87}
$$

This means that the random variable $\textbf {  { y } }$ is distributed according to 

$$
p (\boldsymbol {y}) = \mathcal {N} \left(\boldsymbol {y} \mid \boldsymbol {A} \boldsymbol {\mu}, \boldsymbol {A} \boldsymbol {\Sigma} \boldsymbol {A} ^ {\top}\right).\tag{6.88}
$$

©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 

Let us now consider the reverse transformation: when we know that a random variable has a mean that is a linear transformation of another random variable. For a given full rank matrix $\pmb { A } \in \mathbb { R } ^ { M \times N }$ , where $M \geqslant N$ let $\pmb { y } \in \mathbb { R } ^ { M }$ be a Gaussian random variable with mean Ax, i.e., 

$$
p (\boldsymbol {y}) = \mathcal {N} (\boldsymbol {y} \mid A \boldsymbol {x}, \boldsymbol {\Sigma}).\tag{6.89}
$$

What is the corresponding probability distribution $p ( { \pmb x } ) \ b { \ ? }$ If A is invertible, then we can write $x = A ^ { - 1 } y$ and apply the transformation in the previous paragraph. However, in general A is not invertible, and we use an approach similar to that of the pseudo-inverse (3.57). That is, we premultiply both sides with $A ^ { \top }$ and then invert $A ^ { \top } A$ , which is symmetric and positive definite, giving us the relation 

$$
\boldsymbol {y} = \boldsymbol {A} \boldsymbol {x} \iff (\boldsymbol {A} ^ {\top} \boldsymbol {A}) ^ {- 1} \boldsymbol {A} ^ {\top} \boldsymbol {y} = \boldsymbol {x}.\tag{6.90}
$$

Hence, x is a linear transformation of ${ \mathbf { } } ^ { \prime } { \mathbf { } }$ and we obtain 

$$
p (\boldsymbol {x}) = \mathcal {N} \left(\boldsymbol {x} \mid \left(\boldsymbol {A} ^ {\top} \boldsymbol {A}\right) ^ {- 1} \boldsymbol {A} ^ {\top} \boldsymbol {y}, \left(\boldsymbol {A} ^ {\top} \boldsymbol {A}\right) ^ {- 1} \boldsymbol {A} ^ {\top} \boldsymbol {\Sigma} \boldsymbol {A} \left(\boldsymbol {A} ^ {\top} \boldsymbol {A}\right) ^ {- 1}\right).\tag{6.91}
$$

## 6.5.4 Sampling from Multivariate Gaussian Distributions

We will not explain the subtleties of random sampling on a computer, and the interested reader is referred to Gentle (2004). In the case of a multivariate Gaussian, this process consists of three stages: first, we need a source of pseudo-random numbers that provide a uniform sample in the interval [0,1]; second, we use a non-linear transformation such as the Box-Muller transform (¨ Devroye, 1986) to obtain a sample from a univariate Gaussian; and third, we collate a vector of these samples to obtain a sample from a multivariate standard normal $\mathcal { N } ( \mathbf { 0 } , \pmb { I } )$ 

For a general multivariate Gaussian, that is, where the mean is non zero and the covariance is not the identity matrix, we use the properties of linear transformations of a Gaussian random variable. Assume we are interested in generating samples $\pmb { x } _ { i } , i = 1 , \ldots , n$ , from a multivariate Gaussian distribution with mean $\pmb { \mu }$ and covariance matrix Σ. We would like to construct the sample from a sampler that provides samples from the multivariate standard normal $\mathcal { N } ( \mathbf { 0 } , \pmb { I } )$ 

To obtain samples from a multivariate normal $\mathcal { N } ( \pmb { \mu } , \pmb { \Sigma } )$ , we can use the properties of a linear transformation of a Gaussian random variable: If $\mathbf { \boldsymbol { x } } \sim \mathcal { N } ( \mathbf { \boldsymbol { 0 } } , I )$ , then ${ \pmb y } = { \pmb A } { \pmb x } + { \pmb \mu } _ { 3 }$ , where $\boldsymbol { A } \boldsymbol { A } ^ { \intercal } = \boldsymbol { \Sigma }$ is Gaussian distributed with mean $\pmb { \mu }$ and covariance matrix Σ. One convenient choice of $\pmb { A }$ is to use the Cholesky decomposition (Section 4.3) of the covariance matrix $\pmb { \Sigma } = \pmb { A } \pmb { A } ^ { \top }$ . The Cholesky decomposition has the benefit that A is triangular, leading to efficient computation. 

Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 

## 6.6 Conjugacy and the Exponential Family

Many of the probability distributions “with names” that we find in statistics textbooks were discovered to model particular types of phenomena. For example, we have seen the Gaussian distribution in Section 6.5. The distributions are also related to each other in complex ways (Leemis and McQueston, 2008). For a beginner in the field, it can be overwhelming to figure out which distribution to use. In addition, many of these distributions were discovered at a time that statistics and computation were done by pencil and paper. It is natural to ask what are meaningful concepts in the computing age (Efron and Hastie, 2016). In the previous section, we saw that many of the operations required for inference can be conveniently calculated when the distribution is Gaussian. It is worth recalling at this point the desiderata for manipulating probability distributions in the machine learning context: 

1. There is some “closure property” when applying the rules of probability, e.g., Bayes’ theorem. By closure, we mean that applying a particular operation returns an object of the same type. 

2. As we collect more data, we do not need more parameters to describe the distribution. 

3. Since we are interested in learning from data, we want parameter estimation to behave nicely. 

It turns out that the class of distributions called the exponential family provides the right balance of generality while retaining favorable compu tation and inference properties. Before we introduce the exponential family, let us see three more members of “named” probability distributions, the Bernoulli (Example 6.8), Binomial (Example 6.9), and Beta (Exam ple 6.10) distributions. 

## Example 6.8

The Bernoulli distribution is a distribution for a single binary random variable X with state x $\in \{ 0 , 1 \}$ . It is governed by a single continuous parameter $\mu \in [ 0 , 1 ]$ that represents the probability of $X = 1$ . The Bernoulli distribution Ber $( \mu )$ is defined as 

$$
p (x \mid \mu) = \mu^ {x} (1 - \mu) ^ {1 - x}, \quad x \in \{0, 1 \},\tag{6.92}
$$

$$
\mathbb {E} [ x ] = \mu ,\tag{6.93}
$$

$$
\mathbb {V} [ x ] = \mu (1 - \mu),\tag{6.94}
$$

where $\mathbb { E } [ x ]$ and $\mathbb { V } [ x ]$ are the mean and variance of the binary random variable X. 

An example where the Bernoulli distribution can be used is when we are interested in modeling the probability of “heads” when flipping a coin. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/85e9d2c5bc6dc5683fc11b94a403e9633643cb2995a8dba3ffcc98cf927cd195.jpg)


Figure 6.10 Examples of the Binomial distribution for µ ∈ {0.1, 0.4, 0.75} and N = 15. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/cf83d0b46da7617b01e7a6b1a494af28fe38157a0f1267bf54e7f0ebcd19fed0.jpg)



Number m of observations $x = 1$ in $N = 1 5$ experiments


Remark. The rewriting above of the Bernoulli distribution, where we use Boolean variables as numerical 0 or 1 and express them in the exponents, is a trick that is often used in machine learning textbooks. Another occurence of this is when expressing the Multinomial distribution. 

## Example 6.9 (Binomial Distribution)

The Binomial distribution is a generalization of the Bernoulli distribution to a distribution over integers (illustrated in Figure 6.10). In particular, the Binomial can be used to describe the probability of observing m occurrences of $X = 1$ in a set of N samples from a Bernoulli distribution where $p ( X = 1 ) \ = \mu \in [ 0 , 1 ]$ . The Binomial distribution Bin $( N , \mu )$ is defined as 

$$
p (m \mid N, \mu) = \binom {N} {m} \mu^ {m} (1 - \mu) ^ {N - m},\tag{6.95}
$$

$$
\mathbb {E} [ m ] = N \mu ,\tag{6.96}
$$

$$
\mathbb {V} [ m ] = N \mu (1 - \mu),\tag{6.97}
$$

where $\mathbb { E } [ m ]$ and $\mathbb { V } [ m ]$ are the mean and variance of $m _ { \ i }$ , respectively. 

An example where the Binomial could be used is if we want to describe the probability of observing m “heads” in N coin-flip experiments if the probability for observing head in a single experiment is $\mu .$ 

## Example 6.10 (Beta Distribution)

We may wish to model a continuous random variable on a finite interval. The Beta distribution is a distribution over a continuous random variable $\mu \in [ 0 , 1 ]$ , which is often used to represent the probability for some binary event (e.g., the parameter governing the Bernoulli distribution). The Beta distribution $\mathtt { B e t a } ( \alpha , \beta )$ (illustrated in Figure 6.11) itself is governed by two parameters $\alpha > 0 , \beta > 0$ and is defined as 

$$
p (\mu \mid \alpha , \beta) = \frac {\Gamma (\alpha + \beta)}{\Gamma (\alpha) \Gamma (\beta)} \mu^ {\alpha - 1} (1 - \mu) ^ {\beta - 1}\tag{6.98}
$$

$$
\mathbb {E} [ \mu ] = \frac {\alpha}{\alpha + \beta}, \qquad \mathbb {V} [ \mu ] = \frac {\alpha \beta}{(\alpha + \beta) ^ {2} (\alpha + \beta + 1)}\tag{6.99}
$$

where $\Gamma ( \cdot )$ is the Gamma function defined as 

$$
\Gamma (t) := \int_ {0} ^ {\infty} x ^ {t - 1} \exp (- x) d x, \qquad t > 0.\tag{6.100}
$$

$$
\Gamma (t + 1) = t \Gamma (t).\tag{6.101}
$$

Note that the fraction of Gamma functions in (6.98) normalizes the Beta distribution. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/cf46f3b9c91b096e42641720e92560ce1f9daa49aa7aeec1267792439eab494d.jpg)



Figure 6.11 Examples of the Beta distribution for different values of α and β.


Intuitively, α moves probability mass toward 1, whereas $\beta$ moves prob ability mass toward 0. There are some special cases (Murphy, 2012): 

For $\alpha = 1 = \beta ,$ , we obtain the uniform distribution $\boldsymbol { \mathcal { U } } [ 0 , 1 ]$ 

For $\alpha , \beta < 1$ , we get a bimodal distribution with spikes at 0 and 1. 

For $\alpha , \beta > 1$ , the distribution is unimodal. 

For $\alpha , \beta > 1$ and $\alpha = \beta _ { i }$ , the distribution is unimodal, symmetric, and centered in the interval [0, 1], i.e., the mode/mean is at $\frac { 1 } { 2 }$ . 

Remark. There is a whole zoo of distributions with names, and they are related in different ways to each other (Leemis and McQueston, 2008). It is worth keeping in mind that each named distribution is created for a particular reason, but may have other applications. Knowing the reason behind the creation of a particular distribution often allows insight into how to best use it. We introduced the preceding three distributions to be able to illustrate the concepts of conjugacy (Section 6.6.1) and exponential families (Section 6.6.3). ◇ 

## 6.6.1 Conjugacy

According to Bayes’ theorem (6.23), the posterior is proportional to the product of the prior and the likelihood. The specification of the prior can be tricky for two reasons: First, the prior should encapsulate our knowledge about the problem before we see any data. This is often difficult to describe. Second, it is often not possible to compute the posterior distribution analytically. However, there are some priors that are computationally convenient: conjugate priors. 

Definition 6.13 (Conjugate Prior). A prior is conjugate for the likelihood function if the posterior is of the same form/type as the prior. 

Conjugacy is particularly convenient because we can algebraically calculate our posterior distribution by updating the parameters of the prior distribution. 

Remark. When considering the geometry of probability distributions, conjugate priors retain the same distance structure as the likelihood (Agarwal and Daum´e III, 2010). 

To introduce a concrete example of conjugate priors, we describe in Example 6.11 the Binomial distribution (defined on discrete random variables) and the Beta distribution (defined on continuous random variables). 

## Example 6.11 (Beta-Binomial Conjugacy)

Consider a Binomial random variable $x \sim \mathrm { B i n } ( N , \mu )$ where 

$$
p (x \mid N, \mu) = \binom {N} {x} \mu^ {x} (1 - \mu) ^ {N - x}, \quad x = 0, 1, \ldots , N,\tag{6.102}
$$

is the probability of finding x times the outcome “heads” in $N$ coin flips, where $\mu$ is the probability of a “head”. We place a Beta prior on the parameter $\mu ,$ that is, $\mu \sim$ Beta $( \alpha , \beta )$ , where 

$$
p (\mu \mid \alpha , \beta) = \frac {\Gamma (\alpha + \beta)}{\Gamma (\alpha) \Gamma (\beta)} \mu^ {\alpha - 1} (1 - \mu) ^ {\beta - 1}.\tag{6.103}
$$

If we now observe some outcome $x = h ,$ , that is, we see h heads in $N$ coin flips, we compute the posterior distribution on $\mu$ as 

$$
p (\mu \mid x = h, N, \alpha , \beta) \propto p (x \mid N, \mu) p (\mu \mid \alpha , \beta)\tag{6.104a}
$$

$$
\propto \mu^ {h} (1 - \mu) ^ {(N - h)} \mu^ {\alpha - 1} (1 - \mu) ^ {\beta - 1}\tag{6.104b}
$$

$$
= \mu^ {h + \alpha - 1} (1 - \mu) ^ {(N - h) + \beta - 1}\tag{6.104c}
$$


6.6 Conjugacy and the Exponential Family


<table><tr><td>Likelihood</td><td>Conjugate prior</td><td>Posterior</td></tr><tr><td>Bernoulli</td><td>Beta</td><td>Beta</td></tr><tr><td>Binomial</td><td>Beta</td><td>Beta</td></tr><tr><td>Gaussian</td><td>Gaussian/inverse Gamma</td><td>Gaussian/inverse Gamma</td></tr><tr><td>Gaussian</td><td>Gaussian/inverse Wishart</td><td>Gaussian/inverse Wishart</td></tr><tr><td>Multinomial</td><td>Dirichlet</td><td>Dirichlet</td></tr></table>


Table 6.2 Examples of conjugate priors for common likelihood functions. 


$$
\propto \operatorname{Beta} (h + \alpha , N - h + \beta),\tag{6.104d}
$$

i.e., the posterior distribution is a Beta distribution as the prior, i.e., the Beta prior is conjugate for the parameter $\mu$ in the Binomial likelihood function. 

In the following example, we will derive a result that is similar to the Beta-Binomial conjugacy result. Here we will show that the Beta distribution is a conjugate prior for the Bernoulli distribution. 

## Example 6.12 (Beta-Bernoulli Conjugacy)

Let $x \in \{ 0 , 1 \}$ be distributed according to the Bernoulli distribution with parameter $\theta \in [ 0 , 1 ]$ , that is, $p ( x = 1 | \theta ) = \theta$ . This can also be expressed as $p ( x | \theta ) = \theta ^ { x } \bar { ( 1 - \theta ) } ^ { 1 - x }$ . Let θ be distributed according to a Beta distribution with parameters $\alpha , \beta _ { z }$ , that is, $p ( \theta | \alpha , \beta ) \propto \theta ^ { \alpha - 1 } ( 1 - \theta ) ^ { \beta - 1 }$ 

Multiplying the Beta and the Bernoulli distributions, we get 

$$
p (\theta \mid x, \alpha , \beta) \propto p (x \mid \theta) p (\theta \mid \alpha , \beta)\tag{6.105a}
$$

$$
= \theta^ {x} (1 - \theta) ^ {1 - x} \theta^ {\alpha - 1} (1 - \theta) ^ {\beta - 1}\tag{6.105b}
$$

$$
= \theta^ {\alpha + x - 1} (1 - \theta) ^ {\beta + (1 - x) - 1}\tag{6.105c}
$$

$$
\propto p (\theta \mid \alpha + x, \beta + (1 - x)).\tag{6.105d}
$$

The last line is the Beta distribution with parameters $( \alpha + x , \beta + ( 1 - x ) )$ 

Table 6.2 lists examples for conjugate priors for the parameters of some standard likelihoods used in probabilistic modeling. Distributions such as Multinomial, inverse Gamma, inverse Wishart, and Dirichlet can be found in any statistical text, and are described in Bishop (2006), for example. 

The Beta distribution is the conjugate prior for the parameter $\mu$ in both the Binomial and the Bernoulli likelihood. For a Gaussian likelihood func tion, we can place a conjugate Gaussian prior on the mean. The reason why the Gaussian likelihood appears twice in the table is that we need to distinguish the univariate from the multivariate case. In the univariate (scalar) case, the inverse Gamma is the conjugate prior for the variance. In the multivariate case, we use a conjugate inverse Wishart distribution as a prior on the covariance matrix. The Dirichlet distribution is the conjugate prior for the multinomial likelihood function. For further details, we refer to Bishop (2006). 

## 6.6.2 Sufficient Statistics

Recall that a statistic of a random variable is a deterministic function of that random variable. For example, if $\pmb { x } = [ x _ { 1 } , \ldots , x _ { N } ] ^ { \top }$ is a vector of univariate Gaussian random variables, that is, $x _ { n } \sim \mathcal { N } ( \mu , \sigma ^ { 2 } )$ , then the sample mean $\begin{array} { r } { \hat { \mu } = \frac 1 N ( x _ { 1 } + \dots + x _ { N } ) } \end{array}$ is a statistic. Sir Ronald Fisher discovered the notion of sufficient statistics: the idea that there are statistics that will contain all available information that can be inferred from data corresponding to the distribution under consideration. In other words, sufficient statistics carry all the information needed to make inference about the population, that is, they are the statistics that are sufficient to represent the distribution. 

For a set of distributions parametrized by $\theta ,$ , let X be a random variable with distribution $p ( x \mid \theta _ { 0 } )$ given an unknown $\theta _ { 0 }$ . A vector $\phi ( x )$ of statistics is called sufficient statistics for $\theta _ { 0 }$ if they contain all possible information about $\theta _ { 0 }$ . To be more formal about “contain all possible information”, this means that the probability of x given $\theta$ can be factored into a part that does not depend on $\theta ,$ and a part that depends on $\theta$ only via $\phi ( x )$ The Fisher-Neyman factorization theorem formalizes this notion, which we state in Theorem 6.14 without proof. 

Theorem 6.14 (Fisher-Neyman). [Theorem 6.5 in Lehmann and Casella (1998)] Let X have probability density function $p ( x | \theta )$ . Then the statistics ϕ(x) are sufficient for θ if and only $i f p ( x | \theta )$ can be written in the form 

$$
p (x \mid \theta) = h (x) g _ {\theta} (\phi (x)),\tag{6.106}
$$

where $h ( x )$ is a distribution independent of θ and $g _ { \theta }$ captures all the dependence on θ via sufficient statistics $\phi ( x )$ . 

If $p ( x \mid \theta )$ does not depend on $\theta ,$ then $\phi ( x )$ is trivially a sufficient statistic for any function $\phi .$ . The more interesting case is that $p ( x \mid \theta )$ is dependent only on $\phi ( x )$ and not $x$ itself. In this case, $\phi ( x )$ is a sufficient statistic for $\theta .$ 

In machine learning, we consider a finite number of samples from a distribution. One could imagine that for simple distributions (such as the Bernoulli in Example 6.8) we only need a small number of samples to estimate the parameters of the distributions. We could also consider the opposite problem: If we have a set of data (a sample from an unknown distribution), which distribution gives the best fit? A natural question to ask is, as we observe more data, do we need more parameters $\theta$ to describe the distribution? It turns out that the answer is yes in general, and this is studied in non-parametric statistics (Wasserman, 2007). A converse question is to consider which class of distributions have finite-dimensional sufficient statistics, that is the number of parameters needed to describe them does not increase arbitrarily. The answer is exponential family distributions, described in the following section. 

## 6.6.3 Exponential Family

There are three possible levels of abstraction we can have when considering distributions (of discrete or continuous random variables). At level one (the most concrete end of the spectrum), we have a particular named distribution with fixed parameters, for example a univariate Gaussian $\mathcal { N } ( 0 , 1 )$ with zero mean and unit variance. In machine learning, we often use the second level of abstraction, that is, we fix the parametric form (the univariate Gaussian) and infer the parameters from data. For example, we assume a univariate Gaussian $\mathcal { N } ( \mu , \sigma ^ { 2 } )$ with unknown mean $\mu$ and unknown variance $\sigma ^ { 2 }$ , and use a maximum likelihood fit to determine the best parameters $( \mu , \sigma ^ { 2 } )$ . We will see an example of this when considering linear regression in Chapter 9. A third level of abstraction is to consider families of distributions, and in this book, we consider the exponential family. The univariate Gaussian is an example of a member of the exponential family. Many of the widely used statistical models, including all the “named” models in Table 6.2, are members of the exponential family. They can all be unified into one concept (Brown, 1986). 

Remark. A brief historical anecdote: Like many concepts in mathematics and science, exponential families were independently discovered at the same time by different researchers. In the years 1935–1936, Edwin Pitman in Tasmania, Georges Darmois in Paris, and Bernard Koopman in New York independently showed that the exponential families are the only families that enjoy finite-dimensional sufficient statistics under repeated independent sampling (Lehmann and Casella, 1998). 

An exponential family is a family of probability distributions, parameterized by $\pmb \theta \in \mathbb { R } ^ { D }$ , of the form 

$$
p (\boldsymbol {x} \mid \boldsymbol {\theta}) = h (\boldsymbol {x}) \exp \left(\langle \boldsymbol {\theta}, \phi (\boldsymbol {x}) \rangle - A (\boldsymbol {\theta})\right),\tag{6.107}
$$

where $\phi ( { \pmb x } )$ is the vector of sufficient statistics. In general, any inner product (Section 3.2) can be used in (6.107), and for concreteness we will use the standard dot product here $( \langle \pmb \theta , \phi ( \pmb x ) \rangle = \pmb \theta ^ { \top } \phi ( \pmb x ) )$ . Note that the form of the exponential family is essentially a particular expression of $g _ { \theta } ( \phi ( x ) )$ 1 in the Fisher-Neyman theorem (Theorem 6.14). 

The factor $h ( { \pmb x } )$ can be absorbed into the dot product term by adding another entry $( \log h ( { \pmb x } ) )$ to the vector of sufficient statistics $\phi ( { \pmb x } )$ , and constraining the corresponding parameter $\theta _ { 0 } = 1$ . The term $A ( \pmb \theta )$ is the normalization constant that ensures that the distribution sums up or integrates to one and is called the log-partition function. A good intuitive notion of exponential families can be obtained by ignoring these two terms and considering exponential families as distributions of the form 

$$
p (\boldsymbol {x} \mid \boldsymbol {\theta}) \propto \exp \left(\boldsymbol {\theta} ^ {\top} \phi (\boldsymbol {x})\right).\tag{6.108}
$$

natural parameters 

For this form of parametrization, the parameters $\pmb \theta$ are called the natural parameters. At first glance, it seems that exponential families are a mundane transformation by adding the exponential function to the result of a dot product. However, there are many implications that allow for convenient modeling and efficient computation based on the fact that we can capture information about data in $\phi ( { \pmb x } )$ 

## Example 6.13 (Gaussian as Exponential Family)

Consider the univariate Gaussian distribution ${ \mathcal { N } } ( \mu , \sigma ^ { 2 } )$ . Let $\phi ( x ) = \biggl [ \frac { x } { x ^ { 2 } } \biggr ]$ Then by using the definition of the exponential family, 

$$
p (x \mid \boldsymbol {\theta}) \propto \exp (\theta_ {1} x + \theta_ {2} x ^ {2}).\tag{6.109}
$$

Setting 

$$
\pmb {\theta} = \left[ \frac {\mu}{\sigma^ {2}}, - \frac {1}{2 \sigma^ {2}} \right] ^ {\top}\tag{6.110}
$$

and substituting into (6.109), we obtain 

$$
p (x \mid \boldsymbol {\theta}) \propto \exp \left(\frac {\mu x}{\sigma^ {2}} - \frac {x ^ {2}}{2 \sigma^ {2}}\right) \propto \exp \left(- \frac {1}{2 \sigma^ {2}} (x - \mu) ^ {2}\right).\tag{6.111}
$$

Therefore, the univariate Gaussian distribution is a member of the exponential family with sufficient statistic $\phi ( x ) = { \Big [ } { x } { \Big ] }$ , and natural parameters given by $\pmb \theta$ in (6.110). 

## Example 6.14 (Bernoulli as Exponential Family)

Recall the Bernoulli distribution from Example 6.8 

$$
p (x \mid \mu) = \mu^ {x} (1 - \mu) ^ {1 - x}, \quad x \in \{0, 1 \}.\tag{6.112}
$$

This can be written in exponential family form 

$$
p (x \mid \mu) = \exp \left[ \log \left(\mu^ {x} (1 - \mu) ^ {1 - x}\right) \right]\tag{6.113a}
$$

$$
= \exp \left[ x \log \mu + (1 - x) \log (1 - \mu) \right]\tag{6.113b}
$$

$$
= \exp \left[ x \log \mu - x \log (1 - \mu) + \log (1 - \mu) \right]\tag{6.113c}
$$

$$
= \exp \left[ x \log {\frac {\mu}{1 - \mu}} + \log (1 - \mu) \right].\tag{6.113d}
$$

The last line (6.113d) can be identified as being in exponential family form (6.107) by observing that 

$$
h (x) = 1\tag{6.114}
$$



Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 



$$
\theta = \log {\frac {\mu}{1 - \mu}}\tag{6.115}
$$

$$
\phi (x) = x\tag{6.116}
$$

$$
A (\theta) = - \log (1 - \mu) = \log (1 + \exp (\theta)).\tag{6.117}
$$

The relationship between θ and $\mu$ is invertible so that 

$$
\mu = \frac {1}{1 + \exp (- \theta)}.\tag{6.118}
$$

The relation (6.118) is used to obtain the right equality of (6.117). 

Remark. The relationship between the original Bernoulli parameter $\mu$ and the natural parameter $\theta$ is known as the sigmoid or logistic function. Observe that $\mu \in \mathsf { \Gamma } ( 0 , 1 )$ but $\theta ~ \in ~ \mathbb { R }$ , and therefore the sigmoid function squeezes a real value into the range (0, 1). This property is useful in machine learning, for example it is used in logistic regression (Bishop, 2006, section 4.3.2), as well as as a nonlinear activation functions in neural net works (Goodfellow et al., 2016, chapter 6). 

It is often not obvious how to find the parametric form of the conjugate distribution of a particular distribution (for example, those in Table 6.2). Exponential families provide a convenient way to find conjugate pairs of distributions. Consider the random variable $\dot { X }$ is a member of the exponential family (6.107): 

$$
p (\boldsymbol {x} \mid \boldsymbol {\theta}) = h (\boldsymbol {x}) \exp \left(\langle \boldsymbol {\theta}, \phi (\boldsymbol {x}) \rangle - A (\boldsymbol {\theta})\right).\tag{6.119}
$$

Every member of the exponential family has a conjugate prior (Brown, 1986) 

$$
p (\boldsymbol {\theta} \mid \boldsymbol {\gamma}) = h _ {c} (\boldsymbol {\theta}) \exp \left(\left\langle \left[ \begin{array}{c} \boldsymbol {\gamma} _ {1} \\ \boldsymbol {\gamma} _ {2} \end{array} \right], \left[ \begin{array}{c} \boldsymbol {\theta} \\ - A (\boldsymbol {\theta}) \end{array} \right] \right\rangle - A _ {c} (\boldsymbol {\gamma})\right)  ,\tag{6.120}
$$

where $\gamma = \binom { \gamma _ { 1 } } { \gamma _ { 2 } }$ has dimension dim $( \pmb \theta ) + 1$ . The sufficient statistics of the conjugate prior are $\left[ - \stackrel { \theta } { A ( \theta ) } \right]$ . By using the knowledge of the general form of conjugate priors for exponential families, we can derive functional forms of conjugate priors corresponding to particular distributions. 

## Example 6.15

Recall the exponential family form of the Bernoulli distribution (6.113d) 

$$
p (x \mid \mu) = \exp \left[ x \log {\frac {\mu}{1 - \mu}} + \log (1 - \mu) \right].\tag{6.121}
$$



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



The canonical conjugate prior has the form 

$$
p (\mu \mid \alpha , \beta) = \frac {\mu}{1 - \mu} \exp \left[ \alpha \log \frac {\mu}{1 - \mu} + (\beta + \alpha) \log (1 - \mu) - A _ {c} (\boldsymbol {\gamma}) \right],\tag{6.122}
$$

where we defined $\gamma : = [ \alpha , \beta + \alpha ] ^ { \intercal }$ and $h _ { c } ( \mu ) : = \mu / ( 1 - \mu )$ . Equation (6.122) then simplifies to 

$$
p (\mu \mid \alpha , \beta) = \exp \left[ (\alpha - 1) \log \mu + (\beta - 1) \log (1 - \mu) - A _ {c} (\alpha , \beta) \right].\tag{6.123}
$$

Putting this in non-exponential family form yields 

$$
p (\mu \mid \alpha , \beta) \propto \mu^ {\alpha - 1} (1 - \mu) ^ {\beta - 1},\tag{6.124}
$$

which we identify as the Beta distribution (6.98). In example 6.12, we assumed that the Beta distribution is the conjugate prior of the Bernoulli distribution and showed that it was indeed the conjugate prior. In this example, we derived the form of the Beta distribution by looking at the canonical conjugate prior of the Bernoulli distribution in exponential family form. 

As mentioned in the previous section, the main motivation for exponential families is that they have finite-dimensional sufficient statistics. Additionally, conjugate distributions are easy to write down, and the conjugate distributions also come from an exponential family. From an inference perspective, maximum likelihood estimation behaves nicely because empirical estimates of sufficient statistics are optimal estimates of the population values of sufficient statistics (recall the mean and covariance of a Gaussian). From an optimization perspective, the log-likelihood function is concave, allowing for efficient optimization approaches to be applied (Chapter 7). 

## 6.7 Change of Variables/Inverse Transform

It may seem that there are very many known distributions, but in reality the set of distributions for which we have names is quite limited. Therefore, it is often useful to understand how transformed random variables are distributed. For example, assuming that X is a random variable distributed according to the univariate normal distribution $\mathcal { N } ( 0 , 1 )$ , what is the distribution of $X ^ { 2 } ?$ Another example, which is quite common in machine learning, is, given that $X _ { 1 }$ and $X _ { 2 }$ are univariate standard normal, what is the distribution of ${ \scriptstyle { \frac { 1 } { 2 } } } ( X _ { 1 } + X _ { 2 } ) ?$ 

One option to work out the distribution of ${ \scriptstyle { \frac { 1 } { 2 } } } ( X _ { 1 } + X _ { 2 } )$ is to calculate the mean and variance of $X _ { 1 }$ and $X _ { 2 }$ and then combine them. As we saw in Section $6 . 4 . 4 .$ , we can calculate the mean and variance of resulting random variables when we consider affine transformations of random variables. However, we may not be able to obtain the functional form of the distribution under transformations. Furthermore, we may be interested in nonlinear transformations of random variables for which closed-form expressions are not readily available. 

Remark (Notation). In this section, we will be explicit about random variables and the values they take. Hence, recall that we use capital letters $X , Y$ to denote random variables and small letters $x , y$ to denote the values in the target space $\tau$ that the random variables take. We will explicitly write pmfs of discrete random variables $X$ as $P ( X = x )$ . For continuous random variables X (Section 6.2.2), the pdf is written as $f ( x )$ and the cdf is written as $F _ { X } ( x )$ 

We will look at two approaches for obtaining distributions of transformations of random variables: a direct approach using the definition of a cumulative distribution function and a change-of-variable approach that uses the chain rule of calculus (Section 5.2.2). The change-of-variable approach is widely used because it provides a “recipe” for attempting to compute the resulting distribution due to a transformation. We will explain the techniques for univariate random variables, and will only briefly provide the results for the general case of multivariate random variables. 

Transformations of discrete random variables can be understood directly. Suppose that there is a discrete random variable X with pmf $P ( X =$ $x )$ (Section 6.2.1), and an invertible function $U ( x )$ . Consider the transformed random variable $Y : = U ( X )$ , with pmf $P ( \boldsymbol { Y } = \boldsymbol { y } )$ . Then 

$$
\begin{array}{c c} P (Y = y) = P (U (X) = y) & \text { transformation   of   interest } \\ = P (X = U ^ {- 1} (y)) & \text { inverse } \end{array}\tag{6.125a}
$$

(6.125b) 

where we can observe that $x = U ^ { - 1 } ( y )$ . Therefore, for discrete random variables, transformations directly change the individual events (with the probabilities appropriately transformed). 

## 6.7.1 Distribution Function Technique

The distribution function technique goes back to first principles, and uses the definition of a cdf $F _ { X } ( x ) = P ( X \leqslant x )$ and the fact that its differential is the pdf $f ( x )$ (Wasserman, 2004, chapter 2). For a random variable $X$ and a function $U$ , we find the pdf of the random variable $Y : = U ( X )$ by 

1. Finding the cdf: 

$$
F _ {Y} (y) = P (Y \leqslant y)\tag{6.126}
$$

2. Differentiating the cdf $F _ { Y } ( y )$ to get the pdf $f ( y )$ 

$$
f (y) = \frac {\mathrm{d}}{\mathrm{d} y} F _ {Y} (y).\tag{6.127}
$$

©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 

We also need to keep in mind that the domain of the random variable may have changed due to the transformation by $U$ 

## Example 6.16

Let $X$ be a continuous random variable with probability density function on $0 \leqslant x \leqslant 1$ 

$$
f (x) = 3 x ^ {2}.\tag{6.128}
$$

We are interested in finding the pdf of $Y = X ^ { 2 }$ 

The function $f$ is an increasing function of $x ,$ and therefore the resulting value of $y$ lies in the interval [0, 1]. We obtain 

$$
F _ {Y} (y) = P (Y \leqslant y)
$$

definition of cdf 

$$
= P (X ^ {2} \leqslant y)\tag{6.129a}
$$

transformation of interest 

$$
= P (X \leqslant y ^ {\frac {1}{2}})\tag{6.129b}
$$

inverse 

$$
= F _ {X} (y ^ {\frac {1}{2}})\tag{6.129c}
$$

definition of cdf 

$$
= \int_ {0} ^ {y ^ {\frac {1}{2}}} 3 t ^ {2} \mathrm{d} t\tag{6.129d}
$$

cdf as a definite integral 

$$
= \left[ t ^ {3} \right] _ {t = 0} ^ {t = y ^ {\frac {1}{2}}}\tag{6.129e}
$$

result of integration 

(6.129f) 

$$
= y ^ {\frac {3}{2}}, \quad 0 \leqslant y \leqslant 1.\tag{6.129g}
$$

Therefore, the cdf of $Y$ is 

$$
F _ {Y} (y) = y ^ {\frac {3}{2}}\tag{6.130}
$$

for $0 \leqslant y \leqslant 1$ . To obtain the pdf, we differentiate the cdf 

$$
f (y) = \frac {\mathrm{d}}{\mathrm{d} y} F _ {Y} (y) = \frac {3}{2} y ^ {\frac {1}{2}}\tag{6.131}
$$

for $0 \leqslant y \leqslant 1$ 

Functions that have inverses are called bijective functions (Section 2.7). 

In Example 6.16, we considered a strictly monotonically increasing function $f ( x ) = 3 x ^ { 2 }$ . This means that we could compute an inverse function. In general, we require that the function of interest $y = U ( x )$ has an inverse $x = U ^ { - 1 } ( y )$ . A useful result can be obtained by considering the cumulative distribution function $F _ { X } ( x )$ of a random variable X, and using it as the transformation $U ( x )$ . This leads to the following theorem. 

Theorem 6.15. [Theorem 2.1.10 in Casella and Berger (2002)] Let X be a continuous random variable with a strictly monotonic cumulative distribution function $F _ { X } ( x )$ . Then the random variable Y defined as 

$$
Y := F _ {X} (X)\tag{6.132}
$$

has a uniform distribution. 

Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 

Theorem 6.15 is known as the probability integral transform, and it is used to derive algorithms for sampling from distributions by transforming the result of sampling from a uniform random variable (Bishop, 2006). The algorithm works by first generating a sample from a uniform distribu tion, then transforming it by the inverse cdf (assuming this is available) to obtain a sample from the desired distribution. The probability integral transform is also used for hypothesis testing whether a sample comes from a particular distribution (Lehmann and Romano, 2005). The idea that the output of a cdf gives a uniform distribution also forms the basis of copu las (Nelsen, 2006). 

## 6.7.2 Change of Variables

The distribution function technique in Section 6.7.1 is derived from first principles, based on the definitions of cdfs and using properties of inverses, differentiation, and integration. This argument from first principles relies on two facts: 

1. We can transform the cdf of Y into an expression that is a cdf of X. 2. We can differentiate the cdf to obtain the pdf. 

Let us break down the reasoning step by step, with the goal of understanding the more general change-of-variables approach in Theorem 6.16. 

Remark. The name “change of variables” comes from the idea of changing the variable of integration when faced with a difficult integral. For univariate functions, we use the substitution rule of integration, 

$$
\int f (g (x)) g ^ {\prime} (x) \mathrm{d} x = \int f (u) \mathrm{d} u, \quad \text { where } \quad u = g (x).\tag{6.133}
$$

The derivation of this rule is based on the chain rule of calculus (5.32) and by applying twice the fundamental theorem of calculus. The fundamental theorem of calculus formalizes the fact that integration and differentiation are somehow “inverses” of each other. An intuitive understanding of the rule can be obtained by thinking (loosely) about small changes (differentials) to the equation $u = g ( x )$ , that is by considering $\Delta u = g ^ { \prime } ( x ) \Delta x$ as a differential of $u = g ( x )$ . By substituting $u = g ( x )$ , the argument inside the integral on the right-hand side of (6.133) becomes $f ( g ( x ) )$ ). By pretending that the term du can be approximated by du $\approx \Delta u = g ^ { \prime } ( x ) \Delta x$ , and that dx $\Delta x$ , we obtain (6.133). 

Consider a univariate random variable X, and an invertible function $U _ { ☉ }$ , which gives us another random variable $Y = U ( X )$ . We assume that random variable X has states $x \in [ a , b ]$ . By the definition of the cdf, we have 

$$
F _ {Y} (y) = P (Y \leqslant y).\tag{6.134}
$$



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



We are interested in a function $U$ of the random variable 

$$
P (Y \leqslant y) = P (U (X) \leqslant y),\tag{6.135}
$$

where we assume that the function $U$ is invertible. An invertible function on an interval is either strictly increasing or strictly decreasing. In the case that $U$ is strictly increasing, then its inverse $U ^ { - 1 }$ is also strictly increasing. By applying the inverse $U ^ { - 1 }$ to the arguments of $P ( U ( X ) \leqslant y )$ , we obtain 

$$
P (U (X) \leqslant y) = P (U ^ {- 1} (U (X)) \leqslant U ^ {- 1} (y)) = P (X \leqslant U ^ {- 1} (y)).\tag{6.136}
$$

The right-most term in (6.136) is an expression of the cdf of X. Recall the definition of the cdf in terms of the pdf 

$$
P (X \leqslant U ^ {- 1} (y)) = \int_ {a} ^ {U ^ {- 1} (y)} f (x) \mathrm{d} x.\tag{6.137}
$$

Now we have an expression of the cdf of $Y$ in terms of x: 

$$
F _ {Y} (y) = \int_ {a} ^ {U ^ {- 1} (y)} f (x) \mathrm{d} x.\tag{6.138}
$$

To obtain the pdf, we differentiate (6.138) with respect to $y \colon$ 

$$
f (y) = \frac {\mathrm{d}}{\mathrm{d} y} F _ {y} (y) = \frac {\mathrm{d}}{\mathrm{d} y} \int_ {a} ^ {U ^ {- 1} (y)} f (x) \mathrm{d} x.\tag{6.139}
$$

Note that the integral on the right-hand side is with respect to $x ,$ , but we need an integral with respect to y because we are differentiating with respect to $y .$ . In particular, we use (6.133) to get the substitution 

$$
\int f (U ^ {- 1} (y)) U ^ {- 1 ^ {\prime}} (y) \mathrm{d} y = \int f (x) \mathrm{d} x \quad \text { where } \quad x = U ^ {- 1} (y)  .\tag{6.140}
$$

Using (6.140) on the right-hand side of (6.139) gives us 

$$
f (y) = \frac {\mathrm{d}}{\mathrm{d} y} \int_ {a} ^ {U ^ {- 1} (y)} f _ {x} (U ^ {- 1} (y)) U ^ {- 1 ^ {\prime}} (y) \mathrm{d} y.\tag{6.141}
$$

We then recall that differentiation is a linear operator and we use the subscript x to remind ourselves that $f _ { x } ( U ^ { - 1 } ( y ) )$ is a function of x and not $y .$ Invoking the fundamental theorem of calculus again gives us 

$$
f (y) = f _ {x} \left(U ^ {- 1} (y)\right) \cdot \left(\frac {\mathrm{d}}{\mathrm{d} y} U ^ {- 1} (y)\right).\tag{6.142}
$$

Recall that we assumed that $U$ is a strictly increasing function. For decreasing functions, it turns out that we have a negative sign when we follow the same derivation. We introduce the absolute value of the differential to have the same expression for both increasing and decreasing $U$ : 

$$
f (y) = f _ {x} \left(U ^ {- 1} (y)\right) \cdot \left| \frac {\mathrm{d}}{\mathrm{d} y} U ^ {- 1} (y) \right|.\tag{6.143}
$$

This is called the change-of-variable technique. The term $\begin{array} { r } { \left| { \frac { \operatorname { d } } { \operatorname { d } y } } U ^ { - 1 } ( y ) \right| } \end{array}$ in (6.143) measures how much a unit volume changes when applying $U$ (see also the definition of the Jacobian in Section 5.3). 

Remark. In comparison to the discrete case in (6.125b), we have an additional factor $\begin{array} { r } { \left| { \frac { \operatorname { d } } { \operatorname { d } y } } U ^ { - 1 } ( y ) \right| } \end{array}$ . The continuous case requires more care because $P ( Y = y ) \stackrel { . } { = } 0$ for all $y .$ . The probability density function $f ( y )$ does not have a description as a probability of an event involving $y .$ . ◇ 

So far in this section, we have been studying univariate change of variables. The case for multivariate random variables is analogous, but complicated by fact that the absolute value cannot be used for multivariate functions. Instead, we use the determinant of the Jacobian matrix. Recall from (5.58) that the Jacobian is a matrix of partial derivatives, and that the existence of a nonzero determinant shows that we can invert the Jacobian. Recall the discussion in Section 4.1 that the determinant arises because our differentials (cubes of volume) are transformed into paral lelepipeds by the Jacobian. Let us summarize preceding the discussion in the following theorem, which gives us a recipe for multivariate change of variables. 

Theorem 6.16. [Theorem 17.2 in Billingsley (1995)] Let $f ( { \pmb x } )$ be the value of the probability density of the multivariate continuous random variable $X$ If the vector-valued function $\mathbf {  { y } } = U ( \mathbf {  { x } } )$ is differentiable and invertible for all values within the domain of $\mathbf { \delta } _ { \mathbf { x } , \mathbf { \delta } }$ then for corresponding values of $^ { y , }$ the probability density of $Y = U ( X )$ is given by 

$$
f (\boldsymbol {y}) = f _ {\boldsymbol {x}} \left(U ^ {- 1} (\boldsymbol {y})\right) \cdot \left| \det \left(\frac {\partial}{\partial \boldsymbol {y}} U ^ {- 1} (\boldsymbol {y})\right) \right|.\tag{6.144}
$$

The theorem looks intimidating at first glance, but the key point is that a change of variable of a multivariate random variable follows the procedure of the univariate change of variable. First we need to work out the inverse transform, and substitute that into the density of x. Then we calculate the determinant of the Jacobian and multiply the result. The following example illustrates the case of a bivariate random variable. 

## Example 6.17

Consider a bivariate random variable X with states $\pmb { x } = \left[ \begin{array} { l } { x _ { 1 } } \\ { x _ { 2 } } \end{array} \right]$ and probability density function 

$$
f \left(\left[ \begin{array}{c} x _ {1} \\ x _ {2} \end{array} \right]\right) = \frac {1}{2 \pi} \exp \left(- \frac {1}{2} \left[ \begin{array}{c} x _ {1} \\ x _ {2} \end{array} \right] ^ {\top} \left[ \begin{array}{c} x _ {1} \\ x _ {2} \end{array} \right]\right).\tag{6.145}
$$

We use the change-of-variable technique from Theorem 6.16 to derive the 



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



effect of a linear transformation (Section 2.7) of the random variable. Consider a matrix $A \in \mathbb { R } ^ { 2 \times 2 }$ defined as 

$$
\boldsymbol {A} = \left[ \begin{array}{c c} a & b \\ c & d \end{array} \right].\tag{6.146}
$$

We are interested in finding the probability density function of the transformed bivariate random variable $Y$ with states $y = A x$ 

Recall that for change of variables we require the inverse transformation of x as a function of $\mathbf { \pmb { y } } .$ . Since we consider linear transformations, the inverse transformation is given by the matrix inverse (see Section 2.2.2). For $2 \times 2$ matrices, we can explicitly write out the formula, given by 

$$
\left[ \begin{array}{c} x _ {1} \\ x _ {2} \end{array} \right] = \boldsymbol {A} ^ {- 1} \left[ \begin{array}{c} y _ {1} \\ y _ {2} \end{array} \right] = \frac {1}{a d - b c} \left[ \begin{array}{c c} d & - b \\ - c & a \end{array} \right] \left[ \begin{array}{c} y _ {1} \\ y _ {2} \end{array} \right].\tag{6.147}
$$

Observe that $a d - b c$ is the determinant (Section 4.1) of A. The corresponding probability density function is given by 

$$
f (\boldsymbol {x}) = f \left(\boldsymbol {A} ^ {- 1} \boldsymbol {y}\right) = \frac {1}{2 \pi} \exp \left(- \frac {1}{2} \boldsymbol {y} ^ {\top} \boldsymbol {A} ^ {- \top} \boldsymbol {A} ^ {- 1} \boldsymbol {y}\right).\tag{6.148}
$$

The partial derivative of a matrix times a vector with respect to the vector is the matrix itself (Section 5.5), and therefore 

$$
\frac {\partial}{\partial \boldsymbol {y}} \boldsymbol {A} ^ {- 1} \boldsymbol {y} = \boldsymbol {A} ^ {- 1}.\tag{6.149}
$$

Recall from Section 4.1 that the determinant of the inverse is the inverse of the determinant so that the determinant of the Jacobian matrix is 

$$
\det \left(\frac {\partial}{\partial \boldsymbol {y}} \boldsymbol {A} ^ {- 1} \boldsymbol {y}\right) = \frac {1}{a d - b c}.\tag{6.150}
$$

We are now able to apply the change-of-variable formula from Theorem 6.16 by multiplying (6.148) with (6.150), which yields 

$$
f (\boldsymbol {y}) = f (\boldsymbol {x}) \left| \det \left(\frac {\partial}{\partial \boldsymbol {y}} \boldsymbol {A} ^ {- 1} \boldsymbol {y}\right) \right|\tag{6.151a}
$$

$$
= \frac {1}{2 \pi} \exp \left(- \frac {1}{2} \pmb {y} ^ {\top} \pmb {A} ^ {- \top} \pmb {A} ^ {- 1} \pmb {y}\right) | a d - b c | ^ {- 1}.\tag{6.151b}
$$

While Example 6.17 is based on a bivariate random variable, which allows us to easily compute the matrix inverse, the preceding relation holds for higher dimensions. 

Remark. We saw in Section 6.5 that the density $f ( { \pmb x } )$ in (6.148) is actually the standard Gaussian distribution, and the transformed density $f ( \boldsymbol { y } )$ is a bivariate Gaussian with covariance $\pmb { \Sigma } = \pmb { A } \pmb { A } ^ { \top }$ ◇ 

We will use the ideas in this chapter to describe probabilistic modeling in Section 8.4, as well as introduce a graphical language in Section 8.5. We will see direct machine learning applications of these ideas in Chapters 9 and 11. 

## 6.8 Further Reading

This chapter is rather terse at times. Grinstead and Snell (1997) and Walpole et al. (2011) provide more relaxed presentations that are suitable for self-study. Readers interested in more philosophical aspects of probability should consider Hacking (2001), whereas an approach that is more related to software engineering is presented by Downey (2014). An overview of exponential families can be found in Barndorff-Nielsen (2014). We will see more about how to use probability distributions to model machine learning tasks in Chapter 8. Ironically, the recent surge in interest in neural networks has resulted in a broader appreciation of probabilistic models. For example, the idea of normalizing flows (Jimenez Rezende and Mohamed, 2015) relies on change of variables for transforming random variables. An overview of methods for variational inference as applied to neural networks is described in chapters 16 to 20 of the book by Goodfellow et al. (2016). 

We side stepped a large part of the difficulty in continuous random variables by avoiding measure theoretic questions (Billingsley, 1995; Pollard, 2002), and by assuming without construction that we have real numbers, and ways of defining sets on real numbers as well as their appropriate frequency of occurrence. These details do matter, for example, in the specifi cation of conditional probability $p ( y \mid x )$ for continuous random variables $x , y$ (Proschan and Presnell, 1998). The lazy notation hides the fact that we want to specify that $X \ = \ x$ (which is a set of measure zero). Furthermore, we are interested in the probability density function of y. A more precise notation would have to say $\mathbb { E } _ { y } [ f ( y ) | \sigma ( x ) ]$ , where we take the expectation over y of a test function $f$ conditioned on the σ-algebra of x. A more technical audience interested in the details of probability theory have many options (Jaynes, 2003; MacKay, 2003; Jacod and Protter, 2004; Grimmett and Welsh, 2014), including some very technical discussions (Shiryayev, 1984; Lehmann and Casella, 1998; Dudley, 2002; Bickel and Doksum, 2006; C¸inlar, 2011). An alternative way to approach probability is to start with the concept of expectation, and “work backward” to derive the necessary properties of a probability space (Whittle, 2000). As machine learning allows us to model more intricate distributions on ever more complex types of data, a developer of probabilistic machine learning models would have to understand these more technical aspects. Ma chine learning texts with a probabilistic modeling focus include the books by MacKay (2003); Bishop (2006); Rasmussen and Williams (2006); Barber (2012); Murphy (2012). 



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



## Exercises

6.1 Consider the following bivariate distribution $p ( x , y )$ of two discrete random variables X and $Y$ . 

<table><tr><td>y1</td><td>0.01</td><td>0.02</td><td>0.03</td><td>0.1</td><td>0.1</td></tr><tr><td>y2</td><td>0.05</td><td>0.1</td><td>0.05</td><td>0.07</td><td>0.2</td></tr><tr><td>y3</td><td>0.1</td><td>0.05</td><td>0.03</td><td>0.05</td><td>0.04</td></tr></table>

Compute: 

a. The marginal distributions $p ( x )$ and $p ( y )$ 

b. The conditional distributions $p ( x | Y = y _ { 1 } )$ and $p ( y | X = x _ { 3 } )$ 

6.2 Consider a mixture of two Gaussian distributions (illustrated in Figure 6.4), 

$$
0. 4 \mathcal {N} \left(\left[ \begin{array}{c} 1 0 \\ 2 \end{array} \right], \left[ \begin{array}{c c} 1 & 0 \\ 0 & 1 \end{array} \right]\right) + 0. 6 \mathcal {N} \left(\left[ \begin{array}{c} 0 \\ 0 \end{array} \right], \left[ \begin{array}{c c} 8. 4 & 2. 0 \\ 2. 0 & 1. 7 \end{array} \right]\right).
$$

a. Compute the marginal distributions for each dimension. 

b. Compute the mean, mode and median for each marginal distribution. 

c. Compute the mean and mode for the two-dimensional distribution. 

You have written a computer program that sometimes compiles and sometimes not (code does not change). You decide to model the apparent stochasticity (success vs. no success) x of the compiler using a Bernoulli distribution with parameter $\mu \colon$ 

$$
p (x \mid \mu) = \mu^ {x} (1 - \mu) ^ {1 - x}, \quad x \in \{0, 1 \}.
$$

Choose a conjugate prior for the Bernoulli likelihood and compute the posterior distribution $p ( \mu | x _ { 1 } , \ldots , x _ { N } )$ . 

There are two bags. The first bag contains four mangos and two apples; the second bag contains four mangos and four apples. 

We also have a biased coin, which shows “heads” with probability 0.6 and “tails” with probability 0.4. If the coin shows “heads”. we pick a fruit at random from bag 1; otherwise we pick a fruit at random from bag 2. 

Your friend flips the coin (you cannot see the result), picks a fruit at random from the corresponding bag, and presents you a mango. 

What is the probability that the mango was picked from bag 2? 

Hint: Use Bayes’ theorem. 

6.5 Consider the time-series model 

$$
\begin{array}{c} \boldsymbol {x} _ {t + 1} = \boldsymbol {A} \boldsymbol {x} _ {t} + \boldsymbol {w}, \quad \boldsymbol {w} \sim \mathcal {N} (\boldsymbol {0}, \boldsymbol {Q}) \\ \boldsymbol {y} _ {t} = \boldsymbol {C} \boldsymbol {x} _ {t} + \boldsymbol {v}, \quad \boldsymbol {v} \sim \mathcal {N} (\boldsymbol {0}, \boldsymbol {R}), \end{array}
$$

where w, v are i.i.d. Gaussian noise variables. Further, assume that $p ( \pmb { x } _ { 0 } ) =$ $\mathcal { N } ( \mu _ { 0 } , \Sigma _ { 0 } )$ 

Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 

a. What is the form of $p ( { \pmb x } _ { 0 } , { \pmb x } _ { 1 } , \ldots , { \pmb x } _ { T } ) ?$ Justify your answer (you do not have to explicitly compute the joint distribution). 

b. Assume that $p ( \pmb { x } _ { t } | \pmb { y } _ { 1 } , \dots , \pmb { y } _ { t } ) = \mathcal { N } \big ( \pmb { \mu } _ { t } , \pmb { \Sigma } _ { t } \big )$ 

1. Compute $p ( \pmb { x } _ { t + 1 } \mid \pmb { y } _ { 1 } , \ldots , \pmb { y } _ { t } )$ 

2. Compute $p ( \pmb { x } _ { t + 1 } , \pmb { y } _ { t + 1 } \mid \pmb { y } _ { 1 } , \dots , \pmb { y } _ { t } ) .$ 

3. At time t+1, we observe the value $\pmb { y } _ { t + 1 } = \hat { \pmb { y } }$ . Compute the conditional distribution $p ( \pmb { x } _ { t + 1 } | \pmb { y } _ { 1 } , \dots , \pmb { y } _ { t + 1 } )$ 

6.6 Prove the relationship in $_ { ( 6 . 4 4 ) }$ , which relates the standard definition of the variance to the raw-score expression for the variance. 

6.7 Prove the relationship in (6.45), which relates the pairwise difference between examples in a dataset with the raw-score expression for the variance. 

6.8 Express the Bernoulli distribution in the natural parameter form of the ex ponential family, see (6.107). 

6.9 Express the Binomial distribution as an exponential family distribution. Also express the Beta distribution is an exponential family distribution. Show that the product of the Beta and the Binomial distribution is also a member of the exponential family. 

6.10 Derive the relationship in Section 6.5.2 in two ways: 

a. By completing the square 

b. By expressing the Gaussian in its exponential family form 

The product of two Gaussians $\mathcal { N } \big ( \mathbf { { x } } \mid \mathbf { { a } } , A \big ) \mathcal { N } \big ( \mathbf { { x } } \mid b , B \big )$ is an unnormalized Gaussian distribution $c { \mathcal { N } } ( { \pmb x } | { \pmb c } , { \pmb C } )$ with 

$$
\boldsymbol {C} = \left(\boldsymbol {A} ^ {- 1} + \boldsymbol {B} ^ {- 1}\right) ^ {- 1}
$$

$$
\boldsymbol {c} = \boldsymbol {C} (\boldsymbol {A} ^ {- 1} \boldsymbol {a} + \boldsymbol {B} ^ {- 1} \boldsymbol {b})
$$

$$
c = (2 \pi) ^ {- \frac {D}{2}} | \boldsymbol {A} + \boldsymbol {B} | ^ {- \frac {1}{2}} \exp \left(- \frac {1}{2} (\boldsymbol {a} - \boldsymbol {b}) ^ {\top} (\boldsymbol {A} + \boldsymbol {B}) ^ {- 1} (\boldsymbol {a} - \boldsymbol {b})\right).
$$

Note that the normalizing constant c itself can be considered a (normalized) Gaussian distribution either in a or in b with an “inflated” covariance matrix $A + B , \operatorname { i . e . , } c = { \mathcal { N } } { \big ( } a \mid b , A + B { \big ) } = { \mathcal { N } } { \big ( } b \mid a , A + B { \big ) }$ 

## 6.11 Iterated Expectations.

Consider two random variables $x , y$ with joint distribution $p ( x , y )$ . Show that 

$$
\mathbb {E} _ {X} [ x ] = \mathbb {E} _ {Y} \left[ \mathbb {E} _ {X} [ x \mid y ] \right].
$$

Here, $\mathbb { E } _ { X } [ x \mid y ]$ denotes the expected value of x under the conditional distribution $p ( x \mid y )$ 

## 6.12 Manipulation of Gaussian Random Variables.

Consider a Gaussian random variable $\pmb { x } \sim \mathcal { N } \big ( \pmb { x } | \pmb { \mu } _ { x } , \pmb { \Sigma } _ { x } \big )$ , where $\pmb { x } \in \mathbb { R } ^ { D }$ Furthermore, we have 

$$
\boldsymbol {y} = \boldsymbol {A} \boldsymbol {x} + \boldsymbol {b} + \boldsymbol {w},
$$

where $\pmb { y } \in \mathbb { R } ^ { E } , \pmb { A } \in \mathbb { R } ^ { E \times D } , \pmb { b } \in \mathbb { R } ^ { E }$ , and ${ \pmb w } \sim \mathcal { N } \big ( { \pmb w } | { \bf 0 } , Q \big )$ is independent Gaussian noise. “Independent” implies that x and w are independent random variables and that Q is diagonal. 

a. Write down the likelihood $p ( \pmb { y } \vert \pmb { x } )$ 

b. The distribution $\begin{array} { r } { p ( { \pmb y } ) = \int p ( { \pmb y } | { \pmb x } ) p ( { \pmb x } ) } \end{array}$ dx is Gaussian. Compute the mean $\mu _ { y }$ and the covariance $\Sigma _ { y }$ . Derive your result in detail. 

©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 

c. The random variable $\textbf {  { y } }$ is being transformed according to the measurement mapping 

$$
\boldsymbol {z} = \boldsymbol {C y} + \boldsymbol {v},
$$

where $\boldsymbol { z } \in \mathbb { R } ^ { F } , \boldsymbol { C } \in \mathbb { R } ^ { F \times E }$ , and ${ \pmb v } \sim \mathcal { N } \big ( { \pmb v } \mid { \bf 0 } , { \pmb R } \big )$ is independent Gaussian (measurement) noise. 

Write down $p ( z \mid \pmb { y } )$ 

Compute $p ( z )$ , i.e., the mean $\mu _ { z }$ and the covariance $\Sigma _ { z }$ . Derive your result in detail. 

d. Now, a value $\hat { \pmb y }$ is measured. Compute the posterior distribution $p ( \pmb { x } | \hat { \pmb y } )$ Hint for solution: This posterior is also Gaussian, i.e., we need to determine only its mean and covariance matrix. Start by explicitly computing the joint Gaussian $p ( { \pmb x } , { \pmb y } )$ . This also requires us to compute the cross-covariances $\mathrm { C o v } _ { \pmb { x } , \pmb { y } } [ \pmb { x } , \pmb { y } ]$ and $\mathrm { C o v } _ { { \pmb y } , { \pmb x } } [ { \pmb y } , { \pmb x } ]$ . Then apply the rules for Gaussian conditioning. 

## 6.13 Probability Integral Transformation

Given a continuous random variable $X ,$ , with cdf $F _ { X } ( x )$ , show that the random variable $Y : = F _ { X } ( X )$ is uniformly distributed (Theorem 6.15). 

# Continuous Optimization

Since machine learning algorithms are implemented on a computer, the mathematical formulations are expressed as numerical optimization meth ods. This chapter describes the basic numerical methods for training machine learning models. Training a machine learning model often boils down to finding a good set of parameters. The notion of “good” is determined by the objective function or the probabilistic model, which we will see examples of in the second part of this book. Given an objective function, finding the best value is done using optimization algorithms. 

This chapter covers two main branches of continuous optimization (Figure 7.1): unconstrained and constrained optimization. We will assume in this chapter that our objective function is differentiable (see Chapter 5), hence we have access to a gradient at each location in the space to help us find the optimum value. By convention, most objective functions in machine learning are intended to be minimized, that is, the best value is the minimum value. Intuitively finding the best value is like finding the valleys of the objective function, and the gradients point us uphill. The idea is to move downhill (opposite to the gradient) and hope to find the deepest point. For unconstrained optimization, this is the only concept we need, but there are several design choices, which we discuss in Section 7.1. For constrained optimization, we need to introduce other concepts to manage the constraints (Section 7.2). We will also introduce a special class of problems (convex optimization problems in Section 7.3) where we can make statements about reaching the global optimum. 

Consider the function in Figure 7.2. The function has a global minimum around $x \ = \ - 4 . 5$ , with a function value of approximately 47. Since the function is “smooth,” the gradients can be used to help find the minimum by indicating whether we should take a step to the right or left. This assumes that we are in the correct bowl, as there exists another local minimum around $x = 0 . 7$ . Recall that we can solve for all the stationary points of a function by calculating its derivative and setting it to zero. For 

$$
\ell (x) = x ^ {4} + 7 x ^ {3} + 5 x ^ {2} - 1 7 x + 3,\tag{7.1}
$$

we obtain the corresponding gradient as 

$$
{\frac {\mathrm{d} \ell (x)}{\mathrm{d} x}} = 4 x ^ {3} + 2 1 x ^ {2} + 1 0 x - 1 7.\tag{7.2}
$$

225 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/ce67965678ea69d5ffc1646680d6239114b976f539de137e07ec9bb79d392462.jpg)


Since we consider data and models in $\mathbb { R } ^ { D }$ , the optimization problems we face are continuous optimization problems, as opposed to combinatorial optimization problems for discrete variables. 

Stationary points are the real roots of the derivative, that is, points that have zero gradient. 


Figure 7.1 A mind map of the concepts related to optimization, as presented in this chapter. There are two main ideas: gradient descent and convex optimization.



Continuous Optimization


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/7c940f623ce4d8a61e6fcd70feccfc1bd50a5786eae0166f1e317f14eb37e6db.jpg)


Since this is a cubic equation, it has in general three solutions when set to zero. In the example, two of them are minimums and one is a maximum (around $x = - 1 . 4 )$ . To check whether a stationary point is a minimum or maximum, we need to take the derivative a second time and check whether the second derivative is positive or negative at the stationary point. In our case, the second derivative is 

$$
\frac {\mathrm{d} ^ {2} \ell (x)}{\mathrm{d} x ^ {2}} = 1 2 x ^ {2} + 4 2 x + 1 0.\tag{7.3}
$$

By substituting our visually estimated values of $x = - 4 . 5 , - 1 . 4 , 0 . 7 .$ , we will observe that as expected the middle point is a maximum $\left( \frac { \mathrm { d } ^ { 2 } \ell ( x ) } { \mathrm { d } x ^ { 2 } } < 0 \right)$ and the other two stationary points are minimums. 

Note that we have avoided analytically solving for values of x in the previous discussion, although for low-order polynomials such as the preceding we could do so. In general, we are unable to find analytic solutions, and hence we need to start at some value, say $x _ { 0 } = - 6 ,$ , and follow the negative gradient. The negative gradient indicates that we should go right, but not how far (this is called the step-size). Furthermore, if we had started at the right side $( \boldsymbol { \mathrm { e } } . \boldsymbol { \mathrm { g } } . , x _ { 0 } = 0 )$ the negative gradient would have led us to the wrong minimum. Figure 7.2 illustrates the fact that for $x > - 1$ , the negative gradient points toward the minimum on the right of the figure, which has a larger objective value. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/8247c0ca34e7f81583bdc3f76feec3d26185d416456a59257804e6aaef8ada5f.jpg)



Figure 7.2 Example objective function. Negative gradients are indicated by arrows, and the global minimum is indicated by the dashed blue line.


In Section 7.3, we will learn about a class of functions, called convex functions, that do not exhibit this tricky dependency on the starting point of the optimization algorithm. For convex functions, all local minimums are global minimum. It turns out that many machine learning objective functions are designed such that they are convex, and we will see an example in Chapter 12. 

The discussion in this chapter so far was about a one-dimensional function, where we are able to visualize the ideas of gradients, descent direc tions, and optimal values. In the rest of this chapter we develop the same ideas in high dimensions. Unfortunately, we can only visualize the concepts in one dimension, but some concepts do not generalize directly to higher dimensions, therefore some care needs to be taken when reading. 

## 7.1 Optimization Using Gradient Descent

We now consider the problem of solving for the minimum of a real-valued function 

$$
\min _ {\boldsymbol {x}} f (\boldsymbol {x}),\tag{7.4}
$$



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



where $f : \mathbb { R } ^ { d }  \mathbb { R }$ is an objective function that captures the machine learning problem at hand. We assume that our function f is differentiable, and we are unable to analytically find a solution in closed form. 

Gradient descent is a first-order optimization algorithm. To find a local minimum of a function using gradient descent, one takes steps proportional to the negative of the gradient of the function at the current point. Recall from Section 5.1 that the gradient points in the direction of the steepest ascent. Another useful intuition is to consider the set of lines where the function is at a certain value $( f ( \pmb { x } ) = c$ for some value $c \in \mathbb { R } )$ which are known as the contour lines. The gradient points in a direction that is orthogonal to the contour lines of the function we wish to optimize. 

Let us consider multivariate functions. Imagine a surface (described by the function $f ( { \pmb x } ) )$ with a ball starting at a particular location $\scriptstyle { \pmb x } _ { 0 }$ . When the ball is released, it will move downhill in the direction of steepest descent. Gradient descent exploits the fact that $f ( \pmb { x } _ { 0 } )$ decreases fastest if one moves from $\scriptstyle { \pmb x } _ { 0 }$ in the direction of the negative gradient $( ( \nabla f ) ( \pmb { x } _ { 0 } ) ) ^ { \top }$ of $f { \mathrm { ~ a t ~ } } x _ { 0 }$ . We assume in this book that the functions are differentiable, and refer the reader to more general settings in Section 7.4. Then, if 

$$
\boldsymbol {x} _ {1} = \boldsymbol {x} _ {0} - \gamma ((\nabla f) (\boldsymbol {x} _ {0})) ^ {\top}\tag{7.5}
$$

for a small step-size $\gamma \geqslant 0 .$ , then $f ( \pmb { x } _ { 1 } ) \leqslant f ( \pmb { x } _ { 0 } )$ . Note that we use the transpose for the gradient since otherwise the dimensions will not work out. 

This observation allows us to define a simple gradient descent algorithm: If we want to find a local optimum $f ( x _ { * } )$ of a function $f : \mathbb { R } ^ { n } $ R, $x \mapsto f ( { \pmb x } )$ , we start with an initial guess $\scriptstyle { \pmb x } _ { 0 }$ of the parameters we wish to optimize and then iterate according to 

$$
\boldsymbol {x} _ {i + 1} = \boldsymbol {x} _ {i} - \gamma_ {i} ((\nabla f) (\boldsymbol {x} _ {i})) ^ {\top}.\tag{7.6}
$$

For suitable step-size $\gamma _ { i : }$ , the sequence $f ( { \pmb x } _ { 0 } ) \geqslant f ( { \pmb x } _ { 1 } ) \geqslant . .$ . converges to a local minimum. 

## Example 7.1

Consider a quadratic function in two dimensions 

$$
f \left(\left[ \begin{array}{c} x _ {1} \\ x _ {2} \end{array} \right]\right) = \frac {1}{2} \left[ \begin{array}{c} x _ {1} \\ x _ {2} \end{array} \right] ^ {\top} \left[ \begin{array}{c c} 2 & 1 \\ 1 & 2 0 \end{array} \right] \left[ \begin{array}{c} x _ {1} \\ x _ {2} \end{array} \right] - \left[ \begin{array}{c} 5 \\ 3 \end{array} \right] ^ {\top} \left[ \begin{array}{c} x _ {1} \\ x _ {2} \end{array} \right]\tag{7.7}
$$

with gradient 

$$
\nabla f \left(\left[ \begin{array}{c} x _ {1} \\ x _ {2} \end{array} \right]\right) = \left[ \begin{array}{c} x _ {1} \\ x _ {2} \end{array} \right] ^ {\top} \left[ \begin{array}{c c} 2 & 1 \\ 1 & 2 0 \end{array} \right] - \left[ \begin{array}{c} 5 \\ 3 \end{array} \right] ^ {\top}.\tag{7.8}
$$

Starting at the initial location $\pmb { x } _ { 0 } = [ - 3 , - 1 ] ^ { \top }$ , we iteratively apply (7.6) to obtain a sequence of estimates that converge to the minimum value 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/1338a90c887117eb68145185a1a21ac71812fadab047b0f2e8b42dbc1b9ebc56.jpg)



(illustrated in Figure 7.3). We can see (both from the figure and by plugging $\scriptstyle { \pmb x } _ { 0 }$ into (7.8) with $\gamma = 0 . 0 8 5 )$ that the negative gradient at $\scriptstyle { \pmb x } _ { 0 }$ points north and east, leading to $\pmb { x } _ { 1 } = [ - 1 . 9 8 , 1 . 2 1 ] ^ { \top }$ . Repeating that argument gives us $\pmb { x } _ { 2 } = [ - 1 . 3 2 , - 0 . 4 2 ] ^ { \top }$ , and so on.


Remark. Gradient descent can be relatively slow close to the minimum: Its asymptotic rate of convergence is inferior to many other methods. Using the ball rolling down the hill analogy, when the surface is a long, thin valley, the problem is poorly conditioned (Trefethen and Bau III, 1997). For poorly conditioned convex problems, gradient descent increasingly “zigzags” as the gradients point nearly orthogonally to the shortest direction to a minimum point; see Figure 7.3. 

## 7.1.1 Step-size

As mentioned earlier, choosing a good step-size is important in gradient descent. If the step-size is too small, gradient descent can be slow. If the step-size is chosen too large, gradient descent can overshoot, fail to converge, or even diverge. We will discuss the use of momentum in the next section. It is a method that smoothes out erratic behavior of gradient updates and dampens oscillations. 

Adaptive gradient methods rescale the step-size at each iteration, depending on local properties of the function. There are two simple heuristics (Toussaint, 2012): 

When the function value increases after a gradient step, the step-size was too large. Undo the step and decrease the step-size. 

When the function value decreases the step could have been larger. Try to increase the step-size. 

Although the “undo” step seems to be a waste of resources, using this heuristic guarantees monotonic convergence. 

## Example 7.2 (Solving a Linear Equation System)

When we solve linear equations of the form $\mathbf { { \boldsymbol { A } } } \mathbf { { \boldsymbol { x } } } = \mathbf { { \boldsymbol { b } } } ,$ in practice we solve $\mathbf { } A \mathbf { } x - b = \mathbf { 0 }$ approximately by finding x<sub>∗</sub> that minimizes the squared error 

$$
\left\| \boldsymbol {A} \boldsymbol {x} - \boldsymbol {b} \right\| ^ {2} = (\boldsymbol {A} \boldsymbol {x} - \boldsymbol {b}) ^ {\top} (\boldsymbol {A} \boldsymbol {x} - \boldsymbol {b})\tag{7.9}
$$

if we use the Euclidean norm. The gradient of (7.9) with respect to x is 

$$
\nabla_ {\boldsymbol {x}} = 2 (\boldsymbol {A} \boldsymbol {x} - \boldsymbol {b}) ^ {\top} \boldsymbol {A}.\tag{7.10}
$$

We can use this gradient directly in a gradient descent algorithm. However, for this particular special case, it turns out that there is an analytic solution, which can be found by setting the gradient to zero. We will see more on solving squared error problems in Chapter 9. 

Remark. When applied to the solution of linear systems of equations ${ \pmb A } { \pmb x } =$ $^ { b , }$ gradient descent may converge slowly. The speed of convergence of gradient descent is dependent on the condition number $\begin{array} { r } { \kappa = \frac { \sigma ( \mathbf { \breve { A } } ) _ { \mathrm { m a x } } } { \sigma ( A ) _ { \mathrm { m i n } } } } \end{array}$ , which is the ratio of the maximum to the minimum singular value (Section 4.5) of A. The condition number essentially measures the ratio of the most curved direction versus the least curved direction, which corresponds to our imagery that poorly conditioned problems are long, thin valleys: They are very curved in one direction, but very flat in the other. Instead of directly solving $\mathbf { { \boldsymbol { A } } } \mathbf { { \boldsymbol { x } } } = \mathbf { { \boldsymbol { b } } } ,$ one could instead solve $P ^ { - 1 } ( A x - b ) = \mathbf { 0 } ,$ where $_ { r }$ is called the preconditioner. The goal is to design $\scriptstyle { \dot { \boldsymbol { P } } } ^ { - 1 }$ such that $P ^ { - 1 } A$ has a better condition number, but at the same time $P ^ { - 1 }$ is easy to compute. For further information on gradient descent, preconditioning, and convergence we refer to Boyd and Vandenberghe (2004, chapter 9). $\diamondsuit$ 

## 7.1.2 Gradient Descent With Momentum

As illustrated in Figure 7.3, the convergence of gradient descent may be very slow if the curvature of the optimization surface is such that there are regions that are poorly scaled. The curvature is such that the gradient descent steps hops between the walls of the valley and approaches the optimum in small steps. The proposed tweak to improve convergence is to give gradient descent some memory. 

Gradient descent with momentum (Rumelhart et al., 1986) is a method that introduces an additional term to remember what happened in the previous iteration. This memory dampens oscillations and smoothes out the gradient updates. Continuing the ball analogy, the momentum term emulates the phenomenon of a heavy ball that is reluctant to change directions. The idea is to have a gradient update with memory to implement a moving average. The momentum-based method remembers the update $\Delta { \pmb x } _ { i }$ at each iteration i and determines the next update as a linear combination of the current and previous gradients 

$$
\pmb {x} _ {i + 1} = \pmb {x} _ {i} - \gamma_ {i} ((\nabla f) (\pmb {x} _ {i})) ^ {\top} + \alpha \Delta \pmb {x} _ {i}\tag{7.11}
$$

$$
\Delta \pmb {x} _ {i} = \pmb {x} _ {i} - \pmb {x} _ {i - 1} = \alpha \Delta \pmb {x} _ {i - 1} - \gamma_ {i - 1} ((\nabla f) (\pmb {x} _ {i - 1})) ^ {\top},\tag{7.12}
$$

where $\alpha ~ \in ~ [ 0 , 1 ]$ . Sometimes we will only know the gradient approxi mately. In such cases, the momentum term is useful since it averages out different noisy estimates of the gradient. One particularly useful way to obtain an approximate gradient is by using a stochastic approximation, which we discuss next. 

## 7.1.3 Stochastic Gradient Descent

Computing the gradient can be very time consuming. However, often it is possible to find a “cheap” approximation of the gradient. Approximating the gradient is still useful as long as it points in roughly the same direction as the true gradient. 

Stochastic gradient descent (often shortened as SGD) is a stochastic approximation of the gradient descent method for minimizing an objective function that is written as a sum of differentiable functions. The word stochastic here refers to the fact that we acknowledge that we do not know the gradient precisely, but instead only know a noisy approximation to it. By constraining the probability distribution of the approximate gradients, we can still theoretically guarantee that SGD will converge. 

In machine learning, given $n = 1 , \ldots , N$ data points, we often consider objective functions that are the sum of the losses $L _ { n }$ incurred by each example n. In mathematical notation, we have the form 

$$
L (\boldsymbol {\theta}) = \sum_ {n = 1} ^ {N} L _ {n} (\boldsymbol {\theta}),\tag{7.13}
$$

where $\pmb \theta$ is the vector of parameters of interest, i.e., we want to find θ that minimizes L. An example from regression (Chapter 9) is the negative loglikelihood, which is expressed as a sum over log-likelihoods of individual examples so that 

$$
L (\boldsymbol {\theta}) = - \sum_ {n = 1} ^ {N} \log p (y _ {n} | \boldsymbol {x} _ {n}, \boldsymbol {\theta}),\tag{7.14}
$$

where ${ \pmb x } _ { n } \in \mathbb { R } ^ { D }$ are the training inputs, $y _ { n }$ are the training targets, and θ are the parameters of the regression model. 

Standard gradient descent, as introduced previously, is a “batch” optimization method, i.e., optimization is performed using the full training set by updating the vector of parameters according to 

$$
\boldsymbol {\theta} _ {i + 1} = \boldsymbol {\theta} _ {i} - \gamma_ {i} (\nabla L (\boldsymbol {\theta} _ {i})) ^ {\top} = \boldsymbol {\theta} _ {i} - \gamma_ {i} \sum_ {n = 1} ^ {N} (\nabla L _ {n} (\boldsymbol {\theta} _ {i})) ^ {\top}\tag{7.15}
$$

for a suitable step-size parameter $\gamma _ { i }$ . Evaluating the sum gradient may require expensive evaluations of the gradients from all individual functions $L _ { n }$ . When the training set is enormous and/or no simple formulas exist, evaluating the sums of gradients becomes very expensive. 

Consider the term $\begin{array} { r l } {  { \sum _ { n = 1 } ^ { \bar { N } } ( \nabla L _ { n } ( \pmb { \theta } _ { i } ) ) } } & { { } } \end{array}$ in (7.15). We can reduce the amount of computation by taking a sum over a smaller set of $L _ { n }$ . In contrast to batch gradient descent, which uses all $L _ { n }$ for $n = 1 , \ldots , N$ , we randomly choose a subset of $L _ { n }$ for mini-batch gradient descent. In the extreme case, we randomly select only a single $L _ { n }$ to estimate the gradient. The key insight about why taking a subset of data is sensible is to realize that for gradient descent to converge, we only require that the gradient is an unbiased estimate of the true gradient. In fact the term $\begin{array} { r l } {  { \sum _ { n = 1 } ^ { \bar { N } } ( \nabla L _ { n } ( \pmb { \theta } _ { i } ) ) } \qquad } & { { } } \end{array}$ in (7.15) is an empirical estimate of the expected value (Section 6.4.1) of the gradient. Therefore, any other unbiased empirical estimate of the expected value, for example using any subsample of the data, would suffice for convergence of gradient descent. 

Remark. When the learning rate decreases at an appropriate rate, and subject to relatively mild assumptions, stochastic gradient descent converges almost surely to local minimum (Bottou, 1998). ◇ 

Why should one consider using an approximate gradient? A major reason is practical implementation constraints, such as the size of central processing unit (CPU)/graphics processing unit (GPU) memory or limits on computational time. We can think of the size of the subset used to estimate the gradient in the same way that we thought of the size of a sample when estimating empirical means (Section 6.4.1). Large mini-batch sizes will provide accurate estimates of the gradient, reducing the variance in the parameter update. Furthermore, large mini-batches take advantage of highly optimized matrix operations in vectorized implementations of the cost and gradient. The reduction in variance leads to more stable convergence, but each gradient calculation will be more expensive. 

In contrast, small mini-batches are quick to estimate. If we keep the mini-batch size small, the noise in our gradient estimate will allow us to get out of some bad local optima, which we may otherwise get stuck in. In machine learning, optimization methods are used for training by minimizing an objective function on the training data, but the overall goa is to improve generalization performance (Chapter 8). Since the goal in machine learning does not necessarily need a precise estimate of the minimum of the objective function, approximate gradients using mini-batch approaches have been widely used. Stochastic gradient descent is very effective in large-scale machine learning problems (Bottou et al., 2018), 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/0df6007690fbc6c425d8212066f288b62bb59161ac6f66c16d1b9b144ac0791b.jpg)



Figure 7.4 Illustration of constrained optimization. The unconstrained problem (indicated by the contour lines) has a minimum on the right side (indicated by the circle). The box constraints $( - 1 \leqslant x \leqslant 1$ and $- 1 \leqslant y \leqslant 1 )$ require that the optimal solution is within the box, resulting in an optimal value indicated by the star.



such as training deep neural networks on millions of images (Dean et al., 2012), topic models (Hoffman et al., 2013), reinforcement learning (Mnih et al., 2015), or training of large-scale Gaussian process models (Hensman et al., 2013; Gal et al., 2014).


## 7.2 Constrained Optimization and Lagrange Multipliers

In the previous section, we considered the problem of solving for the minimum of a function 

$$
\min _ {\boldsymbol {x}} f (\boldsymbol {x}),\tag{7.16}
$$

where $f : \mathbb { R } ^ { D }  \mathbb { R }$ 

In this section, we have additional constraints. That is, for real-valued functions $g _ { i } : \mathbb { R } ^ { D } $ <sup>R</sup> for $i = 1 , \ldots , m$ , we consider the constrained optimization problem (see Figure 7.4 for an illustration) 

$$
\begin{array}{r l} \min _ {\boldsymbol {x}} & f (\boldsymbol {x}) \\ \text { subject   to } & g _ {i} (\boldsymbol {x}) \leqslant 0 \quad \text { for   all } \quad i = 1, \ldots , m. \end{array}\tag{7.17}
$$

It is worth pointing out that the functions $f$ and $g _ { i }$ could be non-convex in general, and we will consider the convex case in the next section. 

One obvious, but not very practical, way of converting the constrained problem (7.17) into an unconstrained one is to use an indicator function 

$$
J (\boldsymbol {x}) = f (\boldsymbol {x}) + \sum_ {i = 1} ^ {m} \mathbf {1} (g _ {i} (\boldsymbol {x})),\tag{7.18}
$$

©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 

where $\mathbf { 1 } ( z )$ is an infinite step function 

$$
\mathbf {1} (z) = \left\{ \begin{array}{l l} 0 & \text { if } z \leqslant 0 \\ \infty & \text { otherwise } \end{array} \right..\tag{7.19}
$$

This gives infinite penalty if the constraint is not satisfied, and hence would provide the same solution. However, this infinite step function is equally difficult to optimize. We can overcome this difficulty by introducing Lagrange multipliers. The idea of Lagrange multipliers is to replace the step function with a linear function. 

We associate to problem (7.17) the Lagrangian by introducing the Lagrange multipliers $\lambda _ { i } \geqslant 0$ corresponding to each inequality constraint respectively (Boyd and Vandenberghe, 2004, chapter 4) so that 

$$
\begin{array}{c} \mathfrak {L} (\boldsymbol {x}, \boldsymbol {\lambda}) = f (\boldsymbol {x}) + \sum_ {i = 1} ^ {m} \lambda_ {i} g _ {i} (\boldsymbol {x}) \\ = f (\boldsymbol {x}) + \boldsymbol {\lambda} ^ {\top} \boldsymbol {g} (\boldsymbol {x}), \end{array}\tag{7.20a}
$$

(7.20b) 

where in the last line we have concatenated all constraints $g _ { i } ( { \pmb x } )$ into a vector $\pmb { g } ( \pmb { x } )$ , and all the Lagrange multipliers into a vector $\pmb { \lambda } \in \mathbb { R } ^ { m }$ 

We now introduce the idea of Lagrangian duality. In general, duality in optimization is the idea of converting an optimization problem in one set of variables x (called the primal variables), into another optimization problem in a different set of variables λ (called the dual variables). We introduce two different approaches to duality: In this section, we discuss Lagrangian duality; in Section 7.3.3, we discuss Legendre-Fenchel duality. 

Definition 7.1. The problem in (7.17) 

$$
\begin{array}{r l} \min _ {\boldsymbol {x}} & f (\boldsymbol {x}) \\ \text { subject   to } & g _ {i} (\boldsymbol {x}) \leqslant 0 \quad \text { for   all } \quad i = 1, \ldots , m \end{array}\tag{7.21}
$$

is known as the primal problem, corresponding to the primal variables x. The associated Lagrangian dual problem is given by 

$$
\begin{array}{c c} \max _ {\boldsymbol {\lambda} \in \mathbb {R} ^ {m}} & \mathfrak {D} (\boldsymbol {\lambda}) \\ \text { subject   to } & \boldsymbol {\lambda} \geqslant \boldsymbol {0}, \end{array}\tag{7.22}
$$

where λ are the dual variables and ${ \mathfrak { D } } ( \lambda ) = \operatorname* { m i n } _ { \pmb { x } \in \mathbb { R } ^ { d } } { \mathfrak { L } } ( \pmb { x } , \lambda )$ 

Remark. In the discussion of Definition 7.1, we use two concepts that are also of independent interest (Boyd and Vandenberghe, 2004). 

First is the minimax inequality, which says that for any function with two arguments $\varphi ( { \pmb x } , { \pmb y } )$ , the maximin is less than the minimax, i.e., 

$$
\max _ {\boldsymbol {y}} \min _ {\boldsymbol {x}} \varphi (\boldsymbol {x}, \boldsymbol {y}) \leqslant \min _ {\boldsymbol {x}} \max _ {\boldsymbol {y}} \varphi (\boldsymbol {x}, \boldsymbol {y}).\tag{7.23}
$$

Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 

This inequality can be proved by considering the inequality 

$$
\text { For   all } \boldsymbol {x}, \boldsymbol {y} \quad \min _ {\boldsymbol {x}} \varphi (\boldsymbol {x}, \boldsymbol {y}) \leqslant \max _ {\boldsymbol {y}} \varphi (\boldsymbol {x}, \boldsymbol {y}).\tag{7.24}
$$

Note that taking the maximum over y of the left-hand side of (7.24) maintains the inequality since the inequality is true for all y. Similarly, we can take the minimum over x of the right-hand side of $( 7 . 2 4 )$ to obtain (7.23). 

The second concept is weak duality, which uses (7.23) to show that primal values are always greater than or equal to dual values. This is described in more detail in (7.27). 

Recall that the difference between $J ( x )$ in (7.18) and the Lagrangian in (7.20b) is that we have relaxed the indicator function to a linear function. Therefore, when $\lambda \geqslant 0$ , the Lagrangian $\mathfrak { L } ( \pmb { x } , \pmb { \lambda } )$ is a lower bound of $J ( x )$ . Hence, the maximum of ${ \mathfrak { L } } ( { \pmb x } , { \pmb \lambda } )$ with respect to λ is 

$$
J (\boldsymbol {x}) = \max _ {\boldsymbol {\lambda} \geqslant 0} \mathfrak {L} (\boldsymbol {x}, \boldsymbol {\lambda}).\tag{7.25}
$$

Recall that the original problem was minimizing $J ( x )$ 

$$
\min _ {\boldsymbol {x} \in \mathbb {R} ^ {d}} \max _ {\boldsymbol {\lambda} \geqslant \boldsymbol {0}} \mathfrak {L} (\boldsymbol {x}, \boldsymbol {\lambda}).\tag{7.26}
$$

By the minimax inequality (7.23), it follows that swapping the order of the minimum and maximum results in a smaller value, i.e., 

$$
\min _ {\boldsymbol {x} \in \mathbb {R} ^ {d}} \max _ {\boldsymbol {\lambda} \geqslant \boldsymbol {0}} \mathfrak {L} (\boldsymbol {x}, \boldsymbol {\lambda}) \geqslant \max _ {\boldsymbol {\lambda} \geqslant \boldsymbol {0}} \min _ {\boldsymbol {x} \in \mathbb {R} ^ {d}} \mathfrak {L} (\boldsymbol {x}, \boldsymbol {\lambda}).\tag{7.27}
$$

This is also known as weak duality. Note that the inner part of the righthand side is the dual objective function $\mathfrak { D } ( \boldsymbol { \lambda } )$ and the definition follows. 

In contrast to the original optimization problem, which has constraints, $\mathrm { m i n } _ { { \pmb x } \in \mathbb { R } ^ { d } } { \mathfrak { L } } ( { \pmb x } , \lambda )$ is an unconstrained optimization problem for a given value of λ. If solving $\mathrm { m i n } _ { { \pmb x } \in \mathbb { R } ^ { d } } { \mathfrak { L } } ( { \pmb x } , \lambda )$ is easy, then the overall problem is easy to solve. We can see this by observing from (7.20b) that $\mathfrak { L } ( \pmb { x } , \pmb { \lambda } )$ is affine with respect to λ. Therefore $\scriptstyle \operatorname* { m i n } _ { { \pmb x } \in \mathbb { R } ^ { d } } { \mathfrak { L } } ( { \pmb x } , { \pmb \lambda } )$ is a pointwise minimum of affine functions of $\lambda ,$ and hence ${ \mathfrak { D } } ( \lambda )$ is concave even though $f ( \cdot )$ and $g _ { i } ( \cdot )$ may be nonconvex. The outer problem, maximization over λ, is the maximum of a concave function and can be efficiently computed. 

Assuming $f ( \cdot )$ and $g _ { i } ( \cdot )$ are differentiable, we find the Lagrange dual problem by differentiating the Lagrangian with respect to ${ \mathbf { } } ^ { \mathbf { } } \mathbf { { x } } ,$ , setting the differential to zero, and solving for the optimal value. We will discuss two concrete examples in Sections 7.3.1 and 7.3.2, where $f ( \cdot )$ and $g _ { i } ( \cdot )$ are convex. 

Remark (Equality Constraints). Consider (7.17) with additional equality constraints 

$$
\begin{array}{r l} \min _ {\boldsymbol {x}} & f (\boldsymbol {x}) \\ \text { subject   to } & g _ {i} (\boldsymbol {x}) \leqslant 0 \quad \text { for   all } \quad i = 1, \ldots , m \\ & h _ {j} (\boldsymbol {x}) = 0 \quad \text { for   all } \quad j = 1, \ldots , n. \end{array}\tag{7.28}
$$

©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 

We can model equality constraints by replacing them with two inequality constraints. That is for each equality constraint $h _ { j } ( { \pmb x } ) = 0$ we equivalently replace it by two constraints $h _ { j } ( { \pmb x } ) \leqslant 0$ and $h _ { j } ( { \pmb x } ) \geqslant 0 .$ . It turns out that the resulting Lagrange multipliers are then unconstrained. 

Therefore, we constrain the Lagrange multipliers corresponding to the inequality constraints in (7.28) to be non-negative, and leave the Lagrange multipliers corresponding to the equality constraints unconstrained. 

## 7.3 Convex Optimization

We focus our attention of a particularly useful class of optimization problems, where we can guarantee global optimality. When $f ( \cdot )$ is a convex function, and when the constraints involving $g ( \cdot )$ and $h ( \cdot )$ are convex sets, this is called a convex optimization problem. In this setting, we have strong duality: The optimal solution of the dual problem is the same as the optimal solution of the primal problem. The distinction between convex functions and convex sets are often not strictly presented in machine learning literature, but one can often infer the implied meaning from context. 

Definition 7.2. A set is a convex set if for any $x , y \in { \mathcal { C } }$ and for any scalar θ with $0 \leqslant \theta \leqslant 1$ , we have 

$$
\theta x + (1 - \theta) y \in \mathcal {C}.
$$


Figure 7.5 Example of a convex set.


(7.29) 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/9a9163897b6d0d9c5a3d887d2f25ed614df4cbba6268b34dc9def256918203d0.jpg)



Figure 7.6 Example of a nonconvex set.


Convex sets are sets such that a straight line connecting any two elements of the set lie inside the set. Figures 7.5 and 7.6 illustrate convex and nonconvex sets, respectively. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/71e0b00377215aee1c5afc4001e76d6fdc8db7858073880490fb03ed20db07ff.jpg)


Convex functions are functions such that a straight line between any two points of the function lie above the function. Figure 7.2 shows a nonconvex function, and Figure 7.3 shows a convex function. Another convex function is shown in Figure 7.7. 


convex function concave function


Definition 7.3. Let function $f : \mathbb { R } ^ { D }  \mathbb { R }$ be a function whose domain is a convex set. The function f is a convex function if for all x, y in the domain of $f _ { i }$ , and for any scalar $\theta$ with $0 \leqslant \theta \leqslant 1$ , we have 

$$
f (\theta \boldsymbol {x} + (1 - \theta) \boldsymbol {y}) \leqslant \theta f (\boldsymbol {x}) + (1 - \theta) f (\boldsymbol {y}).\tag{7.30}
$$

Remark. A concave function is the negative of a convex function. 

$$
\diamondsuit
$$

The constraints involving $g ( \cdot )$ and $h ( \cdot )$ in (7.28) truncate functions at a scalar value, resulting in sets. Another relation between convex functions and convex sets is to consider the set obtained by “filling $\mathrm { i n } ^ { \dag }$ a convex function. A convex function is a bowl-like object, and we imagine pouring water into it to fill it up. This resulting filled-in set, called the epigraph of the convex function, is a convex set. 

If a function $f : \mathbb { R } ^ { n }  \mathbb { R }$ is differentiable, we can specify convexity in terms of its gradient $\nabla _ { \pmb { x } } f ( \pmb { x } )$ (Section 5.2). A function $f ( { \pmb x } )$ is convex if and only if for any two points x, y it holds that 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/3356114dbe4d83330c1182c641a89e2a9cfa0d480b6bec66ff7f973a3fa582cb.jpg)



Figure 7.7 Example of a convex function.


$$
f (\boldsymbol {y}) \geqslant f (\boldsymbol {x}) + \nabla_ {\boldsymbol {x}} f (\boldsymbol {x}) ^ {\top} (\boldsymbol {y} - \boldsymbol {x}).\tag{7.31}
$$

If we further know that a function $f ( { \pmb x } )$ is twice differentiable, that is, the Hessian (5.147) exists for all values in the domain of x, then the function $f ( { \pmb x } )$ is convex if and only if $\nabla _ { x } ^ { 2 } f ( x )$ is positive semidefinite (Boyd and Vandenberghe, 2004). 

## Example 7.3

The negative entropy $f ( x ) = x \log _ { 2 } x$ is convex for $x > 0$ . A visualization of the function is shown in Figure $7 . 8 ,$ , and we can see that the function is convex. To illustrate the previous definitions of convexity, let us check the calculations for two points $x = 2$ and $x = 4$ . Note that to prove convexity of $f ( x )$ we would need to check for all points $x \in \mathbb { R }$ 

Recall Definition 7.3. Consider a point midway between the two points (that is $\theta = 0 . 5 )$ ; then the left-hand side is $f ( 0 . 5 \cdot 2 + 0 . 5 \cdot 4 ) = 3 \log _ { 2 } 3$ ≈ 4.75. The right-hand side is $0 . 5 ( 2 \log _ { 2 } 2 ) + 0 . 5 ( 4 \log _ { 2 } 4 ) = 1 + 4 = 5$ . And therefore the definition is satisfied. 

Since $f ( x )$ is differentiable, we can alternatively use (7.31). Calculating the derivative of $f ( x )$ , we obtain 

$$
\nabla_ {x} (x \log_ {2} x) = 1 \cdot \log_ {2} x + x \cdot \frac {1}{x \log_ {e} 2} = \log_ {2} x + \frac {1}{\log_ {e} 2}.\tag{7.32}
$$

Using the same two test points $x = 2$ and $x = 4 .$ , the left-hand side of (7.31) is given by $f ( 4 ) = 8$ . The right-hand side is 

$$
f (\boldsymbol {x}) + \nabla_ {\boldsymbol {x}} ^ {\top} (\boldsymbol {y} - \boldsymbol {x}) = f (2) + \nabla f (2) \cdot (4 - 2)
$$

$$
= 2 + \left(1 + \frac {1}{\log_ {e} 2}\right) \cdot 2 \approx 6. 9.\tag{7.33a}
$$

(7.33b) 



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 




Figure 7.8 The negative entropy function (which is convex) and its tangent at $x = 2 .$


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/89d3e19547c6abe8e27e2bffd72439256c04500f3a247050749094903f89e074.jpg)


We can check that a function or set is convex from first principles by recalling the definitions. In practice, we often rely on operations that preserve convexity to check that a particular function or set is convex. Although the details are vastly different, this is again the idea of closure that we introduced in Chapter 2 for vector spaces. 

## Example 7.4

A nonnegative weighted sum of convex functions is convex. Observe that if $f$ is a convex function, and $\alpha \geqslant 0$ is a nonnegative scalar, then the function $\alpha f$ is convex. We can see this by multiplying α to both sides of the equation in Definition 7.3, and recalling that multiplying a nonnegative number does not change the inequality. 

If $f _ { 1 }$ and $f _ { 2 }$ are convex functions, then we have by the definition 

$$
f _ {1} (\theta \boldsymbol {x} + (1 - \theta) \boldsymbol {y}) \leqslant \theta f _ {1} (\boldsymbol {x}) + (1 - \theta) f _ {1} (\boldsymbol {y})\tag{7.34}
$$

$$
f _ {2} (\theta \boldsymbol {x} + (1 - \theta) \boldsymbol {y}) \leqslant \theta f _ {2} (\boldsymbol {x}) + (1 - \theta) f _ {2} (\boldsymbol {y}).\tag{7.35}
$$

Summing up both sides gives us 

$$
\begin{array}{l} f _ {1} (\theta \boldsymbol {x} + (1 - \theta) \boldsymbol {y}) + f _ {2} (\theta \boldsymbol {x} + (1 - \theta) \boldsymbol {y}) \\ \leqslant \theta f _ {1} (\boldsymbol {x}) + (1 - \theta) f _ {1} (\boldsymbol {y}) + \theta f _ {2} (\boldsymbol {x}) + (1 - \theta) f _ {2} (\boldsymbol {y}), \end{array}\tag{7.36}
$$

where the right-hand side can be rearranged to 

$$
\theta \left(f _ {1} (\boldsymbol {x}) + f _ {2} (\boldsymbol {x})\right) + (1 - \theta) \left(f _ {1} (\boldsymbol {y}) + f _ {2} (\boldsymbol {y})\right),\tag{7.37}
$$

completing the proof that the sum of convex functions is convex. 

Combining the preceding two facts, we see that $\alpha f _ { 1 } ( { \pmb x } ) + \beta f _ { 2 } ( { \pmb x } )$ is convex for $\alpha , \beta \geqslant 0$ This closure property can be extended using a similar argument for nonnegative weighted sums of more than two convex functions. 

Remark. The inequality in (7.30) is sometimes called Jensen’s inequality. In fact, a whole class of inequalities for taking nonnegative weighted sums of convex functions are all called Jensen’s inequality. 

In summary, a constrained optimization problem is called a convex optimization problem if 

$$
\begin{array}{c c c} \min _ {\boldsymbol {x}} f (\boldsymbol {x}) \\ \text {subject to} g _ {i} (\boldsymbol {x}) \leqslant 0 & \text {for all} & i = 1, \ldots , m \\ h _ {j} (\boldsymbol {x}) = 0 & \text {for all} & j = 1, \ldots , n  , \end{array}\tag{7.38}
$$

where all functions $f ( { \pmb x } )$ and $g _ { i } ( { \pmb x } )$ are convex functions, and all $h _ { j } ( { \pmb x } ) =$ 0 are convex sets. In the following, we will describe two classes of convex optimization problems that are widely used and well understood. 

## 7.3.1 Linear Programming

Consider the special case when all the preceding functions are linear, i.e., 

$$
\begin{array}{c c} \min _ {\boldsymbol {x} \in \mathbb {R} ^ {d}} & \boldsymbol {c} ^ {\top} \boldsymbol {x} \\ \text { subject   to } & \boldsymbol {A} \boldsymbol {x} \leqslant \boldsymbol {b}, \end{array}\tag{7.39}
$$

where $\pmb { A } \in \mathbb { R } ^ { m \times d }$ and $\pmb { b } \in \mathbb { R } ^ { m }$ . This is known as a linear program. It has d variables and m linear constraints. The Lagrangian is given by 

$$
\mathfrak {L} (\boldsymbol {x}, \boldsymbol {\lambda}) = \boldsymbol {c} ^ {\top} \boldsymbol {x} + \boldsymbol {\lambda} ^ {\top} (\boldsymbol {A} \boldsymbol {x} - \boldsymbol {b}),\tag{7.40}
$$

where $\pmb { \lambda } \in \mathbb { R } ^ { m }$ is the vector of non-negative Lagrange multipliers. Rearranging the terms corresponding to x yields 

$$
\mathfrak {L} (\boldsymbol {x}, \boldsymbol {\lambda}) = (\boldsymbol {c} + \boldsymbol {A} ^ {\top} \boldsymbol {\lambda}) ^ {\top} \boldsymbol {x} - \boldsymbol {\lambda} ^ {\top} \boldsymbol {b}.\tag{7.41}
$$

Taking the derivative of ${ \mathfrak { L } } ( { \pmb x } , { \pmb \lambda } )$ with respect to x and setting it to zero gives us 

$$
\boldsymbol {c} + \boldsymbol {A} ^ {\top} \boldsymbol {\lambda} = \boldsymbol {0}.\tag{7.42}
$$

Therefore, the dual Lagrangian is ${ \mathfrak { D } } ( \lambda ) = - \lambda ^ { \top } b .$ . Recall we would like to maximize $\mathfrak { D } ( \boldsymbol { \lambda } )$ . In addition to the constraint due to the derivative of $\mathfrak { L } ( \pmb { x } , \pmb { \lambda } )$ being zero, we also have the fact that $\lambda \geqslant \mathbf { 0 } _ { : }$ , resulting in the following dual optimization problem 

$$
\begin{array}{r l} \max _ {\boldsymbol {\lambda} \in \mathbb {R} ^ {m}} & - \boldsymbol {b} ^ {\top} \boldsymbol {\lambda} \\ \text {subject to} & \boldsymbol {c} + \boldsymbol {A} ^ {\top} \boldsymbol {\lambda} = \boldsymbol {0} \\ & \boldsymbol {\lambda} \geqslant \boldsymbol {0}. \end{array}\tag{7.43}
$$

This is also a linear program, but with m variables. We have the choice of solving the primal (7.39) or the dual (7.43) program depending on whether m or d is larger. Recall that d is the number of variables and m is the number of constraints in the primal linear program. 

## Example 7.5 (Linear Program)

Consider the linear program 

$$
\begin{array}{r l} \min _ {\boldsymbol {x} \in \mathbb {R} ^ {2}} & - \left[ \begin{array}{c} 5 \\ 3 \end{array} \right] ^ {\top} \left[ \begin{array}{c} x _ {1} \\ x _ {2} \end{array} \right] \\ \text {subject to} & \left[ \begin{array}{c c} 2 & 2 \\ 2 & - 4 \\ - 2 & 1 \\ 0 & - 1 \\ 0 & 1 \end{array} \right] \left[ \begin{array}{c} x _ {1} \\ x _ {2} \end{array} \right] \leqslant \left[ \begin{array}{c} 3 3 \\ 8 \\ 5 \\ - 1 \\ 8 \end{array} \right] \end{array}\tag{7.44}
$$

with two variables. This program is also shown in Figure 7.9. The objective function is linear, resulting in linear contour lines. The constraint set in standard form is translated into the legend. The optimal value must lie in the shaded (feasible) region, and is indicated by the star. 

Figure 7.9 Illustration of a linear program. The unconstrained problem (indicated by the contour lines) has a minimum on the right side. The optimal value given the constraints are shown by the star. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/34099efcb9f49454bff16bbcb68f276a8a948c0b8608a9f49c738fdaa11bdce1.jpg)



Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com.


## 7.3.2 Quadratic Programming

Consider the case of a convex quadratic objective function, where the constraints are affine, i.e., 

$$
\begin{array}{l l} \min _ {\boldsymbol {x} \in \mathbb {R} ^ {d}} & \frac {1}{2} \boldsymbol {x} ^ {\top} \boldsymbol {Q} \boldsymbol {x} + \boldsymbol {c} ^ {\top} \boldsymbol {x} \\ \text { subject   to } & \boldsymbol {A} \boldsymbol {x} \leqslant \boldsymbol {b}, \end{array}\tag{7.45}
$$

where $\pmb { A } \in \mathbb { R } ^ { m \times d } , \pmb { b } \in \mathbb { R } ^ { m }$ , and $\boldsymbol { c } \in \mathbb { R } ^ { d }$ . The square symmetric matrix $Q \in$ $\mathbb { R } ^ { d \times d }$ is positive definite, and therefore the objective function is convex. This is known as a quadratic program. Observe that it has d variables and m linear constraints. 

## Example 7.6 (Quadratic Program)

Consider the quadratic program 

$$
\min _ {\boldsymbol {x} \in \mathbb {R} ^ {2}} \quad \frac {1}{2} \left[ \begin{array}{c} x _ {1} \\ x _ {2} \end{array} \right] ^ {\top} \left[ \begin{array}{c c} 2 & 1 \\ 1 & 4 \end{array} \right] \left[ \begin{array}{c} x _ {1} \\ x _ {2} \end{array} \right] + \left[ \begin{array}{c} 5 \\ 3 \end{array} \right] ^ {\top} \left[ \begin{array}{c} x _ {1} \\ x _ {2} \end{array} \right]\tag{7.46}
$$

subject to 

$$
\text {o} \quad \left[ \begin{array}{c c} 1 & 0 \\ - 1 & 0 \\ 0 & 1 \\ 0 & - 1 \end{array} \right] \left[ \begin{array}{c} x _ {1} \\ x _ {2} \end{array} \right] \leqslant \left[ \begin{array}{c} 1 \\ 1 \\ 1 \\ 1 \end{array} \right]\tag{7.47}
$$

of two variables. The program is also illustrated in Figure 7.4. The objective function is quadratic with a positive semidefinite matrix $Q ,$ , resulting in elliptical contour lines. The optimal value must lie in the shaded (feasible) region, and is indicated by the star. 

The Lagrangian is given by 

$$
\mathfrak {L} (\boldsymbol {x}, \boldsymbol {\lambda}) = \frac {1}{2} \boldsymbol {x} ^ {\top} \boldsymbol {Q} \boldsymbol {x} + \boldsymbol {c} ^ {\top} \boldsymbol {x} + \boldsymbol {\lambda} ^ {\top} (\boldsymbol {A} \boldsymbol {x} - \boldsymbol {b})\tag{7.48a}
$$

(7.48b) 

where again we have rearranged the terms. Taking the derivative of $\mathfrak { L } ( \pmb { x } , \pmb { \lambda } )$ with respect to x and setting it to zero gives 

$$
\boldsymbol {Q} \boldsymbol {x} + (\boldsymbol {c} + \boldsymbol {A} ^ {\top} \boldsymbol {\lambda}) = \boldsymbol {0}.\tag{7.49}
$$

Since $Q$ is positive definite and therefore invertible, we get 

$$
\boldsymbol {x} = - \boldsymbol {Q} ^ {- 1} (\boldsymbol {c} + \boldsymbol {A} ^ {\top} \boldsymbol {\lambda}).\tag{7.50}
$$

Substituting (7.50) into the primal Lagrangian $\mathfrak { L } ( \pmb { x } , \pmb { \lambda } )$ , we get the dual Lagrangian 

$$
\mathfrak {D} (\boldsymbol {\lambda}) = - \frac {1}{2} (\boldsymbol {c} + \boldsymbol {A} ^ {\top} \boldsymbol {\lambda}) ^ {\top} \boldsymbol {Q} ^ {- 1} (\boldsymbol {c} + \boldsymbol {A} ^ {\top} \boldsymbol {\lambda}) - \boldsymbol {\lambda} ^ {\top} \boldsymbol {b}.\tag{7.51}
$$



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



Therefore, the dual optimization problem is given by 

$$
\begin{array}{l l} \max _ {\boldsymbol {\lambda} \in \mathbb {R} ^ {m}} & - \frac {1}{2} (\boldsymbol {c} + \boldsymbol {A} ^ {\top} \boldsymbol {\lambda}) ^ {\top} \boldsymbol {Q} ^ {- 1} (\boldsymbol {c} + \boldsymbol {A} ^ {\top} \boldsymbol {\lambda}) - \boldsymbol {\lambda} ^ {\top} \boldsymbol {b} \\ \text { subject   to } & \boldsymbol {\lambda} \geqslant \boldsymbol {0}. \end{array}\tag{7.52}
$$

We will see an application of quadratic programming in machine learning in Chapter 12. 

## 7.3.3 Legendre–Fenchel Transform and Convex Conjugate

Let us revisit the idea of duality from Section 7.2, without considering constraints. One useful fact about a convex set is that it can be equivalently described by its supporting hyperplanes. A hyperplane is called a supporting hyperplane of a convex set if it intersects the convex set, and the convex set is contained on just one side of it. Recall that we can fill up a convex function to obtain the epigraph, which is a convex set. Therefore, we can also describe convex functions in terms of their supporting hyperplanes. Furthermore, observe that the supporting hyperplane just touches the convex function, and is in fact the tangent to the function at that point. And recall that the tangent of a function $f ( { \pmb x } )$ at a given point $\scriptstyle { \pmb x } _ { 0 }$ is the evaluation of the gradient of that function at that point $\frac { \mathrm { d } f ( \pmb { x } ) } { \mathrm { d } \pmb { x } } \bigg | _ { \pmb { x } = \pmb { x } _ { 0 } }$ . In summary, because convex sets can be equivalently described by their supporting hyperplanes, convex functions can be equivalently described by a function of their gradient. The Legendre transform formalizes this concept. 

We begin with the most general definition, which unfortunately has a counter-intuitive form, and look at special cases to relate the definition to the intuition described in the preceding paragraph. The Legendre-Fenchel transform is a transformation (in the sense of a Fourier transform) from a convex differentiable function $f ( { \pmb x } )$ to a function that depends on the tangents $s ( { \pmb x } ) = \nabla _ { \pmb x } f ( { \pmb x } )$ ). It is worth stressing that this is a transformation of the function $f ( \cdot )$ and not the variable x or the function evaluated at x. The Legendre-Fenchel transform is also known as the convex conjugate (for reasons we will see soon) and is closely related to duality (Hiriart-Urruty and Lemar´echal, 2001, chapter 5). 

Definition 7.4. The convex conjugate of a function $f : \mathbb { R } ^ { D } \ \to$ <sup>R</sup> is a function $f ^ { * }$ defined by 

$$
f ^ {*} (\pmb {s}) = \sup _ {\pmb {x} \in \mathbb {R} ^ {D}} \left(\langle \pmb {s}, \pmb {x} \rangle - f (\pmb {x})\right).\tag{7.53}
$$

Note that the preceding convex conjugate definition does not need the function $f$ to be convex nor differentiable. In Definition $7 . 4 .$ , we have used a general inner product (Section 3.2) but in the rest of this section we will consider the standard dot product between finite-dimensional vectors $( \langle s , { \pmb x } \rangle = { \pmb s } ^ { \top } { \pmb x } )$ to avoid too many technical details. 

To understand Definition 7.4 in a geometric fashion, consider a nice simple one-dimensional convex and differentiable function, for example $f ( x ) = x ^ { 2 }$ . Note that since we are looking at a one-dimensional problem, hyperplanes reduce to a line. Consider a line $y = s x + c .$ . Recall that we are able to describe convex functions by their supporting hyperplanes, so let us try to describe this function $f ( x )$ by its supporting lines. Fix the gradient of the line $s \in \mathbb { R }$ and for each point $( x _ { 0 } , f ( x _ { 0 } ) )$ on the graph of $f ,$ find the minimum value of c such that the line still intersects $( x _ { 0 } , f ( x _ { 0 } ) )$ ). Note that the minimum value of c is the place where a line with slope s “just touches” the function $f ( x ) \ = \ x ^ { 2 }$ . The line passing through $( x _ { 0 } , f ( x _ { 0 } ) )$ with gradient s is given by 

$$
y - f (x _ {0}) = s (x - x _ {0}).\tag{7.54}
$$

The $y \cdot$ -intercept of this line is $- s x _ { 0 } + f ( x _ { 0 } )$ . The minimum of $c$ for which $y = s x + c$ intersects with the graph of $f$ is therefore 

$$
\inf _ {x _ {0}} - s x _ {0} + f (x _ {0}).\tag{7.55}
$$

The preceding convex conjugate is by convention defined to be the negative of this. The reasoning in this paragraph did not rely on the fact that we chose a one-dimensional convex and differentiable function, and holds for $f : \mathbb { R } ^ { D }  \mathbb { R }$ , which are nonconvex and non-differentiable. 

Remark. Convex differentiable functions such as the example $f ( x ) = x ^ { 2 }$ is a nice special case, where there is no need for the supremum, and there is a one-to-one correspondence between a function and its Legendre transform. Let us derive this from first principles. For a convex differentiable function, we know that at $x _ { 0 }$ the tangent touches $f ( x _ { 0 } )$ so that 

$$
f (x _ {0}) = s x _ {0} + c.\tag{7.56}
$$

Recall that we want to describe the convex function $f ( x )$ in terms of its gradient $\nabla _ { x } f ( x )$ , and that $s = \nabla _ { x } f ( x _ { 0 } )$ . We rearrange to get an expression for c to obtain 

$$
- c = s x _ {0} - f (x _ {0}).\tag{7.57}
$$

Note that $- c$ changes with $x _ { 0 }$ and therefore with $s ,$ which is why we can think of it as a function of $s ,$ which we call 

$$
f ^ {*} (s) := s x _ {0} - f \left(x _ {0}\right).\tag{7.58}
$$

Comparing (7.58) with Definition $7 . 4 ,$ we see that (7.58) is a special case (without the supremum). 

The conjugate function has nice properties; for example, for convex functions, applying the Legendre transform again gets us back to the original function. In the same way that the slope of $f ( x )$ is $s ,$ the slope of $f ^ { * } ( s )$ 

is x. The following two examples show common uses of convex conjugates in machine learning. 

## Example 7.7 (Convex Conjugates)

To illustrate the application of convex conjugates, consider the quadratic function 

$$
f (\boldsymbol {y}) = \frac {\lambda}{2} \boldsymbol {y} ^ {\top} \boldsymbol {K} ^ {- 1} \boldsymbol {y}\tag{7.59}
$$

based on a positive definite matrix $\pmb { K } \in \mathbb { R } ^ { n \times n }$ . We denote the primal variable to be $\pmb { y } \in \mathbb { R } ^ { n }$ and the dual variable to be $\mathbf { \Delta } \alpha \in \mathbb { R } ^ { n }$ 

Applying Definition 7.4, we obtain the function 

$$
f ^ {*} (\boldsymbol {\alpha}) = \sup _ {\boldsymbol {y} \in \mathbb {R} ^ {n}} \left\langle \boldsymbol {y}, \boldsymbol {\alpha} \right\rangle - \frac {\lambda}{2} \boldsymbol {y} ^ {\top} \boldsymbol {K} ^ {- 1} \boldsymbol {y}.\tag{7.60}
$$

Since the function is differentiable, we can find the maximum by taking the derivative and with respect to y setting it to zero. 

$$
\frac {\partial [ \langle \boldsymbol {y} , \boldsymbol {\alpha} \rangle - \frac {\lambda}{2} \boldsymbol {y} ^ {\top} \boldsymbol {K} ^ {- 1} \boldsymbol {y} ]}{\partial \boldsymbol {y}} = (\boldsymbol {\alpha} - \lambda \boldsymbol {K} ^ {- 1} \boldsymbol {y}) ^ {\top}\tag{7.61}
$$

and hence when the gradient is zero we have $\begin{array} { r } { { \bf y } = { \frac { 1 } { \lambda } } { \cal K } { \boldsymbol \alpha } } \end{array}$ . Substituting into (7.60) yields 

$$
f ^ {*} (\boldsymbol {\alpha}) = \frac {1}{\lambda} \boldsymbol {\alpha} ^ {\top} \boldsymbol {K} \boldsymbol {\alpha} - \frac {\lambda}{2} \left(\frac {1}{\lambda} \boldsymbol {K} \boldsymbol {\alpha}\right) ^ {\top} \boldsymbol {K} ^ {- 1} \left(\frac {1}{\lambda} \boldsymbol {K} \boldsymbol {\alpha}\right) = \frac {1}{2 \lambda} \boldsymbol {\alpha} ^ {\top} \boldsymbol {K} \boldsymbol {\alpha}.\tag{7.62}
$$

## Example 7.8

In machine learning, we often use sums of functions; for example, the objective function of the training set includes a sum of the losses for each example in the training set. In the following, we derive the convex conjugate of a sum of losses $\ell ( t )$ , where $\ell : \mathbb { R } $ <sup>R</sup>. This also illustrates the application of the convex conjugate to the vector case. Let $\begin{array} { r } { \mathcal { L } ( t ) = \sum _ { i = 1 } ^ { n } \ell _ { i } ( t _ { i } ) } \end{array}$ Then, 

$$
\mathcal {L} ^ {*} (\boldsymbol {z}) = \sup _ {\boldsymbol {t} \in \mathbb {R} ^ {n}} \left\langle \boldsymbol {z}, \boldsymbol {t} \right\rangle - \sum_ {i = 1} ^ {n} \ell_ {i} (t _ {i})\tag{7.63a}
$$

$$
= \sup _ {\boldsymbol {t} \in \mathbb {R} ^ {n}} \sum_ {i = 1} ^ {n} z _ {i} t _ {i} - \ell_ {i} (t _ {i})
$$

definition of dot product 

(7.63b) 

$$
= \sum_ {i = 1} ^ {n} \sup _ {\boldsymbol {t} \in \mathbb {R} ^ {n}} z _ {i} t _ {i} - \ell_ {i} (t _ {i})\tag{7.63c}
$$



Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 



$$
= \sum_ {i = 1} ^ {n} \ell_ {i} ^ {*} (z _ {i}).
$$

definition of conjugate 

(7.63d) 

Recall that in Section 7.2 we derived a dual optimization problem using Lagrange multipliers. Furthermore, for convex optimization problems we have strong duality, that is the solutions of the primal and dual problem match. The Legendre-Fenchel transform described here also can be used to derive a dual optimization problem. Furthermore, when the function is convex and differentiable, the supremum is unique. To further investigate the relation between these two approaches, let us consider a linear equality constrained convex optimization problem. 

## Example $7 . 9$

Let $f ( \boldsymbol { y } )$ and $g ( { \pmb x } )$ be convex functions, and A a real matrix of appropriate dimensions such that $\pmb { A } \pmb { x } = \pmb { y }$ . Then 

$$
\min _ {\boldsymbol {x}} f (\boldsymbol {A} \boldsymbol {x}) + g (\boldsymbol {x}) = \min _ {\boldsymbol {A} \boldsymbol {x} = \boldsymbol {y}} f (\boldsymbol {y}) + g (\boldsymbol {x}).\tag{7.64}
$$

By introducing the Lagrange multiplier u for the constraints $\pmb { A } \pmb { x } = \pmb { y }$ 

$$
\begin{array}{r l} \min _ {\boldsymbol {A x} = \boldsymbol {y}} f (\boldsymbol {y}) + g (\boldsymbol {x}) & = \min _ {\boldsymbol {x}, \boldsymbol {y}} \max _ {\boldsymbol {u}} f (\boldsymbol {y}) + g (\boldsymbol {x}) + (\boldsymbol {A x} - \boldsymbol {y}) ^ {\top} \boldsymbol {u} \\ & = \max _ {\boldsymbol {u}} \min _ {\boldsymbol {x}, \boldsymbol {y}} f (\boldsymbol {y}) + g (\boldsymbol {x}) + (\boldsymbol {A x} - \boldsymbol {y}) ^ {\top} \boldsymbol {u}, \end{array}\tag{7.65a}
$$

(7.65b) 

where the last step of swapping max and min is due to the fact that $f ( y )$ and $g ( { \pmb x } )$ are convex functions. By splitting up the dot product term and collecting x and ${ \mathbf { } } ^ { \prime } { \mathbf { } }$ 

$$
\max _ {\boldsymbol {u}} \min _ {\boldsymbol {x}, \boldsymbol {y}} f (\boldsymbol {y}) + g (\boldsymbol {x}) + (\boldsymbol {A} \boldsymbol {x} - \boldsymbol {y}) ^ {\top} \boldsymbol {u}\tag{7.66a}
$$

$$
= \max _ {\boldsymbol {u}} \left[ \min _ {\boldsymbol {y}} - \boldsymbol {y} ^ {\top} \boldsymbol {u} + f (\boldsymbol {y}) \right] + \left[ \min _ {\boldsymbol {x}} (\boldsymbol {A x}) ^ {\top} \boldsymbol {u} + g (\boldsymbol {x}) \right]\tag{7.66b}
$$

$$
= \max _ {\boldsymbol {u}} \left\lfloor \min _ {\boldsymbol {y}} - \boldsymbol {y} ^ {\top} \boldsymbol {u} + f (\boldsymbol {y}) \right\rfloor + \left[ \min _ {\boldsymbol {x}} \boldsymbol {x} ^ {\top} \boldsymbol {A} ^ {\top} \boldsymbol {u} + g (\boldsymbol {x}) \right]\tag{7.66c}
$$

Recall the convex conjugate (Definition 7.4) and the fact that dot products are symmetric, 

$$
\max _ {\boldsymbol {u}} \left[ \min _ {\boldsymbol {y}} - \boldsymbol {y} ^ {\top} \boldsymbol {u} + f (\boldsymbol {y}) \right] + \left[ \min _ {\boldsymbol {x}} \boldsymbol {x} ^ {\top} \boldsymbol {A} ^ {\top} \boldsymbol {u} + g (\boldsymbol {x}) \right]\tag{7.67a}
$$

$$
= \max _ {\boldsymbol {u}} - f ^ {*} (\boldsymbol {u}) - g ^ {*} (- \boldsymbol {A} ^ {\top} \boldsymbol {u}).\tag{7.67b}
$$

Therefore, we have shown that 

$$
\min _ {\boldsymbol {x}} f (\boldsymbol {A} \boldsymbol {x}) + g (\boldsymbol {x}) = \max _ {\boldsymbol {u}} - f ^ {*} (\boldsymbol {u}) - g ^ {*} (- \boldsymbol {A} ^ {\top} \boldsymbol {u}).\tag{7.68}
$$



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



The Legendre-Fenchel conjugate turns out to be quite useful for machine learning problems that can be expressed as convex optimization problems. In particular, for convex loss functions that apply independently to each example, the conjugate loss is a convenient way to derive a dual problem. 

## 7.4 Further Reading

Continuous optimization is an active area of research, and we do not try to provide a comprehensive account of recent advances. 

From a gradient descent perspective, there are two major weaknesses which each have their own set of literature. The first challenge is the fact that gradient descent is a first-order algorithm, and does not use information about the curvature of the surface. When there are long valleys, the gradient points perpendicularly to the direction of interest. The idea of momentum can be generalized to a general class of acceleration methods (Nesterov, 2018). Conjugate gradient methods avoid the issues faced by gradient descent by taking previous directions into account (Shewchuk, 1994). Second-order methods such as Newton methods use the Hessian to provide information about the curvature. Many of the choices for choosing step-sizes and ideas like momentum arise by considering the curvature of the objective function (Goh, 2017; Bottou et al., 2018). Quasi-Newton methods such as L-BFGS try to use cheaper computational methods to approximate the Hessian (Nocedal and Wright, 2006). Recently there has been interest in other metrics for computing descent directions, resulting in approaches such as mirror descent (Beck and Teboulle, 2003) and natural gradient (Toussaint, 2012). 

The second challenge is to handle non-differentiable functions. Gradient methods are not well defined when there are kinks in the function. In these cases, subgradient methods can be used (Shor, 1985). For further information and algorithms for optimizing non-differentiable functions, we refer to the book by Bertsekas (1999). There is a vast amount of literature on different approaches for numerically solving continuous optimization problems, including algorithms for constrained optimization problems. Good starting points to appreciate this literature are the books by Luenberger (1969) and Bonnans et al. (2006). A recent survey of continuous optimization is provided by Bubeck (2015). 

Modern applications of machine learning often mean that the size of datasets prohibit the use of batch gradient descent, and hence stochastic gradient descent is the current workhorse of large-scale machine learning methods. Recent surveys of the literature include Hazan (2015) and Bottou et al. (2018). 

For duality and convex optimization, the book by Boyd and Vandenberghe (2004) includes lectures and slides online. A more mathematical treatment is provided by Bertsekas (2009), and recent book by one of the key researchers in the area of optimization is Nesterov (2018). Convex optimization is based upon convex analysis, and the reader interested in more foundational results about convex functions is referred to Rockafellar (1970), Hiriart-Urruty and Lemar´echal (2001), and Borwein and Lewis (2006). Legendre–Fenchel transforms are also covered in the aforementioned books on convex analysis, but a more beginner-friendly presentation is available at Zia et al. (2009). The role of Legendre–Fenchel transforms in the analysis of convex optimization algorithms is surveyed in Polyak (2016). 

## Exercises

## 7.1 Consider the univariate function

$$
f (x) = x ^ {3} + 6 x ^ {2} - 3 x - 5.
$$

Find its stationary points and indicate whether they are maximum, minimum, or saddle points. 

7.2 Consider the update equation for stochastic gradient descent (Equation (7.15)). 

Write down the update when we use a mini-batch size of one. 

7.3 Consider whether the following statements are true or false: 

a. The intersection of any two convex sets is convex. 

b. The union of any two convex sets is convex. 

c. The difference of a convex set A from another convex set B is convex. 

7.4 Consider whether the following statements are true or false: 

a. The sum of any two convex functions is convex. 

b. The difference of any two convex functions is convex. 

c. The product of any two convex functions is convex. 

d. The maximum of any two convex functions is convex. 

7.5 Express the following optimization problem as a standard linear program in matrix notation 

$$
\max _ {\pmb {x} \in \mathbb {R} ^ {2}, \xi \in \mathbb {R}} \pmb {p} ^ {\top} \pmb {x} + \xi
$$

subject to the constraints that $\xi \geqslant 0 , x _ { 0 } \leqslant 0$ and $x _ { 1 } \leqslant 3 .$ 

7.6 Consider the linear program illustrated in Figure 7.9, 

$$
\min _ {\boldsymbol {x} \in \mathbb {R} ^ {2}} - \left[ \begin{array}{c} 5 \\ 3 \end{array} \right] ^ {\top} \left[ \begin{array}{c} x _ {1} \\ x _ {2} \end{array} \right]
$$

$$
\text { subject   to } \left[ \begin{array}{c c} 2 & 2 \\ 2 & - 4 \\ - 2 & 1 \\ 0 & - 1 \\ 0 & 1 \end{array} \right] \left[ \begin{array}{c} x _ {1} \\ x _ {2} \end{array} \right] \leqslant \left[ \begin{array}{c} 3 3 \\ 8 \\ 5 \\ - 1 \\ 8 \end{array} \right]
$$

Derive the dual linear program using Lagrange duality. 

©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 

7.7 Consider the quadratic program illustrated in Figure $7 . 4 _ { ; }$ 

$$
\min _ {\boldsymbol {x} \in \mathbb {R} ^ {2}} \frac {1}{2} \left[ \begin{array}{c} x _ {1} \\ x _ {2} \end{array} \right] ^ {\top} \left[ \begin{array}{c c} 2 & 1 \\ 1 & 4 \end{array} \right] \left[ \begin{array}{c} x _ {1} \\ x _ {2} \end{array} \right] + \left[ \begin{array}{c} 5 \\ 3 \end{array} \right] ^ {\top} \left[ \begin{array}{c} x _ {1} \\ x _ {2} \end{array} \right]
$$

$$
\text { subject   to } \left[ \begin{array}{c c} 1 & 0 \\ - 1 & 0 \\ 0 & 1 \\ 0 & - 1 \end{array} \right] \left[ \begin{array}{c} x _ {1} \\ x _ {2} \end{array} \right] \leqslant \left[ \begin{array}{c} 1 \\ 1 \\ 1 \\ 1 \end{array} \right]
$$

Derive the dual quadratic program using Lagrange duality. 

7.8 Consider the following convex optimization problem 

$$
\min _ {\boldsymbol {w} \in \mathbb {R} ^ {D}} \frac {1}{2} \boldsymbol {w} ^ {\top} \boldsymbol {w}
$$

$$
\text { subject   to } \quad \boldsymbol {w} ^ {\top} \boldsymbol {x} \geqslant 1.
$$

Derive the Lagrangian dual by introducing the Lagrange multiplier λ. 

Consider the negative entropy of $\pmb { x } \in \mathbb { R } ^ { D }$ 

$$
f (\boldsymbol {x}) = \sum_ {d = 1} ^ {D} x _ {d} \log x _ {d}.
$$

Derive the convex conjugate function $f ^ { * } ( s )$ , by assuming the standard dot product. 

Hint: Take the gradient of an appropriate function and set the gradient to zero. 7.10 Consider the function 

$$
f (\pmb {x}) = \frac {1}{2} \pmb {x} ^ {\top} \pmb {A} \pmb {x} + \pmb {b} ^ {\top} \pmb {x} + c,
$$

where A is strictly positive definite, which means that it is invertible. Derive the convex conjugate of $f ( { \pmb x } )$ 

Hint: Take the gradient of an appropriate function and set the gradient to zero. 

7.11 The hinge loss (which is the loss used by the support vector machine) is given by 

$$
L (\alpha) = \max \{0, 1 - \alpha \},
$$

If we are interested in applying gradient methods such as L-BFGS, and do not want to resort to subgradient methods, we need to smooth the kink in the hinge loss. Compute the convex conjugate of the hinge loss $L ^ { * } ( \beta )$ where $\beta$ is the dual variable. Add a $\ell _ { 2 }$ proximal term, and compute the conjugate of the resulting function 

$$
L ^ {*} (\beta) + \frac {\gamma}{2} \beta^ {2},
$$

where $\gamma$ is a given hyperparameter. 

Central Machine Learning Problems 

# When Models Meet Data

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/de9489af0b64352f77cbfdff0bb7140a6ba152c69c6c0b1beb8dcaf7e63a088b.jpg)


In the first part of the book, we introduced the mathematics that form the foundations of many machine learning methods. The hope is that a reader would be able to learn the rudimentary forms of the language of mathematics from the first part, which we will now use to describe and discuss machine learning. The second part of the book introduces four pillars of machine learning: 

Regression (Chapter 9) 

Dimensionality reduction (Chapter 10) 

Density estimation (Chapter 11) 

Classification (Chapter 12) 

The main aim of this part of the book is to illustrate how the mathematical concepts introduced in the first part of the book can be used to design machine learning algorithms that can be used to solve tasks within the remit of the four pillars. We do not intend to introduce advanced machine learning concepts, but instead to provide a set of practical methods that allow the reader to apply the knowledge they gained from the first part of the book. It also provides a gateway to the wider machine learning literature for readers already familiar with the mathematics. 

## 8.1 Data, Models, and Learning

It is worth at this point, to pause and consider the problem that a machine learning algorithm is designed to solve. As discussed in Chapter 1, there are three major components of a machine learning system: data, models, and learning. The main question of machine learning is “What do we mean by good models?”. The word model has many subtleties, and we will revisit it multiple times in this chapter. It is also not entirely obvious how to objectively define the word “good”. One of the guiding principles of machine learning is that good models should perform well on unseen data. This requires us to define some performance metrics, such as accuracy or distance from ground truth, as well as figuring out ways to do well under these performance metrics. This chapter covers a few necessary bits and pieces of mathematical and statistical language that are commonly 

Table 8.1 Example data from a fictitious human resource database that is not in a numerical format. 

<table><tr><td>Name</td><td>Gender</td><td>Degree</td><td>Postcode</td><td>Age</td><td>Annual salary</td></tr><tr><td>Aditya</td><td>M</td><td>MSc</td><td>W21BG</td><td>36</td><td>89563</td></tr><tr><td>Bob</td><td>M</td><td>PhD</td><td>EC1A1BA</td><td>47</td><td>123543</td></tr><tr><td>Chloé</td><td>F</td><td>BEcon</td><td>SW1A1BH</td><td>26</td><td>23989</td></tr><tr><td>Daisuke</td><td>M</td><td>BSc</td><td>SE207AT</td><td>68</td><td>138769</td></tr><tr><td>Elisabeth</td><td>F</td><td>MBA</td><td>SE10AA</td><td>33</td><td>113888</td></tr></table>

used to talk about machine learning models. By doing so, we briefly outline the current best practices for training a model such that the resulting predictor does well on data that we have not yet seen. 

As mentioned in Chapter 1, there are two different senses in which we use the phrase “machine learning algorithm”: training and prediction. We will describe these ideas in this chapter, as well as the idea of selecting among different models. We will introduce the framework of empirical risk minimization in Section 8.2, the principle of maximum likelihood in Section 8.3, and the idea of probabilistic models in Section 8.4. We briefly outline a graphical language for specifying probabilistic models in Section 8.5 and finally discuss model selection in Section 8.6. The rest of this section expands upon the three main components of machine learning: data, models and learning. 

## 8.1.1 Data as Vectors

We assume that our data can be read by a computer, and represented adequately in a numerical format. Data is assumed to be tabular (Figure 8.1), where we think of each row of the table as representing a particular instance or example, and each column to be a particular feature. In recent years, machine learning has been applied to many types of data that do not obviously come in the tabular numerical format, for example genomic sequences, text and image contents of a webpage, and social media graphs. We do not discuss the important and challenging aspects of identifying good features. Many of these aspects depend on domain expertise and require careful engineering, and, in recent years, they have been put under the umbrella of data science (Stray, 2016; Adhikari and DeNero, 2018). 

Even when we have data in tabular format, there are still choices to be made to obtain a numerical representation. For example, in Table 8.1, the gender column (a categorical variable) may be converted into numbers 0 representing “Male” and 1 representing “Female”. Alternatively, the gender could be represented by numbers 1, +1, respectively (as shown in Table 8.2). Furthermore, it is often important to use domain knowledge when constructing the representation, such as knowing that university degrees progress from bachelor’s to master’s to PhD or realizing that the postcode provided is not just a string of characters but actually encodes an area in London. In Table 8.2, we converted the data from Table 8.1 to a numerical format, and each postcode is represented as two numbers, a latitude and longitude. Even numerical data that could potentially be directly read into a machine learning algorithm should be carefully considered for units, scaling, and constraints. Without additional information, one should shift and scale all columns of the dataset such that they have an empirical mean of 0 and an empirical variance of 1. For the purposes of this book, we assume that a domain expert already converted data appropriately, i.e., each input ${ \bf { x } } _ { n }$ is a D-dimensional vector of real numbers, which are called features, attributes, or covariates. We consider a dataset to be of the form as illustrated by Table 8.2. Observe that we have dropped the Name column of Table 8.1 in the new numerical representation. There are two main reasons why this is desirable: (1) we do not expect the identifier (the Name) to be informative for a machine learning task; and (2) we may wish to anonymize the data to help protect the privacy of the employees. 


8.1 Data, Models, and Learning


<table><tr><td>Gender ID</td><td>Degree</td><td>Latitude (in degrees)</td><td>Longitude (in degrees)</td><td>Age</td><td>Annual Salary (in thousands)</td></tr><tr><td>-1</td><td>2</td><td>51.5073</td><td>0.1290</td><td>36</td><td>89.563</td></tr><tr><td>-1</td><td>3</td><td>51.5074</td><td>0.1275</td><td>47</td><td>123.543</td></tr><tr><td>+1</td><td>1</td><td>51.5071</td><td>0.1278</td><td>26</td><td>23.989</td></tr><tr><td>-1</td><td>1</td><td>51.5075</td><td>0.1281</td><td>68</td><td>138.769</td></tr><tr><td>+1</td><td>2</td><td>51.5074</td><td>0.1278</td><td>33</td><td>113.888</td></tr></table>

In this part of the book, we will use N to denote the number of examples in a dataset and index the examples with lowercase $n = 1 , \ldots , N$ We assume that we are given a set of numerical data, represented as an array of vectors (Table 8.2). Each row is a particular individual ${ \bf x } _ { n } ,$ often referred to as an example or data point in machine learning. The subscript n refers to the fact that this is the nth example out of a total of N examples in the dataset. Each column represents a particular feature of interest about the example, and we index the features as $d = 1 , \ldots , D$ . Recall that data is represented as vectors, which means that each example (each data point) is a D-dimensional vector. The orientation of the table originates from the database community, but for some machine learning algorithms (e.g., in Chapter 10) it is more convenient to represent examples as column vectors. 

Let us consider the problem of predicting annual salary from age, based on the data in Table 8.2. This is called a supervised learning problem where we have a label $y _ { n }$ (the salary) associated with each example ${ \bf { x } } _ { n }$ (the age). The label $y _ { n }$ has various other names, including target, response variable, and annotation. A dataset is written as a set of examplelabel pairs $\left\{ ( \pmb { x } _ { 1 } , y _ { 1 } ) , \dotsc , ( \pmb { x } _ { n } , y _ { n } ) , \dotsc , ( \pmb { x } _ { N } , y _ { N } ) \right\}$ . The table of examples $\{ \pmb { x } _ { 1 } , \hdots , \pmb { x } _ { N } \}$ is often concatenated, and written as $\pmb { X } \in \mathbb { R } ^ { N \times D }$ . Figure 8.1 illustrates the dataset consisting of the two rightmost columns of Table 8.2, where x = age and y = salary. 

We use the concepts introduced in the first part of the book to formalize 

Figure 8.1 Toy data for linear regression. Training data in $( x _ { n } , y _ { n } )$ pairs from the rightmost two columns of Table 8.2. We are interested in the salary of a person aged sixty $( x = 6 0 )$ illustrated as a vertical dashed red line, which is not part of the training data. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/b9bf831cf30e0094ae998e98ab7294aa76886719f85c164953dddc1d85ba7c61.jpg)


the machine learning problems such as that in the previous paragraph. Representing data as vectors ${ \bf { x } } _ { n }$ allows us to use concepts from linear algebra (introduced in Chapter 2). In many machine learning algorithms, we need to additionally be able to compare two vectors. As we will see in Chapters 9 and 12, computing the similarity or distance between two examples allows us to formalize the intuition that examples with similar features should have similar labels. The comparison of two vectors requires that we construct a geometry (explained in Chapter 3) and allows us to optimize the resulting learning problem using techniques from Chapter 7. 

Since we have vector representations of data, we can manipulate data to find potentially better representations of it. We will discuss finding good representations in two ways: finding lower-dimensional approximations of the original feature vector, and using nonlinear higher-dimensional combinations of the original feature vector. In Chapter 10, we will see an example of finding a low-dimensional approximation of the original data space by finding the principal components. Finding principal components is closely related to concepts of eigenvalue and singular value decomposition as introduced in Chapter 4. For the high-dimensional representation, we will see an explicit feature map $\phi ( \cdot )$ that allows us to represent inputs ${ \bf { x } } _ { n }$ using a higher-dimensional representation $\phi ( { \pmb x } _ { n } )$ . The main motivation for higher-dimensional representations is that we can construct new features as non-linear combinations of the original features, which in turn may make the learning problem easier. We will discuss the feature map in Section 9.2 and show how this feature map leads to a kernel in Section 12.4. In recent years, deep learning methods (Goodfellow et $\mathrm { { a l . } }$ 2016) have shown promise in using the data itself to learn new good features and have been very successful in areas, such as computer vision, speech recognition, and natural language processing. We will not cover neural networks in this part of the book, but the reader is referred to 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/920b93e039dcaa2042a178405169da410a02bb8052b8dc31418e1fba070f6323.jpg)



Figure 8.2 Example function (black solid diagonal line) and its prediction at x = 60, i.e., f(60) = 100.



Section 5.6 for the mathematical description of backpropagation, a key concept for training neural networks.


## 8.1.2 Models as Functions

Once we have data in an appropriate vector representation, we can get to the business of constructing a predictive function (known as a predictor). In Chapter 1, we did not yet have the language to be precise about models. Using the concepts from the first part of the book, we can now introduce what “model” means. We present two major approaches in this book: a predictor as a function, and a predictor as a probabilistic model. We describe the former here and the latter in the next subsection. 

A predictor is a function that, when given a particular input example (in our case, a vector of features), produces an output. For now, consider the output to be a single number, i.e., a real-valued scalar output. This can be written as 

$$
f: \mathbb {R} ^ {D} \to \mathbb {R},\tag{8.1}
$$

where the input vector x is D-dimensional (has D features), and the function f then applied to it (written as f (x)) returns a real number. Figure 8.2 illustrates a possible function that can be used to compute the value of the prediction for input values x. 

In this book, we do not consider the general case of all functions, which would involve the need for functional analysis. Instead, we consider the special case of linear functions 

$$
f (\boldsymbol {x}) = \boldsymbol {\theta} ^ {\top} \boldsymbol {x} + \theta_ {0}\tag{8.2}
$$

for unknown θ and $\theta _ { 0 }$ . This restriction means that the contents of Chapters 2 and 3 suffice for precisely stating the notion of a predictor for the non-probabilistic (in contrast to the probabilistic view described next) 


Figure 8.3 Example function (black solid diagonal line) and its predictive uncertainty at x = 60 (drawn as a Gaussian).


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/1501cc066b03728f12da918624541d9feae1cac472f189faacf5411298e1532d.jpg)



view of machine learning. Linear functions strike a good balance between the generality of the problems that can be solved and the amount of background mathematics that is needed.


## 8.1.3 Models as Probability Distributions

We often consider data to be noisy observations of some true underlying effect, and hope that by applying machine learning we can identify the signal from the noise. This requires us to have a language for quantifying the effect of noise. We often would also like to have predictors that express some sort of uncertainty, e.g., to quantify the confidence we have about the value of the prediction for a particular test data point. As we have seen in Chapter 6, probability theory provides a language for quantifying uncertainty. Figure 8.3 illustrates the predictive uncertainty of the function as a Gaussian distribution. 

Instead of considering a predictor as a single function, we could consider predictors to be probabilistic models, i.e., models describing the distribution of possible functions. We limit ourselves in this book to the special case of distributions with finite-dimensional parameters, which allows us to describe probabilistic models without needing stochastic processes and random measures. For this special case, we can think about probabilistic models as multivariate probability distributions, which already allow for a rich class of models. 

We will introduce how to use concepts from probability (Chapter 6) to define machine learning models in Section 8.4, and introduce a graphical language for describing probabilistic models in a compact way in Section 8.5. 

## 8.1.4 Learning is Finding Parameters

The goal of learning is to find a model and its corresponding parameters such that the resulting predictor will perform well on unseen data. There are conceptually three distinct algorithmic phases when discussing machine learning algorithms: 

## 1. Prediction or inference

2. Training or parameter estimation 

3. Hyperparameter tuning or model selection 

The prediction phase is when we use a trained predictor on previously unseen test data. In other words, the parameters and model choice is already fixed and the predictor is applied to new vectors representing new input data points. As outlined in Chapter 1 and the previous subsection, we will consider two schools of machine learning in this book, corresponding to whether the predictor is a function or a probabilistic model. When we have a probabilistic model (discussed further in Section 8.4) the predic tion phase is called inference. 

Remark. Unfortunately, there is no agreed upon naming for the different algorithmic phases. The word “inference” is sometimes also used to mean parameter estimation of a probabilistic model, and less often may be also used to mean prediction for non-probabilistic models. 

The training or parameter estimation phase is when we adjust our predictive model based on training data. We would like to find good predictors given training data, and there are two main strategies for doing so: finding the best predictor based on some measure of quality (sometimes called finding a point estimate), or using Bayesian inference. Finding a point estimate can be applied to both types of predictors, but Bayesian inference requires probabilistic models. 

For the non-probabilistic model, we follow the principle of empirical risk minimization, which we describe in Section 8.2. Empirical risk minimization directly provides an optimization problem for finding good parameters. With a statistical model, the principle of maximum likelihood is used to find a good set of parameters (Section 8.3). We can additionally model the uncertainty of parameters using a probabilistic model, which we will look at in more detail in Section 8.4. 

We use numerical methods to find good parameters that “fit” the data, and most training methods can be thought of as hill-climbing approaches to find the maximum of an objective, for example the maximum of a likelihood. To apply hill-climbing approaches we use the gradients described in Chapter 5 and implement numerical optimization approaches from Chapter 7. 

As mentioned in Chapter 1, we are interested in learning a model based on data such that it performs well on future data. It is not enough for the model to only fit the training data well, the predictor needs to perform well on unseen data. We simulate the behavior of our predictor on future unseen data using cross-validation (Section 8.2.4). As we will see in this chapter, to achieve the goal of performing well on unseen data, we will need to balance between fitting well on training data and finding “simple” explanations of the phenomenon. This trade-off is achieved using regularization (Section 8.2.3) or by adding a prior (Section 8.3.2). In philosophy, this is considered to be neither induction nor deduction, but is called abduction. According to the Stanford Encyclopedia of Philosophy, abduction is the process of inference to the best explanation (Douven, 2017). 

We often need to make high-level modeling decisions about the structure of the predictor, such as the number of components to use or the class of probability distributions to consider. The choice of the number of components is an example of a hyperparameter, and this choice can affect the performance of the model significantly. The problem of choosing among different models is called model selection, which we describe in Section 8.6. For non-probabilistic models, model selection is often done using nested cross-validation, which is described in Section 8.6.1. We also use model selection to choose hyperparameters of our model. 

Remark. The distinction between parameters and hyperparameters is somewhat arbitrary, and is mostly driven by the distinction between what can be numerically optimized versus what needs to use search techniques. Another way to consider the distinction is to consider parameters as the explicit parameters of a probabilistic model, and to consider hyperparameters (higher-level parameters) as parameters that control the distribution of these explicit parameters. ◇ 

In the following sections, we will look at three flavors of machine learning: empirical risk minimization (Section 8.2), the principle of maximum likelihood (Section 8.3), and probabilistic modeling (Section 8.4). 

## 8.2 Empirical Risk Minimization

After having all the mathematics under our belt, we are now in a position to introduce what it means to learn. The “learning” part of machine learning boils down to estimating parameters based on training data. 

In this section, we consider the case of a predictor that is a function, and consider the case of probabilistic models in Section 8.3. We describe the idea of empirical risk minimization, which was originally popularized by the proposal of the support vector machine (described in Chapter 12). However, its general principles are widely applicable and allow us to ask the question of what is learning without explicitly constructing probabilistic models. There are four main design choices, which we will cover in detail in the following subsections: 

Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 

Section 8.2.1 What is the set of functions we allow the predictor to take? Section 8.2.2 How do we measure how well the predictor performs on the training data? 

Section 8.2.3 How do we construct predictors from only training data that performs well on unseen test data? 

Section 8.2.4 What is the procedure for searching over the space of models? 

## 8.2.1 Hypothesis Class of Functions

Assume we are given N examples ${ \pmb x } _ { n } \in \mathbb { R } ^ { D }$ and corresponding scalar labels $y _ { n } \in \mathbb { R }$ . We consider the supervised learning setting, where we obtain pairs $( { \pmb x } _ { 1 } , y _ { 1 } ) , \dots , ( { \pmb x } _ { N } , y _ { N } )$ . Given this data, we would like to estimate a predictor $f ( \cdot , { \pmb \theta } ) : \mathbb { R } ^ { D }  \mathbb { R } ,$ , parametrized by θ. We hope to be able to find a good parameter $\pmb { \theta } ^ { * }$ such that we fit the data well, that is, 

$$
f (\boldsymbol {x} _ {n}, \boldsymbol {\theta} ^ {*}) \approx y _ {n} \quad \text { for   all } \quad n = 1, \dots , N.\tag{8.3}
$$

In this section, we use the notation $\hat { y } _ { n } = f ( \pmb { x } _ { n } , \pmb { \theta } ^ { * } )$ to represent the output of the predictor. 

Remark. For ease of presentation, we will describe empirical risk minimization in terms of supervised learning (where we have labels). This simplifies the definition of the hypothesis class and the loss function. It is also common in machine learning to choose a parametrized class of functions, for example affine functions. 

## Example 8.1

We introduce the problem of ordinary least-squares regression to illustrate empirical risk minimization. A more comprehensive account of regression is given in Chapter 9. When the label $y _ { n }$ is real-valued, a popular choice of function class for predictors is the set of affine functions. We choose a more compact notation for an affine function by concatenating an additional unit feature $x ^ { ( 0 ) } = 1$ to ${ \bf { x } } _ { n } .$ , i.e., $\pmb { x } _ { n } = [ 1 , x _ { n } ^ { ( 1 ) } , x _ { n } ^ { ( 2 ) } , \bot \bot , x _ { n } ^ { ( D ) } ] ^ { \top }$ . The parameter vector is correspondingly $\pmb { \theta } = [ \theta _ { 0 } , \theta _ { 1 } , \theta _ { 2 } , \dots , \theta _ { D } ] ^ { \top }$ , allowing us to write the predictor as a linear function 

$$
f (\pmb {x} _ {n}, \pmb {\theta}) = \pmb {\theta} ^ {\top} \pmb {x} _ {n}.\tag{8.4}
$$

This linear predictor is equivalent to the affine model 

$$
f (\boldsymbol {x} _ {n}, \boldsymbol {\theta}) = \theta_ {0} + \sum_ {d = 1} ^ {D} \theta_ {d} x _ {n} ^ {(d)}.\tag{8.5}
$$

The predictor takes the vector of features representing a single example ${ \bf { x } } _ { n }$ as input and produces a real-valued output, i.e., $f : \mathbb { R } ^ { D + 1 } $ <sup>R</sup>. The 



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



previous figures in this chapter had a straight line as a predictor, which means that we have assumed an affine function. 

Instead of a linear function, we may wish to consider non-linear functions as predictors. Recent advances in neural networks allow for efficient computation of more complex non-linear function classes. 

Given the class of functions, we want to search for a good predictor. We now move on to the second ingredient of empirical risk minimization: how to measure how well the predictor fits the training data. 

## 8.2.2 Loss Function for Training

Consider the label $y _ { n }$ for a particular example; and the corresponding prediction ${ \hat { y } } _ { n }$ that we make based on ${ \bf { x } } _ { n }$ . To define what it means to fit the data well, we need to specify a loss function $\ell ( y _ { n } , \hat { y } _ { n } )$ that takes the ground truth label and the prediction as input and produces a non-negative number (referred to as the loss) representing how much error we have made on this particular prediction. Our goal for finding a good parameter vector $\pmb { \theta } ^ { * }$ is to minimize the average loss on the set of N training examples. 

One assumption that is commonly made in machine learning is that the set of examples $( { \pmb x } _ { 1 } , y _ { 1 } ) , \dots , ( { \pmb x } _ { N } , y _ { N } )$ is independent and identically distributed. The word independent (Section 6.4.5) means that two data points $( { \pmb x } _ { i } , y _ { i } )$ and $( \pmb { x } _ { j } , y _ { j } )$ do not statistically depend on each other, meaning that the empirical mean is a good estimate of the population mean (Section 6.4.1). This implies that we can use the empirical mean of the loss on the training data. For a given training set $\left\{ ( \pmb { x } _ { 1 } , y _ { 1 } ) , \dotsc , ( \pmb { x } _ { N } , y _ { N } ) \right\}$ we introduce the notation of an example matrix $X : = [ \pmb { x } _ { 1 } , \dots , \pmb { x } _ { N } ] ^ { \top } \in$ $\mathbb { R } ^ { N \times D }$ and a label vector $\pmb { y } : = [ y _ { 1 } , \dotsc , y _ { N } ] ^ { \top } \in \mathbb { R } ^ { N }$ . Using this matrix notation the average loss is given by 

$$
\mathbf {R} _ {\mathrm{emp}} (f, \pmb {X}, \pmb {y}) = \frac {1}{N} \sum_ {n = 1} ^ {N} \ell (y _ {n}, \hat {y} _ {n}),\tag{8.6}
$$

where $\hat { y } _ { n } = f ( \pmb { x } _ { n } , \pmb { \theta } )$ . Equation (8.6) is called the empirical risk and depends on three arguments, the predictor $f$ and the data X, y. This general strategy for learning is called empirical risk minimization. 

## Example 8.2 (Least-Squares Loss)

Continuing the example of least-squares regression, we specify that we measure the cost of making an error during training using the squared loss $\ell ( y _ { n } , \hat { y } _ { n } ) = ( y _ { n } - \hat { y } _ { n } ) ^ { 2 }$ . We wish to minimize the empirical risk (8.6), which is the average of the losses over the data 

$$
\min _ {\boldsymbol {\theta} \in \mathbb {R} ^ {D}} \frac {1}{N} \sum_ {n = 1} ^ {N} (y _ {n} - f (\boldsymbol {x} _ {n}, \boldsymbol {\theta})) ^ {2},\tag{8.7}
$$

where we substituted the predictor $\hat { y } _ { n } = f ( \pmb { x } _ { n } , \pmb { \theta } )$ . By using our choice of a linear predictor $\begin{array} { r } { f ( \pmb { x } _ { n } , \pmb { \theta } ) = \pmb { \theta } ^ { \top } \pmb { x } _ { n } } \end{array}$ , we obtain the optimization problem 

$$
\min _ {\boldsymbol {\theta} \in \mathbb {R} ^ {D}} \frac {1}{N} \sum_ {n = 1} ^ {N} (y _ {n} - \boldsymbol {\theta} ^ {\top} \boldsymbol {x} _ {n}) ^ {2}.\tag{8.8}
$$

This equation can be equivalently expressed in matrix form 

$$
\min _ {\boldsymbol {\theta} \in \mathbb {R} ^ {D}} \frac {1}{N} \left\| \boldsymbol {y} - \boldsymbol {X} \boldsymbol {\theta} \right\| ^ {2}.\tag{8.9}
$$

This is known as the least-squares problem. There exists a closed-form analytic solution for this by solving the normal equations, which we will discuss in Section 9.2. 

We are not interested in a predictor that only performs well on the training data. Instead, we seek a predictor that performs well (has low risk) on unseen test data. More formally, we are interested in finding a predictor f (with parameters fixed) that minimizes the expected risk 

$$
\mathbf {R} _ {\mathrm{true}} (f) = \mathbb {E} _ {\pmb {x}, y} [ \ell (y, f (\pmb {x})) ],\tag{8.10}
$$

where $y$ is the label and $f ( { \pmb x } )$ is the prediction based on the example x. The notation $\mathbf { R } _ { \mathrm { t r u e } } ( f )$ indicates that this is the true risk if we had access to an infinite amount of data. The expectation is over the (infinite) set of all possible data and labels. There are two practical questions that arise from our desire to minimize expected risk, which we address in the following two subsections: 

How should we change our training procedure to generalize well? 

How do we estimate expected risk from (finite) data? 

Remark. Many machine learning tasks are specified with an associated performance measure, e.g., accuracy of prediction or root mean squared error. The performance measure could be more complex, be cost sensitive, and capture details about the particular application. In principle, the design of the loss function for empirical risk minimization should correspond directly to the performance measure specified by the machine learning task. In practice, there is often a mismatch between the design of the loss function and the performance measure. This could be due to issues such as ease of implementation or efficiency of optimization. 

©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 

## 8.2.3 Regularization to Reduce Overfitting

This section describes an addition to empirical risk minimization that allows it to generalize well (approximately minimizing expected risk). Recall that the aim of training a machine learning predictor is so that we can perform well on unseen data, i.e., the predictor generalizes well. We simulate this unseen data by holding out a proportion of the whole dataset. This hold out set is referred to as the test set. Given a sufficiently rich class of functions for the predictor $f ,$ , we can essentially memorize the training data to obtain zero empirical risk. While this is great to minimize the loss (and therefore the risk) on the training data, we would not expect the predictor to generalize well to unseen data. In practice, we have only a finite set of data, and hence we split our data into a training and a test set. The training set is used to fit the model, and the test set (not seen by the machine learning algorithm during training) is used to evaluate generalization performance. It is important for the user to not cycle back to a new round of training after having observed the test set. We use the subscripts $\operatorname { t r a i n }$ and $\scriptstyle { \mathrm { t e s t } }$ to denote the training and test sets, respectively. We will revisit this idea of using a finite dataset to evaluate expected risk in Section 8.2.4. 

It turns out that empirical risk minimization can lead to overfitting, i.e., the predictor fits too closely to the training data and does not generalize well to new data (Mitchell, 1997). This general phenomenon of having very small average loss on the training set but large average loss on the test set tends to occur when we have little data and a complex hypothesis class. For a particular predictor $f$ (with parameters fixed), the phenomenon of overfitting occurs when the risk estimate from the training data $\mathbf { R } _ { \mathrm { e m p } } ( f , X _ { \mathrm { t r a i n } } , \pmb { y } _ { \mathrm { t r a i n } } )$ underestimates the expected risk $\mathbf { R } _ { \mathrm { t r u e } } ( f )$ Since we estimate the expected risk $\mathbf { R } _ { \mathrm { t r u e } } ( f )$ by using the empirical risk on the test set $\mathbf { R } _ { \mathrm { e m p } } ( f , X _ { \mathrm { t e s t } } , y _ { \mathrm { t e s t } } )$ if the test risk is much larger than the training risk, this is an indication of overfitting. We revisit the idea of overfitting in Section 8.3.3. 

Therefore, we need to somehow bias the search for the minimizer of empirical risk by introducing a penalty term, which makes it harder for the optimizer to return an overly flexible predictor. In machine learning, the penalty term is referred to as regularization. Regularization is a way to compromise between accurate solution of empirical risk minimization and the size or complexity of the solution. 

## Example 8.3 (Regularized Least Squares)

Regularization is an approach that discourages complex or extreme solutions to an optimization problem. The simplest regularization strategy is 

to replace the least-squares problem 

$$
\min _ {\boldsymbol {\theta}} \frac {1}{N} \left\| \boldsymbol {y} - \boldsymbol {X} \boldsymbol {\theta} \right\| ^ {2}.\tag{8.11}
$$

in the previous example with the “regularized” problem by adding a penalty term involving only θ: 

$$
\min _ {\boldsymbol {\theta}} \frac {1}{N} \left\| \boldsymbol {y} - \boldsymbol {X} \boldsymbol {\theta} \right\| ^ {2} + \lambda \left\| \boldsymbol {\theta} \right\| ^ {2}.\tag{8.12}
$$

The additional term $\left\| \pmb \theta \right\| ^ { 2 }$ is called the regularizer, and the parameter $\lambda$ is the regularization parameter. The regularization parameter trades off minimizing the loss on the training set and the magnitude of the parameters $\pmb \theta .$ It often happens that the magnitude of the parameter values becomes relatively large if we run into overfitting (Bishop, 2006). 

The regularization term is sometimes called the penalty term, which biases the vector $\pmb \theta$ to be closer to the origin. The idea of regularization also appears in probabilistic models as the prior probability of the parameters. Recall from Section $6 . 6$ that for the posterior distribution to be of the same form as the prior distribution, the prior and the likelihood need to be conjugate. We will revisit this idea in Section 8.3.2. We will see in Chapter 12 that the idea of the regularizer is equivalent to the idea of a large margin. 

## 8.2.4 Cross-Validation to Assess the Generalization Performance

We mentioned in the previous section that we measure the generalization error by estimating it by applying the predictor on test data. This data is also sometimes referred to as the validation set. The validation set is a subset of the available training data that we keep aside. A practical issue with this approach is that the amount of data is limited, and ideally we would use as much of the data available to train the model. This would require us to keep our validation set $\nu$ small, which then would lead to a noisy estimate (with high variance) of the predictive performance. One solu tion to these contradictory objectives (large training set, large validation set) is to use cross-validation. K-fold cross-validation effectively partitions the data into $K$ chunks, $K - 1$ of which form the training set ${ \mathcal { R } } ,$ and the last chunk serves as the validation set $\nu$ (similar to the idea outlined previously). Cross-validation iterates through (ideally) all combinations of assignments of chunks to $\mathcal { R }$ and $\nu ;$ see Figure 8.4. This procedure is repeated for all $K$ choices for the validation set, and the performance of the model from the K runs is averaged. 

We partition our dataset into two sets $\mathcal { D } = \mathcal { R } \cup \mathcal { V } _ { : }$ , such that they do not overlap $( \mathcal { R } \cap \mathcal { V } = \varnothing )$ , where $\nu$ is the validation set, and train our model on $\mathcal { R }$ . After training, we assess the performance of the predictor $f$ on the 

Figure 8.4 K-fold cross-validation. The dataset is divided into $K = 5$ chunks, K − 1 of which serve as the training set (blue) and one as the validation set (orange hatch). 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/42ccafd187716a140fb84e7ae34865380b1bdf4c4a0c72213119e26360444b8f.jpg)


validation set $( \boldsymbol { \mathrm { e . g . } }$ , by computing root mean square error (RMSE) of the trained model on the validation set). More precisely, for each partition k the training data $\mathcal { R } ^ { ( k ) }$ produces a predictor ${ \bar { \boldsymbol { f } } } ^ { ( k ) }$ , which is then applied to validation set $\mathcal { V } ^ { ( k ) }$ to compute the empirical risk $R ( f ^ { ( k ) } , \mathcal { V } ^ { ( k ) } )$ . We cycle through all possible partitionings of validation and training sets and compute the average generalization error of the predictor. Cross-validation approximates the expected generalization error 

$$
\mathbb {E} _ {\mathcal {V}} [ R (f, \mathcal {V}) ] \approx \frac {1}{K} \sum_ {k = 1} ^ {K} R (f ^ {(k)}, \mathcal {V} ^ {(k)}),\tag{8.13}
$$

where $R ( f ^ { ( k ) } , \mathcal { V } ^ { ( k ) } )$ is the risk (e.g., RMSE) on the validation set $\mathcal { V } ^ { ( k ) }$ for predictor $f ^ { ( k ) }$ . The approximation has two sources: first, due to the finite training set, which results in not the best possible $f ^ { ( k ) }$ ; and second, due to the finite validation set, which results in an inaccurate estimation of the risk $R ( f ^ { ( k ) } , \mathcal { V } ^ { ( k ) } )$ ). A potential disadvantage of K-fold cross-validation is the computational cost of training the model K times, which can be burdensome if the training cost is computationally expensive. In practice, it is often not sufficient to look at the direct parameters alone. For example, we need to explore multiple complexity parameters (e.g., multiple regularization parameters), which may not be direct parameters of the model. Evaluating the quality of the model, depending on these hyperparameters, may result in a number of training runs that is exponential in the number of model parameters. One can use nested cross-validation (Section 8.6.1) to search for good hyperparameters. 

However, cross-validation is an embarrassingly parallel problem, i.e., little effort is needed to separate the problem into a number of parallel tasks. Given sufficient computing resources (e.g., cloud computing, server farms), cross-validation does not require longer than a single performance assessment. 

In this section, we saw that empirical risk minimization is based on the following concepts: the hypothesis class of functions, the loss function and regularization. In Section 8.3, we will see the effect of using a probability distribution to replace the idea of loss functions and regularization. 

## 8.2.5 Further Reading

Due to the fact that the original development of empirical risk minimization (Vapnik, 1998) was couched in heavily theoretical language, many of the subsequent developments have been theoretical. The area of study is called statistical learning theory (Vapnik, 1999; Evgeniou et al., 2000; Hastie et al., 2001; von Luxburg and Scholkopf¨ , 2011). A recent machine learning textbook that builds on the theoretical foundations and develops efficient learning algorithms is Shalev-Shwartz and Ben-David (2014). 

The concept of regularization has its roots in the solution of ill-posed inverse problems (Neumaier, 1998). The approach presented here is called Tikhonov regularization, and there is a closely related constrained version called Ivanov regularization. Tikhonov regularization has deep relationships to the bias-variance trade-off and feature selection (Buhlmann and¨ Van De Geer, 2011). An alternative to cross-validation is bootstrap and jackknife (Efron and Tibshirani, 1993; Davidson and Hinkley, 1997; Hall, 1992). 

Thinking about empirical risk minimization (Section 8.2) as “probability $\mathrm { f r e e } ^ { \prime \prime }$ is incorrect. There is an underlying unknown probability distribution $p ( { \pmb x } , { \pmb y } )$ that governs the data generation. However, the approach of empirical risk minimization is agnostic to that choice of distribution. This is in contrast to standard statistical approaches that explicitly require the knowledge of $p ( { \pmb x } , { \pmb y } )$ . Furthermore, since the distribution is a joint distribution on both examples x and labels $y ,$ the labels can be nondeterministic. In contrast to standard statistics we do not need to specify the noise distribution for the labels $y .$ 

## 8.3 Parameter Estimation

In Section 8.2, we did not explicitly model our problem using probability distributions. In this section, we will see how to use probability distributions to model our uncertainty due to the observation process and our uncertainty in the parameters of our predictors. In Section 8.3.1, we introduce the likelihood, which is analogous to the concept of loss functions (Section 8.2.2) in empirical risk minimization. The concept of priors (Section 8.3.2) is analogous to the concept of regularization (Section 8.2.3). 

## 8.3.1 Maximum Likelihood Estimation

The idea behind maximum likelihood estimation (MLE) is to define a function of the parameters that enables us to find a model that fits the data well. The estimation problem is focused on the likelihood function, or more precisely its negative logarithm. For data represented by a random variable x and for a family of probability densities $p ( { \pmb x } | { \pmb \theta } )$ parametrized by $\theta ,$ the negative log-likelihood is given by 

©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 

$$
\mathcal {L} _ {\boldsymbol {x}} (\boldsymbol {\theta}) = - \log p (\boldsymbol {x} \mid \boldsymbol {\theta}).\tag{8.14}
$$

The notation ${ \mathcal { L } } _ { x } ( \theta )$ emphasizes the fact that the parameter $\pmb \theta$ is varying and the data x is fixed. We very often drop the reference to x when writing the negative log-likelihood, as it is really a function of $\theta ,$ , and write it as $\mathcal { L } ( \pmb \theta )$ when the random variable representing the uncertainty in the data is clear from the context. 

Let us interpret what the probability density $p ( { \pmb x } | { \pmb \theta } )$ is modeling for a fixed value of $\pmb \theta .$ . It is a distribution that models the uncertainty of the data for a given parameter setting. For a given dataset $\mathbf { \delta } _ { \mathbf { x } , \mathbf { \delta } _ { \mathbf { \delta } _ { \mathrm { i } } } }$ , the likelihood allows us to express preferences about different settings of the parameters $\theta ,$ and we can choose the setting that more “likely” has generated the data. 

In a complementary view, if we consider the data to be fixed (because it has been observed), and we vary the parameters $\theta ,$ what does $\mathcal { L } ( \pmb \theta )$ tell us? It tells us how likely a particular setting of $\pmb \theta$ is for the observations ${ \pmb x } .$ Based on this second view, the maximum likelihood estimator gives us the most likely parameter $\pmb \theta$ for the set of data. 

We consider the supervised learning setting, where we obtain pairs $( { \pmb x } _ { 1 } , y _ { 1 } ) , \dots , ( { \pmb x } _ { N } , y _ { N } )$ with ${ \pmb x } _ { n } \in \mathbb { R } ^ { D }$ and labels $y _ { n } \in \mathbb { R }$ . We are interested in constructing a predictor that takes a feature vector ${ \bf { x } } _ { n }$ as input and produces a prediction $y _ { n }$ (or something close to it), i.e., given a vector ${ \bf { x } } _ { n }$ we want the probability distribution of the label $y _ { n } .$ . In other words, we specify the conditional probability distribution of the labels given the examples for the particular parameter setting θ. 

## Example 8.4

The first example that is often used is to specify that the conditional probability of the labels given the examples is a Gaussian distribution. In other words, we assume that we can explain our observation uncertainty by independent Gaussian noise (refer to Section 6.5) with zero mean, $\varepsilon _ { n } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ . We further assume that the linear model $\pmb { x } _ { n } ^ { \top } \pmb { \theta }$ is used for prediction. This means we specify a Gaussian likelihood for each example label pair $( { \pmb x } _ { n } , y _ { n } )$ , 

$$
p (y _ {n} \mid \boldsymbol {x} _ {n}, \boldsymbol {\theta}) = \mathcal {N} \big (y _ {n} \mid \boldsymbol {x} _ {n} ^ {\top} \boldsymbol {\theta}, \sigma^ {2} \big).\tag{8.15}
$$

An illustration of a Gaussian likelihood for a given parameter $\pmb \theta$ is shown in Figure 8.3. We will see in Section 9.2 how to explicitly expand the preceding expression out in terms of the Gaussian distribution. 

We assume that the set of examples $( x _ { 1 } , y _ { 1 } ) , \dotsc , ( x _ { N } , y _ { N } )$ are independent and identically distributed (i.i.d.). The word “independent” (Section 6.4.5) implies that the likelihood involving the whole dataset $( \mathcal { V } = \{ y _ { 1 } , \dots , y _ { N } \}$ and $\mathcal { X } = \{ \pmb { x } _ { 1 } , \hdots , \pmb { x } _ { N } \} )$ factorizes into a product of the likelihoods of 

## 8.3 Parameter Estimation

each individual example 

$$
p (\mathcal {Y} \mid \mathcal {X}, \boldsymbol {\theta}) = \prod_ {n = 1} ^ {N} p (y _ {n} \mid \boldsymbol {x} _ {n}, \boldsymbol {\theta}),\tag{8.16}
$$

where $p ( y _ { n } \mid \pmb { x } _ { n } , \pmb { \theta } )$ is a particular distribution (which was Gaussian in Example 8.4). The expression “identically distributed” means that each term in the product (8.16) is of the same distribution, and all of them share the same parameters. It is often easier from an optimization viewpoint to compute functions that can be decomposed into sums of simpler functions. Hence, in machine learning we often consider the negative log-likelihood 

Recall log(ab) = log(a) + log(b) 

$$
\mathcal {L} (\boldsymbol {\theta}) = - \log p (\mathcal {Y} | \mathcal {X}, \boldsymbol {\theta}) = - \sum_ {n = 1} ^ {N} \log p (y _ {n} | \boldsymbol {x} _ {n}, \boldsymbol {\theta}).\tag{8.17}
$$

While it is temping to interpret the fact that $\pmb \theta$ is on the right of the conditioning in $p ( y _ { n } | \pmb { x } _ { n } , \pmb { \theta } )$ (8.15), and hence should be interpreted as observed and fixed, this interpretation is incorrect. The negative log-likelihood ${ \mathcal { L } } ( \theta )$ is a function of $\pmb { \theta } .$ . Therefore, to find a good parameter vector θ that explains the data $( { \pmb x } _ { 1 } , y _ { 1 } ) , \dots , ( { \pmb x } _ { N } , y _ { N } )$ well, minimize the negative loglikelihood $\mathcal { L } ( \pmb { \theta } )$ with respect to θ. 

Remark. The negative sign in (8.17) is a historical artifact that is due to the convention that we want to maximize likelihood, but numerical optimization literature tends to study minimization of functions. 

## Example 8.5

Continuing on our example of Gaussian likelihoods (8.15), the negative log-likelihood can be rewritten as 

$$
\mathcal {L} (\boldsymbol {\theta}) = - \sum_ {n = 1} ^ {N} \log p (y _ {n} \mid \boldsymbol {x} _ {n}, \boldsymbol {\theta}) = - \sum_ {n = 1} ^ {N} \log \mathcal {N} \bigl (y _ {n} \mid \boldsymbol {x} _ {n} ^ {\top} \boldsymbol {\theta}, \sigma^ {2} \bigr)\tag{8.18a}
$$

$$
= - \sum_ {n = 1} ^ {N} \log {\frac {1}{\sqrt {2 \pi \sigma^ {2}}}} \exp \left(- \frac {(y _ {n} - \pmb {x} _ {n} ^ {\top} \pmb {\theta}) ^ {2}}{2 \sigma^ {2}}\right)\tag{8.18b}
$$

$$
= - \sum_ {n = 1} ^ {N} \log \exp \left(- \frac {(y _ {n} - \pmb {x} _ {n} ^ {\top} \pmb {\theta}) ^ {2}}{2 \sigma^ {2}}\right) - \sum_ {n = 1} ^ {N} \log \frac {1}{\sqrt {2 \pi \sigma^ {2}}}\tag{8.18c}
$$

$$
= \frac {1}{2 \sigma^ {2}} \sum_ {n = 1} ^ {N} (y _ {n} - \boldsymbol {x} _ {n} ^ {\top} \boldsymbol {\theta}) ^ {2} - \sum_ {n = 1} ^ {N} \log \frac {1}{\sqrt {2 \pi \sigma^ {2}}}.\tag{8.18d}
$$

As σ is given, the second term in (8.18d) is constant, and minimizing $\mathcal { L } ( \pmb \theta )$ corresponds to solving the least-squares problem (compare with (8.8)) expressed in the first term. 

It turns out that for Gaussian likelihoods the resulting optimization 

Figure 8.5 For the given data, the maximum likelihood estimate of the parameters results in the black diagonal line. The orange square shows the value of the maximum likelihood prediction at x = 60. 

Figure 8.6 Comparing the predictions with the maximum likelihood estimate and the MAP estimate at x = 60. The prior biases the slope to be less steep and the intercept to be closer to zero. In this example, the bias that moves the intercept closer to zero actually increases the slope. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/c3108433bdcc858831807d2c41ec778cd0dac8cb9016ba1263119343bd9e0927.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/b821dcdce566daf1e3132b10cdd924b00a371a693a1c67a3a56a40b35e62d8fc.jpg)



problem corresponding to maximum likelihood estimation has a closedform solution. We will see more details on this in Chapter 9. Figure 8.5 shows a regression dataset and the function that is induced by the maximum-likelihood parameters. Maximum likelihood estimation may suffer from overfitting (Section 8.3.3), analogous to unregularized empirical risk minimization (Section 8.2.3). For other likelihood functions, i.e., if we model our noise with non-Gaussian distributions, maximum likelihood estimation may not have a closed-form analytic solution. In this case, we resort to numerical optimization methods discussed in Chapter 7.


## 8.3.2 Maximum A Posteriori Estimation

If we have prior knowledge about the distribution of the parameters $\theta ,$ we can multiply an additional term to the likelihood. This additional term is a prior probability distribution on parameters $p ( \pmb \theta )$ . For a given prior, after 

## 8.3 Parameter Estimation

observing some data x, how should we update the distribution of θ? In other words, how should we represent the fact that we have more specific knowledge of θ after observing data x? Bayes’ theorem, as discussed in Section $6 . 3 ,$ gives us a principled tool to update our probability distributions of random variables. It allows us to compute a posterior distribution $p ( \pmb \theta | \pmb x )$ (the more specific knowledge) on the parameters θ from general prior statements (prior distribution) $p ( \pmb \theta )$ and the function $p ( { \pmb x } | { \pmb \theta } )$ that links the parameters θ and the observed data x (called the likelihood): 

$$
p (\boldsymbol {\theta} \mid \boldsymbol {x}) = \frac {p (\boldsymbol {x} \mid \boldsymbol {\theta}) p (\boldsymbol {\theta})}{p (\boldsymbol {x})}.\tag{8.19}
$$

Recall that we are interested in finding the parameter θ that maximizes the posterior. Since the distribution $p ( { \pmb x } )$ does not depend on $\theta ,$ we can ignore the value of the denominator for the optimization and obtain 

$$
p (\boldsymbol {\theta} \mid \boldsymbol {x}) \propto p (\boldsymbol {x} \mid \boldsymbol {\theta}) p (\boldsymbol {\theta}).\tag{8.20}
$$

The preceding proportion relation hides the density of the data $p ( { \pmb x } )$ which may be difficult to estimate. Instead of estimating the minimum of the negative log-likelihood, we now estimate the minimum of the neg ative log-posterior, which is referred to as maximum a posteriori estimation (MAP estimation). An illustration of the effect of adding a zero-mean Gaussian prior is shown in Figure 8.6. 

## Example 8.6

In addition to the assumption of Gaussian likelihood in the previous example, we assume that the parameter vector is distributed as a multivariate Gaussian with zero mean, i.e., $p ( \pmb \theta ) = \mathcal { N } ( \mathbf 0 , \pmb \Sigma )$ , where Σ is the covariance matrix (Section 6.5). Note that the conjugate prior of a Gaussian is also a Gaussian (Section 6.6.1), and therefore we expect the posterior distribution to also be a Gaussian. We will see the details of maximum a posteriori estimation in Chapter 9. 

The idea of including prior knowledge about where good parameters lie is widespread in machine learning. An alternative view, which we saw in Section 8.2.3, is the idea of regularization, which introduces an additional term that biases the resulting parameters to be close to the origin. Maximum a posteriori estimation can be considered to bridge the nonprobabilistic and probabilistic worlds as it explicitly acknowledges the need for a prior distribution but it still only produces a point estimate of the parameters. 

Remark. The maximum likelihood estimate $\pmb { \theta } _ { \mathrm { M L } }$ possesses the following properties (Lehmann and Casella, 1998; Efron and Hastie, 2016): 

Asymptotic consistency: The MLE converges to the true value in the limit of infinitely many observations, plus a random error that is approximately normal. 


Figure 8.7 Model fitting. In a parametrized class $M _ { \theta }$ of models, we optimize the model parameters θ to minimize the distance to the true (unknown) model M<sup>∗</sup>.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/fa01834ca3b063bfa4f4ffa54912547d7a299b13a426c3c28822afc9d57ae9d8.jpg)


The size of the samples necessary to achieve these properties can be quite large. 

The error’s variance decays in $1 / N$ , where N is the number of data points. 

Especially, in the “small” data regime, maximum likelihood estimation can lead to overfitting. 

♢ 

The principle of maximum likelihood estimation (and maximum a posteriori estimation) uses probabilistic modeling to reason about the uncertainty in the data and model parameters. However, we have not yet taken probabilistic modeling to its full extent. In this section, the resulting training procedure still produces a point estimate of the predictor, i.e., training returns one single set of parameter values that represent the best predictor. In Section 8.4, we will take the view that the parameter values should also be treated as random variables, and instead of estimating “best” values of that distribution, we will use the full parameter distribution when making predictions. 

## 8.3.3 Model Fitting

Consider the setting where we are given a dataset, and we are interested in fitting a parametrized model to the data. When we talk about “fitting”, we typically mean optimizing/learning model parameters so that they minimize some loss function, e.g., the negative log-likelihood. With maximum likelihood (Section 8.3.1) and maximum a posteriori estimation (Section 8.3.2), we already discussed two commonly used algorithms for model fitting. 

The parametrization of the model defines a model class $M _ { \theta }$ with which we can operate. For example, in a linear regression setting, we may define the relationship between inputs x and (noise-free) observations $y$ to be $y = a x + b _ { \mathrm { { ; } } }$ , where $\theta : = \{ a , b \}$ are the model parameters. In this case, the model parameters θ describe the family of affine functions, i.e., straight lines with slope $^ { a , }$ which are offset from 0 by b. Assume the data comes 

## 8.3 Parameter Estimation

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/612fd1bae42cdba3cfff10a1861138286fbd73bff36c7a527b74135279e4dfcc.jpg)



(a) Overfitting


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/106359522cfbd0ad02ea7dcad8a565b2c5d15e303f48101902cdb55d157c32a3.jpg)



(b) Underfitting.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/f1db29b0-0362-47b9-950f-0c625142c1b2/62792c05197d26de35c2c21b8ad56f3104f253f5571490c71f1b9ac585c6dde1.jpg)



(c) Fitting well.



Figure 8.8 Fitting (by maximum likelihood) of different model classes to a regression dataset.


from a model $M ^ { * }$ , which is unknown to us. For a given training dataset, we optimize θ so that $M _ { \theta }$ is as close as possible to $M ^ { * }$ , where the “closeness” is defined by the objective function we optimize $( \boldsymbol { \mathrm { e . g . } } ,$ squared loss on the training data). Figure 8.7 illustrates a setting where we have a small model class (indicated by the circle $M _ { \theta } )$ , and the data generation model $M ^ { * }$ lies outside the set of considered models. We begin our parameter search at $M _ { \theta _ { 0 } }$ . After the optimization, i.e., when we obtain the best pos sible parameters $\pmb { \theta } ^ { * }$ , we distinguish three different cases: (i) overfitting, (ii) underfitting, and (iii) fitting well. We will give a high-level intuition of what these three concepts mean. 

Roughly speaking, overfitting refers to the situation where the parametrized model class is too rich to model the dataset generated by $M ^ { * }$ i.e., $M _ { \theta }$ could model much more complicated datasets. For instance, if the dataset was generated by a linear function, and we define $M _ { \theta }$ to be the class of seventh-order polynomials, we could model not only linear func tions, but also polynomials of degree two, three, etc. Models that overfit typically have a large number of parameters. An observation we often make is that the overly flexible model class $M _ { \theta }$ uses all its modeling power to reduce the training error. If the training data is noisy, it will therefore find some useful signal in the noise itself. This will cause enormous problems when we predict away from the training data. Figure 8.8(a) gives an example of overfitting in the context of regression where the model parameters are learned by means of maximum likelihood (see Section 8.3.1). We will discuss overfitting in regression more in Section 9.2.2. 

When we run into underfitting, we encounter the opposite problem where the model class $M _ { \theta }$ is not rich enough. For example, if our dataset was generated by a sinusoidal function, but θ only parametrizes straight lines, the best optimization procedure will not get us close to the true model. However, we still optimize the parameters and find the best straight line that models the dataset. Figure 8.8(b) shows an example of a model that underfits because it is insufficiently flexible. Models that underfit typically have few parameters. 

underfitting 

The third case is when the parametrized model class is about right. Then, our model fits well, i.e., it neither overfits nor underfits. This means our model class is just rich enough to describe the dataset we are given. Figure 8.8(c) shows a model that fits the given dataset fairly well. Ideally, this is the model class we would want to work with since it has good generalization properties. 

In practice, we often define very rich model classes $M _ { \theta }$ with many parameters, such as deep neural networks. To mitigate the problem of overfitting, we can use regularization (Section 8.2.3) or priors (Section 8.3.2). We will discuss how to choose the model class in Section 8.6. 

## 8.3.4 Further Reading

When considering probabilistic models, the principle of maximum likelihood estimation generalizes the idea of least-squares regression for linear models, which we will discuss in detail in Chapter 9. When restricting the predictor to have linear form with an additional nonlinear function $\varphi$ applied to the output, i.e., 

$$
p (y _ {n} | \boldsymbol {x} _ {n}, \boldsymbol {\theta}) = \varphi (\boldsymbol {\theta} ^ {\top} \boldsymbol {x} _ {n}),\tag{8.21}
$$

we can consider other models for other prediction tasks, such as binary classification or modeling count data (McCullagh and Nelder, 1989). An alternative view of this is to consider likelihoods that are from the exponential family (Section 6.6). The class of models, which have linear dependence between parameters and data, and have potentially nonlinear transformation $\varphi$ (called a link function), is referred to as generalized linear models (Agresti, 2002, chapter 4). 

Maximum likelihood estimation has a rich history, and was originally proposed by Sir Ronald Fisher in the 1930s. We will expand upon the idea of a probabilistic model in Section 8.4. One debate among researchers who use probabilistic models, is the discussion between Bayesian and frequentist statistics. As mentioned in Section 6.1.1, it boils down to the definition of probability. Recall from Section 6.1 that one can consider probability to be a generalization (by allowing uncertainty) of logical reasoning (Cheeseman, 1985; Jaynes, 2003). The method of maximum likelihood estimation is frequentist in nature, and the interested reader is pointed to Efron and Hastie (2016) for a balanced view of both Bayesian and frequentist statistics. 

There are some probabilistic models where maximum likelihood estimation may not be possible. The reader is referred to more advanced statistical textbooks, e.g., Casella and Berger (2002), for approaches, such as method of moments, M-estimation, and estimating equations. 

## 8.4 Probabilistic Modeling and Inference

In machine learning, we are frequently concerned with the interpretation and analysis of data, e.g., for prediction of future events and decision making. To make this task more tractable, we often build models that describe the generative process that generates the observed data. 