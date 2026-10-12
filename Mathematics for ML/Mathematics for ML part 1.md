## MATHEMATICS FOR MACHINE LEARNING

Marc Peter Deisenroth A. Aldo Faisal Cheng Soon Ong 

## Contents

Foreword 1
Part I Mathematical Foundations 9
1 Introduction and Motivation 11
1.1 Finding Words for Intuitions 12
1.2 Two Ways to Read This Book 13
1.3 Exercises and Feedback 16
2 Linear Algebra 17
2.1 Systems of Linear Equations 19
2.2 Matrices 22
2.3 Solving Systems of Linear Equations 27
2.4 Vector Spaces 35
2.5 Linear Independence 40
2.6 Basis and Rank 44
2.7 Linear Mappings 48
2.8 Affine Spaces 61
2.9 Further Reading 63
Exercises 64
3 Analytic Geometry 70
3.1 Norms 71
3.2 Inner Products 72
3.3 Lengths and Distances 75
3.4 Angles and Orthogonality 76
3.5 Orthonormal Basis 78
3.6 Orthogonal Complement 79
3.7 Inner Product of Functions 80
3.8 Orthogonal Projections 81
3.9 Rotations 91
3.10 Further Reading 94
Exercises 96
4 Matrix Decompositions 98
4.1 Determinant and Trace 99 

ii Contents
4.2 Eigenvalues and Eigenvectors 105
4.3 Cholesky Decomposition 114
4.4 Eigendecomposition and Diagonalization 115
4.5 Singular Value Decomposition 119
4.6 Matrix Approximation 129
4.7 Matrix Phylogeny 134
4.8 Further Reading 135
Exercises 137
5 Vector Calculus 139
5.1 Differentiation of Univariate Functions 141
5.2 Partial Differentiation and Gradients 146
5.3 Gradients of Vector-Valued Functions 149
5.4 Gradients of Matrices 155
5.5 Useful Identities for Computing Gradients 158
5.6 Backpropagation and Automatic Differentiation 159
5.7 Higher-Order Derivatives 164
5.8 Linearization and Multivariate Taylor Series 165
5.9 Further Reading 170
Exercises 170
6 Probability and Distributions 172
6.1 Construction of a Probability Space 172
6.2 Discrete and Continuous Probabilities 178
6.3 Sum Rule, Product Rule, and Bayes' Theorem 183
6.4 Summary Statistics and Independence 186
6.5 Gaussian Distribution 197
6.6 Conjugacy and the Exponential Family 205
6.7 Change of Variables/Inverse Transform 214
6.8 Further Reading 221
Exercises 222
7 Continuous Optimization 225
7.1 Optimization Using Gradient Descent 227
7.2 Constrained Optimization and Lagrange Multipliers 233
7.3 Convex Optimization 236
7.4 Further Reading 246
Exercises 247
Part II Central Machine Learning Problems 249
8 When Models Meet Data 251
8.1 Data, Models, and Learning 251
8.2 Empirical Risk Minimization 258
8.3 Parameter Estimation 265
8.4 Probabilistic Modeling and Inference 272
8.5 Directed Graphical Models 278 

Contents iii
8.6 Model Selection 283
9 Linear Regression 289
9.1 Problem Formulation 291
9.2 Parameter Estimation 292
9.3 Bayesian Linear Regression 303
9.4 Maximum Likelihood as Orthogonal Projection 313
9.5 Further Reading 315
10 Dimensionality Reduction with Principal Component Analysis 317
10.1 Problem Setting 318
10.2 Maximum Variance Perspective 320
10.3 Projection Perspective 325
10.4 Eigenvector Computation and Low-Rank Approximations 333
10.5 PCA in High Dimensions 335
10.6 Key Steps of PCA in Practice 336
10.7 Latent Variable Perspective 339
10.8 Further Reading 343
11 Density Estimation with Gaussian Mixture Models 348
11.1 Gaussian Mixture Model 349
11.2 Parameter Learning via Maximum Likelihood 350
11.3 EM Algorithm 360
11.4 Latent-Variable Perspective 363
11.5 Further Reading 368
12 Classification with Support Vector Machines 370
12.1 Separating Hyperplanes 372
12.2 Primal Support Vector Machine 374
12.3 Dual Support Vector Machine 383
12.4 Kernels 388
12.5 Numerical Solution 390
12.6 Further Reading 392
References 395
Index 407 

## Foreword

Machine learning is the latest in a long line of attempts to distill human knowledge and reasoning into a form that is suitable for constructing machines and engineering automated systems. As machine learning becomes more ubiquitous and its software packages become easier to use, it is natural and desirable that the low-level technical details are abstracted away and hidden from the practitioner. However, this brings with it the danger that a practitioner becomes unaware of the design decisions and, hence, the limits of machine learning algorithms. 

The enthusiastic practitioner who is interested to learn more about the magic behind successful machine learning algorithms currently faces a daunting set of pre-requisite knowledge: 

Programming languages and data analysis tools 

Large-scale computation and the associated frameworks 

Mathematics and statistics and how machine learning builds on it 

At universities, introductory courses on machine learning tend to spend early parts of the course covering some of these pre-requisites. For historical reasons, courses in machine learning tend to be taught in the computer science department, where students are often trained in the first two areas of knowledge, but not so much in mathematics and statistics. 

Current machine learning textbooks primarily focus on machine learning algorithms and methodologies and assume that the reader is competent in mathematics and statistics. Therefore, these books only spend one or two chapters on background mathematics, either at the beginning of the book or as appendices. We have found many people who want to delve into the foundations of basic machine learning methods who struggle with the mathematical knowledge required to read a machine learning textbook. Having taught undergraduate and graduate courses at universities, we find that the gap between high school mathematics and the mathematics level required to read a standard machine learning textbook is too big for many people. 

This book brings the mathematical foundations of basic machine learning concepts to the fore and collects the information in a single place so that this skills gap is narrowed or even closed. 

## Why Another Book on Machine Learning?

Machine learning builds upon the language of mathematics to express concepts that seem intuitively obvious but that are surprisingly difficult to formalize. Once formalized properly, we can gain insights into the task we want to solve. One common complaint of students of mathematics around the globe is that the topics covered seem to have little relevance to practical problems. We believe that machine learning is an obvious and direct motivation for people to learn mathematics. 

This book is intended to be a guidebook to the vast mathematical literature that forms the foundations of modern machine learning. We motivate the need for mathematical concepts by directly pointing out their usefulness in the context of fundamental machine learning problems. In the interest of keeping the book short, many details and more advanced concepts have been left out. Equipped with the basic concepts presented here, and how they fit into the larger context of machine learning, the reader can find numerous resources for further study, which we provide at the end of the respective chapters. For readers with a mathematical background, this book provides a brief but precisely stated glimpse of machine learning. In contrast to other books that focus on methods and models of machine learning (MacKay, 2003; Bishop, 2006; Alpaydin, 2010; Barber, 2012; Murphy, 2012; Shalev-Shwartz and Ben-David, 2014; Rogers and Girolami, 2016) or programmatic aspects of machine learning (Muller¨ and Guido, 2016; Raschka and Mirjalili, 2017; Chollet and Allaire, 2018), we provide only four representative examples of machine learning algorithms. Instead, we focus on the mathematical concepts behind the models themselves. We hope that readers will be able to gain a deeper understanding of the basic questions in machine learning and connect practical questions arising from the use of machine learning with fundamental choices in the mathematical model. 

We do not aim to write a classical machine learning book. Instead, our intention is to provide the mathematical background, applied to four central machine learning problems, to make it easier to read other machine learning textbooks. 

## Who Is the Target Audience?

As applications of machine learning become widespread in society, we believe that everybody should have some understanding of its underlying principles. This book is written in an academic mathematical style, which enables us to be precise about the concepts behind machine learning. We encourage readers unfamiliar with this seemingly terse style to persevere and to keep the goals of each topic in mind. We sprinkle comments and remarks throughout the text, in the hope that it provides useful guidance with respect to the big picture. 

The book assumes the reader to have mathematical knowledge commonly covered in high school mathematics and physics. For example, the reader should have seen derivatives and integrals before, and geometric vectors in two or three dimensions. Starting from there, we generalize these concepts. Therefore, the target audience of the book includes undergraduate university students, evening learners and learners participating in online machine learning courses. 

In analogy to music, there are three types of interaction that people have with machine learning: 

Astute Listener The democratization of machine learning by the provision of open-source software, online tutorials and cloud-based tools allows users to not worry about the specifics of pipelines. Users can focus on extracting insights from data using off-the-shelf tools. This enables nontech-savvy domain experts to benefit from machine learning. This is sim ilar to listening to music; the user is able to choose and discern between different types of machine learning, and benefits from it. More experienced users are like music critics, asking important questions about the application of machine learning in society such as ethics, fairness, and privacy of the individual. We hope that this book provides a foundation for thinking about the certification and risk management of machine learning systems, and allows them to use their domain expertise to build better machine learning systems. 

Experienced Artist Skilled practitioners of machine learning can plug and play different tools and libraries into an analysis pipeline. The stereotypical practitioner would be a data scientist or engineer who understands machine learning interfaces and their use cases, and is able to perform wonderful feats of prediction from data. This is similar to a virtuoso playing music, where highly skilled practitioners can bring existing instruments to life and bring enjoyment to their audience. Using the mathematics presented here as a primer, practitioners would be able to understand the benefits and limits of their favorite method, and to extend and generalize existing machine learning algorithms. We hope that this book provides the impetus for more rigorous and principled development of machine learning methods. 

Fledgling Composer As machine learning is applied to new domains, developers of machine learning need to develop new methods and extend existing algorithms. They are often researchers who need to understand the mathematical basis of machine learning and uncover relationships between different tasks. This is similar to composers of music who, within the rules and structure of musical theory, create new and amazing pieces. We hope this book provides a high-level overview of other technical books for people who want to become composers of machine learning. There is a great need in society for new researchers who are able to propose and explore novel approaches for attacking the many challenges of learning from data. 



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



## Acknowledgments

We are grateful to many people who looked at early drafts of the book and suffered through painful expositions of concepts. We tried to implement their ideas that we did not vehemently disagree with. We would like to especially acknowledge Christfried Webers for his careful reading of many parts of the book, and his detailed suggestions on structure and presentation. Many friends and colleagues have also been kind enough to provide their time and energy on different versions of each chapter. We have been lucky to benefit from the generosity of the online community, who have suggested improvements via https://github.com, which greatly improved the book. 

The following people have found bugs, proposed clarifications and suggested relevant literature, either via https://github.com or personal communication. Their names are sorted alphabetically. 

Abdul-Ganiy Usman Ellen Broad Adam Gaier Fengkuangtian Zhu Adele Jackson Fiona Condon Aditya Menon Georgios Theodorou Alasdair Tran He Xin Aleksandar Krnjaic Irene Raissa Kameni Alexander Makrigiorgos Jakub Nabaglo Alfredo Canziani James Hensman Ali Shafti Jamie Liu Amr Khalifa Jean Kaddour Andrew Tanggara Jean-Paul Ebejer Angus Gruen Jerry Qiang Antal A. Buss Jitesh Sindhare Antoine Toisoul Le Cann John Lloyd Areg Sarvazyan Jonas Ngnawe Artem Artemev Jon Martin Artyom Stepanov Justin Hsi Bill Kromydas Kai Arulkumaran Bob Williamson Kamil Dreczkowski Boon Ping Lim Lily Wang Chao Qu Lionel Tondji Ngoupey Cheng Li Lydia Knufing¨ Chris Sherlock Mahmoud Aslan Christopher Gray Mark Hartenstein Daniel McNamara Mark van der Wilk Daniel Wood Markus Hegland Darren Siegel Martin Hewing David Johnston Matthew Alger Dawei Chen Matthew Lee 

Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 

## Foreword

<table><tr><td>Maximus McCann</td><td>Shakir Mohamed</td></tr><tr><td>Mengyan Zhang</td><td>Shawn Berry</td></tr><tr><td>Michael Bennett</td><td>Sheikh Abdul Raheem Ali</td></tr><tr><td>Michael Pedersen</td><td>Sheng Xue</td></tr><tr><td>Minjeong Shin</td><td>Sridhar Thiagarajan</td></tr><tr><td>Mohammad Malekzadeh</td><td>Syed Nouman Hasany</td></tr><tr><td>Naveen Kumar</td><td>Szymon Brych</td></tr><tr><td>Nico Montali</td><td>Thomas Bühler</td></tr><tr><td>Oscar Armas</td><td>Timur Sharapov</td></tr><tr><td>Patrick Henriksen</td><td>Tom Melamed</td></tr><tr><td>Patrick Wieschollek</td><td>Vincent Adam</td></tr><tr><td>Pattarawat Chormai</td><td>Vincent Dutordoir</td></tr><tr><td>Paul Kelly</td><td>Vu Minh</td></tr><tr><td>Petros Christodoulou</td><td>Wasim Aftab</td></tr><tr><td>Piotr Januszewski</td><td>Wen Zhi</td></tr><tr><td>Pranav Subramani</td><td>Wojciech Stokowiec</td></tr><tr><td>Quyu Kong</td><td>Xiaonan Chong</td></tr><tr><td>Ragib Zaman</td><td>Xiaowei Zhang</td></tr><tr><td>Rui Zhang</td><td>Yazhou Hao</td></tr><tr><td>Ryan-Rhys Griffiths</td><td>Yicheng Luo</td></tr><tr><td>Salomon Kabongo</td><td>Young Lee</td></tr><tr><td>Samuel Ogunmola</td><td>Yu Lu</td></tr><tr><td>Sandeep Mavadia</td><td>Yun Cheng</td></tr><tr><td>Sarvesh Nikumbh</td><td>Yuxiao Huang</td></tr><tr><td>Sebastian Raschka</td><td>Zac Cranko</td></tr><tr><td>Senanayak Sesh Kumar Karri</td><td>Zijian Cao</td></tr><tr><td>Seung-Heon Baek</td><td>Zoe Nolan</td></tr></table>

Contributors through GitHub, whose real names were not listed on their GitHub profile, are: 

<table><tr><td>SamDataMad</td><td>insad</td><td>empet</td></tr><tr><td>bumptiousmonkey</td><td>HorizonP</td><td>victorBigand</td></tr><tr><td>idoamihai</td><td>cs-maillist</td><td>17SKYE</td></tr><tr><td>deepakiim</td><td>kudo23</td><td>jessjing1995</td></tr></table>

We are also very grateful to Parameswaran Raman and the many anonymous reviewers, organized by Cambridge University Press, who read one or more chapters of earlier versions of the manuscript, and provided constructive criticism that led to considerable improvements. A special mention goes to Dinesh Singh Negi, our LAT X support, for detailed and prompt advice about LAT X-related issues. Last but not least, we are very grateful to our editor Lauren Cowles, who has been patiently guiding us through the gestation process of this book. 

©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 


Table of Symbols


<table><tr><td>Symbol</td><td>Typical meaning</td></tr><tr><td>a,b,c,α,β,γ</td><td>Scalars are lowercase</td></tr><tr><td>x,y,z</td><td>Vectors are bold lowercase</td></tr><tr><td>A,B,C</td><td>Matrices are bold uppercase</td></tr><tr><td>x<eq>^{\top}</eq>,A<eq>^{\top}</eq></td><td>Transpose of a vector or matrix</td></tr><tr><td>A<eq>^{-1}</eq></td><td>Inverse of a matrix</td></tr><tr><td>{x,y}</td><td>Inner product of x and y</td></tr><tr><td>x<eq>^{\top}</eq>y</td><td>Dot product of x and y</td></tr><tr><td>B=(b1,b2,b3)</td><td>(Ordered) tuple</td></tr><tr><td>B=[b1,b2,b3]</td><td>Matrix of column vectors stacked horizontally</td></tr><tr><td>B={b1,b2,b3}</td><td>Set of vectors (unordered)</td></tr><tr><td>Z,N</td><td>Integers and natural numbers, respectively</td></tr><tr><td>R,C</td><td>Real and complex numbers, respectively</td></tr><tr><td>R<eq>^{n}</eq></td><td>n-dimensional vector space of real numbers</td></tr><tr><td>∀x</td><td>Universal quantifier: for all x</td></tr><tr><td>∃x</td><td>Existential quantifier: there exists x</td></tr><tr><td>a:=b</td><td>a is defined as b</td></tr><tr><td>a=:b</td><td>b is defined as a</td></tr><tr><td>a∝b</td><td>a is proportional to b, i.e., a = constant · b</td></tr><tr><td>g∘f</td><td>Function composition: “g after f”</td></tr><tr><td>⇔</td><td>If and only if</td></tr><tr><td>⇒</td><td>Implies</td></tr><tr><td>A,C</td><td>Sets</td></tr><tr><td>a∈A</td><td>a is an element of set A</td></tr><tr><td>∅</td><td>Empty set</td></tr><tr><td>A\B</td><td>A without B: the set of elements in A but not in B</td></tr><tr><td>D</td><td>Number of dimensions; indexed by d=1,...,D</td></tr><tr><td>N</td><td>Number of data points; indexed by n=1,...,N</td></tr><tr><td>I<eq>_{m}</eq></td><td>Identity matrix of size m × m</td></tr><tr><td>0<eq>_{m,n}</eq></td><td>Matrix of zeros of size m × n</td></tr><tr><td>1<eq>_{m,n}</eq></td><td>Matrix of ones of size m × n</td></tr><tr><td>ei</td><td>Standard/canonical vector (where i is the component that is 1)</td></tr><tr><td>dim</td><td>Dimensionality of vector space</td></tr><tr><td>rk(A)</td><td>Rank of matrix A</td></tr><tr><td>Im(Φ)</td><td>Image of linear mapping Φ</td></tr><tr><td>ker(Φ)</td><td>Kernel (null space) of a linear mapping Φ</td></tr><tr><td>span[b1]</td><td>Span (generating set) of b<eq>_{1}</eq></td></tr><tr><td>tr(A)</td><td>Trace of A</td></tr><tr><td>det(A)</td><td>Determinant of A</td></tr><tr><td>|·|</td><td>Absolute value or determinant (depending on context)</td></tr><tr><td>||·||</td><td>Norm; Euclidean, unless specified</td></tr><tr><td>λ</td><td>Eigenvalue or Lagrange multiplier</td></tr><tr><td>Eλ</td><td>Eigenspace corresponding to eigenvalue λ</td></tr></table>


Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 


<table><tr><td>Symbol</td><td>Typical meaning</td></tr><tr><td><eq>\boldsymbol{x} \perp \boldsymbol{y}</eq></td><td>Vectors <eq>\boldsymbol{x}</eq> and <eq>\boldsymbol{y}</eq> are orthogonal</td></tr><tr><td><eq>V</eq></td><td>Vector space</td></tr><tr><td><eq>V^{\perp}</eq></td><td>Orthogonal complement of vector space <eq>V</eq></td></tr><tr><td><eq>\sum_{n=1}^{N} x_n</eq></td><td>Sum of the <eq>x_n: x_1 + \ldots + x_N</eq></td></tr><tr><td><eq>\prod_{n=1}^{N} x_n</eq></td><td>Product of the <eq>x_n: x_1 \cdot \ldots \cdot x_N</eq></td></tr><tr><td><eq>\boldsymbol{\theta}</eq></td><td>Parameter vector</td></tr><tr><td><eq>\frac{\partial f}{\partial x}</eq></td><td>Partial derivative of <eq>f</eq> with respect to <eq>x</eq></td></tr><tr><td><eq>\frac{\mathrm{d}f}{\mathrm{d}x}</eq></td><td>Total derivative of <eq>f</eq> with respect to <eq>x</eq></td></tr><tr><td><eq>\nabla</eq></td><td>Gradient</td></tr><tr><td><eq>f_* = \min_x f(x)</eq></td><td>The smallest function value of <eq>f</eq></td></tr><tr><td><eq>x_* \in \arg \min_x f(x)</eq></td><td>The value <eq>x_*</eq> that minimizes <eq>f</eq> (note: arg min returns a set of values)</td></tr><tr><td><eq>\mathcal{L}</eq></td><td>Lagrangian</td></tr><tr><td><eq>\mathcal{L}</eq></td><td>Negative log-likelihood</td></tr><tr><td><eq>\binom{n}{k}</eq></td><td>Binomial coefficient, <eq>n</eq> choose <eq>k</eq></td></tr><tr><td><eq>\mathrm{V}_X[\boldsymbol{x}]</eq></td><td>Variance of <eq>\boldsymbol{x}</eq> with respect to the random variable <eq>X</eq></td></tr><tr><td><eq>\mathbb{E}_X[\boldsymbol{x}]</eq></td><td>Expectation of <eq>\boldsymbol{x}</eq> with respect to the random variable <eq>X</eq></td></tr><tr><td><eq>\text{Cov}_{X,Y}[\boldsymbol{x}, \boldsymbol{y}]</eq></td><td>Covariance between <eq>\boldsymbol{x}</eq> and <eq>\boldsymbol{y}</eq>.</td></tr><tr><td><eq>X \perp Y | Z</eq></td><td><eq>X</eq> is conditionally independent of <eq>Y</eq> given <eq>Z</eq></td></tr><tr><td><eq>X \sim p</eq></td><td>Random variable <eq>X</eq> is distributed according to <eq>p</eq></td></tr><tr><td><eq>\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})</eq></td><td>Gaussian distribution with mean <eq>\boldsymbol{\mu}</eq> and covariance <eq>\boldsymbol{\Sigma}</eq></td></tr><tr><td><eq>\text{Ber}(\boldsymbol{\mu})</eq></td><td>Bernoulli distribution with parameter <eq>\boldsymbol{\mu}</eq></td></tr><tr><td><eq>\text{Bin}(N, \boldsymbol{\mu})</eq></td><td>Binomial distribution with parameters <eq>N, \boldsymbol{\mu}</eq></td></tr><tr><td><eq>\text{Beta}(\boldsymbol{\alpha}, \boldsymbol{\beta})</eq></td><td>Beta distribution with parameters <eq>\boldsymbol{\alpha}, \boldsymbol{\beta}</eq></td></tr></table>

## Table of Abbreviations and Acronyms

<table><tr><td>Acronym</td><td>Meaning</td></tr><tr><td>e.g.</td><td>Exempli gratia (Latin: for example)</td></tr><tr><td>GMM</td><td>Gaussian mixture model</td></tr><tr><td>i.e.</td><td>Id est (Latin: this means)</td></tr><tr><td>i.i.d.</td><td>Independent, identically distributed</td></tr><tr><td>MAP</td><td>Maximum a posteriori</td></tr><tr><td>MLE</td><td>Maximum likelihood estimation/estimator</td></tr><tr><td>ONB</td><td>Orthonormal basis</td></tr><tr><td>PCA</td><td>Principal component analysis</td></tr><tr><td>PPCA</td><td>Probabilistic principal component analysis</td></tr><tr><td>REF</td><td>Row-echelon form</td></tr><tr><td>SPD</td><td>Symmetric, positive definite</td></tr><tr><td>SVM</td><td>Support vector machine</td></tr></table>

Mathematical Foundations 

# Introduction and Motivation

Machine learning is about designing algorithms that automatically extract valuable information from data. The emphasis here is on “automatic”, i.e., machine learning is concerned about general-purpose methodologies that can be applied to many datasets, while producing something that is meaningful. There are three concepts that are at the core of machine learning: data, a model, and learning. 

Since machine learning is inherently data driven, data is at the core of machine learning. The goal of machine learning is to design generalpurpose methodologies to extract valuable patterns from data, ideally without much domain-specific expertise. For example, given a large corpus of documents (e.g., books in many libraries), machine learning methods can be used to automatically find relevant topics that are shared across documents (Hoffman et al., 2010). To achieve this goal, we design models that are typically related to the process that generates data, similar to the dataset we are given. For example, in a regression setting, the model would describe a function that maps inputs to real-valued outputs. To paraphrase Mitchell (1997): A model is said to learn from data if its performance on a given task improves after the data is taken into account. The goal is to find good models that generalize well to yet unseen data, which we may care about in the future. Learning can be understood as a way to automatically find patterns and structure in data by optimizing the parameters of the model. 

While machine learning has seen many success stories, and software is readily available to design and train rich and flexible machine learning systems, we believe that the mathematical foundations of machine learning are important in order to understand fundamental principles upon which more complicated machine learning systems are built. Understanding these principles can facilitate creating new machine learning solutions, understanding and debugging existing approaches, and learning about the inherent assumptions and limitations of the methodologies we are working with. 

## 1.1 Finding Words for Intuitions

A challenge we face regularly in machine learning is that concepts and words are slippery, and a particular component of the machine learning system can be abstracted to different mathematical concepts. For example, the word “algorithm” is used in at least two different senses in the context of machine learning. In the first sense, we use the phrase “machine learning algorithm” to mean a system that makes predictions based on input data. We refer to these algorithms as predictors. In the second sense, we use the exact same phrase “machine learning algorithm” to mean a system that adapts some internal parameters of the predictor so that it performs well on future unseen input data. Here we refer to this adaptation as training a system. 

This book will not resolve the issue of ambiguity, but we want to highlight upfront that, depending on the context, the same expressions can mean different things. However, we attempt to make the context sufficiently clear to reduce the level of ambiguity. 

The first part of this book introduces the mathematical concepts and foundations needed to talk about the three main components of a machine learning system: data, models, and learning. We will briefly outline these components here, and we will revisit them again in Chapter 8 once we have discussed the necessary mathematical concepts. 

While not all data is numerical, it is often useful to consider data in a number format. In this book, we assume that data has already been appropriately converted into a numerical representation suitable for reading into a computer program. Therefore, we think of data as vectors. As another illustration of how subtle words are, there are (at least) three different ways to think about vectors: a vector as an array of numbers (a computer science view), a vector as an arrow with a direction and magnitude (a physics view), and a vector as an object that obeys addition and scaling (a mathematical view). 

A model is typically used to describe a process for generating data, similar to the dataset at hand. Therefore, good models can also be thought of as simplified versions of the real (unknown) data-generating process, capturing aspects that are relevant for modeling the data and extracting hidden patterns from it. A good model can then be used to predict what would happen in the real world without performing real-world experiments. 

We now come to the crux of the matter, the learning component of machine learning. Assume we are given a dataset and a suitable model. Training the model means to use the data available to optimize some parameters of the model with respect to a utility function that evaluates how well the model predicts the training data. Most training methods can be thought of as an approach analogous to climbing a hill to reach its peak. In this analogy, the peak of the hill corresponds to a maximum of some desired performance measure. However, in practice, we are interested in the model to perform well on unseen data. Performing well on data that we have already seen (training data) may only mean that we found a good way to memorize the data. However, this may not generalize well to unseen data, and, in practical applications, we often need to expose our machine learning system to situations that it has not encountered before. 

Let us summarize the main concepts of machine learning that we cover in this book: 

We represent data as vectors. 

We choose an appropriate model, either using the probabilistic or optimization view. 

We learn from available data by using numerical optimization methods with the aim that the model performs well on data not used for training. 

## 1.2 Two Ways to Read This Book

We can consider two strategies for understanding the mathematics for machine learning: 

Bottom-up: Building up the concepts from foundational to more advanced. This is often the preferred approach in more technical fields, such as mathematics. This strategy has the advantage that the reader at all times is able to rely on their previously learned concepts. Unfortunately, for a practitioner many of the foundational concepts are not particularly interesting by themselves, and the lack of motivation means that most foundational definitions are quickly forgotten. 

Top-down: Drilling down from practical needs to more basic requirements. This goal-driven approach has the advantage that the readers know at all times why they need to work on a particular concept, and there is a clear path of required knowledge. The downside of this strategy is that the knowledge is built on potentially shaky foundations, and the readers have to remember a set of words that they do not have any way of understanding. 

We decided to write this book in a modular way to separate foundational (mathematical) concepts from applications so that this book can be read in both ways. The book is split into two parts, where Part I lays the math ematical foundations and Part II applies the concepts from Part I to a set of fundamental machine learning problems, which form four pillars of machine learning as illustrated in Figure 1.1: regression, dimensionality reduction, density estimation, and classification. Chapters in Part I mostly build upon the previous ones, but it is possible to skip a chapter and work backward if necessary. Chapters in Part II are only loosely coupled and can be read in any order. There are many pointers forward and backward 

Figure 1.1 The foundations and four pillars of machine learning. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/a58265c97d2a7a844bf6c2ca438792b82154aece86209084c98a22a2b01e7037.jpg)


between the two parts of the book to link mathematical concepts with machine learning algorithms. 

Of course there are more than two ways to read this book. Most readers learn using a combination of top-down and bottom-up approaches, sometimes building up basic mathematical skills before attempting more complex concepts, but also choosing topics based on applications of machine learning. 

## Part I Is about Mathematics

The four pillars of machine learning we cover in this book (see Figure 1.1) require a solid mathematical foundation, which is laid out in Part I. 

We represent numerical data as vectors and represent a table of such data as a matrix. The study of vectors and matrices is called linear algebra, which we introduce in Chapter 2. The collection of vectors as a matrix is also described there. 

Given two vectors representing two objects in the real world, we want to make statements about their similarity. The idea is that vectors that are similar should be predicted to have similar outputs by our machine learning algorithm (our predictor). To formalize the idea of similarity between vectors, we need to introduce operations that take two vectors as input and return a numerical value representing their similarity. The construction of similarity and distances is central to analytic geometry and is discussed in Chapter 3. 

In Chapter 4, we introduce some fundamental concepts about matrices and matrix decomposition. Some operations on matrices are extremely useful in machine learning, and they allow for an intuitive interpretation of the data and more efficient learning. 

We often consider data to be noisy observations of some true underlying signal. We hope that by applying machine learning we can identify the signal from the noise. This requires us to have a language for quantifying what “noise” means. We often would also like to have predictors that allow us to express some sort of uncertainty, e.g., to quantify the confidence we have about the value of the prediction at a particular test data point. Quantification of uncertainty is the realm of probability theory and is covered in Chapter 6. 

To train machine learning models, we typically find parameters that maximize some performance measure. Many optimization techniques require the concept of a gradient, which tells us the direction in which to search for a solution. Chapter 5 is about vector calculus and details the concept of gradients, which we subsequently use in Chapter 7, where we talk about optimization to find maxima/minima of functions. 

## Part II Is about Machine Learning

The second part of the book introduces four pillars of machine learning as shown in Figure 1.1. We illustrate how the mathematical concepts introduced in the first part of the book are the foundation for each pillar. Broadly speaking, chapters are ordered by difficulty (in ascending order). 

In Chapter 8, we restate the three components of machine learning (data, models, and parameter estimation) in a mathematical fashion. In addition, we provide some guidelines for building experimental set-ups that guard against overly optimistic evaluations of machine learning systems. Recall that the goal is to build a predictor that performs well on unseen data. 

In Chapter 9, we will have a close look at linear regression, where our objective is to find functions that map inputs $\pmb { x } \in \mathbb { R } ^ { D }$ to corresponding observed function values $y \in \mathbb { R }$ , which we can interpret as the labels of their respective inputs. We will discuss classical model fitting (parameter estimation) via maximum likelihood and maximum a posteriori estimation, as well as Bayesian linear regression, where we integrate the parameters out instead of optimizing them. 

Chapter 10 focuses on dimensionality reduction, the second pillar in Figure 1.1, using principal component analysis. The key objective of dimensionality reduction is to find a compact, lower-dimensional representation of high-dimensional data $\pmb { x } \in \mathbb { R } ^ { D }$ , which is often easier to analyze than the original data. Unlike regression, dimensionality reduction is only concerned about modeling the data – there are no labels associated with a data point x. 

In Chapter 11, we will move to our third pillar: density estimation. The objective of density estimation is to find a probability distribution that describes a given dataset. We will focus on Gaussian mixture models for this purpose, and we will discuss an iterative scheme to find the parameters of this model. As in dimensionality reduction, there are no labels associated with the data points $\pmb { x } \in \mathbb { R } ^ { D }$ . However, we do not seek a low-dimensional representation of the data. Instead, we are interested in a density model that describes the data. 

Chapter 12 concludes the book with an in-depth discussion of the fourth pillar: classification. We will discuss classification in the context of support vector machines. Similar to regression (Chapter 9), we have inputs x and corresponding labels $y .$ However, unlike regression, where the labels were real-valued, the labels in classification are integers, which requires special care. 

## 1.3 Exercises and Feedback

We provide some exercises in Part I, which can be done mostly by pen and paper. For Part II, we provide programming tutorials (jupyter notebooks) to explore some properties of the machine learning algorithms we discuss in this book. 

We appreciate that Cambridge University Press strongly supports our aim to democratize education and learning by making this book freely available for download at 

$$
\mathrm{https://mml-book.com}
$$

where tutorials, errata, and additional materials can be found. Mistakes can be reported and feedback provided using the preceding URL. 

# Linear Algebra

When formalizing intuitive concepts, a common approach is to construct a set of objects (symbols) and a set of rules to manipulate these objects. This is known as an algebra. Linear algebra is the study of vectors and certain rules to manipulate vectors. The vectors many of us know from school are called “geometric vectors”, which are usually denoted by a small arrow above the letter, $\mathrm { e } . g . , \ \vec { x }$ and $\vec { y }$ . In this book, we discuss more general concepts of vectors and use a bold letter to represent them, e.g., x and $\mathbf { \mathscr { y } } .$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/85cf7372a02c0448fab1d285ada07c8caaf9f6d08d7d900c139e3a594edba9e0.jpg)


In general, vectors are special objects that can be added together and multiplied by scalars to produce another object of the same kind. From an abstract mathematical viewpoint, any object that satisfies these two properties can be considered a vector. Here are some examples of such vector objects: 

1. Geometric vectors. This example of a vector may be familiar from high school mathematics and physics. Geometric vectors – see Figure 2.1(a) – are directed segments, which can be drawn (at least in two dimensions). Two geometric vectors $\vec { x }$ , $\vec { y }$ can be added, such that $\stackrel {  } { \pmb { x } } + \stackrel {  } { \pmb { y } } = \stackrel {  } { \pmb { z } }$ is another geometric vector. Furthermore, multiplication by a scalar $\lambda { \vec { x } } , \lambda \in \mathbb { R } ,$ is also a geometric vector. In fact, it is the original vector scaled by $\lambda .$ Therefore, geometric vectors are instances of the vector concepts introduced previously. Interpreting vectors as geometric vec tors enables us to use our intuitions about direction and magnitude to reason about mathematical operations. 

2. Polynomials are also vectors; see Figure 2.1(b): Two polynomials can be added together, which results in another polynomial; and they can be multiplied by a scalar $\lambda \ \in \ \mathbb { R } ,$ , and the result is a polynomial as well. Therefore, polynomials are (rather unusual) instances of vectors. Note that polynomials are very different from geometric vectors. While geometric vectors are concrete “drawings”, polynomials are abstract concepts. However, they are both vectors in the sense previously described. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/926dca932d4326834fa6e0b6254624a48a29970baa7c29b5d88cd2d97e6d7d7b.jpg)



(a) Geometric vectors.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/522499b21d40b9d9e952767f7e4465f7dec9db7f21d55d20adb8b84384c28eb7.jpg)



(b) Polynomials.



Figure 2.1 Different types of vectors. Vectors can be surprising objects, including (a) geometric vectors and (b) polynomials.


3. Audio signals are vectors. Audio signals are represented as a series of numbers. We can add audio signals together, and their sum is a new audio signal. If we scale an audio signal, we also obtain an audio signal. Therefore, audio signals are a type of vector, too. 

4. Elements of $\mathbb { R } ^ { n }$ (tuples of $n$ real numbers) are vectors. $\mathbb { R } ^ { n }$ is more abstract than polynomials, and it is the concept we focus on in this book. For instance, 

$$
\boldsymbol {a} = \left[ \begin{array}{c} 1 \\ 2 \\ 3 \end{array} \right] \in \mathbb {R} ^ {3}\tag{2.1}
$$

is an example of a triplet of numbers. Adding two vectors $\mathbf { \Delta } a , b \in \mathbb { R } ^ { n }$ component-wise results in another vector: $\pmb { a } + \pmb { b } = \pmb { c } \in \mathbb { R } ^ { n }$ . Moreover, multiplying $\pmb { a } \in \mathbb { R } ^ { n }$ by $\lambda \in \mathbb { R }$ results in a scaled vector $\boldsymbol { \lambda } \boldsymbol { a } ~ \in ~ \mathbb { R } ^ { n }$ Considering vectors as elements of $\mathbb { R } ^ { n }$ has an additional benefit that it loosely corresponds to arrays of real numbers on a computer. Many programming languages support array operations, which allow for convenient implementation of algorithms that involve vector operations. 

Linear algebra focuses on the similarities between these vector concepts. We can add them together and multiply them by scalars. We will largely focus on vectors in $\mathbb { R } ^ { n }$ since most algorithms in linear algebra are formulated in $\mathbb { R } ^ { n }$ . We will see in Chapter 8 that we often consider data to be represented as vectors in $\mathbb { R } ^ { n }$ . In this book, we will focus on finitedimensional vector spaces, in which case there is a 1:1 correspondence between any kind of vector and $\mathbb { R } ^ { n }$ . When it is convenient, we will use intuitions about geometric vectors and consider array-based algorithms. 

One major idea in mathematics is the idea of “closure”. This is the question: What is the set of all things that can result from my proposed operations? In the case of vectors: What is the set of vectors that can result by starting with a small set of vectors, and adding them to each other and scaling them? This results in a vector space (Section 2.4). The concept of a vector space and its properties underlie much of machine learning. The concepts introduced in this chapter are summarized in Figure 2.2. 

This chapter is mostly based on the lecture notes and books by Drumm and Weil (2001), Strang (2003), Hogben (2013), Liesen and Mehrmann (2015), as well as Pavel Grinfeld’s Linear Algebra series. Other excellent resources are Gilbert Strang’s Linear Algebra course at MIT and the Linear Algebra Series by 3Blue1Brown. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/12edb14c26944ac49e05cad6d5f6e98cfcee61816e83109dccdc57f20fc4f061.jpg)



Figure 2.2 A mind map of the concepts introduced in this chapter, along with where they are used in other parts of the book.


Linear algebra plays an important role in machine learning and general mathematics. The concepts introduced in this chapter are further expanded to include the idea of geometry in Chapter 3. In Chapter 5, we will discuss vector calculus, where a principled knowledge of matrix operations is essential. In Chapter 10, we will use projections (to be introduced in Section 3.8) for dimensionality reduction with principal component analysis (PCA). In Chapter 9, we will discuss linear regression, where linear algebra plays a central role for solving least-squares problems. 

## 2.1 Systems of Linear Equations

Systems of linear equations play a central part of linear algebra. Many problems can be formulated as systems of linear equations, and linear algebra gives us the tools for solving them. 

Example 2.1 

A company produces products $N _ { 1 } , \ldots , N _ { n }$ for which resources $R _ { 1 } , \ldots , R _ { m }$ are required. To produce a unit of product $N _ { j } , a _ { i j }$ units of resource $R _ { i }$ are needed, where $i = 1 , \ldots , m$ and $j = 1 , \dots , n .$ 

The objective is to find an optimal production plan, i.e., a plan of how many units $x _ { j }$ of product $N _ { j }$ should be produced if a total of $b _ { i }$ units of resource $R _ { i }$ are available and (ideally) no resources are left over. 

If we produce $x _ { 1 } , \ldots , x _ { n }$ units of the corresponding products, we need 



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



a total of 

$$
a _ {i 1} x _ {1} + \dots + a _ {i n} x _ {n}\tag{2.2}
$$

many units of resource $R _ { i }$ . An optimal production plan $( x _ { 1 } , \ldots , x _ { n } ) \in \mathbb { R } ^ { n }$ therefore, has to satisfy the following system of equations: 

$$
\begin{array}{c} a _ {1 1} x _ {1} + \dots + a _ {1 n} x _ {n} = b _ {1} \\ \vdots \\ a _ {m 1} x _ {1} + \dots + a _ {m n} x _ {n} = b _ {m} \end{array} ,\tag{2.3}
$$

where $a _ { i j } \in \mathbb { R }$ and $b _ { i } \in \mathbb { R }$ 

Equation (2.3) is the general form of a system of linear equations, and $x _ { 1 } , \ldots , x _ { n }$ are the unknowns of this system. Every n-tuple $( x _ { 1 } , \ldots , x _ { n } ) \in$ $\mathbb { R } ^ { n }$ that satisfies (2.3) is a solution of the linear equation system. 

## Example 2.2

The system of linear equations 

$$
\begin{array}{ccccccc}x _ {1} & + & x _ {2} & + & x _ {3} & = & 3 \\ x _ {1} & - & x _ {2} & + & 2 x _ {3} & = & 2 \\ 2 x _ {1} & & & + & 3 x _ {3} & = & 1 \\ \end{array}\tag{1}
$$

(2) 

(3) 

(2.4) 

has no solution: Adding the first two equations yields $2 x _ { 1 } + 3 x _ { 3 } = 5$ , which contradicts the third equation (3). 

Let us have a look at the system of linear equations 

$$
\begin{array}{r c l r c l r c l} x _ {1} & + & x _ {2} & + & x _ {3} & = & 3 & (1) \\ x _ {1} & - & x _ {2} & + & 2 x _ {3} & = & 2 & (2) \\ & & x _ {2} & + & x _ {3} & = & 2 & (3) \end{array} .\tag{2.5}
$$

From the first and third equation, it follows that $x _ { 1 } = 1$ . From $( 1 ) + ( 2 )$ we get 2x $+ 3 x _ { 3 } = 5 , { \mathrm { i . e . , ~ } } x _ { 3 } = 1$ . From (3), we then get that $x _ { 2 } = 1$ Therefore, (1, 1, 1) is the only possible and unique solution (verify that (1, 1, 1) is a solution by plugging in). 

As a third example, we consider 

$$
\begin{array}{c c c c c c c c} x _ {1} & + & x _ {2} & + & x _ {3} & = & 3 & (1) \\ x _ {1} & - & x _ {2} & + & 2 x _ {3} & = & 2 & (2) \\ 2 x _ {1} & & & + & 3 x _ {3} & = & 5 & (3) \end{array}\tag{2.6}
$$

Since $( 1 ) + ( 2 ) = ( 3 )$ , we can omit the third equation (redundancy). From (1) and (2), we get $2 x _ { 1 } = 5 { - } 3 x _ { 3 }$ and $2 x _ { 2 } = 1 + x _ { 3 }$ . We define $x _ { 3 } = a \in$ R as a free variable, such that any triplet 

$$
\left(\frac {5}{2} - \frac {3}{2} a, \frac {1}{2} + \frac {1}{2} a, a\right), \quad a \in \mathbb {R}\tag{2.7}
$$



Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 



## 2.1 Systems of Linear Equations

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/c4f4d592a06a0b60fdf056d3885d5c002999a91d1696d02c8fa278f8a370f060.jpg)



Figure 2.3 The solution space of a system of two linear equations with two variables can be geometrically interpreted as the intersection of two lines. Every linear equation represents a line.


is a solution of the system of linear equations, i.e., we obtain a solution set that contains infinitely many solutions. 

In general, for a real-valued system of linear equations we obtain either no, exactly one, or infinitely many solutions. Linear regression (Chapter 9) solves a version of Example 2.1 when we cannot solve the system of linear equations. 

Remark (Geometric Interpretation of Systems of Linear Equations). In a system of linear equations with two variables $x _ { 1 } , x _ { 2 }$ , each linear equation defines a line on the $x _ { 1 } x _ { 2 } – \mathsf { p l a n e }$ . Since a solution to a system of linear equations must satisfy all equations simultaneously, the solution set is the intersection of these lines. This intersection set can be a line (if the linear equations describe the same line), a point, or empty (when the lines are parallel). An illustration is given in Figure 2.3 for the system 

$$
\begin{array}{l} {4 x _ {1} + 4 x _ {2} = 5} \\ {2 x _ {1} - 4 x _ {2} = 1} \end{array}\tag{2.8}
$$

where the solution space is the point $( x _ { 1 } , x _ { 2 } ) = ( 1 , { \textstyle \frac { 1 } { 4 } } )$ . Similarly, for three variables, each linear equation determines a plane in three-dimensional space. When we intersect these planes, i.e., satisfy all linear equations at the same time, we can obtain a solution set that is a plane, a line, a point or empty (when the planes have no common intersection). 

For a systematic approach to solving systems of linear equations, we will introduce a useful compact notation. We collect the coefficients $a _ { i j }$ into vectors and collect the vectors into matrices. In other words, we write the system from (2.3) in the following form: 

$$
\left[ \begin{array}{c} a _ {1 1} \\ \vdots \\ a _ {m 1} \end{array} \right] x _ {1} + \left[ \begin{array}{c} a _ {1 2} \\ \vdots \\ a _ {m 2} \end{array} \right] x _ {2} + \dots + \left[ \begin{array}{c} a _ {1 n} \\ \vdots \\ a _ {m n} \end{array} \right] x _ {n} = \left[ \begin{array}{c} b _ {1} \\ \vdots \\ b _ {m} \end{array} \right]\tag{2.9}
$$

©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 

$$
\Longleftrightarrow \left[ \begin{array}{c c c} a _ {1 1} & \dots & a _ {1 n} \\ \vdots & & \vdots \\ a _ {m 1} & \dots & a _ {m n} \end{array} \right] \left[ \begin{array}{c} x _ {1} \\ \vdots \\ x _ {n} \end{array} \right] = \left[ \begin{array}{c} b _ {1} \\ \vdots \\ b _ {m} \end{array} \right].\tag{2.10}
$$

In the following, we will have a close look at these matrices and define computation rules. We will return to solving linear equations in Section 2.3. 

## 2.2 Matrices

Matrices play a central role in linear algebra. They can be used to compactly represent systems of linear equations, but they also represent linear functions (linear mappings) as we will see later in Section 2.7. Before we discuss some of these interesting topics, let us first define what a matrix is and what kind of operations we can do with matrices. We will see more properties of matrices in Chapter 4. 

matrix 

Definition 2.1 (Matrix). With $m , n \in \mathbb { N }$ a real-valued $( m , n )$ matrix A is an m n-tuple of elements $a _ { i j } , i = 1 , \ldots , m , j = 1 , \ldots , n$ , which is ordered according to a rectangular scheme consisting of m rows and n columns: 

$$
\boldsymbol {A} = \left[ \begin{array}{c c c c} a _ {1 1} & a _ {1 2} & \dots & a _ {1 n} \\ a _ {2 1} & a _ {2 2} & \dots & a _ {2 n} \\ \vdots & \vdots & & \vdots \\ a _ {m 1} & a _ {m 2} & \dots & a _ {m n} \end{array} \right], \quad a _ {i j} \in \mathbb {R}.\tag{2.11}
$$

row 

column row vector column vector Figure 2.4 By stacking its columns, a matrix A can be represented as a long vector a. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/ff1d2380204a7e59279ab02f7613a5acf98b7302a63bf1735205428e3e48b6a8.jpg)


Note the size of the matrices. 

np.einsum(’il, lj’, A, B) 

By convention $( 1 , n )$ -matrices are called rows and $( m , 1 )$ -matrices are called columns. These special matrices are also called row/column vectors. 

<sup>Rm×n</sup> is the set of all real-valued $( m , n )$ -matrices. $\pmb { A } \in \mathbb { R } ^ { m \times n }$ can be equivalently represented as $\pmb { a } \in \mathbb { R } ^ { m n }$ by stacking all n columns of the matrix into a long vector; see Figure 2.4. 

## 2.2.1 Matrix Addition and Multiplication

The sum of two matrices $\pmb { A } \in \mathbb { R } ^ { m \times n } , \pmb { B } \in \mathbb { R } ^ { m \times n }$ is defined as the elementwise sum, i.e., 

$$
\boldsymbol {A} + \boldsymbol {B} := \left[ \begin{array}{c c c} a _ {1 1} + b _ {1 1} & \dots & a _ {1 n} + b _ {1 n} \\ \vdots & & \vdots \\ a _ {m 1} + b _ {m 1} & \dots & a _ {m n} + b _ {m n} \end{array} \right] \in \mathbb {R} ^ {m \times n}.\tag{2.12}
$$

For matrices $\pmb { A } \in \mathbb { R } ^ { m \times n } , \pmb { B } \in \mathbb { R } ^ { n \times k }$ , the elements $c _ { i j }$ of the product $C = A B \in \mathbb { R } ^ { m \times k }$ are computed as 

$$
c _ {i j} = \sum_ {l = 1} ^ {n} a _ {i l} b _ {l j}, \quad i = 1, \dots , m, \quad j = 1, \dots , k.\tag{2.13}
$$

This means, to compute element $c _ { i j }$ we multiply the elements of the ith row of A with the jth column of B and sum them up. Later in Section 3.2, we will call this the dot product of the corresponding row and column. In cases, where we need to be explicit that we are performing multiplication, we use the notation $\pmb { A } \cdot \pmb { B }$ to denote multiplication (explicitly showing $^ { * , * } )$ 

Remark. Matrices can only be multiplied if their “neighboring” dimensions match. For instance, an $n ~ \times$ k-matrix A can be multiplied with a $k \times m \cdot$ matrix B, but only from the left side: 

There are n columns in A and n rows in B so that we can compute $a _ { i l } b _ { l j }$ for $l = 1 , \ldots , n .$ 

$$
\underbrace {A} _ {n \times k} \underbrace {B} _ {k \times m} = \underbrace {C} _ {n \times m}
$$

Commonly, the dot product between two vectors a, b is denoted by a<sup>⊤</sup>b or $\langle a , b \rangle$ 

(2.14) 

The product BA is not defined if m = n since the neighboring dimensions do not match. 

Remark. Matrix multiplication is not defined as an element-wise operation on matrix elements, $\mathbf { i . e . , } c _ { i j } \neq a _ { i j } b _ { i j }$ (even if the size of $A , B$ was chosen appropriately). This kind of element-wise multiplication often appears in programming languages when we multiply (multi-dimensional) arrays with each other, and is called a Hadamard product. 

Hadamard product 

## Example 2.3

$$
\text { For   } \boldsymbol {A} = \left[ \begin{array}{l l l} 1 & 2 & 3 \\ 3 & 2 & 1 \end{array} \right] \in \mathbb {R} ^ {2 \times 3},   \boldsymbol {B} = \left[ \begin{array}{l l} 0 & 2 \\ 1 & - 1 \\ 0 & 1 \end{array} \right] \in \mathbb {R} ^ {3 \times 2},   \text { we   obtain }
$$

$$
\boldsymbol {A} \boldsymbol {B} = \left[ \begin{array}{c c c} 1 & 2 & 3 \\ 3 & 2 & 1 \end{array} \right] \left[ \begin{array}{c c} 0 & 2 \\ 1 & - 1 \\ 0 & 1 \end{array} \right] = \left[ \begin{array}{c c} 2 & 3 \\ 2 & 5 \end{array} \right] \in \mathbb {R} ^ {2 \times 2},\tag{2.15}
$$

$$
\boldsymbol {B} \boldsymbol {A} = \left[ \begin{array}{c c} 0 & 2 \\ 1 & - 1 \\ 0 & 1 \end{array} \right] \left[ \begin{array}{c c c} 1 & 2 & 3 \\ 3 & 2 & 1 \end{array} \right] = \left[ \begin{array}{c c c} 6 & 4 & 2 \\ - 2 & 0 & 2 \\ 3 & 2 & 1 \end{array} \right] \in \mathbb {R} ^ {3 \times 3}.\tag{2.16}
$$

From this example, we can already see that matrix multiplication is not commutative, $\operatorname { i . e . , } A B \neq B A ;$ see also Figure 2.5 for an illustration. 

Definition 2.2 (Identity Matrix). In $\mathbb { R } ^ { n \times n }$ , we define the identity matrix 

$$
\boldsymbol {I} _ {n} := \left[ \begin{array}{c c c c c c} 1 & 0 & \dots & 0 & \dots & 0 \\ 0 & 1 & \dots & 0 & \dots & 0 \\ \vdots & \vdots & \ddots & \vdots & \ddots & \vdots \\ 0 & 0 & \dots & 1 & \dots & 0 \\ \vdots & \vdots & \ddots & \vdots & \ddots & \vdots \\ 0 & 0 & \dots & 0 & \dots & 1 \end{array} \right] \in \mathbb {R} ^ {n \times n}\tag{2.17}
$$

Figure 2.5 Even if both matrix multiplications AB and BA are defined, the dimensions of the results can be different. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/81ab05231006b9850e35bd336a31d992ef12218dda476237c2b5f018faa11f76.jpg)



identity matrix


©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 

as the n n-matrix containing 1 on the diagonal and 0 everywhere else. 

Now that we defined matrix multiplication, matrix addition and the identity matrix, let us have a look at some properties of matrices: 

Associativity: 

$$
\forall \boldsymbol {A} \in \mathbb {R} ^ {m \times n}, \boldsymbol {B} \in \mathbb {R} ^ {n \times p}, \boldsymbol {C} \in \mathbb {R} ^ {p \times q}: (\boldsymbol {A B}) \boldsymbol {C} = \boldsymbol {A} (\boldsymbol {B C})\tag{2.18}
$$

Distributivity: 

$$
\forall \boldsymbol {A}, \boldsymbol {B} \in \mathbb {R} ^ {m \times n}, \boldsymbol {C}, \boldsymbol {D} \in \mathbb {R} ^ {n \times p}: (\boldsymbol {A} + \boldsymbol {B}) \boldsymbol {C} = \boldsymbol {A C} + \boldsymbol {B C}\tag{2.19a}
$$

$$
\boldsymbol {A} (\boldsymbol {C} + \boldsymbol {D}) = \boldsymbol {A C} + \boldsymbol {A D}\tag{2.19b}
$$

Multiplication with the identity matrix: 

$$
\forall \boldsymbol {A} \in \mathbb {R} ^ {m \times n}: \boldsymbol {I} _ {m} \boldsymbol {A} = \boldsymbol {A} \boldsymbol {I} _ {n} = \boldsymbol {A}\tag{2.20}
$$

Note that ${ \pmb { I } } _ { m } \neq { \pmb { I } } _ { n }$ for m $\neq n$ 

## 2.2.2 Inverse and Transpose

Definition 2.3 (Inverse). Consider a square matrix $A \in \mathbb { R } ^ { n \times n }$ . Let matrix $B \in \mathbb { R } ^ { n \times n }$ have the property that $A B = I _ { n } = B A$ . B is called the inverse of A and denoted by $A ^ { - 1 }$ 

Unfortunately, not every matrix A possesses an inverse $A ^ { - 1 }$ . If this inverse does exist, A is called regular/invertible/nonsingular, otherwise singular/noninvertible. When the matrix inverse exists, it is unique. In Section 2.3, we will discuss a general way to compute the inverse of a matrix by solving a system of linear equations. 

Remark (Existence of the Inverse of a $2 \times 2 \cdot$ -matrix). Consider a matrix 

$$
\boldsymbol {A} := \left[ \begin{array}{c c} a _ {1 1} & a _ {1 2} \\ a _ {2 1} & a _ {2 2} \end{array} \right] \in \mathbb {R} ^ {2 \times 2}.\tag{2.21}
$$

If we multiply A with 

$$
\boldsymbol {A} ^ {\prime} := \left[ \begin{array}{c c} a _ {2 2} & - a _ {1 2} \\ - a _ {2 1} & a _ {1 1} \end{array} \right]\tag{2.22}
$$

we obtain 

$$
\boldsymbol {A} \boldsymbol {A} ^ {\prime} = \left[ \begin{array}{c c} a _ {1 1} a _ {2 2} - a _ {1 2} a _ {2 1} & 0 \\ 0 & a _ {1 1} a _ {2 2} - a _ {1 2} a _ {2 1} \end{array} \right] = (a _ {1 1} a _ {2 2} - a _ {1 2} a _ {2 1}) \boldsymbol {I}.\tag{2.23}
$$

Therefore, 

$$
\pmb {A} ^ {- 1} = \frac {1}{a _ {1 1} a _ {2 2} - a _ {1 2} a _ {2 1}} \left[ \begin{array}{c c} a _ {2 2} & - a _ {1 2} \\ - a _ {2 1} & a _ {1 1} \end{array} \right]\tag{2.24}
$$

if and only if $a _ { 1 1 } a _ { 2 2 } - a _ { 1 2 } a _ { 2 1 } \neq 0$ . In Section 4.1, we will see that $a _ { 1 1 } a _ { 2 2 } -$ $a _ { 1 2 } a _ { 2 1 }$ is the determinant of a $2 \times 2 .$ -matrix. Furthermore, we can generally use the determinant to check whether a matrix is invertible. 

## Example 2.4 (Inverse Matrix)

The matrices 

$$
\boldsymbol {A} = \left[ \begin{array}{c c c} 1 & 2 & 1 \\ 4 & 4 & 5 \\ 6 & 7 & 7 \end{array} \right], \quad \boldsymbol {B} = \left[ \begin{array}{c c c} - 7 & - 7 & 6 \\ 2 & 1 & - 1 \\ 4 & 5 & - 4 \end{array} \right]\tag{2.25}
$$

are inverse to each other since $A B = I = B A$ 

Definition 2.4 (Transpose). For $\pmb { A } \in \mathbb { R } ^ { m \times n }$ the matrix $B \in \mathbb { R } ^ { n \times m }$ with $b _ { i j } = a _ { j i }$ is called the transpose of A. We write $\pmb { B } = \pmb { A } ^ { \top }$ 

transpose 

In general, $A ^ { \top }$ can be obtained by writing the columns of A as the rows of $A ^ { \top }$ . The following are important properties of inverses and transposes: 

The main diagonal (sometimes called “principal diagonal”, “primary diagonal”, “leading diagonal”, 

$$
\pmb {A} \pmb {A} ^ {- 1} = \pmb {I} = \pmb {A} ^ {- 1} \pmb {A}\tag{2.26}
$$

$$
(A B) ^ {- 1} = B ^ {- 1} A ^ {- 1}
$$

or “major diagonal”) 

(2.27) 

of a matrix $\pmb { A }$ is the collection of entries 

$$
(\boldsymbol {A} + \boldsymbol {B}) ^ {- 1} \neq \boldsymbol {A} ^ {- 1} + \boldsymbol {B} ^ {- 1}\tag{2.28}
$$

$$
A _ {i j} \text {   where   } i = j.
$$

$$
\left(\boldsymbol {A} ^ {\top}\right) ^ {\top} = \boldsymbol {A}
$$

The scalar case of 

$$
\left(\boldsymbol {A} \boldsymbol {B}\right) ^ {\top} = \boldsymbol {B} ^ {\top} \boldsymbol {A} ^ {\top}\tag{2.29}
$$

$$
(2. 2 8) \text {   is   }\tag{2.30}
$$

$$
\frac {1}{2 + 4} = \frac {1}{6} \neq \frac {1}{2} + \frac {1}{4}.
$$

$$
\left(\boldsymbol {A} + \boldsymbol {B}\right) ^ {\top} = \boldsymbol {A} ^ {\top} + \boldsymbol {B} ^ {\top}\tag{2.31}
$$

Definition 2.5 (Symmetric Matrix). A matrix $A \in \mathbb { R } ^ { n \times n }$ is symmetric if $\pmb { A } = \pmb { A } ^ { \top }$ 

symmetric matrix 

Note that only $( n , n )$ -matrices can be symmetric. Generally, we call $( n , n )$ -matrices also square matrices because they possess the same number of rows and columns. Moreover, if A is invertible, then so is $A ^ { \top }$ , and $( A ^ { - 1 } ) ^ { \top } = ( A ^ { \top } ) ^ { - 1 } = : A ^ { - \top }$ 

Remark (Sum and Product of Symmetric Matrices). The sum of symmet ric matrices $\pmb { A } , \pmb { B } \in \mathbb { R } ^ { n \times n }$ is always symmetric. However, although their product is always defined, it is generally not symmetric: 

$$
\left[ \begin{array}{c c} 1 & 0 \\ 0 & 0 \end{array} \right] \left[ \begin{array}{c c} 1 & 1 \\ 1 & 1 \end{array} \right] = \left[ \begin{array}{c c} 1 & 1 \\ 0 & 0 \end{array} \right].\tag{2.32}
$$

## 2.2.3 Multiplication by a Scalar

Let us look at what happens to matrices when they are multiplied by a scalar $\lambda \in \mathbb { R }$ . Let $\pmb { A } \in \mathbb { R } ^ { m \times n }$ and $\lambda \in \mathbb { R }$ . Then $\lambda A = K , K _ { i j } = \lambda a _ { i j }$ Practically, λ scales each element of A. For $\lambda , \psi \in \mathbb { R } ,$ the following holds: 

associativity 

Associativity: 

$$
(\lambda \psi) \boldsymbol {C} = \lambda (\psi \boldsymbol {C}), \quad \boldsymbol {C} \in \mathbb {R} ^ {m \times n}
$$

$\lambda ( B C ) = ( \lambda B ) C = B ( \lambda C ) = ( B C ) \lambda , \quad B \in \mathbb { R } ^ { m \times n } , C \in \mathbb { R } ^ { n \times k } .$ 

Note that this allows us to move scalar values around. 

distributivity 

$( \lambda C ) ^ { \top } = C ^ { \top } \lambda ^ { \top } = C ^ { \top } \lambda = \lambda C ^ { \top }$ since $\lambda = \lambda ^ { \top }$ for all $\lambda \in \mathbb { R }$ 

Distributivity: 

$$
(\lambda + \psi) \boldsymbol {C} = \lambda \boldsymbol {C} + \psi \boldsymbol {C}, \quad \boldsymbol {C} \in \mathbb {R} ^ {m \times n}
$$

$$
\lambda (\boldsymbol {B} + \boldsymbol {C}) = \lambda \boldsymbol {B} + \lambda \boldsymbol {C}, \quad \boldsymbol {B}, \boldsymbol {C} \in \mathbb {R} ^ {m \times n}
$$

## Example 2.5 (Distributivity)

If we define 

$$
\boldsymbol {C} := \left[ \begin{array}{c c} 1 & 2 \\ 3 & 4 \end{array} \right],\tag{2.33}
$$

then for any $\lambda , \psi \in \mathbb { R }$ we obtain 

$$
(\lambda + \psi) \boldsymbol {C} = \left[ \begin{array}{c c} (\lambda + \psi) 1 & (\lambda + \psi) 2 \\ (\lambda + \psi) 3 & (\lambda + \psi) 4 \end{array} \right] = \left[ \begin{array}{c c} \lambda + \psi & 2 \lambda + 2 \psi \\ 3 \lambda + 3 \psi & 4 \lambda + 4 \psi \end{array} \right]
$$

$$
= \left[ \begin{array}{c c} \lambda & 2 \lambda \\ 3 \lambda & 4 \lambda \end{array} \right] + \left[ \begin{array}{c c} \psi & 2 \psi \\ 3 \psi & 4 \psi \end{array} \right] = \lambda C + \psi C.\tag{2.34a}
$$

(2.34b) 

## 2.2.4 Compact Representations of Systems of Linear Equations

If we consider the system of linear equations 

$$
\begin{array}{l} 2 x _ {1} + 3 x _ {2} + 5 x _ {3} = 1 \\ 4 x _ {1} - 2 x _ {2} - 7 x _ {3} = 8 \\ 9 x _ {1} + 5 x _ {2} - 3 x _ {3} = 2 \end{array}\tag{2.35}
$$

and use the rules for matrix multiplication, we can write this equation system in a more compact form as 

$$
\left[ \begin{array}{c c c} 2 & 3 & 5 \\ 4 & - 2 & - 7 \\ 9 & 5 & - 3 \end{array} \right] \left[ \begin{array}{c} x _ {1} \\ x _ {2} \\ x _ {3} \end{array} \right] = \left[ \begin{array}{c} 1 \\ 8 \\ 2 \end{array} \right].\tag{2.36}
$$

Note that $x _ { 1 }$ scales the first column, $x _ { 2 }$ the second one, and $x _ { 3 }$ the third one. 

Generally, a system of linear equations can be compactly represented in their matrix form as $\mathbf { \nabla } A \mathbf { \sigma } x = b ;$ see (2.3), and the product Ax is a (linear) combination of the columns of $\pmb { A }$ . We will discuss linear combinations in more detail in Section 2.5. 

Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 

## 2.3 Solving Systems of Linear Equations

In (2.3), we introduced the general form of an equation system, i.e., 

$$
\begin{array}{c} {a _ {1 1} x _ {1} + \dots + a _ {1 n} x _ {n} = b _ {1}} \\ {\vdots} \\ {a _ {m 1} x _ {1} + \dots + a _ {m n} x _ {n} = b _ {m},} \end{array}\tag{2.37}
$$

where $a _ { i j } \in \mathbb { R }$ and $b _ { i } \in \mathbb { R }$ are known constants and $x _ { j }$ are unknowns, $i = 1 , \ldots , m , j = 1 , \ldots , n$ . Thus far, we saw that matrices can be used as a compact way of formulating systems of linear equations so that we can write $\mathbf { { A } } \mathbf { { x } } = \mathbf { { b } } ,$ see (2.10). Moreover, we defined basic matrix operations, such as addition and multiplication of matrices. In the following, we will focus on solving systems of linear equations and provide an algorithm for finding the inverse of a matrix. 

## 2.3.1 Particular and General Solution

Before discussing how to generally solve systems of linear equations, let us have a look at an example. Consider the system of equations 

$$
\left[ \begin{array}{c c c c} 1 & 0 & 8 & - 4 \\ 0 & 1 & 2 & 1 2 \end{array} \right] \left[ \begin{array}{c} x _ {1} \\ x _ {2} \\ x _ {3} \\ x _ {4} \end{array} \right] = \left[ \begin{array}{c} 4 2 \\ 8 \end{array} \right].\tag{2.38}
$$

The system has two equations and four unknowns. Therefore, in general we would expect infinitely many solutions. This system of equations is in a particularly easy form, where the first two columns consist of a 1 and a 0. Remember that we want to find scalars $x _ { 1 } , \ldots , x _ { 4 } $ , such that $\textstyle \sum _ { i = 1 } ^ { 4 } x _ { i } c _ { i } = b ,$ where we define $c _ { i }$ to be the ith column of the matrix and b the right-hand-side of (2.38). A solution to the problem in (2.38) can be found immediately by taking 42 times the first column and 8 times the second column so that 

$$
\boldsymbol {b} = \left[ \begin{array}{c} 4 2 \\ 8 \end{array} \right] = 4 2 \left[ \begin{array}{c} 1 \\ 0 \end{array} \right] + 8 \left[ \begin{array}{c} 0 \\ 1 \end{array} \right].\tag{2.39}
$$

Therefore, a solution is $[ 4 2 , 8 , 0 , 0 ] ^ { \top }$ . This solution is called a particular solution or special solution. However, this is not the only solution of this system of linear equations. To capture all the other solutions, we need to be creative in generating 0 in a non-trivial way using the columns of the matrix: Adding 0 to our special solution does not change the special solution. To do so, we express the third column using the first two columns (which are of this very simple form) 

$$
\left[ \begin{array}{c} 8 \\ 2 \end{array} \right] = 8 \left[ \begin{array}{c} 1 \\ 0 \end{array} \right] + 2 \left[ \begin{array}{c} 0 \\ 1 \end{array} \right]\tag{2.40}
$$

©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 

general solution 

so that ${ \bf 0 } = 8 { { c } _ { 1 } } + 2 { { c } _ { 2 } } - 1 { { c } _ { 3 } } + 0 { { c } _ { 4 } }$ and $( x _ { 1 } , x _ { 2 } , x _ { 3 } , x _ { 4 } ) = ( 8 , 2 , - 1 , 0 )$ . In fact, any scaling of this solution by $\lambda _ { 1 } \in \mathbb { R }$ produces the 0 vector, i.e., 

$$
\left[ \begin{array}{c c c c} 1 & 0 & 8 & - 4 \\ 0 & 1 & 2 & 1 2 \end{array} \right] \left(\lambda_ {1} \left[ \begin{array}{c} 8 \\ 2 \\ - 1 \\ 0 \end{array} \right]\right) = \lambda_ {1} (8 \boldsymbol {c} _ {1} + 2 \boldsymbol {c} _ {2} - \boldsymbol {c} _ {3}) = \boldsymbol {0}.\tag{2.41}
$$

Following the same line of reasoning, we express the fourth column of the matrix in (2.38) using the first two columns and generate another set of non-trivial versions of 0 as 

$$
\left[ \begin{array}{c c c c} 1 & 0 & 8 & - 4 \\ 0 & 1 & 2 & 1 2 \end{array} \right] \left(\lambda_ {2} \left[ \begin{array}{c} - 4 \\ 1 2 \\ 0 \\ - 1 \end{array} \right]\right) = \lambda_ {2} (- 4 \boldsymbol {c} _ {1} + 1 2 \boldsymbol {c} _ {2} - \boldsymbol {c} _ {4}) = \mathbf {0}\tag{2.42}
$$

for any $\lambda _ { 2 } \in \mathbb { R }$ . Putting everything together, we obtain all solutions of the equation system in (2.38), which is called the general solution, as the set 

$$
\left\{\boldsymbol {x} \in \mathbb {R} ^ {4}: \boldsymbol {x} = \left[ \begin{array}{c} 4 2 \\ 8 \\ 0 \\ 0 \end{array} \right] + \lambda_ {1} \left[ \begin{array}{c} 8 \\ 2 \\ - 1 \\ 0 \end{array} \right] + \lambda_ {2} \left[ \begin{array}{c} - 4 \\ 1 2 \\ 0 \\ - 1 \end{array} \right], \lambda_ {1}, \lambda_ {2} \in \mathbb {R} \right\}.\tag{2.43}
$$

Remark. The general approach we followed consisted of the following three steps: 

1. Find a particular solution to $\mathbf { A } { \boldsymbol { \mathbf { \mathit { x } } } } = \mathbf { \mathit { b } }$ 

2. Find all solutions to $\pmb { A } \pmb { x } = \mathbf { 0 }$ 

3. Combine the solutions from steps 1. and 2. to the general solution. 

Neither the general nor the particular solution is unique. 

♢ 

The system of linear equations in the preceding example was easy to solve because the matrix in (2.38) has this particularly convenient form, which allowed us to find the particular and the general solution by inspection. However, general equation systems are not of this simple form. Fortunately, there exists a constructive algorithmic way of transforming any system of linear equations into this particularly simple form: Gaussian elimination. Key to Gaussian elimination are elementary transformations of systems of linear equations, which transform the equation system into a simple form. Then, we can apply the three steps to the simple form that we just discussed in the context of the example in (2.38). 

## 2.3.2 Elementary Transformations

Key to solving a system of linear equations are elementary transformations that keep the solution set the same, but that transform the equation system into a simpler form: 

Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 

## 2.3 Solving Systems of Linear Equations

Exchange of two equations (rows in the matrix representing the system of equations) 

Multiplication of an equation (row) with a constant $\lambda \in \mathbb { R } \backslash \{ 0 \}$ 

Addition of two equations (rows) 

## Example 2.6

For $a \in \mathbb { R } ,$ , we seek all solutions of the following system of equations: 

$$
\begin{array}{ccccccccc}- 2 x _ {1} & + & 4 x _ {2} & - & 2 x _ {3} & - & x _ {4} & + & 4 x _ {5} & = & - 3 \\ 4 x _ {1} & - & 8 x _ {2} & + & 3 x _ {3} & - & 3 x _ {4} & + & x _ {5} & = & 2 \\ x _ {1} & - & 2 x _ {2} & + & x _ {3} & - & x _ {4} & + & x _ {5} & = & 0 \\ x _ {1} & - & 2 x _ {2} & & & - & 3 x _ {4} & + & 4 x _ {5} & = & a \\ \end{array} .\tag{2.44}
$$

We start by converting this system of equations into the compact matrix notation $A x = b .$ We no longer mention the variables x explicitly and build the augmented matrix (in the form $\left[ A \left| b \right| \right)$ 

$$
\left[ \begin{array}{c c c c c c} - 2 & 4 & - 2 & - 1 & 4 & - 3 \\ 4 & - 8 & 3 & - 3 & 1 & 2 \\ 1 & - 2 & 1 & - 1 & 1 & 0 \\ 1 & - 2 & 0 & - 3 & 4 & a \end{array} \right] \text {Swap with R_{3}}
$$

where we used the vertical line to separate the left-hand side from the right-hand side in (2.44). We use <sup>⇝</sup> to indicate a transformation of the augmented matrix using elementary transformations. 

Swapping Rows 1 and 3 leads to 

$$
\left[ \begin{array}{r r r r r r} 1 & - 2 & 1 & - 1 & 1 & 0 \\ 4 & - 8 & 3 & - 3 & 1 & 2 \\ - 2 & 4 & - 2 & - 1 & 4 & - 3 \\ 1 & - 2 & 0 & - 3 & 4 & a \end{array} \right] \begin{array}{l} - 4 R _ {1} \\ + 2 R _ {1} \\ - R _ {1} \end{array}
$$

The augmented matrix -A | b compactly represents the system of linear equations Ax = b. 

When we now apply the indicated transformations $( \boldsymbol { \mathrm { e . g . } }$ , subtract Row 1 four times from Row 2), we obtain 

$$
\left[ \begin{array}{c c c c c c} 1 & - 2 & 1 & - 1 & 1 & 0 \\ 0 & 0 & - 1 & 1 & - 3 & 2 \\ 0 & 0 & 0 & - 3 & 6 & - 3 \\ 0 & 0 & - 1 & - 2 & 3 & a \end{array} \right] - R _ {2} - R _ {3}
$$

$$
\rightsquigarrow \quad \left[\begin{array}{c c c c c c}1&- 2&1&- 1&1&0\\0&0&- 1&1&- 3&2\\0&0&0&- 3&6&- 3\\0&0&0&0&0&a + 1\end{array}\right] \cdot (- 1) \cdot (- \frac {1}{3})
$$

$$
\rightsquigarrow \quad \left[\begin{array}{c c c c c c}1&- 2&1&- 1&1&0\\0&0&1&- 1&3&- 2\\0&0&0&1&- 2&1\\0&0&0&0&0&a + 1\end{array}\right]
$$

©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 

basic variable free variable 

This (augmented) matrix is in a convenient form, the row-echelon form (REF). Reverting this compact notation back into the explicit notation with the variables we seek, we obtain 

$$
\begin{array}{r c l r c l r c l r c l} x _ {1} & - & 2 x _ {2} & + & x _ {3} & - & x _ {4} & + & x _ {5} & = & 0 \\ & & & & x _ {3} & - & x _ {4} & + & 3 x _ {5} & = & - 2 \\ & & & & & & x _ {4} & - & 2 x _ {5} & = & 1 \\ & & & & & & & & 0 & = & a + 1 \end{array}\tag{2.45}
$$

Only for $a = - 1$ this system can be solved. A particular solution is 

$$
{\left[ \begin{array}{l} x _ {1} \\ x _ {2} \\ x _ {3} \\ x _ {4} \\ x _ {5} \end{array} \right]} = {\left[ \begin{array}{l} 2 \\ 0 \\ - 1 \\ 1 \\ 0 \end{array} \right]}\tag{2.46}
$$

The general solution, which captures the set of all possible solutions, is 

$$
\left\{\boldsymbol {x} \in \mathbb {R} ^ {5}: \boldsymbol {x} = \left[ \begin{array}{c} 2 \\ 0 \\ - 1 \\ 1 \\ 0 \end{array} \right] + \lambda_ {1} \left[ \begin{array}{c} 2 \\ 1 \\ 0 \\ 0 \\ 0 \end{array} \right] + \lambda_ {2} \left[ \begin{array}{c} 2 \\ 0 \\ - 1 \\ 2 \\ 1 \end{array} \right], \quad \lambda_ {1}, \lambda_ {2} \in \mathbb {R} \right\}.\tag{2.47}
$$

In the following, we will detail a constructive way to obtain a particular and general solution of a system of linear equations. 

Remark (Pivots and Staircase Structure). The leading coefficient of a row (first nonzero number from the left) is called the pivot and is always strictly to the right of the pivot of the row above it. Therefore, any equation system in row-echelon form always has a “staircase” structure. 

Definition 2.6 (Row-Echelon Form). A matrix is in row-echelon form if 

All rows that contain only zeros are at the bottom of the matrix; correspondingly, all rows that contain at least one nonzero element are on top of rows that contain only zeros. 

Looking at nonzero rows only, the first nonzero number from the left (also called the pivot or the leading coefficient) is always strictly to the right of the pivot of the row above it. 

Remark (Basic and Free Variables). The variables corresponding to the pivots in the row-echelon form are called basic variables and the other variables are free variables. For example, in (2.45), $x _ { 1 } , x _ { 3 } , x _ { 4 }$ are basic variables, whereas $x _ { 2 } , x _ { 5 }$ are free variables. ◇ 

Remark (Obtaining a Particular Solution). The row-echelon form makes 

## 2.3 Solving Systems of Linear Equations

our lives easier when we need to determine a particular solution. To do this, we express the right-hand side of the equation system using the pivot columns, such that $\begin{array} { r } { \pmb { b } = \sum _ { i = 1 } ^ { P } \lambda _ { i } \pmb { p } _ { i } } \end{array}$ , where $p _ { i } , i = 1 , \ldots , P$ , are the pivot columns. The $\lambda _ { i }$ are determined easiest if we start with the rightmost pivot column and work our way to the left. 

In the previous example, we would try to find $\lambda _ { 1 } , \lambda _ { 2 } , \lambda _ { 3 }$ so that 

$$
\lambda_ {1} \left[ \begin{array}{c} 1 \\ 0 \\ 0 \\ 0 \end{array} \right] + \lambda_ {2} \left[ \begin{array}{c} 1 \\ 1 \\ 0 \\ 0 \end{array} \right] + \lambda_ {3} \left[ \begin{array}{c} - 1 \\ - 1 \\ 1 \\ 0 \end{array} \right] = \left[ \begin{array}{c} 0 \\ - 2 \\ 1 \\ 0 \end{array} \right].\tag{2.48}
$$

From here, we find relatively directly that $\lambda _ { 3 } = 1 , \lambda _ { 2 } = - 1 , \lambda _ { 1 } = 2$ . When we put everything together, we must not forget the non-pivot columns for which we set the coefficients implicitly to 0. Therefore, we get the particular solution $\pmb { x } = [ 2 , 0 , - 1 , 1 , 0 ] ^ { \top }$ 

Remark (Reduced Row Echelon Form). An equation system is in reduced row-echelon form (also: row-reduced echelon form or row canonical form) if 

It is in row-echelon form. 

Every pivot is 1. 

The pivot is the only nonzero entry in its column. 

The reduced row-echelon form will play an important role later in Section 2.3.3 because it allows us to determine the general solution of a system of linear equations in a straightforward way. 

Remark (Gaussian Elimination). Gaussian elimination is an algorithm that performs elementary transformations to bring a system of linear equations into reduced row-echelon form. 

## Example 2.7 (Reduced Row Echelon Form)

Verify that the following matrix is in reduced row-echelon form (the pivots are in bold): 

$$
\boldsymbol {A} = \left[ \begin{array}{c c c c c} \mathbf {1} & 3 & 0 & 0 & 3 \\ 0 & 0 & \mathbf {1} & 0 & 9 \\ 0 & 0 & 0 & \mathbf {1} & - 4 \end{array} \right].\tag{2.49}
$$

The key idea for finding the solutions of $\pmb { A } \pmb { x } = \pmb { 0 }$ is to look at the nonpivot columns, which we will need to express as a (linear) combination of the pivot columns. The reduced row echelon form makes this relatively straightforward, and we express the non-pivot columns in terms of sums and multiples of the pivot columns that are on their left: The second column is 3 times the first column (we can ignore the pivot columns on the right of the second column). Therefore, to obtain 0, we need to subtract 



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



the second column from three times the first column. Now, we look at the fifth column, which is our second non-pivot column. The fifth column can be expressed as 3 times the first pivot column, 9 times the second pivot column, and 4 times the third pivot column. We need to keep track of the indices of the pivot columns and translate this into 3 times the first column, 0 times the second column (which is a non-pivot column), 9 times the third column (which is our second pivot column), and 4 times the fourth column (which is the third pivot column). Then we need to subtract the fifth column to obtain 0. In the end, we are still solving a homogeneous equation system. 

To summarize, all solutions of $\pmb { A } \pmb { x } = \mathbf { 0 } , \pmb { x } \in \mathbb { R } ^ { 5 }$ are given by 

$$
\left\{\boldsymbol {x} \in \mathbb {R} ^ {5}: \boldsymbol {x} = \lambda_ {1} \left[ \begin{array}{c} 3 \\ - 1 \\ 0 \\ 0 \\ 0 \end{array} \right] + \lambda_ {2} \left[ \begin{array}{c} 3 \\ 0 \\ 9 \\ - 4 \\ - 1 \end{array} \right], \quad \lambda_ {1}, \lambda_ {2} \in \mathbb {R} \right\}.\tag{2.50}
$$

## 2.3.3 The Minus-1 Trick

In the following, we introduce a practical trick for reading out the solutions x of a homogeneous system of linear equations $\ \mathbf { A } { \boldsymbol { x } } \ = \ \mathbf { 0 } ;$ , where $\pmb { A } \in \mathbb { R } ^ { k \times n } , \pmb { x } \in \mathbb { R } ^ { n }$ 

To start, we assume that A is in reduced row-echelon form without any rows that just contain zeros, i.e., 

$$
\boldsymbol {A} = \left[ \begin{array}{c c c c c c c c c c c c c c c} 0 & \dots & 0 & \mathbf {1} & * & \dots & * & 0 & * & \dots & * & 0 & * & \dots & * \\ \vdots & & \vdots & 0 & 0 & \dots & 0 & \mathbf {1} & * & \dots & * & \vdots & \vdots & & \vdots \\ \vdots & & \vdots & \vdots & \vdots & & \vdots & 0 & \vdots & & \vdots & \vdots & \vdots & & \vdots \\ \vdots & & \vdots & \vdots & \vdots & & \vdots & \vdots & \vdots & & \vdots & 0 & \vdots & & \vdots \\ 0 & \dots & 0 & 0 & 0 & \dots & 0 & 0 & 0 & \dots & 0 & \mathbf {1} & * & \dots & * \end{array} \right],\tag{2.51}
$$

where can be an arbitrary real number, with the constraints that the first nonzero entry per row must be 1 and all other entries in the corresponding column must be 0. The columns $j _ { 1 } , \dots , j _ { k }$ with the pivots (marked in bold) are the standard unit vectors $e _ { 1 } , \ldots , e _ { k } \in \mathbb { R } ^ { k }$ . We extend this matrix to an $n \times n$ -matrix A<sup>˜</sup> by adding $n - k$ rows of the form 

$$
\left[ \begin{array}{c c c c c c c} 0 & \dots & 0 & - 1 & 0 & \dots & 0 \end{array} \right]\tag{2.52}
$$

so that the diagonal of the augmented matrix $\tilde { \boldsymbol { A } }$ contains either 1 $. \ \mathrm { o r } - 1$ Then, the columns of A<sup>˜</sup> that contain the 1 as pivots are solutions of 

## 2.3 Solving Systems of Linear Equations

the homogeneous equation system $\mathbf { { \boldsymbol { A } } } \mathbf { { \boldsymbol { x } } } = \mathbf { { \boldsymbol { 0 } } }$ . To be more precise, these columns form a basis (Section 2.6.1) of the solution space of $\mathbf { { \boldsymbol { A } } } \mathbf { { \boldsymbol { x } } } = \mathbf { { \boldsymbol { 0 } } } $ which we will later call the kernel or null space (see Section 2.7.3). 

## Example 2.8 (Minus-1 Trick)

Let us revisit the matrix in (2.49), which is already in reduced REF: 

$$
\boldsymbol {A} = \left[ \begin{array}{c c c c c} 1 & 3 & 0 & 0 & 3 \\ 0 & 0 & 1 & 0 & 9 \\ 0 & 0 & 0 & 1 & - 4 \end{array} \right].\tag{2.53}
$$

We now augment this matrix to a $5 \times 5$ matrix by adding rows of the form (2.52) at the places where the pivots on the diagonal are missing and obtain 

$$
\tilde {\boldsymbol {A}} = \left[ \begin{array}{c c c c c} 1 & 3 & 0 & 0 & 3 \\ 0 & - \mathbf {1} & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 & 9 \\ 0 & 0 & 0 & 1 & - 4 \\ 0 & 0 & 0 & 0 & - \mathbf {1} \end{array} \right].\tag{2.54}
$$

From this form, we can immediately read out the solutions of $\mathbf { A } \mathbf { { x } } = \mathbf { 0 }$ by taking the columns of ${ \tilde { \cal A } } _ { i }$ , which contain 1 on the diagonal: 

$$
\left\{\boldsymbol {x} \in \mathbb {R} ^ {5}: \boldsymbol {x} = \lambda_ {1} \left[ \begin{array}{c} 3 \\ - 1 \\ 0 \\ 0 \\ 0 \end{array} \right] + \lambda_ {2} \left[ \begin{array}{c} 3 \\ 0 \\ 9 \\ - 4 \\ - 1 \end{array} \right], \quad \lambda_ {1}, \lambda_ {2} \in \mathbb {R} \right\},\tag{2.55}
$$

which is identical to the solution in (2.50) that we obtained by “insight”. 

## Calculating the Inverse

To compute the inverse $A ^ { - 1 }$ of $\pmb { A } \in \mathbb { R } ^ { n \times n }$ , we need to find a matrix X that satisfies $A X \ = \ I _ { n }$ . Then, $\pmb { X } = \pmb { A } ^ { - 1 }$ . We can write this down as a set of simultaneous linear equations $A X \ = \ I _ { n }$ , where we solve for $\pmb { X } = \left[ \pmb { x } _ { 1 } | \cdot \cdot \cdot | \pmb { x } _ { n } \right]$ . We use the augmented matrix notation for a compact representation of this set of systems of linear equations and obtain 

$$
\left[ \boldsymbol {A} | \boldsymbol {I} _ {n} \right] \quad \rightsquigarrow \dots \rightsquigarrow \quad \left[ \boldsymbol {I} _ {n} | \boldsymbol {A} ^ {- 1} \right].\tag{2.56}
$$

This means that if we bring the augmented equation system into reduced row-echelon form, we can read out the inverse on the right-hand side of the equation system. Hence, determining the inverse of a matrix is equivalent to solving systems of linear equations. 

©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 

Example 2.9 (Calculating an Inverse Matrix by Gaussian Elimination) To determine the inverse of 

$$
\boldsymbol {A} = \left[ \begin{array}{c c c c} 1 & 0 & 2 & 0 \\ 1 & 1 & 0 & 0 \\ 1 & 2 & 0 & 1 \\ 1 & 1 & 1 & 1 \end{array} \right]\tag{2.57}
$$

we write down the augmented matrix 

$$
\left[ \begin{array}{c c c c c c c c} 1 & 0 & 2 & 0 & 1 & 0 & 0 & 0 \\ 1 & 1 & 0 & 0 & 0 & 1 & 0 & 0 \\ 1 & 2 & 0 & 1 & 0 & 0 & 1 & 0 \\ 1 & 1 & 1 & 1 & 0 & 0 & 0 & 1 \end{array} \right]
$$

and use Gaussian elimination to bring it into reduced row-echelon form 

$$
\left[ \begin{array}{c c c c c c c c} 1 & 0 & 0 & 0 & - 1 & 2 & - 2 & 2 \\ 0 & 1 & 0 & 0 & 1 & - 1 & 2 & - 2 \\ 0 & 0 & 1 & 0 & 1 & - 1 & 1 & - 1 \\ 0 & 0 & 0 & 1 & - 1 & 0 & - 1 & 2 \end{array} \right],
$$

such that the desired inverse is given as its right-hand side: 

$$
\boldsymbol {A} ^ {- 1} = \left[ \begin{array}{c c c c} - 1 & 2 & - 2 & 2 \\ 1 & - 1 & 2 & - 2 \\ 1 & - 1 & 1 & - 1 \\ - 1 & 0 & - 1 & 2 \end{array} \right].\tag{2.58}
$$

We can verify that (2.58) is indeed the inverse by performing the multiplication $A \bar { A } ^ { - 1 }$ and observing that we recover $\pmb { I } _ { 4 }$ 

## 2.3.4 Algorithms for Solving a System of Linear Equations

In the following, we briefly discuss approaches to solving a system of linear equations of the form $\mathbf { } A \mathbf { } x = b$ . We make the assumption that a solution exists. Should there be no solution, we need to resort to approximate solutions, which we do not cover in this chapter. One way to solve the approximate problem is using the approach of linear regression, which we discuss in detail in Chapter 9. 

In special cases, we may be able to determine the inverse $A ^ { - 1 }$ , such that the solution of $A x \ = \ b$ is given as $x = A ^ { - 1 } b$ . However, this is only possible if A is a square matrix and invertible, which is often not the case. Otherwise, under mild assumptions (i.e., A needs to have linearly independent columns) we can use the transformation 

$$
\boldsymbol {A} \boldsymbol {x} = \boldsymbol {b} \iff \boldsymbol {A} ^ {\top} \boldsymbol {A} \boldsymbol {x} = \boldsymbol {A} ^ {\top} \boldsymbol {b} \iff \boldsymbol {x} = (\boldsymbol {A} ^ {\top} \boldsymbol {A}) ^ {- 1} \boldsymbol {A} ^ {\top} \boldsymbol {b}\tag{2.59}
$$

Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 

and use the Moore-Penrose pseudo-inverse $( A ^ { \top } A ) ^ { - 1 } A ^ { \top }$ to determine the solution (2.59) that solves $\mathbf { { A } } \mathbf { { x } } = \mathbf { { b } } ,$ which also corresponds to the minimum norm least-squares solution. A disadvantage of this approach is that it requires many computations for the matrix-matrix product and computing the inverse of $A ^ { \top } A$ . Moreover, for reasons of numerical precision it is generally not recommended to compute the inverse or pseudo-inverse. In the following, we therefore briefly discuss alternative approaches to solving systems of linear equations. 

Gaussian elimination plays an important role when computing determinants (Section 4.1), checking whether a set of vectors is linearly independent (Section 2.5), computing the inverse of a matrix (Section 2.2.2), computing the rank of a matrix (Section 2.6.2), and determining a basis of a vector space (Section 2.6.1). Gaussian elimination is an intuitive and constructive way to solve a system of linear equations with thousands of variables. However, for systems with millions of variables, it is impractical as the required number of arithmetic operations scales cubically in the number of simultaneous equations. 

In practice, systems of many linear equations are solved indirectly, by either stationary iterative methods, such as the Richardson method, the Jacobi method, the Gauß-Seidel method, and the successive over-relaxation method, or Krylov subspace methods, such as conjugate gradients, generalized minimal residual, or biconjugate gradients. We refer to the books by Stoer and Burlirsch (2002), Strang (2003), and Liesen and Mehrmann (2015) for further details. 

Let $^ { \mathbf { \delta x } }$ be a solution of $\mathbf { } A \mathbf { } x = b$ . The key idea of these iterative methods is to set up an iteration of the form 

$$
\pmb {x} ^ {(k + 1)} = \pmb {C x} ^ {(k)} + \pmb {d}\tag{2.60}
$$

for suitable C and d that reduces the residual error $\| \pmb { x } ^ { ( k + 1 ) } - \pmb { x } _ { * } \|$ in every iteration and converges to ${ \pmb x } _ { \ast }$ . We will introduce norms $\| \cdot \| .$ , which allow us to compute similarities between vectors, in Section 3.1. 

## 2.4 Vector Spaces

Thus far, we have looked at systems of linear equations and how to solve them (Section 2.3). We saw that systems of linear equations can be compactly represented using matrix-vector notation (2.10). In the following, we will have a closer look at vector spaces, i.e., a structured space in which vectors live. 

In the beginning of this chapter, we informally characterized vectors as objects that can be added together and multiplied by a scalar, and they remain objects of the same type. Now, we are ready to formalize this, and we will start by introducing the concept of a group, which is a set of elements and an operation defined on these elements that keeps some structure of the set intact. 

©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 

## 2.4.1 Groups

Groups play an important role in computer science. Besides providing a fundamental framework for operations on sets, they are heavily used in cryptography, coding theory, and graphics. 

Definition 2.7 (Group). Consider a set $\mathcal { G }$ and an operation $\otimes : { \mathcal { G } } \times { \mathcal { G } } \to { \mathcal { G } }$ defined on $\mathcal { G }$ . Then $G : = ( \mathcal { G } , \otimes )$ is called a group if the following hold: 

1. Closure of under $\otimes : \forall x , y \in { \mathcal { G } } : x \otimes y \in { \mathcal { G } }$ 

2. Associativity: x, $y , z \in \mathcal { G } : ( x \otimes y ) \otimes z = x \otimes ( y \otimes z )$ 

3. Neutral element: $\exists e \in { \mathcal { G } } \forall x \in { \mathcal { G } } : x \otimes e = x { \mathrm { ~ a n d ~ } } e \otimes x = x$ 

4. Inverse element: x $\in \mathcal { G } \exists y \in \mathcal { G } : x \otimes y = e$ and $y \otimes x = e ,$ , where e is the neutral element. We often write $x ^ { - 1 }$ to denote the inverse element of $x .$ . 

Remark. The inverse element is defined with respect to the operation $\otimes$ and does not necessarily mean $\textstyle { \frac { 1 } { x } }$ $\diamondsuit$ 

If additionally x, $y \in \mathcal { G } : x \otimes y = y \otimes x .$ , then $G = ( \mathcal { G } , \otimes )$ is an Abelian group (commutative). 

## Example 2.10 (Groups)

Let us have a look at some examples of sets with associated operations and see whether they are groups: 

- $( \mathbb { Z } , + )$ is an Abelian group. 

<sup>N</sup> <sub>0</sub> := <sup>N</sup> ∪ {0} 

$( \mathbb { N } _ { 0 } , + )$ is not a group: Although $( \mathbb { N } _ { 0 } , + )$ possesses a neutral element (0), the inverse elements are missing. 

$( \mathbb { Z } , \cdot )$ is not a group: Although $( \mathbb { Z } , \cdot )$ contains a neutral element (1), the inverse elements for any $z \in \mathbb { Z } , z \neq \pm 1$ , are missing. 

■ $( \mathbb { R } , \cdot )$ is not a group since 0 does not possess an inverse element. 

$( \mathbb { R } \backslash \{ 0 \} , \cdot )$ is Abelian. 

$( \mathbb { R } ^ { n } , + ) , ( \mathbb { Z } ^ { n } , + ) , n \in \mathbb { N }$ are Abelian if + is defined componentwise, i.e., 

$$
(x _ {1}, \dots , x _ {n}) + (y _ {1}, \dots , y _ {n}) = (x _ {1} + y _ {1}, \dots , x _ {n} + y _ {n}).\tag{2.61}
$$

Then, $( x _ { 1 } , \cdot \cdot \cdot , x _ { n } ) ^ { - 1 } : = ( - x _ { 1 } , \cdot \cdot \cdot , - x _ { n } )$ is the inverse element and $e = ( 0 , \cdots , 0 )$ is the neutral element. 

$\left( \mathbb { R } ^ { m \times n } , + \right)$ , the set of m n-matrices is Abelian (with componentwise addition as defined in (2.61)). 

Let us have a closer look at $( \mathbb { R } ^ { n \times n } , \cdot )$ , i.e., the set of n n-matrices with matrix multiplication as defined in (2.13). 

– Closure and associativity follow directly from the definition of matrix multiplication. 

– Neutral element: The identity matrix ${ \cal { I } } _ { n }$ is the neutral element with respect to matrix multiplication “ ” in $( \mathbb { R } ^ { n \times n } , \cdot )$ 

– Inverse element: If the inverse exists (A is regular), then $A ^ { - 1 }$ is the inverse element of $A \in \mathbb { R } ^ { n \times n }$ and in exactly this case $( \mathbb { R } ^ { n \times n } , \cdot )$ is a group, called the general linear group. 

Definition 2.8 (General Linear Group). The set of regular (invertible) matrices $\pmb { A } \in \mathbb { R } ^ { n \times n }$ is a group with respect to matrix multiplication as defined in (2.13) and is called general linear group $G L ( n , \mathbb { R } )$ . However, since matrix multiplication is not commutative, the group is not Abelian. 

## 2.4.2 Vector Spaces

When we discussed groups, we looked at sets $\mathcal { G }$ and inner operations on $\mathcal { G } , \mathrm { i . e . }$ , mappings ${ \mathcal { G } } \times { \mathcal { G } } \to { \mathcal { G } }$ that only operate on elements in . In the following, we will consider sets that in addition to an inner operation $+$ also contain an outer operation , the multiplication of a vector $\boldsymbol { x } \in \mathcal G$ by a scalar $\lambda \in \mathbb { R }$ . We can think of the inner operation as a form of addition, and the outer operation as a form of scaling. Note that the inner/outer operations have nothing to do with inner/outer products. 

Definition 2.9 (Vector Space). A real-valued vector space $V = ( \nu , + , \cdot )$ is a set with two operations 

$$
+: \mathcal {V} \times \mathcal {V} \rightarrow \mathcal {V}\tag{2.62}
$$

$$
\cdot : \mathbb {R} \times \mathcal {V} \to \mathcal {V}\tag{2.63}
$$

where 

1. $( \nu , + )$ is an Abelian group 

2. Distributivity: 

$$
1. \forall \lambda \in \mathbb {R}, \boldsymbol {x}, \boldsymbol {y} \in \mathcal {V}: \lambda \cdot (\boldsymbol {x} + \boldsymbol {y}) = \lambda \cdot \boldsymbol {x} + \lambda \cdot \boldsymbol {y}
$$

$$
2. \forall \lambda , \psi \in \mathbb {R}, \boldsymbol {x} \in \mathcal {V}: (\lambda + \psi) \cdot \boldsymbol {x} = \lambda \cdot \boldsymbol {x} + \psi \cdot \boldsymbol {x}
$$

3. Associativity (outer operation): $\forall \lambda , \psi \in \mathbb { R } , \pmb { x } \in \mathcal { V } : \lambda \cdot ( \psi \cdot \pmb { x } ) = ( \lambda \psi ) \cdot \pmb { x }$ 

4. Neutral element with respect to the outer operation: $\forall x \in \mathcal { V } : 1 \cdot x = x$ 

The elements $\mathbf { \pmb { x } } \in \mathbf { \Xi } V$ are called vectors. The neutral element of $( \nu , + )$ is the zero vector $\mathbf { 0 } = [ 0 , \ldots , 0 ] ^ { \top }$ , and the inner operation + is called vector addition. The elements $\lambda \in \mathbb { R }$ are called scalars and the outer operation is a multiplication by scalars. Note that a scalar product is something different, and we will get to this in Section 3.2. 

Remark. A “vector multiplication” $\mathbf { \Delta } a b , \mathbf { \Delta } a , b \in \mathbb { R } ^ { n }$ , is not defined. Theoret ically, we could define an element-wise multiplication, such that ${ \pmb { c } } = { \pmb { a } } { \pmb { b } }$ with $c _ { j } = a _ { j } b _ { j }$ . This “array multiplication” is common to many programming languages but makes mathematically limited sense using the standard rules for matrix multiplication: By treating vectors as $n \times 1$ matrices (which we usually do), we can use the matrix multiplication as defined in (2.13). However, then the dimensions of the vectors do not match. Only the following multiplications for vectors are defined: $\mathbf { \boldsymbol { a } } \mathbf { \boldsymbol { b } } ^ { \top } \in \mathbb { R } ^ { n \times n }$ (outer product), ${ \pmb a } ^ { \top } { \pmb b } \in \mathbb { R }$ (inner/scalar/dot product). ◇ 

## Example 2.11 (Vector Spaces)

Let us have a look at some important examples: 

$\mathcal { V } = \mathbb { R } ^ { n } , n \in \mathbb { N }$ is a vector space with operations defined as follows: 

$$
\boldsymbol {x} + \boldsymbol {y} = (x _ {1}, \dots , x _ {n}) + (y _ {1}, \dots , y _ {n}) = (x _ {1} + y _ {1}, \dots , x _ {n} + y _ {n})
$$

$$
\boldsymbol {y} \in \mathbb {R} ^ {n}
$$

– Multiplication by scalars: $\lambda { \pmb x } = \lambda ( x _ { 1 } , \ldots , x _ { n } ) = ( \lambda x _ { 1 } , \ldots , \lambda x _ { n } )$ for all $\lambda \in \mathbb { R } , x \in \mathbb { R } ^ { n }$ 

V ${ \bf \Phi } ^ { \prime } = \mathbb { R } ^ { m \times n } , m , n \in \mathbb { N }$ is a vector space with 

– Addition: $A + B = \left[ \begin{array} { c c c } { a _ { 1 1 } + b _ { 1 1 } } & { \cdots } & { a _ { 1 n } + b _ { 1 n } } \\ { \vdots } & { } & { \vdots } \\ { a _ { m 1 } + b _ { m 1 } } & { \cdots } & { a _ { m n } + b _ { m n } } \end{array} \right]$ is defined elementwise for all $A , B \in \mathcal { V }$ 

– Multiplication by scalars: $\lambda \pmb { A } = \left[ \begin{array} { c c c } { \lambda a _ { 1 1 } } & { \cdots } & { \lambda a _ { 1 n } } \\ { \vdots } & { } & { \vdots } \\ { \lambda a _ { m 1 } } & { \cdots } & { \lambda a _ { m n } } \end{array} \right]$ as defined in Section 2.2. Remember that $\mathbb { R } ^ { m \times n }$ is equivalent to $\mathbb { R } ^ { m n }$ 

$\gamma = \mathbb { C } .$ , with the standard definition of addition of complex numbers. 

Remark. In the following, we will denote a vector space $( \nu , + , \cdot )$ by V when + and are the standard vector addition and scalar multiplication. Moreover, we will use the notation $\textbf { \textit { x } } \in \textbf { \textit { V } }$ for vectors in to simplify notation. ◇ 

Remark. The vector spaces $\mathbb { R } ^ { n } , \mathbb { R } ^ { n \times 1 } , \mathbb { R } ^ { 1 \times n }$ are only different in the way we write vectors. In the following, we will not make a distinction between $\mathbb { R } ^ { n }$ and $\mathbb { R } ^ { n \times 1 }$ , which allows us to write n-tuples as column vectors 

$$
\boldsymbol {x} = \left[ \begin{array}{c} x _ {1} \\ \vdots \\ x _ {n} \end{array} \right].\tag{2.64}
$$

This simplifies the notation regarding vector space operations. However, we do distinguish between $\mathbb { R } ^ { n \times 1 }$ and $\mathbb { R } ^ { 1 \times n }$ (the row vectors) to avoid confusion with matrix multiplication. By default, we write x to denote a column vector, and a row vector is denoted by $\pmb { x } ^ { \top }$ , the transpose of $\mathbf { \pmb { x } } . \quad \begin{array} { r l } \end{array} \qquad \mathrm { \diamondsuit }$ 

## 2.4.3 Vector Subspaces

In the following, we will introduce vector subspaces. Intuitively, they are sets contained in the original vector space with the property that when we perform vector space operations on elements within this subspace, we will never leave it. In this sense, they are “closed”. Vector subspaces are a key idea in machine learning. For example, Chapter 10 demonstrates how to use vector subspaces for dimensionality reduction. 

Definition 2.10 (Vector Subspace). Let $V = ( \nu , + , \cdot )$ be a vector space and $\mathcal { U } \subseteq \mathcal { V } , \mathcal { U } \neq \emptyset$ . Then $U = ( \mathcal { U } , + , \cdot )$ is called vector subspace of $V$ (or linear subspace) if U is a vector space with the vector space operations + and restricted to $\boldsymbol { \mathcal { U } } \times \boldsymbol { \mathcal { U } }$ and $\mathbb { R } \times \mathcal { U }$ . We write $U \subseteq V$ to denote a subspace U of V . 

If $\mathcal { U } \subseteq \mathcal { V }$ and V is a vector space, then U naturally inherits many properties directly from V because they hold for all $\mathbf { \boldsymbol { x } } \in \mathcal { V } ,$ , and in particular for all $\pmb { x } \in \mathcal { U } \subseteq \mathcal { V }$ . This includes the Abelian group properties, the distributivity, the associativity and the neutral element. To determine whether $( \mathcal { U } , + , \cdot )$ is a subspace of V we still do need to show 

1. $\mathcal { U } \neq \emptyset ,$ in particular: $\mathbf { 0 } \in \mathcal { U }$ 

2. Closure of U : 

a. With respect to the outer operation: $\forall \lambda \in \mathbb { R } \forall x \in \mathcal { U } : \lambda \pmb { x } \in \mathcal { U } .$ 

b. With respect to the inner operation: $\forall x , y \in \mathcal { U } : x + y \in \mathcal { U } .$ 

## Example 2.12 (Vector Subspaces)

Let us have a look at some examples: 

For every vector space $V ,$ the trivial subspaces are V itself and 0 . 

Only example D in Figure 2.6 is a subspace of $\mathbb { R } ^ { 2 }$ (with the usual inner/ outer operations). In A and C, the closure property is violated; B does not contain 0. 

The solution set of a homogeneous system of linear equations $\pmb { A } \pmb { x } = \mathbf { 0 }$ with n unknowns $\pmb { x } = [ x _ { 1 } , \ldots , x _ { n } ] ^ { \top }$ is a subspace of $\mathbb { R } ^ { n }$ 

The solution of an inhomogeneous system of linear equations ${ \pmb A } { \pmb x } =$ b, $\pm \mathbf { 0 }$ is not a subspace of <sup>Rn</sup>. 

The intersection of arbitrarily many subspaces is a subspace itself. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/efb19954cef66f4a4ccd62964b65c757ec53547aadabbd6935d9a02025fec1b9.jpg)



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020).



Figure 2.6 Not all subsets of <sup>R2</sup> are subspaces. In A and C, the closure property is violated; B does not contain 0. Only D is a subspace.


Remark. Every subspace $U \subseteq ( \mathbb { R } ^ { n } , + , \cdot )$ is the solution space of a homogeneous system of linear equations $\pmb { A } \pmb { x } = \mathbf { 0 }$ for $\pmb { x } \in \mathbb { R } ^ { n }$ ◇ 

## 2.5 Linear Independence

In the following, we will have a close look at what we can do with vectors (elements of the vector space). In particular, we can add vectors together and multiply them with scalars. The closure property guarantees that we end up with another vector in the same vector space. It is possible to find a set of vectors with which we can represent every vector in the vector space by adding them together and scaling them. This set of vectors is a basis, and we will discuss them in Section 2.6.1. Before we get there, we will need to introduce the concepts of linear combinations and linear independence. 

Definition 2.11 (Linear Combination). Consider a vector space V and a finite number of vectors $\pmb { x } _ { 1 } , \dotsc , \pmb { x } _ { k } \in V$ . Then, every $v \in V$ of the form 

$$
\boldsymbol {v} = \lambda_ {1} \boldsymbol {x} _ {1} + \dots + \lambda_ {k} \boldsymbol {x} _ {k} = \sum_ {i = 1} ^ {k} \lambda_ {i} \boldsymbol {x} _ {i} \in V\tag{2.65}
$$

with $\lambda _ { 1 } , \ldots , \lambda _ { k } \in \mathbb { R }$ is a linear combination of the vectors $\pmb { x } _ { 1 } , \ldots , \pmb { x } _ { k }$ 

The 0-vector can always be written as the linear combination of k vectors $\pmb { x } _ { 1 } , \ldots , \pmb { x } _ { k }$ because $\mathbf { 0 } = \textstyle \sum _ { i = 1 } ^ { k }$ 0x is always true. In the following, we are interested in non-trivial linear combinations of a set of vectors to represent 0, i.e., linear combinations of vectors $\pmb { x } _ { 1 } , \ldots , \pmb { x } _ { k }$ , where not all coefficients λ in (2.65) are 0. 

Definition 2.12 (Linear (In)dependence). Let us consider a vector space V with $k \in \mathbb { N }$ and $\pmb { x } _ { 1 } , \dotsc , \pmb { x } _ { k } \in V$ . If there is a non-trivial linear combination, such that $\begin{array} { r } { \mathbf { 0 } = \sum _ { i = 1 } ^ { k } \lambda _ { i } \mathbf { x } _ { i } } \end{array}$ with at least one $\lambda _ { i } \neq 0 .$ , the vectors $\pmb { x } _ { 1 } , \ldots , \pmb { x } _ { k }$ are linearly dependent. If only the trivial solution exists, i.e., $\lambda _ { 1 } = . . . = \lambda _ { k } = 0$ the vectors $\pmb { x } _ { 1 } , \ldots , \pmb { x } _ { k }$ are linearly independent. 

Linear independence is one of the most important concepts in linear algebra. Intuitively, a set of linearly independent vectors consists of vectors that have no redundancy, i.e., if we remove any of those vectors from the set, we will lose something. Throughout the next sections, we will formalize this intuition more. 

## Example 2.13 (Linearly Dependent Vectors)

A geographic example may help to clarify the concept of linear independence. A person in Nairobi (Kenya) describing where Kigali (Rwanda) is might say ,“You can get to Kigali by first going 506 km Northwest to Kampala (Uganda) and then 374 km Southwest.”. This is sufficient information to describe the location of Kigali because the geographic coordinate system may be considered a two-dimensional vector space (ignoring altitude and the Earth’s curved surface). The person may add, “It is about 751 km West of here.” Although this last statement is true, it is not necessary to find Kigali given the previous information (see Figure 2.7 for an illustration). In this example, the “506 km Northwest” vector (blue) and the “374 km Southwest” vector (purple) are linearly independent. This means the Southwest vector cannot be described in terms of the Northwest vector, and vice versa. However, the third “751 km West” vector (black) is a linear combination of the other two vectors, and it makes the set of vectors linearly dependent. Equivalently, given “751 km West” and “374 km Southwest” can be linearly combined to obtain “506 km Northwest”. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/3f678f4022f5ea883a956fdb3265808e7991133b5abc9c25a223c2a9425ccc4d.jpg)



Figure 2.7 Geographic example (with crude approximations to cardinal directions) of linearly dependent vectors in a two-dimensional space (plane).


Remark. The following properties are useful to find out whether vectors are linearly independent: 

k vectors are either linearly dependent or linearly independent. There is no third option. 

If at least one of the vectors $\pmb { x } _ { 1 } , \ldots , \pmb { x } _ { k }$ is 0 then they are linearly dependent. The same holds if two vectors are identical. 

The vectors $\{ \pmb { x } _ { 1 } , \dots , \pmb { x } _ { k } : \pmb { x } _ { i } \neq \pmb { 0 } , i = 1 , \dots , k \} , k \geqslant 2 ,$ , are linearly dependent if and only if (at least) one of them is a linear combination of the others. In particular, if one vector is a multiple of another vector, i.e., $\pmb { x } _ { i } = \lambda \pmb { x } _ { j } , \lambda \in \mathbb { R }$ then the set $\{ \pmb { x } _ { 1 } , \dots , \pmb { x } _ { k } : \pmb { x } _ { i } \neq \mathbf { 0 } , i = 1 , \dots , k \}$ is linearly dependent. 

A practical way of checking whether vectors $\pmb { x } _ { 1 } , \dotsc , \pmb { x } _ { k } \in V$ are linearly independent is to use Gaussian elimination: Write all vectors as columns of a matrix A and perform Gaussian elimination until the matrix is in row echelon form (the reduced row-echelon form is unnecessary here): 

– The pivot columns indicate the vectors, which are linearly independent of the vectors on the left. Note that there is an ordering of vectors when the matrix is built. 

– The non-pivot columns can be expressed as linear combinations of the pivot columns on their left. For instance, the row-echelon form 

$$
\left[ \begin{array}{c c c} 1 & 3 & 0 \\ 0 & 0 & 2 \end{array} \right]\tag{2.66}
$$

tells us that the first and third columns are pivot columns. The second column is a non-pivot column because it is three times the first column. 

All column vectors are linearly independent if and only if all columns are pivot columns. If there is at least one non-pivot column, the columns (and, therefore, the corresponding vectors) are linearly dependent. 

## Example 2.14

Consider $\mathbb { R } ^ { 4 }$ with 

$$
\boldsymbol {x} _ {1} = \left[ \begin{array}{c} 1 \\ 2 \\ - 3 \\ 4 \end{array} \right], \quad \boldsymbol {x} _ {2} = \left[ \begin{array}{c} 1 \\ 1 \\ 0 \\ 2 \end{array} \right], \quad \boldsymbol {x} _ {3} = \left[ \begin{array}{c} - 1 \\ - 2 \\ 1 \\ 1 \end{array} \right].\tag{2.67}
$$

To check whether they are linearly dependent, we follow the general approach and solve 

$$
\lambda_ {1} \boldsymbol {x} _ {1} + \lambda_ {2} \boldsymbol {x} _ {2} + \lambda_ {3} \boldsymbol {x} _ {3} = \lambda_ {1} \left[ \begin{array}{c} 1 \\ 2 \\ - 3 \\ 4 \end{array} \right] + \lambda_ {2} \left[ \begin{array}{c} 1 \\ 1 \\ 0 \\ 2 \end{array} \right] + \lambda_ {3} \left[ \begin{array}{c} - 1 \\ - 2 \\ 1 \\ 1 \end{array} \right] = \mathbf {0}\tag{2.68}
$$

for $\lambda _ { 1 } , \ldots , \lambda _ { 3 }$ . We write the vectors $\mathbf { { x } } _ { i } , i = 1 , 2 , 3 .$ , as the columns of a matrix and apply elementary row operations until we identify the pivot columns: 

$$
\left[\begin{array}{c c c}1&1&- 1\\2&1&- 2\\- 3&0&1\\4&2&1\end{array}\right] \quad \rightsquigarrow \dots \rightsquigarrow \quad \left[\begin{array}{c c c}1&1&- 1\\0&1&0\\0&0&1\\0&0&0\end{array}\right].\tag{2.69}
$$

Here, every column of the matrix is a pivot column. Therefore, there is no non-trivial solution, and we require $\lambda _ { 1 } = 0 , \lambda _ { 2 } = 0 , \lambda _ { 3 } = 0$ to solve the equation system. Hence, the vectors ${ \pmb x } _ { 1 } , { \pmb x } _ { 2 } , { \pmb x } _ { 3 }$ are linearly independent. 



Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 



Remark. Consider a vector space V with k linearly independent vectors $\boldsymbol { b } _ { 1 } , \ldots , \boldsymbol { b } _ { k }$ and m linear combinations 

$$
\begin{array}{c} \boldsymbol {x} _ {1} = \sum_ {i = 1} ^ {k} \lambda_ {i 1} \boldsymbol {b} _ {i}, \\ \vdots \end{array}\tag{2.70}
$$

$$
\boldsymbol {x} _ {m} = \sum_ {i = 1} ^ {k} \lambda_ {i m} \boldsymbol {b} _ {i}.
$$

Defining $B = [ b _ { 1 } , \ldots , b _ { k } ]$ as the matrix whose columns are the linearly independent vectors $b _ { 1 } , \ldots , b _ { k }$ , we can write 

$$
\boldsymbol {x} _ {j} = \boldsymbol {B} \boldsymbol {\lambda} _ {j}, \quad \boldsymbol {\lambda} _ {j} = \left[ \begin{array}{c} \lambda_ {1 j} \\ \vdots \\ \lambda_ {k j} \end{array} \right], \quad j = 1, \ldots , m,\tag{2.71}
$$

in a more compact form. 

We want to test whether $\pmb { x } _ { 1 } , \ldots , \pmb { x } _ { m }$ are linearly independent. For this purpose, we follow the general approach of testing when $\begin{array} { r } { \sum _ { j = 1 } ^ { m } \psi _ { j } \pmb { x } _ { j } = \pmb { 0 } } \end{array}$ With (2.71), we obtain 

$$
\sum_ {j = 1} ^ {m} \psi_ {j} \boldsymbol {x} _ {j} = \sum_ {j = 1} ^ {m} \psi_ {j} \boldsymbol {B} \boldsymbol {\lambda} _ {j} = \boldsymbol {B} \sum_ {j = 1} ^ {m} \psi_ {j} \boldsymbol {\lambda} _ {j}.\tag{2.72}
$$

This means that $\{ \pmb { x } _ { 1 } , \ldots , \pmb { x } _ { m } \}$ are linearly independent if and only if the column vectors $\{ \lambda _ { 1 } , \ldots , \lambda _ { m } \}$ are linearly independent. 

Remark. In a vector space $V ,$ , m linear combinations of k vectors $\pmb { x } _ { 1 } , \ldots , \pmb { x } _ { k }$ are linearly dependent if m $> k$ 

## Example 2.15

Consider a set of linearly independent vectors $b _ { 1 } , b _ { 2 } , b _ { 3 } , b _ { 4 } \in \mathbb { R } ^ { n }$ and 

$$
\begin{array}{r c l r c l r c l} \boldsymbol {x} _ {1} & = & \boldsymbol {b} _ {1} & - & 2 \boldsymbol {b} _ {2} & + & \boldsymbol {b} _ {3} & - & \boldsymbol {b} _ {4} \\ \boldsymbol {x} _ {2} & = & - 4 \boldsymbol {b} _ {1} & - & 2 \boldsymbol {b} _ {2} & & & + & 4 \boldsymbol {b} _ {4} \\ \boldsymbol {x} _ {3} & = & 2 \boldsymbol {b} _ {1} & + & 3 \boldsymbol {b} _ {2} & - & \boldsymbol {b} _ {3} & - & 3 \boldsymbol {b} _ {4} \\ \boldsymbol {x} _ {4} & = & 1 7 \boldsymbol {b} _ {1} & - & 1 0 \boldsymbol {b} _ {2} & + & 1 1 \boldsymbol {b} _ {3} & + & \boldsymbol {b} _ {4} \end{array} .\tag{2.73}
$$

Are the vectors $\pmb { x } _ { 1 } , \dotsc , \pmb { x } _ { 4 } \in \mathbb { R } ^ { n }$ linearly independent? To answer this question, we investigate whether the column vectors 

$$
\left\{\left[ \begin{array}{c} 1 \\ - 2 \\ 1 \\ - 1 \end{array} \right], \left[ \begin{array}{c} - 4 \\ - 2 \\ 0 \\ 4 \end{array} \right], \left[ \begin{array}{c} 2 \\ 3 \\ - 1 \\ - 3 \end{array} \right], \left[ \begin{array}{c} 1 7 \\ - 1 0 \\ 1 1 \\ 1 \end{array} \right] \right\}\tag{2.74}
$$



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



are linearly independent. The reduced row-echelon form of the corresponding linear equation system with coefficient matrix 

$$
\boldsymbol {A} = \left[ \begin{array}{c c c c} 1 & - 4 & 2 & 1 7 \\ - 2 & - 2 & 3 & - 1 0 \\ 1 & 0 & - 1 & 1 1 \\ - 1 & 4 & - 3 & 1 \end{array} \right]\tag{2.75}
$$

is given as 

$$
\left[ \begin{array}{c c c c} 1 & 0 & 0 & - 7 \\ 0 & 1 & 0 & - 1 5 \\ 0 & 0 & 1 & - 1 8 \\ 0 & 0 & 0 & 0 \end{array} \right].\tag{2.76}
$$

We see that the corresponding linear equation system is non-trivially solvable: The last column is not a pivot column, and $\pmb { x } _ { 4 } = - 7 \pmb { x } _ { 1 } - 1 5 \pmb { x } _ { 2 } - 1 8 \pmb { x } _ { 3 }$ Therefore, $\pmb { x } _ { 1 } , \ldots , \pmb { x } _ { 4 }$ are linearly dependent as $\mathbf { x } _ { 4 }$ can be expressed as a linear combination of $\pmb { x } _ { 1 } , \ldots , \pmb { x } _ { 3 }$ 

## 2.6 Basis and Rank

In a vector space $V ,$ we are particularly interested in sets of vectors $\mathcal { A }$ that possess the property that any vector $v \in V$ can be obtained by a linear combination of vectors in ${ \mathcal { A } } .$ These vectors are special vectors, and in the following, we will characterize them. 

## 2.6.1 Generating Set and Basis

Definition 2.13 (Generating Set and Span). Consider a vector space $V =$ $( \nu , + , \cdot )$ and set of vectors $\mathcal { A } = \{ \pmb { x } _ { 1 } , \dots , \pmb { x } _ { k } \} \subseteq \mathcal { V }$ . If every vector ${ \pmb v } \in$ $\nu$ can be expressed as a linear combination of $\pmb { x } _ { 1 } , \dots , \pmb { x } _ { k } , \pmb { \mathcal { A } }$ is called a generating set of $V .$ The set of all linear combinations of vectors in $\mathcal { A }$ is called the span of . If spans the vector space $V _ { s }$ , we write $V = \operatorname { s p a n } [ A ]$ or $V = \operatorname { s p a n } [ \pmb { x } _ { 1 } , \dots , \pmb { x } _ { k } ]$ 

Generating sets are sets of vectors that span vector (sub)spaces, i.e., every vector can be represented as a linear combination of the vectors in the generating set. Now, we will be more specific and characterize the smallest generating set that spans a vector (sub)space. 

Definition 2.14 (Basis). Consider a vector space $V = ( \nu , + , \cdot )$ and ${ \mathcal { A } } \subseteq$ . A generating set of V is called minimal if there exists no smaller set $\tilde { \mathcal { A } } \subsetneq \mathcal { A } \subseteq \mathcal { V }$ that spans V . Every linearly independent generating set of $V$ is minimal and is called a basis of $V$ 

## 2.6 Basis and Rank

Let $V = ( \nu , + , \cdot )$ be a vector space and $B \subseteq \gamma , B \neq \varnothing$ . Then, the following statements are equivalent: 

$\boldsymbol { B }$ is a basis of $V$ . 

$\boldsymbol { B }$ is a minimal generating set. 

■ $\boldsymbol { B }$ is a maximal linearly independent set of vectors in $V , \mathrm { i . e . }$ ., adding any other vector to this set will make it linearly dependent. 

Every vector $\pmb { x } \in V$ is a linear combination of vectors from $B ,$ , and every linear combination is unique, i.e., with 

$$
\boldsymbol {x} = \sum_ {i = 1} ^ {k} \lambda_ {i} \boldsymbol {b} _ {i} = \sum_ {i = 1} ^ {k} \psi_ {i} \boldsymbol {b} _ {i}\tag{2.77}
$$

and $\lambda _ { i } , \psi _ { i } \in \mathbb { R } , b _ { i } \in \mathcal { B }$ it follows that $\lambda _ { i } = \psi _ { i } , i = 1 , \ldots , k .$ 

## Example 2.16

In $\mathbb { R } ^ { 3 }$ , the canonical/standard basis is 

$$
\mathcal {B} = \left\{\left[ \begin{array}{c} 1 \\ 0 \\ 0 \end{array} \right], \left[ \begin{array}{c} 0 \\ 1 \\ 0 \end{array} \right], \left[ \begin{array}{c} 0 \\ 0 \\ 1 \end{array} \right] \right\}.\tag{2.78}
$$

Different bases in $\mathbb { R } ^ { 3 }$ are 

$$
\mathcal {B} _ {1} = \left\{\left[ \begin{array}{l} 1 \\ 0 \\ 0 \end{array} \right], \left[ \begin{array}{l} 1 \\ 1 \\ 0 \end{array} \right], \left[ \begin{array}{l} 1 \\ 1 \\ 1 \end{array} \right] \right\}, \mathcal {B} _ {2} = \left\{\left[ \begin{array}{l} 0. 5 \\ 0. 8 \\ 0. 4 \end{array} \right], \left[ \begin{array}{l} 1. 8 \\ 0. 3 \\ 0. 3 \end{array} \right], \left[ \begin{array}{l} - 2. 2 \\ - 1. 3 \\ 3. 5 \end{array} \right] \right\}.\tag{2.79}
$$

The set 

$$
\mathcal {A} = \left\{\left[ \begin{array}{c} 1 \\ 2 \\ 3 \\ 4 \end{array} \right], \left[ \begin{array}{c} 2 \\ - 1 \\ 0 \\ 2 \end{array} \right], \left[ \begin{array}{c} 1 \\ 1 \\ 0 \\ - 4 \end{array} \right] \right\}\tag{2.80}
$$

is linearly independent, but not a generating set (and no basis) of $\mathbb { R } ^ { 4 }$ For instance, the vector $[ 1 , 0 , 0 , 0 ] ^ { - }$ cannot be obtained by a linear combination of elements in . 

Remark. Every vector space V possesses a basis . The preceding examples show that there can be many bases of a vector space $V ,$ , i.e., there is no unique basis. However, all bases possess the same number of elements, the basis vectors. 

We only consider finite-dimensional vector spaces $V$ . In this case, the dimension of V is the number of basis vectors of $V _ { i }$ , and we write dim(V ). If $U \subseteq V$ is a subspace of $V ,$ then dim $( U ) \leqslant \dim ( V )$ and $\dim ( U ) =$ 

The dimension of a vector space corresponds to the number of its basi vectors. 

dim(V ) if and only if $U = V$ . Intuitively, the dimension of a vector space can be thought of as the number of independent directions in this vector space. 

Remark. The dimension of a vector space is not necessarily the number of elements in a vector. For instance, the vector space $V = \operatorname { s p a n } [ \left[ 0 \right] ]$ is one-dimensional, although the basis vector possesses two elements. $\diamondsuit$ Remark. A basis of a subspace $U = \operatorname { s p a n } [ \pmb { x } _ { 1 } , \ldots , \pmb { x } _ { m } ] \subseteq \mathbb { R } ^ { n }$ can be found by executing the following steps: 

1. Write the spanning vectors as columns of a matrix A 

2. Determine the row-echelon form of A. 

3. The spanning vectors associated with the pivot columns are a basis of $U .$ 

## Example 2.17 (Determining a Basis)

For a vector subspace $U \subseteq \mathbb { R } ^ { 5 }$ , spanned by the vectors 

$$
\boldsymbol {x} _ {1} = \left[ \begin{array}{c} 1 \\ 2 \\ - 1 \\ - 1 \\ - 1 \end{array} \right], \quad \boldsymbol {x} _ {2} = \left[ \begin{array}{c} 2 \\ - 1 \\ 1 \\ 2 \\ - 2 \end{array} \right], \quad \boldsymbol {x} _ {3} = \left[ \begin{array}{c} 3 \\ - 4 \\ 3 \\ 5 \\ - 3 \end{array} \right], \quad \boldsymbol {x} _ {4} = \left[ \begin{array}{c} - 1 \\ 8 \\ - 5 \\ - 6 \\ 1 \end{array} \right] \in \mathbb {R} ^ {5},\tag{2.81}
$$

we are interested in finding out which vectors $\pmb { x } _ { 1 } , \ldots , \pmb { x } _ { 4 }$ are a basis for U . For this, we need to check whether $\pmb { x } _ { 1 } , \ldots , \pmb { x } _ { 4 }$ are linearly independent. Therefore, we need to solve 

$$
\sum_ {i = 1} ^ {4} \lambda_ {i} \pmb {x} _ {i} = \mathbf {0},\tag{2.82}
$$

which leads to a homogeneous system of equations with matrix 

$$
\left[ \boldsymbol {x} _ {1}, \boldsymbol {x} _ {2}, \boldsymbol {x} _ {3}, \boldsymbol {x} _ {4} \right] = \left[ \begin{array}{c c c c} 1 & 2 & 3 & - 1 \\ 2 & - 1 & - 4 & 8 \\ - 1 & 1 & 3 & - 5 \\ - 1 & 2 & 5 & - 6 \\ - 1 & - 2 & - 3 & 1 \end{array} \right].\tag{2.83}
$$

With the basic transformation rules for systems of linear equations, we obtain the row-echelon form 

$$
\left[\begin{array}{r r r r}1&2&3&- 1\\2&- 1&- 4&8\\- 1&1&3&- 5\\- 1&2&5&- 6\\- 1&- 2&- 3&1\end{array}\right] \quad \rightsquigarrow \dots \rightsquigarrow \quad \left[\begin{array}{r r r r}1&2&3&- 1\\0&1&2&- 2\\0&0&0&1\\0&0&0&0\\0&0&0&0\end{array}\right].
$$

Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 

Since the pivot columns indicate which set of vectors is linearly independent, we see from the row-echelon form that ${ \pmb x } _ { 1 } , { \pmb x } _ { 2 } , { \pmb x } _ { 4 }$ are linearly independent (because the system of linear equations $\lambda _ { 1 } \pmb { x } _ { 1 } + \lambda _ { 2 } \pmb { x } _ { 2 } + \lambda _ { 4 } \pmb { x } _ { 4 } = \mathbf { 0 }$ can only be solved with $\lambda _ { 1 } = \lambda _ { 2 } = \lambda _ { 4 } = 0 )$ . Therefore, $\{ \pmb { x } _ { 1 } , \pmb { x } _ { 2 } , \pmb { x } _ { 4 } \}$ is a basis of U . 

## 2.6.2 Rank

The number of linearly independent columns of a matrix $\pmb { { A } } \in \mathbb { R } ^ { m \times n }$ equals the number of linearly independent rows and is called the rank of A and is denoted by rk(A). 

Remark. The rank of a matrix has some important properties: 

$\operatorname { r k } ( A ) = \operatorname { r k } ( A ^ { \top } )$ , i.e., the column rank equals the row rank. 

The columns of $\pmb { A } \in \mathbb { R } ^ { m \times n }$ span a subspace $U \subseteq \mathbb { R } ^ { m }$ with $\dim ( U ) =$ $\operatorname { r k } ( A )$ . Later we will call this subspace the image or range. A basis of U can be found by applying Gaussian elimination to A to identify the pivot columns. 

The rows of $\pmb { { A } } \in \mathbb { R } ^ { m \times n }$ span a subspace $W \subseteq \mathbb { R } ^ { n }$ with $\dim ( W ) =$ $\operatorname { r k } ( A )$ . A basis of W can be found by applying Gaussian elimination to $A ^ { \top }$ ⊤ 

For all $\pmb { A } \in \mathbb { R } ^ { n \times n }$ it holds that A is regular (invertible) if and only if $\operatorname { r k } ( A ) = n$ 

For all $\pmb { { A } } \in \mathbb { R } ^ { m \times n }$ and all $\pmb { b } \in \mathbb { R } ^ { m }$ it holds that the linear equation system $\mathbf { A } { \boldsymbol { \mathbf { \mathit { x } } } } = \mathbf { \boldsymbol { \mathit { b } } }$ can be solved if and only if $\operatorname { r k } ( A ) = \operatorname { r k } ( A | b )$ , where A b denotes the augmented system. 

For $\pmb { A } \in \mathbb { R } ^ { m \times n }$ the subspace of solutions for $\pmb { A } \pmb { x } = \mathbf { 0 }$ possesses dimension $n - \operatorname { r k } ( A )$ . Later, we will call this subspace the kernel or the null space. 

A matrix $\pmb { A } \in \mathbb { R } ^ { m \times n }$ has full rank if its rank equals the largest possible rank for a matrix of the same dimensions. This means that the rank of a full-rank matrix is the lesser of the number of rows and columns, i.e., $\operatorname { r k } ( A ) = \operatorname* { m i n } ( m , n )$ . A matrix is said to be rank deficient if it does not have full rank. 

Example 2.18 (Rank) 

$$
\boldsymbol {A} = \left[ \begin{array}{l l l} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 0 & 0 & 0 \end{array} \right].
$$

A has two linearly independent rows/columns so that rk $( A ) = 2 .$ 



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



$$
\boldsymbol {A} = \left[ \begin{array}{c c c} 1 & 2 & 1 \\ - 2 & - 3 & 1 \\ 3 & 5 & 0 \end{array} \right].
$$

We use Gaussian elimination to determine the rank: 

$$
\left[\begin{array}{c c c}1&2&1\\- 2&- 3&1\\3&5&0\end{array}\right] \quad \rightsquigarrow \dots \rightsquigarrow \quad \left[\begin{array}{c c c}1&2&1\\0&1&3\\0&0&0\end{array}\right]\tag{2.84}
$$

Here, we see that the number of linearly independent rows and columns is 2, such that rk $( A ) = 2$ 

## 2.7 Linear Mappings

In the following, we will study mappings on vector spaces that preserve their structure, which will allow us to define the concept of a coordinate. In the beginning of the chapter, we said that vectors are objects that can be added together and multiplied by a scalar, and the resulting object is still a vector. We wish to preserve this property when applying the mapping: Consider two real vector spaces $V , W$ . A mapping Φ $: V  W$ preserves the structure of the vector space if 

$$
\Phi (\boldsymbol {x} + \boldsymbol {y}) = \Phi (\boldsymbol {x}) + \Phi (\boldsymbol {y})\tag{2.85}
$$

$$
\Phi (\lambda \boldsymbol {x}) = \lambda \Phi (\boldsymbol {x})\tag{2.86}
$$

for all $\mathbf { { } } x , y \in \ { \mathrm { ~ V ~ } }$ and $\lambda \in \mathbb { R }$ . We can summarize this in the following definition: 

Definition 2.15 (Linear Mapping). For vector spaces $V , W ,$ , a mapping $\Phi : V \to W$ is called a linear mapping (or vector space homomorphism/ linear transformation) if 

$$
\forall \boldsymbol {x}, \boldsymbol {y} \in V \forall \lambda , \psi \in \mathbb {R}: \Phi (\lambda \boldsymbol {x} + \psi \boldsymbol {y}) = \lambda \Phi (\boldsymbol {x}) + \psi \Phi (\boldsymbol {y}).\tag{2.87}
$$

It turns out that we can represent linear mappings as matrices (Section 2.7.1). Recall that we can also collect a set of vectors as columns of a matrix. When working with matrices, we have to keep in mind what the matrix represents: a linear mapping or a collection of vectors. We will see more about linear mappings in Chapter 4. Before we continue, we will briefly introduce special mappings. 

Definition 2.16 (Injective, Surjective, Bijective). Consider a mapping $\Phi :$ $\nu \to \mathcal W$ , where $\nu , w$ can be arbitrary sets. Then Φ is called 

Injective if $\forall x , y \in \mathcal { V } : \Phi ( x ) = \Phi ( y ) \implies x = y .$ 

Surjective if $\Phi ( \nu ) = \mathcal { W } .$ 

Bijective if it is injective and surjective. 

## 2.7 Linear Mappings

If Φ is surjective, then every element in can be “reached” from using Φ. A bijective Φ can be “undone”, i.e., there exists a mapping $\Psi$ : $\mathcal { W }  \mathcal { V }$ so that $\Psi \circ \Phi ( { \pmb x } ) = { \pmb x }$ . This mapping Ψ is then called the inverse of Φ and normally denoted by $\Phi ^ { - 1 }$ 

With these definitions, we introduce the following special cases of linear mappings between vector spaces $V$ and $W { : }$ : 

Isomorphism: Φ $: V  W$ linear and bijective 

Endomorphism: $\Phi : V  V$ linear 

Automorphism: $\Phi : V  V$ linear and bijective 

We define $\operatorname { i d } _ { V } : V \to V , x \mapsto x$ as the identity mapping or identity automorphism in $V$ 

## Example 2.19 (Homomorphism)

The mapping $\Phi : \mathbb { R } ^ { 2 }  \mathbb { C } , \ \Phi ( { \pmb x } ) = x _ { 1 } + i x _ { 2 } .$ , is a homomorphism: 

$$
\begin{array}{c} \Phi \left(\left[ \begin{array}{l} x _ {1} \\ x _ {2} \end{array} \right] + \left[ \begin{array}{l} y _ {1} \\ y _ {2} \end{array} \right]\right) = (x _ {1} + y _ {1}) + i (x _ {2} + y _ {2}) = x _ {1} + i x _ {2} + y _ {1} + i y _ {2} \\ = \Phi \left(\left[ \begin{array}{l} x _ {1} \\ x _ {2} \end{array} \right]\right) + \Phi \left(\left[ \begin{array}{l} y _ {1} \\ y _ {2} \end{array} \right]\right) \\ \Phi \left(\lambda \left[ \begin{array}{l} x _ {1} \\ x _ {2} \end{array} \right]\right) = \lambda x _ {1} + \lambda i x _ {2} = \lambda (x _ {1} + i x _ {2}) = \lambda \Phi \left(\left[ \begin{array}{l} x _ {1} \\ x _ {2} \end{array} \right]\right). \end{array}\tag{2.88}
$$

This also justifies why complex numbers can be represented as tuples in $\mathbb { R } ^ { 2 } \colon$ : There is a bijective linear mapping that converts the elementwise addition of tuples in $\mathbb { R } ^ { 2 }$ into the set of complex numbers with the corresponding addition. Note that we only showed linearity, but not the bijection. 

Theorem 2.17 (Theorem 3.59 in Axler (2015)). Finite-dimensional vector spaces $V$ and $W$ are isomorphic $i f$ and only $i f \dim ( V ) = \dim ( W )$ 

Theorem 2.17 states that there exists a linear, bijective mapping between two vector spaces of the same dimension. Intuitively, this means that vector spaces of the same dimension are kind of the same thing, as they can be transformed into each other without incurring any loss. 

Theorem 2.17 also gives us the justification to treat $\mathbb { R } ^ { m \times n }$ (the vector space of $m \times n { \mathrm { - m a t r i c e s } } )$ and $\mathbb { R } ^ { m n }$ (the vector space of vectors of length mn) the same, as their dimensions are mn, and there exists a linear, bijective mapping that transforms one into the other. 

Remark. Consider vector spaces $V , W , X$ . Then: 

For linear mappings $\Phi : V \to W$ and $\Psi : W \to X$ , the mapping $\Psi \circ \Phi : V  X$ is also linear. 

If $\Phi : V \to W$ is an isomorphism, then $\Phi ^ { - 1 } : W \to V$ is an isomorphism, too. 

Figure 2.8 Two different coordinate systems defined by two sets of basis vectors. A vector x has different coordinate representations depending on which coordinate system is chosen. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/f1d1c2863a0c8551572a589dfcfde0d1bb47f98c1bbc4e8f47071196845ab25b.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/e687ee801bac54bddc93cec87abdb4a94c3c232e4e40553203fb21c0d94d5fc5.jpg)



If Φ : $V  W ,$ Ψ : $V  W$ are linear, then $\Phi + \Psi$ and λΦ, $\lambda \in \mathbb { R } ,$ are linear, too.


## 2.7.1 Matrix Representation of Linear Mappings

Any n-dimensional vector space is isomorphic to $\mathbb { R } ^ { n }$ (Theorem 2.17). We consider a basis $\{ b _ { 1 } , \ldots , b _ { n } \}$ of an n-dimensional vector space V . In the following, the order of the basis vectors will be important. Therefore, we write 

$$
B = (\boldsymbol {b} _ {1}, \dots , \boldsymbol {b} _ {n})\tag{2.89}
$$

and call this n-tuple an ordered basis of $V .$ 

Remark (Notation). We are at the point where notation gets a bit tricky. Therefore, we summarize some parts here. $B = ( b _ { 1 } , \ldots , b _ { n } )$ is an ordered basis, $B = \{ b _ { 1 } , \ldots , b _ { n } \}$ is an (unordered) basis, and $B = [ \pmb { b } _ { 1 } , \dots , \pmb { b } _ { n } ]$ is a matrix whose columns are the vectors $b _ { 1 } , \ldots , b _ { n }$ ◇ 

Definition 2.18 (Coordinates). Consider a vector space $V$ and an ordered basis $B = ( b _ { 1 } , \ldots , b _ { n } )$ of $V$ . For any $\pmb { x } \in V$ we obtain a unique representation (linear combination) 

$$
\boldsymbol {x} = \alpha_ {1} \boldsymbol {b} _ {1} + \dots + \alpha_ {n} \boldsymbol {b} _ {n}\tag{2.90}
$$

of x with respect to $B .$ . Then $\alpha _ { 1 } , \ldots , \alpha _ { n }$ are the coordinates of x with respect to $B ,$ and the vector 

$$
\boldsymbol {\alpha} = \left[ \begin{array}{c} \alpha_ {1} \\ \vdots \\ \alpha_ {n} \end{array} \right] \in \mathbb {R} ^ {n}\tag{2.91}
$$

is the coordinate vector/coordinate representation of x with respect to the ordered basis B. 

Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 

## 2.7 Linear Mappings

A basis effectively defines a coordinate system. We are familiar with the Cartesian coordinate system in two dimensions, which is spanned by the canonical basis vectors $e _ { 1 } , e _ { 2 }$ . In this coordinate system, a vector $\pmb { x } \in \mathbb { R } ^ { 2 }$ has a representation that tells us how to linearly combine $e _ { 1 }$ and $e _ { 2 }$ to obtain ${ \pmb x } .$ . However, any basis of $\mathbb { R } ^ { 2 }$ defines a valid coordinate system, and the same vector x from before may have a different coordinate representation in the $( b _ { 1 } , b _ { 2 } )$ basis. In Figure 2.8, the coordinates of x with respect to the standard basis $( e _ { 1 } , e _ { 2 } ) \mathrm { i } s [ 2 , 2 ] ^ { \top }$ . However, with respect to the basis $\left( \boldsymbol { b } _ { 1 } , \boldsymbol { b } _ { 2 } \right)$ the same vector x is represented as $[ 1 . 0 9 , 0 . 7 2 ] ^ { \top }$ , i.e., $\pmb { x } = 1 . 0 9 b _ { 1 } + 0 . 7 2 b _ { 2 }$ . In the following sections, we will discover how to obtain this representation. 

## Example 2.20

Let us have a look at a geometric vector $\pmb { x } \in \mathbb { R } ^ { 2 }$ with coordinates $[ 2 , 3 ] ^ { \top }$ with respect to the standard basis $( e _ { 1 } , e _ { 2 } )$ of $\mathbb { R } ^ { 2 }$ . This means, we can write $\pmb { x } = 2 e _ { 1 } + 3 e _ { 2 }$ . However, we do not have to choose the standard basis to represent this vector. If we use the basis vectors $\pmb { b } _ { 1 } = [ 1 , - 1 ] ^ { \top } , \pmb { b } _ { 2 } = [ 1 , 1 ] ^ { \top }$ we will obtain the coordinates $\mathring { 2 } [ - 1 , 5 ] ^ { \top }$ to represent the same vector with respect to $( b _ { 1 } , b _ { 2 } )$ (see Figure 2.9). 


Figure 2.9 Different coordinate representations of a vector $^ { x , }$ depending on the choice of basis.


Remark. For an n-dimensional vector space V and an ordered basis B of $V ,$ the mapping $\Phi : \mathbb { R } ^ { n } \to V , \Phi ( e _ { i } ) = \pmb { b } _ { i } , i = 1 , \dots , n $ , is linear (and because of Theorem 2.17 an isomorphism), where $( e _ { 1 } , \ldots , e _ { n } )$ is the standard basis of $\mathbb { R } ^ { n }$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/468b26074b58d3fe320981728f2dad6fadf510fae345454676e5dd2f636e3f47.jpg)


Now we are ready to make an explicit connection between matrices and linear mappings between finite-dimensional vector spaces. 

Definition 2.19 (Transformation Matrix). Consider vector spaces $V , W$ with corresponding (ordered) bases $B = ( b _ { 1 } , \ldots , b _ { n } )$ and $C = ( c _ { 1 } , \ldots , c _ { m } )$ Moreover, we consider a linear mapping $\Phi : V  W$ . For $j \in \{ 1 , \ldots , n \}$ 

$$
\Phi (\boldsymbol {b} _ {j}) = \alpha_ {1 j} \boldsymbol {c} _ {1} + \dots + \alpha_ {m j} \boldsymbol {c} _ {m} = \sum_ {i = 1} ^ {m} \alpha_ {i j} \boldsymbol {c} _ {i}\tag{2.92}
$$

is the unique representation of $\Phi ( b _ { j } )$ with respect to $C .$ . Then, we call the m n-matrix $\pmb { A } _ { \Phi }$ , whose elements are given by 

$$
A _ {\Phi} (i, j) = \alpha_ {i j},\tag{2.93}
$$

the transformation matrix of Φ (with respect to the ordered bases B of V and $C$ of W ). 

The coordinates of $\Phi ( \pmb { b } _ { i } )$ with respect to the ordered basis C of W are the j-th column of $A _ { \Phi } .$ Consider (finite-dimensional) vector spaces $V , W$ with ordered bases $B , C$ and a linear mapping $\Phi : V \to W$ with transformation matrix $A _ { \Phi }$ . If xˆ is the coordinate vector of $\textbf { \em x } \in \textbf { \textit { V } }$ with respect to $B$ and $\hat { y }$ the coordinate vector of ${ \pmb y } = { \pmb \Phi } ( { \pmb x } ) \in W$ with respect to $C _ { i }$ then 

$$
\hat {\boldsymbol {y}} = \boldsymbol {A} _ {\Phi} \hat {\boldsymbol {x}}.\tag{2.94}
$$

This means that the transformation matrix can be used to map coordinates with respect to an ordered basis in $V$ to coordinates with respect to an ordered basis in $W$ 

## Example 2.21 (Transformation Matrix)

Consider a homomorphism $\Phi \ : \ V \ \to \ W$ and ordered bases $B \ =$ $( b _ { 1 } , \ldots , b _ { 3 } )$ of V and $C = ( c _ { 1 } , \ldots , c _ { 4 } )$ of W . With 

$$
\begin{array}{l} \Phi (\boldsymbol {b} _ {1}) = \boldsymbol {c} _ {1} - \boldsymbol {c} _ {2} + 3 \boldsymbol {c} _ {3} - \boldsymbol {c} _ {4} \\ \Phi (\boldsymbol {b} _ {2}) = 2 \boldsymbol {c} _ {1} + \boldsymbol {c} _ {2} + 7 \boldsymbol {c} _ {3} + 2 \boldsymbol {c} _ {4} \\ \Phi (\boldsymbol {b} _ {3}) = 3 \boldsymbol {c} _ {2} + \boldsymbol {c} _ {3} + 4 \boldsymbol {c} _ {4} \end{array}\tag{2.95}
$$

the transformation matrix $A _ { \Phi }$ with respect to $B$ and $C$ satisfies $\Phi ( \boldsymbol { b } _ { k } ) =$ $\textstyle \sum _ { i = 1 } ^ { 4 } \alpha _ { i k } { \pmb { c } } _ { i }$ for $k = 1 , \ldots , 3$ and is given as 

$$
\boldsymbol {A} _ {\Phi} = \left[ \boldsymbol {\alpha} _ {1}, \boldsymbol {\alpha} _ {2}, \boldsymbol {\alpha} _ {3} \right] = \left[ \begin{array}{c c c} 1 & 2 & 0 \\ - 1 & 1 & 3 \\ 3 & 7 & 1 \\ - 1 & 2 & 4 \end{array} \right],\tag{2.96}
$$

where the $\alpha _ { j } , j = 1 , 2 , 3$ , are the coordinate vectors of $\Phi ( \pmb { b } _ { j } )$ with respect to $C .$ 

## Example 2.22 (Linear Transformations of Vectors)

Figure 2.10 Three examples of linear transformations of the vectors shown as dots in (a); (b) Rotation by 45<sup>◦</sup>; (c) Stretching of the horizontal coordinates by 2; (d) Combination of reflection, rotation and stretching. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/223c57e3d45155d84f271ac2ff4cf9ff8e5af7a54f2ce327b969e4628cd4a991.jpg)



(a) Original data.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/bfd2fc9be8e2eba7f222eb67fddbf707a501e753d2699ace7ad9e51ac234c376.jpg)



(b) Rotation by 45<sup>◦</sup>.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/9e0c1f16fe6b3c743e3f5dfc23385ac94812cc1b46d4de69027df19d9026baa3.jpg)



(c) Stretch along the horizontal axis.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/2627fa3c90999dd595f224e64ebf8ba0b4d29ebf827b5d549e9837977f7c7da9.jpg)



(d) General linear mapping.


We consider three linear transformations of a set of vectors in $\mathbb { R } ^ { 2 }$ with the transformation matrices 

$$
\boldsymbol {A} _ {1} = \left[ \begin{array}{c c} \cos (\frac {\pi}{4}) & - \sin (\frac {\pi}{4}) \\ \sin (\frac {\pi}{4}) & \cos (\frac {\pi}{4}) \end{array} \right], \boldsymbol {A} _ {2} = \left[ \begin{array}{c c} 2 & 0 \\ 0 & 1 \end{array} \right], \boldsymbol {A} _ {3} = \frac {1}{2} \left[ \begin{array}{c c} 3 & - 1 \\ 1 & - 1 \end{array} \right].\tag{2.97}
$$

Figure 2.10 gives three examples of linear transformations of a set of vectors. Figure 2.10(a) shows 400 vectors in $\mathbb { R } ^ { 2 }$ , each of which is represented by a dot at the corresponding $( x _ { 1 } , x _ { 2 } )$ -coordinates. The vectors are arranged in a square. When we use matrix $\pmb { A } _ { 1 }$ in (2.97) to linearly transform each of these vectors, we obtain the rotated square in Figure 2.10(b). If we apply the linear mapping represented by $A _ { 2 }$ , we obtain the rectangle in Figure 2.10(c) where each $x _ { 1 }$ -coordinate is stretched by 2. Figure 2.10(d) shows the original square from Figure 2.10(a) when linearly transformed using $A _ { 3 } ,$ which is a combination of a reflection, a rotation, and a stretch. 

## 2.7.2 Basis Change

In the following, we will have a closer look at how transformation matrices of a linear mapping $\Phi : V $ W change if we change the bases in $V$ and $W$ . Consider two ordered bases 

$$
B = (\boldsymbol {b} _ {1}, \dots , \boldsymbol {b} _ {n}), \quad \tilde {B} = (\tilde {\boldsymbol {b}} _ {1}, \dots , \tilde {\boldsymbol {b}} _ {n})\tag{2.98}
$$

of $V$ and two ordered bases 

$$
C = (\pmb {c} _ {1}, \dots , \pmb {c} _ {m}), \quad \tilde {C} = (\tilde {\pmb {c}} _ {1}, \dots , \tilde {\pmb {c}} _ {m})\tag{2.99}
$$

of $W .$ . Moreover, $A _ { \Phi } \in \mathbb { R } ^ { m \times n }$ is the transformation matrix of the linear mapping $\Phi : V  W$ with respect to the bases $B$ and $C ,$ and $\tilde { \mathbf { A } } _ { \Phi } \in \mathbb { R } ^ { m \times n }$ is the corresponding transformation mapping with respect to B<sup>˜</sup> and $\tilde { C }$ In the following, we will investigate how $\pmb { A }$ and $\tilde { \boldsymbol { A } }$ are related, $\mathrm { i . e . }$ , how/ whether we can transform $A _ { \Phi }$ into $\tilde { A } _ { \Phi }$ if we choose to perform a basis change from $B , C$ to $\tilde { B } , \tilde { C }$ 

Remark. We effectively get different coordinate representations of the identity mapping id . In the context of Figure 2.9, this would mean to map coordinates with respect to $( e _ { 1 } , e _ { 2 } )$ onto coordinates with respect to $( b _ { 1 } , b _ { 2 } )$ without changing the vector ${ \pmb x } .$ . By changing the basis and correspondingly the representation of vectors, the transformation matrix with respect to this new basis can have a particularly simple form that allows for straightforward computation. 

## Example 2.23 (Basis Change)

Consider a transformation matrix 

$$
\boldsymbol {A} = \left[ \begin{array}{c c} 2 & 1 \\ 1 & 2 \end{array} \right]\tag{2.100}
$$

with respect to the canonical basis in $\mathbb { R } ^ { 2 }$ . If we define a new basis 

$$
B = (\left[ \begin{array}{c} 1 \\ 1 \end{array} \right], \left[ \begin{array}{c} 1 \\ - 1 \end{array} \right])\tag{2.101}
$$



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



we obtain a diagonal transformation matrix 

$$
\tilde {\pmb {A}} = \left[ \begin{array}{c c} 3 & 0 \\ 0 & 1 \end{array} \right]\tag{2.102}
$$

with respect to $B ,$ , which is easier to work with than A. 

In the following, we will look at mappings that transform coordinate vectors with respect to one basis into coordinate vectors with respect to a different basis. We will state our main result first and then provide an explanation. 

Theorem 2.20 (Basis Change). For a linear mapping $\Phi : V \to W$ , ordered bases 

$$
B = (\boldsymbol {b} _ {1}, \dots , \boldsymbol {b} _ {n}), \quad \tilde {B} = (\tilde {\boldsymbol {b}} _ {1}, \dots , \tilde {\boldsymbol {b}} _ {n})\tag{2.103}
$$

of $V$ and 

$$
C = (\pmb {c} _ {1}, \dots , \pmb {c} _ {m}), \quad \tilde {C} = (\tilde {\pmb {c}} _ {1}, \dots , \tilde {\pmb {c}} _ {m})\tag{2.104}
$$

of $W ,$ and a transformation matrix $A _ { \Phi }$ of Φ with respect to $B$ and $C ,$ the corresponding transformation matrix $\tilde { A } _ { \Phi }$ with respect to the bases $\tilde { B }$ and $\tilde { C }$ is given as 

$$
\tilde {\boldsymbol {A}} _ {\Phi} = \boldsymbol {T} ^ {- 1} \boldsymbol {A} _ {\Phi} \boldsymbol {S}.\tag{2.105}
$$

Here, $\ b { S } \in \mathbb { R } ^ { n \times n }$ is the transformation matrix of id that maps coordinates with respect to $\tilde { B }$ onto coordinates with respect to $B ,$ and $\pmb { T } \in \mathbb { R } ^ { m \times m }$ is the transformation matrix of $\mathrm { i d } _ { W }$ that maps coordinates with respect to $\tilde { C }$ onto coordinates with respect to $C .$ 

Proof Following Drumm and Weil (2001), we can write the vectors of the new basis $\tilde { B }$ of $V$ as a linear combination of the basis vectors of $B ,$ such that 

$$
\tilde {\boldsymbol {b}} _ {j} = s _ {1 j} \boldsymbol {b} _ {1} + \dots + s _ {n j} \boldsymbol {b} _ {n} = \sum_ {i = 1} ^ {n} s _ {i j} \boldsymbol {b} _ {i}, \quad j = 1, \dots , n.\tag{2.106}
$$

Similarly, we write the new basis vectors $\tilde { C }$ of $W$ as a linear combination of the basis vectors of $C _ { i }$ , which yields 

$$
\tilde {\boldsymbol {c}} _ {k} = t _ {1 k} \boldsymbol {c} _ {1} + \dots + t _ {m k} \boldsymbol {c} _ {m} = \sum_ {l = 1} ^ {m} t _ {l k} \boldsymbol {c} _ {l}, \quad k = 1, \ldots , m.\tag{2.107}
$$

We define $S = ( ( s _ { i j } ) ) \in \mathbb { R } ^ { n \times n }$ as the transformation matrix that maps coordinates with respect to $\tilde { B }$ onto coordinates with respect to $B$ and $\pmb { T } = \left( \left( t _ { l k } \right) \right) \in \mathbb { R } ^ { m \times m }$ as the transformation matrix that maps coordinates with respect to $\tilde { C }$ onto coordinates with respect to C. In particular, the jth column of $_ { s }$ is the coordinate representation of $\tilde { \boldsymbol { b } } _ { j }$ with respect to $B$ and the kth column of T is the coordinate representation of $\tilde { c } _ { k }$ with respect to C. Note that both S and $\mathbf { T }$ are regular. 

We are going to look at $\Phi ( \tilde { \pmb { b } } _ { j } )$ from two perspectives. First, applying the mapping Φ, we get that for all $j = 1 , \dotsc , n$ 

$$
\Phi (\tilde {\boldsymbol {b}} _ {j}) = \sum_ {k = 1} ^ {m} \underbrace {\tilde {a} _ {k j} \tilde {\boldsymbol {c}} _ {k}} _ {\in W} \stackrel {(2. 1 0 7)} {=} \sum_ {k = 1} ^ {m} \tilde {a} _ {k j} \sum_ {l = 1} ^ {m} t _ {l k} \boldsymbol {c} _ {l} = \sum_ {l = 1} ^ {m} \left(\sum_ {k = 1} ^ {m} t _ {l k} \tilde {a} _ {k j}\right) \boldsymbol {c} _ {l},\tag{2.108}
$$

where we first expressed the new basis vectors $\tilde { \boldsymbol { c } } _ { k } ~ \in ~ W$ as linear combinations of the basis vectors $\mathbf { \boldsymbol { c } } _ { l } ~ \in ~ W$ and then swapped the order of summation. 

Alternatively, when we express the $\tilde { \pmb { b } } _ { j } \in V$ as linear combinations of $b _ { j } \in V$ , we arrive at 

$$
\Phi (\tilde {\boldsymbol {b}} _ {j}) \stackrel {(2. 1 0 6)} {=} \Phi \left(\sum_ {i = 1} ^ {n} s _ {i j} \boldsymbol {b} _ {i}\right) = \sum_ {i = 1} ^ {n} s _ {i j} \Phi (\boldsymbol {b} _ {i}) = \sum_ {i = 1} ^ {n} s _ {i j} \sum_ {l = 1} ^ {m} a _ {l i} \boldsymbol {c} _ {l}\tag{2.109a}
$$

$$
= \sum_ {l = 1} ^ {m} \left(\sum_ {i = 1} ^ {n} a _ {l i} s _ {i j}\right) \boldsymbol {c} _ {l}, \quad j = 1, \ldots , n,\tag{2.109b}
$$

where we exploited the linearity of Φ. Comparing (2.108) and (2.109b), it follows for all $j = 1 , \dotsc , n$ and $l = 1 , \ldots , m$ that 

$$
\sum_ {k = 1} ^ {m} t _ {l k} \tilde {a} _ {k j} = \sum_ {i = 1} ^ {n} a _ {l i} s _ {i j}\tag{2.110}
$$

and, therefore, 

$$
\boldsymbol {T} \tilde {\boldsymbol {A}} _ {\Phi} = \boldsymbol {A} _ {\Phi} \boldsymbol {S} \in \mathbb {R} ^ {m \times n},\tag{2.111}
$$

such that 

$$
\tilde {\boldsymbol {A}} _ {\Phi} = \boldsymbol {T} ^ {- 1} \boldsymbol {A} _ {\Phi} \boldsymbol {S},\tag{2.112}
$$

which proves Theorem 2.20. 

Theorem 2.20 tells us that with a basis change in $V \left( B \right)$ is replaced with $\tilde { B } )$ and $W$ (C is replaced with $\tilde { C } )$ , the transformation matrix $A _ { \Phi }$ of a linear mapping $\Phi : V  W$ is replaced by an equivalent matrix $\tilde { \mathbf { A } } _ { \Phi }$ with 

$$
\tilde {\boldsymbol {A}} _ {\Phi} = \boldsymbol {T} ^ {- 1} \boldsymbol {A} _ {\Phi} \boldsymbol {S}.\tag{2.113}
$$

Figure 2.11 illustrates this relation: Consider a homomorphism $\Phi : V $ W and ordered bases B, $\tilde { B }$ of $V$ and $C , { \tilde { C } }$ of W . The mapping $\Phi _ { C B }$ is an instantiation of $\Phi$ and maps basis vectors of B onto linear combinations of basis vectors of C. Assume that we know the transformation matrix $A _ { \Phi }$ of $\Phi _ { C B }$ with respect to the ordered bases B, C. When we perform a basis change from $B$ to $\tilde { B }$ in $V$ and from $C$ to $\tilde { C }$ in $W _ { i }$ , we can determine the 

Vector spaces 

Ordered bases 

$$
\begin{array}{c c} V \xrightarrow {\Phi} W & V \xrightarrow {\Phi} W \\ B \xrightarrow {\Phi_ {C B}} C & B \xrightarrow {\Phi_ {C B}} C \\ \Psi_ {B \tilde {B}} \Big | \begin{array}{c c} S & T \\ \tilde {A} _ {\Phi} & \tilde {C} \end{array} \Big | \Xi_ {C \tilde {C}} & \Psi_ {B \tilde {B}} \Big | \begin{array}{c c} S & T ^ {- 1} \\ \tilde {B} & \tilde {C} \end{array} \Big | \Xi_ {\tilde {C} \tilde {C}} = \Xi_ {C \tilde {C}} ^ {- 1} \end{array}
$$

corresponding transformation matrix $\tilde { \mathbf { A } } _ { \Phi }$ as follows: First, we find the matrix representation of the linear mapping $\Psi _ { B \tilde { B } } : V  V$ that maps coordinates with respect to the new basis $\tilde { B }$ onto the (unique) coordinates with respect to the $\bar { \iota } _ { \mathrm { ( o l d ) } } ,$ basis $B$ (in $V )$ . Then, we use the transformation matrix $A _ { \Phi }$ of $\Phi _ { C B } : V \to W$ to map these coordinates onto the coordinates with respect to $C$ in W . Finally, we use a linear mapping $\Xi _ { \tilde { C } C } : W \to W$ to map the coordinates with respect to $C$ onto coordinates with respect to ${ \tilde { C } } .$ Therefore, we can express the linear mapping $\Phi _ { \tilde { C } \tilde { B } }$ as a composition of linear mappings that involve the $ { { } ^ { \circ } } \mathrm { o l d } ^ { \prime \prime }$ basis: 

$$
\Phi_ {\tilde {C} \tilde {B}} = \Xi_ {\tilde {C} C} \circ \Phi_ {C B} \circ \Psi_ {B \tilde {B}} = \Xi_ {C \tilde {C}} ^ {- 1} \circ \Phi_ {C B} \circ \Psi_ {B \tilde {B}}.\tag{2.114}
$$

Concretely, we use $\Psi _ { B \tilde { B } } = \mathrm { i d } _ { V }$ and $\Xi _ { C \tilde { C } } = \mathrm { i d } _ { W }$ , i.e., the identity mappings that map vectors onto themselves, but with respect to a different basis. 

Definition 2.21 (Equivalence). Two matrices $\mathbf { \Delta } \mathbf { A } , \tilde { \mathbf { A } } \in \mathbb { R } ^ { m \times n }$ are equivalent if there exist regular matrices $S ~ \in ~ \mathbb { R } ^ { n \times n }$ and $\pmb { T } \in \mathbb { R } ^ { m \times m }$ , such that $\tilde { \pmb { A } } = \pmb { T } ^ { - 1 } \pmb { A } \pmb { S }$ 

Definition 2.22 (Similarity). Two matrices $\pmb { A } , \tilde { \pmb { A } } \in \mathbb { R } ^ { n \times n }$ are similar if there exists a regular matrix $\ b { S } \in \mathbb { R } ^ { n \times n }$ with $\tilde { \pmb { A } } = \pmb { S } ^ { - 1 } \pmb { A } \pmb { S }$ 

Remark. Similar matrices are always equivalent. However, equivalent matrices are not necessarily similar. $\diamondsuit$ 

Remark. Consider vector spaces $V , W , X$ . From the remark that follows Theorem 2.17, we already know that for linear mappings $\Phi : V \to W$ and $\Psi : W \to X$ the mapping $\Psi \circ \Phi : V  X$ is also linear. With transformation matrices $A _ { \Phi }$ and $\pmb { A } _ { \Psi }$ of the corresponding mappings, the overall transformation matrix is $\pmb { A } _ { \Psi \circ \Phi } = \pmb { A } _ { \Psi } \pmb { A } _ { \Phi }$ $\diamondsuit$ 

In light of this remark, we can look at basis changes from the perspective of composing linear mappings: 

$A _ { \Phi }$ is the transformation matrix of a linear mapping $\Phi _ { C B } : V \to W$ with respect to the bases $B , C .$ 

${ \tilde { \mathbf { A } } } _ { \Phi }$ is the transformation matrix of the linear mapping $\Phi _ { \tilde { C } \tilde { B } } : V \to W$ with respect to the bases $\tilde { B } , \tilde { C } .$ 

S is the transformation matrix of a linear mapping $\Psi _ { B \tilde { B } } : V $ V (automorphism) that represents $\tilde { B }$ in terms of $B .$ . Normally, $\Psi = \operatorname { i d } _ { V }$ is the identity mapping in $V$ 

T is the transformation matrix of a linear mapping $\Xi _ { C \tilde { C } } : W \to W$ (automorphism) that represents $\tilde { C }$ in terms of C. Normally, $\Xi = \mathrm { i d } _ { W }$ is the identity mapping in $W$ 

If we (informally) write down the transformations just in terms of bases, then $A _ { \Phi } : B \  \ C , \ \tilde { A } _ { \Phi } : \ \tilde { B } \  \ \tilde { C } , \ S : \ \tilde { B } \  \ \tilde { B } , \ T : \ \tilde { C } \  \ C$ and $\pmb { T } ^ { - 1 } : C  \tilde { C } ,$ , and 

$$
\tilde {B} \rightarrow \tilde {C} = \tilde {B} \rightarrow B \rightarrow C \rightarrow \tilde {C}\tag{2.115}
$$

$$
\tilde {\boldsymbol {A}} _ {\Phi} = \boldsymbol {T} ^ {- 1} \boldsymbol {A} _ {\Phi} \boldsymbol {S}.\tag{2.116}
$$

Note that the execution order in (2.116) is from right to left because vectors are multiplied at the right-hand side so that $\pmb { x } \mapsto \pmb { S x } \mapsto \pmb { A } _ { \Phi } ( \pmb { S x } ) \mapsto$ $\pmb { T } ^ { - 1 } \big ( \pmb { A } _ { \Phi } ( \pmb { S } \pmb { x } ) \big ) = \tilde { \pmb { A } } _ { \Phi } \pmb { x }$ 

## Example 2.24 (Basis Change)

Consider a linear mapping $\dot { \boldsymbol { \mathbf { \ell } } } : \mathbb { R } ^ { 3 }  \mathbb { R } ^ { 4 }$ whose transformation matrix is 

$$
\boldsymbol {A} _ {\Phi} = \left[ \begin{array}{c c c} 1 & 2 & 0 \\ - 1 & 1 & 3 \\ 3 & 7 & 1 \\ - 1 & 2 & 4 \end{array} \right]\tag{2.117}
$$

with respect to the standard bases 

$$
B = (\left[ \begin{array}{c} 1 \\ 0 \\ 0 \end{array} \right], \left[ \begin{array}{c} 0 \\ 1 \\ 0 \end{array} \right], \left[ \begin{array}{c} 0 \\ 0 \\ 1 \end{array} \right]), \quad C = (\left[ \begin{array}{c} 1 \\ 0 \\ 0 \\ 0 \end{array} \right], \left[ \begin{array}{c} 0 \\ 1 \\ 0 \\ 0 \end{array} \right], \left[ \begin{array}{c} 0 \\ 0 \\ 1 \\ 0 \end{array} \right], \left[ \begin{array}{c} 0 \\ 0 \\ 0 \\ 1 \end{array} \right]).\tag{2.118}
$$

We seek the transformation matrix $\tilde { \mathbf { A } } _ { \Phi }$ of Φ with respect to the new bases 

$$
\tilde {B} = (\left[ \begin{array}{c} 1 \\ 1 \\ 0 \end{array} \right], \left[ \begin{array}{c} 0 \\ 1 \\ 1 \end{array} \right], \left[ \begin{array}{c} 1 \\ 0 \\ 1 \end{array} \right]) \in \mathbb {R} ^ {3}, \quad \tilde {C} = (\left[ \begin{array}{c} 1 \\ 1 \\ 0 \\ 0 \end{array} \right], \left[ \begin{array}{c} 1 \\ 0 \\ 1 \\ 0 \end{array} \right], \left[ \begin{array}{c} 0 \\ 1 \\ 1 \\ 0 \end{array} \right], \left[ \begin{array}{c} 1 \\ 0 \\ 0 \\ 1 \end{array} \right]).\tag{2.119}
$$

Then, 

$$
\boldsymbol {S} = \left[ \begin{array}{c c c} 1 & 0 & 1 \\ 1 & 1 & 0 \\ 0 & 1 & 1 \end{array} \right], \qquad \boldsymbol {T} = \left[ \begin{array}{c c c c} 1 & 1 & 0 & 1 \\ 1 & 0 & 1 & 0 \\ 0 & 1 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{array} \right],\tag{2.120}
$$

where the ith column of S is the coordinate representation of $\tilde { \pmb { b } } _ { i }$ in terms of the basis vectors of B. Since B is the standard basis, the coordinate representation is straightforward to find. For a general basis $B _ { z }$ we would need to solve a linear equation system to find the $\lambda _ { i }$ such that 



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



$\begin{array} { r } { \sum _ { i = 1 } ^ { 3 } \lambda _ { i } \pmb { b } _ { i } = \tilde { \pmb { b } } _ { j } , j = 1 , \dots , 3 } \end{array}$ . Similarly, the jth column of $_ { \pmb { T } }$ is the coordinate representation of $\tilde { \boldsymbol { c } } _ { j }$ in terms of the basis vectors of C. 

Therefore, we obtain 

$$
\begin{array}{c} \tilde {\boldsymbol {A}} _ {\Phi} = \boldsymbol {T} ^ {- 1} \boldsymbol {A} _ {\Phi} \boldsymbol {S} = \frac {1}{2} \left[ \begin{array}{c c c c} 1 & 1 & - 1 & - 1 \\ 1 & - 1 & 1 & - 1 \\ - 1 & 1 & 1 & 1 \\ 0 & 0 & 0 & 2 \end{array} \right] \left[ \begin{array}{c c c} 3 & 2 & 1 \\ 0 & 4 & 2 \\ 1 0 & 8 & 4 \\ 1 & 6 & 3 \end{array} \right] \\ = \left[ \begin{array}{c c c} - 4 & - 4 & - 2 \\ 6 & 0 & 0 \\ 4 & 8 & 4 \\ 1 & 6 & 3 \end{array} \right]. \end{array}\tag{2.121a}
$$

(2.121b) 

In Chapter 4, we will be able to exploit the concept of a basis change to find a basis with respect to which the transformation matrix of an endomorphism has a particularly simple (diagonal) form. In Chapter 10, we will look at a data compression problem and find a convenient basis onto which we can project the data while minimizing the compression loss. 

## 2.7.3 Image and Kernel

The image and kernel of a linear mapping are vector subspaces with certain important properties. In the following, we will characterize them more carefully. 

Definition 2.23 (Image and Kernel). 

For $\Phi : V  W$ , we define the kernel/null space 

$$
\ker (\Phi) := \Phi^ {- 1} (\mathbf {0} _ {W}) = \{\boldsymbol {v} \in V: \Phi (\boldsymbol {v}) = \mathbf {0} _ {W} \}\tag{2.122}
$$

and the image/range 

$$
\operatorname{Im} (\Phi) := \Phi (V) = \left\{\boldsymbol {w} \in W | \exists \boldsymbol {v} \in V: \Phi (\boldsymbol {v}) = \boldsymbol {w} \right\}.\tag{2.123}
$$

We also call V and W also the domain and codomain of Φ, respectively. 

Intuitively, the kernel is the set of vectors $v \in V$ that Φ maps onto the neutral element ${ \mathbf { 0 } } _ { W } \in W$ . The image is the set of vectors $\mathbf { \pmb { w } } \in \mathbf { \nabla } W$ that can be “reached” by Φ from any vector in V . An illustration is given in Figure 2.12. 

Remark. Consider a linear mapping $\Phi : V \to W$ , where $V , W$ are vector spaces. 

It always holds that $\Phi ( { \bf 0 } _ { V } ) = { \bf 0 } _ { W }$ and, therefore, ${ \mathbf { 0 } _ { V } } \in \mathop { \ker } ( \Phi )$ . In particular, the null space is never empty. 

${ \mathrm { I m } } ( \Phi ) \subseteq W$ is a subspace of $W ,$ and $\ker ( \Phi ) \subseteq V$ is a subspace of $V .$ . 

## 2.7 Linear Mappings

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/d8429da36fa1617499bc221662a600c2cd2c44e517a9f849a2c6f9f4ea1f18d2.jpg)



Figure 2.12 Kernel and image of a linear mapping Φ : V → W.


Φ is injective (one-to-one) if and only if $\ker ( \Phi ) = \{ \mathbf { 0 } \}$ 

Remark (Null Space and Column Space). Let us consider $\pmb { A } \in \mathbb { R } ^ { m \times n }$ and a linear mapping $\Phi : \mathbb { R } ^ { n }  \mathbb { R } ^ { m }$ , x $\rangle \mapsto A x$ 

For $\pmb { A } = [ \pmb { a } _ { 1 } , \dots , \pmb { a } _ { n } ]$ , where $\mathbf { a } _ { i }$ are the columns of A, we obtain 

$$
\begin{array}{c} \operatorname{Im} (\Phi) = \{\boldsymbol {A x}: \boldsymbol {x} \in \mathbb {R} ^ {n} \} = \left\{\sum_ {i = 1} ^ {n} x _ {i} \boldsymbol {a} _ {i}: x _ {1}, \ldots , x _ {n} \in \mathbb {R} \right\} \\ = \operatorname{span} [ \boldsymbol {a} _ {1}, \ldots , \boldsymbol {a} _ {n} ] \subseteq \mathbb {R} ^ {m}, \end{array}\tag{2.124a}
$$

(2.124b) 

$. . \mathsf { e . , }$ the image is the span of the columns of A, also called the column space. Therefore, the column space (image) is a subspace of $\mathbb { R } ^ { m }$ , where m is the “height” of the matrix. 

r $\ b ( A ) = \dim ( \operatorname { I m } ( \Phi ) )$ 

The kernel/null space ker(Φ) is the general solution to the homogeneous system of linear equations $\mathbf { { \boldsymbol { A } } } \mathbf { { \boldsymbol { x } } } = \mathbf { { \boldsymbol { 0 } } }$ and captures all possible linear combinations of the elements in $\mathbb { R } ^ { n }$ that produce $\mathbf { 0 } \in \mathbb { R } ^ { m }$ 

The kernel is a subspace of <sup>Rn</sup>, where n is the “width” of the matrix. 

The kernel focuses on the relationship among the columns, and we can use it to determine whether/how we can express a column as a linear combination of other columns. 

## Example 2.25 (Image and Kernel of a Linear Mapping) The mapping

$$
\Phi : \mathbb {R} ^ {4} \to \mathbb {R} ^ {2}, \quad \left[ \begin{array}{c} x _ {1} \\ x _ {2} \\ x _ {3} \\ x _ {4} \end{array} \right] \mapsto \left[ \begin{array}{c c c c} 1 & 2 & - 1 & 0 \\ 1 & 0 & 0 & 1 \end{array} \right] \left[ \begin{array}{c} x _ {1} \\ x _ {2} \\ x _ {3} \\ x _ {4} \end{array} \right] = \left[ \begin{array}{c} x _ {1} + 2 x _ {2} - x _ {3} \\ x _ {1} + x _ {4} \end{array} \right]\tag{2.125a}
$$



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



$$
= x _ {1} \left[ \begin{array}{c} 1 \\ 1 \end{array} \right] + x _ {2} \left[ \begin{array}{c} 2 \\ 0 \end{array} \right] + x _ {3} \left[ \begin{array}{c} - 1 \\ 0 \end{array} \right] + x _ {4} \left[ \begin{array}{c} 0 \\ 1 \end{array} \right]\tag{2.125b}
$$

is linear. To determine Im(Φ), we can take the span of the columns of the transformation matrix and obtain 

$$
\mathrm{Im} (\Phi) = \mathrm{span} [ \left[ \begin{array}{c} 1 \\ 1 \end{array} \right], \left[ \begin{array}{c} 2 \\ 0 \end{array} \right], \left[ \begin{array}{c} - 1 \\ 0 \end{array} \right], \left[ \begin{array}{c} 0 \\ 1 \end{array} \right] ].\tag{2.126}
$$

To compute the kernel (null space) of Φ, we need to solve $A x = \mathbf { 0 } , { \mathrm { i . e . } }$ we need to solve a homogeneous equation system. To do this, we use Gaussian elimination to transform A into reduced row-echelon form: 

$$
\left[\begin{array}{c c c c}1&2&- 1&0\\1&0&0&1\end{array}\right] \quad \rightsquigarrow \dots \rightsquigarrow \quad \left[\begin{array}{c c c c}1&0&0&1\\0&1&- \frac {1}{2}&- \frac {1}{2}\end{array}\right].\tag{2.127}
$$

This matrix is in reduced row-echelon form, and we can use the Minus-1 Trick to compute a basis of the kernel (see Section 2.3.3). Alternatively, we can express the non-pivot columns (columns 3 and 4) as linear combinations of the pivot columns (columns 1 and 2). The third column ${ \bf { a } } _ { 3 }$ is equivalent to $- \frac 1 2$ times the second column $\mathbf { a } _ { 2 }$ . Therefore, ${ \bf 0 } = { \bf a } _ { 3 } + \frac { 1 } { 2 } { \bf a } _ { 2 }$ . In the same way, we see that $\mathbf { a } _ { 4 } = \mathbf { a } _ { 1 } - \frac { 1 } { 2 } \mathbf { a } _ { 2 }$ and, therefore, $\mathbf { 0 } = { \pmb a } _ { 1 } - \frac { 1 } { 2 } { \pmb a } _ { 2 } - { \pmb a } _ { 4 }$ Overall, this gives us the kernel (null space) as 

$$
\ker (\Phi) = \operatorname{span} [ \left[ \begin{array}{c} 0 \\ \frac {1}{2} \\ 1 \\ 0 \end{array} \right], \left[ \begin{array}{c} - 1 \\ \frac {1}{2} \\ 0 \\ 1 \end{array} \right] ].\tag{2.128}
$$

Theorem 2.24 (Rank-Nullity Theorem). For vector spaces $V , W$ and a linear mapping $\Phi : V  W$ it holds that 

$$
\dim (\ker (\Phi)) + \dim (\operatorname{Im} (\Phi)) = \dim (V).\tag{2.129}
$$

The rank-nullity theorem is also referred to as the fundamental theorem of linear mappings (Axler, 2015, theorem 3.22). The following are direct consequences of Theorem 2.24: 

If dim $. ( \mathrm { I m } ( \Phi ) ) <$ dim(V ), then ker(Φ) is non-trivial, i.e., the kernel contains more than $\mathbf { 0 } _ { V }$ and $\dim ( \ker ( \Phi ) ) \geqslant 1$ 

If $A _ { \Phi }$ is the transformation matrix of Φ with respect to an ordered basis and dim $( \operatorname { I m } ( \Phi ) ) < \dim ( V )$ , then the system of linear equations ${ \pmb { A } } _ { \Phi } { \pmb x } =$ 0 has infinitely many solutions. 

If $\dim ( V ) = \dim ( W )$ , then the three-way equivalence 

Φ is injective $\iff \Phi$ is surjective $\iff \Phi$ is bijective 

holds since Im $( \Phi ) \subseteq W$ 

Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 

## 2.8 Affine Spaces

In the following, we will take a closer look at spaces that are offset from the ${ \mathrm { o r i g i n } } , { \mathrm { i . e . } } ,$ spaces that are no longer vector subspaces. Moreover, we will briefly discuss properties of mappings between these affine spaces, which resemble linear mappings. 

Remark. In the machine learning literature, the distinction between linear and affine is sometimes not clear so that we can find references to affine spaces/mappings as linear spaces/mappings. 

## 2.8.1 Affine Subspaces

Definition 2.25 (Affine Subspace). Let V be a vector space, $\pmb { x } _ { 0 } \in V$ and $U \subseteq V$ a subspace. Then the subset 

$$
L = \boldsymbol {x} _ {0} + U := \left\{\boldsymbol {x} _ {0} + \boldsymbol {u}: \boldsymbol {u} \in U \right\}\tag{2.130a}
$$

$$
= \left\{\boldsymbol {v} \in V | \exists \boldsymbol {u} \in U: \boldsymbol {v} = \boldsymbol {x} _ {0} + \boldsymbol {u} \right\} \subseteq V\tag{2.130b}
$$

is called affine subspace or linear manifold of V . U is called direction or direction space, and $\scriptstyle { \pmb x } _ { 0 }$ is called support point. In Chapter 12, we refer to such a subspace as a hyperplane. 

Note that the definition of an affine subspace excludes 0 if ${ \pmb x } _ { 0 } \not \in \ U$ Therefore, an affine subspace is not a (linear) subspace (vector subspace) of V for ${ \pmb x } _ { 0 } \not \in U$ 

Examples of affine subspaces are points, lines, and planes in $\mathbb { R } ^ { 3 }$ , which do not (necessarily) go through the origin. 

Remark. Consider two affine subspaces $L = { \pmb x } _ { 0 } + U$ and $\tilde { L } = \tilde { \mathbf { x } } _ { 0 } + \tilde { U }$ of a vector space V . Then, $L \subseteq { \tilde { L } }$ if and only if $U \subseteq { \tilde { U } }$ and ${ \pmb x } _ { 0 } - { \pmb { \tilde { x } } } _ { 0 } \in \tilde { U }$ 

Affine subspaces are often described by parameters: Consider a k-dimensional affine space ${ \boldsymbol { L } } = { \mathbf { { x } } } _ { 0 } + { \boldsymbol { U } }$ of V . If $\left( \pmb { b } _ { 1 } , \dots , \pmb { b } _ { k } \right)$ is an ordered basis of $U _ { s }$ , then every element $\boldsymbol { x } \in L$ can be uniquely described as 

$$
\boldsymbol {x} = \boldsymbol {x} _ {0} + \lambda_ {1} \boldsymbol {b} _ {1} + \dots + \lambda_ {k} \boldsymbol {b} _ {k},\tag{2.131}
$$

where $\lambda _ { 1 } , \ldots , \lambda _ { k } \in \mathbb { R }$ . This representation is called parametric equation of L with directional vectors $\boldsymbol { b } _ { 1 } , \ldots , \boldsymbol { b } _ { k }$ and parameters $\lambda _ { 1 } , \ldots , \lambda _ { k }$ 

## Example 2.26 (Affine Subspaces)

One-dimensional affine subspaces are called lines and can be written as $\pmb { y } = \pmb { x } _ { 0 } + \lambda \pmb { b } _ { 1 }$ where $\lambda \in \mathbb { R }$ and $U = \mathrm { s p a n } [ b _ { 1 } ] \ \subseteq \ \mathbb { R } ^ { n }$ is a onedimensional subspace of $\mathbb { R } ^ { n }$ . This means that a line is defined by a support point $\scriptstyle { \pmb x } _ { 0 }$ and a vector $b _ { 1 }$ that defines the direction. See Figure 2.13 for an illustration. 



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



Two-dimensional affine subspaces of $\mathbb { R } ^ { n }$ are called planes. The parametric equation for planes is $\pmb { y } = \pmb { x } _ { 0 } + \lambda _ { 1 } \pmb { b } _ { 1 } + \lambda _ { 2 } \pmb { b } _ { 2 }$ , where $\lambda _ { 1 } , \lambda _ { 2 } \in \mathbb { R }$ and $U = \operatorname { s p a n } [ b _ { 1 } , b _ { 2 } ] \subseteq \mathbb { R } ^ { n }$ . This means that a plane is defined by a support point $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ and two linearly independent vectors $b _ { 1 } , b _ { 2 }$ that span the direction space. 

Figure 2.13 Lines are affine subspaces. Vectors y on a line x<sub>0</sub> + λb<sub>1</sub> lie in an affine subspace $L$ with support point $\scriptstyle { \mathbf { a } } _ { 0 }$ and direction $b _ { 1 }$ 

In $\mathbb { R } ^ { n }$ , the $( n - 1 )$ -dimensional affine subspaces are called hyperplanes, and the corresponding parametric equation is $\begin{array} { r } { \pmb { y } = \pmb { x } _ { 0 } + \sum _ { i = 1 } ^ { n - 1 } \lambda _ { i } \pmb { b } _ { i } } \end{array}$ where $b _ { 1 } , \ldots , b _ { n - 1 }$ form a basis of an $( n - 1 )$ )-dimensional subspace $U$ of $\mathbb { R } ^ { n }$ . This means that a hyperplane is defined by a support point $\scriptstyle { \mathbf { \mathscr { x } } } _ { \mathrm { 0 } }$ and $( n - 1 )$ linearly independent vectors $b _ { 1 } , \ldots , b _ { n - 1 }$ that span the direction space. In $\mathbb { R } ^ { 2 }$ , a line is also a hyperplane. In $\mathbb { R } ^ { 3 }$ , a plane is also a hyperplane. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/92db126f7f6feb3b961b67f57d186f2f9030278179317b190e80cc38be91f8e9.jpg)


Remark (Inhomogeneous systems of linear equations and affine subspaces). For $\pmb { A } \in \mathbb { R } ^ { m \times n }$ and $\pmb { x } \in \mathbb { R } ^ { m }$ , the solution of the system of linear equations $A \lambda = x$ is either the empty set or an affine subspace of $\mathbb { R } ^ { n }$ of dimension $n - \operatorname { r k } ( A )$ . In particular, the solution of the linear equation $\lambda _ { 1 } b _ { 1 } + . . . + \lambda _ { n } b _ { n } = x$ , where $( \lambda _ { 1 } , \ldots , \lambda _ { n } ) \neq ( 0 , \ldots , 0 )$ , is a hyperplane in $\mathbb { R } ^ { n }$ 

In $\mathbb { R } ^ { n }$ , every k-dimensional affine subspace is the solution of an inhomogeneous system of linear equations $\mathbf { { A } } { \boldsymbol { x } } = { \boldsymbol { b } } $ , where $\pmb { A } \in \mathbb { R } ^ { m \times n } , \pmb { b } \in$ $\mathbb { R } ^ { m }$ and rk $\mathbf { \nabla } [ \mathbf { A } ) = n - k$ . Recall that for homogeneous equation systems $\pmb { A } \pmb { x } = \pmb { 0 }$ the solution was a vector subspace, which we can also think of as a special affine space with support point $\mathbf { \mathscr { x } } _ { 0 } = \mathbf { 0 }$ ◇ 

## 2.8.2 Affine Mappings

Similar to linear mappings between vector spaces, which we discussed in Section 2.7, we can define affine mappings between two affine spaces. Linear and affine mappings are closely related. Therefore, many properties that we already know from linear mappings, e.g., that the composition of linear mappings is a linear mapping, also hold for affine mappings. 

Definition 2.26 (Affine Mapping). For two vector spaces $V , W ,$ , a linear 

mapping $\Phi : V \to W$ , and $\mathbf { \pmb { a } } \in W$ , the mapping 

$$
\phi : V \to W\tag{2.132}
$$

$$
\boldsymbol {x} \mapsto \boldsymbol {a} + \Phi (\boldsymbol {x})\tag{2.133}
$$

is an affine mapping from V to W . The vector a is called the translation vector of $\phi .$ 

Every affine mapping $\phi : V \to W$ is also the composition of a linear mapping $\Phi : V \to W$ and a translation $\tau : W \to W$ in $W ,$ , such that $\phi = \tau \circ \Phi$ . The mappings Φ and τ are uniquely determined. 

The composition $\phi ^ { \prime } \circ \phi$ of affine mappings $\phi : V \to W , \phi ^ { \prime } : W \to X$ is affine. 

$\operatorname { I f } \phi$ is bijective, affine mappings keep the geometric structure invariant. They then also preserve the dimension and parallelism. 

## 2.9 Further Reading

There are many resources for learning linear algebra, including the textbooks by Strang (2003), Golan (2007), Axler (2015), and Liesen and Mehrmann (2015). There are also several online resources that we mentioned in the introduction to this chapter. We only covered Gaussian elimination here, but there are many other approaches for solving systems of linear equations, and we refer to numerical linear algebra textbooks by Stoer and Burlirsch (2002), Golub and Van Loan (2012), and Horn and Johnson (2013) for an in-depth discussion. 

In this book, we distinguish between the topics of linear algebra $( \boldsymbol { \mathrm { e . g . } }$ vectors, matrices, linear independence, basis) and topics related to the geometry of a vector space. In Chapter 3, we will introduce the inner product, which induces a norm. These concepts allow us to define angles, lengths and distances, which we will use for orthogonal projections. Projections turn out to be key in many machine learning algorithms, such as linear regression and principal component analysis, both of which we will cover in Chapters 9 and $^ { 1 0 , }$ respectively. 

## Exercises

2.1 We consider $( \mathbb { R } \backslash \{ - 1 \} , \star )$ , where 

$$
a \star b := a b + a + b, \qquad a, b \in \mathbb {R} \backslash \{- 1 \}\tag{2.134}
$$

a. Show that $( \mathbb { R } \backslash \{ - 1 \} , \star )$ is an Abelian group. 

b. Solve 

$$
3 \star x \star x = 1 5
$$

in the Abelian group $( \mathbb { R } \backslash \{ - 1 \} , \star )$ , where ⋆ is defined in (2.134). 

2.2 Let n be in $\mathbb { N } \backslash \{ 0 \}$ . Let $k , x$ be in $\mathbb { Z } .$ . We define the congruence class $\bar { k }$ of the integer k as the set 

$$
\begin{array}{r l} & {\overline {{k}} = \{x \in \mathbb {Z} \mid x - k = 0 (\mathrm{mod} n) \}} \\ & {\quad = \{x \in \mathbb {Z} \mid \exists a \in \mathbb {Z} \colon (x - k = n \cdot a) \}.} \end{array}
$$

We now define $\mathbb { Z } / n \mathbb { Z }$ (sometimes written $\mathbb { Z } _ { n } )$ as the set of all congruence classes modulo n. Euclidean division implies that this set is a finite set containing n elements: 

$$
\mathbb {Z} _ {n} = \{\overline {{0}}, \overline {{1}}, \dots , \overline {{n - 1}} \}
$$

For all ${ \overline { { a } } } , { \overline { { b } } } \in \mathbb { Z } _ { n }$ , we define 

$$
\overline {{a}} \oplus \overline {{b}} := \overline {{a + b}}
$$

a. Show that $( \mathbb { Z } _ { n } , \oplus )$ is a group. Is it Abelian? 

b. We now define another operation $\otimes$ for all a and b in $\mathbb { Z } _ { n }$ as 

$$
\overline {{a}} \otimes \overline {{b}} = \overline {{a \times b}},\tag{2.135}
$$

where $a \times b$ represents the usual multiplication in $\mathbb { Z }$ 

Let $n = 5 \AA$ . Draw the times table of the elements of $\mathbb { Z } _ { 5 } \backslash \{ \overline { { 0 } } \}$ under $\otimes ,$ , i.e., calculate the products $\overline { { a } } \otimes \overline { { b } }$ for all a and b in $\mathbb { Z } _ { 5 } \backslash \{ \overline { { 0 } } \}$ 

Hence, show that $\mathbb { Z } _ { 5 } \backslash \{ \overline { { 0 } } \}$ is closed under $\otimes$ and possesses a neutral element for $\otimes .$ . Display the inverse of all elements in $\mathbb { Z } _ { 5 } \backslash \{ \overline { { 0 } } \}$ under $\otimes .$ Conclude that $( \mathbb { Z } _ { 5 } \backslash \{ \overline { { 0 } } \} , \otimes )$ is an Abelian group. 

c. Show that $( \mathbb { Z } _ { 8 } \backslash \{ \overline { { 0 } } \} , \otimes )$ is not a group. 

d. We recall that the B´ezout theorem states that two integers a and b are relatively prime $( \mathrm { i . e . , } g c d ( a , b ) = 1 )$ if and only if there exist two integers u and v such that au + bv = 1. Show that $( \mathbb { Z } _ { n } \backslash \{ \overline { { 0 } } \} , \otimes )$ is a group if and only i $\mathbf { \dot { \gamma } } _ { n } \in \mathbb { N } \backslash \{ 0 \}$ is prime. 

2.3 Consider the set $\mathcal { G }$ of $3 \times 3$ matrices defined as follows: 

$$
\mathcal {G} = \left\{\left[ \begin{array}{c c c} 1 & x & z \\ 0 & 1 & y \\ 0 & 0 & 1 \end{array} \right] \in \mathbb {R} ^ {3 \times 3}   \middle |   x, y, z \in \mathbb {R} \right\}
$$

We define · as the standard matrix multiplication. 

Is $( { \mathcal { G } } , \cdot )$ a group? If yes, is it Abelian? Justify your answer. 

2.4 Compute the following matrix products, if possible: 

Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 

a. 

$$
\left[ \begin{array}{c c} 1 & 2 \\ 4 & 5 \\ 7 & 8 \end{array} \right] \left[ \begin{array}{c c c} 1 & 1 & 0 \\ 0 & 1 & 1 \\ 1 & 0 & 1 \end{array} \right]
$$

b. 

$$
{\left[ \begin{array}{l l l} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{array} \right]} {\left[ \begin{array}{l l l} 1 & 1 & 0 \\ 0 & 1 & 1 \\ 1 & 0 & 1 \end{array} \right]}
$$

c. 

$$
\left[ \begin{array}{c c c} 1 & 1 & 0 \\ 0 & 1 & 1 \\ 1 & 0 & 1 \end{array} \right] \left[ \begin{array}{c c c} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{array} \right]
$$

d. 

$$
\left[ \begin{array}{c c c c} 1 & 2 & 1 & 2 \\ 4 & 1 & - 1 & - 4 \end{array} \right] \left[ \begin{array}{c c} 0 & 3 \\ 1 & - 1 \\ 2 & 1 \\ 5 & 2 \end{array} \right]
$$

e. 

$$
\left[ \begin{array}{c c} 0 & 3 \\ 1 & - 1 \\ 2 & 1 \\ 5 & 2 \end{array} \right] \left[ \begin{array}{c c c c} 1 & 2 & 1 & 2 \\ 4 & 1 & - 1 & - 4 \end{array} \right]
$$

2.5 Find the set S of all solutions in x of the following inhomogeneous linear systems $\mathbf { { A } } \mathbf { { x } } = \mathbf { { b } } ,$ where A and b are defined as follows: 

a. 

$$
\boldsymbol {A} = \left[ \begin{array}{c c c c} 1 & 1 & - 1 & - 1 \\ 2 & 5 & - 7 & - 5 \\ 2 & - 1 & 1 & 3 \\ 5 & 2 & - 4 & 2 \end{array} \right], \quad \boldsymbol {b} = \left[ \begin{array}{c} 1 \\ - 2 \\ 4 \\ 6 \end{array} \right]
$$

b. 

$$
\boldsymbol {A} = \left[ \begin{array}{c c c c c} 1 & - 1 & 0 & 0 & 1 \\ 1 & 1 & 0 & - 3 & 0 \\ 2 & - 1 & 0 & 1 & - 1 \\ - 1 & 2 & 0 & - 2 & - 1 \end{array} \right], \quad \boldsymbol {b} = \left[ \begin{array}{c} 3 \\ 6 \\ 5 \\ - 1 \end{array} \right]
$$

2.6 Using Gaussian elimination, find all solutions of the inhomogeneous equation system $\mathbf { A } { \boldsymbol { \mathbf { \mathit { x } } } } = \mathbf { \mathit { b } }$ with 

$$
\boldsymbol {A} = \left[ \begin{array}{c c c c c c} 0 & 1 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 & 1 & 0 \\ 0 & 1 & 0 & 0 & 0 & 1 \end{array} \right], \quad \boldsymbol {b} = \left[ \begin{array}{c} 2 \\ - 1 \\ 1 \end{array} \right].
$$

©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 

Find all solutions in $\pmb { x } = \left[ \begin{array} { l } { x _ { 1 } } \\ { x _ { 2 } } \\ { x _ { 3 } } \end{array} \right] \in \mathbb { R } ^ { 3 }$ of the equation system $\mathbf { } A \mathbf { } x \mathrm { ~ = ~ } 1 2 x ,$ where 

$$
\boldsymbol {A} = \left[ \begin{array}{c c c} 6 & 4 & 3 \\ 6 & 0 & 9 \\ 0 & 8 & 0 \end{array} \right]
$$

and $\textstyle \sum _ { i = 1 } ^ { 3 } x _ { i } = 1$ 

2.8 Determine the inverses of the following matrices if possible: 

$$
\boldsymbol {A} = \left[ \begin{array}{c c c} 2 & 3 & 4 \\ 3 & 4 & 5 \\ 4 & 5 & 6 \end{array} \right]
$$

b. 

$$
\boldsymbol {A} = \left[ \begin{array}{c c c c} 1 & 0 & 1 & 0 \\ 0 & 1 & 1 & 0 \\ 1 & 1 & 0 & 1 \\ 1 & 1 & 1 & 0 \end{array} \right]
$$

2.9 Which of the following sets are subspaces of $\mathbb { R } ^ { 3 } ?$ 

a. $A = \{ ( \lambda , \lambda + \mu ^ { 3 } , \lambda - \mu ^ { 3 } ) \mid \lambda , \mu \in \mathbb { R } \}$ 

b. $B = \{ ( \lambda ^ { 2 } , - \lambda ^ { 2 } , 0 ) \mid \lambda \in \mathbb { R } \}$ 

c. $\operatorname { L e t } \gamma \operatorname { b e } \operatorname { i n } \mathbb { R } .$ 

$$
C = \left\{\left(\xi_ {1}, \xi_ {2}, \xi_ {3}\right) \in \mathbb {R} ^ {3} \mid \xi_ {1} - 2 \xi_ {2} + 3 \xi_ {3} = \gamma \right\}
$$

d. $D = \{ ( \xi _ { 1 } , \xi _ { 2 } , \xi _ { 3 } ) \in \mathbb { R } ^ { 3 } \mid \xi _ { 2 } \in \mathbb { Z } \}$ 

2.10 Are the following sets of vectors linearly independent? 

$$
\boldsymbol {x} _ {1} = \left[ \begin{array}{l} 2 \\ - 1 \\ 3 \end{array} \right], \quad \boldsymbol {x} _ {2} = \left[ \begin{array}{l} 1 \\ 1 \\ - 2 \end{array} \right], \quad \boldsymbol {x} _ {3} = \left[ \begin{array}{l} 3 \\ - 3 \\ 8 \end{array} \right]
$$

b. 

$$
\boldsymbol {x} _ {1} = \left[ \begin{array}{l} 1 \\ 2 \\ 1 \\ 0 \\ 0 \end{array} \right], \quad \boldsymbol {x} _ {2} = \left[ \begin{array}{l} 1 \\ 1 \\ 0 \\ 1 \\ 1 \end{array} \right], \quad \boldsymbol {x} _ {3} = \left[ \begin{array}{l} 1 \\ 0 \\ 0 \\ 1 \\ 1 \end{array} \right]
$$

2.11 Write 

$$
\boldsymbol {y} = \left[ \begin{array}{c} 1 \\ - 2 \\ 5 \end{array} \right]
$$

as linear combination of 

$$
\boldsymbol {x} _ {1} = \left[ \begin{array}{l} 1 \\ 1 \\ 1 \end{array} \right], \quad \boldsymbol {x} _ {2} = \left[ \begin{array}{l} 1 \\ 2 \\ 3 \end{array} \right], \quad \boldsymbol {x} _ {3} = \left[ \begin{array}{l} 2 \\ - 1 \\ 1 \end{array} \right]
$$

Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 

Exercises 

2.12 Consider two subspaces of $\mathbb { R } ^ { 4 }$ : 

$$
U _ {1} = \operatorname{span} [ \left[ \begin{array}{c} 1 \\ 1 \\ - 3 \\ 1 \end{array} \right], \left[ \begin{array}{c} 2 \\ - 1 \\ 0 \\ - 1 \end{array} \right], \left[ \begin{array}{c} - 1 \\ 1 \\ - 1 \\ 1 \end{array} \right], U _ {2} = \operatorname{span} [ \left[ \begin{array}{c} - 1 \\ - 2 \\ 2 \\ 1 \end{array} \right], \left[ \begin{array}{c} 2 \\ - 2 \\ 0 \\ 0 \end{array} \right], \left[ \begin{array}{c} - 3 \\ 6 \\ - 2 \\ - 1 \end{array} \right].
$$

Determine a basis of $U _ { 1 } \cap U _ { 2 }$ 

2.13 Consider two subspaces $U _ { 1 }$ and $U _ { 2 }$ , where $U _ { 1 }$ is the solution space of the homogeneous equation system $\pmb { A } _ { 1 } \pmb { x } = \mathbf { 0 }$ and $U _ { 2 }$ is the solution space of the homogeneous equation system $\pmb { A } _ { 2 } \pmb { x } = \mathbf { 0 }$ with 

$$
\boldsymbol {A} _ {1} = \left[ \begin{array}{c c c} 1 & 0 & 1 \\ 1 & - 2 & - 1 \\ 2 & 1 & 3 \\ 1 & 0 & 1 \end{array} \right], \quad \boldsymbol {A} _ {2} = \left[ \begin{array}{c c c} 3 & - 3 & 0 \\ 1 & 2 & 3 \\ 7 & - 5 & 2 \\ 3 & - 1 & 2 \end{array} \right].
$$

a. Determine the dimension of $U _ { 1 } , U _ { 2 }$ 

b. Determine bases of $U _ { 1 }$ and $U _ { 2 }$ . 

c. Determine a basis of $U _ { 1 } \cap U _ { 2 }$ 

2.14 Consider two subspaces $U _ { 1 }$ and $U _ { 2 }$ , where $U _ { 1 }$ is spanned by the columns of $\pmb { A } _ { 1 }$ and $U _ { 2 }$ is spanned by the columns of $A _ { 2 }$ with 

$$
\boldsymbol {A} _ {1} = \left[ \begin{array}{c c c} 1 & 0 & 1 \\ 1 & - 2 & - 1 \\ 2 & 1 & 3 \\ 1 & 0 & 1 \end{array} \right], \quad \boldsymbol {A} _ {2} = \left[ \begin{array}{c c c} 3 & - 3 & 0 \\ 1 & 2 & 3 \\ 7 & - 5 & 2 \\ 3 & - 1 & 2 \end{array} \right].
$$

a. Determine the dimension of $U _ { 1 } , U _ { 2 }$ 

b. Determine bases of $U _ { 1 }$ and $U _ { 2 }$ 

c. Determine a basis of $U _ { 1 } \cap U _ { 2 }$ 

2.15 Let $F = \{ ( x , y , z ) \in \mathbb { R } ^ { 3 } \mid x + y - z = 0 \}$ and $G = \{ ( a - b , a + b , a - 3 b ) \mid a , b \in \mathbb { R } \}$ 

a. Show that F and G are subspaces of $\mathbb { R } ^ { 3 }$ 

b. Calculate $F \cap G$ without resorting to any basis vector. 

c. Find one basis for F and one for $G ,$ calculate $F \cap G$ using the basis vectors previously found and check your result with the previous question. 

2.16 Are the following mappings linear? 

a. Let $a , b \in \mathbb { R } .$ 

$$
\begin{array}{c} \Phi : L ^ {1} ([ a, b ]) \to \mathbb {R} \\ f \mapsto \Phi (f) = \int_ {a} ^ {b} f (x) d x  , \end{array}
$$

where $L ^ { 1 } ( [ a , b ] )$ ) denotes the set of integrable functions on $[ a , b ]$ 

$$
\begin{array}{c} \Phi : C ^ {1} \to C ^ {0} \\ f \mapsto \Phi (f) = f ^ {\prime}, \end{array}
$$

where for $k \geqslant 1 , C ^ { k }$ denotes the set of $k$ times continuously differen tiable functions, and $C ^ { 0 }$ denotes the set of continuous functions. 

©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 

c. 

$$
\begin{array}{c} \Phi : \mathbb {R} \to \mathbb {R} \\ x \mapsto \Phi (x) = \cos (x) \end{array}
$$

d. 

$$
\Phi : \mathbb {R} ^ {3} \to \mathbb {R} ^ {2}
$$

$$
\boldsymbol {x} \mapsto \left[ \begin{array}{c c c} 1 & 2 & 3 \\ 1 & 4 & 3 \end{array} \right] \boldsymbol {x}
$$

e. Let θ be in $[ 0 , 2 \pi [$ and 

$$
\Phi : \mathbb {R} ^ {2} \to \mathbb {R} ^ {2}
$$

$$
\pmb {x} \mapsto \left[ \begin{array}{l l} \cos (\theta) & \sin (\theta) \\ - \sin (\theta) & \cos (\theta) \end{array} \right] \pmb {x}
$$

2.17 Consider the linear mapping 

$$
\Phi : \mathbb {R} ^ {3} \to \mathbb {R} ^ {4}
$$

$$
\Phi \left(\left[ \begin{array}{c} x _ {1} \\ x _ {2} \\ x _ {3} \end{array} \right]\right) = \left[ \begin{array}{c} 3 x _ {1} + 2 x _ {2} + x _ {3} \\ x _ {1} + x _ {2} + x _ {3} \\ x _ {1} - 3 x _ {2} \\ 2 x _ {1} + 3 x _ {2} + x _ {3} \end{array} \right]
$$

Find the transformation matrix $A _ { \Phi }$ . 

Determine rk( $A _ { \Phi } )$ . 

Compute the kernel and image of Φ. What are $\dim ( \ker ( \Phi ) )$ and $\dim ( \operatorname { I m } ( \Phi ) ) ?$ 

2.18 Let $E$ be a vector space. Let $f$ and $g$ be two automorphisms on $E$ such that $f \circ g = \operatorname { i d } _ { E } ~ ( \operatorname { i . e . , } ~ f \circ g$ is the identity mapping id $_ { E } )$ . Show that ker $\left( f \right) =$ ker $( g \circ f ) , \operatorname { I m } ( g ) = \operatorname { I m } ( g \circ f )$ and that ker(f) ∩ Im $( g ) = \{ \mathbf { 0 } _ { E } \}$ 

2.19 Consider an endomorphism Φ : $\mathbb { R } ^ { 3 }  \mathbb { R } ^ { 3 }$ whose transformation matrix (with respect to the standard basis in $\mathbb { R } ^ { 3 } )$ is 

$$
\boldsymbol {A} _ {\Phi} = \left[ \begin{array}{c c c} 1 & 1 & 0 \\ 1 & - 1 & 0 \\ 1 & 1 & 1 \end{array} \right].
$$

a. Determine ker(Φ) and Im(Φ). 

b. Determine the transformation matrix $\tilde { \mathbf { A } } _ { \Phi }$ with respect to the basis 

$$
B = (\left[ \begin{array}{c} 1 \\ 1 \\ 1 \end{array} \right], \left[ \begin{array}{c} 1 \\ 2 \\ 1 \end{array} \right], \left[ \begin{array}{c} 1 \\ 0 \\ 0 \end{array} \right]),
$$

i.e., perform a basis change toward the new basis $B .$ 

2.20 Let us consider $b _ { 1 } , b _ { 2 } , b _ { 1 } ^ { \prime } , b _ { 2 } ^ { \prime }$ , 4 vectors of $\mathbb { R } ^ { 2 }$ expressed in the standard basis of $\mathbb { R } ^ { 2 }$ as 

$$
\boldsymbol {b} _ {1} = \left[ \begin{array}{c} 2 \\ 1 \end{array} \right], \quad \boldsymbol {b} _ {2} = \left[ \begin{array}{c} - 1 \\ - 1 \end{array} \right], \quad \boldsymbol {b} _ {1} ^ {\prime} = \left[ \begin{array}{c} 2 \\ - 2 \end{array} \right], \quad \boldsymbol {b} _ {2} ^ {\prime} = \left[ \begin{array}{c} 1 \\ 1 \end{array} \right]
$$

and let us define two ordered bases $B = ( b _ { 1 } , b _ { 2 } )$ and $B ^ { \prime } = ( b _ { 1 } ^ { \prime } , b _ { 2 } ^ { \prime } )$ of $\textstyle \mathbb { R } ^ { 2 }$ . 

a. Show that B and $B ^ { \prime }$ are two bases of $\mathbb { R } ^ { 2 }$ and draw those basis vectors. 

b. Compute the matrix $P _ { 1 }$ that performs a basis change from $B ^ { \prime }$ to $B .$ . 

c. We consider $_ { c _ { 1 } , c _ { 2 } , c _ { 3 } }$ , three vectors of $\mathbb { R } ^ { 3 }$ defined in the standard basis of $\mathbb { R } ^ { 3 }$ as 

$$
\boldsymbol {c} _ {1} = \left[ \begin{array}{l} 1 \\ 2 \\ - 1 \end{array} \right], \quad \boldsymbol {c} _ {2} = \left[ \begin{array}{l} 0 \\ - 1 \\ 2 \end{array} \right], \quad \boldsymbol {c} _ {3} = \left[ \begin{array}{l} 1 \\ 0 \\ - 1 \end{array} \right]
$$

and we define $C = ( c _ { 1 } , c _ { 2 } , c _ { 3 } )$ 

(i) Show that C is a basis of $\mathbb { R } ^ { 3 }$ , e.g., by using determinants (see Section 4.1). 

(ii) Let us call $C ^ { \prime } = ( c _ { 1 } ^ { \prime } , c _ { 2 } ^ { \prime } , c _ { 3 } ^ { \prime } )$ the standard basis of $\mathbb { R } ^ { 3 }$ . Determine the matrix $\scriptstyle P _ { 2 }$ that performs the basis change from $C$ to $C ^ { \prime }$ . 

d. We consider a homomorphism $\Phi : \mathbb { R } ^ { 2 } \longrightarrow \mathbb { R } ^ { 3 }$ , such that 

$$
\begin{array}{r c l} \Phi (\boldsymbol {b} _ {1} + \boldsymbol {b} _ {2}) & = & \boldsymbol {c} _ {2} + \boldsymbol {c} _ {3} \\ \Phi (\boldsymbol {b} _ {1} - \boldsymbol {b} _ {2}) & = & 2 \boldsymbol {c} _ {1} - \boldsymbol {c} _ {2} + 3 \boldsymbol {c} _ {3} \end{array}
$$

where $B = ( b _ { 1 } , b _ { 2 } )$ and $C = ( c _ { 1 } , c _ { 2 } , c _ { 3 } )$ are ordered bases of $\textstyle \mathbb { R } ^ { 2 }$ and $\mathbb { R } ^ { 3 }$ , respectively. 

Determine the transformation matrix $A _ { \Phi }$ of Φ with respect to the $\mathrm { o r \mathrm { \cdot } }$ dered bases B and C. 

e. Determine $A ^ { \prime } ,$ the transformation matrix of Φ with respect to the bases $B ^ { \prime }$ and $C ^ { \prime }$ . 

f. Let us consider the vector $\pmb { x } \in \mathbb { R } ^ { 2 }$ whose coordinates in $B ^ { \prime }$ are $[ 2 , 3 ] ^ { \top }$ In other words, ${ \pmb x } = 2 b _ { 1 } ^ { \prime } + 3 b _ { 2 } ^ { \prime }$ 

(i) Calculate the coordinates of x in $B .$ . 

(ii) Based on that, compute the coordinates of $\Phi ( { \pmb x } )$ expressed in $C .$ 

(iii) Then, write $\Phi ( { \pmb x } )$ in terms of $\mathbf { } c _ { 1 } ^ { \prime } , c _ { 2 } ^ { \prime } , c _ { 3 } ^ { \prime }$ 

(iv) Use the representation of x in $B ^ { \prime }$ and the matrix $A ^ { \prime }$ to find this result directly. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/21bbf4bc23a7b3c390ae2b3b5eadffbb8b8142e70ba192ca5a33ef2dfb147274.jpg)


## Analytic Geometry

In Chapter 2, we studied vectors, vector spaces, and linear mappings at a general but abstract level. In this chapter, we will add some geometric interpretation and intuition to all of these concepts. In particular, we will look at geometric vectors and compute their lengths and distances or angles between two vectors. To be able to do this, we equip the vector space with an inner product that induces the geometry of the vector space. Inner products and their corresponding norms and metrics capture the intuitive notions of similarity and distances, which we use to develop the support vector machine in Chapter 12. We will then use the concepts of lengths and angles between vectors to discuss orthogonal projections, which will play a central role when we discuss principal component analysis in Chapter 10 and regression via maximum likelihood estimation in Chapter 9. Figure 3.1 gives an overview of how concepts in this chapter are related and how they are connected to other chapters of the book. 

Figure 3.1 A mind map of the concepts introduced in this chapter, along with when they are used in other parts of the book. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/b3f6024bb71d853b9980f0f602aac86178b7cdd38cd6fa062b1720042ff8bdcc.jpg)



This material is published by Cambridge University Press as Mathematics for Machine Learning by Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong (2020). This version is free to view and download for personal use only. Not for re-distribution, re-sale, or use in derivative works. ©by M. P. Deisenroth, A. A. Faisal, and C. S. Ong, 2024. https://mml-book.com.


3.1 Norms 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/858dc9fd3b6f70a2bf4d9a4aaef854ed85a99e471c5463d109e5fbe270b20918.jpg)


Figure 3.3 For different norms, the red lines indicate the set of vectors with norm 1. Left: Manhattan norm; Right: Euclidean distance. 

## 3.1 Norms

When we think of geometric vectors, i.e., directed line segments that start at the origin, then intuitively the length of a vector is the distance of the “end” of this directed line segment from the origin. In the following, we will discuss the notion of the length of vectors using the concept of a norm. 

Definition 3.1 (Norm). A norm on a vector space V is a function 

norm 

$$
\| \cdot \|: V \to \mathbb {R},\tag{3.1}
$$

$$
\boldsymbol {x} \mapsto \| \boldsymbol {x} \|,\tag{3.2}
$$

which assigns each vector x its length $\| \pmb { x } \| \in \mathbb { R }$ , such that for all $\lambda \in \mathbb { R }$ and $\mathbf { { } } x , y \in V $ the following hold: 

length 

Absolutely homogeneous: $\| \lambda \pmb { x } \| = | \lambda | \| \pmb { x } \|$ 

Triangle inequality: $\| { \pmb x } + { \pmb y } \| \leqslant \| { \pmb x } \| + \| { \pmb y } \|$ 

Positive definite: $\| \pmb { x } \| \geqslant 0$ and $\| { \pmb x } \| = 0 \iff { \pmb x } = \mathbf { 0 }$ 

absolutely homogeneous triangle inequality positive definite 

Figure 3.2 Triangle inequality. 

In geometric terms, the triangle inequality states that for any triangle, the sum of the lengths of any two sides must be greater than or equal to the length of the remaining side; see Figure 3.2 for an illustration. Definition 3.1 is in terms of a general vector space V (Section 2.4), but in this book we will only consider a finite-dimensional vector space $\mathbb { R } ^ { n }$ Recall that for a vector $\pmb { x } \in \mathbb { R } ^ { n }$ we denote the elements of the vector using a subscript, that is, $x _ { i }$ is the $i ^ { \mathrm { t h } }$ element of the vector x. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/8438bb7886d7693424538032bd4bf5e6a357483bab3a247138892fea29fd8a7a.jpg)


## Example 3.1 (Manhattan Norm)

The Manhattan norm on $\mathbb { R } ^ { n }$ is defined for $\pmb { x } \in \mathbb { R } ^ { n }$ as 

Manhattan norm 

$$
\left\| \boldsymbol {x} \right\| _ {1} := \sum_ {i = 1} ^ {n} \left| x _ {i} \right|,\tag{3.3}
$$

where $| \cdot |$ is the absolute value. The left panel of Figure 3.3 shows all vectors $\textbf { \em x } \in \ \mathbb { R } ^ { 2 }$ with $\| \pmb { x } \| _ { 1 } = 1$ . The Manhattan norm is also called $\ell _ { 1 } \quad \quad \ell _ { 1 }$ norm norm. 



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



## Example 3.2 (Euclidean Norm)

The Euclidean norm of $\pmb { x } \in \mathbb { R } ^ { n }$ is defined as 

$$
\| \boldsymbol {x} \| _ {2} := \sqrt {\sum_ {i = 1} ^ {n} x _ {i} ^ {2}} = \sqrt {\boldsymbol {x} ^ {\top} \boldsymbol {x}}\tag{3.4}
$$

and computes the Euclidean distance of x from the origin. The right panel of Figure 3.3 shows all vectors $\textbf { \em x } \in \ \mathbb { R } ^ { 2 }$ with $\| \pmb { x } \| _ { 2 } = 1$ . The Euclidean norm is also called $\ell _ { 2 }$ norm. 

Remark. Throughout this book, we will use the Euclidean norm (3.4) by default if not stated otherwise. ◇ 

## 3.2 Inner Products

Inner products allow for the introduction of intuitive geometrical concepts, such as the length of a vector and the angle or distance between two vectors. A major purpose of inner products is to determine whether vectors are orthogonal to each other. 

## 3.2.1 Dot Product

We may already be familiar with a particular type of inner product, the scalar product/dot product in <sup>Rn</sup>, which is given by 

$$
\boldsymbol {x} ^ {\top} \boldsymbol {y} = \sum_ {i = 1} ^ {n} x _ {i} y _ {i}.\tag{3.5}
$$

We will refer to this particular inner product as the dot product in this book. However, inner products are more general concepts with specific properties, which we will now introduce. 

## 3.2.2 General Inner Products

Recall the linear mapping from Section 2.7, where we can rearrange the mapping with respect to addition and multiplication with a scalar. A bilinear mapping Ω is a mapping with two arguments, and it is linear in each argument, i.e., when we look at a vector space V then it holds that for all x, $\pmb { y } , \pmb { z } \in V , \ \lambda , \psi \in \mathbb { R }$ that 

$$
\Omega (\lambda \boldsymbol {x} + \psi \boldsymbol {y}, \boldsymbol {z}) = \lambda \Omega (\boldsymbol {x}, \boldsymbol {z}) + \psi \Omega (\boldsymbol {y}, \boldsymbol {z})\tag{3.6}
$$

$$
\Omega (\boldsymbol {x}, \lambda \boldsymbol {y} + \psi \boldsymbol {z}) = \lambda \Omega (\boldsymbol {x}, \boldsymbol {y}) + \psi \Omega (\boldsymbol {x}, \boldsymbol {z}).\tag{3.7}
$$

Here, (3.6) asserts that Ω is linear in the first argument, and (3.7) asserts that Ω is linear in the second argument (see also (2.87)). 

Definition 3.2. Let V be a vector space and $\Omega : V \times V $ <sup>R</sup> be a bilinear mapping that takes two vectors and maps them onto a real number. Then 

Ω is called symmetric if $\Omega ( x , y ) = \Omega ( y , x )$ for all $x , y \in V , \mathrm { i . e . }$ , the order of the arguments does not matter. 

Ω is called positive definite if 

$$
\forall \boldsymbol {x} \in V \backslash \{\boldsymbol {0} \}: \Omega (\boldsymbol {x}, \boldsymbol {x}) > 0, \quad \Omega (\boldsymbol {0}, \boldsymbol {0}) = 0.\tag{3.8}
$$

Definition 3.3. Let $V$ be a vector space and $\Omega : V \times V $ <sup>R</sup> be a bilinear mapping that takes two vectors and maps them onto a real number. Then 

A positive definite, symmetric bilinear mapping $\Omega : V \times V $ <sup>R</sup> is called an inner product on V . We typically write $\langle { \pmb x } , { \pmb y } \rangle$ instead of $\Omega ( { \pmb x } , { \pmb y } )$ . 

The pair $\left( V , \langle \cdot , \cdot \rangle \right)$ is called an inner product space or (real) vector space with inner product. If we use the dot product defined in (3.5), we call $\left( V , \langle \cdot , \cdot \rangle \right)$ a Euclidean vector space. 

We will refer to these spaces as inner product spaces in this book. 

## Example 3.3 (Inner Product That Is Not the Dot Product)

Consider $V = \mathbb { R } ^ { 2 }$ . If we define 

$$
\langle \boldsymbol {x}, \boldsymbol {y} \rangle := x _ {1} y _ {1} - (x _ {1} y _ {2} + x _ {2} y _ {1}) + 2 x _ {2} y _ {2}\tag{3.9}
$$

then $\langle \cdot , \cdot \rangle$ is an inner product but different from the dot product. The proof will be an exercise. 

## 3.2.3 Symmetric, Positive Definite Matrices

Symmetric, positive definite matrices play an important role in machine learning, and they are defined via the inner product. In Section 4.3, we will return to symmetric, positive definite matrices in the context of matrix decompositions. The idea of symmetric positive semidefinite matrices is key in the definition of kernels (Section $1 2 . 4 )$ 

Consider an n-dimensional vector space V with an inner product $\langle \cdot , \cdot \rangle$ $V \times V \to \mathbb { R }$ (see Definition 3.3) and an ordered basis $B = ( b _ { 1 } , \ldots , b _ { n } )$ of $V .$ . Recall from Section 2.6.1 that any vectors x, $\pmb { y } \in V$ can be written as linear combinations of the basis vectors so that $\begin{array} { r } { \pmb { x } = \sum _ { i = 1 } ^ { n } { \psi _ { i } \pmb { b } _ { i } } \in V } \end{array}$ and $\begin{array} { r } { \pmb { y } = \sum _ { j = 1 } ^ { n } \lambda _ { j } \pmb { b } _ { j } \in V } \end{array}$ for suitable $\psi _ { i } , \lambda _ { j } \in \mathbb { R }$ . Due to the bilinearity of the inner product, it holds for all $\mathbf { { \pmb { x } } } , \mathbf { { \pmb { y } } } \in V$ that 

$$
\langle \boldsymbol {x}, \boldsymbol {y} \rangle = \left\langle \sum_ {i = 1} ^ {n} \psi_ {i} \boldsymbol {b} _ {i}, \sum_ {j = 1} ^ {n} \lambda_ {j} \boldsymbol {b} _ {j} \right\rangle = \sum_ {i = 1} ^ {n} \sum_ {j = 1} ^ {n} \psi_ {i} \left\langle \boldsymbol {b} _ {i}, \boldsymbol {b} _ {j} \right\rangle \lambda_ {j} = \hat {\boldsymbol {x}} ^ {\top} \boldsymbol {A} \hat {\boldsymbol {y}},\tag{3.10}
$$

where $A _ { i j } : = \langle b _ { i } , b _ { j } \rangle$ and $\hat { \pmb x } , \hat { \pmb y }$ are the coordinates of x and y with respect to the basis B. This implies that the inner product $\langle \cdot , \cdot \rangle$ is uniquely determined through A. The symmetry of the inner product also means that A is symmetric. Furthermore, the positive definiteness of the inner product implies that 

$$
\forall \boldsymbol {x} \in V \backslash \{\mathbf {0} \}: \boldsymbol {x} ^ {\top} \boldsymbol {A} \boldsymbol {x} > 0.\tag{3.11}
$$

Definition 3.4 (Symmetric, Positive Definite Matrix). A symmetric matrix $\pmb { A } \in \mathbb { R } ^ { n \times n }$ that satisfies (3.11) is called symmetric, positive definite, or just positive definite. If only <sup>⩾</sup> holds in (3.11), then A is called symmetric, positive semidefinite. 

Example 3.4 (Symmetric, Positive Definite Matrices) Consider the matrices 

$$
\boldsymbol {A} _ {1} = \left[ \begin{array}{c c} 9 & 6 \\ 6 & 5 \end{array} \right], \quad \boldsymbol {A} _ {2} = \left[ \begin{array}{c c} 9 & 6 \\ 6 & 3 \end{array} \right].\tag{3.12}
$$

$\pmb { A } _ { 1 }$ is positive definite because it is symmetric and 

$$
\pmb {x} ^ {\top} \pmb {A} _ {1} \pmb {x} = \left[ \begin{array}{c c} x _ {1} & x _ {2} \end{array} \right] \left[ \begin{array}{c c} 9 & 6 \\ 6 & 5 \end{array} \right] \left[ \begin{array}{c} x _ {1} \\ x _ {2} \end{array} \right]\tag{3.13a}
$$

$$
= 9 x _ {1} ^ {2} + 1 2 x _ {1} x _ {2} + 5 x _ {2} ^ {2} = (3 x _ {1} + 2 x _ {2}) ^ {2} + x _ {2} ^ {2} > 0\tag{3.13b}
$$

for all ${ \pmb x } \in V \backslash \{ { \bf 0 } \}$ . In contrast, $A _ { 2 }$ is symmetric but not positive definite because $x ^ { \top } A _ { 2 } { \dot { x } } = 9 x _ { 1 } ^ { 2 } + 1 2 x _ { 1 } x _ { 2 } + 3 x _ { 2 } ^ { 2 } = ( 3 x _ { 1 } + 2 x _ { 2 } ) ^ { 2 } - x _ { 2 } ^ { 2 }$ can be less than 0, e.g., for $\pmb { x } = [ 2 , - 3 ] ^ { \top }$ 

If $A \in \mathbb { R } ^ { n \times n }$ is symmetric, positive definite, then 

$$
\langle \boldsymbol {x}, \boldsymbol {y} \rangle = \hat {\boldsymbol {x}} ^ {\top} \boldsymbol {A} \hat {\boldsymbol {y}}\tag{3.14}
$$

defines an inner product with respect to an ordered basis $B ,$ where xˆ and $\hat { y }$ are the coordinate representations of $\mathbf { { } } x , y \in V $ with respect to B. 

Theorem 3.5. For a real-valued, finite-dimensional vector space $V$ and an ordered basis B of V , it holds that $\langle \cdot , \cdot \rangle : V \times V  \mathbb { R }$ is an inner product if and only if there exists a symmetric, positive definite matrix $A \in \mathbb { R } ^ { n \times n }$ with 

$$
\langle \boldsymbol {x}, \boldsymbol {y} \rangle = \hat {\boldsymbol {x}} ^ {\top} \boldsymbol {A} \hat {\boldsymbol {y}}.\tag{3.15}
$$

The following properties hold if $\pmb { A } \in \mathbb { R } ^ { n \times n }$ is symmetric and positive definite: 

The null space (kernel) of A consists only of 0 because ${ \pmb x } ^ { \top } { \pmb A } { \pmb x } > 0$ for all $\mathbf { \boldsymbol { x } } \neq \mathbf { \boldsymbol { 0 } }$ . This implies that Ax = 0 if $\mathbf { \boldsymbol { x } } \neq \mathbf { 0 }$ 

The diagonal elements $a _ { i i }$ of A are positive because $a _ { i i } = e _ { i } ^ { \top } A e _ { i } > 0$ , where $e _ { i }$ is the ith vector of the standard basis in $\mathbb { R } ^ { n }$ 

## 3.3 Lengths and Distances

In Section 3.1, we already discussed norms that we can use to compute the length of a vector. Inner products and norms are closely related in the sense that any inner product induces a norm 

$$
\| \boldsymbol {x} \| := \sqrt {\langle \boldsymbol {x} , \boldsymbol {x} \rangle}\tag{3.16}
$$

in a natural way, such that we can compute lengths of vectors using the inner product. However, not every norm is induced by an inner product. The Manhattan norm (3.3) is an example of a norm without a corresponding inner product. In the following, we will focus on norms that are induced by inner products and introduce geometric concepts, such as lengths, distances, and angles. 

Remark (Cauchy-Schwarz Inequality). For an inner product vector space $\left( V , \langle \cdot , \cdot \rangle \right)$ the induced norm $\| \cdot \|$ satisfies the Cauchy-Schwarz inequality 

$$
| \langle \boldsymbol {x}, \boldsymbol {y} \rangle | \leqslant \| \boldsymbol {x} \| \| \boldsymbol {y} \|.\tag{3.17}
$$

## Example 3.5 (Lengths of Vectors Using Inner Products)

In geometry, we are often interested in lengths of vectors. We can now use an inner product to compute them using (3.16). Let us take $\pmb { x } = [ 1 , 1 ] ^ { \top } \in$ $\mathbb { R } ^ { 2 }$ . If we use the dot product as the inner product, with (3.16) we obtain 

$$
\| \boldsymbol {x} \| = \sqrt {\boldsymbol {x} ^ {\top} \boldsymbol {x}} = \sqrt {1 ^ {2} + 1 ^ {2}} = \sqrt {2}\tag{3.18}
$$

as the length of x. Let us now choose a different inner product: 

$$
\langle \boldsymbol {x}, \boldsymbol {y} \rangle := \boldsymbol {x} ^ {\top} \left[ \begin{array}{c c} 1 & - \frac {1}{2} \\ - \frac {1}{2} & 1 \end{array} \right] \boldsymbol {y} = x _ {1} y _ {1} - \frac {1}{2} (x _ {1} y _ {2} + x _ {2} y _ {1}) + x _ {2} y _ {2}  .\tag{3.19}
$$

If we compute the norm of a vector, then this inner product returns smaller values than the dot product if $x _ { 1 }$ and $x _ { 2 }$ have the same sign (and $x _ { 1 } x _ { 2 } >$ 0); otherwise, it returns greater values than the dot product. With this inner product, we obtain 

$$
\langle \boldsymbol {x}, \boldsymbol {x} \rangle = x _ {1} ^ {2} - x _ {1} x _ {2} + x _ {2} ^ {2} = 1 - 1 + 1 = 1 \implies \| \boldsymbol {x} \| = \sqrt {1} = 1,\tag{3.20}
$$

such that x is “shorter” with this inner product than with the dot product. 

Definition 3.6 (Distance and Metric). Consider an inner product space $\left( V , \langle \cdot , \cdot \rangle \right)$ . Then 

$$
d (\boldsymbol {x}, \boldsymbol {y}) := \| \boldsymbol {x} - \boldsymbol {y} \| = \sqrt {\langle \boldsymbol {x} - \boldsymbol {y} , \boldsymbol {x} - \boldsymbol {y} \rangle}\tag{3.21}
$$

is called the distance between x and $\textbf {  { y } }$ for x, $\textit { \textbf { y } } \in \textit { V }$ . If we use the dot product as the inner product, then the distance is called Euclidean distance. 

The mapping 

$$
d: V \times V \to \mathbb {R}\tag{3.22}
$$

$$
(\boldsymbol {x}, \boldsymbol {y}) \mapsto d (\boldsymbol {x}, \boldsymbol {y})\tag{3.23}
$$

is called a metric. 

Remark. Similar to the length of a vector, the distance between vectors does not require an inner product: a norm is sufficient. If we have a norm induced by an inner product, the distance may vary depending on the choice of the inner product. ◇ 

A metric d satisfies the following: 

1. d is positive definite, i.e., $d ( { \pmb x } , { \pmb y } ) \geqslant 0$ for all $\mathbf { { } } x , y \in V \mathbf { { } }$ and $d ( { \pmb x } , { \pmb y } ) =$ $0 \iff x = y$ 

2. d is symmetric, i.e., $d ( \pmb { x } , \pmb { y } ) = d ( \pmb { y } , \pmb { x } )$ for all $\mathbf { { } } x , y \in V $ 

3. Triangle inequality: $d ( \pmb { x } , z ) \leqslant d ( \pmb { x } , \pmb { y } ) + d ( \pmb { y } , z )$ for all $\pmb { x } , \pmb { y } , \pmb { z } \in V$ 

Remark. At first glance, the lists of properties of inner products and metrics look very similar. However, by comparing Definition 3.3 with Definition 3.6 we observe that $\langle { \pmb x } , { \pmb y } \rangle$ and $d ( { \pmb x } , { \pmb y } )$ behave in opposite directions. Very similar x and $\textbf {  { y } }$ will result in a large value for the inner product and a small value for the metric. ◇ 

Figure 3.4 When restricted to $[ 0 , \pi ]$ then $f ( \omega ) = \cos ( \omega )$ returns a unique number in the interval [−1, 1]. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/71ca38215a13380f80bb65b3d757e5578a596384e0a12a5e2d1f8cb05e3bbec2.jpg)


## 3.4 Angles and Orthogonality

In addition to enabling the definition of lengths of vectors, as well as the distance between two vectors, inner products also capture the geometry of a vector space by defining the angle $\omega$ between two vectors. We use the Cauchy-Schwarz inequality (3.17) to define angles $\omega$ in inner product spaces between two vectors x, y, and this notion coincides with our intuition in $\mathbb { R } ^ { 2 }$ and $\mathbb { R } ^ { 3 }$ . Assume that $\pmb { x } \neq \mathbf { 0 } , \pmb { y } \neq \mathbf { 0 }$ . Then 

$$
- 1 \leqslant \frac {\langle \boldsymbol {x} , \boldsymbol {y} \rangle}{\| \boldsymbol {x} \| \| \boldsymbol {y} \|} \leqslant 1.\tag{3.24}
$$

Therefore, there exists a unique $\omega \in [ 0 , \pi ]$ , illustrated in Figure 3.4, with 

$$
\cos \omega = \frac {\langle \boldsymbol {x} , \boldsymbol {y} \rangle}{\| \boldsymbol {x} \| \| \boldsymbol {y} \|}.\tag{3.25}
$$

angle 

The number $\omega$ is the angle between the vectors $_ { \pmb { x } }$ and $\mathbf { \pmb { y } } .$ . Intuitively, the angle between two vectors tells us how similar their orientations are. For example, using the dot product, the angle between x and $\begin{array} { r } { \pmb { y } = 4 \pmb { x } } \end{array}$ , i.e., y is a scaled version of $^ { x , }$ is 0: Their orientation is the same. 

## Example 3.6 (Angle between Vectors)

Let us compute the angle between $\pmb { x } = [ 1 , 1 ] ^ { \top } \in \mathbb { R } ^ { 2 }$ and $\pmb { y } = [ 1 , 2 ] ^ { \top } \in \mathbb { R } ^ { 2 } .$ see Figure 3.5, where we use the dot product as the inner product. Then we get 

Figure 3.5 The angle ω between two vectors x, y is computed using the inner product. 

$$
\cos \omega = \frac {\langle \boldsymbol {x} , \boldsymbol {y} \rangle}{\sqrt {\langle \boldsymbol {x} , \boldsymbol {x} \rangle \langle \boldsymbol {y} , \boldsymbol {y} \rangle}} = \frac {\boldsymbol {x} ^ {\top} \boldsymbol {y}}{\sqrt {\boldsymbol {x} ^ {\top} \boldsymbol {x y} ^ {\top} \boldsymbol {y}}} = \frac {3}{\sqrt {1 0}},\tag{3.26}
$$

and the angle between the two vectors is arccos $\left( { \frac { 3 } { \sqrt { 1 0 } } } \right) \approx 0 . 3 2 { \mathrm { r a d } }$ , which corresponds to about $1 8 ^ { \circ }$ 

A key feature of the inner product is that it also allows us to characterize vectors that are orthogonal. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/e933921813fab44b8364a966cdb289ab9043be4f04ec0997a8fe644aca9370ab.jpg)


Definition 3.7 (Orthogonality). Two vectors x and y are orthogonal if and only if $\langle { \pmb x } , { \pmb y } \rangle = 0 .$ , and we write $\mathbf {  { x } } \perp \mathbf {  { y } }$ . If additionally $\| { \pmb x } \| = 1 = \| { \pmb y } \|$ 2 i.e., the vectors are unit vectors, then x and y are orthonormal. 

orthonormal 

An implication of this definition is that the 0-vector is orthogonal to every vector in the vector space. 

Remark. Orthogonality is the generalization of the concept of perpendic ularity to bilinear forms that do not have to be the dot product. In our context, geometrically, we can think of orthogonal vectors as having a right angle with respect to a specific inner product. 

## Example 3.7 (Orthogonal Vectors)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/8a343639640f722c92b60f9fc71b52604b11ba4803c652e63b1269519de74672.jpg)



Figure 3.6 The angle ω between two vectors x, y can change depending on the inner product.


Consider two vectors $\pmb { x } = [ 1 , 1 ] ^ { \top } , \pmb { y } = [ - 1 , 1 ] ^ { \top } \in \mathbb { R } ^ { 2 }$ ; see Figure 3.6. We are interested in determining the angle ω between them using two different inner products. Using the dot product as the inner product yields an angle $\omega$ between x and $\textbf {  { y } }$ of $9 0 ^ { \circ }$ , such that $\textbf { \em x } \perp \textbf { \em y }$ . However, if we choose the inner product 

$$
\langle \boldsymbol {x}, \boldsymbol {y} \rangle = \boldsymbol {x} ^ {\top} \left[ \begin{array}{c c} 2 & 0 \\ 0 & 1 \end{array} \right] \boldsymbol {y},\tag{3.27}
$$



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



we get that the angle ω between x and $\textbf {  { y } }$ is given by 

$$
\cos \omega = \frac {\langle \boldsymbol {x} , \boldsymbol {y} \rangle}{\| \boldsymbol {x} \| \| \boldsymbol {y} \|} = - \frac {1}{3} \implies \omega \approx 1. 9 1 \mathrm{rad} \approx 1 0 9. 5 ^ {\circ},\tag{3.28}
$$

and x and y are not orthogonal. Therefore, vectors that are orthogonal with respect to one inner product do not have to be orthogonal with respect to a different inner product. 

Definition 3.8 (Orthogonal Matrix). A square matrix $\pmb { A } \in \mathbb { R } ^ { n \times n }$ is an orthogonal matrix if and only if its columns are orthonormal so that 

$$
\boldsymbol {A} \boldsymbol {A} ^ {\top} = \boldsymbol {I} = \boldsymbol {A} ^ {\top} \boldsymbol {A},\tag{3.29}
$$

which implies that 

$$
\boldsymbol {A} ^ {- 1} = \boldsymbol {A} ^ {\top},\tag{3.30}
$$

i.e., the inverse is obtained by simply transposing the matrix. 

Transformations by orthogonal matrices are special because the length of a vector x is not changed when transforming it using an orthogonal matrix A. For the dot product, we obtain 

$$
\left\| \boldsymbol {A} \boldsymbol {x} \right\| ^ {2} = (\boldsymbol {A} \boldsymbol {x}) ^ {\top} (\boldsymbol {A} \boldsymbol {x}) = \boldsymbol {x} ^ {\top} \boldsymbol {A} ^ {\top} \boldsymbol {A} \boldsymbol {x} = \boldsymbol {x} ^ {\top} \boldsymbol {I} \boldsymbol {x} = \boldsymbol {x} ^ {\top} \boldsymbol {x} = \| \boldsymbol {x} \| ^ {2}.\tag{3.31}
$$

Moreover, the angle between any two vectors x, y, as measured by their inner product, is also unchanged when transforming both of them using an orthogonal matrix A. Assuming the dot product as the inner product, the angle of the images Ax and $\pmb { A } \pmb { y }$ is given as 

$$
\cos \omega = \frac {(\boldsymbol {A} \boldsymbol {x}) ^ {\top} (\boldsymbol {A} \boldsymbol {y})}{\| \boldsymbol {A} \boldsymbol {x} \| \| \boldsymbol {A} \boldsymbol {y} \|} = \frac {\boldsymbol {x} ^ {\top} \boldsymbol {A} ^ {\top} \boldsymbol {A} \boldsymbol {y}}{\sqrt {\boldsymbol {x} ^ {\top} \boldsymbol {A} ^ {\top} \boldsymbol {A} \boldsymbol {x} \boldsymbol {y} ^ {\top} \boldsymbol {A} ^ {\top} \boldsymbol {A} \boldsymbol {y}}} = \frac {\boldsymbol {x} ^ {\top} \boldsymbol {y}}{\| \boldsymbol {x} \| \| \boldsymbol {y} \|},\tag{3.32}
$$

which gives exactly the angle between x and $\mathbf { \nabla } _ { \mathbf { \mu } _ { y } . }$ This means that orthogonal matrices A with $\pmb { A } ^ { \top } = \pmb { A } ^ { - 1 }$ preserve both angles and distances. It turns out that orthogonal matrices define transformations that are rotations (with the possibility of flips). In Section 3.9, we will discuss more details about rotations. 

## 3.5 Orthonormal Basis

In Section 2.6.1, we characterized properties of basis vectors and found that in an n-dimensional vector space, we need n basis vectors, i.e., n vectors that are linearly independent. In Sections 3.3 and 3.4, we used inner products to compute the length of vectors and the angle between vectors. In the following, we will discuss the special case where the basis vectors are orthogonal to each other and where the length of each basis vector is 1. We will call this basis then an orthonormal basis. 

## 3.6 Orthogonal Complement

Let us introduce this more formally. 

Definition 3.9 (Orthonormal Basis). Consider an n-dimensional vector space $V$ and a basis $\{ b _ { 1 } , \ldots , b _ { n } \}$ of $V$ . If 

$$
\langle \boldsymbol {b} _ {i}, \boldsymbol {b} _ {j} \rangle = 0 \quad \text { for } i \neq j\tag{3.33}
$$

$$
\langle \pmb {b} _ {i}, \pmb {b} _ {i} \rangle = 1\tag{3.34}
$$

for all $i , j = 1 , \ldots , n$ then the basis is called an orthonormal basis (ONB). If only (3.33) is satisfied, then the basis is called an orthogonal basis. Note that (3.34) implies that every basis vector has length/norm 1. 

Recall from Section 2.6.1 that we can use Gaussian elimination to find a basis for a vector space spanned by a set of vectors. Assume we are given a set $\{ \tilde { b } _ { 1 } , \dots , \tilde { b } _ { n } \}$ of non-orthogonal and unnormalized basis vectors. We concatenate them into a matrix $\tilde { \pmb { B } } = [ \tilde { \pmb { b } } _ { 1 } , \dots , \tilde { \pmb { b } } _ { n } ]$ and apply Gaussian elimination to the augmented matrix (Section 2.3.2) $[ \tilde { B } \tilde { B } ^ { ' } | \tilde { B } ]$ to obtain an orthonormal basis. This constructive way to iteratively build an orthonor mal basis $\{ b _ { 1 } , \ldots , b _ { n } \}$ is called the Gram-Schmidt process (Strang, 2003). 

## Example 3.8 (Orthonormal Basis)

The canonical/standard basis for a Euclidean vector space $\mathbb { R } ^ { n }$ is an $\mathrm { o r ^ { - } }$ thonormal basis, where the inner product is the dot product of vectors. 

In $\mathbb { R } ^ { 2 }$ , the vectors 

$$
\boldsymbol {b} _ {1} = \frac {1}{\sqrt {2}} \left[ \begin{array}{c} 1 \\ 1 \end{array} \right], \quad \boldsymbol {b} _ {2} = \frac {1}{\sqrt {2}} \left[ \begin{array}{c} 1 \\ - 1 \end{array} \right]\tag{3.35}
$$

form an orthonormal basis since $\pmb { b } _ { 1 } ^ { \top } \pmb { b } _ { 2 } = 0$ and $\| \pmb { b } _ { 1 } \| = 1 = \| \pmb { b } _ { 2 } \|$ 

We will exploit the concept of an orthonormal basis in Chapter 12 and Chapter 10 when we discuss support vector machines and principal component analysis. 

## 3.6 Orthogonal Complement

Having defined orthogonality, we will now look at vector spaces that are orthogonal to each other. This will play an important role in Chapter 10, when we discuss linear dimensionality reduction from a geometric perspective. 

Consider a D-dimensional vector space V and an M-dimensional subspace $U \subseteq V$ . Then its orthogonal complement $U ^ { \perp }$ is a $( D - M )$ -dimensional subspace of $V$ and contains all vectors in $V$ that are orthogonal to every vector in U. Furthermore, $U \cap U ^ { \bot } = \{ { \bf 0 } \}$ so that any vector $\pmb { x } \in V$ can be 

Figure 3.7 A plane U in a three-dimensional vector space can be described by its normal vector, which spans its orthogonal complement $U ^ { \perp }$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/770929b083911ae5f0854ea111ab448d82fde74c9f1b75f1d11bcdd430eb9900.jpg)


uniquely decomposed into 

$$
\boldsymbol {x} = \sum_ {m = 1} ^ {M} \lambda_ {m} \boldsymbol {b} _ {m} + \sum_ {j = 1} ^ {D - M} \psi_ {j} \boldsymbol {b} _ {j} ^ {\perp}, \quad \lambda_ {m}, \psi_ {j} \in \mathbb {R},\tag{3.36}
$$

where $\left( \pmb { b } _ { 1 } , \dots , \pmb { b } _ { M } \right)$ is a basis of $U$ and $( \pmb { b } _ { 1 } ^ { \perp } , \ldots , \pmb { b } _ { D - M } ^ { \perp } )$ is a basis of $U ^ { \perp }$ 

Therefore, the orthogonal complement can also be used to describe a plane $U$ (two-dimensional subspace) in a three-dimensional vector space. More specifically, the vector w with $\| \pmb { w } \| = 1$ , which is orthogonal to the plane $U ,$ is the basis vector of $U ^ { \perp }$ . Figure 3.7 illustrates this setting. All vectors that are orthogonal to w must (by construction) lie in the plane $U .$ . The vector w is called the normal vector of $U$ 

Generally, orthogonal complements can be used to describe hyperplanes in n-dimensional vector and affine spaces. 

## 3.7 Inner Product of Functions

Thus far, we looked at properties of inner products to compute lengths, angles and distances. We focused on inner products of finite-dimensional vectors. In the following, we will look at an example of inner products of a different type of vectors: inner products of functions. 

The inner products we discussed so far were defined for vectors with a finite number of entries. We can think of a vector $\pmb { x } \in \mathbb { R } ^ { n }$ as a function with n function values. The concept of an inner product can be generalized to vectors with an infinite number of entries (countably infinite) and also continuous-valued functions (uncountably infinite). Then the sum over individual components of vectors (see Equation (3.5) for example) turns into an integral. 

An inner product of two functions $u : \mathbb { R }  \mathbb { R }$ and $v : \mathbb { R }  \mathbb { R }$ can be defined as the definite integral 

$$
\langle u, v \rangle := \int_ {a} ^ {b} u (x) v (x) d x\tag{3.37}
$$

Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 

## 3.8 Orthogonal Projections

for lower and upper limits $a , b < \infty$ , respectively. As with our usual inner product, we can define norms and orthogonality by looking at the inner product. If (3.37) evaluates to 0, the functions u and v are orthogonal. To make the preceding inner product mathematically precise, we need to take care of measures and the definition of integrals, leading to the definition of a Hilbert space. Furthermore, unlike inner products on finite-dimensional vectors, inner products on functions may diverge (have infinite value). All this requires diving into some more intricate details of real and functional analysis, which we do not cover in this book. 

## Example 3.9 (Inner Product of Functions)

If we choose $u = \sin ( x )$ and $v = \cos ( x )$ , the integrand $f ( x ) = u ( x ) v ( x )$ of (3.37), is shown in Figure 3.8. We see that this function is odd, i.e., $f ( - x ) = - f ( x )$ . Therefore, the integral with limits $a = - \pi , b = \pi$ of this product evaluates to 0. Therefore, sin and cos are orthogonal functions. 

Figure 3.8 f(x) = sin(x) cos(x). 

Remark. It also holds that the collection of functions 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/13c6727df5813b113aba97d7190cdffc4d40e588c6e66be5e8a1ad8c2b76511d.jpg)


$$
\{1, \cos (x), \cos (2 x), \cos (3 x), \dots \}\tag{3.38}
$$

is orthogonal if we integrate from π to π, i.e., any pair of functions are orthogonal to each other. The collection of functions in (3.38) spans a large subspace of the functions that are even and periodic on $[ - \pi , \pi )$ , and projecting functions onto this subspace is the fundamental idea behind Fourier series. 

In Section $6 . 4 . 6 ,$ , we will have a look at a second type of unconventional inner products: the inner product of random variables. 

## 3.8 Orthogonal Projections

Projections are an important class of linear transformations (besides rotations and reflections) and play an important role in graphics, coding theory, statistics and machine learning. In machine learning, we often deal with data that is high-dimensional. High-dimensional data is often hard to analyze or visualize. However, high-dimensional data quite often possesses the property that only a few dimensions contain most information, and most other dimensions are not essential to describe key properties of the data. When we compress or visualize high-dimensional data, we will lose information. To minimize this compression loss, we ideally find the most informative dimensions in the data. As discussed in Chapter 1, data can be represented as vectors, and in this chapter, we will discuss some of the fundamental tools for data compression. More specifically, we can project the original high-dimensional data onto a lower-dimensional feature space and work in this lower-dimensional space to learn more about the dataset and extract relevant patterns. For example, machine learning algorithms, such as principal component analysis (PCA) by Pearson (1901) and Hotelling (1933) and deep neural networks (e.g., deep auto-encoders (Deng et al., 2010)), heavily exploit the idea of dimensionality reduction. In the following, we will focus on orthogonal projections, which we will use in Chapter 10 for linear dimensionality reduction and in Chapter 12 for classification. Even linear regression, which we discuss in Chapter 9, can be interpreted using orthogonal projections. For a given lower-dimensional subspace, orthogonal projections of high-dimensional data retain as much information as possible and minimize the difference/ error between the original data and the corresponding projection. An illustration of such an orthogonal projection is given in Figure 3.9. Before we detail how to obtain these projections, let us define what a projection actually is. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/57afe34b39ef8f17dafbdf993cdf6a34261eb7e49fb50284ac9b843a7ae29c27.jpg)


Definition 3.10 (Projection). Let V be a vector space and $U \subseteq V$ a subspace of V . A linear mapping $\pi : V  U$ is called a projection if $\pi ^ { 2 } = \pi \circ \pi = \pi$ 

Since linear mappings can be expressed by transformation matrices (see Section 2.7), the preceding definition applies equally to a special kind of transformation matrices, the projection matrices $P _ { \pi }$ , which exhibit the property that $P _ { \pi } ^ { 2 } = P _ { \pi }$ 

In the following, we will derive orthogonal projections of vectors in the inner product space $\left( \mathbb { R } ^ { n } , \langle \cdot , \cdot \rangle \right)$ onto subspaces. We will start with onedimensional subspaces, which are also called lines. If not mentioned otherwise, we assume the dot product $\langle \pmb { x } , \pmb { y } \rangle = \pmb { x } ^ { \top } \pmb { y }$ as the inner product. 

## 3.8.1 Projection onto One-Dimensional Subspaces (Lines)

Assume we are given a line (one-dimensional subspace) through the origin with basis vector $b ~ \in ~ \mathbb { R } ^ { n }$ . The line is a one-dimensional subspace $U \subseteq \mathbb { R } ^ { n }$ spanned by b. When we project $\pmb { x } \in \mathbb { R } ^ { n }$ onto $U _ { i }$ , we seek the vector $\pi _ { U } ( { \pmb x } ) \in U$ that is closest to x. Using geometric arguments, let 

## 3.8 Orthogonal Projections

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/fd2f39f66d90da5a623089b24c30ed6d4c5dc44e562ee9966ca5b5ee9a735ce4.jpg)



(a) Projection of $\mathbf { r } \in \mathbb { R } ^ { 2 }$ onto a subspace U with basis vector b.



83


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/6cb1f2230a37553d55ede777dbe62bc4940f5788756f4814f56bf3cb5c65da3e.jpg)



Figure 3.10 Examples of projections onto one-dimensional subspaces.



(b) Projection of a two-dimensional vector x with $\| \pmb { x } \| = 1$ onto a one-dimensional subspace spanned by b.


us characterize some properties of the projection $\pi _ { U } ( { \pmb x } )$ (Figure 3.10(a) serves as an illustration): 

The projection $\pi _ { U } ( { \pmb x } )$ is closest to ${ \mathbf { } } ^ { \mathbf { } } \mathbf { { \mathbf { { x } } } } ,$ where “closest” implies that the distance $\| { \pmb x } - \pi _ { U } ( { \pmb x } ) \|$ is minimal. It follows that the segment $\pi _ { U } ( { \pmb x } ) - { \pmb x }$ from $\pi _ { U } ( { \pmb x } )$ to x is orthogonal to $U$ , and therefore the basis vector b of $U$ . The orthogonality condition yields $\langle \pi _ { U } ( { \pmb x } ) - { \pmb x } , { \pmb b } \rangle = 0$ since angles between vectors are defined via the inner product. 

The projection $\pi _ { U } ( { \pmb x } )$ of x onto $U$ must be an element of U and, therefore, a multiple of the basis vector b that spans U . Hence, $\pi _ { U } ( { \pmb x } ) = \lambda { \pmb b }$ 2 for some $\lambda \in \mathbb { R }$ 

In the following three steps, we determine the coordinate $\lambda ,$ , the projection $\pi _ { U } ( { \pmb x } ) \in U$ , and the projection matrix ${ \pmb P } _ { \pi }$ that maps any $\mathbf { x } \in \mathbb { R } ^ { n }$ onto $U { : }$ 

1. Finding the coordinate $\lambda .$ The orthogonality condition yields 

$$
\langle \boldsymbol {x} - \pi_ {U} (\boldsymbol {x}), \boldsymbol {b} \rangle = 0 \stackrel {{\pi_ {U} (\boldsymbol {x}) = \lambda \boldsymbol {b}}} {{\Longleftrightarrow}} \langle \boldsymbol {x} - \lambda \boldsymbol {b}, \boldsymbol {b} \rangle = 0.\tag{3.39}
$$

We can now exploit the bilinearity of the inner product and arrive at 

$$
\langle \boldsymbol {x}, \boldsymbol {b} \rangle - \lambda \langle \boldsymbol {b}, \boldsymbol {b} \rangle = 0 \iff \lambda = \frac {\langle \boldsymbol {x} , \boldsymbol {b} \rangle}{\langle \boldsymbol {b} , \boldsymbol {b} \rangle} = \frac {\langle \boldsymbol {b} , \boldsymbol {x} \rangle}{\| \boldsymbol {b} \| ^ {2}}.\tag{3.40}
$$

In the last step, we exploited the fact that inner products are symmetric. If we choose $\langle \cdot , \cdot \rangle$ to be the dot product, we obtain 

$$
\lambda = \frac {\boldsymbol {b} ^ {\top} \boldsymbol {x}}{\boldsymbol {b} ^ {\top} \boldsymbol {b}} = \frac {\boldsymbol {b} ^ {\top} \boldsymbol {x}}{\| \boldsymbol {b} \| ^ {2}}.\tag{3.41}
$$

$\operatorname { I f } \left\| \boldsymbol { b } \right\| = 1$ , then the coordinate λ of the projection is given by ${ \pmb b } ^ { \top } { \pmb x }$ 

©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 

2. Finding the projection point $\pi _ { U } ( { \pmb x } ) \in U$ . Since $\pi _ { U } ( { \pmb x } ) = \lambda { \pmb b } .$ , we immediately obtain with (3.40) that 

$$
\pi_ {U} (\boldsymbol {x}) = \lambda \boldsymbol {b} = \frac {\langle \boldsymbol {x} , \boldsymbol {b} \rangle}{\| \boldsymbol {b} \| ^ {2}} \boldsymbol {b} = \frac {\boldsymbol {b} ^ {\top} \boldsymbol {x}}{\| \boldsymbol {b} \| ^ {2}} \boldsymbol {b},\tag{3.42}
$$

where the last equality holds for the dot product only. We can also compute the length of $\pi _ { U } ( { \pmb x } )$ by means of Definition 3.1 as 

$$
\left\| \pi_ {U} (\boldsymbol {x}) \right\| = \left\| \lambda \boldsymbol {b} \right\| = | \lambda | \| \boldsymbol {b} \|.\tag{3.43}
$$

Hence, our projection is of length $| \lambda |$ times the length of b. This also adds the intuition that λ is the coordinate of $\pi _ { U } ( { \pmb x } )$ with respect to the basis vector b that spans our one-dimensional subspace $U$ 

If we use the dot product as an inner product, we get 

$$
\left\| \pi_ {U} (\boldsymbol {x}) \right\| \stackrel {(3. 4 2)} {=} \frac {\left| \boldsymbol {b} ^ {\top} \boldsymbol {x} \right|}{\left\| \boldsymbol {b} \right\| ^ {2}} \left\| \boldsymbol {b} \right\| \stackrel {(3. 2 5)} {=} | \cos \omega | \| \boldsymbol {x} \| \| \boldsymbol {b} \| \frac {\left\| \boldsymbol {b} \right\|}{\left\| \boldsymbol {b} \right\| ^ {2}} = | \cos \omega | \| \boldsymbol {x} \|.\tag{3.44}
$$

Here, ω is the angle between x and b. This equation should be familiar from trigonometry: $\operatorname { I f } \| \pmb { x } \| = 1$ , then x lies on the unit circle. It follows that the projection onto the horizontal axis spanned by b is exactly cos ω, and the length of the corresponding vector $\pi _ { U } ( { \pmb x } ) = | \cos { \omega } |$ . An illustration is given in Figure 3.10(b). 

3. Finding the projection matrix ${ \pmb P } _ { \pi }$ . We know that a projection is a linear mapping (see Definition 3.10). Therefore, there exists a projection matrix $P _ { \pi } ,$ such that $\pi _ { U } ( { \pmb x } ) = P _ { \pi } { \pmb x }$ . With the dot product as inner product and 

$$
\pi_ {U} (\boldsymbol {x}) = \lambda \boldsymbol {b} = \boldsymbol {b} \lambda = \boldsymbol {b} \frac {\boldsymbol {b} ^ {\top} \boldsymbol {x}}{\| \boldsymbol {b} \| ^ {2}} = \frac {\boldsymbol {b b} ^ {\top}}{\| \boldsymbol {b} \| ^ {2}} \boldsymbol {x},\tag{3.45}
$$

we immediately see that 

$$
\boldsymbol {P} _ {\pi} = \frac {\boldsymbol {b b} ^ {\top}}{\| \boldsymbol {b} \| ^ {2}}.\tag{3.46}
$$

Note that $b b ^ { \top }$ (and, consequently, $\mathbf { \nabla } _ { P _ { \pi } } )$ is a symmetric matrix (of rank 1), and $\| \pmb { b } \| ^ { 2 } = \langle \pmb { b } , \pmb { b } \rangle$ is a scalar. 

The projection matrix $P _ { \pi }$ projects any vector $\pmb { x } \in \mathbb { R } ^ { n }$ onto the line through the origin with direction b (equivalently, the subspace $U$ spanned by b). 

Remark. The projection $\pi _ { U } ( \pmb { x } ) \in \mathbb { R } ^ { n }$ is still an n-dimensional vector and not a scalar. However, we no longer require n coordinates to represent the projection, but only a single one if we want to express it with respect to the basis vector b that spans the subspace $U { : } \lambda$ ◇ 

Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 

## 3.8 Orthogonal Projections

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/5e413489f4affd9339a47aed03adec55f808c709b4faf19c89b38ec46e468e5d.jpg)


Figure 3.11 Projection onto a two-dimensional subspace U with basis $b _ { 1 } , b _ { 2 }$ . The projection π<sub>U</sub> (x) of x $\in \mathbb { R } ^ { 3 }$ onto U can be expressed as a linear combination of $b _ { 1 } , b _ { 2 }$ and the displacement vector ${ \pmb x } - \pi _ { U } ( { \pmb x } )$ is orthogonal to both $\pmb { b } _ { 1 }$ and $b _ { 2 }$ 

## Example 3.10 (Projection onto a Line)

Find the projection matrix ${ \pmb P } _ { \pi }$ onto the line through the origin spanned by $\pmb { b } = \left[ 1 \quad 2 \quad 2 \right] ^ { \top }$ . b is a direction and a basis of the one-dimensional subspace (line through origin). 

With (3.46), we obtain 

$$
\boldsymbol {P} _ {\pi} = \frac {\boldsymbol {b b} ^ {\top}}{\boldsymbol {b} ^ {\top} \boldsymbol {b}} = \frac {1}{9} \left[ \begin{array}{l} 1 \\ 2 \\ 2 \end{array} \right] \left[ \begin{array}{l l l} 1 & 2 & 2 \end{array} \right] = \frac {1}{9} \left[ \begin{array}{l l l} 1 & 2 & 2 \\ 2 & 4 & 4 \\ 2 & 4 & 4 \end{array} \right].\tag{3.47}
$$

Let us now choose a particular x and see whether it lies in the subspace spanned by b. For $\pmb { x } = \left\lceil 1 \quad 1 \quad 1 \right\rceil ^ { \top }$ , the projection is 

$$
\pi_ {U} (\boldsymbol {x}) = \boldsymbol {P} _ {\pi} \boldsymbol {x} = \frac {1}{9} \left[ \begin{array}{c c c} 1 & 2 & 2 \\ 2 & 4 & 4 \\ 2 & 4 & 4 \end{array} \right] \left[ \begin{array}{c} 1 \\ 1 \\ 1 \end{array} \right] = \frac {1}{9} \left[ \begin{array}{c} 5 \\ 1 0 \\ 1 0 \end{array} \right] \in \operatorname{span} [ \left[ \begin{array}{c} 1 \\ 2 \\ 2 \end{array} \right].\tag{3.48}
$$

Note that the application of $P _ { \pi }$ to $\pi _ { U } ( { \pmb x } )$ does not change anything, i.e., $\pmb { P } _ { \pi } \pi _ { U } ( \pmb { x } ) = \pi _ { U } ( \pmb { x } )$ . This is expected because according to Definition 3.10, we know that a projection matrix ${ \pmb P } _ { \pi }$ satisfies $\boldsymbol { P } _ { \pi } ^ { 2 } \boldsymbol { x } = \boldsymbol { P } _ { \pi } \boldsymbol { x }$ for all x. 

Remark. With the results from Chapter 4, we can show that $\pi _ { U } ( { \pmb x } )$ is an eigenvector of $P _ { \pi . }$ , and the corresponding eigenvalue is 1. 

## 3.8.2 Projection onto General Subspaces

In the following, we look at orthogonal projections of vectors $\pmb { x } \in \mathbb { R } ^ { n }$ onto lower-dimensional subspaces $U \subseteq \mathbb { R } ^ { n }$ with $\dim ( U ) = m \geqslant 1$ . An illustration is given in Figure 3.11. 

Assume that $\left( b _ { 1 } , \ldots , b _ { m } \right)$ is an ordered basis of U . Any projection $\pi _ { U } ( { \pmb x } )$ onto U is necessarily an element of U. Therefore, they can be represented 

If U is given by a set of spanning vectors, which are not a basis, make sure you determine a basis $\pmb { b } _ { 1 } , \dots , \pmb { b } _ { m }$ before proceeding. 

The basis vectors form the columns of $B \in \mathbb { R } ^ { n \times m } ;$ , where $B = [ b _ { 1 } , \dots , b _ { m } ]$ 

as linear combinations of the basis vectors $b _ { 1 } , \ldots , b _ { m }$ of $U _ { z }$ , such that $\begin{array} { r } { \pi _ { U } ( { \pmb x } ) = \sum _ { i = 1 } ^ { m } \lambda _ { i } \pmb { b } _ { i } } \end{array}$ 

As in the 1D case, we follow a three-step procedure to find the projection $\pi _ { U } ( { \pmb x } )$ and the projection matrix $P _ { \pi }$ : 

1. Find the coordinates $\lambda _ { 1 } , \ldots , \lambda _ { m }$ of the projection (with respect to the basis of U ), such that the linear combination 

$$
\pi_ {U} (\boldsymbol {x}) = \sum_ {i = 1} ^ {m} \lambda_ {i} \boldsymbol {b} _ {i} = B \boldsymbol {\lambda},\tag{3.49}
$$

$$
\boldsymbol {B} = \left[ \boldsymbol {b} _ {1}, \dots , \boldsymbol {b} _ {m} \right] \in \mathbb {R} ^ {n \times m}, \quad \boldsymbol {\lambda} = \left[ \lambda_ {1}, \dots , \lambda_ {m} \right] ^ {\top} \in \mathbb {R} ^ {m},\tag{3.50}
$$

is closest to $\pmb { x } \in \mathbb { R } ^ { n }$ . As in the 1D case, “closest” means “minimum distance”, which implies that the vector connecting $\pi _ { U } ( \pmb { x } ) \in U$ and $\pmb { x } \in \mathbb { R } ^ { n }$ must be orthogonal to all basis vectors of $U$ . Therefore, we obtain m simultaneous conditions (assuming the dot product as the inner product) 

$$
\langle \pmb {b} _ {1}, \pmb {x} - \pi_ {U} (\pmb {x}) \rangle = \pmb {b} _ {1} ^ {\top} (\pmb {x} - \pi_ {U} (\pmb {x})) = 0\tag{3.51}
$$

$$
\langle \pmb {b} _ {m}, \pmb {x} - \pi_ {U} (\pmb {x}) \rangle = \pmb {b} _ {m} ^ {\top} (\pmb {x} - \pi_ {U} (\pmb {x})) = 0\tag{3.52}
$$

which, with $\pi _ { U } ( { \pmb x } ) = B \lambda$ , can be written as 

$$
\boldsymbol {b} _ {1} ^ {\top} (\boldsymbol {x} - \boldsymbol {B} \boldsymbol {\lambda}) = 0\tag{3.53}
$$

$$
\begin{array}{c} \vdots \\ \boldsymbol {b} _ {m} ^ {\top} (\boldsymbol {x} - \boldsymbol {B} \boldsymbol {\lambda}) = 0 \end{array}\tag{3.54}
$$

such that we obtain a homogeneous linear equation system 

$$
\left[ \begin{array}{c} \boldsymbol {b} _ {1} ^ {\top} \\ \vdots \\ \boldsymbol {b} _ {m} ^ {\top} \end{array} \right] \left[ \begin{array}{c} \boldsymbol {x} - \boldsymbol {B} \boldsymbol {\lambda} \end{array} \right] = \mathbf {0} \iff \boldsymbol {B} ^ {\top} (\boldsymbol {x} - \boldsymbol {B} \boldsymbol {\lambda}) = \mathbf {0}\tag{3.55}
$$

$$
\Longleftrightarrow \boldsymbol {B} ^ {\top} \boldsymbol {B} \lambda = \boldsymbol {B} ^ {\top} \boldsymbol {x}.\tag{3.56}
$$

The last expression is called normal equation. Since $\pmb { b } _ { 1 } , \dots , \pmb { b } _ { m }$ are a basis of $U$ and, therefore, linearly independent, $\boldsymbol { B } ^ { \intercal } \boldsymbol { B } \in \mathbb { R } ^ { m \times m }$ is regular and can be inverted. This allows us to solve for the coefficients/ coordinates 

$$
\boldsymbol {\lambda} = \left(\boldsymbol {B} ^ {\top} \boldsymbol {B}\right) ^ {- 1} \boldsymbol {B} ^ {\top} \boldsymbol {x}.\tag{3.57}
$$

The matrix $( B ^ { \top } B ) ^ { - 1 } B ^ { \top }$ is also called the pseudo-inverse of $B ,$ , which can be computed for non-square matrices B. It only requires that $B ^ { \top } B$ is positive definite, which is the case if B is full rank. In practical applications $( \boldsymbol { \mathrm { e . g . } }$ , linear regression), we often add a “jitter term” ϵI to 

$B ^ { \top } B$ to guarantee increased numerical stability and positive definiteness. This “ridge” can be rigorously derived using Bayesian inference. See Chapter 9 for details. 

2. Find the projection $\pi _ { U } ( { \pmb x } ) \in U$ . We already established that $\pi _ { U } ( { \pmb x } ) =$ Bλ. Therefore, with (3.57) 

$$
\pi_ {U} (\boldsymbol {x}) = \boldsymbol {B} (\boldsymbol {B} ^ {\top} \boldsymbol {B}) ^ {- 1} \boldsymbol {B} ^ {\top} \boldsymbol {x}.\tag{3.58}
$$

3. Find the projection matrix ${ \pmb P } _ { \pi }$ . From (3.58), we can immediately see that the projection matrix that solves $\pmb { P } _ { \pi } \pmb { x } = \pi _ { U } ( \pmb { x } )$ must be 

$$
\boldsymbol {P} _ {\pi} = \boldsymbol {B} (\boldsymbol {B} ^ {\top} \boldsymbol {B}) ^ {- 1} \boldsymbol {B} ^ {\top}.\tag{3.59}
$$

Remark. The solution for projecting onto general subspaces includes the 1D case as a special case: If dim $( U ) = 1$ , then $B ^ { \top } B \in \mathbb { R }$ is a scalar and we can rewrite the projection matrix in (3.59) $\begin{array} { r } { \pmb { P } _ { \pi } = \pmb { B } ( \pmb { B } ^ { \top } \pmb { B } ) ^ { - 1 } \pmb { B } ^ { \top } } \end{array}$ as $\begin{array} { r } { { P } _ { \pi } = \frac { B B ^ { \top } } { B ^ { \top } B } } \end{array}$ , which is exactly the projection matrix in (3.46). ◇ 

## Example 3.11 (Projection onto a Two-dimensional Subspace)

For a subspace $U = \operatorname { s p a n } [ \left[ 1 \atop 1 \right] , \left[ 1 \atop 2 \right] ] \subseteq \mathbb { R } ^ { 3 }$ and $\pmb { x } = \begin{array} { l } { \left[ 6 \right] } \\ { 0 } \\ { \left[ 0 \right] } \end{array} \in \mathbb { R } ^ { 3 }$ find the coordinates λ of x in terms of the subspace $U _ { i }$ , the projection point $\pi _ { U } ( { \pmb x } )$ and the projection matrix ${ \pmb P } _ { \pi }$ 

First, we see that the generating set of $U$ is a basis (linear indepen-

dence) and write the basis vectors of U into a matrix $\pmb { { \cal B } } = \left[ \begin{array} { l l } { 1 } & { 0 } \\ { 1 } & { 1 } \\ { 1 } & { 2 } \end{array} \right]$ 

Second, we compute the matrix $B ^ { \top } B$ and the vector $B ^ { \intercal }$ x as 

$$
\boldsymbol {B} ^ {\top} \boldsymbol {B} = \left[ \begin{array}{c c c} 1 & 1 & 1 \\ 0 & 1 & 2 \end{array} \right] \left[ \begin{array}{c c} 1 & 0 \\ 1 & 1 \\ 1 & 2 \end{array} \right] = \left[ \begin{array}{c c} 3 & 3 \\ 3 & 5 \end{array} \right], \quad \boldsymbol {B} ^ {\top} \boldsymbol {x} = \left[ \begin{array}{c c c} 1 & 1 & 1 \\ 0 & 1 & 2 \end{array} \right] \left[ \begin{array}{c} 6 \\ 0 \\ 0 \end{array} \right] = \left[ \begin{array}{c} 6 \\ 0 \end{array} \right].\tag{3.60}
$$

Third, we solve the normal equation $B ^ { \top } B \pmb { \lambda } = B ^ { \top }$ x to find λ: 

$$
\left[ \begin{array}{c c} 3 & 3 \\ 3 & 5 \end{array} \right] \left[ \begin{array}{c} \lambda_ {1} \\ \lambda_ {2} \end{array} \right] = \left[ \begin{array}{c} 6 \\ 0 \end{array} \right] \iff \boldsymbol {\lambda} = \left[ \begin{array}{c} 5 \\ - 3 \end{array} \right].\tag{3.61}
$$

Four ${ \mathrm { h } } ,$ the projection $\pi _ { U } ( { \pmb x } )$ of x onto $U ,$ , i.e., into the column space of $B _ { i }$ , can be directly computed via 

$$
\pi_ {U} (\boldsymbol {x}) = \boldsymbol {B} \boldsymbol {\lambda} = \left[ \begin{array}{c} 5 \\ 2 \\ - 1 \end{array} \right].\tag{3.62}
$$



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



The corresponding projection error is the norm of the difference vector between the original vector and its projection onto U , i.e., 

$$
\left\| \boldsymbol {x} - \pi_ {U} (\boldsymbol {x}) \right\| = \left\| \left[ \begin{array}{c c c} 1 & - 2 & 1 \end{array} \right] ^ {\top} \right\| = \sqrt {6}.\tag{3.63}
$$

Fifth, the projection matrix (for any $\pmb { x } \in \mathbb { R } ^ { 3 } )$ is given by 

$$
\boldsymbol {P} _ {\pi} = \boldsymbol {B} (\boldsymbol {B} ^ {\top} \boldsymbol {B}) ^ {- 1} \boldsymbol {B} ^ {\top} = \frac {1}{6} \left[ \begin{array}{c c c} 5 & 2 & - 1 \\ 2 & 2 & 2 \\ - 1 & 2 & 5 \end{array} \right].\tag{3.64}
$$

To verify the results, we can (a) check whether the displacement vector $\pi _ { U } ( { \pmb x } ) - { \pmb x }$ is orthogonal to all basis vectors of U , and (b) verify that $P _ { \pi } = P _ { \pi } ^ { 2 }$ (see Definition 3.10). 

Remark. The projections $\pi _ { U } ( { \pmb x } )$ are still vectors in <sup>Rn</sup> although they lie in an m-dimensional subspace $U \subseteq \mathbb { R } ^ { n }$ . However, to represent a projected vector we only need the m coordinates $\lambda _ { 1 } , \ldots , \lambda _ { m }$ with respect to the basis vectors $\pmb { b } _ { 1 } , \dots , \pmb { b } _ { m }$ of $U$ ◇ 

Remark. In vector spaces with general inner products, we have to pay attention when computing angles and distances, which are defined by means of the inner product. 

Projections allow us to look at situations where we have a linear system Ax = b without a solution. Recall that this means that b does not lie in the span of A, i.e., the vector b does not lie in the subspace spanned by the columns of A. Given that the linear equation cannot be solved exactly, we can find an approximate solution. The idea is to find the vector in the subspace spanned by the columns of A that is closest to b, i.e., we compute the orthogonal projection of b onto the subspace spanned by the columns of A. This problem arises often in practice, and the solution is called the least-squares solution (assuming the dot product as the inner product) of an overdetermined system. This is discussed further in Section 9.4. Using reconstruction errors (3.63) is one possible approach to derive principal component analysis (Section 10.3). 

Remark. We just looked at projections of vectors x onto a subspace U with basis vectors $\{ b _ { 1 } , \ldots , b _ { k } \}$ . If this basis is an ONB, i.e., (3.33) and (3.34) are satisfied, the projection equation (3.58) simplifies greatly to 

$$
\pi_ {U} (\pmb {x}) = \pmb {B B} ^ {\top} \pmb {x}\tag{3.65}
$$

since $B ^ { \top } B = I$ with coordinates 

$$
\boldsymbol {\lambda} = \boldsymbol {B} ^ {\top} \boldsymbol {x}.\tag{3.66}
$$

This means that we no longer have to compute the inverse from (3.58), which saves computation time. ◇ 

## 3.8.3 Gram-Schmidt Orthogonalization

Projections are at the core of the Gram-Schmidt method that allows us to constructively transform any basis $\left( b _ { 1 } , \ldots , b _ { n } \right)$ of an n-dimensional vector space $V$ into an orthogonal/orthonormal basis $( \pmb { u } _ { 1 } , \dots , \pmb { u } _ { n } )$ of $V .$ . This basis always exists (Liesen and Mehrmann, 2015) and $\mathrm { s p a n } [ \pmb { b } _ { 1 } , \dots , \pmb { b } _ { n } ] =$ span $[ { \pmb u } _ { 1 } , \dots , { \pmb u } _ { n } ]$ . The Gram-Schmidt orthogonalization method iteratively constructs an orthogonal basis $\left( \pmb { u } _ { 1 } , \dots , \pmb { u } _ { n } \right)$ from any basis $\left( b _ { 1 } , \ldots , b _ { n } \right)$ of V as follows: 

$$
\boldsymbol {u} _ {1} := \boldsymbol {b} _ {1}\tag{3.67}
$$

$$
\boldsymbol {u} _ {k} := \boldsymbol {b} _ {k} - \pi_ {\text { span } [ \boldsymbol {u} _ {1}, \dots , \boldsymbol {u} _ {k - 1} ]} (\boldsymbol {b} _ {k}), \quad k = 2, \dots , n.\tag{3.68}
$$

In (3.68), the kth basis vector $ { \boldsymbol { b } } _ { k }$ is projected onto the subspace spanned by the first $k - 1$ constructed orthogonal vectors $u _ { 1 } , \ldots , u _ { k - 1 } ;$ see Section 3.8.2. This projection is then subtracted from $b _ { k }$ and yields a vector $\mathbf { \Delta } \mathbf { u } _ { k }$ that is orthogonal to the $( k - 1 )$ -dimensional subspace spanned by $\pmb { u } _ { 1 } , \ldots , \pmb { u } _ { k - 1 }$ . Repeating this procedure for all n basis vectors $b _ { 1 } , \ldots , b _ { n }$ yields an orthogonal basis $\left( \pmb { u } _ { 1 } , \dots , \pmb { u } _ { n } \right)$ of $V .$ . If we normalize the $\mathbf { \Delta } \mathbf { u } _ { k }$ , we obtain an ONB where $\| \boldsymbol { u } _ { k } \| = 1$ for $k = 1 , \dots , n$ 

## Example 3.12 (Gram-Schmidt Orthogonalization)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/b33fb5eff40482bb71e1b8ccf16a5a93f54b9241c309696278a86a2405d6bdd9.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/fd6aee964c3436d69e61f0f6e9a2c698c37c5a4dc1fa2766dc22436142481983.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/91aa5f86642a4f0ed3547bfd3a9b48c0e5d95153d24e0eda171e9bbee8b3ba35.jpg)



(a) Original non-orthogonal (b) First new basis vector (c) Orthogonal basis vectors u<sub>1</sub> basis vectors $b _ { 1 } , b _ { 2 }$ $\pmb { u } _ { 1 } = \pmb { b } _ { 1 }$ and projection of $\scriptstyle b _ { 2 }$ and ${ \pmb u } _ { 2 } = { \pmb b } _ { 2 } - \pi _ { \mathrm { s p a n } [ { \pmb u } _ { 1 } ] } ( { \pmb b } _ { 2 } )$ onto the subspace spanned by


Consider a basis $( b _ { 1 } , b _ { 2 } )$ ) of $\mathbb { R } ^ { 2 }$ , where 

Figure 3.12 Gram-Schmidt orthogonalization. (a) non-orthogonal basis $( b _ { 1 } , b _ { 2 } )$ of $\mathbb { R } ^ { 2 } ;$ (b) first constructed basis vector ${ \pmb u } _ { 1 }$ and orthogonal projection of $\pmb { b } _ { 2 }$ onto span $[ { \pmb u } _ { 1 } ] ;$ (c) orthogonal basis $( \mathbf { \boldsymbol { u } } _ { 1 } , \mathbf { \boldsymbol { u } } _ { 2 } )$ of $\mathbb { R } ^ { 2 }$ 

$$
\boldsymbol {b} _ {1} = \left[ \begin{array}{c} 2 \\ 0 \end{array} \right], \quad \boldsymbol {b} _ {2} = \left[ \begin{array}{c} 1 \\ 1 \end{array} \right];\tag{3.69}
$$

see also Figure 3.12(a). Using the Gram-Schmidt method, we construct an orthogonal basis $( \pmb { u } _ { 1 } , \pmb { u } _ { 2 } )$ of $\mathbb { R } ^ { 2 }$ as follows (assuming the dot product as the inner product): 

$$
\boldsymbol {u} _ {1} := \boldsymbol {b} _ {1} = \left[ \begin{array}{c} 2 \\ 0 \end{array} \right],\tag{3.70}
$$

$$
\boldsymbol {u} _ {2} := \boldsymbol {b} _ {2} - \pi_ {\text {span} [ \boldsymbol {u} _ {1} ]} (\boldsymbol {b} _ {2}) \stackrel {(3. 4 5)} {=} \boldsymbol {b} _ {2} - \frac {\boldsymbol {u} _ {1} \boldsymbol {u} _ {1} ^ {\top}}{\| \boldsymbol {u} _ {1} \| ^ {2}} \boldsymbol {b} _ {2} = \left[ \begin{array}{c} 1 \\ 1 \end{array} \right] - \left[ \begin{array}{c c} 1 & 0 \\ 0 & 0 \end{array} \right] \left[ \begin{array}{c} 1 \\ 1 \end{array} \right] = \left[ \begin{array}{c} 0 \\ 1 \end{array} \right].\tag{3.71}
$$



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 




90



Figure 3.13 Projection onto an affine space. (a) original setting; (b) setting shifted by −x<sub>0</sub> so that x − x<sub>0</sub> can be projected onto the direction space U; (c) projection is translated back to x<sub>0</sub> + π<sub>U</sub>(x − x<sub>0</sub>), which gives the final orthogonal projection $\pi _ { L } ( { \pmb x } )$


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/383b18ed1ebee1f744b81aced3e21ed5305afef6f1423994afa6c0cf644cf578.jpg)



(b) Reduce problem to projection π<sub>U</sub> onto vector subspace.



(c) Add support point back in to get affine projection $\pi _ { L }$



These steps are illustrated in Figures 3.12(b) and (c). We immediately see that $\mathbf { \delta u } _ { 1 }$ and ${ \pmb u } _ { 2 }$ are orthogonal, i.e., $\pmb { u } _ { 1 } ^ { \top } \pmb { u } _ { 2 } = 0$


## 3.8.4 Projection onto Affine Subspaces

Thus far, we discussed how to project a vector onto a lower-dimensional subspace U . In the following, we provide a solution to projecting a vector onto an affine subspace. 

Consider the setting in Figure 3.13(a). We are given an affine space $L =$ $x _ { 0 } + U$ , where $b _ { 1 } , b _ { 2 }$ are basis vectors of $U$ . To determine the orthogonal projection $\pi _ { L } ( \pmb { x } )$ of x onto $L ,$ we transform the problem into a problem that we know how to solve: the projection onto a vector subspace. In order to get there, we subtract the support point $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ from x and from $L ,$ so that $L - x _ { 0 } = U$ is exactly the vector subspace $U .$ . We can now use the orthogonal projections onto a subspace we discussed in Section 3.8.2 and obtain the projection $\pi _ { U } ( \pmb { x } - \pmb { x } _ { 0 } )$ , which is illustrated in Figure 3.13(b). This projection can now be translated back into L by adding $\scriptstyle { \mathbf { x } } _ { 0 } ,$ , such that we obtain the orthogonal projection onto an affine space $L$ as 

$$
\pi_ {L} (\boldsymbol {x}) = \boldsymbol {x} _ {0} + \pi_ {U} (\boldsymbol {x} - \boldsymbol {x} _ {0}),\tag{3.72}
$$

where $\pi _ { U } ( \cdot )$ is the orthogonal projection onto the subspace $U$ , i.e., the direction space of $L ;$ see Figure 3.13(c). 

From Figure 3.13, it is also evident that the distance of x from the affine space $L$ is identical to the distance of ${ \boldsymbol { \mathbf { \mathit { x } } } } - { \boldsymbol { \mathbf { \mathit { x } } } } _ { 0 }$ from U , i.e., 

$$
d (\boldsymbol {x}, L) = \left\| \boldsymbol {x} - \pi_ {L} (\boldsymbol {x}) \right\| = \left\| \boldsymbol {x} - \left(\boldsymbol {x} _ {0} + \pi_ {U} (\boldsymbol {x} - \boldsymbol {x} _ {0})\right) \right\|\tag{3.73a}
$$

$$
= d (\boldsymbol {x} - \boldsymbol {x} _ {0}, \pi_ {U} (\boldsymbol {x} - \boldsymbol {x} _ {0})) = d (\boldsymbol {x} - \boldsymbol {x} _ {0}, U).\tag{3.73b}
$$

We will use projections onto an affine subspace to derive the concept of a separating hyperplane in Section 12.1. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/c6c380fd2673c8ff46f2159e3e2d73bb4d077fa8fcf7b459513295aa66f77b6b.jpg)



Figure 3.14 A rotation rotates objects in a plane about the origin. If the rotation angle is positive, we rotate counterclockwise.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/fdfbc7b93c38d74266c1359dde96ecfec4ff49e72e93aa879a15ae362945ec08.jpg)



Figure 3.15 The robotic arm needs to rotate its joints in order to pick up objects or to place them correctly. Figure taken from (Deisenroth et al., 2015).


## 3.9 Rotations

Length and angle preservation, as discussed in Section 3.4, are the two characteristics of linear mappings with orthogonal transformation matrices. In the following, we will have a closer look at specific orthogonal transformation matrices, which describe rotations. 

A rotation is a linear mapping (more specifically, an automorphism of a Euclidean vector space) that rotates a plane by an angle θ about the origin, i.e., the origin is a fixed point. For a positive angle $\theta > 0 _ { : }$ , by common convention, we rotate in a counterclockwise direction. An example is shown in Figure 3.14, where the transformation matrix is 

$$
\boldsymbol {R} = \left[ \begin{array}{c c} - 0. 3 8 & - 0. 9 2 \\ 0. 9 2 & - 0. 3 8 \end{array} \right].\tag{3.74}
$$

Important application areas of rotations include computer graphics and robotics. For example, in robotics, it is often important to know how to rotate the joints of a robotic arm in order to pick up or place an object, see Figure 3.15. 



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



Figure 3.16 Rotation of the standard basis in $\mathbb { R } ^ { 2 }$ by an angle θ. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/fd118c2e076247f06bb7800c9e015a13d613865152d4eb2803996898ef5add6f.jpg)



3.9.1 Rotations in $\mathbb { R } ^ { 2 }$


Consider the standard basis ${ \bigg \{ } e _ { 1 } = { \bigg [ } { \bigg ] } { \bigg ] } , e _ { 2 } = { \bigg [ } { \boldsymbol { 0 } } { \bigg ] } { \bigg \} }$ of $\mathbb { R } ^ { 2 }$ , which defines the standard coordinate system in $\mathbb { R } ^ { 2 }$ . We aim to rotate this coordinate system by an angle θ as illustrated in Figure 3.16. Note that the rotated vectors are still linearly independent and, therefore, are a basis of $\mathbb { R } ^ { 2 }$ . This means that the rotation performs a basis change. 

Rotations Φ are linear mappings so that we can express them by a rotation matrix $R ( \theta )$ . Trigonometry (see Figure 3.16) allows us to determine the coordinates of the rotated axes (the image of Φ) with respect to the standard basis in $\mathbb { R } ^ { 2 }$ . We obtain 

$$
\Phi (\boldsymbol {e} _ {1}) = \left[ \begin{array}{c} \cos \theta \\ \sin \theta \end{array} \right], \quad \Phi (\boldsymbol {e} _ {2}) = \left[ \begin{array}{c} - \sin \theta \\ \cos \theta \end{array} \right].\tag{3.75}
$$

Therefore, the rotation matrix that performs the basis change into the rotated coordinates $R ( \theta )$ is given as 

$$
\boldsymbol {R} (\theta) = \left[ \begin{array}{c c} \Phi (\boldsymbol {e} _ {1}) & \Phi (\boldsymbol {e} _ {2}) \end{array} \right] = \left[ \begin{array}{c c} \cos \theta & - \sin \theta \\ \sin \theta & \cos \theta \end{array} \right].\tag{3.76}
$$

## 3.9.2 Rotations in $\mathbb { R } ^ { 3 }$

In contrast to the $\mathbb { R } ^ { 2 }$ case, in $\mathbb { R } ^ { 3 }$ we can rotate any two-dimensional plane about a one-dimensional axis. The easiest way to specify the general rotation matrix is to specify how the images of the standard basis $e _ { 1 } , e _ { 2 } , e _ { 3 }$ are supposed to be rotated, and making sure these images $R e _ { 1 } , R e _ { 2 } , R e _ { 3 }$ are orthonormal to each other. We can then obtain a general rotation matrix R by combining the images of the standard basis. 

To have a meaningful rotation angle, we have to define what “counterclockwise” means when we operate in more than two dimensions. We use the convention that a “counterclockwise” (planar) rotation about an axis refers to a rotation about an axis when we look at the axis “head on, from the end toward the origin”. In $\mathbb { R } ^ { 3 }$ , there are therefore three (planar) rotations about the three standard basis vectors (see Figure 3.17): 

## 3.9 Rotations

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/a8aa54c7ab778af62cb5b2eca916fa44fca9779ee64b7e5c822b41e9721986e4.jpg)



Figure 3.17 Rotation of a vector (gray) in <sup>R3</sup> by an angle θ about the e<sub>3</sub>-axis. The rotated vector is shown in blue.


Rotation about the $e _ { 1 }$ -axis 

$$
\boldsymbol {R} _ {1} (\theta) = \left[ \begin{array}{c c c} \Phi (\boldsymbol {e} _ {1}) & \Phi (\boldsymbol {e} _ {2}) & \Phi (\boldsymbol {e} _ {3}) \end{array} \right] = \left[ \begin{array}{c c c} 1 & 0 & 0 \\ 0 & \cos \theta & - \sin \theta \\ 0 & \sin \theta & \cos \theta \end{array} \right].\tag{3.77}
$$

Here, the $e _ { 1 }$ coordinate is fixed, and the counterclockwise rotation is performed in the $e _ { 2 } e _ { 3 }$ plane. 

Rotation about the e -axis 

$$
\boldsymbol {R} _ {2} (\theta) = \left[ \begin{array}{c c c} \cos \theta & 0 & \sin \theta \\ 0 & 1 & 0 \\ - \sin \theta & 0 & \cos \theta \end{array} \right].\tag{3.78}
$$

If we rotate the $e _ { 1 } e _ { 3 }$ plane about the $e _ { 2 }$ axis, we need to look at the $e _ { 2 }$ axis from its $ { \mathrm { ^ { * } t i p } } ^ { \eta }$ toward the origin. 

Rotation about the $e _ { 3 }$ -axis 

$$
\boldsymbol {R} _ {3} (\theta) = \left[ \begin{array}{c c c} \cos \theta & - \sin \theta & 0 \\ \sin \theta & \cos \theta & 0 \\ 0 & 0 & 1 \end{array} \right].\tag{3.79}
$$

Figure 3.17 illustrates this. 

## 3.9.3 Rotations in n Dimensions

The generalization of rotations from 2D and 3D to n-dimensional Euclidean vector spaces can be intuitively described as fixing $n - 2$ dimensions and restrict the rotation to a two-dimensional plane in the n-dimensional space. As in the three-dimensional case, we can rotate any plane (two-dimensional subspace of <sup>Rn</sup>). 

Definition 3.11 (Givens Rotation). Let V be an n-dimensional Euclidean vector space and $\Phi : V  V$ an automorphism with transformation ma-

trix 

$$
\boldsymbol {R} _ {i j} (\theta) := \left[ \begin{array}{c c c c c} \boldsymbol {I} _ {i - 1} & \boldsymbol {0} & \dots & \dots & \boldsymbol {0} \\ \boldsymbol {0} & \cos \theta & \boldsymbol {0} & - \sin \theta & \boldsymbol {0} \\ \boldsymbol {0} & \boldsymbol {0} & \boldsymbol {I} _ {j - i - 1} & \boldsymbol {0} & \boldsymbol {0} \\ \boldsymbol {0} & \sin \theta & \boldsymbol {0} & \cos \theta & \boldsymbol {0} \\ \boldsymbol {0} & \dots & \dots & \boldsymbol {0} & \boldsymbol {I} _ {n - j} \end{array} \right] \in \mathbb {R} ^ {n \times n},\tag{3.80}
$$

for $1 \leqslant i < j \leqslant$ n and $\theta \in \mathbb { R } .$ . Then $R _ { i j } ( \theta )$ is called a Givens rotation. Essentially, $R _ { i j } ( \theta )$ is the identity matrix $\scriptstyle { { \cal { I } } _ { n } }$ with 

$$
r _ {i i} = \cos \theta , r _ {i j} = - \sin \theta , r _ {j i} = \sin \theta , r _ {j j} = \cos \theta .\tag{3.81}
$$

In two dimensions $( \mathrm { i } . \mathsf { e } . , n = 2 )$ , we obtain (3.76) as a special case. 

## 3.9.4 Properties of Rotations

Rotations exhibit a number of useful properties, which can be derived by considering them as orthogonal matrices (Definition 3.8): 

Rotations preserve distances, $\mathrm { i . e . , } \| x - y \| = \| R _ { \theta } ( { \pmb x } ) - { \pmb R } _ { \theta } ( \pmb y ) \|$ . In other words, rotations leave the distance between any two points unchanged after the transformation. 

Rotations preserve angles, $\mathrm { i . e . , }$ , the angle between $\scriptstyle { R _ { \theta } } x$ and $\scriptstyle R _ { \theta } { \boldsymbol { y } }$ equals the angle between x and $\mathbf { \pmb { y } } .$ 

Rotations in three (or more) dimensions are generally not commutative. Therefore, the order in which rotations are applied is important, even if they rotate about the same point. Only in two dimensions vector rotations are commutative, such that $R ( \phi ) R ( \theta ) \ : = \ : R ( \theta ) R ( \phi )$ for all $\phi , \theta \in [ 0 , 2 \pi )$ . They form an Abelian group (with multiplication) only if they rotate about the same point (e.g., the origin). 

## 3.10 Further Reading

In this chapter, we gave a brief overview of some of the important concepts of analytic geometry, which we will use in later chapters of the book. For a broader and more in-depth overview of some of the concepts we presented, we refer to the following excellent books: Axler (2015) and Boyd and Vandenberghe (2018). 

Inner products allow us to determine specific bases of vector (sub)spaces, where each vector is orthogonal to all others (orthogonal bases) using the Gram-Schmidt method. These bases are important in optimization and numerical algorithms for solving linear equation systems. For instance, Krylov subspace methods, such as conjugate gradients or the generalized minimal residual method (GMRES), minimize residual errors that are orthogonal to each other (Stoer and Burlirsch, 2002). 

In machine learning, inner products are important in the context of kernel methods (Scholkopf and Smola¨ , 2002). Kernel methods exploit the fact that many linear algorithms can be expressed purely by inner product computations. Then, the “kernel trick” allows us to compute these inner products implicitly in a (potentially infinite-dimensional) feature space, without even knowing this feature space explicitly. This allowed the “non-linearization” of many algorithms used in machine learning, such as kernel-PCA (Scholkopf et al.¨ , 1997) for dimensionality reduction. Gaus sian processes (Rasmussen and Williams, 2006) also fall into the category of kernel methods and are the current state of the art in probabilistic regression (fitting curves to data points). The idea of kernels is explored further in Chapter 12. 

Projections are often used in computer graphics, e.g., to generate shadows. In optimization, orthogonal projections are often used to (iteratively) minimize residual errors. This also has applications in machine learning, e.g., in linear regression where we want to find a (linear) function that minimizes the residual errors, i.e., the lengths of the orthogonal projections of the data onto the linear function (Bishop, 2006). We will investigate this further in Chapter 9. PCA (Pearson, 1901; Hotelling, 1933) also uses projections to reduce the dimensionality of high-dimensional data. We will discuss this in more detail in Chapter 10. 

## Exercises

3.1 Show that $\langle \cdot , \cdot \rangle$ defined for all $\pmb { x } = [ x _ { 1 } , x _ { 2 } ] ^ { \top } \in \mathbb { R } ^ { 2 }$ and $\pmb { y } = [ y _ { 1 } , y _ { 2 } ] ^ { \top } \in \mathbb { R } ^ { 2 }$ by 

$$
\langle \boldsymbol {x}, \boldsymbol {y} \rangle := x _ {1} y _ {1} - (x _ {1} y _ {2} + x _ {2} y _ {1}) + 2 (x _ {2} y _ {2})
$$

is an inner product. 

3.2 Consider $\mathbb { R } ^ { 2 }$ with $\langle \cdot , \cdot \rangle$ defined for all x and $\pmb { y }$ in $\mathbb { R } ^ { 2 }$ as 

$$
\langle \boldsymbol {x}, \boldsymbol {y} \rangle := \boldsymbol {x} ^ {\top} \underbrace {\left[ \begin{array}{c c} 2 & 0 \\ 1 & 2 \end{array} \right]} _ {=: \boldsymbol {A}} \boldsymbol {y}.
$$

Is $\langle \cdot , \cdot \rangle$ an inner product? 

3.3 Compute the distance between 

$$
\boldsymbol {x} = \left[ \begin{array}{c} 1 \\ 2 \\ 3 \end{array} \right], \quad \boldsymbol {y} = \left[ \begin{array}{c} - 1 \\ - 1 \\ 0 \end{array} \right]
$$

using 

a. $\langle \pmb { x } , \pmb { y } \rangle : = \pmb { x } ^ { \top } \pmb { y }$ 

2 1 0 b. $\langle \pmb { x } , \pmb { y } \rangle : = \pmb { x } ^ { \top } \pmb { A } \pmb { y }$ A := 1 3 −1 0 −1 2 

3.4 Compute the angle between 

$$
\boldsymbol {x} = \left[ \begin{array}{c} 1 \\ 2 \end{array} \right], \quad \boldsymbol {y} = \left[ \begin{array}{c} - 1 \\ - 1 \end{array} \right]
$$

using 

a. $\langle \pmb { x } , \pmb { y } \rangle : = \pmb { x } ^ { \top } \pmb { y }$ 

$$
\text { b. } \langle \boldsymbol {x}, \boldsymbol {y} \rangle := \boldsymbol {x} ^ {\top} \boldsymbol {B} \boldsymbol {y}, \quad \boldsymbol {B} := \left[ \begin{array}{c c} 2 & 1 \\ 1 & 3 \end{array} \right]
$$

3.5 Consider the Euclidean vector space $\mathbb { R } ^ { 5 }$ with the dot product. A subspace $U \subseteq \mathbb { R } ^ { 5 }$ and $\pmb { x } \in \mathbb { R } ^ { 5 }$ are given by 

$$
U = \operatorname{span} [ \left[ \begin{array}{c} 0 \\ - 1 \\ 2 \\ 0 \\ 2 \end{array} \right], \left[ \begin{array}{c} 1 \\ - 3 \\ 1 \\ - 1 \\ 2 \end{array} \right], \left[ \begin{array}{c} - 3 \\ 4 \\ 1 \\ 2 \\ 1 \end{array} \right], \left[ \begin{array}{c} - 1 \\ - 3 \\ 5 \\ 0 \\ 7 \end{array} \right], \quad \boldsymbol {x} = \left[ \begin{array}{c} - 1 \\ - 9 \\ - 1 \\ 4 \\ 1 \end{array} \right].
$$

a. Determine the orthogonal projection $\pi _ { U } ( { \pmb x } )$ of x onto $U$ 

b. Determine the distance $d ( { \pmb x } , U )$ 

3.6 Consider $\mathbb { R } ^ { 3 }$ with the inner product 

$$
\langle \boldsymbol {x}, \boldsymbol {y} \rangle := \boldsymbol {x} ^ {\top} \left[ \begin{array}{c c c} 2 & 1 & 0 \\ 1 & 2 & - 1 \\ 0 & - 1 & 2 \end{array} \right] \boldsymbol {y}.
$$

Furthermore, we define $e _ { 1 } , e _ { 2 }$ , e as the standard/canonical basis in $\mathbb { R } ^ { 3 }$ 

Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 

a. Determine the orthogonal projection $\pi _ { U } ( e _ { 2 } )$ of $e _ { 2 }$ onto 

$$
U = \operatorname{span} [ \boldsymbol {e} _ {1}, \boldsymbol {e} _ {3} ].
$$

Hint: Orthogonality is defined through the inner product. 

b. Compute the distance $d ( e _ { 2 } , U )$ 

c. Draw the scenario: standard basis vectors and $\pi _ { U } ( e _ { 2 } )$ 

3.7 Let V be a vector space and π an endomorphism of V . 

a. Prove that π is a projection if and only if $\operatorname { i d } _ { V } - \pi$ is a projection, where $\operatorname { i d } _ { V }$ is the identity endomorphism on $V .$ 

b. Assume now that π is a projection. Calculate Im $\left( \operatorname { i d } _ { V } - \pi \right)$ and $\ker ( \mathrm { i d } _ { V } - \pi )$ as a function of Im(π) and ker(π). 

3.8 Using the Gram-Schmidt method, turn the basis $B ~ = ~ ( b _ { 1 } , b _ { 2 } )$ of a twodimensional subspace $U \subseteq \mathbb { R } ^ { 3 }$ into an ONB $C = ( c _ { 1 } , c _ { 2 } )$ of U, where 

$$
\boldsymbol {b} _ {1} := \left[ \begin{array}{c} 1 \\ 1 \\ 1 \end{array} \right], \quad \boldsymbol {b} _ {2} := \left[ \begin{array}{c} - 1 \\ 2 \\ 0 \end{array} \right].
$$

3.9 Let $n \in \mathbb { N }$ and let $x _ { 1 } , \ldots , x _ { n } > 0$ be n positive real numbers so that $x _ { 1 } +$ $\ldots + x _ { n } = 1$ . Use the Cauchy-Schwarz inequality and show that 

a. $\begin{array} { r } { \sum _ { i = 1 } ^ { n } x _ { i } ^ { 2 } \geqslant \frac { 1 } { n } } \\ { \sum _ { i = 1 } ^ { n } \frac { 1 } { x _ { i } } \geqslant n ^ { 2 } } \end{array}$ b. 

Hint: Think about the dot product on $\mathbb { R } ^ { n }$ . Then, choose specific vectors x, $\pmb { y } \in \mathbb { R } ^ { n }$ and apply the Cauchy-Schwarz inequality. 

3.10 Rotate the vectors 

$$
\boldsymbol {x} _ {1} := \left[ \begin{array}{c} 2 \\ 3 \end{array} \right], \quad \boldsymbol {x} _ {2} := \left[ \begin{array}{c} 0 \\ - 1 \end{array} \right]
$$

by $3 0 ^ { \circ }$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/bbeface1cae4f51b6774c4ef2660af4673a9be62af40dbcb4ebf5228764e0409.jpg)


# Matrix Decompositions

In Chapters 2 and 3, we studied ways to manipulate and measure vectors, projections of vectors, and linear mappings. Mappings and transformations of vectors can be conveniently described as operations performed by matrices. Moreover, data is often represented in matrix form as well, e.g., where the rows of the matrix represent different people and the columns describe different features of the people, such as weight, height, and socioeconomic status. In this chapter, we present three aspects of matrices: how to summarize matrices, how matrices can be decomposed, and how these decompositions can be used for matrix approximations. 

We first consider methods that allow us to describe matrices with just a few numbers that characterize the overall properties of matrices. We will do this in the sections on determinants (Section 4.1) and eigenvalues (Section 4.2) for the important special case of square matrices. These characteristic numbers have important mathematical consequences and allow us to quickly grasp what useful properties a matrix has. From here we will proceed to matrix decomposition methods: An analogy for matrix decomposition is the factoring of numbers, such as the factoring of 21 into prime numbers 7 3. For this reason matrix decomposition is also often referred to as matrix factorization. Matrix decompositions are used to describe a matrix by means of a different representation using factors of interpretable matrices. 

We will first cover a square-root-like operation for symmetric, positive definite matrices, the Cholesky decomposition (Section 4.3). From here we will look at two related methods for factorizing matrices into canonical forms. The first one is known as matrix diagonalization (Section 4.4), which allows us to represent the linear mapping using a diagonal transformation matrix if we choose an appropriate basis. The second method, singular value decomposition (Section 4.5), extends this factorization to non-square matrices, and it is considered one of the fundamental concepts in linear algebra. These decompositions are helpful, as matrices representing numerical data are often very large and hard to analyze. We conclude the chapter with a systematic overview of the types of matrices and the characteristic properties that distinguish them in the form of a matrix taxonomy (Section 4.7). 

The methods that we cover in this chapter will become important in 


4.1 Determinant and Trace


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/ec8e1380e8dd4f9769849ce112764b7c9f055ba761741d49c65c8fb9608c40fb.jpg)



Figure 4.1 A mind map of the concepts introduced in this chapter, along with where they are used in other parts of the book.



both subsequent mathematical chapters, such as Chapter $^ { 6 , }$ but also in applied chapters, such as dimensionality reduction in Chapters 10 or density estimation in Chapter 11. This chapter’s overall structure is depicted in the mind map of Figure 4.1.


## 4.1 Determinant and Trace

Determinants are important concepts in linear algebra. A determinant is a mathematical object in the analysis and solution of systems of linear equations. Determinants are only defined for square matrices $A \in \mathbb { R } ^ { n \times n }$ ， i.e., matrices with the same number of rows and columns. In this book, we write the determinant as det(A) or sometimes as $| A |$ so that 

$$
\det (\boldsymbol {A}) = \left| \begin{array}{c c c c} a _ {1 1} & a _ {1 2} & \ldots & a _ {1 n} \\ a _ {2 1} & a _ {2 2} & \ldots & a _ {2 n} \\ \vdots & & \ddots & \vdots \\ a _ {n 1} & a _ {n 2} & \ldots & a _ {n n} \end{array} \right|.\tag{4.1}
$$

The determinant of a square matrix $A \in \mathbb { R } ^ { n \times n }$ is a function that maps A 

The determinant notation |A| must not be confused with the absolute value. 

onto a real number. Before providing a definition of the determinant for general $n \times n$ matrices, let us have a look at some motivating examples, and define determinants for some special matrices. 

## Example 4.1 (Testing for Matrix Invertibility)

Let us begin with exploring if a square matrix A is invertible (see Section 2.2.2). For the smallest cases, we already know when a matrix is invertible. If A is a $1 \times 1$ matrix, i.e., it is a scalar number, then $\begin{array} { r } { \pmb { A } = \pmb { a } \implies \pmb { A } ^ { - 1 } = \frac { 1 } { a } } \end{array}$ . Thus $\begin{array} { r } { a \ \frac { 1 } { a } = 1 } \end{array}$ holds, if and only if $a \neq 0$ 

For $2 \times 2$ matrices, by the definition of the inverse (Definition 2.3), we know that $\pmb { A } \pmb { A } ^ { - 1 } = \pmb { I }$ . Then, with (2.24), the inverse of A is 

$$
\pmb {A} ^ {- 1} = \frac {1}{a _ {1 1} a _ {2 2} - a _ {1 2} a _ {2 1}} \left[ \begin{array}{c c} a _ {2 2} & - a _ {1 2} \\ - a _ {2 1} & a _ {1 1} \end{array} \right].\tag{4.2}
$$

Hence, A is invertible if and only if 

$$
a _ {1 1} a _ {2 2} - a _ {1 2} a _ {2 1} \neq 0.\tag{4.3}
$$

This quantity is the determinant of $A \in \mathbb { R } ^ { 2 \times 2 }$ , i.e., 

$$
\det (\boldsymbol {A}) = \left| \begin{array}{c c} a _ {1 1} & a _ {1 2} \\ a _ {2 1} & a _ {2 2} \end{array} \right| = a _ {1 1} a _ {2 2} - a _ {1 2} a _ {2 1}.\tag{4.4}
$$

Example 4.1 points already at the relationship between determinants and the existence of inverse matrices. The next theorem states the same result for $n \times n$ matrices. 

Theorem 4.1. For any square matrix $A \in \mathbb { R } ^ { n \times n }$ it holds that A is invertible if and only $i f \operatorname* { d e t } ( A ) \neq 0$ 

We have explicit (closed-form) expressions for determinants of small matrices in terms of the elements of the matrix. For $n = 1$ 

$$
\det (\boldsymbol {A}) = \det (a _ {1 1}) = a _ {1 1}.\tag{4.5}
$$

For $n = 2 .$ 

$$
\det (\boldsymbol {A}) = \left| \begin{array}{c c} a _ {1 1} & a _ {1 2} \\ a _ {2 1} & a _ {2 2} \end{array} \right| = a _ {1 1} a _ {2 2} - a _ {1 2} a _ {2 1},\tag{4.6}
$$

which we have observed in the preceding example. 

For $n = 3$ (known as Sarrus’ rule), 

$$
\begin{array}{r l} & {\left| \begin{array}{l l l} a _ {1 1} & a _ {1 2} & a _ {1 3} \\ a _ {2 1} & a _ {2 2} & a _ {2 3} \\ a _ {3 1} & a _ {3 2} & a _ {3 3} \end{array} \right| = a _ {1 1} a _ {2 2} a _ {3 3} + a _ {2 1} a _ {3 2} a _ {1 3} + a _ {3 1} a _ {1 2} a _ {2 3}} \\ & {- a _ {3 1} a _ {2 2} a _ {1 3} - a _ {1 1} a _ {3 2} a _ {2 3} - a _ {2 1} a _ {1 2} a _ {3 3}.} \end{array}\tag{4.7}
$$

Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 

## 4.1 Determinant and Trace

For a memory aid of the product terms in Sarrus’ rule, try tracing the elements of the triple products in the matrix. 

We call a square matrix $_ { \pmb { T } }$ an upper-triangular matrix if $T _ { i j } ~ = ~ 0$ for $i > j , \mathsf { i . e . }$ ., the matrix is zero below its diagonal. Analogously, we define a lower-triangular matrix as a matrix with zeros above its diagonal. For a triangular matrix $\pmb { T } \in \mathbb { R } ^ { n \times n }$ , the determinant is the product of the diagonal elements, i.e., 

$$
\det (\boldsymbol {T}) = \prod_ {i = 1} ^ {n} T _ {i i}.\tag{4.8}
$$

## Example 4.2 (Determinants as Measures of Volume)

The notion of a determinant is natural when we consider it as a mapping from a set of n vectors spanning an object in $\mathbb { R } ^ { n }$ . It turns out that the determinant det $( A )$ is the signed volume of an n-dimensional parallelepiped formed by columns of the matrix A. 

For $n = 2$ , the columns of the matrix form a parallelogram; see $\mathrm { F i g } .$ ure 4.2. As the angle between vectors gets smaller, the area of a parallelogram shrinks, too. Consider two vectors ${ \pmb { b } } , { \pmb g }$ that form the columns of a matrix $A = [ b , g ]$ . Then, the absolute value of the determinant of A is the area of the parallelogram with vertices $\mathbf { 0 } ,  { b } ,  { g } ,  { b } +  { g }$ . In particular, if ${ \pmb { b } } , { \pmb g }$ are linearly dependent so that $\pmb { b } = \lambda \pmb { g }$ for some $\lambda \in \mathbb { R }$ , they no longer form a two-dimensional parallelogram. Therefore, the corresponding area is 0. On the contrary, if $b , \pmb { g }$ are linearly independent and are multiples of the canonical basis vectors $e _ { 1 } , e _ { 2 }$ then they can be written as $\pmb { b } = \left[ \begin{array} { l } { \bar { b } } \\ { 0 } \end{array} \right]$ and $\pmb { g } = \left\lceil \begin{array} { l l } { 0 } \\ { g } \end{array} \right\rceil$ , and the determinant is ${ \left| \begin{array} { l l } { b } & { 0 } \\ { 0 } & { g } \end{array} \right| } = b g - 0 = b g .$ 

The sign of the determinant indicates the orientation of the spanning vectors ${ \pmb { b } } , { \pmb g }$ with respect to the standard basis $( e _ { 1 } , e _ { 2 } )$ . In our figure, flipping the order to g, b swaps the columns of A and reverses the orientation of the shaded area. This becomes the familiar formula: area = height length. This intuition extends to higher dimensions. In $\mathbb { R } ^ { 3 }$ , we consider three vectors $r , b , g \in \mathbb { R } ^ { 3 }$ spanning the edges of a parallelepiped, i.e., a solid with faces that are parallel parallelograms (see Figure 4.3). The absolute value of the determinant of the $3 \times 3$ matrix $[ r , b , g ]$ is the volume of the solid. Thus, the determinant acts as a function that measures the signed volume formed by column vectors composed in a matrix. 

upper-triangular matrix 

Consider the three linearly independent vectors $r , g , b \in \mathbb { R } ^ { 3 }$ given as 

$$
\boldsymbol {r} = \left[ \begin{array}{c} 2 \\ 0 \\ - 8 \end{array} \right], \quad \boldsymbol {g} = \left[ \begin{array}{c} 6 \\ 1 \\ 0 \end{array} \right], \quad \boldsymbol {b} = \left[ \begin{array}{c} 1 \\ 4 \\ - 1 \end{array} \right].\tag{4.9}
$$

lower-triangular matrix 

The determinant is the signed volume of the parallelepiped formed by the columns of the matrix. 

Figure 4.2 The area of the parallelogram (shaded region) spanned by the vectors b and g is |det([b, g])|. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/31cffd86f4f74931e9fe82a1b872036e12b638e6db97686e05c1e9712316b02c.jpg)


Figure 4.3 The volume of the parallelepiped (shaded volume) spanned by vectors $\mathbf { \delta } _ { r , b , g }$ is |det([r, b, g])|. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/1c7b9bb184df9a9e6379bebdd448701daa0b4f93ef95c0135e736b3157a01525.jpg)


The sign of the determinant indicates the orientation of the spanning vectors. 



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



Writing these vectors as the columns of a matrix 

$$
\boldsymbol {A} = [ \boldsymbol {r}, \boldsymbol {g}, \boldsymbol {b} ] = \left[ \begin{array}{c c c} 2 & 6 & 1 \\ 0 & 1 & 4 \\ - 8 & 0 & - 1 \end{array} \right]\tag{4.10}
$$

allows us to compute the desired volume as 

$$
V = | \det (\boldsymbol {A}) | = 1 8 6.\tag{4.11}
$$

Computing the determinant of an $n \times n$ matrix requires a general algorithm to solve the cases for $n > 3 .$ , which we are going to explore in the following. Theorem 4.2 below reduces the problem of computing the determinant of an n n matrix to computing the determinant of $( n - 1 ) \times ( n - 1 )$ matrices. By recursively applying the Laplace expansion (Theorem 4.2), we can therefore compute determinants of $n \times n$ matrices by ultimately computing determinants of $2 \times 2$ matrices. 

Laplace expansion 

Theorem 4.2 (Laplace Expansion). Consider a matrix $\pmb { A } \in \mathbb { R } ^ { n \times n }$ . Then, for all $j = 1 , \dots , n \colon$ 

$\operatorname* { d e t } ( A _ { k , j } )$ is called a minor and $( - 1 ) ^ { k + j } \operatorname* { d e t } ( A _ { k , j } )$ a cofactor. 

1. Expansion along column $j$ 

$$
\det (\boldsymbol {A}) = \sum_ {k = 1} ^ {n} (- 1) ^ {k + j} a _ {k j} \det (\boldsymbol {A} _ {k, j}).\tag{4.12}
$$

2. Expansion along row $j$ 

$$
\det (\boldsymbol {A}) = \sum_ {k = 1} ^ {n} (- 1) ^ {k + j} a _ {j k} \det (\boldsymbol {A} _ {j, k}).\tag{4.13}
$$

Here $\mathbf { A } _ { k , j } \in \mathbb { R } ^ { ( n - 1 ) \times ( n - 1 ) }$ is the submatrix of A that we obtain when deleting row k and column $j .$ 

## Example 4.3 (Laplace Expansion)

Let us compute the determinant of 

$$
\boldsymbol {A} = \left[ \begin{array}{c c c} 1 & 2 & 3 \\ 3 & 1 & 2 \\ 0 & 0 & 1 \end{array} \right]\tag{4.14}
$$

using the Laplace expansion along the first row. Applying (4.13) yields 

$$
\begin{array}{c} \left| \begin{array}{c c c} 1 & 2 & 3 \\ 3 & 1 & 2 \\ 0 & 0 & 1 \end{array} \right| = (- 1) ^ {1 + 1} \cdot 1 \left| \begin{array}{c c} 1 & 2 \\ 0 & 1 \end{array} \right| \\ + (- 1) ^ {1 + 2} \cdot 2 \left| \begin{array}{c c} 3 & 2 \\ 0 & 1 \end{array} \right| + (- 1) ^ {1 + 3} \cdot 3 \left| \begin{array}{c c} 3 & 1 \\ 0 & 0 \end{array} \right|. \end{array}\tag{4.15}
$$



Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 



We use (4.6) to compute the determinants of all $2 \times 2$ matrices and obtain 

$$
\det (\boldsymbol {A}) = 1 (1 - 0) - 2 (3 - 0) + 3 (0 - 0) = - 5.\tag{4.16}
$$

For completeness we can compare this result to computing the determinant using Sarrus’ rule (4.7): 

$$
\det (\boldsymbol {A}) = 1 \cdot 1 \cdot 1 + 3 \cdot 0 \cdot 3 + 0 \cdot 2 \cdot 2 - 0 \cdot 1 \cdot 3 - 1 \cdot 0 \cdot 2 - 3 \cdot 2 \cdot 1 = 1 - 6 = - 5.\tag{4.17}
$$

For $A \in \mathbb { R } ^ { n \times n }$ the determinant exhibits the following properties: 

The determinant of a matrix product is the product of the corresponding determinants, de $( A B ) = \operatorname* { d e t } ( A ) \operatorname* { d e t } ( B )$ 

Determinants are invariant to transposition, i.e., det $( \boldsymbol { A } ) = \operatorname* { d e t } ( \boldsymbol { A } ^ { \top } )$ 

If A is regular (invertible), then det $\begin{array} { r } { ( A ^ { - 1 } ) = \frac { 1 } { \operatorname* { d e t } ( A ) } } \end{array}$ 

Similar matrices (Definition 2.22) possess the same determinant. Therefore, for a linear mapping $\Phi : V \to V$ all transformation matrices $A _ { \Phi }$ of Φ have the same determinant. Thus, the determinant is invariant to the choice of basis of a linear mapping. 

Adding a multiple of a column/row to another one does not change det(A). 

Multiplication of a column/row with $\lambda \ \in \ \mathbb { R }$ scales det(A) by λ. In particular, det $( \lambda A ) = \lambda ^ { n } \operatorname* { d e t } ( A )$ 

Swapping two rows/columns changes the sign of $\operatorname* { d e t } ( A )$ 

Because of the last three properties, we can use Gaussian elimination (see Section 2.1) to compute $\operatorname* { d e t } ( A )$ by bringing A into row-echelon form. We can stop Gaussian elimination when we have A in a triangular form where the elements below the diagonal are all 0. Recall from (4.8) that the determinant of a triangular matrix is the product of the diagonal elements. 

Theorem 4.3. A square matrix $A \in \mathbb { R } ^ { n \times n }$ has det $( A ) \neq 0$ if and only $i f$ r ${ \mathfrak { z } } ( A ) = n$ . In other words, A is invertible $i f$ and only if it is full rank. 

When mathematics was mainly performed by hand, the determinant calculation was considered an essential way to analyze matrix invertibil ity. However, contemporary approaches in machine learning use direct numerical methods that superseded the explicit calculation of the determinant. For example, in Chapter 2, we learned that inverse matrices can be computed by Gaussian elimination. Gaussian elimination can thus be used to compute the determinant of a matrix. 

Determinants will play an important theoretical role for the following sections, especially when we learn about eigenvalues and eigenvectors (Section 4.2) through the characteristic polynomial. 

Definition 4.4. The trace of a square matrix $A \in \mathbb { R } ^ { n \times n }$ is defined as 

$$
\operatorname{tr} (\boldsymbol {A}) := \sum_ {i = 1} ^ {n} a _ {i i},\tag{4.18}
$$

i.e. , the trace is the sum of the diagonal elements of A. 

The trace satisfies the following properties: 

$\operatorname { t r } ( A + B ) = \operatorname { t r } ( A ) + \operatorname { t r } ( B )$ for A, $B \in \mathbb { R } ^ { n \times n }$ 

$\operatorname { t r } ( \alpha A ) = \alpha \mathrm { t r } ( A ) , \alpha \in \mathbb { R }$ for $A \in \mathbb { R } ^ { n \times n }$ 

$\operatorname { t r } ( I _ { n } ) = n$ 

tr(AB) = tr(BA) for A <sup>Rn×k</sup>, B <sup>Rk×n</sup> 

It can be shown that only one function satisfies these four properties together – the trace (Gohberg et al., 2012). 

The properties of the trace of matrix products are more general. Specifically, the trace is invariant under cyclic permutations, i.e., 

$$
\operatorname{tr} (\boldsymbol {A K L}) = \operatorname{tr} (\boldsymbol {K L A})\tag{4.19}
$$

for matrices $\pmb { A } \in \mathbb { R } ^ { a \times k } , \pmb { K } \in \mathbb { R } ^ { k \times l } , \pmb { L } \in \mathbb { R } ^ { l \times a }$ . This property generalizes to products of an arbitrary number of matrices. As a special case of (4.19), it follows that for two vectors x, $\pmb { y } \in \mathbb { R } ^ { n }$ 

$$
\operatorname{tr} \left(\boldsymbol {x} \boldsymbol {y} ^ {\top}\right) = \operatorname{tr} \left(\boldsymbol {y} ^ {\top} \boldsymbol {x}\right) = \boldsymbol {y} ^ {\top} \boldsymbol {x} \in \mathbb {R}.\tag{4.20}
$$

Given a linear mapping $\Phi : { \cal V }  { \cal V } ;$ , where V is a vector space, we define the trace of this map by using the trace of matrix representation of Φ. For a given basis of $V ,$ we can describe Φ by means of the transformation matrix $A .$ . Then the trace of Φ is the trace of $A .$ . For a different basis of $V _ { i }$ , it holds that the corresponding transformation matrix B of Φ can be obtained by a basis change of the form $S ^ { - 1 } A S$ for suitable S (see Section 2.7.2). For the corresponding trace of $\Phi ,$ this means 

$$
\operatorname{tr} (\boldsymbol {B}) = \operatorname{tr} \left(\boldsymbol {S} ^ {- 1} \boldsymbol {A} \boldsymbol {S}\right) \stackrel {(4. 1 9)} {=} \operatorname{tr} \left(\boldsymbol {A} \boldsymbol {S} \boldsymbol {S} ^ {- 1}\right) = \operatorname{tr} (\boldsymbol {A}).\tag{4.21}
$$

Hence, while matrix representations of linear mappings are basis dependent the trace of a linear mapping Φ is independent of the basis. 

In this section, we covered determinants and traces as functions characterizing a square matrix. Taking together our understanding of determinants and traces we can now define an important equation describing a matrix A in terms of a polynomial, which we will use extensively in the following sections. 

Definition 4.5 (Characteristic Polynomial). For $\lambda \in \mathbb { R }$ and a square matrix $A \in \mathbb { R } ^ { n \times n }$ 

$$
\begin{array}{l} p _ {\boldsymbol {A}} (\lambda) := \det (\boldsymbol {A} - \lambda \boldsymbol {I}) \\ \qquad = c _ {0} + c _ {1} \lambda + c _ {2} \lambda^ {2} + \dots + c _ {n - 1} \lambda^ {n - 1} + (- 1) ^ {n} \lambda^ {n}, \end{array}\tag{4.22a}
$$

(4.22b) 

$c _ { 0 } , \dotsc , c _ { n - 1 } \in \mathbb { R } ,$ is the characteristic polynomial of A. In particular, 

4.2 Eigenvalues and Eigenvectors 

$$
c _ {0} = \det (\boldsymbol {A}),\tag{4.23}
$$

$$
c _ {n - 1} = (- 1) ^ {n - 1} \operatorname{tr} (\boldsymbol {A}).\tag{4.24}
$$

The characteristic polynomial (4.22a) will allow us to compute eigenvalues and eigenvectors, covered in the next section. 

## 4.2 Eigenvalues and Eigenvectors

We will now get to know a new way to characterize a matrix and its associ ated linear mapping. Recall from Section 2.7.1 that every linear mapping has a unique transformation matrix given an ordered basis. We can interpret linear mappings and their associated transformation matrices by performing an “eigen” analysis. As we will see, the eigenvalues of a linear mapping will tell us how a special set of vectors, the eigenvectors, is transformed by the linear mapping. 

Definition 4.6. Let $\pmb { A } \in \mathbb { R } ^ { n \times n }$ be a square matrix. Then $\lambda \in \mathbb { R }$ is an eigenvalue of A and $\pmb { x } \in \mathbb { R } ^ { n } \backslash \{ \mathbf { 0 } \}$ is the corresponding eigenvector of A if 

$$
\boldsymbol {A} \boldsymbol {x} = \lambda \boldsymbol {x}.\tag{4.25}
$$

We call (4.25) the eigenvalue equation. 

Remark. In the linear algebra literature and software, it is often a conven tion that eigenvalues are sorted in descending order, so that the largest eigenvalue and associated eigenvector are called the first eigenvalue and its associated eigenvector, and the second largest called the second eigenvalue and its associated eigenvector, and so on. However, textbooks and publications may have different or no notion of orderings. We do not want to presume an ordering in this book if not stated explicitly. 

The following statements are equivalent: 

λ is an eigenvalue of $A \in \mathbb { R } ^ { n \times n }$ 

There exists an $\pmb { x } \in \mathbb { R } ^ { n } \backslash \{ \mathbf { 0 } \}$ with $A x \ = \ \lambda x$ , or equivalently, $( A -$ $\lambda I _ { n } ) { \pmb x } = { \bf 0 }$ can be solved non-trivially, $\mathrm { i . e . , } x \neq \mathbf { 0 }$ 

$\operatorname { \mathrm { { \cdot k } } } ( A - \lambda I _ { n } ) < n$ 

det $( { \cal A } - \lambda I _ { n } ) = 0 \quad$ 

Definition 4.7 (Collinearity and Codirection). Two vectors that point in the same direction are called codirected. Two vectors are collinear if they point in the same or the opposite direction. 

Remark (Non-uniqueness of eigenvectors). If x is an eigenvector of A associated with eigenvalue $\lambda ,$ then for any $c \in \mathbb { R } \backslash \{ 0 \}$ it holds that cx is an eigenvector of A with the same eigenvalue since 

$$
\boldsymbol {A} (c \boldsymbol {x}) = c \boldsymbol {A} \boldsymbol {x} = c \lambda \boldsymbol {x} = \lambda (c \boldsymbol {x}).\tag{4.26}
$$

Thus, all vectors that are collinear to x are also eigenvectors of A. 



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



Theorem 4.8. $\lambda \in \mathbb { R }$ is an eigenvalue of $\pmb { A } \in \mathbb { R } ^ { n \times n }$ if and only $i f \lambda$ is a root of the characteristic polynomial $p _ { A } ( \lambda ) o f A$ 

Definition 4.9. Let a square matrix A have an eigenvalue $\lambda _ { i } .$ . The algebraic multiplicity of $\lambda _ { i }$ is the number of times the root appears in the characteristic polynomial. 

Definition 4.10 (Eigenspace and Eigenspectrum). For $A \in \mathbb { R } ^ { n \times n }$ , the set of all eigenvectors of A associated with an eigenvalue λ spans a subspace of $\mathbb { R } ^ { n }$ , which is called the eigenspace of A with respect to λ and is denoted by $E _ { \lambda }$ . The set of all eigenvalues of A is called the eigenspectrum, or just spectrum, of $A .$ 

If λ is an eigenvalue of $A \in \mathbb { R } ^ { n \times n }$ , then the corresponding eigenspace $E _ { \lambda }$ is the solution space of the homogeneous system of linear equations $( A - \lambda I ) x = 0$ . Geometrically, the eigenvector corresponding to a nonzero eigenvalue points in a direction that is stretched by the linear mapping. The eigenvalue is the factor by which it is stretched. If the eigenvalue is negative, the direction of the stretching is flipped. 

## Example 4.4 (The Case of the Identity Matrix)

The identity matrix $\pmb { I } \in \mathbb { R } ^ { n \times n }$ has characteristic polynomial $p _ { I } ( \lambda ) \ =$ det $( I - \lambda I ) = ( 1 - \lambda ) ^ { n } = 0 _ { ! }$ , which has only one eigenvalue $\lambda = 1$ that occurs n times. Moreover, $I x = \lambda x = 1$ x holds for all vectors x $\in \mathbb { R } ^ { n } \backslash \{ \mathbf { 0 } \}$ Because of this, the sole eigenspace $E _ { 1 }$ of the identity matrix spans n dimensions, and all n standard basis vectors of $\mathbb { R } ^ { n }$ are eigenvectors of I. 

Useful properties regarding eigenvalues and eigenvectors include the following: 

A matrix A and its transpose $A ^ { \top }$ possess the same eigenvalues, but not necessarily the same eigenvectors. 

The eigenspace $E _ { \lambda }$ is the null space of $A - \lambda I$ since 

$$
\boldsymbol {A} \boldsymbol {x} = \lambda \boldsymbol {x} \iff \boldsymbol {A} \boldsymbol {x} - \lambda \boldsymbol {x} = \boldsymbol {0}
$$

$$
\Longleftrightarrow (\boldsymbol {A} - \lambda \boldsymbol {I}) \boldsymbol {x} = \boldsymbol {0} \Longleftrightarrow \boldsymbol {x} \in \ker (\boldsymbol {A} - \lambda \boldsymbol {I}).\tag{4.27a}
$$

(4.27b) 

Similar matrices (see Definition 2.22) possess the same eigenvalues. Therefore, a linear mapping $\Phi$ has eigenvalues that are independent of the choice of basis of its transformation matrix. This makes eigenvalues, together with the determinant and the trace, key characteristic parameters of a linear mapping as they are all invariant under basis change. 

Symmetric, positive definite matrices always have positive, real eigenvalues. 

Example 4.5 (Computing Eigenvalues, Eigenvectors, and Eigenspaces) 

Let us find the eigenvalues and eigenvectors of the $2 \times 2$ matrix 

$$
\boldsymbol {A} = \left[ \begin{array}{c c} 4 & 2 \\ 1 & 3 \end{array} \right].\tag{4.28}
$$

Step 1: Characteristic Polynomial. From our definition of the eigenvector x = 0 and eigenvalue λ of $A _ { i }$ , there will be a vector such that $A x = \lambda x , \mathrm { i . e . , } ( A - \lambda I ) x = 0$ . Since x $\neq \mathbf { 0 }$ , this requires that the kernel (null space) of $A - \lambda I$ contains more elements than just 0. This means that $A - \lambda I$ is not invertible and therefore det $( A - \lambda I ) = 0$ . Hence, we need to compute the roots of the characteristic polynomial (4.22a) to find the eigenvalues. 

Step 2: Eigenvalues. The characteristic polynomial is 

$$
p _ {\boldsymbol {A}} (\lambda) = \det (\boldsymbol {A} - \lambda \boldsymbol {I})\tag{4.29a}
$$

$$
= \det \left(\left[ \begin{array}{c c} 4 & 2 \\ 1 & 3 \end{array} \right] - \left[ \begin{array}{c c} \lambda & 0 \\ 0 & \lambda \end{array} \right]\right) = \left| \begin{array}{c c} 4 - \lambda & 2 \\ 1 & 3 - \lambda \end{array} \right|\tag{4.29b}
$$

$$
= (4 - \lambda) (3 - \lambda) - 2 \cdot 1.\tag{4.29c}
$$

We factorize the characteristic polynomial and obtain 

$$
p (\lambda) = (4 - \lambda) (3 - \lambda) - 2 \cdot 1 = 1 0 - 7 \lambda + \lambda^ {2} = (2 - \lambda) (5 - \lambda)\tag{4.30}
$$

giving the roots $\lambda _ { 1 } = 2$ and $\lambda _ { 2 } = 5$ 

Step 3: Eigenvectors and Eigenspaces. We find the eigenvectors that correspond to these eigenvalues by looking at vectors x such that 

$$
\left[ \begin{array}{c c} 4 - \lambda & 2 \\ 1 & 3 - \lambda \end{array} \right] \boldsymbol {x} = \boldsymbol {0}.\tag{4.31}
$$

For $\lambda = 5$ we obtain 

$$
\left[ \begin{array}{c c} 4 - 5 & 2 \\ 1 & 3 - 5 \end{array} \right] \left[ \begin{array}{c} x _ {1} \\ x _ {2} \end{array} \right] = \left[ \begin{array}{c c} - 1 & 2 \\ 1 & - 2 \end{array} \right] \left[ \begin{array}{c} x _ {1} \\ x _ {2} \end{array} \right] = \mathbf {0}.\tag{4.32}
$$

We solve this homogeneous system and obtain a solution space 

$$
E _ {5} = \operatorname{span} [ \left[ \begin{array}{c} 2 \\ 1 \end{array} \right].\tag{4.33}
$$

This eigenspace is one-dimensional as it possesses a single basis vector. 

Analogously, we find the eigenvector for $\lambda = 2$ by solving the homoge neous system of equations 

$$
\left[ \begin{array}{c c} 4 - 2 & 2 \\ 1 & 3 - 2 \end{array} \right] \boldsymbol {x} = \left[ \begin{array}{c c} 2 & 2 \\ 1 & 1 \end{array} \right] \boldsymbol {x} = \boldsymbol {0}.\tag{4.34}
$$



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



This means any vector $\pmb { x } = \left\lceil \frac { x _ { 1 } } { x _ { 2 } } \right\rceil$ , where $x _ { 2 } = - x _ { 1 }$ , such as $\left[ \begin{array} { l } { 1 } \\ { - 1 } \end{array} \right]$ , is an eigenvector with eigenvalue 2. The corresponding eigenspace is given as 

$$
E _ {2} = \operatorname{span} [ \left[ \begin{array}{c} 1 \\ - 1 \end{array} \right] ].\tag{4.35}
$$

The two eigenspaces $E _ { 5 }$ and $E _ { 2 }$ in Example 4.5 are one-dimensional as they are each spanned by a single vector. However, in other cases we may have multiple identical eigenvalues (see Definition 4.9) and the eigenspace may have more than one dimension. 

Definition 4.11. Let $\lambda _ { i }$ be an eigenvalue of a square matrix A. Then the geometric multiplicity of $\lambda _ { i }$ is the number of linearly independent eigenvectors associated with $\lambda _ { i } .$ . In other words, it is the dimensionality of the eigenspace spanned by the eigenvectors associated with $\lambda _ { i }$ 

Remark. A specific eigenvalue’s geometric multiplicity must be at least one because every eigenvalue has at least one associated eigenvector. An eigenvalue’s geometric multiplicity cannot exceed its algebraic multiplicity, but it may be lower. ◇ 

## Example 4.6

The matrix $A = { \left\lceil \begin{array} { l l } { 2 } & { 1 } \\ { 0 } & { 2 } \end{array} \right\rceil }$ has two repeated eigenvalues $\lambda _ { 1 } = \lambda _ { 2 } = 2$ and an algebraic multiplicity of 2. The eigenvalue has, however, only one distinct unit eigenvector $\pmb { x } _ { 1 } = \left[ \begin{array} { l } { 1 } \\ { 0 } \end{array} \right]$ and, thus, geometric multiplicity 1. 

## Graphical Intuition in Two Dimensions

Let us gain some intuition for determinants, eigenvectors, and eigenvalues using different linear mappings. Figure 4.4 depicts five transformation matrices $A _ { 1 } , \dotsc , A _ { 5 }$ and their impact on a square grid of points, centered at the origin: 

$A _ { 1 } = { \left[ \begin{array} { l l } { { \frac { 1 } { 2 } } } & { 0 } \\ { 0 } & { 2 } \end{array} \right] }$ . The direction of the two eigenvectors correspond to the canonical basis vectors in $\mathbb { R } ^ { 2 }$ , i.e., to two cardinal axes. The vertical axis is extended by a factor of 2 (eigenvalue $\lambda _ { 1 } = 2 )$ , and the horizontal axis is compressed by factor $\frac { 1 } { 2 }$ (eigenvalue $\begin{array} { r } { \lambda _ { 2 } = \frac { 1 } { 2 } ) } \end{array}$ ). The mapping is area preserving (det $( A _ { 1 } ) = 1 \mathbf { \bar { = } } 2 \cdot \frac { 1 } { 2 } )$ 

1 $A _ { 2 } = { \bigg [ } { \frac { 1 } { 0 } } { \frac { 1 } { 2 } } { \bigg ] }$ corresponds to a shearing mapping , i.e., it shears the points along the horizontal axis to the right if they are on the positive half of the vertical axis, and to the left vice versa. This mapping is area preserving $\mathrm { ( d e t } ( A _ { 2 } ) = 1 )$ . The eigenvalue $\lambda _ { 1 } = 1 = \lambda _ { 2 }$ is repeated and the eigenvectors are collinear (drawn here for emphasis in two opposite directions). This indicates that the mapping acts only along one direction (the horizontal axis). 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/8b28d801f98ca53171ab24886a489f67c4157b11c2cebe6693283ada6c540ee3.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/67121d96511091a3d98f7afd29df71ba8d42b2a26debaab384edf74bb4899e89.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/d7d9e090b91b886df829bdcf9c5a994b47d24a9fb93b01450dba8a9818a9a54d.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/f51361dbe55e9c68237d5c5de22e21e066b244d04b4f16213cf44ab5a14e12ec.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/bd15ed056b736d9a6d95a6ca34d9eeab9ad5600c8b1eb17aa590d627e242d8fb.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/e6338673299ecbb7cf330c31331467de01340af789edf3dd2c0333301f7360b6.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/d4cfae1b55d21073c46b1c3ad041ef7253612e7cc2425ff2cc55e33506cdf2bb.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/6f665935a978093e5156230cbba1681130696b52ebf195667e162c80dcb6060b.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/b3da63a7914ffada83e66395420950b412098225289f0258b14bc94dcfa5caaf.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/80b461b7c17210e9caee564a7e5599ede117569b287a8b299c7b828772682c44.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/c58420f2708bb8f3d6e70ea4afada24bcdf36dd1294a8348f92310ed3f541a55.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/e8d536a0a96441019e54fc675ac0f9672fe7c60856907ddc8d1b5b9ba295c343.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/d1cea2106fc6a90ee1b1c6e1d3b315267d3c42a5b1c4ea44e1871f7514457e3d.jpg)


$A _ { 3 } = { \textstyle \left[ \cos \left( { \frac { \pi } { 6 } } \right) - \sin \left( { \frac { \pi } { 6 } } \right) \right] } = { \textstyle \frac { 1 } { 2 } } { \textstyle \left[ \begin{array} { l l } { { \sqrt { 3 } } } & { { - 1 } } \\ { { 1 } } & { { \sqrt { 3 } } } \end{array} \right] }$ The matrix $A _ { 3 }$ rotates the points by $\textstyle { \frac { \pi } { 6 } } \operatorname { r a d } = 3 0 ^ { \circ }$ counter-clockwise and has only complex eigenvalues, reflecting that the mapping is a rotation (hence, no eigenvectors are drawn). A rotation has to be volume preserving, and so the determinant is 1. For more details on rotations, we refer to Section 3.9. 

$A _ { 4 } = \left[ { \begin{array} { l l } { 1 } & { - 1 } \\ { - 1 } & { 1 } \end{array} } \right]$ represents a mapping in the standard basis that collapses a two-dimensional domain onto one dimension. Since one eigen-

Figure 4.4 Determinants and eigenspaces. Overview of five linear mappings and their associated transformation matrices $A _ { i } \in \mathbb { R } ^ { 2 \times 2 }$ projecting 400 color-coded points x $\in \mathbb { R } ^ { 2 }$ (left column) onto target points ${ \pmb A } _ { i } { \pmb x }$ (right column). The central column depicts the first eigenvector, stretched by its associated eigenvalue $\lambda _ { 1 } ,$ and the second eigenvector stretched by its eigenvalue $\lambda _ { 2 } .$ . Each row depicts the effect of one of five transformation matrices $\mathbf { A } _ { i }$ with respect to the standard basis. 

value is 0, the space in direction of the (blue) eigenvector corresponding to $\lambda _ { 1 } = 0$ collapses, while the orthogonal (red) eigenvector stretches space by a factor $\lambda _ { 2 } = 2$ . Therefore, the area of the image is 0. 

$A _ { 5 } = { \left\lceil \begin{array} { l l } { 1 } & { { \frac { 1 } { 2 } } } \\ { { \frac { 1 } { 2 } } } & { 1 } \end{array} \right\rceil }$ is a shear-and-stretch mapping that scales space by 75% since $\vert \overline { { { \mathrm { d e t } } } } ( \bar { A } _ { 5 } ) \vert ~ = ~ \frac { 3 } { 4 }$ . It stretches space along the (red) eigenvector of $\lambda _ { 2 }$ by a factor 1.5 and compresses it along the orthogonal (blue) eigenvector by a factor 0.5. 

## Example 4.7 (Eigenspectrum of a Biological Neural Network)

Figure 4.5 Caenorhabditis elegans neural network (Kaiser and Hilgetag, 2006).(a) Symmetrized connectivity matrix; (b) Eigenspectrum. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/16ec5bb40e35b2f301c040a33fe75135109c0dfc17eb84ea863480d3e1e2a9ee.jpg)



(a) Connectivity matrix.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/2eda8656c2f10268fb49e4f671a87331160ec15838af58e3a6c83b88e193e9cc.jpg)



(b) Eigenspectrum.


Methods to analyze and learn from network data are an essential component of machine learning methods. The key to understanding networks is the connectivity between network nodes, especially if two nodes are connected to each other or not. In data science applications, it is often useful to study the matrix that captures this connectivity data. 

We build a connectivity/adjacency matrix $\pmb { A } \in \mathbb { R } ^ { 2 7 7 \times 2 7 7 }$ of the complete neural network of the worm C.Elegans. Each row/column represents one of the 277 neurons of this worm’s brain. The connectivity matrix A has a value of $a _ { i j } = 1$ if neuron i talks to neuron $j$ through a synapse, and $a _ { i j } = 0$ otherwise. The connectivity matrix is not symmetric, which implies that eigenvalues may not be real valued. Therefore, we compute a symmetrized version of the connectivity matrix as $\pmb { A } _ { s y m } : = \pmb { A } + \pmb { A } ^ { \top }$ . This new matrix $A _ { \it { s y m } }$ is shown in Figure 4.5(a) and has a nonzero value $a _ { i j }$ if and only if two neurons are connected (white pixels), irrespective of the direction of the connection. In Figure 4.5(b), we show the corresponding eigenspectrum of $A _ { \it { s y m } }$ . The horizontal axis shows the index of the eigenvalues, sorted in descending order. The vertical axis shows the corresponding eigenvalue. The S-like shape of this eigenspectrum is typical for many biological neural networks. The underlying mechanism responsible for this is an area of active neuroscience research. 

Theorem 4.12. The eigenvectors $\pmb { x } _ { 1 } , \ldots , \pmb { x } _ { n }$ of a matrix $A \in \mathbb { R } ^ { n \times n }$ with n distinct eigenvalues $\lambda _ { 1 } , \ldots , \lambda _ { n }$ are linearly independent. 

This theorem states that eigenvectors of a matrix with n distinct eigenvalues form a basis of $\mathbb { R } ^ { n }$ 

Definition 4.13. A square matrix $\pmb { A } \in \mathbb { R } ^ { n \times n }$ is defective if it possesses fewer than n linearly independent eigenvectors. 

A non-defective matrix $\pmb { A } \in \mathbb { R } ^ { n \times n }$ does not necessarily require n distinct eigenvalues, but it does require that the eigenvectors form a basis of $\mathbb { R } ^ { n }$ . Looking at the eigenspaces of a defective matrix, it follows that the sum of the dimensions of the eigenspaces is less than n. Specifically, a defective matrix has at least one eigenvalue $\lambda _ { i }$ with an algebraic multiplicity $m > 1$ and a geometric multiplicity of less than $m .$ 

Remark. A defective matrix cannot have n distinct eigenvalues, as distinct eigenvalues have linearly independent eigenvectors (Theorem 4.12). 

Theorem 4.14. Given a matrix $\pmb { A } \in \mathbb { R } ^ { m \times n }$ , we can always obtain a symmetric, positive semidefinite matrix $\ b { S } \in \mathbb { R } ^ { n \times n }$ by defining 

$$
\boldsymbol {S} := \boldsymbol {A} ^ {\top} \boldsymbol {A}.\tag{4.36}
$$

Remark. If $\operatorname { r k } ( A ) = n ,$ , then $S : = A ^ { \top } A$ is symmetric, positive definite. 

Understanding why Theorem 4.14 holds is insightful for how we can use symmetrized matrices: Symmetry requires $S \stackrel { } { = } S ^ { \top }$ , and by inserting (4.36) we obtain $\pmb { S } = \pmb { A } ^ { \top } \pmb { A } = \ - \check { \pmb { A } } ^ { \top } ( \pmb { \mathring { A } } ^ { \top } ) ^ { \top } = ( \pmb { A } ^ { \top } \pmb { A } ) ^ { \top } = \pmb { S } ^ { \dagger }$ . Moreover, positive semidefiniteness (Section 3.2.3) requires that ${ \pmb x } ^ { \top } { \pmb S } { \pmb x } \geqslant 0$ and inserting (4.36) we obtain $\pmb { x } ^ { \top } \pmb { S } \pmb { x } = \pmb { x } ^ { \top } \pmb { A } ^ { \top } \pmb { \mathring { A } } \pmb { x } = ( \pmb { x } ^ { \top } \pmb { A } ^ { \top } ) ( \pmb { A } \pmb { x } ) =$ $( A x ) ^ { \top } ( A x ) \geqslant 0 .$ , because the dot product computes a sum of squares (which are themselves non-negative). 

Theorem 4.15 (Spectral Theorem). If $A \in \mathbb { R } ^ { n \times n }$ is symmetric, there exists an orthonormal basis of the corresponding vector space V consisting of eigenvectors of A, and each eigenvalue is real. 

A direct implication of the spectral theorem is that the eigendecomposition of a symmetric matrix A exists (with real eigenvalues), and that we can find an ONB of eigenvectors so that $\pmb { A } = \bar { \pmb { P } } \pmb { D } \pmb { P } ^ { \top }$ , where D is diagonal and the columns of $_ { \pmb { P } }$ contain the eigenvectors. 

## Example 4.8

Consider the matrix 

$$
\boldsymbol {A} = \left[ \begin{array}{c c c} 3 & 2 & 2 \\ 2 & 3 & 2 \\ 2 & 2 & 3 \end{array} \right].\tag{4.37}
$$



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



The characteristic polynomial of A is 

$$
p _ {\boldsymbol {A}} (\lambda) = - (\lambda - 1) ^ {2} (\lambda - 7),\tag{4.38}
$$

so that we obtain the eigenvalues $\lambda _ { 1 } ~ = ~ 1$ and $\lambda _ { 2 } \ = \ 7 ,$ where $\lambda _ { 1 }$ is a repeated eigenvalue. Following our standard procedure for computing eigenvectors, we obtain the eigenspaces 

$$
E _ {1} = \operatorname{span} [ \underbrace {\left[ \begin{array}{c} - 1 \\ 1 \\ 0 \end{array} \right]} _ {=: \boldsymbol {x} _ {1}}, \underbrace {\left[ \begin{array}{c} - 1 \\ 0 \\ 1 \end{array} \right]} _ {=: \boldsymbol {x} _ {2}} ], \quad E _ {7} = \operatorname{span} [ \underbrace {\left[ \begin{array}{c} 1 \\ 1 \\ 1 \end{array} \right]} _ {=: \boldsymbol {x} _ {3}} ].\tag{4.39}
$$

We see that $\mathbf { { x } _ { 3 } }$ is orthogonal to both $\mathbf { x } _ { 1 }$ and $\mathbf { \mathcal { x } } _ { 2 }$ . However, since $\pmb { x } _ { 1 } ^ { \top } \pmb { x } _ { 2 } =$ $1 \neq 0 ,$ they are not orthogonal. The spectral theorem (Theorem 4.15) states that there exists an orthogonal basis, but the one we have is not orthogonal. However, we can construct one. 

To construct such a basis, we exploit the fact that $\mathbf { \boldsymbol { x } } _ { 1 } , \mathbf { \boldsymbol { x } } _ { 2 }$ are eigenvectors associated with the same eigenvalue λ. Therefore, for any $\alpha , \beta \in \mathbb { R }$ it holds that 

$$
\boldsymbol {A} \left(\alpha \boldsymbol {x} _ {1} + \beta \boldsymbol {x} _ {2}\right) = \boldsymbol {A} \boldsymbol {x} _ {1} \alpha + \boldsymbol {A} \boldsymbol {x} _ {2} \beta = \lambda \left(\alpha \boldsymbol {x} _ {1} + \beta \boldsymbol {x} _ {2}\right),\tag{4.40}
$$

i.e., any linear combination of $\mathbf { x } _ { 1 }$ and $\mathbf { \boldsymbol { x } } _ { 2 }$ is also an eigenvector of A associated with λ. The Gram-Schmidt algorithm (Section 3.8.3) is a method for iteratively constructing an orthogonal/orthonormal basis from a set of basis vectors using such linear combinations. Therefore, even if $\mathbf { x } _ { 1 }$ and $\mathbf { \boldsymbol { x } } _ { 2 }$ are not orthogonal, we can apply the Gram-Schmidt algorithm and find eigenvectors associated with $\lambda _ { 1 } = 1$ that are orthogonal to each other (and to $\mathbf { { x } } _ { 3 } )$ . In our example, we will obtain 

$$
\boldsymbol {x} _ {1} ^ {\prime} = \left[ \begin{array}{c} - 1 \\ 1 \\ 0 \end{array} \right], \quad \boldsymbol {x} _ {2} ^ {\prime} = \frac {1}{2} \left[ \begin{array}{c} - 1 \\ - 1 \\ 2 \end{array} \right],\tag{4.41}
$$

which are orthogonal to each other, orthogonal to ${ \pmb x } _ { 3 } ,$ and eigenvectors of A associated with $\lambda _ { 1 } = 1$ 

Before we conclude our considerations of eigenvalues and eigenvectors it is useful to tie these matrix characteristics together with the concepts of the determinant and the trace. 

Theorem 4.16. The determinant of a matrix $\pmb { A } \in \mathbb { R } ^ { n \times n }$ is the product of its eigenvalues, i.e., 

$$
\det (\boldsymbol {A}) = \prod_ {i = 1} ^ {n} \lambda_ {i},\tag{4.42}
$$

where $\lambda _ { i } \in \mathbb { C }$ are (possibly repeated) eigenvalues of A. 

Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/1114bcc9503eb2e115ced732b63492b180b089cc0515ce6984d6f9296a995c79.jpg)


Theorem 4.17. The trace $o f a$ matrix $A \in \mathbb { R } ^ { n \times n }$ is the sum of its eigenvalues, $i . e . _ { \cdot }$ 

$$
t r (\boldsymbol {A}) = \sum_ {i = 1} ^ {n} \lambda_ {i},\tag{4.43}
$$

Figure 4.6 Geometric interpretation of eigenvalues. The eigenvectors of A get stretched by the corresponding eigenvalues. The area of the unit square changes by $| \lambda _ { 1 } \lambda _ { 2 } |$ , the perimeter changes by a factor of $\frac { 1 } { 2 } ( | \lambda _ { 1 } | + | \lambda _ { 2 } | )$ 

where $\lambda _ { i } \in \mathbb { C }$ are (possibly repeated) eigenvalues of A. 

Let us provide a geometric intuition of these two theorems. Consider a matrix $A \in \mathbb { R } ^ { 2 \times 2 }$ that possesses two linearly independent eigenvectors $\mathbf { \delta x } _ { 1 } , \mathbf { \delta x } _ { 2 } .$ . For this example, we assume $( \pmb { x } _ { 1 } , \pmb { x } _ { 2 } )$ are an ONB of $\mathbb { R } ^ { 2 }$ so that they are orthogonal and the area of the square they span is 1; see Figure 4.6. From Section 4.1, we know that the determinant computes the change of area of unit square under the transformation A. In this example, we can compute the change of area explicitly: Mapping the eigenvectors using A gives us vectors ${ \pmb v } _ { 1 } = { \pmb A } { \pmb x } _ { 1 } = \lambda _ { 1 } { \pmb x } _ { 1 }$ and ${ \pmb v } _ { 2 } = { \pmb A } { \pmb x } _ { 2 } = \lambda _ { 2 } { \pmb x } _ { 2 }$ , i.e., the new vectors ${ \mathbf { } } v _ { i }$ are scaled versions of the eigenvectors ${ \mathbf { } } x _ { i } ,$ and the scaling factors are the corresponding eigenvalues $\lambda _ { i } , v _ { 1 } , v _ { 2 }$ are still orthogonal, and the area of the rectangle they span is $| \lambda _ { 1 } \lambda _ { 2 } |$ 

Given that $\mathbf { \boldsymbol { x } } _ { 1 } , \mathbf { \boldsymbol { x } } _ { 2 }$ (in our example) are orthonormal, we can directly compute the perimeter of the unit square as $2 ( 1 + 1 )$ . Mapping the eigenvectors using A creates a rectangle whose perimeter is $2 ( | \lambda _ { 1 } | + | \lambda _ { 2 } | )$ Therefore, the sum of the absolute values of the eigenvalues tells us how the perimeter of the unit square changes under the transformation matrix A. 

## Example 4.9 (Google’s PageRank – Webpages as Eigenvectors)

Google uses the eigenvector corresponding to the maximal eigenvalue of a matrix A to determine the rank of a page for search. The idea for the PageRank algorithm, developed at Stanford University by Larry Page and Sergey Brin in 1996, was that the importance of any web page can be approximated by the importance of pages that link to it. For this, they write down all web sites as a huge directed graph that shows which page links to which. PageRank computes the weight (importance) $x _ { i } \geqslant 0$ of a web site $a _ { i }$ by counting the number of pages pointing to $a _ { i }$ . Moreover, PageRank takes into account the importance of the web sites that link to $a _ { i }$ . The navigation behavior of a user is then modeled by a transition matrix A of this graph that tells us with what (click) probability somebody will end up 



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



on a different web site. The matrix A has the property that for any initial rank/importance vector x of a web site the sequence $x , A x , A ^ { 2 } x , \ldots$ converges to a vector $\pmb { x } ^ { * }$ . This vector is called the PageRank and satisfies $\pmb { A } \pmb { x } ^ { * } = \pmb { x } ^ { * }$ , i.e., it is an eigenvector (with corresponding eigenvalue 1) of A. After normalizing $\mathbf { \boldsymbol { x } } ^ { * }$ , such that $\| \pmb { x } ^ { * } \| = 1$ , we can interpret the entries as probabilities. More details and different perspectives on PageRank can be found in the original technical report (Page et al., 1999). 

## 4.3 Cholesky Decomposition

There are many ways to factorize special types of matrices that we encounter often in machine learning. In the positive real numbers, we have the square-root operation that gives us a decomposition of the number into identical components, e.g., $9 = 3 \cdot 3$ . For matrices, we need to be careful that we compute a square-root-like operation on positive quantities. For symmetric, positive definite matrices (see Section 3.2.3), we can choose from a number of square-root equivalent operations. The Cholesky decomposition/Cholesky factorization provides a square-root equivalent operation on symmetric, positive definite matrices that is useful in practice. 

Theorem 4.18 (Cholesky Decomposition). A symmetric, positive definite matrix A can be factorized into a product $A = L L ^ { \top }$ , where L is a lowertriangular matrix with positive diagonal elements: 

$$
\left[ \begin{array}{c c c} a _ {1 1} & \dots & a _ {1 n} \\ \vdots & \ddots & \vdots \\ a _ {n 1} & \dots & a _ {n n} \end{array} \right] = \left[ \begin{array}{c c c} l _ {1 1} & \dots & 0 \\ \vdots & \ddots & \vdots \\ l _ {n 1} & \dots & l _ {n n} \end{array} \right] \left[ \begin{array}{c c c} l _ {1 1} & \dots & l _ {n 1} \\ \vdots & \ddots & \vdots \\ 0 & \dots & l _ {n n} \end{array} \right].\tag{4.44}
$$

L is called the Cholesky factor of A, and L is unique. 

Example 4.10 (Cholesky Factorization) 

Consider a symmetric, positive definite matrix $\pmb { A } \in \mathbb { R } ^ { 3 \times 3 }$ . We are interested in finding its Cholesky factorization $\pmb { A } = \pmb { L } \pmb { L } ^ { \top } , \mathrm { i . e . }$ 

$$
\boldsymbol {A} = \left[ \begin{array}{c c c} a _ {1 1} & a _ {2 1} & a _ {3 1} \\ a _ {2 1} & a _ {2 2} & a _ {3 2} \\ a _ {3 1} & a _ {3 2} & a _ {3 3} \end{array} \right] = \boldsymbol {L} \boldsymbol {L} ^ {\top} = \left[ \begin{array}{c c c} l _ {1 1} & 0 & 0 \\ l _ {2 1} & l _ {2 2} & 0 \\ l _ {3 1} & l _ {3 2} & l _ {3 3} \end{array} \right] \left[ \begin{array}{c c c} l _ {1 1} & l _ {2 1} & l _ {3 1} \\ 0 & l _ {2 2} & l _ {3 2} \\ 0 & 0 & l _ {3 3} \end{array} \right].\tag{4.45}
$$

Multiplying out the right-hand side yields 

$$
\boldsymbol {A} = \left[ \begin{array}{c c c} l _ {1 1} ^ {2} & l _ {2 1} l _ {1 1} & l _ {3 1} l _ {1 1} \\ l _ {2 1} l _ {1 1} & l _ {2 1} ^ {2} + l _ {2 2} ^ {2} & l _ {3 1} l _ {2 1} + l _ {3 2} l _ {2 2} \\ l _ {3 1} l _ {1 1} & l _ {3 1} l _ {2 1} + l _ {3 2} l _ {2 2} & l _ {3 1} ^ {2} + l _ {3 2} ^ {2} + l _ {3 3} ^ {2} \end{array} \right].\tag{4.46}
$$



Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 



Comparing the left-hand side of (4.45) and the right-hand side of (4.46) shows that there is a simple pattern in the diagonal elements $l _ { i i } \mathbf { \cdot }$ 

$$
l _ {1 1} = \sqrt {a _ {1 1}}, l _ {2 2} = \sqrt {a _ {2 2} - l _ {2 1} ^ {2}}, l _ {3 3} = \sqrt {a _ {3 3} - (l _ {3 1} ^ {2} + l _ {3 2} ^ {2})}.\tag{4.47}
$$

Similarly for the elements below the diagonal $( l _ { i j }$ , where $i > j )$ , there is also a repeating pattern: 

$$
l _ {2 1} = \frac {1}{l _ {1 1}} a _ {2 1}, \quad l _ {3 1} = \frac {1}{l _ {1 1}} a _ {3 1}, \quad l _ {3 2} = \frac {1}{l _ {2 2}} (a _ {3 2} - l _ {3 1} l _ {2 1}).\tag{4.48}
$$

Thus, we constructed the Cholesky decomposition for any symmetric, positive definite $3 \times 3$ matrix. The key realization is that we can backward calculate what the components $l _ { i j }$ for the $\pmb { L }$ should be, given the values $a _ { i j }$ for A and previously computed values of $l _ { i j }$ 

The Cholesky decomposition is an important tool for the numerical computations underlying machine learning. Here, symmetric positive definite matrices require frequent manipulation, e.g., the covariance matrix of a multivariate Gaussian variable (see Section 6.5) is symmetric, positive definite. The Cholesky factorization of this covariance matrix allows us to generate samples from a Gaussian distribution. It also allows us to perform a linear transformation of random variables, which is heavily exploited when computing gradients in deep stochastic models, such as the variational auto-encoder (Jimenez Rezende et al., 2014; Kingma and Welling, 2014). The Cholesky decomposition also allows us to compute determinants very efficiently. Given the Cholesky decomposition $\bar { A } = L L ^ { \top }$ , we know that det $( A ) = \operatorname* { d e t } ( L ) \operatorname* { d e t } ( L ^ { \top } ) = \operatorname* { d e t } ( L ) ^ { 2 }$ . Since $\pmb { L }$ is a triangular matrix, the determinant is simply the product of its diagonal entries so that det $\begin{array} { r } { \left( \boldsymbol { A } \right) = \prod _ { i } l _ { i i } ^ { 2 } } \end{array}$ . Thus, many numerical software packages use the Cholesky decomposition to make computations more efficient. 

## 4.4 Eigendecomposition and Diagonalization

A diagonal matrix is a matrix that has value zero on all off-diagonal elements, i.e., they are of the form 

$$
\boldsymbol {D} = \left[ \begin{array}{c c c} c _ {1} & \dots & 0 \\ \vdots & \ddots & \vdots \\ 0 & \dots & c _ {n} \end{array} \right].\tag{4.49}
$$

They allow fast computation of determinants, powers, and inverses. The determinant is the product of its diagonal entries, a matrix power $D ^ { k }$ is given by each diagonal element raised to the power $k ,$ and the inverse $\pmb { D } ^ { - 1 }$ is the reciprocal of its diagonal elements if all of them are nonzero. 

In this section, we will discuss how to transform matrices into diagonal form. This is an important application of the basis change we discussed in Section 2.7.2 and eigenvalues from Section 4.2. 

Recall that two matrices A, D are similar (Definition 2.22) if there exists an invertible matrix $P ,$ , such that $\pmb { { \cal D } } = \pmb { { \cal P } } ^ { - 1 } \pmb { { \cal A } } \pmb { { \cal P } }$ . More specifically, we will look at matrices A that are similar to diagonal matrices D that contain the eigenvalues of A on the diagonal. 

Definition 4.19 (Diagonalizable). A matrix $\pmb { A } \in \mathbb { R } ^ { n \times n }$ is diagonalizable if it is similar to a diagonal matrix, i.e., if there exists an invertible matrix $P \in \mathbb { R } ^ { n \times n }$ such that $\pmb { { \cal D } } = \pmb { { \cal P } } ^ { - 1 } \pmb { { \cal A } } \pmb { { \cal P } }$ 

In the following, we will see that diagonalizing a matrix $A \in \mathbb { R } ^ { n \times n }$ is a way of expressing the same linear mapping but in another basis (see Section 2.6.1), which will turn out to be a basis that consists of the eigenvectors of A. 

Let $A \in \mathbb { R } ^ { n \times n }$ , let $\lambda _ { 1 } , \ldots , \lambda _ { n }$ be a set of scalars, and let $p _ { 1 } , \ldots , p _ { n }$ be a set of vectors in <sup>Rn</sup>. We define $P : = [ \pmb { p } _ { 1 } , \ldots , \pmb { p } _ { n } ]$ and let $D \in \mathbb { R } ^ { n \times n }$ be a diagonal matrix with diagonal entries $\lambda _ { 1 } , \ldots , \lambda _ { n }$ . Then we can show that 

$$
A P = P D\tag{4.50}
$$

if and only if $\lambda _ { 1 } , \ldots , \lambda _ { n }$ are the eigenvalues of A and $p _ { 1 } , \ldots , p _ { n }$ are corresponding eigenvectors of A. 

We can see that this statement holds because 

$$
\boldsymbol {A} \boldsymbol {P} = \boldsymbol {A} [ \boldsymbol {p} _ {1}, \dots , \boldsymbol {p} _ {n} ] = [ \boldsymbol {A} \boldsymbol {p} _ {1}, \dots , \boldsymbol {A} \boldsymbol {p} _ {n} ],\tag{4.51}
$$

$$
\boldsymbol {P} \boldsymbol {D} = [ \boldsymbol {p} _ {1}, \ldots , \boldsymbol {p} _ {n} ] \left[ \begin{array}{c c c} \lambda_ {1} & & 0 \\ & \ddots & \\ 0 & & \lambda_ {n} \end{array} \right] = [ \lambda_ {1} \boldsymbol {p} _ {1}, \ldots , \lambda_ {n} \boldsymbol {p} _ {n} ].\tag{4.52}
$$

Thus, (4.50) implies that 

$$
\boldsymbol {A} \boldsymbol {p} _ {1} = \lambda_ {1} \boldsymbol {p} _ {1}\tag{4.53}
$$

$$
\begin{array}{c} \vdots \\ A \boldsymbol {p} _ {n} = \lambda_ {n} \boldsymbol {p} _ {n}. \end{array}\tag{4.54}
$$

Therefore, the columns of P must be eigenvectors of A. 

Our definition of diagonalization requires that $P \in \mathbb { R } ^ { n \times n }$ is invertible, i.e., P has full rank (Theorem 4.3). This requires us to have n linearly independent eigenvectors $p _ { 1 } , \ldots , p _ { n } ,$ , i.e., the $\mathbf { \nabla } _ { \pmb { p } _ { i } }$ form a basis of $\mathbb { R } ^ { n }$ 

Theorem 4.20 (Eigendecomposition). A square matrix $A \in \mathbb { R } ^ { n \times n }$ can be factored into 

$$
\boldsymbol {A} = \boldsymbol {P D P} ^ {- 1},\tag{4.55}
$$

where $P \in \mathbb { R } ^ { n \times n }$ and D is a diagonal matrix whose diagonal entries are the eigenvalues of A, if and only if the eigenvectors of A form a basis of <sup>Rn</sup>. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/120b166123e1ba2a862324498ae065aa449c3e81408310eddf33e7e536d24da5.jpg)


Theorem 4.20 implies that only non-defective matrices can be diagonalized and that the columns of P are the n eigenvectors of $\pmb { A }$ . For symmetric matrices we can obtain even stronger outcomes for the eigenvalue decom position. 

Theorem 4.21. A symmetric matrix $\ b { S } \in \mathbb { R } ^ { n \times n }$ can always be diagonalized. 

Theorem 4.21 follows directly from the spectral theorem 4.15. Moreover, the spectral theorem states that we can find an ONB of eigenvectors of $\mathbb { R } ^ { n }$ . This makes $_ { \pmb { P } }$ an orthogonal matrix so that $\pmb { D } = \pmb { P } ^ { \top } \pmb { A } \pmb { P }$ 

Remark. The Jordan normal form of a matrix offers a decomposition that works for defective matrices (Lang, 1987) but is beyond the scope of this book. 

## Geometric Intuition for the Eigendecomposition

We can interpret the eigendecomposition of a matrix as follows (see also Figure 4.7): Let A be the transformation matrix of a linear mapping with respect to the standard basis $e _ { i }$ (blue arrows). $P ^ { - 1 }$ performs a basis change from the standard basis into the eigenbasis. Then, the diagonal D scales the vectors along these axes by the eigenvalues $\lambda _ { i } .$ . Finally, $_ { r }$ transforms these scaled vectors back into the standard/canonical coordinates yielding $\lambda _ { i } { \pmb p } _ { i }$ 

Figure 4.7 Intuition behind the eigendecomposition as sequential transformations. Top-left to bottom-left: $P ^ { - 1 }$ performs a basis change (here drawn in $\mathbb { R } ^ { \overset { \sim } { 2 } }$ and depicted as a rotation-like operation) from the standard basis into the eigenbasis. Bottom-left to bottom-right: D performs a scaling along the remapped orthogonal eigenvectors, depicted here by a circle being stretched to an ellipse. Bottom-right to top-right: $_ { P }$ undoes the basis change (depicted as a reverse rotation) and restores the original coordinate frame. 

## Example 4.11 (Eigendecomposition)

Let us compute the eigendecomposition of $\begin{array} { r } { A = \frac { 1 } { 2 } \left\lceil \begin{array} { l l } { 5 } & { - 2 } \\ { - 2 } & { 5 } \end{array} \right\rceil } \end{array}$ 

Step 1: Compute eigenvalues and eigenvectors. The characteristic 

polynomial of A is 

$$
\det (\boldsymbol {A} - \lambda \boldsymbol {I}) = \det \left(\left[ \begin{array}{c c} \frac {5}{2} - \lambda & - 1 \\ - 1 & \frac {5}{2} - \lambda \end{array} \right]\right)\tag{4.56a}
$$

$$
= (\frac {5}{2} - \lambda) ^ {2} - 1 = \lambda^ {2} - 5 \lambda + \frac {2 1}{4} = (\lambda - \frac {7}{2}) (\lambda - \frac {3}{2}).\tag{4.56b}
$$

Therefore, the eigenvalues of A are $\begin{array} { r } { \lambda _ { 1 } = \frac { 7 } { 2 } } \end{array}$ and $\begin{array} { r } { \lambda _ { 2 } = \frac { 3 } { 2 } } \end{array}$ (the roots of the characteristic polynomial), and the associated (normalized) eigenvectors are obtained via 

$$
\boldsymbol {A} \boldsymbol {p} _ {1} = \frac {7}{2} \boldsymbol {p} _ {1}, \quad \boldsymbol {A} \boldsymbol {p} _ {2} = \frac {3}{2} \boldsymbol {p} _ {2}.\tag{4.57}
$$

This yields 

$$
\boldsymbol {p} _ {1} = \frac {1}{\sqrt {2}} \left[ \begin{array}{c} 1 \\ - 1 \end{array} \right], \quad \boldsymbol {p} _ {2} = \frac {1}{\sqrt {2}} \left[ \begin{array}{c} 1 \\ 1 \end{array} \right].\tag{4.58}
$$

Step 2: Check for existence. The eigenvectors $\mathbf { \boldsymbol { p } } _ { 1 } , \mathbf { \boldsymbol { p } } _ { 2 }$ form a basis of $\mathbb { R } ^ { 2 }$ Therefore, A can be diagonalized. 

Step 3: Construct the matrix $_ { \pmb { P } }$ to diagonalize A. We collect the eigenvectors of A in P so that 

$$
\boldsymbol {P} = \left[ \boldsymbol {p} _ {1}, \boldsymbol {p} _ {2} \right] = \frac {1}{\sqrt {2}} \left[ \begin{array}{c c} 1 & 1 \\ - 1 & 1 \end{array} \right].\tag{4.59}
$$

We then obtain 

$$
\boldsymbol {P} ^ {- 1} \boldsymbol {A} \boldsymbol {P} = \left[ \begin{array}{c c} \frac {7}{2} & 0 \\ 0 & \frac {3}{2} \end{array} \right] = \boldsymbol {D}.\tag{4.60}
$$

Figure 4.7 visualizes the 

eigendecomposition 

Equivalently, we get (exploiting that $P ^ { - 1 } = P ^ { \top }$ since the eigenvectors $\pmb { p } _ { 1 }$ and $\pmb { p } _ { 2 }$ in this example form an ONB) 

$$
\boldsymbol {A} = \left[ \begin{array}{c c} 5 & - 2 \\ - 2 & 5 \end{array} \right]
$$

as a sequence of linear 

$$
\underbrace {\frac {1}{2} \left[ \begin{array}{c c} 5 & - 2 \\ - 2 & 5 \end{array} \right]} _ {A} = \underbrace {\frac {1}{\sqrt {2}} \left[ \begin{array}{c c} 1 & 1 \\ - 1 & 1 \end{array} \right]} _ {P} \underbrace {\left[ \begin{array}{c c} \frac {7}{2} & 0 \\ 0 & \frac {3}{2} \end{array} \right]} _ {D} \underbrace {\frac {1}{\sqrt {2}} \left[ \begin{array}{c c} 1 & - 1 \\ 1 & 1 \end{array} \right]} _ {P ^ {- 1}}.\tag{4.61}
$$

transformations. 

Diagonal matrices D can efficiently be raised to a power. Therefore, we can find a matrix power for a matrix $A \in \mathbb { R } ^ { n \times n }$ via the eigenvalue decomposition (if it exists) so that 

$$
\boldsymbol {A} ^ {k} = \left(\boldsymbol {P D P} ^ {- 1}\right) ^ {k} = \boldsymbol {P D} ^ {k} \boldsymbol {P} ^ {- 1}.\tag{4.62}
$$

Computing $D ^ { k }$ is efficient because we apply this operation individually to any diagonal element. 

Assume that the eigendecomposition $A = P D P ^ { - 1 }$ exists. Then, 

$$
\det (\boldsymbol {A}) = \det (\boldsymbol {P D P} ^ {- 1}) = \det (\boldsymbol {P}) \det (\boldsymbol {D}) \det (\boldsymbol {P} ^ {- 1})\tag{4.63a}
$$

4.5 Singular Value Decomposition 

$$
= \det (\boldsymbol {D}) = \prod_ {i} d _ {i i}\tag{4.63b}
$$

allows for an efficient computation of the determinant of A. 

The eigenvalue decomposition requires square matrices. It would be useful to perform a decomposition on general matrices. In the next section, we introduce a more general matrix decomposition technique, the singular value decomposition. 

## 4.5 Singular Value Decomposition

The singular value decomposition (SVD) of a matrix is a central matrix decomposition method in linear algebra. It has been referred to as the “fundamental theorem of linear algebra” (Strang, 1993) because it can be applied to all matrices, not only to square matrices, and it always exists. Moreover, as we will explore in the following, the SVD of a matrix $A ,$ which represents a linear mapping $\Phi : V \to W$ , quantifies the change between the underlying geometry of these two vector spaces. We recommend the work by Kalman (1996) and Roy and Banerjee (2014) for a deeper overview of the mathematics of the SVD. 

Theorem 4.22 (SVD Theorem). Let $\pmb { A } \in \mathbb { R } ^ { m \times n }$ be a rectangular matrix of rank $r \in [ 0 , \operatorname* { m i n } ( m , n ) ]$ . The SVD of A is a decomposition of the form 

$$
m \boxed {A} ^ {n} = m \boxed {U} ^ {m} m \boxed {\Sigma} ^ {n} \boxed {V ^ {\top}} _ {u} ^ {n}\tag{4.64}
$$

with an orthogonal matrix $\pmb { U } \in \mathbb { R } ^ { m \times m }$ with column vectors $\mathbf { { u } } _ { i } , i = 1 , \ldots , m ,$ and an orthogonal matrix $V \in \mathbb { R } ^ { n \times n }$ with column vectors $v _ { j } , j = 1 , \dotsc , n$ Moreover, Σ is an $m \times n$ matrix with $\Sigma _ { i i } = \sigma _ { i } \geqslant 0$ and $\Sigma _ { i j } = 0 , \ : i \neq j$ 

The diagonal entries $\sigma _ { i } , i = 1 , \ldots , r .$ , of Σ are called the singular values, $\mathbf { \Delta } \mathbf { u } _ { i }$ are called the left-singular vectors, and ${ \pmb v } _ { j }$ are called the right-singular vectors. By convention, the singular values are ordered, i.e., $\sigma _ { 1 } \geqslant \sigma _ { 2 } \geqslant$ $\sigma _ { r } \geqslant 0$ 

The singular value matrix Σ is unique, but it requires some attention. Observe that the $\pmb { \Sigma } \in \mathbb { R } ^ { m \times n }$ is rectangular. In particular, Σ is of the same size as A. This means that Σ has a diagonal submatrix that contains the singular values and needs additional zero padding. Specifically, if $m > n ,$ then the matrix Σ has diagonal structure up to row n and then consists of 

Figure 4.8 Intuition behind the SVD of a matrix $\pmb { A } \in \mathbb { R } ^ { 3 \times 2 }$ as sequential transformations. Top-left to bottom-left: $\mathbf { } _ { V } ^ { \top }$ performs a basis change in $\mathbb { R } ^ { 2 }$ Bottom-left to bottom-right: Σ scales and maps from $\mathbb { R } ^ { 2 }$ to $\mathbb { R } ^ { 3 }$ . The ellipse in the bottom-right lives in $\mathbb { R } ^ { 3 } .$ . The third dimension is orthogonal to the surface of the elliptical disk. Bottom-right to top-right: U performs a basis change within <sup>R3</sup>. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/af356776763df665515e80f011dc277e015c05dace77b415b88c0a4dd07d2e0a.jpg)


$\mathbf { 0 } ^ { \top }$ row vectors from $n + 1$ to m below so that 

$$
\boldsymbol {\Sigma} = \left[ \begin{array}{c c c} \sigma_ {1} & 0 & 0 \\ 0 & \ddots & 0 \\ 0 & 0 & \sigma_ {n} \\ 0 & \ldots & 0 \\ \vdots & & \vdots \\ 0 & \ldots & 0 \end{array} \right].\tag{4.65}
$$

If $m \ < \ n$ , the matrix Σ has a diagonal structure up to column m and columns that consist of 0 from $m + 1$ to n: 

$$
\boldsymbol {\Sigma} = \left[ \begin{array}{c c c c c c} \sigma_ {1} & 0 & 0 & 0 & \ldots & 0 \\ 0 & \ddots & 0 & \vdots & & \vdots \\ 0 & 0 & \sigma_ {m} & 0 & \ldots & 0 \end{array} \right].\tag{4.66}
$$

Remark. The SVD exists for any matrix $\pmb { A } \in \mathbb { R } ^ { m \times n }$ 

♢ 

## 4.5.1 Geometric Intuitions for the SVD

The SVD offers geometric intuitions to describe a transformation matrix A. In the following, we will discuss the SVD as sequential linear transformations performed on the bases. In Example 4.12, we will then apply transformation matrices of the SVD to a set of vectors in $\mathbb { R } ^ { 2 }$ , which allows us to visualize the effect of each transformation more clearly. 

The SVD of a matrix can be interpreted as a decomposition of a corresponding linear mapping (recall Section 2.7.1) $\Phi : \mathbb { R } ^ { n }  \mathbb { R } ^ { m }$ into three operations; see Figure 4.8. The SVD intuition follows superficially a similar structure to our eigendecomposition intuition, see Figure $4 . 7 \colon$ Broadly speaking, the SVD performs a basis change via $V ^ { ^ { \top } }$ followed by a scaling and augmentation (or reduction) in dimensionality via the singular value matrix Σ. Finally, it performs a second basis change via $U .$ . The SVD entails a number of important details and caveats, which is why we will review our intuition in more detail. 

Assume we are given a transformation matrix of a linear mapping $\Phi$ : $\mathbb { R } ^ { n } \to \mathbb { R } ^ { m }$ with respect to the standard bases B and $C$ of $\mathbb { R } ^ { n }$ and $\mathbb { R } ^ { m } { } _ { : }$ respectively. Moreover, assume a second basis $\tilde { B }$ of $\mathbb { R } ^ { n }$ and $\tilde { C }$ of $\mathbb { R } ^ { m }$ . Then 

1. The matrix $V$ performs a basis change in the domain $\mathbb { R } ^ { n }$ from $\tilde { B }$ (represented by the red and orange vectors ${ \pmb v } _ { 1 }$ and ${ \pmb v } _ { 2 }$ in the top-left of Figure 4.8) to the standard basis B. $\pmb { V } ^ { \top } = \pmb { V } ^ { - 1 }$ performs a basis change from $B$ to $\tilde { B } .$ . The red and orange vectors are now aligned with the canonical basis in the bottom-left of Figure 4.8. 

2. Having changed the coordinate system to ${ \tilde { B } } _ { : }$ , Σ scales the new coordinates by the singular values $\sigma _ { i }$ (and adds or deletes dimensions), i.e., Σ is the transformation matrix of Φ with respect to $\tilde { B }$ and ${ \tilde { C } } ,$ represented by the red and orange vectors being stretched and lying in the $e _ { 1 } – e _ { 2 }$ plane, which is now embedded in a third dimension in the bottom-right of Figure 4.8. 

3. U performs a basis change in the codomain $\mathbb { R } ^ { m }$ from $\tilde { C }$ into the canonical basis of $\mathbb { R } ^ { m }$ , represented by a rotation of the red and orange vectors out of the $e _ { 1 } – e _ { 2 }$ plane. This is shown in the top-right of Figure 4.8. 

The SVD expresses a change of basis in both the domain and codomain. This is in contrast with the eigendecomposition that operates within the same vector space, where the same basis change is applied and then undone. What makes the SVD special is that these two different bases are simultaneously linked by the singular value matrix Σ. 

## Example 4.12 (Vectors and the SVD)

Consider a mapping of a square grid of vectors $\mathcal { X } \in \mathbb { R } ^ { 2 }$ that fit in a box of size $2 \times 2$ centered at the origin. Using the standard basis, we map these vectors using 

$$
\begin{array}{r l} \boldsymbol {A} & = \left[ \begin{array}{c c} 1 & - 0. 8 \\ 0 & 1 \\ 1 & 0 \end{array} \right] = \boldsymbol {U} \boldsymbol {\Sigma} \boldsymbol {V} ^ {\top} \\ & = \left[ \begin{array}{c c c} - 0. 7 9 & 0 & - 0. 6 2 \\ 0. 3 8 & - 0. 7 8 & - 0. 4 9 \\ - 0. 4 8 & - 0. 6 2 & 0. 6 2 \end{array} \right] \left[ \begin{array}{c c} 1. 6 2 & 0 \\ 0 & 1. 0 \\ 0 & 0 \end{array} \right] \left[ \begin{array}{c c} - 0. 7 8 & 0. 6 2 \\ - 0. 6 2 & - 0. 7 8 \end{array} \right]. \end{array}\tag{4.67a}
$$

(4.67b) 

We start with a set of vectors $\mathcal { X }$ (colored dots; see top-left panel of Figure 4.9) arranged in a grid. We then apply $V ^ { \top } \in \mathbb { R } ^ { 2 \times \hat { 2 } }$ , which rotates $\mathcal { X } .$ The rotated vectors are shown in the bottom-left panel of Figure 4.9. We now map these vectors using the singular value matrix Σ to the codomain $\mathbb { R } ^ { 3 }$ (see the bottom-right panel in Figure 4.9). Note that all vectors lie in the $x _ { 1 } { - } x _ { 2 }$ plane. The third coordinate is always 0. The vectors in the $x _ { 1 } { - } x _ { 2 }$ plane have been stretched by the singular values. 

The direct mapping of the vectors by A to the codomain $\mathbb { R } ^ { 3 }$ equals the transformation of by $U \pmb { \Sigma } \pmb { V } ^ { \top }$ , where U performs a rotation within the codomain $\mathbb { R } ^ { 3 }$ so that the mapped vectors are no longer restricted to the $x _ { 1 } { - } x _ { 2 }$ plane; they still are on a plane as shown in the top-right panel of Figure 4.9. 

Figure 4.9 SVD and mapping of vectors (represented by discs). The panels follow the same anti-clockwise structure of Figure 4.8. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/d6fcb7030571fdbf1ed6b1a801f88279ac963b858aade000254b4864338296f8.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/d204446950d64933ab5697fda3517fb34eb7e727379d4cf84b87a65d7c27bb3d.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/c7a42bc147eadd2b2874cce40dc6ba16162319cc50db9a1e7812b656ba000320.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/f21b64cc5f112c1237089bf9f7e5284fe502c65861669962b69785ce35c98d97.jpg)


## 4.5.2 Construction of the SVD

We will next discuss why the SVD exists and show how to compute it in detail. The SVD of a general matrix shares some similarities with the eigendecomposition of a square matrix. 

Remark. Compare the eigendecomposition of an SPD matrix 

$$
\pmb {S} = \pmb {S} ^ {\top} = \pmb {P D P} ^ {\top}\tag{4.68}
$$

Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 

## 4.5 Singular Value Decomposition

with the corresponding SVD 

$$
\boldsymbol {S} = \boldsymbol {U} \boldsymbol {\Sigma} \boldsymbol {V} ^ {\top}.\tag{4.69}
$$

If we set 

$$
\boldsymbol {U} = \boldsymbol {P} = \boldsymbol {V}, \quad \boldsymbol {D} = \boldsymbol {\Sigma},\tag{4.70}
$$

we see that the SVD of SPD matrices is their eigendecomposition. 

In the following, we will explore why Theorem 4.22 holds and how the SVD is constructed. Computing the SVD of $\pmb { A } \in \mathbb { R } ^ { m \times n }$ is equivalent to finding two sets of orthonormal bases $U = ( \pmb { u } _ { 1 } , \dots , \pmb { u } _ { m } )$ and $V =$ $( { \pmb v } _ { 1 } , \ldots , { \pmb v } _ { n } )$ of the codomain $\mathbb { R } ^ { m }$ and the domain $\mathbb { R } ^ { n }$ , respectively. From these ordered bases, we will construct the matrices U and $V$ 

Our plan is to start with constructing the orthonormal set of rightsingular vectors $\pmb { v } _ { 1 } , \dots , \pmb { v } _ { n } \in \mathbb { R } ^ { n }$ . We then construct the orthonormal set of left-singular vectors $\pmb { u } _ { 1 } , \ldots , \pmb { u } _ { m } \in \mathbb { R } ^ { m }$ . Thereafter, we will link the two and require that the orthogonality of the ${ \mathbf { } } v _ { i }$ is preserved under the transformation of $A .$ . This is important because we know that the images $\mathbf { \nabla } _ { \mathbf { A } v _ { i } }$ form a set of orthogonal vectors. We will then normalize these images by scalar factors, which will turn out to be the singular values. 

Let us begin with constructing the right-singular vectors. The spectral theorem (Theorem 4.15) tells us that the eigenvectors of a symmetric matrix form an ONB, which also means it can be diagonalized. Moreover, from Theorem 4.14 we can always construct a symmetric, positive semidefinite matrix $\pmb { A } ^ { \top } \pmb { A } \ \in \ \mathbb { R } ^ { n \times n }$ from any rectangular matrix ${ \textbf { \textit { A } } } \in$ $\mathbb { R } ^ { m \times n }$ . Thus, we can always diagonalize $A ^ { \top } A$ and obtain 

$$
\boldsymbol {A} ^ {\top} \boldsymbol {A} = \boldsymbol {P} \boldsymbol {D} \boldsymbol {P} ^ {\top} = \boldsymbol {P} \left[ \begin{array}{c c c} \lambda_ {1} & \dots & 0 \\ \vdots & \ddots & \vdots \\ 0 & \dots & \lambda_ {n} \end{array} \right] \boldsymbol {P} ^ {\top},\tag{4.71}
$$

where $_ { \pmb { P } }$ is an orthogonal matrix, which is composed of the orthonormal eigenbasis. The $\lambda _ { i } \geqslant 0$ are the eigenvalues of $\boldsymbol { A } ^ { \top } \boldsymbol { A }$ . Let us assume the SVD of A exists and inject (4.64) into (4.71). This yields 

$$
\boldsymbol {A} ^ {\top} \boldsymbol {A} = (\boldsymbol {U} \boldsymbol {\Sigma} \boldsymbol {V} ^ {\top}) ^ {\top} (\boldsymbol {U} \boldsymbol {\Sigma} \boldsymbol {V} ^ {\top}) = \boldsymbol {V} \boldsymbol {\Sigma} ^ {\top} \boldsymbol {U} ^ {\top} \boldsymbol {U} \boldsymbol {\Sigma} \boldsymbol {V} ^ {\top},\tag{4.72}
$$

where $U , V$ are orthogonal matrices. Therefore, with $U ^ { \top } U = I$ we obtain 

$$
\boldsymbol {A} ^ {\top} \boldsymbol {A} = \boldsymbol {V} \boldsymbol {\Sigma} ^ {\top} \boldsymbol {\Sigma} \boldsymbol {V} ^ {\top} = \boldsymbol {V} \left[ \begin{array}{c c c} \sigma_ {1} ^ {2} & 0 & 0 \\ 0 & \ddots & 0 \\ 0 & 0 & \sigma_ {n} ^ {2} \end{array} \right] \boldsymbol {V} ^ {\top}.\tag{4.73}
$$

Comparing now (4.71) and (4.73), we identify 

$$
\boldsymbol {V} ^ {\top} = \boldsymbol {P} ^ {\top},
$$

$$
\sigma_ {i} ^ {2} = \lambda_ {i}.\tag{4.74}
$$

(4.75) 



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



Therefore, the eigenvectors of $A ^ { \top } A$ that compose P are the right-singular vectors V of A (see (4.74)). The eigenvalues of $A ^ { \top } A$ are the squared singular values of Σ (see (4.75)). 

To obtain the left-singular vectors $U _ { z }$ , we follow a similar procedure. We start by computing the SVD of the symmetric matrix $\boldsymbol { A } \boldsymbol { A } ^ { \intercal } \in \mathbb { R } ^ { m \times m }$ (instead of the previous $\pmb { A } ^ { \top } \pmb { A } \in \mathbb { R } ^ { n \times n } )$ . The SVD of A yields 

$$
\begin{array}{l} \boldsymbol {A} \boldsymbol {A} ^ {\top} = (\boldsymbol {U} \boldsymbol {\Sigma} \boldsymbol {V} ^ {\top}) (\boldsymbol {U} \boldsymbol {\Sigma} \boldsymbol {V} ^ {\top}) ^ {\top} = \boldsymbol {U} \boldsymbol {\Sigma} \boldsymbol {V} ^ {\top} \boldsymbol {V} \boldsymbol {\Sigma} ^ {\top} \boldsymbol {U} ^ {\top} \\ = \boldsymbol {U} \left[ \begin{array}{c c c} \sigma_ {1} ^ {2} & 0 & 0 \\ 0 & \ddots & 0 \\ 0 & 0 & \sigma_ {m} ^ {2} \end{array} \right] \boldsymbol {U} ^ {\top}. \end{array}\tag{4.76a}
$$

(4.76b) 

The spectral theorem tells us that $\pmb { A } \pmb { A } ^ { \top } = \pmb { S } \pmb { D } \pmb { S } ^ { \top }$ can be diagonalized and we can find an ONB of eigenvectors of $A A ^ { \top }$ , which are collected in S. The orthonormal eigenvectors of $A A ^ { \top }$ are the left-singular vectors U and form an orthonormal basis in the codomain of the SVD. 

This leaves the question of the structure of the matrix Σ. Since $A A ^ { \top }$ and $A ^ { \top } A$ have the same nonzero eigenvalues (see page 106), the nonzero entries of the Σ matrices in the SVD for both cases have to be the same. 

The last step is to link up all the parts we touched upon so far. We have an orthonormal set of right-singular vectors in $V$ . To finish the construction of the SVD, we connect them with the orthonormal vectors $U .$ . To reach this goal, we use the fact the images of the ${ \mathbf { } } v _ { i }$ under A have to be orthogonal, too. We can show this by using the results from Section 3.4. We require that the inner product between $\mathbf { \nabla } _ { \mathbf { A } v _ { i } }$ and $\mathbf { \nabla } A v _ { j }$ must be 0 for $i \neq j$ . For any two orthogonal eigenvectors $\mathbf { } { \pmb v } _ { i } , { \pmb v } _ { j } , i \neq j$ , it holds that 

$$
\left(\boldsymbol {A} \boldsymbol {v} _ {i}\right) ^ {\top} \left(\boldsymbol {A} \boldsymbol {v} _ {j}\right) = \boldsymbol {v} _ {i} ^ {\top} \left(\boldsymbol {A} ^ {\top} \boldsymbol {A}\right) \boldsymbol {v} _ {j} = \boldsymbol {v} _ {i} ^ {\top} \left(\lambda_ {j} \boldsymbol {v} _ {j}\right) = \lambda_ {j} \boldsymbol {v} _ {i} ^ {\top} \boldsymbol {v} _ {j} = 0.\tag{4.77}
$$

For the case $m \ \geqslant \ r ,$ , it holds that $\{ A v _ { 1 } , \ldots , A v _ { r } \}$ is a basis of an $r \mathrm { - }$ dimensional subspace of $\mathbb { R } ^ { m }$ 

To complete the SVD construction, we need left-singular vectors that are orthonormal: We normalize the images of the right-singular vectors $\mathbf { A } \pmb { v } _ { i }$ and obtain 

$$
\boldsymbol {u} _ {i} := \frac {\boldsymbol {A} \boldsymbol {v} _ {i}}{\| \boldsymbol {A} \boldsymbol {v} _ {i} \|} = \frac {1}{\sqrt {\lambda_ {i}}} \boldsymbol {A} \boldsymbol {v} _ {i} = \frac {1}{\sigma_ {i}} \boldsymbol {A} \boldsymbol {v} _ {i},\tag{4.78}
$$

where the last equality was obtained from (4.75) and (4.76b), showing us that the eigenvalues of $A A ^ { \top }$ are such that $\sigma _ { i } ^ { 2 } = \lambda _ { i }$ 

Therefore, the eigenvectors of $A ^ { \top } A$ , which we know are the rightsingular vectors ${ \mathbf { } } v _ { i s }$ and their normalized images under A, the left-singular vectors ${ \bf { u } } _ { i } ,$ , form two self-consistent ONBs that are connected through the singular value matrix Σ. 

Let us rearrange (4.78) to obtain the singular value equation 

$$
\boldsymbol {A} \boldsymbol {v} _ {i} = \sigma_ {i} \boldsymbol {u} _ {i}, \quad i = 1, \dots , r.\tag{4.79}
$$

## 4.5 Singular Value Decomposition

This equation closely resembles the eigenvalue equation (4.25), but the vectors on the left- and the right-hand sides are not the same. 

For $n < m , ( 4 . 7 9 )$ holds only for $i \leqslant n ,$ but (4.79) says nothing about the $\mathbf { \Delta } \mathbf { u } _ { i }$ for $i > n$ . However, we know by construction that they are orthonormal. Conversely, for $m < n , ( 4 . 7 9 )$ holds only for $i \leqslant m$ . For $i > m$ we have $\mathbf { } A \pmb { v } _ { i } = \mathbf { 0 }$ and we still know that the ${ \mathbf { } } v _ { i }$ form an orthonormal set. This means that the SVD also supplies an orthonormal basis of the kernel (null space) of $A ,$ , the set of vectors x with $\pmb { A } \pmb { x } = \mathbf { 0 }$ (see Section 2.7.3). 

Concatenating the ${ \mathbf { } } v _ { i }$ as the columns of V and the $\mathbf { \Delta } \mathbf { u } _ { i }$ as the columns of U yields 

$$
\boldsymbol {A} \boldsymbol {V} = \boldsymbol {U} \boldsymbol {\Sigma},\tag{4.80}
$$

where Σ has the same dimensions as $\pmb { A }$ and a diagonal structure for rows $1 , \ldots , r$ . Hence, right-multiplying with $V ^ { ^ { \top } }$ yields $\mathbf { \pmb { A } } = \pmb { U } \pmb { \Sigma V } ^ { \top }$ , which is the SVD of $\pmb { A }$ . 

## Example 4.13 (Computing the SVD)

Let us find the singular value decomposition of 

$$
\boldsymbol {A} = \left[ \begin{array}{c c c} 1 & 0 & 1 \\ - 2 & 1 & 0 \end{array} \right].\tag{4.81}
$$

The SVD requires us to compute the right-singular vectors $v _ { j } ,$ the singular values $\sigma _ { k }$ , and the left-singular vectors $\mathbf { \Delta } \mathbf { u } _ { i }$ 

Step 1: Right-singular vectors as the eigenbasis of $A ^ { \top } A$ 

We start by computing 

$$
\boldsymbol {A} ^ {\top} \boldsymbol {A} = \left[ \begin{array}{c c} 1 & - 2 \\ 0 & 1 \\ 1 & 0 \end{array} \right] \left[ \begin{array}{c c c} 1 & 0 & 1 \\ - 2 & 1 & 0 \end{array} \right] = \left[ \begin{array}{c c c} 5 & - 2 & 1 \\ - 2 & 1 & 0 \\ 1 & 0 & 1 \end{array} \right].\tag{4.82}
$$

We compute the singular values and right-singular vectors $\boldsymbol { v } _ { j }$ through the eigenvalue decomposition of $A ^ { \top } A$ , which is given as 

$$
\boldsymbol {A} ^ {\top} \boldsymbol {A} = \left[ \begin{array}{c c c} \frac {5}{\sqrt {3 0}} & 0 & \frac {- 1}{\sqrt {6}} \\ \frac {- 2}{\sqrt {3 0}} & \frac {1}{\sqrt {5}} & \frac {- 2}{\sqrt {6}} \\ \frac {1}{\sqrt {3 0}} & \frac {2}{\sqrt {5}} & \frac {1}{\sqrt {6}} \end{array} \right] \left[ \begin{array}{c c c} 6 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{array} \right] \left[ \begin{array}{c c c} \frac {5}{\sqrt {3 0}} & \frac {- 2}{\sqrt {3 0}} & \frac {1}{\sqrt {3 0}} \\ 0 & \frac {1}{\sqrt {5}} & \frac {2}{\sqrt {5}} \\ \frac {- 1}{\sqrt {6}} & \frac {- 2}{\sqrt {6}} & \frac {1}{\sqrt {6}} \end{array} \right] = \boldsymbol {P D P} ^ {\top},\tag{4.83}
$$

and we obtain the right-singular vectors as the columns of $_ { \pmb { P } }$ so that 

$$
\boldsymbol {V} = \boldsymbol {P} = \left[ \begin{array}{c c c} \frac {5}{\sqrt {3 0}} & 0 & \frac {- 1}{\sqrt {6}} \\ \frac {- 2}{\sqrt {3 0}} & \frac {1}{\sqrt {5}} & \frac {- 2}{\sqrt {6}} \\ \frac {1}{\sqrt {3 0}} & \frac {2}{\sqrt {5}} & \frac {1}{\sqrt {6}} \end{array} \right].\tag{4.84}
$$

Step 2: Singular-value matrix. 

As the singular values $\sigma _ { i }$ are the square roots of the eigenvalues of 



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



$A ^ { \top } A$ we obtain them straight from D. Since rk $( A ) = 2$ , there are only two nonzero singular values: $\sigma _ { 1 } = \sqrt { 6 }$ and $\sigma _ { 2 } = 1$ . The singular value matrix must be the same size as $A _ { i }$ , and we obtain 

$$
\boldsymbol {\Sigma} = \left[ \begin{array}{c c c} \sqrt {6} & 0 & 0 \\ 0 & 1 & 0 \end{array} \right].\tag{4.85}
$$

Step 3: Left-singular vectors as the normalized image of the rightsingular vectors. 

We find the left-singular vectors by computing the image of the rightsingular vectors under A and normalizing them by dividing them by their corresponding singular value. We obtain 

$$
\pmb {u} _ {1} = \frac {1}{\sigma_ {1}} \pmb {A} \pmb {v} _ {1} = \frac {1}{\sqrt {6}} \left[ \begin{array}{l l l} 1 & 0 & 1 \\ - 2 & 1 & 0 \end{array} \right] \left[ \begin{array}{l} \frac {5}{\sqrt {3 0}} \\ \frac {- 2}{\sqrt {3 0}} \\ \frac {1}{\sqrt {3 0}} \end{array} \right] = \left[ \begin{array}{l} \frac {1}{\sqrt {5}} \\ - \frac {2}{\sqrt {5}} \end{array} \right],\tag{4.86}
$$

$$
\boldsymbol {u} _ {2} = \frac {1}{\sigma_ {2}} \boldsymbol {A} \boldsymbol {v} _ {2} = \frac {1}{1} \left[ \begin{array}{c c c} 1 & 0 & 1 \\ - 2 & 1 & 0 \end{array} \right] \left[ \begin{array}{c} 0 \\ \frac {1}{\sqrt {5}} \\ \frac {2}{\sqrt {5}} \end{array} \right] = \left[ \begin{array}{c} \frac {2}{\sqrt {5}} \\ \frac {1}{\sqrt {5}} \end{array} \right],\tag{4.87}
$$

$$
\boldsymbol {U} = \left[ \boldsymbol {u} _ {1}, \boldsymbol {u} _ {2} \right] = \frac {1}{\sqrt {5}} \left[ \begin{array}{c c} 1 & 2 \\ - 2 & 1 \end{array} \right].\tag{4.88}
$$

Note that on a computer the approach illustrated here has poor numerical behavior, and the SVD of A is normally computed without resorting to the eigenvalue decomposition of $A ^ { \top } A$ 

## 4.5.3 Eigenvalue Decomposition vs. Singular Value Decomposition

Let us consider the eigendecomposition $A = P D P ^ { - 1 }$ and the SVD $A =$ $U \pmb { \Sigma } \pmb { V } ^ { \top }$ and review the core elements of the past sections. 

The SVD always exists for any matrix $\mathbb { R } ^ { m \times n }$ . The eigendecomposition is only defined for square matrices $\mathbb { R } ^ { n \times n }$ and only exists if we can find a basis of eigenvectors of $\mathbb { R } ^ { n }$ 

The vectors in the eigendecomposition matrix $_ { P }$ are not necessarily orthogonal, $\mathrm { i . e . }$ , the change of basis is not a simple rotation and scaling. On the other hand, the vectors in the matrices $U$ and $V$ in the SVD are orthonormal, so they do represent rotations. 

Both the eigendecomposition and the SVD are compositions of three linear mappings: 

1. Change of basis in the domain 

2. Independent scaling of each new basis vector and mapping from domain to codomain 

3. Change of basis in the codomain 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/05ef5abdc7cbf0bd94e659da2f938634ad2a1b48eb2e8199bf9c2bf79471e823.jpg)



Figure 4.10 Movie ratings of three people for four movies and its SVD decomposition.


A key difference between the eigendecomposition and the SVD is that in the SVD, domain and codomain can be vector spaces of different dimensions. 

In the SVD, the left- and right-singular vector matrices U and V are generally not inverse of each other (they perform basis changes in dif ferent vector spaces). In the eigendecomposition, the basis change matrices P and $P ^ { - 1 }$ are inverses of each other. 

In the SVD, the entries in the diagonal matrix Σ are all real and nonnegative, which is not generally true for the diagonal matrix in the eigendecomposition. 

The SVD and the eigendecomposition are closely related through their projections 

– The left-singular vectors of A are eigenvectors of $A A ^ { \top }$ 

– The right-singular vectors of A are eigenvectors of $A ^ { \top } A$ 

– The nonzero singular values of A are the square roots of the nonzero eigenvalues of both $A A ^ { \top }$ and $A ^ { \top } A$ 

For symmetric matrices $A \in \mathbb { R } ^ { n \times n }$ , the eigenvalue decomposition and the SVD are one and the same, which follows from the spectral theorem 4.15. 

Example 4.14 (Finding Structure in Movie Ratings and Consumers) Let us add a practical interpretation of the SVD by analyzing data on people and their preferred movies. Consider three viewers (Ali, Beatrix, Chandra) rating four different movies (Star Wars, Blade Runner, Amelie, Delicatessen). Their ratings are values between 0 (worst) and 5 (best) and encoded in a data matrix $\pmb { A } \in \mathbb { R } ^ { 4 \times 3 }$ as shown in Figure 4.10. Each row represents a movie and each column a user. Thus, the column vectors of movie ratings, one for each viewer, are $\begin{array} { r } { \pmb { x } _ { \mathrm { A l i } } , \pmb { x } _ { \mathrm { B e a t r i x } } , \pmb { x } _ { \mathrm { C h a n d r a } } . } \end{array}$ 

These two “spaces” are only meaningfully spanned by the respective viewer and movie data if the data itself covers a sufficient diversity of viewers and movies. 

Factoring A using the SVD offers us a way to capture the relationships of how people rate movies, and especially if there is a structure linking which people like which movies. Applying the SVD to our data matrix A makes a number of assumptions: 

1. All viewers rate movies consistently using the same linear mapping. 

2. There are no errors or noise in the ratings. 

3. We interpret the left-singular vectors $\mathbf { \Delta } \mathbf { u } _ { i }$ as stereotypical movies and the right-singular vectors ${ \pmb v } _ { j }$ as stereotypical viewers. 

We then make the assumption that any viewer’s specific movie preferences can be expressed as a linear combination of the ${ \pmb v } _ { j }$ . Similarly, any movie’s like-ability can be expressed as a linear combination of the $\mathbf { \Delta } \mathbf { u } _ { i }$ . Therefore, a vector in the domain of the SVD can be interpreted as a viewer in the “space” of stereotypical viewers, and a vector in the codomain of the SVD correspondingly as a movie in the “space” of stereotypical movies. Let us inspect the SVD of our movie-user matrix. The first left-singular vector ${ \pmb u } _ { 1 }$ has large absolute values for the two science fiction movies and a large first singular value (red shading in Figure 4.10). Thus, this groups a type of users with a specific set of movies (science fiction theme). Similarly, the first right-singular ${ \pmb v } _ { 1 }$ shows large absolute values for Ali and Beatrix, who give high ratings to science fiction movies (green shading in Figure 4.10). This suggests that ${ \pmb v } _ { 1 }$ reflects the notion of a science fiction lover. 

Similarly, ${ \bf { u } } _ { 2 } ,$ seems to capture a French art house film theme, and ${ \pmb v } _ { 2 }$ indicates that Chandra is close to an idealized lover of such movies. An idealized science fiction lover is a purist and only loves science fiction movies, so a science fiction lover ${ \pmb v } _ { 1 }$ gives a rating of zero to everything but science fiction themed—this logic is implied by the diagonal substructure for the singular value matrix Σ. A specific movie is therefore represented by how it decomposes (linearly) into its stereotypical movies. Likewise, a person would be represented by how they decompose (via linear combination) into movie themes. 

It is worth to briefly discuss SVD terminology and conventions, as there are different versions used in the literature. While these differences can be confusing, the mathematics remains invariant to them. 

For convenience in notation and abstraction, we use an SVD notation where the SVD is described as having two square left- and right-singular vector matrices, but a non-square singular value matrix. Our definition (4.64) for the SVD is sometimes called the full SVD. 

Some authors define the SVD a bit differently and focus on square singular matrices. Then, for $\pmb { A } \in \mathbb { R } ^ { m \times n }$ and m $\geqslant n _ { : }$ 

$$
\mathbf {A} _ {m \times n} = \mathbf {U} _ {m \times n} \sum_ {n \times n} \mathbf {V} ^ {\top}.\tag{4.89}
$$

Sometimes this formulation is called the reduced SVD (e.g., Datta (2010)) or the SVD (e.g., Press et al. (2007)). This alternative format changes merely how the matrices are constructed but leaves the mathematical structure of the SVD unchanged. The convenience of this alternative formulation is that Σ is diagonal, as in the eigenvalue decomposition. 

In Section 4.6, we will learn about matrix approximation techniques using the SVD, which is also called the truncated SVD. 

It is possible to define the SVD of a rank-r matrix A so that U is an $m \times r$ matrix, Σ a diagonal matrix $r \times r ,$ and V an $r \times n$ matrix. This construction is very similar to our definition, and ensures that the diagonal matrix Σ has only nonzero entries along the diagonal. The main convenience of this alternative notation is that Σ is diagonal, as in the eigenvalue decomposition. 

A restriction that the SVD for A only applies to $m \times n$ matrices with m $> n$ is practically unnecessary. When $m < n ,$ the SVD decomposition will yield Σ with more zero columns than rows and, consequently, the singular values $\sigma _ { m + 1 } , \ldots , \sigma _ { n }$ are 0. 

The SVD is used in a variety of applications in machine learning from least-squares problems in curve fitting to solving systems of linear equations. These applications harness various important properties of the SVD, its relation to the rank of a matrix, and its ability to approximate matrices of a given rank with lower-rank matrices. Substituting a matrix with its SVD has often the advantage of making calculation more robust to numerical rounding errors. As we will explore in the next section, the SVD’s ability to approximate matrices with “simpler” matrices in a principled manner opens up machine learning applications ranging from dimensionality reduction and topic modeling to data compression and clustering. 

## 4.6 Matrix Approximation

We considered the SVD as a way to factorize $\pmb { A } = \pmb { U } \pmb { \Sigma } \pmb { V } ^ { \top } \in \mathbb { R } ^ { m \times n }$ into the product of three matrices, where $\pmb { U } \in \mathbb { R } ^ { m \times m }$ and $V \in \mathbb { R } ^ { n \times n }$ are orthogonal and Σ contains the singular values on its main diagonal. Instead of doing the full SVD factorization, we will now investigate how the SVD allows us to represent a matrix A as a sum of simpler (low-rank) matrices $A _ { i }$ , which lends itself to a matrix approximation scheme that is cheaper to compute than the full SVD. 

We construct a rank-1 matrix $\pmb { A } _ { i } \in \mathbb { R } ^ { m \times n }$ as 

$$
\boldsymbol {A} _ {i} := \boldsymbol {u} _ {i} \boldsymbol {v} _ {i} ^ {\top},\tag{4.90}
$$

which is formed by the outer product of the ith orthogonal column vector of U and V . Figure 4.11 shows an image of Stonehenge, which can be represented by a matrix $\pmb { A } \in \mathbb { R } ^ { 1 4 3 2 \times 1 9 1 0 }$ , and some outer products $A _ { i } ,$ as defined in (4.90). 

©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 

Figure 4.11 Image processing with the SVD. (a) The original grayscale image is a $1 , 4 3 2 \times 1 , 9 1 0$ matrix of values between 0 (black) and 1 (white). (b)–(f) Rank-1 matrices 

$A _ { 1 } , \dotsc , A _ { 5 }$ and their corresponding singular values $\sigma _ { 1 } , \ldots , \sigma _ { 5 } .$ The grid-like structure of each rank-1 matrix is imposed by the outer-product of the left and right-singular vectors. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/7d8480eb4bb51f8e439e469b32be4d60bbd66082431d98da3caa2a96175c56de.jpg)



(a) Original image A.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/15fda871866c1490e5bca2077c51d9ac9338d03c98839c6e8af8f9436c42982f.jpg)



Matrix Decompositions



(b) A<sub>1</sub>, σ<sub>1</sub> ≈ 228, 052.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/1cf9df7859854c4759e7b3bf32bdad17b755b1bdfc698556739b8636aba8eab0.jpg)



(c) A , σ ≈ 40, 647.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/9abcf77d5372aac7e22e61a7716a8691fdb5c729199ff74268c197fdab35bf9f.jpg)



(d) $\pmb { A } _ { 3 }$ , $\sigma _ { 3 }$ ≈ 26, 125.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/b86f700df59243e39cc405027180d261ae9cb00f40822062159345ed5ac4b48d.jpg)



(e) $\mathbf { A } _ { 4 } ,$ $\sigma _ { 4 }$ ≈ 20, 232.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/905ebda9a1f03b92a30fc3e4c7dac1c900c4950bde422cc6080bd136491262da.jpg)



(f) $A _ { 5 } ,$ $\sigma _ { 5 }$ ≈ 15, 436.


A matrix $\pmb { A } \in \mathbb { R } ^ { m \times n }$ of rank r can be written as a sum of rank-1 matrices $\mathbf { A } _ { i }$ so that 

$$
\boldsymbol {A} = \sum_ {i = 1} ^ {r} \sigma_ {i} \boldsymbol {u} _ {i} \boldsymbol {v} _ {i} ^ {\top} = \sum_ {i = 1} ^ {r} \sigma_ {i} \boldsymbol {A} _ {i},\tag{4.91}
$$

where the outer-product matrices $A _ { i }$ are weighted by the ith singular value $\sigma _ { i }$ . We can see why (4.91) holds: The diagonal structure of the singular value matrix Σ multiplies only matching left- and right-singular vectors $\mathbf { \boldsymbol { u } } _ { i } \mathbf { \boldsymbol { v } } _ { i } ^ { \top }$ and scales them by the corresponding singular value $\sigma _ { i }$ . All terms $\begin{array} { r } { \sum _ { i j } { \pmb u } _ { i } { \pmb v } _ { j } ^ { \top } } \end{array}$ vanish for $i \neq j$ because Σ is a diagonal matrix. Any terms $i > r$ vanish because the corresponding singular values are 0. 

In (4.90), we introduced rank-1 matrices $A _ { i }$ . We summed up the $r$ individual rank-1 matrices to obtain a rank-r matrix $A ;$ see (4.91). If the sum does not run over all matrices $A _ { i } , i = 1 , \ldots , r ,$ but only up to an intermediate value $k < r$ , we obtain a rank-k approximation 

$$
\widehat {\boldsymbol {A}} (k) := \sum_ {i = 1} ^ {k} \sigma_ {i} \boldsymbol {u} _ {i} \boldsymbol {v} _ {i} ^ {\top} = \sum_ {i = 1} ^ {k} \sigma_ {i} \boldsymbol {A} _ {i}\tag{4.92}
$$

of A with r $\hat { \bf \Phi } ( \widehat { \bf A } ( k ) ) = k$ . Figure 4.12 shows low-rank approximations $\hat { \bf A } ( k )$ of an original image A of Stonehenge. The shape of the rocks becomes increasingly visible and clearly recognizable in the rank-5 approximation. While the original image requires 1, 432 1, 910 = 2, 735, 120 numbers, the rank-5 approximation requires us only to store the five singular values and the five left- and right-singular vectors (1, 432 and 1, 910- dimensional each) for a total of $5 \cdot ( 1 , 4 3 2 + 1 , 9 1 0 + 1 ) = 1 6$ , 715 numbers – just above 0.6% of the original. 

To measure the difference (error) between A and its rank-k approximation $\hat { \bf A } ( k )$ , we need the notion of a norm. In Section 3.1, we already used 

## 4.6 Matrix Approximation

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/77618b93272e20e6f255adae5e7da0cbc35bc4ff06e840335f725eecfef6d62f.jpg)



(a) Original image A.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/97f5140f080ee84e102d0de0fbd28bd1163a07312139396b86140490f368eec0.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/c9a0be6a084de8443fe255929a779b77d9e16f40467d7e1fbffa4c27d7816ae9.jpg)



(b) Rank-1 approximation A (1).(c) Rank-2 approximation $\widehat { \mathbf { A } } ( 2 )$


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/965b04cce77edd0a07308a4a0473e315e6840e966615d497231473d9e073baf2.jpg)



Figure 4.12 Image reconstruction with the SVD. (a) Original image. (b)–(f) Image reconstruction using the low-rank approximation of the SVD, where the rank-k approximation is given by A (k) = $\textstyle \sum _ { i = 1 } ^ { k } \sigma _ { i } A _ { i }$


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/a88aea11eed5641d771d5d141e27de81ec33feb2315c92a82044375a7cc7e940.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/bd2886a0-f569-4311-898d-9dc7e40afe95/60abc16b59520df351fa6e126286b656cf01004110ed8f9175260010345315ba.jpg)



(d) Rank-3 approximation $\widehat { \mathbf { A } } ( 3 )$ .(e) Rank-4 approximation A (4).(f) Rank-5 approximation $\widehat { \mathbf { A } } ( 5 )$


norms on vectors that measure the length of a vector. By analogy we can also define norms on matrices. 

Definition 4.23 (Spectral Norm of a Matrix). For $\pmb { x } \in \mathbb { R } ^ { n } \backslash \{ \mathbf { 0 } \}$ , the spectral norm of a matrix $\pmb { A } \in \mathbb { R } ^ { m \times n }$ is defined as 

$$
\| \boldsymbol {A} \| _ {2} := \max _ {\boldsymbol {x}} \frac {\| \boldsymbol {A x} \| _ {2}}{\| \boldsymbol {x} \| _ {2}}.\tag{4.93}
$$

We introduce the notation of a subscript in the matrix norm (left-hand side), similar to the Euclidean norm for vectors (right-hand side), which has subscript 2. The spectral norm (4.93) determines how long any vector x can at most become when multiplied by A. 

Theorem 4.24. The spectral norm of A is its largest singular value $\sigma _ { 1 }$ . 

We leave the proof of this theorem as an exercise. 

Theorem 4.25 (Eckart-Young Theorem (Eckart and Young, 1936)). Consider a matrix $\pmb { A } \in \mathbb { R } ^ { m \times n }$ of rank r and let $B \in \mathbb { R } ^ { m \times n }$ be a matrix of rank k. For any $k \leqslant r$ with $\begin{array} { r } { \hat { \pmb { A } } ( k ) = \sum _ { i = 1 } ^ { k } \sigma _ { i } \pmb { u } _ { i } \pmb { v } _ { i } ^ { \top } } \end{array}$ it holds that 

$$
\widehat {\boldsymbol {A}} (k) = \operatorname{argmin} _ {\mathrm{rk} (\boldsymbol {B}) = k} \left\| \boldsymbol {A} - \boldsymbol {B} \right\| _ {2},\tag{4.94}
$$

$$
\left\| \pmb {A} - \widehat {\pmb {A}} (k) \right\| _ {2} = \sigma_ {k + 1}.\tag{4.95}
$$

The Eckart-Young theorem states explicitly how much error we introduce by approximating A using a rank-k approximation. We can interpret the rank-k approximation obtained with the SVD as a projection of the full-rank matrix A onto a lower-dimensional space of rank-at-most-k matrices. Of all possible projections, the SVD minimizes the error (with respect to the spectral norm) between A and any rank-k approximation. 

We can retrace some of the steps to understand why (4.95) should hold. 

We observe that the difference between $A - { \widehat { A } } ( k )$ is a matrix containing the sum of the remaining rank-1 matrices 

$$
\boldsymbol {A} - \widehat {\boldsymbol {A}} (k) = \sum_ {i = k + 1} ^ {r} \sigma_ {i} \boldsymbol {u} _ {i} \boldsymbol {v} _ {i} ^ {\top}.\tag{4.96}
$$

By Theorem 4.24, we immediately obtain $\sigma _ { k + 1 }$ as the spectral norm of the difference matrix. Let us have a closer look at (4.94). If we assume that there is another matrix B with $\operatorname { r k } ( B ) \leqslant k$ , such that 

$$
\left\| \boldsymbol {A} - \boldsymbol {B} \right\| _ {2} <   \left\| \boldsymbol {A} - \widehat {\boldsymbol {A}} (k) \right\| _ {2},\tag{4.97}
$$

then there exists an at least $( n - k )$ -dimensional null space $Z \subseteq \mathbb { R } ^ { n }$ , such that $\mathbf { \boldsymbol { x } } \in Z$ implies that $\pmb { B x = 0 }$ . Then it follows that 

$$
\left\| \boldsymbol {A} \boldsymbol {x} \right\| _ {2} = \left\| (\boldsymbol {A} - \boldsymbol {B}) \boldsymbol {x} \right\| _ {2},\tag{4.98}
$$

and by using a version of the Cauchy-Schwartz inequality (3.17) that encompasses norms of matrices, we obtain 

$$
\left\| \boldsymbol {A} \boldsymbol {x} \right\| _ {2} \leqslant \left\| \boldsymbol {A} - \boldsymbol {B} \right\| _ {2} \left\| \boldsymbol {x} \right\| _ {2} <   \sigma_ {k + 1} \left\| \boldsymbol {x} \right\| _ {2}.\tag{4.99}
$$

However, there exists a $( k + 1 )$ )-dimensional subspace where $\left. A x \right. _ { 2 } \geqslant$ $\sigma _ { k + 1 } \left\| \pmb { x } \right\| _ { 2 } ,$ which is spanned by the right-singular vectors $v _ { j } , j \leqslant k + 1$ of A. Adding up dimensions of these two spaces yields a number greater than $n ,$ as there must be a nonzero vector in both spaces. This is a contradiction of the rank-nullity theorem (Theorem 2.24) in Section 2.7.3. 

The Eckart-Young theorem implies that we can use SVD to reduce a rank-r matrix A to a rank-k matrix $\hat { \pmb { A } }$ in a principled, optimal (in the spectral norm sense) manner. We can interpret the approximation of A by a rank-k matrix as a form of lossy compression. Therefore, the low-rank approximation of a matrix appears in many machine learning applications, e.g., image processing, noise filtering, and regularization of ill-posed problems. Furthermore, it plays a key role in dimensionality reduction and principal component analysis, as we will see in Chapter 10. 

## Example 4.15 (Finding Structure in Movie Ratings and Consumers (continued))

Coming back to our movie-rating example, we can now apply the concept of low-rank approximations to approximate the original data matrix. Recall that our first singular value captures the notion of science fiction theme in movies and science fiction lovers. Thus, by using only the first singular value term in a rank-1 decomposition of the movie-rating matrix, we obtain the predicted ratings 

$$
\boldsymbol {A} _ {1} = \boldsymbol {u} _ {1} \boldsymbol {v} _ {1} ^ {\top} = \left[ \begin{array}{l} - 0. 6 7 1 0 \\ - 0. 7 1 9 7 \\ - 0. 0 9 3 9 \\ - 0. 1 5 1 5 \end{array} \right] \left[ \begin{array}{l l l} - 0. 7 3 6 7 & - 0. 6 5 1 5 & - 0. 1 8 1 1 \end{array} \right]\tag{4.100a}
$$

Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 

$$
= \left[ \begin{array}{c c c} 0. 4 9 4 3 & 0. 4 3 7 2 & 0. 1 2 1 5 \\ 0. 5 3 0 2 & 0. 4 6 8 9 & 0. 1 3 0 3 \\ 0. 0 6 9 2 & 0. 0 6 1 2 & 0. 0 1 7 0 \\ 0. 1 1 1 6 & 0. 0 9 8 7 & 0. 0 2 7 4 \end{array} \right]\tag{4.100b}
$$

This first rank-1 approximation $\pmb { A } _ { 1 }$ is insightful: it tells us that Ali and Beatrix like science fiction movies, such as Star Wars and Bladerunner (entries have values $> 0 . 4 )$ , but fails to capture the ratings of the other movies by Chandra. This is not surprising, as Chandra’s type of movies is not captured by the first singular value. The second singular value gives us a better rank-1 approximation for those movie-theme lovers: 

$$
\begin{array}{r l} \boldsymbol {A} _ {2} & = \boldsymbol {u} _ {2} \boldsymbol {v} _ {2} ^ {\top} = \left[ \begin{array}{c} 0. 0 2 3 6 \\ 0. 2 0 5 4 \\ - 0. 7 7 0 5 \\ - 0. 6 0 3 0 \end{array} \right] \left[ \begin{array}{c c c} 0. 0 8 5 2 & 0. 1 7 6 2 & - 0. 9 8 0 7 \end{array} \right] \\ & = \left[ \begin{array}{c c c} 0. 0 0 2 0 & 0. 0 0 4 2 & - 0. 0 2 3 1 \\ 0. 0 1 7 5 & 0. 0 3 6 2 & - 0. 2 0 1 4 \\ - 0. 0 6 5 6 & - 0. 1 3 5 8 & 0. 7 5 5 6 \\ - 0. 0 5 1 4 & - 0. 1 0 6 3 & 0. 5 9 1 4 \end{array} \right]. \end{array}\tag{4.101a}
$$

(4.101b) 

In this second rank-1 approximation $A _ { 2 }$ , we capture Chandra’s ratings and movie types well, but not the science fiction movies. This leads us to consider the rank-2 approximation $\widehat { \mathbf { A } } ( 2 )$ , where we combine the first two rank-1 approximations 

$$
\widehat {\boldsymbol {A}} (2) = \sigma_ {1} \boldsymbol {A} _ {1} + \sigma_ {2} \boldsymbol {A} _ {2} = \left[ \begin{array}{c c c} 4. 7 8 0 1 & 4. 2 4 1 9 & 1. 0 2 4 4 \\ 5. 2 2 5 2 & 4. 7 5 2 2 & - 0. 0 2 5 0 \\ 0. 2 4 9 3 & - 0. 2 7 4 3 & 4. 9 7 2 4 \\ 0. 7 4 9 5 & 0. 2 7 5 6 & 4. 0 2 7 8 \end{array} \right].\tag{4.102}
$$

$\widehat { \mathbf { A } } ( 2 )$ is similar to the original movie ratings table 

$$
\boldsymbol {A} = \left[ \begin{array}{c c c} 5 & 4 & 1 \\ 5 & 5 & 0 \\ 0 & 0 & 5 \\ 1 & 0 & 4 \end{array} \right],\tag{4.103}
$$

and this suggests that we can ignore the contribution of $A _ { 3 }$ . We can interpret this so that in the data table there is no evidence of a third movietheme/movie-lovers category. This also means that the entire space of movie-themes/movie-lovers in our example is a two-dimensional space spanned by science fiction and French art house movies and lovers. 