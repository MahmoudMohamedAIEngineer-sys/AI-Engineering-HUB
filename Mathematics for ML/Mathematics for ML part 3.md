For example, we can describe the outcome of a coin-flip experiment (“heads” or “tails”) in two steps. First, we define a parameter $\mu ,$ which describes the probability of “heads” as the parameter of a Bernoulli distribution (Chapter 6); second, we can sample an outcome $x \in \{ { \mathrm { h e a d } } ,$ , tail from the Bernoulli distribution $p ( x | \mu ) = \operatorname { B e r } ( \mu )$ . The parameter $\mu$ gives rise to a specific dataset $\mathcal { X }$ and depends on the coin used. Since $\mu$ is unknown in advance and can never be observed directly, we need mechanisms to learn something about $\mu$ given observed outcomes of coin-flip experiments. In the following, we will discuss how probabilistic modeling can be used for this purpose. 

## 8.4.1 Probabilistic Models

Probabilistic models represent the uncertain aspects of an experiment as probability distributions. The benefit of using probabilistic models is that they offer a unified and consistent set of tools from probability theory (Chapter 6) for modeling, inference, prediction, and model selection. 

In probabilistic modeling, the joint distribution $p ( { \pmb x } , { \pmb \theta } )$ of the observed variables x and the hidden parameters θ is of central importance: It encapsulates information from the following: 

The prior and the likelihood (product rule, Section 6.3). 

The marginal likelihood p(x), which will play an important role in model selection (Section 8.6), can be computed by taking the joint distribution and integrating out the parameters (sum rule, Section 6.3). 

The posterior, which can be obtained by dividing the joint by the marginal likelihood. 

Only the joint distribution has this property. Therefore, a probabilistic model is specified by the joint distribution of all its random variables. 

## 8.4.2 Bayesian Inference

A key task in machine learning is to take a model and the data to uncover the values of the model’s hidden variables θ given the observed variables x. In Section 8.3.1, we already discussed two ways for estimating model parameters θ using maximum likelihood or maximum a posteriori estimation. In both cases, we obtain a single-best value for $\pmb \theta$ so that the key algorithmic problem of parameter estimation is solving an optimization problem. Once these point estimates $\pmb { \theta } ^ { * }$ are known, we use them to make predictions. More specifically, the predictive distribution will be $p ( { \pmb x } | \pmb \theta ^ { * } )$ , where we use $\pmb { \theta } ^ { * }$ in the likelihood function. 

As discussed in Section 6.3, focusing solely on some statistic of the posterior distribution (such as the parameter $\pmb { \theta } ^ { * }$ that maximizes the posterior) leads to loss of information, which can be critical in a system that uses the prediction $p ( { \pmb x } | \pmb \theta ^ { * } )$ to make decisions. These decision-making systems typically have different objective functions than the likelihood, a squared-error loss or a mis-classification error. Therefore, having the full posterior distribution around can be extremely useful and leads to more robust decisions. Bayesian inference is about finding this posterior distribution (Gelman et al., 2004). For a dataset $\mathcal { X }$ , a parameter prior $p ( \pmb \theta )$ , and a likelihood function, the posterior 

$$
p (\pmb {\theta} \mid \mathcal {X}) = \frac {p (\mathcal {X} \mid \pmb {\theta}) p (\pmb {\theta})}{p (\mathcal {X})}, \qquad p (\mathcal {X}) = \int p (\mathcal {X} \mid \pmb {\theta}) p (\pmb {\theta}) \mathrm{d} \pmb {\theta},\tag{8.22}
$$

is obtained by applying Bayes’ theorem. The key idea is to exploit Bayes’ theorem to invert the relationship between the parameters $\pmb \theta$ and the data $\mathcal { X }$ (given by the likelihood) to obtain the posterior distribution $p ( \pmb \theta | \mathcal { X } )$ 

The implication of having a posterior distribution on the parameters is that it can be used to propagate uncertainty from the parameters to the data. More specifically, with a distribution $p ( \pmb \theta )$ on the parameters our predictions will be 

$$
p (\pmb {x}) = \int p (\pmb {x} | \pmb {\theta}) p (\pmb {\theta}) \mathrm{d} \pmb {\theta} = \mathbb {E} _ {\pmb {\theta}} [ p (\pmb {x} | \pmb {\theta}) ],\tag{8.23}
$$

and they no longer depend on the model parameters $\theta ,$ which have been marginalized/integrated out. Equation (8.23) reveals that the prediction is an average over all plausible parameter values $\theta _ { i }$ , where the plausibility is encapsulated by the parameter distribution $p ( \pmb \theta )$ 

Having discussed parameter estimation in Section 8.3 and Bayesian inference here, let us compare these two approaches to learning. Parameter estimation via maximum likelihood or MAP estimation yields a consistent point estimate $\pmb { \theta } ^ { * }$ of the parameters, and the key computational problem to be solved is optimization. In contrast, Bayesian inference yields a (posterior) distribution, and the key computational problem to be solved is integration. Predictions with point estimates are straightforward, whereas predictions in the Bayesian framework require solving another integration problem; see (8.23). However, Bayesian inference gives us a principled way to incorporate prior knowledge, account for side information, and incorporate structural knowledge, all of which is not easily done in the context of parameter estimation. Moreover, the propagation of parameter uncertainty to the prediction can be valuable in decision-making systems for risk assessment and exploration in the context of data-efficient learning (Deisenroth et al., 2015; Kamthe and Deisenroth, 2018). 

While Bayesian inference is a mathematically principled framework for learning about parameters and making predictions, there are some practical challenges that come with it because of the integration problems we need to solve; see (8.22) and (8.23). More specifically, if we do not choose a conjugate prior on the parameters (Section 6.6.1), the integrals in (8.22) and (8.23) are not analytically tractable, and we cannot compute the posterior, the predictions, or the marginal likelihood in closed form. In these cases, we need to resort to approximations. Here, we can use stochastic approximations, such as Markov chain Monte Carlo (MCMC) (Gilks et al., 1996), or deterministic approximations, such as the Laplace approximation (Bishop, 2006; Barber, 2012; Murphy, 2012), variational inference (Jordan et al., 1999; Blei et al., 2017), or expectation propagation (Minka, 2001a). 

Despite these challenges, Bayesian inference has been successfully applied to a variety of problems, including large-scale topic modeling (Hoffman et al., 2013), click-through-rate prediction (Graepel et al., 2010), data-efficient reinforcement learning in control systems (Deisenroth et al., 2015), online ranking systems (Herbrich et al., 2007), and large-scale recommender systems. There are generic tools, such as Bayesian optimization (Brochu et al., 2009; Snoek et al., 2012; Shahriari et al., 2016), that are very useful ingredients for an efficient search of meta parameters of models or algorithms. 

Remark. In the machine learning literature, there can be a somewhat ar bitrary separation between (random) “variables” and “parameters”. While parameters are estimated (e.g., via maximum likelihood), variables are usually marginalized out. In this book, we are not so strict with this separation because, in principle, we can place a prior on any parameter and integrate it out, which would then turn the parameter into a random variable according to the aforementioned separation. 

## 8.4.3 Latent-Variable Models

In practice, it is sometimes useful to have additional latent variables z (besides the model parameters θ) as part of the model (Moustaki et al., 2015). These latent variables are different from the model parameters θ as they do not parametrize the model explicitly. Latent variables may describe the data-generating process, thereby contributing to the interpretability of the model. They also often simplify the structure of the model and allow us to define simpler and richer model structures. Simplification of the model structure often goes hand in hand with a smaller number of model parameters (Paquet, 2008; Murphy, 2012). Learning in latent-variable models (at least via maximum likelihood) can be done in a principled way using the expectation maximization (EM) algorithm (Demp ster et al., 1977; Bishop, 2006). Examples, where such latent variables are helpful, are principal component analysis for dimensionality reduction (Chapter 10), Gaussian mixture models for density estimation (Chapter 11), hidden Markov models (Maybeck, 1979) or dynamical systems (Ghahramani and Roweis, 1999; Ljung, 1999) for time-series modeling, and meta learning and task generalization (Hausman et al., 2018; Sæ- mundsson et al., 2018). Although the introduction of these latent variables may make the model structure and the generative process easier, learning in latent-variable models is generally hard, as we will see in Chapter 11. 

Since latent-variable models also allow us to define the process that generates data from parameters, let us have a look at this generative process. Denoting data by $^ { \mathbf { \delta x } , }$ the model parameters by θ and the latent variables by z, we obtain the conditional distribution 

$$
p (\boldsymbol {x} \mid \boldsymbol {z}, \boldsymbol {\theta})\tag{8.24}
$$

that allows us to generate data for any model parameters and latent variables. Given that z are latent variables, we place a prior $p ( z )$ on them. 

As the models we discussed previously, models with latent variables can be used for parameter learning and inference within the frameworks we discussed in Sections 8.3 and 8.4.2. To facilitate learning $( \mathsf { e . g . }$ , by means of maximum likelihood estimation or Bayesian inference), we follow a two-step procedure. First, we compute the likelihood $p ( { \pmb x } | { \pmb \theta } )$ of the model, which does not depend on the latent variables. Second, we use this likelihood for parameter estimation or Bayesian inference, where we use exactly the same expressions as in Sections 8.3 and 8.4.2, respectively. 

Since the likelihood function $p ( { \pmb x } | { \pmb \theta } )$ is the predictive distribution of the data given the model parameters, we need to marginalize out the latent variables so that 

$$
p (\boldsymbol {x} \mid \boldsymbol {\theta}) = \int p (\boldsymbol {x} \mid \boldsymbol {z}, \boldsymbol {\theta}) p (\boldsymbol {z}) \mathrm{d} \boldsymbol {z},\tag{8.25}
$$

The likelihood is a function of the data and the model parameters, but is independent of the latent variables. 

where $p ( { \pmb x } | { \pmb z } , { \pmb \theta } )$ is given in (8.24) and $p ( z )$ is the prior on the latent variables. Note that the likelihood must not depend on the latent variables $z ,$ but it is only a function of the data x and the model parameters θ. 

The likelihood in (8.25) directly allows for parameter estimation via maximum likelihood. MAP estimation is also straightforward with an additional prior on the model parameters $\pmb \theta$ as discussed in Section 8.3.2. Moreover, with the likelihood (8.25) Bayesian inference (Section 8.4.2) in a latent-variable model works in the usual way: We place a prior $p ( \pmb \theta )$ on the model parameters and use Bayes’ theorem to obtain a posterior distribution 

$$
p (\pmb {\theta} \mid \mathcal {X}) = \frac {p (\mathcal {X} \mid \pmb {\theta}) p (\pmb {\theta})}{p (\mathcal {X})}\tag{8.26}
$$

over the model parameters given a dataset $\mathcal { X } .$ . The posterior in (8.26) can be used for predictions within a Bayesian inference framework; see (8.23). 

One challenge we have in this latent-variable model is that the likelihood $p ( \mathcal { X } | \pmb \theta )$ requires the marginalization of the latent variables according to (8.25). Except when we choose a conjugate prior $p ( z )$ for $p ( { \pmb x } | { \pmb z } , { \pmb \theta } )$ , the marginalization in (8.25) is not analytically tractable, and we need to resort to approximations (Bishop, 2006; Paquet, 2008; Murphy, 2012; Moustaki et al., 2015). 

Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 

Similar to the parameter posterior (8.26) we can compute a posterior on the latent variables according to 

$$
p (\boldsymbol {z} \mid \mathcal {X}) = \frac {p (\mathcal {X} \mid \boldsymbol {z}) p (\boldsymbol {z})}{p (\mathcal {X})}, \qquad p (\mathcal {X} \mid \boldsymbol {z}) = \int p (\mathcal {X} \mid \boldsymbol {z}, \pmb {\theta}) p (\pmb {\theta}) \mathrm{d} \pmb {\theta},\tag{8.27}
$$

where $p ( z )$ is the prior on the latent variables and $p ( \mathcal { X } | z )$ requires us to integrate out the model parameters $\pmb { \theta } .$ 

Given the difficulty of solving integrals analytically, it is clear that marginalizing out both the latent variables and the model parameters at the same time is not possible in general (Bishop, 2006; Murphy, 2012). A quantity that is easier to compute is the posterior distribution on the latent variables, but conditioned on the model parameters, i.e., 

$$
p (\boldsymbol {z} | \mathcal {X}, \boldsymbol {\theta}) = \frac {p (\mathcal {X} | \boldsymbol {z} , \boldsymbol {\theta}) p (\boldsymbol {z})}{p (\mathcal {X} | \boldsymbol {\theta})},\tag{8.28}
$$

where $p ( z )$ is the prior on the latent variables and $p ( \mathcal { X } | z , \theta )$ is given in (8.24). 

In Chapters 10 and 11, we derive the likelihood functions for PCA and Gaussian mixture models, respectively. Moreover, we compute the posterior distributions (8.28) on the latent variables for both PCA and Gaussian mixture models. 

Remark. In the following chapters, we may not be drawing such a clear distinction between latent variables z and uncertain model parameters $\pmb \theta$ and call the model parameters “latent” or “hidden” as well because they are unobserved. In Chapters 10 and 11, where we use the latent variables $z ,$ we will pay attention to the difference as we will have two different types of hidden variables: model parameters θ and latent variables z. $\diamondsuit$ 

We can exploit the fact that all the elements of a probabilistic model are random variables to define a unified language for representing them. In Section 8.5, we will see a concise graphical language for representing the structure of probabilistic models. We will use this graphical language to describe the probabilistic models in the subsequent chapters. 

## 8.4.4 Further Reading

Probabilistic models in machine learning (Bishop, 2006; Barber, 2012; Murphy, 2012) provide a way for users to capture uncertainty about data and predictive models in a principled fashion. Ghahramani (2015) presents a short review of probabilistic models in machine learning. Given a probabilistic model, we may be lucky enough to be able to compute parameters of interest analytically. However, in general, analytic solutions are rare, and computational methods such as sampling (Gilks et al., 1996; Brooks et al., 2011) and variational inference (Jordan et al., 1999; Blei et al., 

2017) are used. Moustaki et al. (2015) and Paquet (2008) provide a good overview of Bayesian inference in latent-variable models. 

In recent years, several programming languages have been proposed that aim to treat the variables defined in software as random variables corresponding to probability distributions. The objective is to be able to write complex functions of probability distributions, while under the hood the compiler automatically takes care of the rules of Bayesian inference. This rapidly changing field is called probabilistic programming. 

## 8.5 Directed Graphical Models

In this section, we introduce a graphical language for specifying a probabilistic model, called the directed graphical model. It provides a compact and succinct way to specify probabilistic models, and allows the reader to visually parse dependencies between random variables. A graphical model visually captures the way in which the joint distribution over all random variables can be decomposed into a product of factors depending only on a subset of these variables. In Section 8.4, we identified the joint distribution of a probabilistic model as the key quantity of interest because it comprises information about the prior, the likelihood, and the posterior. However, the joint distribution by itself can be quite complicated, and it does not tell us anything about structural properties of the probabilistic model. For example, the joint distribution $\textstyle p ( a , b , c )$ does not tell us anything about independence relations. This is the point where graphical models come into play. This section relies on the concepts of independence and conditional independence, as described in Section 6.4.5. 

In a graphical model, nodes are random variables. In Figure 8.9(a), the nodes represent the random variables $a , b , c .$ Edges represent probabilistic relations between variables, e.g., conditional probabilities. 

Remark. Not every distribution can be represented in a particular choice of graphical model. A discussion of this can be found in Bishop (2006). $\diamondsuit$ 

Probabilistic graphical models have some convenient properties: 

They are a simple way to visualize the structure of a probabilistic model. 

They can be used to design or motivate new kinds of statistical models. 

Inspection of the graph alone gives us insight into properties, e.g., conditional independence. 

Complex computations for inference and learning in statistical models can be expressed in terms of graphical manipulations. 

## 8.5.1 Graph Semantics

Directed graphical models/Bayesian networks are a method for representing conditional dependencies in a probabilistic model. They provide a visual description of the conditional probabilities, hence, providing a simple lan guage for describing complex interdependence. The modular description also entails computational simplification. Directed links (arrows) between two nodes (random variables) indicate conditional probabilities. For example, the arrow between a and b in Figure 8.9(a) gives the conditional probability $p ( b | a )$ of b given a. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/8eb494363aafcf11438208e27ff9fcd5141e8e72ea9bc18385d01180954498bf.jpg)



(a) Fully connected.



With additional assumptions, the arrows can be used to indicate causal relationships (Pearl, 2009).


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/67ab54c1351ff03fb8d0ea8c60eadcfe094d0e09e97ba8aaac56a2f6f9920e27.jpg)



(b) Not fully connected.


Directed graphical models can be derived from joint distributions if we know something about their factorization. 

## Example 8.7

Consider the joint distribution 

$$
p (a, b, c) = p (c \mid a, b) p (b \mid a) p (a)\tag{8.29}
$$

of three random variables a, b, c. The factorization of the joint distribution in (8.29) tells us something about the relationship between the random variables: 

c depends directly on a and b. 

b depends directly on a. 

a depends neither on b nor on c. 

For the factorization in (8.29), we obtain the directed graphical model in Figure 8.9(a). 

In general, we can construct the corresponding directed graphical mode from a factorized joint distribution as follows: 

1. Create a node for all random variables. 

2. For each conditional distribution, we add a directed link (arrow) to the graph from the nodes corresponding to the variables on which the distribution is conditioned. 

The graph layout depends on the choice of factorization of the joint distribution. 

We discussed how to get from a known factorization of the joint distribution to the corresponding directed graphical model. Now, we will do 

Figure 8.9 Examples of directed graphical models. 

exactly the opposite and describe how to extract the joint distribution of a set of random variables from a given graphical model. 

## Example 8.8

Looking at the graphical model in Figure 8.9(b), we exploit two properties: 

The joint distribution $p ( x _ { 1 } , \ldots , x _ { 5 } )$ we seek is the product of a set of conditionals, one for each node in the graph. In this particular example, we will need five conditionals. 

Each conditional depends only on the parents of the corresponding node in the graph. For example, $x _ { 4 }$ will be conditioned on $x _ { 2 }$ 

These two properties yield the desired factorization of the joint distribution 

$$
p (x _ {1}, x _ {2}, x _ {3}, x _ {4}, x _ {5}) = p (x _ {1}) p (x _ {5}) p (x _ {2} \mid x _ {5}) p (x _ {3} \mid x _ {1}, x _ {2}) p (x _ {4} \mid x _ {2})\tag{8.30}
$$

In general, the joint distribution $p ( \pmb { x } ) = p ( x _ { 1 } , . . . , x _ { K } )$ is given as 

$$
p (\boldsymbol {x}) = \prod_ {k = 1} ^ {K} p (x _ {k} \mid \mathrm{Pa} _ {k}),\tag{8.31}
$$

where $\mathrm { P a } _ { k }$ means “the parent nodes of $x _ { k } ^ { \dag }$ . Parent nodes of $x _ { k }$ are nodes that have arrows pointing to $x _ { k }$ . 

We conclude this subsection with a concrete example of the coin-flip experiment. Consider a Bernoulli experiment (Example 6.8) where the probability that the outcome x of this experiment is “heads” is 

$$
p (x \mid \mu) = \operatorname{Ber} (\mu).\tag{8.32}
$$

We now repeat this experiment $N$ times and observe outcomes $x _ { 1 } , \ldots , x _ { N }$ so that we obtain the joint distribution 

$$
p \left(x _ {1}, \dots , x _ {N} \mid \mu\right) = \prod_ {n = 1} ^ {N} p \left(x _ {n} \mid \mu\right).\tag{8.33}
$$

The expression on the right-hand side is a product of Bernoulli distributions on each individual outcome because the experiments are independent. Recall from Section 6.4.5 that statistical independence means that the distribution factorizes. To write the graphical model down for this setting, we make the distinction between unobserved/latent variables and observed variables. Graphically, observed variables are denoted by shaded nodes so that we obtain the graphical model in Figure 8.10(a). We see that the single parameter $\mu$ is the same for all $x _ { n } , n = 1 , \ldots , N$ as the outcomes $x _ { n }$ are identically distributed. $\mathsf { A }$ more compact, but equivalent, graphical model for this setting is given in Figure 8.10(b), where we use the plate notation. The plate (box) repeats everything inside (in this case, the observations $x _ { n } )$ N times. Therefore, both graphical models are equivalent, but the plate notation is more compact. Graphical models immediately allow us to place a hyperprior on µ. A hyperprior is a second layer of prior distributions on the parameters of the first layer of priors. Figure 8.10(c) places a $\mathtt { B e t a } ( \alpha , \beta )$ prior on the latent variable µ. If we treat α and $\beta$ as deterministic parameters, i.e., not random variables, we omit the circle around it. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/5e338c6e4d44aade46139a68b9798ce5746f161d9f1e488f9f5eb35ee51ca5cc.jpg)



Figure 8.10 Graphical models for a repeated Bernoulli experiment.


## 8.5.2 Conditional Independence and d-Separation

Directed graphical models allow us to find conditional independence (Section 6.4.5) relationship properties of the joint distribution only by looking at the graph. A concept called d-separation (Pearl, 1988) is key to this. 

Consider a general directed graph in which , , are arbitrary nonin tersecting sets of nodes (whose union may be smaller than the complete set of nodes in the graph). We wish to ascertain whether a particular conditional independence statement, “ is conditionally independent of given ${ \mathcal { C } } ^ { \mathfrak { n } }$ , denoted by 

$$
\mathcal {A} \perp \perp \mathcal {B} | \mathcal {C},\tag{8.34}
$$

is implied by a given directed acyclic graph. To do so, we consider all possible trails (paths that ignore the direction of the arrows) from any node in to any nodes in . Any such path is said to be blocked if it includes any node such that either of the following are true: 

The arrows on the path meet either head to tail or tail to tail at the node, and the node is in the set . 

The arrows meet head to head at the node, and neither the node nor any of its descendants is in the set . 

If all paths are blocked, then is said to be d-separated from by ${ \mathcal { C } } ,$ and the joint distribution over all of the variables in the graph will satisfy AⅡB|C. 



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



Figure 8.12 Three types of graphical models: (a) Directed graphical models (Bayesian networks); (b) Undirected graphical models (Markov random fields); (c) Factor graphs. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/0d2418dd669f513653499548fc1bd50b7fb6cc1873227a718722beb9e7bef6e3.jpg)



(a) Directed graphical model


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/c0fbe07e50a21a4809ab7b3ed673e4fc206a3aded747be43ca1b885e93e2c221.jpg)



(b) Undirected model



graphical


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/cf351798f458ebe76c2a2994fa565bceeb71a12bd4d8d8c0ecc5279d436859a1.jpg)



(c) Factor graph


## Example 8.9 (Conditional Independence)

Figure 8.11 D-separation example. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/7c836aa0d1a5960d911b778f959a5e54a71546e31b809d0eb5fcb00c7274b7f6.jpg)


Consider the graphical model in Figure 8.11. Visual inspection gives us 

$$
b \perp d \mid a, c\tag{8.35}
$$

$$
a \perp c | b\tag{8.36}
$$

$$
b \not \perp d | c\tag{8.37}
$$

$$
a \not \perp c | b, e\tag{8.38}
$$

Directed graphical models allow a compact representation of probabilistic models, and we will see examples of directed graphical models in Chapters 9, 10, and 11. The representation, along with the concept of conditional independence, allows us to factorize the respective probabilistic models into expressions that are easier to optimize. 

The graphical representation of the probabilistic model allows us to visually see the impact of design choices we have made on the structure of the model. We often need to make high-level assumptions about the structure of the model. These modeling assumptions (hyperparameters) affect the prediction performance, but cannot be selected directly using the approaches we have seen so far. We will discuss different ways to choose the structure in Section 8.6. 

Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 

## 8.5.3 Further Reading

An introduction to probabilistic graphical models can be found in Bishop (2006, chapter 8), and an extensive description of the different applications and corresponding algorithmic implications can be found in the book by Koller and Friedman (2009). There are three main types of probabilistic graphical models: 

Directed graphical models (Bayesian networks); see Figure 8.12(a) 

Undirected graphical models (Markov random fields); see Figure 8.12(b) 

Factor graphs; see Figure 8.12(c) 

Graphical models allow for graph-based algorithms for inference and learning, e.g., via local message passing. Applications range from rank ing in online games (Herbrich et al., 2007) and computer vision (e.g., image segmentation, semantic labeling, image denoising, image restoration (Kittler and Foglein¨ , 1984; Sucar and Gillies, 1994; Shotton et al., 2006; Szeliski et al., 2008)) to coding theory (McEliece et al., 1998), solving linear equation systems (Shental et al., 2008), and iterative Bayesian state estimation in signal processing (Bickson et al., 2007; Deisenroth and Mohamed, 2012). 

One topic that is particularly important in real applications that we do not discuss in this book is the idea of structured prediction (Bakir et al., 2007; Nowozin et al., 2014), which allows machine learning models to tackle predictions that are structured, for example sequences, trees, and graphs. The popularity of neural network models has allowed more flexible probabilistic models to be used, resulting in many useful applications of structured models (Goodfellow et al., 2016, chapter 16). In recent years, there has been a renewed interest in graphical models due to their applications to causal inference (Pearl, 2009; Imbens and Rubin, 2015; Peters et al., 2017; Rosenbaum, 2017). 

## 8.6 Model Selection

In machine learning, we often need to make high-level modeling decisions that critically influence the performance of the model. The choices we make (e.g., the functional form of the likelihood) influence the number and type of free parameters in the model and thereby also the flexibility and expressivity of the model. More complex models are more flexible in the sense that they can be used to describe more datasets. For instance, a polynomial of degree 1 (a line $y = a _ { 0 } + a _ { 1 } x )$ can only be used to describe linear relations between inputs x and observations y. A polynomial of degree 2 can additionally describe quadratic relationships between inputs and observations. 

One would now think that very flexible models are generally preferable to simple models because they are more expressive. A general problem 

A polynomial y = a +a x+a x<sup>2</sup> can also describe linear functions by setting a = 0, i.e., it is strictly more expressive than a first-order polynomial. 

Figure 8.13 Nested cross-validation. We perform two levels of K-fold cross-validation. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/7009e1fe9e4c44e5c52ffe3f4567a82b8036b19d3b87e91287d0ca5962e5d8dd.jpg)


is that at training time we can only use the training set to evaluate the performance of the model and learn its parameters. However, the performance on the training set is not really what we are interested in. In Section 8.3, we have seen that maximum likelihood estimation can lead to overfitting, especially when the training dataset is small. Ideally, our model (also) works well on the test set (which is not available at training time). Therefore, we need some mechanisms for assessing how a model generalizes to unseen test data. Model selection is concerned with exactly this problem. 

## 8.6.1 Nested Cross-Validation

We have already seen an approach (cross-validation in Section 8.2.4) that can be used for model selection. Recall that cross-validation provides an estimate of the generalization error by repeatedly splitting the dataset into training and validation sets. We can apply this idea one more time, i.e., for each split, we can perform another round of cross-validation. This is sometimes referred to as nested cross-validation; see Figure 8.13. The inner level is used to estimate the performance of a particular choice of model or hyperparameter on a internal validation set. The outer level is used to estimate generalization performance for the best choice of model chosen by the inner loop. We can test different model and hyperparameter choices in the inner loop. To distinguish the two levels, the set used to estimate the generalization performance is often called the test set and the set used for choosing the best model is called the validation set. The inner loop estimates the expected value of the generalization error for a given model (8.39), by approximating it using the empirical error on the validation set, i.e., 

$$
\mathbb {E} _ {\mathcal {V}} [ \mathbf {R} (\mathcal {V} \mid M) ] \approx \frac {1}{K} \sum_ {k = 1} ^ {K} \mathbf {R} (\mathcal {V} ^ {(k)} \mid M),\tag{8.39}
$$

where ${ \bf R } ( \nu | M )$ is the empirical risk (e.g., root mean square error) on the validation set for model M. We repeat this procedure for all models and choose the model that performs best. Note that cross-validation not only gives us the expected generalization error, but we can also obtain highorder statistics, e.g., the standard error, an estimate of how uncertain the mean estimate is. Once the model is chosen, we can evaluate the final performance on the test set. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/75b6343f849a38d9fd7248004144789ecef9fb6b7fe8f495fae5d10a4d14f81d.jpg)



Figure 8.14 Bayesian inference embodies Occam’s razor. The horizontal axis describes the space of all possible datasets D. The evidence (vertical axis) evaluates how well a model predicts available data. Since $p ( \mathcal { D } \mid M _ { i } )$ needs to integrate to 1, we should choose the model with the greatest evidence. Adapted from MacKay (2003).


## 8.6.2 Bayesian Model Selection

There are many approaches to model selection, some of which are covered in this section. Generally, they all attempt to trade off model complexity and data fit. We assume that simpler models are less prone to overfitting than complex models, and hence the objective of model selection is to find the simplest model that explains the data reasonably well. This concept is also known as Occam’s razor. 

Remark. If we treat model selection as a hypothesis testing problem, we are looking for the simplest hypothesis that is consistent with the data (Murphy, 2012). $\diamondsuit$ 

One may consider placing a prior on models that favors simpler models. However, it is not necessary to do this: An “automatic Occam’s Razor” is quantitatively embodied in the application of Bayesian probability (Smith and Spiegelhalter, 1980; Jefferys and Berger, 1992; MacKay, 1992). Figure 8.14, adapted from MacKay (2003), gives us the basic intuition why complex and very expressive models may turn out to be a less probable choice for modeling a given dataset . Let us think of the horizontal axis representing the space of all possible datasets . If we are interested in the posterior probability $p ( M _ { i } \mid \mathcal { D } )$ of model $M _ { i }$ given the data , we can employ Bayes’ theorem. Assuming a uniform prior $p ( M )$ over all models, Bayes’ theorem rewards models in proportion to how much they predicted the data that occurred. This prediction of the data given model $M _ { i } , p ( \mathcal { D } | M _ { i } )$ , is called the evidence for $M _ { i } . \mathrm { A }$ simple model $M _ { 1 }$ can only predict a small number of datasets, which is shown by $p ( \mathcal { D } \vert M _ { 1 } ) ;$ ; a more powerful model $M _ { 2 }$ that has, e.g., more free parameters than $M _ { 1 }$ , is able 

These predictions are quantified by a normalized probability distribution on D, i.e., it needs to integrate/sum to 1. evidence 

Bayesian model selection generative process Figure 8.15 Illustration of the hierarchical generative process in Bayesian model selection. We place a prior p(M) on the set of models. For each model, there is a distribution $p ( \pmb \theta \vert M )$ on the corresponding model parameters, which is used to generate the data D. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/0169358098b0f6a10894aa9585662d91f61f1dfe2a499bf77199927509bc0559.jpg)



model evidence marginal likelihood


to predict a greater variety of datasets. This means, however, that $M _ { 2 }$ does not predict the datasets in region C as well as $M _ { 1 }$ . Suppose that equal prior probabilities have been assigned to the two models. Then, if the dataset falls into region $C _ { i }$ , the less powerful model $M _ { 1 }$ is the more probable model. 

Earlier in this chapter, we argued that models need to be able to explain the data, i.e., there should be a way to generate data from a given model. Furthermore, if the model has been appropriately learned from the data, then we expect that the generated data should be similar to the empirical data. For this, it is helpful to phrase model selection as a hierarchical inference problem, which allows us to compute the posterior distribution over models. 

Let us consider a finite number of models $M = \left\{ M _ { 1 } , \dots , M _ { K } \right\}$ , where each model $M _ { k }$ possesses parameters $\pmb { \theta } _ { k }$ . In Bayesian model selection, we place a prior $p ( M )$ on the set of models. The corresponding generative process that allows us to generate data from this model is 

$$
M _ {k} \sim p (M)\tag{8.40}
$$

$$
\boldsymbol {\theta} _ {k} \sim p (\boldsymbol {\theta} \mid M _ {k})\tag{8.41}
$$

$$
\mathcal {D} \sim p (\mathcal {D} | \pmb {\theta} _ {k})\tag{8.42}
$$

and illustrated in Figure 8.15. Given a training set ${ \mathcal { D } } ,$ we apply Bayes theorem and compute the posterior distribution over models as 

$$
p (M _ {k} \mid \mathcal {D}) \propto p (M _ {k}) p (\mathcal {D} \mid M _ {k}).\tag{8.43}
$$

Note that this posterior no longer depends on the model parameters $\pmb { \theta } _ { k }$ because they have been integrated out in the Bayesian setting since 

$$
p (\mathcal {D} \mid M _ {k}) = \int p (\mathcal {D} \mid \boldsymbol {\theta} _ {k}) p (\boldsymbol {\theta} _ {k} \mid M _ {k}) d \boldsymbol {\theta} _ {k},\tag{8.44}
$$

where $p ( \pmb \theta _ { k } | M _ { k } )$ is the prior distribution of the model parameters $\pmb { \theta } _ { k }$ of model $M _ { k }$ . The term (8.44) is referred to as the model evidence or marginal likelihood. From the posterior in (8.43), we determine the MAP estimate 

$$
M ^ {*} = \arg \max _ {M _ {k}} p (M _ {k} \mid \mathcal {D}).\tag{8.45}
$$

With a uniform prior $\begin{array} { r } { p ( M _ { k } ) = \frac { 1 } { K } } \end{array}$ , which gives every model equal (prior) probability, determining the MAP estimate over models amounts to picking the model that maximizes the model evidence (8.44). 

Remark (Likelihood and Marginal Likelihood). There are some important differences between a likelihood and a marginal likelihood (evidence): While the likelihood is prone to overfitting, the marginal likelihood is typically not as the model parameters have been marginalized out $( \mathrm { i . e . }$ , we no longer have to fit the parameters). Furthermore, the marginal likelihood automatically embodies a trade-off between model complexity and data fit (Occam’s razor). ◇ 

## 8.6.3 Bayes Factors for Model Comparison

Consider the problem of comparing two probabilistic models $M _ { 1 } , M _ { 2 }$ given a dataset . If we compute the posteriors $p ( M _ { 1 } \mid \mathcal { D } )$ and $p ( M _ { 2 } | \mathcal { D } )$ we can compute the ratio of the posteriors 

$$
\underbrace {\frac {p (M _ {1} \mid \mathcal {D})}{p (M _ {2} \mid \mathcal {D})}} _ {\text {posterior odds}} = \frac {\frac {p (\mathcal {D} \mid M _ {1}) p (M _ {1})}{p (\mathcal {D})}}{\frac {p (\mathcal {D} \mid M _ {2}) p (M _ {2})}{p (\mathcal {D})}} = \underbrace {\frac {p (M _ {1})}{p (M _ {2})}} _ {\text {prior odds}} \underbrace {\frac {p (\mathcal {D} \mid M _ {1})}{p (\mathcal {D} \mid M _ {2})}} _ {\text {Bayes factor}}.\tag{8.46}
$$

The ratio of the posteriors is also called the posterior odds. The first fraction on the right-hand side of (8.46), the prior odds, measures how much our prior (initial) beliefs favor $M _ { 1 }$ over $M _ { 2 }$ . The ratio of the marginal likelihoods (second fraction on the right-hand-side) is called the Bayes factor and measures how well the data is predicted by $M _ { 1 }$ compared to $M _ { 2 }$ 

Remark. The Jeffreys-Lindley paradox states that the “Bayes factor always favors the simpler model since the probability of the data under a complex model with a diffuse prior will be very small” (Murphy, 2012). Here, a diffuse prior refers to a prior that does not favor specific models, i.e., many models are a priori plausible under this prior. $\diamondsuit$ 

If we choose a uniform prior over models, the prior odds term in (8.46) is 1, i.e., the posterior odds is the ratio of the marginal likelihoods (Bayes factor) 

$$
\frac {p (\mathcal {D} \mid M _ {1})}{p (\mathcal {D} \mid M _ {2})}.\tag{8.47}
$$

If the Bayes factor is greater than 1, we choose model $M _ { 1 } ,$ , otherwise model $M _ { 2 }$ . In a similar way to frequentist statistics, there are guidelines on the size of the ratio that one should consider before ”significance” of the result (Jeffreys, 1961). 

Remark (Computing the Marginal Likelihood). The marginal likelihood plays an important role in model selection: We need to compute Bayes factors (8.46) and posterior distributions over models (8.43). 

Unfortunately, computing the marginal likelihood requires us to solve an integral (8.44). This integration is generally analytically intractable, and we will have to resort to approximation techniques, e.g., numerical integration (Stoer and Burlirsch, 2002), stochastic approximations using Monte Carlo (Murphy, 2012), or Bayesian Monte Carlo techniques (O’Hagan, 1991; Rasmussen and Ghahramani, 2003). 

However, there are special cases in which we can solve it. In Section 6.6.1, we discussed conjugate models. If we choose a conjugate parameter prior $p ( \pmb \theta )$ , we can compute the marginal likelihood in closed form. In Chapter 9, we will do exactly this in the context of linear regression. 

We have seen a brief introduction to the basic concepts of machine learning in this chapter. For the rest of this part of the book we will see how the three different flavors of learning in Sections 8.2, 8.3, and 8.4 are applied to the four pillars of machine learning (regression, dimensionality reduction, density estimation, and classification). 

## 8.6.4 Further Reading

We mentioned at the start of the section that there are high-level modeling choices that influence the performance of the model. Examples include the following: 

The degree of a polynomial in a regression setting 

The number of components in a mixture model 

The network architecture of a (deep) neural network 

The type of kernel in a support vector machine 

The dimensionality of the latent space in PCA 

The learning rate (schedule) in an optimization algorithm 

Rasmussen and Ghahramani (2001) showed that the automatic Occam’s razor does not necessarily penalize the number of parameters in a model, but it is active in terms of the complexity of functions. They also showed that the automatic Occam’s razor also holds for Bayesian nonparametric models with many parameters, e.g., Gaussian processes. 

If we focus on the maximum likelihood estimate, there exist a number of heuristics for model selection that discourage overfitting. They are called information criteria, and we choose the model with the largest value. The Akaike information criterion (AIC) (Akaike, 1974) 

$$
\log p (\boldsymbol {x} \mid \boldsymbol {\theta}) - M\tag{8.48}
$$

corrects for the bias of the maximum likelihood estimator by addition of a penalty term to compensate for the overfitting of more complex models with lots of parameters. Here, M is the number of model parameters. The AIC estimates the relative information lost by a given model. 

The Bayesian information criterion (BIC) (Schwarz, 1978) 

$$
\log p (\pmb {x}) = \log \int p (\pmb {x} | \pmb {\theta}) p (\pmb {\theta}) \mathrm{d} \pmb {\theta} \approx \log p (\pmb {x} | \pmb {\theta}) - \frac {1}{2} M \log N\tag{8.49}
$$

can be used for exponential family distributions. Here, N is the number of data points and M is the number of parameters. BIC penalizes model complexity more heavily than AIC. 

# Linear Regression

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/d56499ad72dda63cd3693b0598490433fb7f6677aa67614ed8c50f0bd252db93.jpg)


In the following, we will apply the mathematical concepts from Chapters 2, $5 , 6 ,$ and $7$ to solve linear regression (curve fitting) problems. In regression, we aim to find a function $f$ that maps inputs x $\in \mathbb { R } ^ { D }$ to corresponding function values $f ( \pmb { x } ) \in \mathbb { R }$ . We assume we are given a set of training inputs ${ \bf { x } } _ { n }$ and corresponding noisy observations $y _ { n } = f ( \pmb { x } _ { n } ) + \epsilon ,$ , where ϵ is an i.i.d. random variable that describes measurement/observation noise and potentially unmodeled processes (which we will not consider further in this chapter). Throughout this chapter, we assume zero-mean Gaussian noise. Our task is to find a function that not only models the training data, but generalizes well to predicting function values at input locations that are not part of the training data (see Chapter 8). An illustration of such a regression problem is given in Figure 9.1. A typical regression setting is given in Figure 9.1(a): For some input values $x _ { n } ,$ we observe (noisy) function values $y _ { n } = f ( x _ { n } ) + \epsilon .$ . The task is to infer the function f that generated the data and generalizes well to function values at new input locations. A possible solution is given in Figure 9.1(b), where we also show three distributions centered at the function values $f ( x )$ that represent the noise in the data. 

Regression is a fundamental problem in machine learning, and regression problems appear in a diverse range of research areas and applica-

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/c10da196d6d331a147bd5ec206310193f9d0019731f094a1c52a981a8307528b.jpg)



(a) Regression problem: observed noisy function values from which we wish to infer the underlying function that generated the data.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/23e4a6e3c08d7b3833914b89b1d95b67b853e6b0719ff2e5a9c9cc485fd59f23.jpg)



Figure 9.1 (a) Dataset; (b) possible solution to the regression problem.



(b) Regression solution: possible function that could have generated the data (blue) with indication of the measurement noise of the function value at the corresponding inputs (orange distributions).


Normally, the type of noise could also be a “model choice”, but we fix the noise to be Gaussian in this chapter. 

tions, including time-series analysis (e.g., system identification), control and robotics (e.g., reinforcement learning, forward/inverse model learning), optimization (e.g., line searches, global optimization), and deeplearning applications (e.g., computer games, speech-to-text translation, image recognition, automatic video annotation). Regression is also a key ingredient of classification algorithms. Finding a regression function requires solving a variety of problems, including the following: 

Choice of the model (type) and the parametrization of the regression function. Given a dataset, what function classes (e.g., polynomials) are good candidates for modeling the data, and what particular parametrization (e.g., degree of the polynomial) should we choose? Model selection, as discussed in Section 8.6, allows us to compare various models to find the simplest model that explains the training data reasonably well. 

Finding good parameters. Having chosen a model of the regression function, how do we find good model parameters? Here, we will need to look at different loss/objective functions (they determine what a “good” fit is) and optimization algorithms that allow us to minimize this loss. 

Overfitting and model selection. Overfitting is a problem when the regression function fits the training data “too well” but does not generalize to unseen test data. Overfitting typically occurs if the underlying model (or its parametrization) is overly flexible and expressive; see Section 8.6. We will look at the underlying reasons and discuss ways to mitigate the effect of overfitting in the context of linear regression. 

Relationship between loss functions and parameter priors. Loss functions (optimization objectives) are often motivated and induced by probabilistic models. We will look at the connection between loss functions and the underlying prior assumptions that induce these losses. 

Uncertainty modeling. In any practical setting, we have access to only a finite, potentially large, amount of (training) data for selecting the model class and the corresponding parameters. Given that this finite amount of training data does not cover all possible scenarios, we may want to describe the remaining parameter uncertainty to obtain a measure of confidence of the model’s prediction at test time; the smaller the training set, the more important uncertainty modeling. Consistent modeling of uncertainty equips model predictions with confidence bounds. 

In the following, we will be using the mathematical tools from Chapters 3, 5, 6 and 7 to solve linear regression problems. We will discuss maximum likelihood and maximum a posteriori (MAP) estimation to find optimal model parameters. Using these parameter estimates, we will have a brief look at generalization errors and overfitting. Toward the end of this chapter, we will discuss Bayesian linear regression, which allows us to reason about model parameters at a higher level, thereby removing some of the problems encountered in maximum likelihood and MAP estimation. 

## 9.1 Problem Formulation

Because of the presence of observation noise, we will adopt a probabilistic approach and explicitly model the noise using a likelihood function. More specifically, throughout this chapter, we consider a regression problem with the likelihood function 

$$
p (y \mid \boldsymbol {x}) = \mathcal {N} \bigl (y \mid f (\boldsymbol {x}), \sigma^ {2} \bigr).\tag{9.1}
$$

Here, $\pmb { x } \in \mathbb { R } ^ { D }$ are inputs and $y \in \mathbb { R }$ are noisy function values (targets). With (9.1), the functional relationship between x and y is given as 

$$
y = f (\boldsymbol {x}) + \epsilon ,\tag{9.2}
$$

where $\epsilon \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ is independent, identically distributed (i.i.d.) Gaussian measurement noise with mean 0 and variance $\sigma ^ { 2 }$ . Our objective is to find a function that is close (similar) to the unknown function $f$ that generated the data and that generalizes well. 

In this chapter, we focus on parametric models, i.e., we choose a parametrized function and find parameters θ that “work well” for modeling the data. For the time being, we assume that the noise variance $\sigma ^ { 2 }$ is known and focus on learning the model parameters θ. In linear regression, we consider the special case that the parameters θ appear linearly in our model. An example of linear regression is given by 

$$
p (y \mid \boldsymbol {x}, \boldsymbol {\theta}) = \mathcal {N} \big (y \mid \boldsymbol {x} ^ {\top} \boldsymbol {\theta}, \sigma^ {2} \big)\tag{9.3}
$$

$$
\Longleftrightarrow y = \boldsymbol {x} ^ {\top} \boldsymbol {\theta} + \epsilon , \quad \epsilon \sim \mathcal {N} (0, \sigma^ {2}),\tag{9.4}
$$

where $\pmb { \theta } \in \mathbb { R } ^ { D }$ are the parameters we seek. The class of functions described by $( 9 . 4 )$ are straight lines that pass through the origin. In (9.4), we chose a parametrization $f ( \mathbf { x } ) = \mathbf { x } ^ { \top } \pmb \theta .$ 

The likelihood in (9.3) is the probability density function of $y$ evaluated at ${ \pmb x } ^ { \top } { \pmb \theta } .$ . Note that the only source of uncertainty originates from the observation noise (as x and $\pmb \theta$ are assumed known in (9.3)). Without observation noise, the relationship between x and y would be deterministic and (9.3) would be a Dirac delta. 

## Example 9.1

For $x , \theta \in \mathbb { R }$ the linear regression model in (9.4) describes straight lines (linear functions), and the parameter θ is the slope of the line. Figure 9.2(a) shows some example functions for different values of θ. 

The linear regression model in (9.3)–(9.4) is not only linear in the parameters, but also linear in the inputs x. Figure 9.2(a) shows examples of such functions. We will see later that $y = \phi ^ { \top } ( { \pmb x } ) { \pmb \theta }$ for nonlinear transformations $\phi$ is also a linear regression model because “linear regression” 

292 

Figure 9.2 Linear regression example. (a) Example functions that fall into this category; (b) training set; (c) maximum likelihood estimate. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/ee575e980c52fe4adf027322de48dc66717d598b76160d2b4b18e05cc78de525.jpg)



(a) Example functions (straight lines) that can be described using the linear model in (9.4).


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/7cdda173a791cca960988d2c07637dcdb94f11b93ad4353ce109a3ebcca6e7cf.jpg)



(b) Training set.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/31639c80aff9ea7a7497d1eca51f959ea3070bc5a25127bb68884a3ddbf327d3.jpg)



(c) Maximum likelihood estimate.


refers to models that are “linear in the parameters”, i.e., models that describe a function by a linear combination of input features. Here, a “feature” is a representation $\phi ( { \pmb x } )$ of the inputs ${ \pmb x } .$ 

In the following, we will discuss in more detail how to find good parameters θ and how to evaluate whether a parameter set “works well”. For the time being, we assume that the noise variance $\sigma ^ { 2 }$ is known. 

## 9.2 Parameter Estimation

training set 

Figure 9.3 Probabilistic graphical model for linear regression. Observed random variables are shaded, deterministic/ known values are without circles. 

Consider the linear regression setting (9.4) and assume we are given a training set $\mathcal { D } : = \{ ( \pmb { x } _ { 1 } , y _ { 1 } ) , \dots , ( \pmb { x } _ { N } , y _ { N } ) \}$ consisting of N inputs ${ \pmb x } _ { n } \in \qquad $ $\mathbb { R } ^ { D }$ and corresponding observations/targets $y _ { n } \in \mathbb { R } , n = 1 , . . . , N$ . The corresponding graphical model is given in Figure 9.3. Note that $y _ { i }$ and $y _ { j }$ are conditionally independent given their respective inputs ${ \pmb x } _ { i } , { \pmb x } _ { j }$ so that the likelihood factorizes according to 

$$
p (\mathcal {Y} \mid \mathcal {X}, \boldsymbol {\theta}) = p (y _ {1}, \dots , y _ {N} \mid \boldsymbol {x} _ {1}, \dots , \boldsymbol {x} _ {N}, \boldsymbol {\theta})\tag{9.5a}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/02229863eadc2a883dd23d2e3eb3c8da661aea3810f23948b1cfaeb37e377e23.jpg)


$$
= \prod_ {n = 1} ^ {N} p (y _ {n} \mid \boldsymbol {x} _ {n}, \boldsymbol {\theta}) = \prod_ {n = 1} ^ {N} \mathcal {N} \left(y _ {n} \mid \boldsymbol {x} _ {n} ^ {\top} \boldsymbol {\theta}, \sigma^ {2}\right),\tag{9.5b}
$$

where we defined ${ \mathcal { X } } : = \{ { \pmb x } _ { 1 } , \ldots , { \pmb x } _ { N } \}$ and $\mathcal { Y } : = \{ y _ { 1 } , . . . , y _ { N } \}$ as the sets of training inputs and corresponding targets, respectively. The likelihood and the factors $p ( y _ { n } \mid \pmb { x } _ { n } , \pmb { \theta } )$ are Gaussian due to the noise distribution; see (9.3). 

In the following, we will discuss how to find optimal parameters ${ \pmb { \theta } } ^ { * } \in$ $\mathbb { R } ^ { D }$ for the linear regression model (9.4). Once the parameters $\pmb { \theta } ^ { * }$ are found, we can predict function values by using this parameter estimate in (9.4) so that at an arbitrary test input ${ \pmb x } _ { \mathrm { : } }$ <sub>∗</sub> the distribution of the corresponding target $y _ { * }$ is 

$$
p (y _ {*} \mid \boldsymbol {x} _ {*}, \boldsymbol {\theta} ^ {*}) = \mathcal {N} \big (y _ {*} \mid \boldsymbol {x} _ {*} ^ {\top} \boldsymbol {\theta} ^ {*}, \sigma^ {2} \big).\tag{9.6}
$$

In the following, we will have a look at parameter estimation by maximizing the likelihood, a topic that we already covered to some degree in Section 8.3. 

## 9.2.1 Maximum Likelihood Estimation

A widely used approach to finding the desired parameters $\pmb { \theta } _ { \mathrm { M L } }$ is maximum likelihood estimation, where we find parameters $\pmb { \theta } _ { \mathrm { M I } }$ that maximize the likelihood (9.5b). Intuitively, maximizing the likelihood means maximizing the predictive distribution of the training data given the model parameters. We obtain the maximum likelihood parameters as 

$$
\boldsymbol {\theta} _ {\mathrm{ML}} \in \arg \max _ {\boldsymbol {\theta}} p (\mathcal {Y} | \mathcal {X}, \boldsymbol {\theta}).\tag{9.7}
$$

Remark. The likelihood $p ( \pmb { y } | \pmb { x } , \pmb { \theta } )$ is not a probability distribution in θ: It is simply a function of the parameters θ but does not integrate to 1 $( \mathrm { i . e . , }$ it is unnormalized), and may not even be integrable with respect to $\pmb \theta .$ However, the likelihood in (9.7) is a normalized probability distribution in y. $\diamondsuit$ 

To find the desired parameters $\pmb { \theta } _ { \mathrm { M L } }$ that maximize the likelihood, we typically perform gradient ascent (or gradient descent on the negative likelihood). In the case of linear regression we consider here, however, a closed-form solution exists, which makes iterative gradient descent un necessary. In practice, instead of maximizing the likelihood directly, we apply the log-transformation to the likelihood function and minimize the negative log-likelihood. 

Remark (Log-Transformation). Since the likelihood (9.5b) is a product of N Gaussian distributions, the log-transformation is useful since (a) it does not suffer from numerical underflow, and (b) the differentiation rules will turn out simpler. More specifically, numerical underflow will be a problem when we multiply N probabilities, where N is the number of data points, since we cannot represent very small numbers, such as $1 0 ^ { - 2 5 6 }$ Furthermore, the log-transform will turn the product into a sum of logprobabilities such that the corresponding gradient is a sum of individual gradients, instead of a repeated application of the product rule (5.46) to compute the gradient of a product of N terms. 

To find the optimal parameters $\pmb { \theta } _ { \mathrm { M L } }$ of our linear regression problem, we minimize the negative log-likelihood 

$$
- \log p (\mathcal {Y} \mid \mathcal {X}, \boldsymbol {\theta}) = - \log \prod_ {n = 1} ^ {N} p (y _ {n} \mid \boldsymbol {x} _ {n}, \boldsymbol {\theta}) = - \sum_ {n = 1} ^ {N} \log p (y _ {n} \mid \boldsymbol {x} _ {n}, \boldsymbol {\theta}),\tag{9.8}
$$

where we exploited that the likelihood (9.5b) factorizes over the number of data points due to our independence assumption on the training set. 

In the linear regression model (9.4), the likelihood is Gaussian (due to the Gaussian additive noise term), such that we arrive at 

$$
\log p (y _ {n} \mid \boldsymbol {x} _ {n}, \boldsymbol {\theta}) = - \frac {1}{2 \sigma^ {2}} (y _ {n} - \boldsymbol {x} _ {n} ^ {\top} \boldsymbol {\theta}) ^ {2} + \mathrm{const},\tag{9.9}
$$

where the constant includes all terms independent of θ. Using (9.9) in the 

Maximizing the likelihood means maximizing the predictive distribution of the (training) data given the parameters. The likelihood is not a probability distribution in the parameters. 

negative log-likelihood (9.8), we obtain (ignoring the constant terms) 

The negative log-likelihood function is also called error function. design matrix The squared error is often used as a measure of distance. Recall from Section 3.1 that $\| \pmb { x } \| ^ { 2 } = \pmb { x } ^ { \top } \pmb { x }$ if we choose the dot product as the inner product. 

Ignoring the possibility of duplicate data points, rk $\mathbf { \boldsymbol { x } } ) = D$ ${ \mathrm { i f ~ } } N \geqslant D , { \mathrm { i . e . , } }$ we do not have more parameters than data points. 

$$
\begin{array}{l} \mathcal {L} (\boldsymbol {\theta}) := \frac {1}{2 \sigma^ {2}} \sum_ {n = 1} ^ {N} (y _ {n} - \boldsymbol {x} _ {n} ^ {\top} \boldsymbol {\theta}) ^ {2} \\ \qquad = \frac {1}{2 \sigma^ {2}} (\boldsymbol {y} - \boldsymbol {X} \boldsymbol {\theta}) ^ {\top} (\boldsymbol {y} - \boldsymbol {X} \boldsymbol {\theta}) = \frac {1}{2 \sigma^ {2}} \| \boldsymbol {y} - \boldsymbol {X} \boldsymbol {\theta} \| ^ {2}, \end{array}\tag{9.10a}
$$

(9.10b) 

where we define the design matrix $\pmb { X } : = [ \pmb { x } _ { 1 } , \ldots , \pmb { x } _ { N } ] ^ { \top } \in \mathbb { R } ^ { N \times D }$ as the collection of training inputs and $\pmb { y } : = [ y _ { 1 } , \dots , y _ { N } ] ^ { \top } \in \mathbb { R } ^ { N }$ as a vector that collects all training targets. Note that the nth row in the design matrix X corresponds to the training input ${ \bf { x } } _ { n }$ . In (9.10b), we used the fact that the sum of squared errors between the observations $y _ { n }$ and the corresponding model prediction $\pmb { x } _ { n } ^ { \top } \pmb { \theta }$ equals the squared distance between y and Xθ. 

With (9.10b), we have now a concrete form of the negative log-likelihood function we need to optimize. We immediately see that (9.10b) is quadratic in $\pmb { \theta } .$ This means that we can find a unique global solution $\pmb { \theta } _ { \mathrm { M L } }$ for minimizing the negative log-likelihood ${ \mathcal { L } } .$ . We can find the global optimum by computing the gradient of ${ \mathcal { L } } ,$ setting it to 0 and solving for θ. 

Using the results from Chapter 5, we compute the gradient of $\mathcal { L }$ with respect to the parameters as 

$$
\frac {\mathrm{d} \mathcal {L}}{\mathrm{d} \boldsymbol {\theta}} = \frac {\mathrm{d}}{\mathrm{d} \boldsymbol {\theta}} \left(\frac {1}{2 \sigma^ {2}} (\boldsymbol {y} - \boldsymbol {X} \boldsymbol {\theta}) ^ {\top} (\boldsymbol {y} - \boldsymbol {X} \boldsymbol {\theta})\right)\tag{9.11a}
$$

$$
= \frac {1}{2 \sigma^ {2}} \frac {\mathrm{d}}{\mathrm{d} \boldsymbol {\theta}} \left(\boldsymbol {y} ^ {\top} \boldsymbol {y} - 2 \boldsymbol {y} ^ {\top} \boldsymbol {X} \boldsymbol {\theta} + \boldsymbol {\theta} ^ {\top} \boldsymbol {X} ^ {\top} \boldsymbol {X} \boldsymbol {\theta}\right)\tag{9.11b}
$$

$$
= \frac {1}{\sigma^ {2}} (- \pmb {y} ^ {\top} \pmb {X} + \pmb {\theta} ^ {\top} \pmb {X} ^ {\top} \pmb {X}) \in \mathbb {R} ^ {1 \times D}.\tag{9.11c}
$$

The maximum likelihood estimator $\pmb { \theta } _ { \mathrm { M L } }$ solves $\begin{array} { r } { \frac { \mathrm { d } \mathcal { L } } { \mathrm { d } \theta } = \mathbf { 0 } ^ { \top } } \end{array}$ (necessary optimality condition) and we obtain 

$$
\frac {\mathrm{d} \mathcal {L}}{\mathrm{d} \boldsymbol {\theta}} = \mathbf {0} ^ {\top} \stackrel {(9. 1 1 \mathrm{c})} {\Longleftrightarrow} \boldsymbol {\theta} _ {\mathrm{ML}} ^ {\top} \boldsymbol {X} ^ {\top} \boldsymbol {X} = \boldsymbol {y} ^ {\top} \boldsymbol {X}\tag{9.12a}
$$

$$
\Longleftrightarrow \boldsymbol {\theta} _ {\mathrm{ML}} ^ {\top} = \boldsymbol {y} ^ {\top} \boldsymbol {X} (\boldsymbol {X} ^ {\top} \boldsymbol {X}) ^ {- 1}\tag{9.12b}
$$

$$
\Longleftrightarrow \boldsymbol {\theta} _ {\mathrm{ML}} = (\boldsymbol {X} ^ {\top} \boldsymbol {X}) ^ {- 1} \boldsymbol {X} ^ {\top} \boldsymbol {y}.\tag{9.12c}
$$

We could right-multiply the first equation by $( X ^ { \top } X ) ^ { - 1 }$ because $\boldsymbol { X } ^ { \top } \boldsymbol { X }$ is positive definite if rk $( \pmb { X } ) = \pmb { D }$ , where rk $( X )$ denotes the rank of X. 

Remark. Setting the gradient to $\mathbf { 0 } ^ { \top }$ is a necessary and sufficient condition, and we obtain a global minimum since the Hessian $\nabla _ { \pmb { \theta } } ^ { 2 } { \mathcal { L } } ( \pmb { \theta } ) = \pmb { X } ^ { \top } \pmb { X } \in$ $\mathbb { R } ^ { D \times D }$ is positive definite. ◇ 

Remark. The maximum likelihood solution in (9.12c) requires us to solve a system of linear equations of the form $\mathbf { { \boldsymbol { A } } } \theta = \mathbf { \boldsymbol { b } }$ with $\bar { A } = ( X ^ { \top } X )$ and $\pmb { b } = \pmb { X } ^ { \top } \pmb { y }$ 

## Example 9.2 (Fitting Lines)

Let us have a look at Figure 9.2, where we aim to fit a straight line $f ( x ) =$ $\theta x$ , where θ is an unknown slope, to a dataset using maximum likelihood estimation. Examples of functions in this model class (straight lines) are shown in Figure $9 . 2 ( \mathrm { a } )$ . For the dataset shown in Figure 9.2(b), we find the maximum likelihood estimate of the slope parameter θ using (9.12c) and obtain the maximum likelihood linear function in Figure 9.2(c). 

## Maximum Likelihood Estimation with Features

So far, we considered the linear regression setting described in (9.4), which allowed us to fit straight lines to data using maximum likelihood estimation. However, straight lines are not sufficiently expressive when it comes to fitting more interesting data. Fortunately, linear regression offers us a way to fit nonlinear functions within the linear regression framework: Since “linear regression” only refers to “linear in the parameters”, we can perform an arbitrary nonlinear transformation $\phi ( { \pmb x } )$ of the inputs x and then linearly combine the components of this transformation. The corresponding linear regression model is 

$$
\begin{array}{c} p (y \mid \boldsymbol {x}, \boldsymbol {\theta}) = \mathcal {N} \big (y \mid \boldsymbol {\phi} ^ {\top} (\boldsymbol {x}) \boldsymbol {\theta},   \sigma^ {2} \big) \\ \Longleftrightarrow y = \boldsymbol {\phi} ^ {\top} (\boldsymbol {x}) \boldsymbol {\theta} + \epsilon = \sum_ {k = 0} ^ {K - 1} \theta_ {k} \phi_ {k} (\boldsymbol {x}) + \epsilon  , \end{array}\tag{9.13}
$$

where $\phi : \mathbb { R } ^ { D }  \mathbb { R } ^ { K }$ is a (nonlinear) transformation of the inputs x and $\phi _ { k } : \mathbb { R } ^ { D } $ <sup>R</sup> is the kth component of the feature vector $\phi$ . Note that the model parameters θ still appear only linearly. 

## Example 9.3 (Polynomial Regression)

We are concerned with a regression problem $y = \phi ^ { \top } ( x ) \theta { + } \epsilon .$ , where $x \in \mathbb { R }$ and $\pmb \theta \in \mathbb { R } ^ { K }$ . A transformation that is often used in this context is 

$$
\boldsymbol {\phi} (x) = \left[ \begin{array}{c} \phi_ {0} (x) \\ \phi_ {1} (x) \\ \vdots \\ \phi_ {K - 1} (x) \end{array} \right] = \left[ \begin{array}{c} 1 \\ x \\ x ^ {2} \\ x ^ {3} \\ \vdots \\ x ^ {K - 1} \end{array} \right] \in \mathbb {R} ^ {K}.\tag{9.14}
$$

This means that we $^ { \mathfrak { s o } } \mathrm { l i f f } ^ { \mathfrak { n } }$ the original one-dimensional input space into a K-dimensional feature space consisting of all monomials $x ^ { k }$ for $k =$ $0 , \ldots , K - 1$ With these features, we can model polynomials of degree <sup>⩽</sup> K 1 within the framework of linear regression: A polynomial of degree 



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



K 1 is 

$$
f (x) = \sum_ {k = 0} ^ {K - 1} \theta_ {k} x ^ {k} = \boldsymbol {\phi} ^ {\top} (x) \boldsymbol {\theta},\tag{9.15}
$$

where $\phi$ is defined in (9.14) and $\pmb { \theta } = [ \theta _ { 0 } , \ldots , \theta _ { K - 1 } ] ^ { \top } \in \mathbb { R } ^ { K }$ contains the (linear) parameters $\theta _ { k }$ 

Let us now have a look at maximum likelihood estimation of the parameters θ in the linear regression model (9.13). We consider training inputs ${ \pmb x } _ { n } \in \mathbb { R } ^ { D }$ and targets $y _ { n } \in \mathbb R , n = 1 , . . . , N$ , and define the feature matrix (design matrix) as 

$$
\boldsymbol {\Phi} := \left[ \begin{array}{c} \boldsymbol {\phi} ^ {\top} (\boldsymbol {x} _ {1}) \\ \vdots \\ \boldsymbol {\phi} ^ {\top} (\boldsymbol {x} _ {N}) \end{array} \right] = \left[ \begin{array}{c c c} \phi_ {0} (\boldsymbol {x} _ {1}) & \dots & \phi_ {K - 1} (\boldsymbol {x} _ {1}) \\ \phi_ {0} (\boldsymbol {x} _ {2}) & \dots & \phi_ {K - 1} (\boldsymbol {x} _ {2}) \\ \vdots & & \vdots \\ \phi_ {0} (\boldsymbol {x} _ {N}) & \dots & \phi_ {K - 1} (\boldsymbol {x} _ {N}) \end{array} \right] \in \mathbb {R} ^ {N \times K},\tag{9.16}
$$

where $\Phi _ { i j } = \phi _ { j } ( \pmb { x } _ { i } )$ and $\phi _ { j } : \mathbb { R } ^ { D }  \mathbb { R } .$ 

Example 9.4 (Feature Matrix for Second-order Polynomials) For a second-order polynomial and N training points $x _ { n } ~ \in ~ \mathbb { R } , n ~ =$ $1 , \ldots , N$ , the feature matrix is 

$$
\boldsymbol {\Phi} = \left[ \begin{array}{c c c} 1 & x _ {1} & x _ {1} ^ {2} \\ 1 & x _ {2} & x _ {2} ^ {2} \\ \vdots & \vdots & \vdots \\ 1 & x _ {N} & x _ {N} ^ {2} \end{array} \right].\tag{9.17}
$$

With the feature matrix Φ defined in (9.16), the negative log-likelihood for the linear regression model (9.13) can be written as 

$$
- \log p (\mathcal {Y} \mid \mathcal {X}, \boldsymbol {\theta}) = \frac {1}{2 \sigma^ {2}} (\boldsymbol {y} - \boldsymbol {\Phi} \boldsymbol {\theta}) ^ {\top} (\boldsymbol {y} - \boldsymbol {\Phi} \boldsymbol {\theta}) + \text { const }.\tag{9.18}
$$

Comparing (9.18) with the negative log-likelihood in (9.10b) for the “feature-free” model, we immediately see we just need to replace X with Φ. Since both X and Φ are independent of the parameters $\pmb \theta$ that we wish to optimize, we arrive immediately at the maximum likelihood estimate 

$$
\boldsymbol {\theta} _ {\mathrm{ML}} = \left(\boldsymbol {\Phi} ^ {\top} \boldsymbol {\Phi}\right) ^ {- 1} \boldsymbol {\Phi} ^ {\top} \boldsymbol {y}\tag{9.19}
$$

for the linear regression problem with nonlinear features defined in (9.13). Remark. When we were working without features, we required $\boldsymbol { X } ^ { \top } \boldsymbol { X }$ to be invertible, which is the case when $\operatorname { r k } ( { \boldsymbol { X } } ) = D ,$ i.e., the columns of X 

## 9.2 Parameter Estimation

are linearly independent. In (9.19), we therefore require $\Phi ^ { \top } \Phi \in \mathbb { R } ^ { K \times K }$ to be invertible. This is the case if and only if rk $( \Phi ) = K$ 

## Example 9.5 (Maximum Likelihood Polynomial Fit)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/1414261d146867558986ddd3ca27822570240a5c48c177471e25aeaa8f47684e.jpg)



(a) Regression dataset.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/2f1901600e6138670bff67e3d97e7383f1962248dcdd9fdeea504692eac93e1e.jpg)



Figure 9.4 Polynomial regression: (a) dataset consisting of (x<sub>n</sub>, y<sub>n</sub>) pairs, $n = 1 , \ldots , 1 0 ;$ (b) maximum likelihood polynomial of degree 4.


(b) Polynomial of degree 4 determined by maximum likelihood estimation. 

Consider the dataset in Figure 9.4(a). The dataset consists of $N = 1 0$ pairs $( x _ { n } , y _ { n } )$ , where $x _ { n } \sim \mathcal { U } [ - 5 , 5 ]$ and $y _ { n } = - \sin ( x _ { n } / 5 ) + \cos ( x _ { n } ) + \epsilon _ { \scriptscriptstyle 3 }$ where $\epsilon \sim \mathcal { N } ( 0 , 0 . 2 ^ { 2 } )$ 

We fit a polynomial of degree 4 using maximum likelihood estimation, i.e., parameters $\pmb { \theta } _ { \mathrm { M L } }$ are given in (9.19). The maximum likelihood estimate yields function values $\phi ^ { \top } ( x _ { * } ) \theta _ { \mathrm { M L } }$ at any test location $x _ { * }$ . The result is shown in Figure 9.4(b). 

## Estimating the Noise Variance

Thus far, we assumed that the noise variance $\sigma ^ { 2 }$ is known. However, we can also use the principle of maximum likelihood estimation to obtain the maximum likelihood estimator $\sigma _ { \mathrm { M L } } ^ { 2 }$ for the noise variance. To do this, we follow the standard procedure: We write down the log-likelihood, compute its derivative with respect to $\sigma ^ { 2 } > 0$ , set it to 0, and solve. The log-likelihood is given by 

$$
\log p (\mathcal {Y} | \mathcal {X}, \boldsymbol {\theta}, \sigma^ {2}) = \sum_ {n = 1} ^ {N} \log \mathcal {N} \big (y _ {n} | \boldsymbol {\phi} ^ {\top} (\boldsymbol {x} _ {n}) \boldsymbol {\theta}, \sigma^ {2} \big)\tag{9.20a}
$$

$$
= \sum_ {n = 1} ^ {N} \left(- \frac {1}{2} \log (2 \pi) - \frac {1}{2} \log \sigma^ {2} - \frac {1}{2 \sigma^ {2}} (y _ {n} - \boldsymbol {\phi} ^ {\top} (\boldsymbol {x} _ {n}) \boldsymbol {\theta}) ^ {2}\right)\tag{9.20b}
$$

$$
= - \frac {N}{2} \log \sigma^ {2} - \frac {1}{2 \sigma^ {2}} \underbrace {\sum_ {n = 1} ^ {N} (y _ {n} - \boldsymbol {\phi} ^ {\top} (\boldsymbol {x} _ {n}) \boldsymbol {\theta}) ^ {2}} _ {=: s} + \text { const }.\tag{9.20c}
$$



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



The partial derivative of the log-likelihood with respect to $\sigma ^ { 2 }$ is then 

$$
\frac {\partial \log p (\mathcal {Y} | \mathcal {X} , \pmb {\theta} , \sigma^ {2})}{\partial \sigma^ {2}} = - \frac {N}{2 \sigma^ {2}} + \frac {1}{2 \sigma^ {4}} s = 0\tag{9.21a}
$$

$$
\Longleftrightarrow \frac {N}{2 \sigma^ {2}} = \frac {s}{2 \sigma^ {4}}\tag{9.21b}
$$

so that we identify 

$$
\sigma_ {\mathrm{ML}} ^ {2} = \frac {s}{N} = \frac {1}{N} \sum_ {n = 1} ^ {N} (y _ {n} - \boldsymbol {\phi} ^ {\top} (\pmb {x} _ {n}) \pmb {\theta}) ^ {2}.\tag{9.22}
$$

Therefore, the maximum likelihood estimate of the noise variance is the empirical mean of the squared distances between the noise-free function values $\phi ^ { \top } ( { \pmb x } _ { n } ) \pmb \theta$ and the corresponding noisy observations $y _ { n }$ at input locations ${ \bf { x } } _ { n }$ 

## 9.2.2 Overfitting in Linear Regression

We just discussed how to use maximum likelihood estimation to fit linear models (e.g., polynomials) to data. We can evaluate the quality of the model by computing the error/loss incurred. One way of doing this is to compute the negative log-likelihood (9.10b), which we minimized to determine the maximum likelihood estimator. Alternatively, given that the noise parameter $\sigma ^ { 2 }$ is not a free model parameter, we can ignore the scaling by $1 / \sigma ^ { 2 }$ , so that we end up with a squared-error-loss function $\lVert y - \bar { \Phi } \theta \rVert ^ { 2 }$ . Instead of using this squared loss, we often use the root mean square error (RMSE) 

$$
\sqrt {\frac {1}{N} \left\| \boldsymbol {y} - \boldsymbol {\Phi} \boldsymbol {\theta} \right\| ^ {2}} = \sqrt {\frac {1}{N} \sum_ {n = 1} ^ {N} (y _ {n} - \boldsymbol {\phi} ^ {\top} (\boldsymbol {x} _ {n}) \boldsymbol {\theta}) ^ {2}},\tag{9.23}
$$

which (a) allows us to compare errors of datasets with different sizes and (b) has the same scale and the same units as the observed function values $y _ { n }$ . For example, if we fit a model that maps post-codes (x is given in latitude, longitude) to house prices (y-values are EUR) then the RMSE is also measured in EUR, whereas the squared error is given in EUR<sup>2</sup>. If we choose to include the factor $\sigma ^ { 2 }$ from the original negative log-likelihood (9.10b), then we end up with a unitless objective, i.e., in the preceding example, our objective would no longer be in EUR or $\mathrm { E U R ^ { 2 } }$ 

For model selection (see Section 8.6), we can use the RMSE (or the negative log-likelihood) to determine the best degree of the polynomial by finding the polynomial degree M that minimizes the objective. Given that the polynomial degree is a natural number, we can perform a brute-force search and enumerate all (reasonable) values of M. For a training set of size N it is sufficient to test $0 \leqslant M \leqslant N - 1$ . For $M < N$ , the maximum likelihood estimator is unique. For $M \geqslant N$ , we have more parameters 

## 9.2 Parameter Estimation

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/bc189ff857c1de9bd96ac71678b19bed1b28f61e9d2e04808d1ed0cb13d54538.jpg)



(a) $M = 0$


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/af525894d244040143832bc74f7b6b4cf1ed50a275b28e57ff209cc397ea366d.jpg)



(b) M = 1


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/bed320ed18b6bf0ad0efc29042df9c41abb9555a97c30aa3492cfe3805035228.jpg)



299



(c) $M = 3$



Figure 9.5 Maximum likelihood fits for different polynomial degrees M.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/e50f723a58cfa2a4e7d461ddb1822781a7ae0c96709086587d934d0faf447a65.jpg)



(d) $M = 4$


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/e4ba9749295be922c8bec20169da980caa730a4fa0bf468ecb7e9fc12444d307.jpg)



(e) $M = 6$


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/d352f2d53a3138ab4754a7e9390e43f3c8fc828425cbce5cf72fbe752f147774.jpg)



(f) $M = 9$


than data points, and would need to solve an underdetermined system of linear equations $( \Phi ^ { \top } \Phi$ in (9.19) would also no longer be invertible) so that there are infinitely many possible maximum likelihood estimators. 

Figure 9.5 shows a number of polynomial fits determined by maximum likelihood for the dataset from Figure $9 . 4 ( \mathrm { a } )$ with $N = 1 0$ observations. We notice that polynomials of low degree (e.g., constants $( M = 0 )$ or linear $( M = 1 ) )$ fit the data poorly and, hence, are poor representations of the true underlying function. For degrees $M = 3 , \ldots , 6 ,$ , the fits look plausible and smoothly interpolate the data. When we go to higher-degree polynomials, we notice that they fit the data better and better. In the extreme case of $M = N - 1 = 9$ , the function will pass through every single data point. However, these high-degree polynomials oscillate wildly and are a poor representation of the underlying function that generated the data, such that we suffer from overfitting. 

Remember that the goal is to achieve good generalization by making accurate predictions for new (unseen) data. We obtain some quantitative insight into the dependence of the generalization performance on the polynomial of degree M by considering a separate test set comprising 200 data points generated using exactly the same procedure used to generate the training set. As test inputs, we chose a linear grid of 200 points in the interval of [ 5, 5]. For each choice of M, we evaluate the RMSE (9.23) for both the training data and the test data. 

Looking now at the test error, which is a qualitive measure of the gen eralization properties of the corresponding polynomial, we notice that initially the test error decreases; see Figure 9.6 (orange). For fourth-order polynomials, the test error is relatively low and stays relatively constant up to degree 5. However, from degree 6 onward the test error increases signif icantly, and high-order polynomials have very bad generalization properties. In this particular example, this also is evident from the corresponding 

The case of M = N − 1 is extreme in the sense that otherwise the null space of the corresponding system of linear equations would be non-trivial, and we would have infinitely many optimal solutions to the linear regression problem. 

overfitting Note that the noise variance $\sigma ^ { 2 } > 0 .$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/6fe7f0107bd4ce01c54ca5fd8a65088dfe172fadea5680fc6da841bfbb7f2351.jpg)


maximum likelihood fits in Figure 9.5. Note that the training error (blue curve in Figure 9.6) never increases when the degree of the polynomial increases. In our example, the best generalization (the point of the smallest test error) is obtained for a polynomial of degree $M = 4$ 

## 9.2.3 Maximum A Posteriori Estimation

We just saw that maximum likelihood estimation is prone to overfitting. We often observe that the magnitude of the parameter values becomes relatively large if we run into overfitting (Bishop, 2006). 

To mitigate the effect of huge parameter values, we can place a prior distribution $p ( \pmb \theta )$ on the parameters. The prior distribution explicitly encodes what parameter values are plausible (before having seen any data). For example, a Gaussian prior $p ( \theta ) \ = \ \mathcal { N } ( 0 , 1 )$ on a single parameter $\theta$ encodes that parameter values are expected lie in the interval $[ - 2 , 2 ]$ (two standard deviations around the mean value). Once a dataset $x , y$ is available, instead of maximizing the likelihood we seek parameters that maximize the posterior distribution $p ( \pmb \theta | \mathcal { X } , \mathcal { Y } )$ . This procedure is called maximum a posteriori (MAP) estimation. 

The posterior over the parameters $\theta ,$ given the training data $x , y$ , is obtained by applying Bayes’ theorem (Section 6.3) as 

$$
p (\boldsymbol {\theta} \mid \mathcal {X}, \mathcal {Y}) = \frac {p (\mathcal {Y} \mid \mathcal {X} , \boldsymbol {\theta}) p (\boldsymbol {\theta})}{p (\mathcal {Y} \mid \mathcal {X})}.\tag{9.24}
$$

Since the posterior explicitly depends on the parameter prior $p ( \pmb \theta )$ , the prior will have an effect on the parameter vector we find as the maximizer of the posterior. We will see this more explicitly in the following. The parameter vector $\theta _ { \mathrm { M A P } }$ that maximizes the posterior (9.24) is the MAP estimate. 

To find the MAP estimate, we follow steps that are similar in flavor to maximum likelihood estimation. We start with the log-transform and compute the log-posterior as 

$$
\log p (\boldsymbol {\theta} \mid \mathcal {X}, \mathcal {Y}) = \log p (\mathcal {Y} \mid \mathcal {X}, \boldsymbol {\theta}) + \log p (\boldsymbol {\theta}) + \mathrm{const},\tag{9.25}
$$

Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 

where the constant comprises the terms that are independent of θ. We see that the log-posterior in (9.25) is the sum of the log-likelihood $p ( \mathcal { V } | \mathcal { X } , \pmb { \theta } )$ and the log-prior log $p ( \pmb \theta )$ so that the MAP estimate will be a “compromise” between the prior (our suggestion for plausible parameter values before observing data) and the data-dependent likelihood. 

To find the MAP estimate $\theta _ { \mathrm { M A P } } { \mathrm { . } }$ , we minimize the negative log-posterior distribution with respect to $\theta ,$ i.e., we solve 

$$
\boldsymbol {\theta} _ {\text { MAP }} \in \arg \min _ {\boldsymbol {\theta}} \left\{- \log p (\mathcal {Y} \mid \mathcal {X}, \boldsymbol {\theta}) - \log p (\boldsymbol {\theta}) \right\}.\tag{9.26}
$$

The gradient of the negative log-posterior with respect to θ is 

$$
- \frac {\mathrm{d} \log p (\pmb {\theta} | \mathcal {X} , \mathcal {Y})}{\mathrm{d} \pmb {\theta}} = - \frac {\mathrm{d} \log p (\mathcal {Y} | \mathcal {X} , \pmb {\theta})}{\mathrm{d} \pmb {\theta}} - \frac {\mathrm{d} \log p (\pmb {\theta})}{\mathrm{d} \pmb {\theta}},\tag{9.27}
$$

where we identify the first term on the right-hand side as the gradient of the negative log-likelihood from (9.11c). 

With a (conjugate) Gaussian prior $p ( \pmb \theta ) = \mathcal { N } \big ( \mathbf 0 , b ^ { 2 } \pmb I \big )$ on the parameters $\theta ,$ the negative log-posterior for the linear regression setting (9.13), we obtain the negative log posterior 

$$
- \log p (\boldsymbol {\theta} \mid \mathcal {X}, \mathcal {Y}) = \frac {1}{2 \sigma^ {2}} (\boldsymbol {y} - \boldsymbol {\Phi} \boldsymbol {\theta}) ^ {\top} (\boldsymbol {y} - \boldsymbol {\Phi} \boldsymbol {\theta}) + \frac {1}{2 b ^ {2}} \boldsymbol {\theta} ^ {\top} \boldsymbol {\theta} + \text { const }  .\tag{9.28}
$$

Here, the first term corresponds to the contribution from the log-likelihood, and the second term originates from the log-prior. The gradient of the logposterior with respect to the parameters $\pmb \theta$ is then 

$$
- \frac {\mathrm{d} \log p (\boldsymbol {\theta} \mid \mathcal {X} , \mathcal {Y})}{\mathrm{d} \boldsymbol {\theta}} = \frac {1}{\sigma^ {2}} (\boldsymbol {\theta} ^ {\top} \boldsymbol {\Phi} ^ {\top} \boldsymbol {\Phi} - \boldsymbol {y} ^ {\top} \boldsymbol {\Phi}) + \frac {1}{b ^ {2}} \boldsymbol {\theta} ^ {\top}.\tag{9.29}
$$

We will find the MAP estimate $\theta _ { \mathrm { M A P } }$ by setting this gradient to $\mathbf { 0 } ^ { \top }$ and solving for $\theta _ { \mathrm { M A P } }$ . We obtain 

$$
\frac {1}{\sigma^ {2}} (\pmb {\theta} ^ {\top} \pmb {\Phi} ^ {\top} \pmb {\Phi} - \pmb {y} ^ {\top} \pmb {\Phi}) + \frac {1}{b ^ {2}} \pmb {\theta} ^ {\top} = \pmb {0} ^ {\top}\tag{9.30a}
$$

$$
\Longleftrightarrow \boldsymbol {\theta} ^ {\top} \left(\frac {1}{\sigma^ {2}} \boldsymbol {\Phi} ^ {\top} \boldsymbol {\Phi} + \frac {1}{b ^ {2}} \boldsymbol {I}\right) - \frac {1}{\sigma^ {2}} \boldsymbol {y} ^ {\top} \boldsymbol {\Phi} = \boldsymbol {0} ^ {\top}\tag{9.30b}
$$

$$
\Longleftrightarrow \boldsymbol {\theta} ^ {\top} \left(\boldsymbol {\Phi} ^ {\top} \boldsymbol {\Phi} + \frac {\sigma^ {2}}{b ^ {2}} \boldsymbol {I}\right) = \boldsymbol {y} ^ {\top} \boldsymbol {\Phi}\tag{9.30c}
$$

$$
\Longleftrightarrow \boldsymbol {\theta} ^ {\top} = \boldsymbol {y} ^ {\top} \boldsymbol {\Phi} \left(\boldsymbol {\Phi} ^ {\top} \boldsymbol {\Phi} + \frac {\sigma^ {2}}{b ^ {2}} \boldsymbol {I}\right) ^ {- 1}\tag{9.30d}
$$

so that the MAP estimate is (by transposing both sides of the last equality) 

$$
\boldsymbol {\theta} _ {\mathrm{MAP}} = \left(\boldsymbol {\Phi} ^ {\top} \boldsymbol {\Phi} + \frac {\sigma^ {2}}{b ^ {2}} \boldsymbol {I}\right) ^ {- 1} \boldsymbol {\Phi} ^ {\top} \boldsymbol {y}.\tag{9.31}
$$

Comparing the MAP estimate in (9.31) with the maximum likelihood estimate in (9.19), we see that the only difference between both solutions is the additional term $\begin{array} { r } { \frac { \sigma ^ { 2 } } { b ^ { 2 } } I } \end{array}$ in the inverse matrix. This term ensures that 

$\Phi ^ { \top }$ Φ is symmetric, positive semi definite. The additional term in (9.31) is strictly positive definite so that the inverse exists. 

$\begin{array} { r } { \Phi ^ { \top } \Phi + \frac { \sigma ^ { 2 } } { b ^ { 2 } } I } \end{array}$ is symmetric and strictly positive definite (i.e., its inverse exists and the MAP estimate is the unique solution of a system of linear equations). Moreover, it reflects the impact of the regularizer. 

## Example 9.6 (MAP Estimation for Polynomial Regression)

In the polynomial regression example from Section 9.2.1, we place a Gaussian prior $p ( \pmb \theta ) = \mathcal { N } ( \mathbf 0 , I )$ on the parameters θ and determine the MAP estimates according to (9.31). In Figure 9.7, we show both the maximum likelihood and the MAP estimates for polynomials of degree 6 (left) and degree 8 (right). The prior (regularizer) does not play a significant role for the low-degree polynomial, but keeps the function relatively smooth for higher-degree polynomials. Although the MAP estimate can push the boundaries of overfitting, it is not a general solution to this problem, so we need a more principled approach to tackle overfitting. 

Figure 9.7 Polynomial regression: maximum likelihood and MAP estimates. (a) Polynomials of degree 6; (b) polynomials of degree 8. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/a848da53be8664f1882a796b813eaa021586fd3008ad2726dc9c298e0df9cd85.jpg)



(a) Polynomials of degree 6.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/46a0758ac48190ddd5a42200a6f1c3f8228e946cce1bca11d93cc943af04d5b6.jpg)



(b) Polynomials of degree 8.


## 9.2.4 MAP Estimation as Regularization

Instead of placing a prior distribution on the parameters $\theta ,$ it is also possible to mitigate the effect of overfitting by penalizing the amplitude of the parameter by means of regularization. In regularized least squares, we consider the loss function 

$$
\left\| \boldsymbol {y} - \boldsymbol {\Phi} \boldsymbol {\theta} \right\| ^ {2} + \lambda \left\| \boldsymbol {\theta} \right\| _ {2} ^ {2},\tag{9.32}
$$

which we minimize with respect to $\pmb \theta$ (see Section 8.2.3). Here, the first term is a data $- f i t$ term (also called misfit term), which is proportional to the negative log-likelihood; see (9.10b). The second term is called the regularizer, and the regularization parameter $\lambda \geqslant 0$ controls the “strictness” of the regularization. 

Remark. Instead of the Euclidean norm $\left\| \cdot \right\| _ { 2 }$ , we can choose any p-norm $\left\| \cdot \right\| _ { p }$ in (9.32). In practice, smaller values for $p$ lead to sparser solutions. Here, “sparse” means that many parameter values $\theta _ { d } = 0 _ { i }$ , which is also useful for variable selection. For $p = 1$ , the regularizer is called LASSO (least absolute shrinkage and selection operator) and was proposed by Tibshirani (1996). $\diamondsuit$ 

The regularizer $\lambda \left. \pmb { \theta } \right. _ { 2 } ^ { 2 }$ in (9.32) can be interpreted as a negative log-Gaussian prior, which we use in MAP estimation; see (9.26). More specifically, with a Gaussian prior $p ( \pmb \theta ) = \mathcal { N } ( \mathbf 0 , b ^ { 2 } \pmb I )$ , we obtain the negative log-Gaussian prior 

$$
- \log p (\boldsymbol {\theta}) = \frac {1}{2 b ^ {2}} \| \boldsymbol {\theta} \| _ {2} ^ {2} + \text { const }\tag{9.33}
$$

so that for $\begin{array} { r } { \lambda = \frac { 1 } { 2 b ^ { 2 } } } \end{array}$ the regularization term and the negative log-Gaussian prior are identical. 

Given that the regularized least-squares loss function in (9.32) consists of terms that are closely related to the negative log-likelihood plus a negative log-prior, it is not surprising that, when we minimize this loss, we obtain a solution that closely resembles the MAP estimate in (9.31). More specifically, minimizing the regularized least-squares loss function yields 

$$
\boldsymbol {\theta} _ {\mathrm{RLS}} = \left(\boldsymbol {\Phi} ^ {\top} \boldsymbol {\Phi} + \lambda \boldsymbol {I}\right) ^ {- 1} \boldsymbol {\Phi} ^ {\top} \boldsymbol {y},\tag{9.34}
$$

which is identical to the MAP estimate in (9.31) for $\textstyle \lambda = { \frac { \sigma ^ { 2 } } { b ^ { 2 } } }$ , where $\sigma ^ { 2 }$ is the noise variance and $b ^ { 2 }$ the variance of the (isotropic) Gaussian prior $p ( \pmb \theta ) = \mathcal { N } \big ( \mathbf 0 , b ^ { 2 } \pmb I \big )$ 

So far, we have covered parameter estimation using maximum likelihood and MAP estimation where we found point estimates $\pmb { \theta } ^ { * }$ that optimize an objective function (likelihood or posterior). We saw that both maximum likelihood and MAP estimation can lead to overfitting. In the next section, we will discuss Bayesian linear regression, where we use Bayesian inference (Section 8.4) to find a posterior distribution over the unknown parameters, which we subsequently use to make predictions. More specifically, for predictions we will average over all plausible sets of parameters instead of focusing on a point estimate. 

## 9.3 Bayesian Linear Regression

Previously, we looked at linear regression models where we estimated the model parameters θ, e.g., by means of maximum likelihood or MAP estimation. We discovered that MLE can lead to severe overfitting, in particular, in the small-data regime. MAP addresses this issue by placing a prior on the parameters that plays the role of a regularizer. 

Bayesian linear regression pushes the idea of the parameter prior a step further and does not even attempt to compute a point estimate of the parameters, but instead the full posterior distribution over the parameters is taken into account when making predictions. This means we do not fit any parameters, but we compute a mean over all plausible parameters settings (according to the posterior). 

©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 

## 9.3.1 Model

In Bayesian linear regression, we consider the model 

$$
\begin{array}{l l} \text {prior} & p (\boldsymbol {\theta}) = \mathcal {N} \big (\boldsymbol {m} _ {0},   \boldsymbol {S} _ {0} \big)  , \\ \text {likelihood} & p (y \mid \boldsymbol {x}, \boldsymbol {\theta}) = \mathcal {N} \big (y \mid \boldsymbol {\phi} ^ {\top} (\boldsymbol {x}) \boldsymbol {\theta},   \sigma^ {2} \big)  , \end{array}\tag{9.35}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/b712af0cd3b7ca9422c4e201a1c9fc3dcaf9df5d8ef48caea98d0ab5dbfe5f7b.jpg)


where we now explicitly place a Gaussian prior $p ( \pmb \theta ) = \mathcal { N } \big ( m _ { 0 } , S _ { 0 } \big )$ on $\theta ,$ which turns the parameter vector into a random variable. This allows us to write down the corresponding graphical model in Figure 9.8, where we made the parameters of the Gaussian prior on $\pmb \theta$ explicit. The full probabilistic model, i.e., the joint distribution of observed and unobserved random variables, y and $\theta ,$ , respectively, is 

$$
p (y, \boldsymbol {\theta} \mid \boldsymbol {x}) = p (y \mid \boldsymbol {x}, \boldsymbol {\theta}) p (\boldsymbol {\theta}).\tag{9.36}
$$

## 9.3.2 Prior Predictions

In practice, we are usually not so much interested in the parameter values $\pmb \theta$ themselves. Instead, our focus often lies in the predictions we make with those parameter values. In a Bayesian setting, we take the parameter distribution and average over all plausible parameter settings when we make predictions. More specifically, to make predictions at an input $^ { \mathbf { x } } { } _ { * }$ we integrate out $\pmb \theta$ and obtain 

$$
p (y _ {*} \mid \boldsymbol {x} _ {*}) = \int p (y _ {*} \mid \boldsymbol {x} _ {*}, \pmb {\theta}) p (\pmb {\theta}) \mathrm{d} \pmb {\theta} = \mathbb {E} _ {\pmb {\theta}} [ p (y _ {*} \mid \boldsymbol {x} _ {*}, \pmb {\theta}) ],\tag{9.37}
$$

which we can interpret as the average prediction of $y _ { * } \mid x _ { * } , \theta$ for all plausible parameters $\pmb \theta$ according to the prior distribution $p ( \pmb \theta )$ . Note that predictions using the prior distribution only require us to specify the input $^ { \mathbf { x } } { } _ { * }$ , but no training data. 

In our model (9.35), we chose a conjugate (Gaussian) prior on $\pmb \theta$ so that the predictive distribution is Gaussian as well (and can be computed in closed form): With the prior distribution $p ( \pmb \theta ) = \mathcal { N } \big ( m _ { 0 } , S _ { 0 } \big )$ , we obtain the predictive distribution as 

$$
p (y _ {*} \mid \boldsymbol {x} _ {*}) = \mathcal {N} \big (\phi^ {\top} (\boldsymbol {x} _ {*}) \boldsymbol {m} _ {0}, \phi^ {\top} (\boldsymbol {x} _ {*}) \boldsymbol {S} _ {0} \phi (\boldsymbol {x} _ {*}) + \sigma^ {2} \big),\tag{9.38}
$$

where we exploited that (i) the prediction is Gaussian due to conjugacy (see Section $6 . 6 )$ and the marginalization property of Gaussians (see Section 6.5), (ii) the Gaussian noise is independent so that 

$$
\mathbb {V} [ y _ {*} ] = \mathbb {V} _ {\pmb {\theta}} [ \pmb {\phi} ^ {\top} (\pmb {x} _ {*}) \pmb {\theta} ] + \mathbb {V} _ {\epsilon} [ \epsilon ],\tag{9.39}
$$

and (iii) $y _ { * }$ is a linear transformation of $\pmb \theta$ so that we can apply the rules for computing the mean and covariance of the prediction analytically by using (6.50) and (6.51), respectively. In (9.38), the term $\phi ^ { \top } ( { \pmb x } _ { * } ) S _ { 0 } \phi ( { \pmb x } _ { * } )$ in the predictive variance explicitly accounts for the uncertainty associated with the parameters $\theta ,$ whereas $\sigma ^ { 2 }$ is the uncertainty contribution due to the measurement noise. 

If we are interested in predicting noise-free function values $f ( x _ { * } ) =$ $\phi ^ { \top } ( x _ { * } ) \theta$ instead of the noise-corrupted targets $y _ { * }$ we obtain 

$$
p (f (\boldsymbol {x} _ {*})) = \mathcal {N} \big (\boldsymbol {\phi} ^ {\top} (\boldsymbol {x} _ {*}) \boldsymbol {m} _ {0}, \boldsymbol {\phi} ^ {\top} (\boldsymbol {x} _ {*}) \boldsymbol {S} _ {0} \boldsymbol {\phi} (\boldsymbol {x} _ {*}) \big),\tag{9.40}
$$

which only differs from (9.38) in the omission of the noise variance $\sigma ^ { 2 }$ in the predictive variance. 

Remark (Distribution over Functions). Since we can represent the distribution $p ( \pmb \theta )$ using a set of samples $\pmb \theta _ { i }$ and every sample $\pmb \theta _ { i }$ gives rise to a function $f _ { i } ( \cdot ) = \mathbf { \bar { \theta } } _ { i } ^ { \top } \boldsymbol { \phi } ( \cdot )$ , it follows that the parameter distribution $p ( \pmb \theta )$ induces a distribution $p ( f ( \cdot ) )$ over functions. Here we use the notation ( ) to explicitly denote a functional relationship. 

## Example 9.7 (Prior over Functions)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/95449cc77c20abd3466c2de35380838c314d980a7f8a60f103bdc6f7976ee824.jpg)



(a) Prior distribution over functions.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/16e59b439379b32a3d2839a4cb3d9dee4fbd0bc14f0c27b73167fa44ae4edfcc.jpg)



(b) Samples from the prior distribution over functions.


Let us consider a Bayesian linear regression problem with polynomials of degree 5. We choose a parameter prior $p ( \pmb \theta ) = \mathcal { N } \big ( \mathbf 0 , \frac { 1 } { 4 } \pmb I \big )$ . Figure 9.9 visualizes the induced prior distribution over functions (shaded area: dark gray: 67% confidence bound; light gray: 95% confidence bound) induced by this parameter prior, including some function samples from this prior. 

A function sample is obtained by first sampling a parameter vector $\theta _ { i } \sim p ( \theta )$ and then computing $f _ { i } ( \cdot ) = \pmb { \theta } _ { i } ^ { \top } \phi ( \cdot )$ . We used 200 input locations $x _ { * } ~ \in ~ [ - 5 , 5 ]$ to which we apply the feature function $\phi ( \cdot )$ . The uncertainty (represented by the shaded area) in Figure 9.9 is solely due to the parameter uncertainty because we considered the noise-free predictive distribution (9.40). 

So $\operatorname { f a r } ,$ we looked at computing predictions using the parameter prior $p ( \pmb \theta )$ . However, when we have a parameter posterior (given some training data , ), the same principles for prediction and inference hold as in (9.37) – we just need to replace the prior $p ( \pmb \theta )$ with the posterior $p ( \pmb \theta | \mathcal { X } , \mathcal { Y } )$ . In the following, we will derive the posterior distribution in detail before using it to make predictions. 

## 9.3.3 Posterior Distribution

Given a training set of inputs ${ \pmb x } _ { n } \in \mathbb { R } ^ { D }$ and corresponding observations $y _ { n } \in \mathbb { R } , n = 1 , \dots , N$ , we compute the posterior over the parameters using Bayes’ theorem as 

$$
p (\boldsymbol {\theta} \mid \mathcal {X}, \mathcal {Y}) = \frac {p (\mathcal {Y} \mid \mathcal {X} , \boldsymbol {\theta}) p (\boldsymbol {\theta})}{p (\mathcal {Y} \mid \mathcal {X})},\tag{9.41}
$$

where $\mathcal { X }$ is the set of training inputs and the collection of corresponding training targets. Furthermore, $p ( \mathcal { V } | \mathcal { X } , \pmb { \theta } )$ is the likelihood, $p ( \pmb \theta )$ the parameter prior, and 

$$
p (\mathcal {Y} \mid \mathcal {X}) = \int p (\mathcal {Y} \mid \mathcal {X}, \boldsymbol {\theta}) p (\boldsymbol {\theta}) \mathrm{d} \boldsymbol {\theta} = \mathbb {E} _ {\boldsymbol {\theta}} [ p (\mathcal {Y} \mid \mathcal {X}, \boldsymbol {\theta}) ]\tag{9.42}
$$

the marginal likelihood/evidence, which is independent of the parameters $\pmb \theta$ and ensures that the posterior is normalized, i.e., it integrates to 1. We can think of the marginal likelihood as the likelihood averaged over all possible parameter settings (with respect to the prior distribution $p ( \pmb \theta ) )$ . 

Theorem 9.1 (Parameter Posterior). In our model (9.35), the parameter posterior (9.41) can be computed in closed form as 

$$
p (\boldsymbol {\theta} \mid \mathcal {X}, \mathcal {Y}) = \mathcal {N} \bigl (\boldsymbol {\theta} \mid \boldsymbol {m} _ {N}, \boldsymbol {S} _ {N} \bigr),\tag{9.43a}
$$

$$
\boldsymbol {S} _ {N} = (\boldsymbol {S} _ {0} ^ {- 1} + \sigma^ {- 2} \boldsymbol {\Phi} ^ {\top} \boldsymbol {\Phi}) ^ {- 1},\tag{9.43b}
$$

$$
\pmb {m} _ {N} = \pmb {S} _ {N} (\pmb {S} _ {0} ^ {- 1} \pmb {m} _ {0} + \sigma^ {- 2} \pmb {\Phi} ^ {\top} \pmb {y}),\tag{9.43c}
$$

where the subscript N indicates the size of the training set. 

Proof Bayes’ theorem tells us that the posterior $p ( \pmb \theta | \mathcal { X } , \mathcal { Y } )$ is proportional to the product of the likelihood $p ( \mathcal { V } | \mathcal { X } , \pmb { \theta } )$ and the prior $p ( \pmb \theta )$ 

$$
\text { Posterior } \quad p (\boldsymbol {\theta} \mid \mathcal {X}, \mathcal {Y}) = \frac {p (\mathcal {Y} \mid \mathcal {X} , \boldsymbol {\theta}) p (\boldsymbol {\theta})}{p (\mathcal {Y} \mid \mathcal {X})}\tag{9.44a}
$$

$$
\text { Likelihood } \quad p (\mathcal {Y} \mid \mathcal {X}, \boldsymbol {\theta}) = \mathcal {N} (\boldsymbol {y} \mid \boldsymbol {\Phi} \boldsymbol {\theta}, \sigma^ {2} \boldsymbol {I})\tag{9.44b}
$$

$$
\text { Prior } \quad p (\boldsymbol {\theta}) = \mathcal {N} \bigl (\boldsymbol {\theta} \mid \boldsymbol {m} _ {0},   \boldsymbol {S} _ {0} \bigr).\tag{9.44c}
$$

Instead of looking at the product of the prior and the likelihood, we can transform the problem into log-space and solve for the mean and covariance of the posterior by completing the squares. 

The sum of the log-prior and the log-likelihood is 

$$
\log \mathcal {N} (\boldsymbol {y} \mid \boldsymbol {\Phi} \boldsymbol {\theta}, \sigma^ {2} \boldsymbol {I}) + \log \mathcal {N} (\boldsymbol {\theta} \mid \boldsymbol {m} _ {0}, \boldsymbol {S} _ {0})\tag{9.45a}
$$

$$
= - \frac {1}{2} \left(\sigma^ {- 2} (\boldsymbol {y} - \boldsymbol {\Phi} \boldsymbol {\theta}) ^ {\top} (\boldsymbol {y} - \boldsymbol {\Phi} \boldsymbol {\theta}) + (\boldsymbol {\theta} - \boldsymbol {m} _ {0}) ^ {\top} S _ {0} ^ {- 1} (\boldsymbol {\theta} - \boldsymbol {m} _ {0})\right) + \text { const }\tag{9.45b}
$$

where the constant contains terms independent of $\pmb \theta .$ We will ignore the constant in the following. We now factorize (9.45b), which yields 

$$
\begin{array}{l} - \frac {1}{2} \left(\sigma^ {- 2} \boldsymbol {y} ^ {\top} \boldsymbol {y} - 2 \sigma^ {- 2} \boldsymbol {y} ^ {\top} \boldsymbol {\Phi} \boldsymbol {\theta} + \boldsymbol {\theta} ^ {\top} \sigma^ {- 2} \boldsymbol {\Phi} ^ {\top} \boldsymbol {\Phi} \boldsymbol {\theta} + \boldsymbol {\theta} ^ {\top} \boldsymbol {S} _ {0} ^ {- 1} \boldsymbol {\theta} \right. \\ - 2 \boldsymbol {m} _ {0} ^ {\top} \boldsymbol {S} _ {0} ^ {- 1} \boldsymbol {\theta} + \boldsymbol {m} _ {0} ^ {\top} \boldsymbol {S} _ {0} ^ {- 1} \boldsymbol {m} _ {0}) \\ = - \frac {1}{2} \left(\boldsymbol {\theta} ^ {\top} \left(\sigma^ {- 2} \boldsymbol {\Phi} ^ {\top} \boldsymbol {\Phi} + \boldsymbol {S} _ {0} ^ {- 1}\right) \boldsymbol {\theta} - 2 \left(\sigma^ {- 2} \boldsymbol {\Phi} ^ {\top} \boldsymbol {y} + \boldsymbol {S} _ {0} ^ {- 1} \boldsymbol {m} _ {0}\right) ^ {\top} \boldsymbol {\theta}\right) + \text {const}, \end{array} \tag {9.46a}\tag{9.46b}
$$

where the constant contains the black terms in (9.46a), which are independent of $\pmb { \theta } .$ . The orange terms are terms that are linear in $\theta ,$ and the blue terms are the ones that are quadratic in $\pmb \theta .$ Inspecting (9.46b), we find that this equation is quadratic in $\pmb { \theta } .$ The fact that the unnormalized log-posterior distribution is a (negative) quadratic form implies that the posterior is Gaussian, i.e., 

$$
p (\boldsymbol {\theta} \mid \mathcal {X}, \mathcal {Y}) = \exp (\log p (\boldsymbol {\theta} \mid \mathcal {X}, \mathcal {Y})) \propto \exp (\log p (\mathcal {Y} \mid \mathcal {X}, \boldsymbol {\theta}) + \log p (\boldsymbol {\theta}))\tag{9.47a}
$$

$$
\propto \exp \left(- \frac {1}{2} \big (\boldsymbol {\theta} ^ {\top} (\sigma^ {- 2} \boldsymbol {\Phi} ^ {\top} \boldsymbol {\Phi} + \boldsymbol {S} _ {0} ^ {- 1}) \boldsymbol {\theta} - 2 (\sigma^ {- 2} \boldsymbol {\Phi} ^ {\top} \boldsymbol {y} + \boldsymbol {S} _ {0} ^ {- 1} \boldsymbol {m} _ {0}) ^ {\top} \boldsymbol {\theta} \big)\right),\tag{9.47b}
$$

where we used (9.46b) in the last expression. 

The remaining task is it to bring this (unnormalized) Gaussian into the form that is proportional to $\mathcal { N } ( \pmb { \theta } | \pmb { m } _ { N } , \pmb { S } _ { N } )$ , i.e., we need to identify the mean $\mathbf {  { m } } _ { N }$ and the covariance matrix $S _ { N }$ . To do this, we use the concept of completing the squares. The desired log-posterior is 

$$
\begin{array}{l} \log \mathcal {N} \big (\boldsymbol {\theta} \mid \boldsymbol {m} _ {N},   \boldsymbol {S} _ {N} \big) = - \frac {1}{2} (\boldsymbol {\theta} - \boldsymbol {m} _ {N}) ^ {\top} \boldsymbol {S} _ {N} ^ {- 1} (\boldsymbol {\theta} - \boldsymbol {m} _ {N}) + \text {const} \\ \qquad = - \frac {1}{2} \big (\boldsymbol {\theta} ^ {\top} \boldsymbol {S} _ {N} ^ {- 1} \boldsymbol {\theta} - 2 \boldsymbol {m} _ {N} ^ {\top} \boldsymbol {S} _ {N} ^ {- 1} \boldsymbol {\theta} + \boldsymbol {m} _ {N} ^ {\top} \boldsymbol {S} _ {N} ^ {- 1} \boldsymbol {m} _ {N} \big). \end{array}\tag{9.48a}
$$

(9.48b) 

Here, we factorized the quadratic form $( { \pmb \theta } - { \pmb m } _ { N } ) ^ { \top } { \pmb S } _ { N } ^ { - 1 } ( { \pmb \theta } - { \pmb m } _ { N } )$ into a term that is quadratic in $\pmb \theta$ alone (blue), a term that is linear in $\pmb \theta$ (orange), and a constant term (black). This allows us now to find $S _ { N }$ and m<sub>N</sub> by matching the colored expressions in (9.46b) and (9.48b), which yields 

$$
\boldsymbol {S} _ {N} ^ {- 1} = \boldsymbol {\Phi} ^ {\top} \sigma^ {- 2} \boldsymbol {I} \boldsymbol {\Phi} + \boldsymbol {S} _ {0} ^ {- 1}\tag{9.49a}
$$

$$
\Longleftrightarrow \boldsymbol {S} _ {N} = \left(\sigma^ {- 2} \boldsymbol {\Phi} ^ {\top} \boldsymbol {\Phi} + \boldsymbol {S} _ {0} ^ {- 1}\right) ^ {- 1}\tag{9.49b}
$$

and 

$$
\pmb {m} _ {N} ^ {\top} \pmb {S} _ {N} ^ {- 1} = \left(\sigma^ {- 2} \pmb {\Phi} ^ {\top} \pmb {y} + \pmb {S} _ {0} ^ {- 1} \pmb {m} _ {0}\right) ^ {\top}\tag{9.50a}
$$

$$
\Longleftrightarrow \boldsymbol {m} _ {N} = \boldsymbol {S} _ {N} \left(\sigma^ {- 2} \boldsymbol {\Phi} ^ {\top} \boldsymbol {y} + \boldsymbol {S} _ {0} ^ {- 1} \boldsymbol {m} _ {0}\right).\tag{9.50b}
$$



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



Remark (General Approach to Completing the Squares). If we are given an equation 

$$
\boldsymbol {x} ^ {\top} \boldsymbol {A} \boldsymbol {x} - 2 \boldsymbol {a} ^ {\top} \boldsymbol {x} + \operatorname{const} _ {1},\tag{9.51}
$$

where $\pmb { A }$ is symmetric and positive definite, which we wish to bring into the form 

$$
\left(\boldsymbol {x} - \boldsymbol {\mu}\right) ^ {\top} \boldsymbol {\Sigma} (\boldsymbol {x} - \boldsymbol {\mu}) + \operatorname{const} _ {2},\tag{9.52}
$$

we can do this by setting 

$$
\boldsymbol {\Sigma} := \boldsymbol {A},\tag{9.53}
$$

$$
\pmb {\mu} := \pmb {\Sigma} ^ {- 1} \pmb {a}\tag{9.54}
$$

and $\mathrm { c o n s t _ { 2 } } = \mathrm { c o n s t _ { 1 } } - \mu ^ { \top } \Sigma \mu$ 

♢ 

We can see that the terms inside the exponential in (9.47b) are of the form (9.51) with 

$$
\boldsymbol {A} := \sigma^ {- 2} \boldsymbol {\Phi} ^ {\top} \boldsymbol {\Phi} + \boldsymbol {S} _ {0} ^ {- 1},\tag{9.55}
$$

$$
\boldsymbol {a} := \sigma^ {- 2} \boldsymbol {\Phi} ^ {\top} \boldsymbol {y} + \boldsymbol {S} _ {0} ^ {- 1} \boldsymbol {m} _ {0}.\tag{9.56}
$$

Since ${ \mathbf { } } A , a$ can be difficult to identify in equations like (9.46a), it is often helpful to bring these equations into the form (9.51) that decouples quadratic term, linear terms, and constants, which simplifies finding the desired solution. 

## 9.3.4 Posterior Predictions

In (9.37), we computed the predictive distribution of $y _ { * }$ at a test input x using the parameter prior $p ( \pmb \theta )$ . In principle, predicting with the parameter posterior $p ( \pmb \theta | \mathcal { X } , \mathcal { Y } )$ is not fundamentally different given that in our conjugate model the prior and posterior are both Gaussian (with different parameters). Therefore, by following the same reasoning as in Section 9.3.2, we obtain the (posterior) predictive distribution 

$$
p (y _ {*} \mid \mathcal {X}, \mathcal {Y}, \boldsymbol {x} _ {*}) = \int p (y _ {*} \mid \boldsymbol {x} _ {*}, \boldsymbol {\theta}) p (\boldsymbol {\theta} \mid \mathcal {X}, \mathcal {Y}) \mathrm{d} \boldsymbol {\theta}\tag{9.57a}
$$

$$
= \int \mathcal {N} \left(y _ {*} \mid \phi^ {\top} \left(\boldsymbol {x} _ {*}\right) \boldsymbol {\theta}, \sigma^ {2}\right) \mathcal {N} \left(\boldsymbol {\theta} \mid \boldsymbol {m} _ {N}, \boldsymbol {S} _ {N}\right) \mathrm{d} \boldsymbol {\theta}\tag{9.57b}
$$

$$
= \mathcal {N} \big (y _ {*} \mid \boldsymbol {\phi} ^ {\top} (\boldsymbol {x} _ {*}) \boldsymbol {m} _ {N}, \boldsymbol {\phi} ^ {\top} (\boldsymbol {x} _ {*}) \boldsymbol {S} _ {N} \boldsymbol {\phi} (\boldsymbol {x} _ {*}) + \sigma^ {2} \big)\tag{9.57c}
$$

<sup>E</sup>[y<sub>∗</sub> | X , Y, x<sub>∗</sub>] = ϕ<sup>⊤</sup>(x<sub>∗</sub>)m<sub>N</sub> = $\phi ^ { \top } ( { \pmb x } _ { * } ) { \pmb \theta } _ { \mathrm { M A P } } ,$ 

The term ${ \phi } ^ { \top } ( { \pmb x } _ { * } ) { \pmb S } _ { N } { \phi } ( { \pmb x } _ { * } )$ reflects the posterior uncertainty associated with the parameters θ. Note that $S _ { N }$ depends on the training inputs through Φ; see (9.43b). The predictive mean $\phi ^ { \top } ( { \pmb x } _ { * } ) m _ { N }$ coincides with the predictions made with the MAP estimate $\theta _ { \mathrm { M A P } }$ 

Remark (Marginal Likelihood and Posterior Predictive Distribution). By replacing the integral in (9.57a), the predictive distribution can be equivalently written as the expectation $\mathbb { E } _ { \pmb { \theta } \mid x , y } [ p ( y _ { \ast } \mid \pmb { x } _ { \ast } , \pmb { \theta } ) ]$ , where the expectation is taken with respect to the parameter posterior $p ( \pmb \theta | \mathcal { X } , \mathcal { Y } )$ 

Writing the posterior predictive distribution in this way highlights a close resemblance to the marginal likelihood (9.42). The key difference between the marginal likelihood and the posterior predictive distribution are (i) the marginal likelihood can be thought of predicting the training targets y and not the test targets $y _ { * }$ , and (ii) the marginal likelihood averages with respect to the parameter prior and not the parameter posterior. ◇ 

Remark (Mean and Variance of Noise-Free Function Values). In many cases, we are not interested in the predictive distribution $p ( y _ { * } | \mathcal { X } , \mathcal { Y } , x _ { * } )$ of a (noisy) observation $y _ { * }$ . Instead, we would like to obtain the distribution of the (noise-free) function values $f ( { \pmb x } _ { * } ) = \phi ^ { \top } ( { \pmb x } _ { * } ) \pmb \theta$ . We determine the corresponding moments by exploiting the properties of means and variances, which yields 

$$
\begin{array}{r} \mathbb {E} [ f (\pmb {x} _ {*}) | \mathcal {X}, \mathcal {Y} ] = \mathbb {E} _ {\pmb {\theta}} [ \phi^ {\top} (\pmb {x} _ {*}) \pmb {\theta} | \mathcal {X}, \mathcal {Y} ] = \phi^ {\top} (\pmb {x} _ {*}) \mathbb {E} _ {\pmb {\theta}} [ \pmb {\theta} | \mathcal {X}, \mathcal {Y} ] \\ = \phi^ {\top} (\pmb {x} _ {*}) \pmb {m} _ {N} = \pmb {m} _ {N} ^ {\top} \phi (\pmb {x} _ {*}), \end{array}\tag{9.58}
$$

$$
\begin{array}{r l} & {\mathbb {V} _ {\boldsymbol {\theta}} [ f (\boldsymbol {x} _ {*}) \mid \mathcal {X}, \mathcal {Y} ] = \mathbb {V} _ {\boldsymbol {\theta}} [ \phi^ {\top} (\boldsymbol {x} _ {*}) \boldsymbol {\theta} \mid \mathcal {X}, \mathcal {Y} ]} \\ & {\qquad = \phi^ {\top} (\boldsymbol {x} _ {*}) \mathbb {V} _ {\boldsymbol {\theta}} [ \boldsymbol {\theta} \mid \mathcal {X}, \mathcal {Y} ] \phi (\boldsymbol {x} _ {*})} \\ & {\qquad = \phi^ {\top} (\boldsymbol {x} _ {*}) S _ {N} \phi (\boldsymbol {x} _ {*}).} \end{array}\tag{9.59}
$$

We see that the predictive mean is the same as the predictive mean for noisy observations as the noise has mean 0, and the predictive variance only differs by $\sigma ^ { 2 }$ , which is the variance of the measurement noise: When we predict noisy function values, we need to include $\sigma ^ { 2 }$ as a source of uncertainty, but this term is not needed for noise-free predictions. Here, the only remaining uncertainty stems from the parameter posterior. 

Remark (Distribution over Functions). The fact that we integrate out the parameters θ induces a distribution over functions: If we sample $\theta _ { i } \sim$ $p ( \pmb \theta | \mathcal { X } , \mathcal { Y } )$ from the parameter posterior, we obtain a single function realization $\theta _ { i } ^ { \top } \phi ( \cdot )$ . The mean function, i.e., the set of all expected function values $\mathbb { E } _ { \pmb { \theta } } [ \bar { f } ( \cdot ) | \pmb { \theta } , \mathcal { X } , \mathcal { Y } ]$ , of this distribution over functions is $m _ { N } ^ { \top } \phi ( \cdot )$ The (marginal) variance, i.e., the variance of the function $f ( \cdot )$ , is given by $\phi ^ { \top } ( \cdot ) S _ { N } \phi ( \cdot )$ 

## Example 9.8 (Posterior over Functions)

Let us revisit the Bayesian linear regression problem with polynomials of degree 5. We choose a parameter prior $p ( \pmb \theta ) = \mathcal { N } \big ( \mathbf 0 , \frac { 1 } { 4 } \pmb I \big )$ . Figure 9.9 visualizes the prior over functions induced by the parameter prior and sample functions from this prior. 



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



Figure 9.10 shows the posterior over functions that we obtain via Bayesian linear regression. The training dataset is shown in panel (a); panel (b) shows the posterior distribution over functions, including the functions we would obtain via maximum likelihood and MAP estimation. The function we obtain using the MAP estimate also corresponds to the posterior mean function in the Bayesian linear regression setting. Panel (c) shows some plausible realizations (samples) of functions under that posterior over functions. 

Figure 9.10 Bayesian linear regression and posterior over functions. (a) training data; (b) posterior distribution over functions; (c) Samples from the posterior over functions. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/658e0d245a0696d7d3b69d990017721b5f228532a24552b8e3b907742a0568e8.jpg)



(a) Training data.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/a70dc9c6a989b4f91da35fe1dbce1b220f222878aa7ad3c809e4d6156974c2e2.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/a98c3d190a3cb2526e5b0a9bb7702a14b24f5878d1f3a79ac05369c0329dfc57.jpg)



(b) Posterior over functions rep- (c) Samples from the posterior resented by the marginal uncer- over functions, which are intainties (shaded) showing the duced by the samples from the 67% and 95% predictive con- parameter posterior. fidence bounds, the maximum likelihood estimate (MLE) and the MAP estimate (MAP), the latter of which is identical to the posterior mean function.


Figure 9.11 shows some posterior distributions over functions induced by the parameter posterior. For different polynomial degrees $M ,$ , the left panels show the maximum likelihood function $\theta _ { \mathrm { M L } } ^ { \top } \phi ( \cdot )$ , the MAP function $\theta _ { \mathrm { M A P } } ^ { \top } \phi ( \cdot )$ (which is identical to the posterior mean function), and the 67% and 95% predictive confidence bounds obtained by Bayesian linear regression, represented by the shaded areas. 

The right panels show samples from the posterior over functions: Here, we sampled parameters $\pmb \theta _ { i }$ from the parameter posterior and computed the function $\phi ^ { \top } ( { \pmb x } _ { * } ) \pmb \theta _ { i }$ , which is a single realization of a function under the posterior distribution over functions. For low-order polynomials, the parameter posterior does not allow the parameters to vary much: The sampled functions are nearly identical. When we make the model more flexible by adding more parameters (i.e., we end up with a higher-order polynomial), these parameters are not sufficiently constrained by the posterior, and the sampled functions can be easily visually separated. We also see in the corresponding panels on the left how the uncertainty increases, especially at the boundaries. 

Although for a seventh-order polynomial the MAP estimate yields a reasonable fit, the Bayesian linear regression model additionally tells us that 

## 9.3 Bayesian Linear Regression

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/57284a30b0683f40132825d5c8bfc17d606ad9c6a7734605cf4e91e958914d04.jpg)



(a) Posterior distribution for polynomials of degree M = 3 (left) and samples from the posterior over functions (right).


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/9ed028577076dcc3245de7e3ebc95555b4028b3dd33ac09ba0507e1f379cc58c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/45e6daf4af6e7b46620522411d4770211ea90b5df1cb70b916816a76a79437ad.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/deb75a0e80ecb4c56e68d8f925b63332579b6e544132e293c3510ff2cf321651.jpg)



Figure 9.11 Bayesian linear regression. Left panels: Shaded areas indicate the 67% (dark gray) and 95% (light gray) predictive confidence bounds. The mean of the Bayesian linear regression model coincides with the MAP estimate. The predictive uncertainty is the sum of the noise term and the posterior parameter uncertainty, which depends on the location of the test input. Right panels: sampled functions from the posterior distribution.



(b) Posterior distribution for polynomials of degree M = 5 (left) and samples from the posterior over functions (right).


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/8ddf1aa656a91bb62e00adbdc21f70a4d59103a664086e80dc84f852ea0d0f81.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/2f2e0455113fd3ffafb3d01397277677bd2ca76d14e916d63b0641c41344a2df.jpg)



(c) Posterior distribution for polynomials of degree M = 7 (left) and samples from the posterior over functions (right).


the posterior uncertainty is huge. This information can be critical when we use these predictions in a decision-making system, where bad decisions can have significant consequences (e.g., in reinforcement learning or robotics). 

©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 

## 9.3.5 Computing the Marginal Likelihood

In Section 8.6.2, we highlighted the importance of the marginal likelihood for Bayesian model selection. In the following, we compute the marginal likelihood for Bayesian linear regression with a conjugate Gaussian prior on the parameters, i.e., exactly the setting we have been discussing in this chapter. 

Just to recap, we consider the following generative process: 

$$
\boldsymbol {\theta} \sim \mathcal {N} (\boldsymbol {m} _ {0}, \boldsymbol {S} _ {0})\tag{9.60a}
$$

$$
y _ {n} \mid \boldsymbol {x} _ {n}, \boldsymbol {\theta} \sim \mathcal {N} \left(\boldsymbol {x} _ {n} ^ {\top} \boldsymbol {\theta}, \sigma^ {2}\right),\tag{9.60b}
$$

The marginal likelihood can be interpreted as the expected likelihood under the prior, $\mathrm { i . e . , }$ $\mathbb { E } _ { \pmb { \theta } } [ p ( \mathcal { V } | \mathcal { X } , \pmb { \theta } ) ]$ . 

$n = 1 , \ldots , N$ . The marginal likelihood is given by 

$$
p (\mathcal {Y} \mid \mathcal {X}) = \int p (\mathcal {Y} \mid \mathcal {X}, \pmb {\theta}) p (\pmb {\theta}) \mathrm{d} \pmb {\theta}\tag{9.61a}
$$

$$
= \int \mathcal {N} (\boldsymbol {y} \mid \boldsymbol {X} \boldsymbol {\theta}, \sigma^ {2} \boldsymbol {I}) \mathcal {N} (\boldsymbol {\theta} \mid \boldsymbol {m} _ {0}, \boldsymbol {S} _ {0}) \mathrm{d} \boldsymbol {\theta},\tag{9.61b}
$$

where we integrate out the model parameters θ. We compute the marginal likelihood in two steps: First, we show that the marginal likelihood is Gaussian (as a distribution in $_ { y ) }$ ; second, we compute the mean and covariance of this Gaussian. 

1. The marginal likelihood is Gaussian: From Section 6.5.2, we know that (i) the product of two Gaussian random variables is an (unnormalized) Gaussian distribution, and (ii) a linear transformation of a Gaussian random variable is Gaussian distributed. In (9.61b), we require a linear transformation to bring $\mathcal { N } ( \pmb { y } | \pmb { X \theta } , \sigma ^ { 2 } \pmb { I } )$  into the form $\mathcal { N } ( \pmb { \theta } | \pmb { \mu } , \pmb { \Sigma } )$ for some $\mu , \Sigma$ . Once this is done, the integral can be solved in closed form. The result is the normalizing constant of the product of the two Gaussians. The normalizing constant itself has Gaussian shape; see (6.76). 

2. Mean and covariance. We compute the mean and covariance matrix of the marginal likelihood by exploiting the standard results for means and covariances of affine transformations of random variables; see Section 6.4.4. The mean of the marginal likelihood is computed as 

$$
\mathbb {E} [ \mathcal {Y} \mid \mathcal {X} ] = \mathbb {E} _ {\boldsymbol {\theta}, \epsilon} [ \boldsymbol {X} \boldsymbol {\theta} + \epsilon ] = \boldsymbol {X} \mathbb {E} _ {\boldsymbol {\theta}} [ \boldsymbol {\theta} ] = \boldsymbol {X} \boldsymbol {m} _ {0}.\tag{9.62}
$$

Note that $\epsilon \sim \mathcal { N } ( 0 , \sigma ^ { 2 } I )$ is a vector of i.i.d. random variables. The covariance matrix is given as 

$$
\operatorname{Cov} [ \mathcal {Y} | \mathcal {X} ] = \operatorname{Cov} _ {\boldsymbol {\theta}, \epsilon} [ \boldsymbol {X} \boldsymbol {\theta} + \epsilon ] = \operatorname{Cov} _ {\boldsymbol {\theta}} [ \boldsymbol {X} \boldsymbol {\theta} ] + \sigma^ {2} \boldsymbol {I}\tag{9.63a}
$$

$$
= \boldsymbol {X} \operatorname{Cov} _ {\boldsymbol {\theta}} [ \boldsymbol {\theta} ] \boldsymbol {X} ^ {\top} + \sigma^ {2} \boldsymbol {I} = \boldsymbol {X} \boldsymbol {S} _ {0} \boldsymbol {X} ^ {\top} + \sigma^ {2} \boldsymbol {I}.\tag{9.63b}
$$

Hence, the marginal likelihood is 

$$
p (\mathcal {Y} | \mathcal {X}) = (2 \pi) ^ {- \frac {N}{2}} \det (\boldsymbol {X} \boldsymbol {S} _ {0} \boldsymbol {X} ^ {\top} + \sigma^ {2} \boldsymbol {I}) ^ {- \frac {1}{2}}\tag{9.64a}
$$

$$
\cdot \exp \left(- \frac {1}{2} (\boldsymbol {y} - \boldsymbol {X m} _ {0}) ^ {\top} (\boldsymbol {X S} _ {0} \boldsymbol {X} ^ {\top} + \sigma^ {2} \boldsymbol {I}) ^ {- 1} (\boldsymbol {y} - \boldsymbol {X m} _ {0})\right)
$$

## 9.4 Maximum Likelihood as Orthogonal Projection

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/658ef19927483af1c247b221ababaa9a0984c075bdc815cdeb3002c6d97c59ed.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/d0ec79a748154d87a172d35f2c99291aede47d0a0238c328067ed6d6c15216ad.jpg)



(a) Regression dataset consisting of noisy observations $y _ { n }$ (blue) of function values $f ( x _ { n } )$ at input locations $x _ { n }$



Figure 9.12 Geometric interpretation of least squares. (a) Dataset; (b) maximum likelihood solution interpreted as a projection.



(b) The orange dots are the projections of the noisy observations (blue dots) onto the line $\theta _ { \mathrm { M L } } x$ . The maximum likelihood solution to a linear regression problem finds a subspace (line) onto which the overall projection error (orange lines) of the observations is minimized.


$$
= \mathcal {N} (\boldsymbol {y} \mid \boldsymbol {X m} _ {0}, \boldsymbol {X S} _ {0} \boldsymbol {X} ^ {\top} + \sigma^ {2} \boldsymbol {I}).\tag{9.64b}
$$

Given the close connection with the posterior predictive distribution (see Remark on Marginal Likelihood and Posterior Predictive Distribution earlier in this section), the functional form of the marginal likelihood should not be too surprising. 

## 9.4 Maximum Likelihood as Orthogonal Projection

Having crunched through much algebra to derive maximum likelihood and MAP estimates, we will now provide a geometric interpretation of maximum likelihood estimation. Let us consider a simple linear regression setting 

$$
y = x \theta + \epsilon , \quad \epsilon \sim \mathcal {N} (0, \sigma^ {2}),\tag{9.65}
$$

in which we consider linear functions $f : \mathbb { R }  \mathbb { R }$ that go through the origin (we omit features here for clarity). The parameter θ determines the slope of the line. Figure 9.12(a) shows a one-dimensional dataset. 

With a training data set $\left\{ ( x _ { 1 } , y _ { 1 } ) , \dotsc , ( x _ { N } , y _ { N } ) \right\}$ we recall the results from Section 9.2.1 and obtain the maximum likelihood estimator for the slope parameter as 

$$
\theta_ {\mathrm{ML}} = \left(\boldsymbol {X} ^ {\top} \boldsymbol {X}\right) ^ {- 1} \boldsymbol {X} ^ {\top} \boldsymbol {y} = \frac {\boldsymbol {X} ^ {\top} \boldsymbol {y}}{\boldsymbol {X} ^ {\top} \boldsymbol {X}} \in \mathbb {R},\tag{9.66}
$$

where $\begin{array} { r } { \pmb { X } = [ x _ { 1 } , \dotsc , x _ { N } ] ^ { \top } \in \mathbb { R } ^ { N } , \pmb { y } = [ y _ { 1 } , \dotsc , y _ { N } ] ^ { \top } \in \mathbb { R } ^ { N } . } \end{array}$ 

This means for the training inputs X we obtain the optimal (maximum likelihood) reconstruction of the training targets as 

$$
\pmb {X} \theta_ {\mathrm{ML}} = \pmb {X} \frac {\pmb {X} ^ {\top} \pmb {y}}{\pmb {X} ^ {\top} \pmb {X}} = \frac {\pmb {X} \pmb {X} ^ {\top}}{\pmb {X} ^ {\top} \pmb {X}} \pmb {y},\tag{9.67}
$$



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



i.e., we obtain the approximation with the minimum least-squares error between y and $X \theta .$ 

As we are looking for a solution of $y = X \theta$ , we can think of linear regression as a problem for solving systems of linear equations. Therefore, we can relate to concepts from linear algebra and analytic geometry that we discussed in Chapters 2 and 3. In particular, looking carefully at (9.67) we see that the maximum likelihood estimator $\theta _ { \mathrm { M L } }$ in our example from (9.65) effectively does an orthogonal projection of $\textbf {  { y } }$ onto the one-dimensional subspace spanned by $\boldsymbol { X }$ . Recalling the results on orthogonal projections from Section 3.8, we identify $\frac { x x ^ { \top } } { X ^ { \top } X }$ as the projection matrix, $\theta _ { \mathrm { M L } }$ as the coordinates of the projection onto the one-dimensional subspace of $\mathbb { R } ^ { N }$ spanned by X and $\pmb { X \theta } _ { \mathrm { M L } }$ as the orthogonal projection of y onto this subspace. 

Therefore, the maximum likelihood solution provides also a geometrically optimal solution by finding the vectors in the subspace spanned by X that are “closest” to the corresponding observations y, where “closest” means the smallest (squared) distance of the function values $y _ { n }$ to $x _ { n } \theta$ . This is achieved by orthogonal projections. Figure 9.12(b) shows the projection of the noisy observations onto the subspace that minimizes the squared distance between the original dataset and its projection (note that the x-coordinate is fixed), which corresponds to the maximum likelihood solution. 

In the general linear regression case where 

$$
y = \boldsymbol {\phi} ^ {\top} (\boldsymbol {x}) \boldsymbol {\theta} + \epsilon , \quad \epsilon \sim \mathcal {N} (0, \sigma^ {2})\tag{9.68}
$$

with vector-valued features $\pmb { \phi } ( \pmb { x } ) \in \mathbb { R } ^ { K }$ , we again can interpret the maximum likelihood result 

$$
\pmb {y} \approx \Phi \pmb {\theta} _ {\mathrm{ML}},\tag{9.69}
$$

$$
\boldsymbol {\theta} _ {\mathrm{ML}} = \left(\boldsymbol {\Phi} ^ {\top} \boldsymbol {\Phi}\right) ^ {- 1} \boldsymbol {\Phi} ^ {\top} \boldsymbol {y}\tag{9.70}
$$

as a projection onto a K-dimensional subspace of $\mathbb { R } ^ { N }$ , which is spanned by the columns of the feature matrix $\Phi ;$ see Section 3.8.2. 

If the feature functions $\phi _ { k }$ that we use to construct the feature matrix Φ are orthonormal (see Section 3.7), we obtain a special case where the columns of Φ form an orthonormal basis (see Section 3.5), such that $\Phi ^ { \top } \Phi = I$ . This will then lead to the projection 

$$
\boldsymbol {\Phi} \left(\boldsymbol {\Phi} ^ {\top} \boldsymbol {\Phi}\right) ^ {- 1} \boldsymbol {\Phi} ^ {\top} \boldsymbol {y} = \boldsymbol {\Phi} \boldsymbol {\Phi} ^ {\top} \boldsymbol {y} = \left(\sum_ {k = 1} ^ {K} \phi_ {k} \phi_ {k} ^ {\top}\right) \boldsymbol {y}\tag{9.71}
$$

so that the maximum likelihood projection is simply the sum of projections of $\textbf {  { y } }$ onto the individual basis vectors $\phi _ { k } , \mathrm { i . e . }$ , the columns of $\Phi$ . Furthermore, the coupling between different features has disappeared due to the orthogonality of the basis. Many popular basis functions in signal processing, such as wavelets and Fourier bases, are orthogonal basis functions. 

## 9.5 Further Reading

When the basis is not orthogonal, one can convert a set of linearly inde pendent basis functions to an orthogonal basis by using the Gram-Schmidt process; see Section 3.8.3 and (Strang, 2003). 

## 9.5 Further Reading

In this chapter, we discussed linear regression for Gaussian likelihoods and conjugate Gaussian priors on the parameters of the model. This allowed for closed-form Bayesian inference. However, in some applications we may want to choose a different likelihood function. For example, in a binary classification setting, we observe only two possible (categorical) outcomes, and a Gaussian likelihood is inappropriate in this setting. Instead, we can choose a Bernoulli likelihood that will return a probability of the predicted label to be 1 (or 0). We refer to the books by Barber (2012), Bishop (2006), and Murphy (2012) for an in-depth introduction to classification problems. A different example where non-Gaussian likelihoods are important is count data. Counts are non-negative integers, and in this case a Binomial or Poisson likelihood would be a better choice than a Gaussian. All these examples fall into the category of generalized linear models, a flexible generalization of linear regression that allows for response variables that have error distributions other than a Gaussian distribution. The GLM generalizes linear regression by allowing the linear model to be related to the observed values via a smooth and invertible function $\sigma ( \cdot )$ that may be nonlinear so that $y = \sigma ( f ( { \pmb x } ) )$ ), where $f ( { \pmb x } ) = { \pmb \theta } ^ { \top } \phi ( { \pmb x } )$ is the linear regression model from (9.13). We can therefore think of a generalized linear model in terms of function composition $y = \sigma \circ f ,$ , where $f$ is a linear regression model and σ the activation function. Note that although we are talking about “generalized linear models”, the outputs y are no longer linear in the parameters θ. In logistic regression, we choose the logistic sigmoid $\begin{array} { r } { \sigma ( f ) = \frac { 1 } { 1 + \exp ( - f ) } \in [ 0 , 1 ] } \end{array}$ ], which can be interpreted as the probability of observing $y = 1$ of a Bernoulli random variable $y \in \{ 0 , 1 \}$ The function $\sigma ( \cdot )$ is called transfer function or activation function, and its inverse is called the canonical link function. From this perspective, it is also clear that generalized linear models are the building blocks of (deep) feedforward neural networks: If we consider a generalized linear model $\pmb { y } = \sigma ( \pmb { A } \pmb { x } + \pmb { b } )$ , where A is a weight matrix and b a bias vector, we identify this generalized linear model as a single-layer neural network with activation function $\sigma ( \cdot )$ . We can now recursively compose these functions via 

$$
\begin{array}{c} \boldsymbol {x} _ {k + 1} = \boldsymbol {f} _ {k} (\boldsymbol {x} _ {k}) \\ \boldsymbol {f} _ {k} (\boldsymbol {x} _ {k}) = \sigma_ {k} (\boldsymbol {A} _ {k} \boldsymbol {x} _ {k} + \boldsymbol {b} _ {k}) \end{array}\tag{9.72}
$$

for $k = 0 , \ldots , K - 1$ , where $\scriptstyle { \pmb x } _ { 0 }$ are the input features and $\scriptstyle { \pmb x } _ { K } \ = \ { \pmb y }$ are the observed outputs, such that $\pmb { f } _ { K - 1 } \circ \cdots \circ \pmb { f } _ { 0 }$ is a K-layer deep neural network. Therefore, the building blocks of this deep neural network are the generalized linear models defined in (9.72). Neural networks (Bishop, 1995; Goodfellow et al., 2016) are significantly more expressive and flexible than linear regression models. However, maximum likelihood parameter estimation is a non-convex optimization problem, and marginalization of the parameters in a fully Bayesian setting is analytically intractable. 

We briefly hinted at the fact that a distribution over parameters induces a distribution over regression functions. Gaussian processes (Rasmussen and Williams, 2006) are regression models where the concept of a distribution over function is central. Instead of placing a distribution over parameters, a Gaussian process places a distribution directly on the space of functions without the “detour” via the parameters. To do so, the Gaussian process exploits the kernel trick (Scholkopf and Smola¨ , 2002), which allows us to compute inner products between two function values $f ( \pmb { x } _ { i } ) , f ( \pmb { x } _ { j } )$ only by looking at the corresponding input $\mathbf { \Delta } _ { x _ { i } , \mathbf { \Delta } _ { x _ { j } } }$ . A Gaussian process is closely related to both Bayesian linear regression and support vector regression but can also be interpreted as a Bayesian neural network with a single hidden layer where the number of units tends to infinity (Neal, 1996; Williams, 1997). Excellent introductions to Gaussian processes can be found in MacKay (1998) and Rasmussen and Williams (2006). 

We focused on Gaussian parameter priors in the discussions in this chapter, because they allow for closed-form inference in linear regression models. However, even in a regression setting with Gaussian likelihoods, we may choose a non-Gaussian prior. Consider a setting, where the inputs are $\pmb { x } \in \mathbb { R } ^ { D }$ and our training set is small and of size $N \ll D$ . This means that the regression problem is underdetermined. In this case, we can choose a parameter prior that enforces sparsity, i.e., a prior that tries to set as many parameters to 0 as possible (variable selection). This prior provides a stronger regularizer than the Gaussian prior, which often leads to an increased prediction accuracy and interpretability of the model. The Laplace prior is one example that is frequently used for this purpose. A linear regression model with the Laplace prior on the parameters is equivalent to linear regression with L1 regularization (LASSO) (Tibshirani, 1996). The Laplace distribution is sharply peaked at zero (its first derivative is discontinuous) and it concentrates its probability mass closer to zero than the Gaussian distribution, which encourages parameters to be 0. Therefore, the nonzero parameters are relevant for the regression problem, which is the reason why we also speak of “variable selection”. 

## Dimensionality Reduction with Principal Component Analysis

Working directly with high-dimensional data, such as images, comes with some difficulties: It is hard to analyze, interpretation is difficult, visualization is nearly impossible, and (from a practical point of view) storage of the data vectors can be expensive. However, high-dimensional data often has properties that we can exploit. For example, high-dimensional data is often overcomplete, i.e., many dimensions are redundant and can be ex plained by a combination of other dimensions. Furthermore, dimensions in high-dimensional data are often correlated so that the data possesses an intrinsic lower-dimensional structure. Dimensionality reduction exploits structure and correlation and allows us to work with a more compact representation of the data, ideally without losing information. We can think of dimensionality reduction as a compression technique, similar to jpeg or mp3, which are compression algorithms for images and music. 

In this chapter, we will discuss principal component analysis (PCA), an algorithm for linear dimensionality reduction. PCA, proposed by Pearson (1901) and Hotelling (1933), has been around for more than 100 years and is still one of the most commonly used techniques for data compression and data visualization. It is also used for the identification of simple patterns, latent factors, and structures of high-dimensional data. In the 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/b87ad0ae70ac96f1ef6f414570322a317eae4e212bd2c21df7ce5dfe86425ca5.jpg)



(a) Dataset with x and x coordinates.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/18ededb03a41ff345ea8f06ee57323384f4f8ebbabe44f1cc8bd23e30b780223.jpg)



(b) Compressed dataset where only the x coordinate is relevant.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/3b8fced6d041bcba427032ff9699fa728829941c5f8935d1aa459d83e898239a.jpg)



A 640 × 480 pixel color image is a data point in a million-dimensiona space, where every pixel responds to three dimensions, one for each color channel (red, green, blue).


Figure 10.1 Illustration: dimensionality reduction. (a) The original dataset does not vary much along the x direction. (b) The data from (a) can be represented using the x -coordinate alone with nearly no loss. 

signal processing community, PCA is also known as the Karhunen-Lo`eve transform. In this chapter, we derive PCA from first principles, drawing on our understanding of basis and basis change (Sections 2.6.1 and 2.7.2), projections (Section 3.8), eigenvalues (Section 4.2), Gaussian distributions (Section 6.5), and constrained optimization (Section 7.2). 

Dimensionality reduction generally exploits a property of high-dimensional data (e.g., images) that it often lies on a low-dimensional subspace. Figure 10.1 gives an illustrative example in two dimensions. Although the data in Figure 10.1(a) does not quite lie on a line, the data does not vary much in the x -direction, so that we can express it as if it were on a line – with nearly no loss; see Figure 10.1(b). To describe the data in Figure 10.1(b), only the $x _ { 1 }$ -coordinate is required, and the data lies in a one-dimensional subspace of $\mathbb { R } ^ { 2 }$ 

## 10.1 Problem Setting

In PCA, we are interested in finding projections $\scriptstyle { \tilde { \mathbf { x } } } _ { n }$ of data points ${ \bf { x } } _ { n }$ that are as similar to the original data points as possible, but which have a significantly lower intrinsic dimensionality. Figure 10.1 gives an illustration of what this could look like. 

More concretely, we consider an i.i.d. dataset ${ \mathcal { X } } = \{ { \pmb x } _ { 1 } , \ldots , { \pmb x } _ { N } \} , { \pmb x } _ { n } \in$ $\mathbb { R } ^ { D }$ , with mean 0 that possesses the data covariance matrix (6.42) 

$$
\boldsymbol {S} = \frac {1}{N} \sum_ {n = 1} ^ {N} \boldsymbol {x} _ {n} \boldsymbol {x} _ {n} ^ {\top}.\tag{10.1}
$$

Furthermore, we assume there exists a low-dimensional compressed representation (code) 

$$
\pmb {z} _ {n} = \pmb {B} ^ {\top} \pmb {x} _ {n} \in \mathbb {R} ^ {M}\tag{10.2}
$$

of ${ \pmb x } _ { n }$ , where we define the projection matrix 

$$
\boldsymbol {B} := \left[ \boldsymbol {b} _ {1}, \dots , \boldsymbol {b} _ {M} \right] \in \mathbb {R} ^ {D \times M}.\tag{10.3}
$$

We assume that the columns of B are orthonormal (Definition 3.7) so that $\pmb { b } _ { i } ^ { \top } \pmb { b } _ { j } = 0$ if and only if $i \neq j$ and $\pmb { b } _ { i } ^ { \top } \pmb { b } _ { i } = 1$ . We seek an M-dimensional subspace $U \subseteq \mathbb { R } ^ { D } , \mathrm { d i m } ( U ) = M < \dot { D }$ onto which we project the data. We denote the projected data by $\tilde { \pmb { x } } _ { n } \in U$ , and their coordinates (with respect to the basis vectors $b _ { 1 } , \dots , b _ { M }$ of U) by $z _ { n }$ . Our aim is to find projections $\tilde { \pmb { x } } _ { n } \in \mathbb { R } ^ { D }$ (or equivalently the codes $z _ { n }$ and the basis vectors $b _ { 1 } , \dotsc , b _ { M } )$ so that they are as similar to the original data ${ \bf { x } } _ { n }$ and minimize the loss due to compression. 

## Example 10.1 (Coordinate Representation/Code)

Consider $\mathbb { R } ^ { 2 }$ with the canonical basis $\mathbf { e } _ { 1 } = [ 1 , 0 ] ^ { \top } , \mathbf { e } _ { 2 } = [ 0 , 1 ] ^ { \top }$ . From 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/5761c818093bc7024b1d5724bd43d984b2aaeec552088d371f92deedae841f0a.jpg)



Figure 10.2 Graphical illustration of PCA. In PCA, we find a compressed version z of original data x The compressed data can be reconstructed into x˜, which lives in the original data space, but has an intrinsic lower-dimensional representation than x.


Chapter 2, we know that $\pmb { x } \in \mathbb { R } ^ { 2 }$ can be represented as a linear combination of these basis vectors, e.g., 

$$
\left[ \begin{array}{c} 5 \\ 3 \end{array} \right] = 5 e _ {1} + 3 e _ {2}.\tag{10.4}
$$

However, when we consider vectors of the form 

$$
\tilde {\boldsymbol {x}} = \left[ \begin{array}{c} 0 \\ z \end{array} \right] \in \mathbb {R} ^ {2}, \quad z \in \mathbb {R},\tag{10.5}
$$

they can always be written as $0 e _ { 1 } + z e _ { 2 }$ . To represent these vectors it is sufficient to remember/store the coordinate/code z of x˜ with respect to the $e _ { 2 }$ vector. 

More precisely, the set of x˜ vectors (with the standard vector addition and scalar multiplication) forms a vector subspace U (see Section 2.4) with dim $( U ) = 1$ because $U = \mathrm { s p a n } [ e _ { 2 } ]$ 

In Section 10.2, we will find low-dimensional representations that retain as much information as possible and minimize the compression loss. An alternative derivation of PCA is given in Section 10.3, where we will be looking at minimizing the squared reconstruction error $\| \pmb { x } _ { n } - \tilde { \pmb { x } } _ { n } \| ^ { 2 }$ between the original data ${ \bf { x } } _ { n }$ and its projection ${ \tilde { \mathbf { x } } } _ { n }$ 

Figure 10.2 illustrates the setting we consider in PCA, where z represents the lower-dimensional representation of the compressed data x˜ and plays the role of a bottleneck, which controls how much information can flow between x and x˜. In PCA, we consider a linear relationship between the original data x and its low-dimensional code z so that $z = B ^ { \intercal }$ x and $\tilde { \boldsymbol { x } } = B \boldsymbol { z }$ for a suitable matrix B. Based on the motivation of thinking of PCA as a data compression technique, we can interpret the arrows in Figure 10.2 as a pair of operations representing encoders and decoders. The linear mapping represented by B can be thought of as a decoder, which maps the low-dimensional code $\boldsymbol { z } \in \mathbb { R } ^ { M }$ back into the original data space $\mathbb { R } ^ { D }$ . Similarly, $B ^ { \top }$ can be thought of an encoder, which encodes the original data x as a low-dimensional (compressed) code z. 

Throughout this chapter, we will use the MNIST digits dataset as a re-

## 0123956389

occurring example, which contains 60,000 examples of handwritten digits 0 through 9. Each digit is a grayscale image of size 28 28, i.e., it contains 784 pixels so that we can interpret every image in this dataset as a vector $\pmb { x } \in \mathbb { R } ^ { 7 8 4 }$ . Examples of these digits are shown in Figure 10.3. 

## 10.2 Maximum Variance Perspective

Figure 10.1 gave an example of how a two-dimensional dataset can be represented using a single coordinate. In Figure 10.1(b), we chose to ignore the $x _ { 2 } .$ -coordinate of the data because it did not add too much information so that the compressed data is similar to the original data in Figure 10.1(a). We could have chosen to ignore the $x _ { 1 }$ -coordinate, but then the compressed data had been very dissimilar from the original data, and much information in the data would have been lost. 

If we interpret information content in the data as how “space filling” the dataset is, then we can describe the information contained in the data by looking at the spread of the data. From Section 6.4.1, we know that the variance is an indicator of the spread of the data, and we can derive PCA as a dimensionality reduction algorithm that maximizes the variance in the low-dimensional representation of the data to retain as much information as possible. Figure 10.4 illustrates this. 

Considering the setting discussed in Section 10.1, our aim is to find a matrix B (see (10.3)) that retains as much information as possible when compressing data by projecting it onto the subspace spanned by the columns $b _ { 1 } , \dots , b _ { M }$ of B. Retaining most information after data compression is equivalent to capturing the largest amount of variance in the low-dimensional code (Hotelling, 1933). 

Remark. (Centered Data) For the data covariance matrix in (10.1), we assumed centered data. We can make this assumption without loss of generality: Let us assume that $\pmb { \mu }$ is the mean of the data. Using the properties of the variance, which we discussed in Section 6.4.4, we obtain 

$$
\mathbb {V} _ {\boldsymbol {z}} [ \boldsymbol {z} ] = \mathbb {V} _ {\boldsymbol {x}} [ \boldsymbol {B} ^ {\top} (\boldsymbol {x} - \boldsymbol {\mu}) ] = \mathbb {V} _ {\boldsymbol {x}} [ \boldsymbol {B} ^ {\top} \boldsymbol {x} - \boldsymbol {B} ^ {\top} \boldsymbol {\mu} ] = \mathbb {V} _ {\boldsymbol {x}} [ \boldsymbol {B} ^ {\top} \boldsymbol {x} ],\tag{10.6}
$$

i.e., the variance of the low-dimensional code does not depend on the mean of the data. Therefore, we assume without loss of generality that the data has mean 0 for the remainder of this section. With this assumption the mean of the low-dimensional code is also 0 since $\begin{array} { r } { \mathbb { E } _ { z } [ z ] = \mathbb { E } _ { x } [ B ^ { \top } x ] = } \end{array}$ $\boldsymbol { B } ^ { \intercal } \mathbb { E } _ { \boldsymbol { x } } [ \boldsymbol { x } ] = \mathbf { 0 }$ ◇ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/0978b26bf36b0195507a15128ce1892a2c083741f446fe68f741715aa9335d44.jpg)



Figure 10.4 PCA finds a lower-dimensional subspace (line) that maintains as much variance (spread of the data) as possible when the data (blue) is projected onto this subspace (orange).


10.2.1 Direction with Maximal Variance 

We maximize the variance of the low-dimensional code using a sequential approach. We start by seeking a single vector $\pmb { b } _ { 1 } \in \mathbb { R } ^ { D }$ that maximizes the variance of the projected data, i.e., we aim to maximize the variance of the first coordinate $z _ { 1 }$ of $\boldsymbol { z } \in \mathbb { R } ^ { M }$ so that 

$$
V _ {1} := \mathbb {V} [ z _ {1} ] = \frac {1}{N} \sum_ {n = 1} ^ {N} z _ {1 n} ^ {2}\tag{10.7}
$$

is maximized, where we exploited the i.i.d. assumption of the data and defined $z _ { 1 n }$ as the first coordinate of the low-dimensional representation $\pmb { z } _ { n } \in \mathbb { R } ^ { M }$ of ${ \pmb x } _ { n } \in \mathbb { R } ^ { D }$ . Note that first component of $z _ { n }$ is given by 

$$
\boldsymbol {z} _ {1 n} = \boldsymbol {b} _ {1} ^ {\top} \boldsymbol {x} _ {n},\tag{10.8}
$$

i.e., it is the coordinate of the orthogonal projection of ${ \bf { x } } _ { n }$ onto the onedimensional subspace spanned by $b _ { 1 }$ (Section 3.8). We substitute (10.8) into (10.7), which yields 

$$
\begin{array}{l} V _ {1} = \frac {1}{N} \sum_ {n = 1} ^ {N} (\boldsymbol {b} _ {1} ^ {\top} \boldsymbol {x} _ {n}) ^ {2} = \frac {1}{N} \sum_ {n = 1} ^ {N} \boldsymbol {b} _ {1} ^ {\top} \boldsymbol {x} _ {n} \boldsymbol {x} _ {n} ^ {\top} \boldsymbol {b} _ {1} \\ = \boldsymbol {b} _ {1} ^ {\top} \left(\frac {1}{N} \sum_ {n = 1} ^ {N} \boldsymbol {x} _ {n} \boldsymbol {x} _ {n} ^ {\top}\right) \boldsymbol {b} _ {1} = \boldsymbol {b} _ {1} ^ {\top} \boldsymbol {S b} _ {1}, \end{array}\tag{10.9a}
$$

(10.9b) 

where $\pmb { S }$ is the data covariance matrix defined in (10.1). In (10.9a), we have used the fact that the dot product of two vectors is symmetric with respect to its arguments, that is, $\bar { \pmb { b } } _ { 1 } ^ { \top } \pmb { x } _ { n } = \pmb { x } _ { n } ^ { \top } \pmb { b } _ { 1 }$ 

Notice that arbitrarily increasing the magnitude of the vector $b _ { 1 }$ increases $V _ { 1 }$ , that is, a vector $b _ { 1 }$ that is two times longer can result in $V _ { 1 }$ that is potentially four times larger. Therefore, we restrict all solutions to $\left. \boldsymbol { b } _ { 1 } \right. ^ { 2 } \bar { = } 1$ , which results in a constrained optimization problem in which we seek the direction along which the data varies most. 

With the restriction of the solution space to unit vectors the vector $b _ { 1 }$ that points in the direction of maximum variance can be found by the 

principal component 

constrained optimization problem 

$$
\begin{array}{l} \underset {\boldsymbol {b} _ {1}} {\max} \boldsymbol {b} _ {1} ^ {\top} \boldsymbol {S} \boldsymbol {b} _ {1} \\ \text {subject to} \| \boldsymbol {b} _ {1} \| ^ {2} = 1. \end{array}\tag{10.10}
$$

Following Section 7.2, we obtain the Lagrangian 

$$
\mathfrak {L} (\boldsymbol {b} _ {1}, \lambda) = \boldsymbol {b} _ {1} ^ {\top} \boldsymbol {S} \boldsymbol {b} _ {1} + \lambda_ {1} (1 - \boldsymbol {b} _ {1} ^ {\top} \boldsymbol {b} _ {1})\tag{10.11}
$$

to solve this constrained optimization problem. The partial derivatives of L with respect to $b _ { 1 }$ and $\lambda _ { 1 }$ are 

$$
\frac {\partial \mathfrak {L}}{\partial \boldsymbol {b} _ {1}} = 2 \boldsymbol {b} _ {1} ^ {\top} \boldsymbol {S} - 2 \lambda_ {1} \boldsymbol {b} _ {1} ^ {\top}, \quad \frac {\partial \mathfrak {L}}{\partial \lambda_ {1}} = 1 - \boldsymbol {b} _ {1} ^ {\top} \boldsymbol {b} _ {1},\tag{10.12}
$$

respectively. Setting these partial derivatives to 0 gives us the relations 

$$
\boldsymbol {S} \boldsymbol {b} _ {1} = \lambda_ {1} \boldsymbol {b} _ {1},\tag{10.13}
$$

$$
\boldsymbol {b} _ {1} ^ {\top} \boldsymbol {b} _ {1} = 1.\tag{10.14}
$$

By comparing this with the definition of an eigenvalue decomposition (Section 4.4), we see that $b _ { 1 }$ is an eigenvector of the data covariance matrix ${ \cal S } ,$ and the Lagrange multiplier $\lambda _ { 1 }$ plays the role of the corresponding eigenvalue. This eigenvector property (10.13) allows us to rewrite our variance objective (10.10) as 

$$
V _ {1} = \boldsymbol {b} _ {1} ^ {\top} \boldsymbol {S} \boldsymbol {b} _ {1} = \lambda_ {1} \boldsymbol {b} _ {1} ^ {\top} \boldsymbol {b} _ {1} = \lambda_ {1},\tag{10.15}
$$

i.e., the variance of the data projected onto a one-dimensional subspace equals the eigenvalue that is associated with the basis vector $b _ { 1 }$ that spans this subspace. Therefore, to maximize the variance of the low-dimensional code, we choose the basis vector associated with the largest eigenvalue of the data covariance matrix. This eigenvector is called the first principa component. We can determine the effect/contribution of the principal component $b _ { 1 }$ in the original data space by mapping the coordinate $z _ { 1 n }$ back into data space, which gives us the projected data point 

$$
\tilde {\pmb {x}} _ {n} = \pmb {b} _ {1} z _ {1 n} = \pmb {b} _ {1} \pmb {b} _ {1} ^ {\top} \pmb {x} _ {n} \in \mathbb {R} ^ {D}\tag{10.16}
$$

in the original data space. 

Remark. Although $\tilde { \mathbf { x } } _ { n }$ is a D-dimensional vector, it only requires a single coordinate $z _ { 1 n }$ to represent it with respect to the basis vector $\pmb { b } _ { 1 } \in \mathbb { R } ^ { D }$ . 

## 10.2.2 M-dimensional Subspace with Maximal Variance

Assume we have found the first $m - 1$ principal components as the $m - 1$ eigenvectors of S that are associated with the largest m 1 eigenvalues. Since S is symmetric, the spectral theorem (Theorem 4.15) states that we can use these eigenvectors to construct an orthonormal eigenbasis of an $( m - 1 )$ )-dimensional subspace of $\mathbb { R } ^ { D }$ . Generally, the mth principal component can be found by subtracting the effect of the first $m - 1$ principal components $b _ { 1 } , \ldots , b _ { m - 1 }$ from the data, thereby trying to find principal components that compress the remaining information. We then arrive at the new data matrix 

$$
\hat {\boldsymbol {X}} := \boldsymbol {X} - \sum_ {i = 1} ^ {m - 1} \boldsymbol {b} _ {i} \boldsymbol {b} _ {i} ^ {\top} \boldsymbol {X} = \boldsymbol {X} - \boldsymbol {B} _ {m - 1} \boldsymbol {X},\tag{10.17}
$$

where $\pmb { X } = [ \pmb { x } _ { 1 } , \dots , \pmb { x } _ { N } ] \in \mathbb { R } ^ { D \times N }$ contains the data points as column vectors and $\begin{array} { r } { \pmb { { B } } _ { m - 1 } : = \sum _ { i = 1 } ^ { m - 1 } \pmb { b } _ { i } \pmb { b } _ { i } ^ { \top } } \end{array}$ is a projection matrix that projects onto the subspace spanned by $b _ { 1 } , \ldots , b _ { m - 1 }$ 

Remark (Notation). Throughout this chapter, we do not follow the convention of collecting data $\pmb { x } _ { 1 } , \ldots , \pmb { x } _ { N }$ as the rows of the data matrix, but we define them to be the columns of $\boldsymbol { X }$ . This means that our data matrix X is a $D \times N$ matrix instead of the conventional $N \times D$ matrix. The reason for our choice is that the algebra operations work out smoothly without the need to either transpose the matrix or to redefine vectors as row vectors that are left-multiplied onto matrices. 

To find the mth principal component, we maximize the variance 

$$
V _ {m} = \mathbb {V} [ z _ {m} ] = \frac {1}{N} \sum_ {n = 1} ^ {N} z _ {m n} ^ {2} = \frac {1}{N} \sum_ {n = 1} ^ {N} (\pmb {b} _ {m} ^ {\top} \hat {\pmb {x}} _ {n}) ^ {2} = \pmb {b} _ {m} ^ {\top} \hat {\pmb {S}} \pmb {b} _ {m},\tag{10.18}
$$

subject to $\left\| \pmb { b } _ { m } \right\| ^ { 2 } = 1$ , where we followed the same steps as in (10.9b) and defined S<sup>ˆ</sup> as the data covariance matrix of the transformed dataset $\hat { \mathcal X } : = \{ \hat { \pmb x } _ { 1 } , \hdots , \hat { \pmb x } _ { N } \}$ . As previously, when we looked at the first principal component alone, we solve a constrained optimization problem and discover that the optimal solution $b _ { m }$ is the eigenvector of $\hat { \boldsymbol { S } }$ that is associated with the largest eigenvalue of $\hat { S }$ . 

It turns out that $b _ { m }$ is also an eigenvector of $S .$ . More generally, the sets of eigenvectors of $_ { s }$ and $\hat { \boldsymbol S }$ are identical. Since both S and $\hat { \boldsymbol { S } }$ are symmetric, we can find an ONB of eigenvectors (spectral theorem 4.15), i.e., there exist $D$ distinct eigenvectors for both $_ { s }$ and $\hat { \boldsymbol { S } } .$ . Next, we show that every eigenvector of S is an eigenvector of $\hat { \boldsymbol { S } } .$ . Assume we have already found eigenvectors $b _ { 1 } , \ldots , b _ { m - 1 }$ of $\hat { \boldsymbol { S } } .$ . Consider an eigenvector $\mathbf { } _ { b _ { i } }$ of $S _ { z }$ i.e., $S b _ { i } = \lambda _ { i } \pmb { b } _ { i }$ . In general, 

$$
\hat {\boldsymbol {S}} \boldsymbol {b} _ {i} = \frac {1}{N} \hat {\boldsymbol {X}} \hat {\boldsymbol {X}} ^ {\top} \boldsymbol {b} _ {i} = \frac {1}{N} (\boldsymbol {X} - \boldsymbol {B} _ {m - 1} \boldsymbol {X}) (\boldsymbol {X} - \boldsymbol {B} _ {m - 1} \boldsymbol {X}) ^ {\top} \boldsymbol {b} _ {i}\tag{10.19a}
$$

$$
= \left(\boldsymbol {S} - \boldsymbol {S} \boldsymbol {B} _ {m - 1} - \boldsymbol {B} _ {m - 1} \boldsymbol {S} + \boldsymbol {B} _ {m - 1} \boldsymbol {S} \boldsymbol {B} _ {m - 1}\right) \boldsymbol {b} _ {i}.\tag{10.19b}
$$

We distinguish between two cases. $\mathrm { I f } ~ i ~ \geqslant ~ m , ~ \mathrm { i . e . , } ~ b _ { i }$ is an eigenvector that is not among the first $m - 1$ principal components, then $b _ { i }$ is orthogonal to the first $m - 1$ principal components and $B _ { m - 1 } b _ { i } = 0 . { \mathrm { I f } } i < m , { \mathrm { i . e . } } ,$ $\mathbf { } _ { b _ { i } }$ is among the first $m - 1$ principal components, then $\mathbf { } _ { b _ { i } }$ is a basis vector of the principal subspace onto which $B _ { m - 1 }$ projects. Since $b _ { 1 } , \ldots , b _ { m - 1 }$ are an ONB of this principal subspace, we obtain $B _ { m - 1 } \pmb { b } _ { i } = \pmb { b } _ { i }$ . The two cases can be summarized as follows: 

$$
\boldsymbol {B} _ {m - 1} \boldsymbol {b} _ {i} = \boldsymbol {b} _ {i} \quad \text {if} i <   m, \qquad \boldsymbol {B} _ {m - 1} \boldsymbol {b} _ {i} = \mathbf {0} \quad \text {if} i \geqslant m.\tag{10.20}
$$

In the case $i \geqslant m ,$ by using (10.20) in (10.19b), we obtain $\hat { S } b _ { i } = ( S -$ $B _ { m - 1 } { \cal S } ) b _ { i } = { \cal S } b _ { i } = \lambda _ { i } b _ { i }$ , i.e., b is also an eigenvector of $\hat { \boldsymbol { S } }$ with eigenvalue $\lambda _ { i }$ . Specifically, 

$$
\hat {\pmb {S}} \pmb {b} _ {m} = \pmb {S} \pmb {b} _ {m} = \lambda_ {m} \pmb {b} _ {m}.\tag{10.21}
$$

Equation (10.21) reveals that $b _ { m }$ is not only an eigenvector of S but also of $\hat { \boldsymbol { S } } .$ . Specifically, $\lambda _ { m }$ is the largest eigenvalue of $\hat { \boldsymbol { S } }$ and $\lambda _ { m }$ is the mth largest eigenvalue of S, and both have the associated eigenvector $b _ { m }$ 

In the case $i < m ,$ by using (10.20) in (10.19b), we obtain 

$$
\hat {\boldsymbol {S}} \boldsymbol {b} _ {i} = (\boldsymbol {S} - \boldsymbol {S} \boldsymbol {B} _ {m - 1} - \boldsymbol {B} _ {m - 1} \boldsymbol {S} + \boldsymbol {B} _ {m - 1} \boldsymbol {S} \boldsymbol {B} _ {m - 1}) \boldsymbol {b} _ {i} = \boldsymbol {0} = 0 \boldsymbol {b} _ {i}\tag{10.22}
$$

This means that $b _ { 1 } , \dotsc , b _ { m - 1 }$ are also eigenvectors of $\hat { \cal S } ,$ , but they are associated with eigenvalue 0 so that $b _ { 1 } , \dotsc , b _ { m - 1 }$ span the null space of $\hat { \boldsymbol S }$ . 

Overall, every eigenvector of S is also an eigenvector of $\hat { \boldsymbol { S } } .$ . However, if the eigenvectors of S are part of the $( m - 1 )$ dimensional principal subspace, then the associated eigenvalue of $\hat { \boldsymbol { S } }$ is 0. 

With the relation (10.21) and $\pmb { b } _ { m } ^ { \top } \pmb { b } _ { m } = 1$ , the variance of the data projected onto the mth principal component is 

$$
V _ {m} = \boldsymbol {b} _ {m} ^ {\top} \boldsymbol {S} \boldsymbol {b} _ {m} \stackrel {(1 0. 2 1)} {=} \lambda_ {m} \boldsymbol {b} _ {m} ^ {\top} \boldsymbol {b} _ {m} = \lambda_ {m}.\tag{10.23}
$$

This means that the variance of the data, when projected onto an $M -$ dimensional subspace, equals the sum of the eigenvalues that are associated with the corresponding eigenvectors of the data covariance matrix. 

## Example 10.2 (Eigenvalues of MNIST “8”)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/f236f3d6bd0d26c9924bb66e9eb11152f8082b52beb93f8c4fb6e795503c10c0.jpg)



(a) Eigenvalues (sorted in descending order) of the data covariance matrix of all digits $\mathbf { \boldsymbol { \mathfrak { s } } } ^ { \flat }$ in the MNIST training set.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/2fc58958dcadd22ce74862faf6436f591953b4d43a629f34cefd99073558f2b1.jpg)



(b) Variance captured by the principal components.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/ce9f199755c0cc3fcfcc16346119a9ba1da9c11b6fa89e4f7ae8d0c8dcdbb8cd.jpg)



Figure 10.6 Illustration of the projection approach: Find a subspace (line) that minimizes the length of the difference vector between projected (orange) and original (blue) data.


Taking all digits $" 8 "$ in the MNIST training data, we compute the eigenvalues of the data covariance matrix. Figure 10.5(a) shows the 200 largest eigenvalues of the data covariance matrix. We see that only a few of them have a value that differs significantly from 0. Therefore, most of the variance, when projecting data onto the subspace spanned by the corresponding eigenvectors, is captured by only a few principal components, as shown in Figure 10.5(b). 

Overall, to find an M-dimensional subspace of $\mathbb { R } ^ { D }$ that retains as much information as possible, PCA tells us to choose the columns of the matrix B in (10.3) as the M eigenvectors of the data covariance matrix S that are associated with the M largest eigenvalues. The maximum amount of variance PCA can capture with the first M principal components is 

$$
V _ {M} = \sum_ {m = 1} ^ {M} \lambda_ {m},\tag{10.24}
$$

where the $\lambda _ { m }$ are the M largest eigenvalues of the data covariance matrix S. Consequently, the variance lost by data compression via PCA is 

$$
J _ {M} := \sum_ {j = M + 1} ^ {D} \lambda_ {j} = V _ {D} - V _ {M}.\tag{10.25}
$$

Instead of these absolute quantities, we can define the relative variance captured as $\frac { V _ { M } } { V _ { D } }$ , and the relative variance lost by compression as $\begin{array} { r } { 1 - \frac { V _ { M } } { V _ { D } } } \end{array}$ 

## 10.3 Projection Perspective

In the following, we will derive PCA as an algorithm that directly minimizes the average reconstruction error. This perspective allows us to interpret PCA as implementing an optimal linear auto-encoder. We will draw heavily from Chapters 2 and 3. 

In the previous section, we derived PCA by maximizing the variance in the projected space to retain as much information as possible. In the 

Figure 10.7 Simplified projection setting. (a) A vector x $\mathbf { \Lambda } \in \mathbb { R } ^ { 2 }$ (red cross) shall be projected onto a one-dimensional subspace $U \subseteq \mathbb { R } ^ { 2 }$ spanned by b. (b) shows the difference vectors between x and some candidates x˜. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/567e506d73cfb839187621d8799e42581bf8a77763c3fa6741ad58a790840532.jpg)



(a) Setting.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/23b5ba2a21cfc1a59caef087dcf7b8050c14bb34df23dbc75c9b0af9477447cc.jpg)



(b) Differences $\mathbf { x } - \tilde { \mathbf { x } } _ { i }$ for 50 different $\tilde { \mathbf { x } } _ { i }$ are shown by the red lines.


following, we will look at the difference vectors between the original data ${ \bf { x } } _ { n }$ and their reconstruction $\scriptstyle { \tilde { \mathbf { x } } } _ { n }$ and minimize this distance so that ${ \bf { x } } _ { n }$ and $\scriptstyle { \tilde { \mathbf { x } } } _ { n }$ are as close as possible. Figure 10.6 illustrates this setting. 

Vectors $\tilde { \mathbf { r } } \in U$ could be vectors on a plane in $\mathbb { R } ^ { 3 } .$ . The dimensionality of the plane is 2, but the vectors still have three coordinates with respect to the standard basis of $\mathbb { R } ^ { 3 }$ 

Assume an (ordered) orthonormal basis (ONB) $B = ( b _ { 1 } , \ldots , b _ { D } )$ of $\mathbb { R } ^ { D }$ i.e., $\pmb { b } _ { i } ^ { \top } \pmb { b } _ { j } = 1$ if and only if $i = j$ and 0 otherwise. 

## 10.3.1 Setting and Objective

From Section 2.5 we know that for a basis $\left( b _ { 1 } , \ldots , b _ { D } \right)$ of $\mathbb { R } ^ { D }$ any ${ \pmb x } \in$ $\mathbb { R } ^ { D }$ can be written as a linear combination of the basis vectors of <sup>RD</sup>, i.e., 

$$
\boldsymbol {x} = \sum_ {d = 1} ^ {D} \zeta_ {d} \boldsymbol {b} _ {d} = \sum_ {m = 1} ^ {M} \zeta_ {m} \boldsymbol {b} _ {m} + \sum_ {j = M + 1} ^ {D} \zeta_ {j} \boldsymbol {b} _ {j}\tag{10.26}
$$

for suitable coordinates $\zeta _ { d } \in \mathbb { R } .$ 

We are interested in finding vectors $\tilde { \textbf { \textit { x } } } \in \ \mathbb { R } ^ { D }$ , which live in lowerdimensional subspace $U \subseteq \mathbb { R } ^ { D }$ , dim $( U ) = M$ , so that 

$$
\tilde {\boldsymbol {x}} = \sum_ {m = 1} ^ {M} z _ {m} \boldsymbol {b} _ {m} \in U \subseteq \mathbb {R} ^ {D}\tag{10.27}
$$

is as similar to x as possible. Note that at this point we need to assume that the coordinates $z _ { m }$ of x˜ and $\zeta _ { m }$ of x are not identical. 

In the following, we use exactly this kind of representation of x˜ to find optimal coordinates z and basis vectors $b _ { 1 } , \dots , b _ { M }$ such that x˜ is as similar to the original data point x as possible, i.e., we aim to minimize the (Euclidean) distance $\lVert x - \tilde { { \boldsymbol { x } } } \rVert$ . Figure 10.7 illustrates this setting. 

Without loss of generality, we assume that the dataset ${ \mathcal { X } } = \{ { \pmb x } _ { 1 } , \ldots , { \pmb x } _ { N } \}$ , ${ \pmb x } _ { n } \in \mathbb { R } ^ { D }$ , is centered at $\mathbf { 0 } , \mathrm { i . e . , } \mathbb { E } [ \mathcal { X } ] = \mathbf { 0 }$ . Without the zero-mean assumption, we would arrive at exactly the same solution, but the notation would be substantially more cluttered. 

We are interested in finding the best linear projection of $\mathcal { X }$ onto a lowerdimensional subspace U of $\bar { \mathbb { R } } ^ { D }$ with dim $( U ) = M$ and orthonormal basis vectors $b _ { 1 } , \dots , b _ { M }$ . We will call this subspace U the principal subspace. The projections of the data points are denoted by 

$$
\tilde {\boldsymbol {x}} _ {n} := \sum_ {m = 1} ^ {M} z _ {m n} \boldsymbol {b} _ {m} = \boldsymbol {B} \boldsymbol {z} _ {n} \in \mathbb {R} ^ {D},\tag{10.28}
$$

where $z _ { n } : = [ z _ { 1 n } , \ldots , z _ { M n } ] ^ { \top } \in \mathbb { R } ^ { M }$ is the coordinate vector of ${ \tilde { \mathbf { x } } } _ { n }$ with respect to the basis $\left( \pmb { b } _ { 1 } , \dots , \pmb { b } _ { M } \right)$ . More specifically, we are interested in having the $\scriptstyle { \tilde { \mathbf { x } } } _ { n }$ as similar to ${ \bf { x } } _ { n }$ as possible. 

The similarity measure we use in the following is the squared distance (Euclidean norm) $\| { \pmb x } - { \tilde { { \pmb x } } } \| ^ { 2 }$ between x and x˜. We therefore define our ob jective as minimizing the average squared Euclidean distance (reconstruction reconstruction error error) (Pearson, 1901) 

$$
J _ {M} := \frac {1}{N} \sum_ {n = 1} ^ {N} \| \boldsymbol {x} _ {n} - \tilde {\boldsymbol {x}} _ {n} \| ^ {2},\tag{10.29}
$$

where we make it explicit that the dimension of the subspace onto which we project the data is $M$ . In order to find this optimal linear projection, we need to find the orthonormal basis of the principal subspace and the coordinates $\boldsymbol { z } _ { n } \in \mathbb { R } ^ { M }$ of the projections with respect to this basis. 

To find the coordinates $z _ { n }$ and the ONB of the principal subspace, we follow a two-step approach. First, we optimize the coordinates $z _ { n }$ for a given ONB $\left( \pmb { b } _ { 1 } , \dots , \pmb { b } _ { M } \right)$ ; second, we find the optimal ONB. 

## 10.3.2 Finding Optimal Coordinates

Let us start by finding the optimal coordinates $z _ { 1 n } , \ldots , z _ { M n }$ of the projections $\scriptstyle { \tilde { \mathbf { x } } } _ { n }$ for $n = 1 , \ldots , N$ . Consider Figure $1 0 . 7 ( \mathrm { b } )$ , where the principal subspace is spanned by a single vector b. Geometrically speaking, finding the optimal coordinates z corresponds to finding the representation of the linear projection x˜ with respect to b that minimizes the distance between $\tilde { \boldsymbol { x } } - \boldsymbol { x }$ . From Figure 10.7(b), it is clear that this will be the orthogonal projection, and in the following we will show exactly this. 

We assume an ONB $\left( \pmb { b } _ { 1 } , \dots , \pmb { b } _ { M } \right)$ of $U \subseteq \mathbb { R } ^ { D }$ . To find the optimal coordinates $z _ { m }$ with respect to this basis, we require the partial derivatives 

$$
\frac {\partial J _ {M}}{\partial z _ {i n}} = \frac {\partial J _ {M}}{\partial \tilde {\pmb {x}} _ {n}} \frac {\partial \tilde {\pmb {x}} _ {n}}{\partial z _ {i n}},\tag{10.30a}
$$

$$
\frac {\partial J _ {M}}{\partial \tilde {\pmb {x}} _ {n}} = - \frac {2}{N} (\pmb {x} _ {n} - \tilde {\pmb {x}} _ {n}) ^ {\top} \in \mathbb {R} ^ {1 \times D},\tag{10.30b}
$$

©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 

$$
\pmb {x} \in \mathbb {R} ^ {2}
$$

$$
\tilde {\boldsymbol {x}} \in U.
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/1f0e22880b3cd31d3a663ec474ed353d2b5263f7e5a73733a6ea8be528a814f7.jpg)



(a) Distances $\lVert \pmb { x } - \tilde { \pmb { x } } \rVert$ for some ${ \tilde { \pmb { x } } } = z _ { 1 } { \pmb { b } } \in$ U = span[b]; see panel (b) for the setting.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/83aab24a2b1914672beaa36046706c3ec0724762804910ffd39f103db9a3eb86.jpg)



(b) The vector x˜ that minimizes the distance in panel (a) is its orthogonal projection onto U. The coordinate of the projection x˜ with respect to the basis vector b that spans U is the factor we need to scale b in order to “reach” x˜.


$$
\frac {\partial \tilde {\boldsymbol {x}} _ {n}}{\partial z _ {i n}} \stackrel {(1 0. 2 8)} {=} \frac {\partial}{\partial z _ {i n}} \left(\sum_ {m = 1} ^ {M} z _ {m n} \boldsymbol {b} _ {m}\right) = \boldsymbol {b} _ {i}\tag{10.30c}
$$

for $i = 1 , \ldots , M _ { \mathrm { { \scriptsize ~ ; ~ } } }$ , such that we obtain 

$$
\frac {\partial J _ {M}}{\partial z _ {i n}} \stackrel {{(1 0. 3 0 \mathrm{b})}} {{=}} - \frac {2}{N} (\boldsymbol {x} _ {n} - \tilde {\boldsymbol {x}} _ {n}) ^ {\top} \boldsymbol {b} _ {i} \stackrel {{(1 0. 2 8)}} {{=}} - \frac {2}{N} \left(\boldsymbol {x} _ {n} - \sum_ {m = 1} ^ {M} z _ {m n} \boldsymbol {b} _ {m}\right) ^ {\top} \boldsymbol {b} _ {i}\tag{10.31a}
$$

$$
\stackrel {\mathrm{ONB}} {=} - \frac {2}{N} (\boldsymbol {x} _ {n} ^ {\top} \boldsymbol {b} _ {i} - z _ {i n} \boldsymbol {b} _ {i} ^ {\top} \boldsymbol {b} _ {i}) = - \frac {2}{N} (\boldsymbol {x} _ {n} ^ {\top} \boldsymbol {b} _ {i} - z _ {i n}).\tag{10.31b}
$$

The coordinates of the optimal projection of x with respect to the basis vectors $b _ { 1 } , \dots , \pmb { b } _ { M }$ are the coordinates of the orthogonal projection of ${ \pmb x } _ { n }$ onto the principal subspace. 

since $b _ { i } ^ { \top } b _ { i } = 1$ . Setting this partial derivative to 0 yields immediately the optimal coordinates 

$$
z _ {i n} = \boldsymbol {x} _ {n} ^ {\top} \boldsymbol {b} _ {i} = \boldsymbol {b} _ {i} ^ {\top} \boldsymbol {x} _ {n}\tag{10.32}
$$

for $i = 1 , \ldots , M$ and $n = 1 , \ldots , N$ . This means that the optimal coordinates $z _ { i n }$ of the projection $\scriptstyle { \tilde { \mathbf { x } } } _ { n }$ are the coordinates of the orthogonal projection (see Section 3.8) of the original data point ${ \bf { x } } _ { n }$ onto the onedimensional subspace that is spanned by $b _ { i }$ . Consequently: 

The optimal linear projection $\scriptstyle { \tilde { \mathbf { x } } } _ { n }$ of ${ \bf { x } } _ { n }$ is an orthogonal projection. 

The coordinates of $\tilde { \mathbf { x } } _ { n }$ with respect to the basis $\left( \pmb { b } _ { 1 } , \dots , \pmb { b } _ { M } \right)$ are the coordinates of the orthogonal projection of ${ \bf { x } } _ { n }$ onto the principal subspace. 

An orthogonal projection is the best linear mapping given the objective (10.29). 

The coordinates $\zeta _ { m }$ of x in (10.26) and the coordinates $z _ { m }$ of x˜ in (10.27) 

must be identical for $m = 1 , \ldots , M$ since $U ^ { \perp } = \operatorname { s p a n } [ { \pmb b } _ { M + 1 } , \ldots , { \pmb b } _ { D } ]$ is the orthogonal complement (see Section $3 . 6 )$ of $U = \operatorname { s p a n } [ \pmb { b } _ { 1 } , \dots , \pmb { b } _ { M } ]$ 

Remark (Orthogonal Projections with Orthonormal Basis Vectors). Let us briefly recap orthogonal projections from Section 3.8. If $\left( b _ { 1 } , \ldots , b _ { D } \right)$ is an orthonormal basis of $\mathbb { R } ^ { D }$ then 

$$
\tilde {\boldsymbol {x}} = \boldsymbol {b} _ {j} (\boldsymbol {b} _ {j} ^ {\top} \boldsymbol {b} _ {j}) ^ {- 1} \boldsymbol {b} _ {j} ^ {\top} \boldsymbol {x} = \boldsymbol {b} _ {j} \boldsymbol {b} _ {j} ^ {\top} \boldsymbol {x} \in \mathbb {R} ^ {D}\tag{10.33}
$$

is the orthogonal projection of x onto the subspace spanned by the jth basis vector, and $z _ { j } = b _ { j } ^ { \intercal }$ x is the coordinate of this projection with respect to the basis vector $b _ { j }$ that spans that subspace since $z _ { j } \pmb { b } _ { j } = \tilde { \pmb { x } }$ . Figure 10.8(b) illustrates this setting. 

More generally, if we aim to project onto an M-dimensional subspace of $\mathbb { R } ^ { D }$ , we obtain the orthogonal projection of x onto the M-dimensional subspace with orthonormal basis vectors $b _ { 1 } , \dots , b _ { M }$ as 

$$
\tilde {\boldsymbol {x}} = \boldsymbol {B} (\underbrace {\boldsymbol {B} ^ {\top} \boldsymbol {B}} _ {= I}) ^ {- 1} \boldsymbol {B} ^ {\top} \boldsymbol {x} = \boldsymbol {B} \boldsymbol {B} ^ {\top} \boldsymbol {x},\tag{10.34}
$$

where we defined $B : = [ \pmb { b } _ { 1 } , \dotsc , \pmb { b } _ { M } ] \in \mathbb { R } ^ { D \times M }$ . The coordinates of this projection with respect to the ordered basis $\left( \pmb { b } _ { 1 } , \dots , \pmb { b } _ { M } \right)$ are $z : = B ^ { \top } x$ as discussed in Section 3.8. 

We can think of the coordinates as a representation of the projected vector in a new coordinate system defined by $\left( \pmb { b } _ { 1 } , \dots , \pmb { b } _ { M } \right)$ . Note that although $\tilde { \textbf { \em x } } \in \ \mathbb { R } ^ { D }$ , we only need M coordinates $z _ { 1 } , \dots , z _ { M }$ to represent this vector; the other $D - M$ coordinates with respect to the basis vectors $\left( { \pmb { b } } _ { M + 1 } , \ldots , { \pmb { b } } _ { D } \right)$ are always 0. 

So far we have shown that for a given ONB we can find the optimal coordinates of x˜ by an orthogonal projection onto the principal subspace. In the following, we will determine what the best basis is. 

## 10.3.3 Finding the Basis of the Principal Subspace

To determine the basis vectors $b _ { 1 } , \dots , b _ { M }$ of the principal subspace, we rephrase the loss function (10.29) using the results we have so far. This will make it easier to find the basis vectors. To reformulate the loss func tion, we exploit our results from before and obtain 

$$
\tilde {\boldsymbol {x}} _ {n} = \sum_ {m = 1} ^ {M} z _ {m n} \boldsymbol {b} _ {m} \stackrel {(1 0. 3 2)} {=} \sum_ {m = 1} ^ {M} (\boldsymbol {x} _ {n} ^ {\top} \boldsymbol {b} _ {m}) \boldsymbol {b} _ {m}.\tag{10.35}
$$

We now exploit the symmetry of the dot product, which yields 

$$
\tilde {\boldsymbol {x}} _ {n} = \left(\sum_ {m = 1} ^ {M} \boldsymbol {b} _ {m} \boldsymbol {b} _ {m} ^ {\top}\right) \boldsymbol {x} _ {n}.\tag{10.36}
$$



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



Figure 10.9 Orthogonal projection and displacement vectors. When projecting data points ${ \pmb x } _ { n }$ (blue) onto subspace $U _ { 1 }$ we obtain $\scriptstyle { \tilde { \mathbf { x } } } _ { n }$ (orange). The displacement vector $\tilde { \pmb x } _ { n } - { \pmb x } _ { n }$ lies completely in the orthogonal complement $U _ { 2 }$ of $U _ { 1 }$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/d5721e3fb91c04d61c8a4a7865d7f4d82faa62c187580223340f0e91513ecfed.jpg)


Since we can generally write the original data point ${ \bf { x } } _ { n }$ as a linear combination of all basis vectors, it holds that 

$$
\begin{array}{l} \boldsymbol {x} _ {n} = \sum_ {d = 1} ^ {D} z _ {d n} \boldsymbol {b} _ {d} \stackrel {(1 0. 3 2)} {=} \sum_ {d = 1} ^ {D} (\boldsymbol {x} _ {n} ^ {\top} \boldsymbol {b} _ {d}) \boldsymbol {b} _ {d} = \left(\sum_ {d = 1} ^ {D} \boldsymbol {b} _ {d} \boldsymbol {b} _ {d} ^ {\top}\right) \boldsymbol {x} _ {n} \\ = \left(\sum_ {m = 1} ^ {M} \boldsymbol {b} _ {m} \boldsymbol {b} _ {m} ^ {\top}\right) \boldsymbol {x} _ {n} + \left(\sum_ {j = M + 1} ^ {D} \boldsymbol {b} _ {j} \boldsymbol {b} _ {j} ^ {\top}\right) \boldsymbol {x} _ {n}, \end{array}\tag{10.37a}
$$

(10.37b) 

where we split the sum with $D$ terms into a sum over M and a sum over $D - M$ terms. With this result, we find that the displacement vector $\begin{array} { r l } { \pmb { x } _ { n } - \tilde { \pmb { x } } _ { n } , \mathrm { i . e . } } \end{array}$ , the difference vector between the original data point and its projection, is 

$$
\begin{array}{c} \boldsymbol {x} _ {n} - \tilde {\boldsymbol {x}} _ {n} = \left(\sum_ {j = M + 1} ^ {D} \boldsymbol {b} _ {j} \boldsymbol {b} _ {j} ^ {\top}\right) \boldsymbol {x} _ {n} \\ = \sum_ {j = M + 1} ^ {D} (\boldsymbol {x} _ {n} ^ {\top} \boldsymbol {b} _ {j}) \boldsymbol {b} _ {j}. \end{array}\tag{10.38a}
$$

(10.38b) 

This means the difference is exactly the projection of the data point onto the orthogonal complement of the principal subspace: We identify the matrix $\begin{array} { r } { \sum _ { j = M + 1 } ^ { D } { \pmb { b } } _ { j } { \pmb { b } } _ { j } ^ { \top } } \end{array}$ in (10.38a) as the projection matrix that performs this projection. Hence the displacement vector ${ \pmb x } _ { n } - \tilde { { \pmb x } } _ { 1 }$ lies in the subspace that is orthogonal to the principal subspace as illustrated in Figure 10.9. 

Remark (Low-Rank Approximation). In (10.38a), we saw that the projection matrix, which projects x onto $\tilde { \mathbf { x } } ,$ is given by 

$$
\sum_ {m = 1} ^ {M} \boldsymbol {b} _ {m} \boldsymbol {b} _ {m} ^ {\top} = \boldsymbol {B} \boldsymbol {B} ^ {\top}.\tag{10.39}
$$

By construction as a sum of rank-one matrices $\pmb { b } _ { m } \pmb { b } _ { m } ^ { \top }$ we see that $B B ^ { \top }$ i symmetric and has rank M. Therefore, the average squared reconstruction error can also be written as 

$$
\frac {1}{N} \sum_ {n = 1} ^ {N} \left\| \boldsymbol {x} _ {n} - \tilde {\boldsymbol {x}} _ {n} \right\| ^ {2} = \frac {1}{N} \sum_ {n = 1} ^ {N} \left\| \boldsymbol {x} _ {n} - \boldsymbol {B B} ^ {\top} \boldsymbol {x} _ {n} \right\| ^ {2}\tag{10.40a}
$$

$$
= \frac {1}{N} \sum_ {n = 1} ^ {N} \left\| (\boldsymbol {I} - \boldsymbol {B} \boldsymbol {B} ^ {\top}) \boldsymbol {x} _ {n} \right\| ^ {2}.\tag{10.40b}
$$

Finding orthonormal basis vectors $b _ { 1 } , \dots , b _ { M }$ , which minimize the difference between the original data ${ \bf { x } } _ { n }$ and their projections $\tilde { \mathbf { x } } _ { n } ,$ is equivalent to finding the best rank-M approximation $B B ^ { \top }$ of the identity matrix I (see Section 4.6). 

Now we have all the tools to reformulate the loss function (10.29). 

$$
J _ {M} = \frac {1}{N} \sum_ {n = 1} ^ {N} \| \boldsymbol {x} _ {n} - \tilde {\boldsymbol {x}} _ {n} \| ^ {2} \stackrel {(1 0. 3 8 \mathrm{b})} {=} \frac {1}{N} \sum_ {n = 1} ^ {N} \left\| \sum_ {j = M + 1} ^ {D} (\boldsymbol {b} _ {j} ^ {\top} \boldsymbol {x} _ {n}) \boldsymbol {b} _ {j} \right\| ^ {2}.\tag{10.41}
$$

We now explicitly compute the squared norm and exploit the fact that the $b _ { j }$ form an ONB, which yields 

$$
\begin{array}{l} J _ {M} = \frac {1}{N} \sum_ {n = 1} ^ {N} \sum_ {j = M + 1} ^ {D} (\boldsymbol {b} _ {j} ^ {\top} \boldsymbol {x} _ {n}) ^ {2} = \frac {1}{N} \sum_ {n = 1} ^ {N} \sum_ {j = M + 1} ^ {D} \boldsymbol {b} _ {j} ^ {\top} \boldsymbol {x} _ {n} \boldsymbol {b} _ {j} ^ {\top} \boldsymbol {x} _ {n} \\ = \frac {1}{N} \sum_ {n = 1} ^ {N} \sum_ {j = M + 1} ^ {D} \boldsymbol {b} _ {j} ^ {\top} \boldsymbol {x} _ {n} \boldsymbol {x} _ {n} ^ {\top} \boldsymbol {b} _ {j}, \end{array}\tag{10.42a}
$$

(10.42b) 

where we exploited the symmetry of the dot product in the last step to write $\pmb { b } _ { j } ^ { \top } \pmb { x } _ { n } = \pmb { x } _ { n } ^ { \top } \pmb { b } _ { j }$ . We now swap the sums and obtain 

$$
\begin{array}{l} J _ {M} = \sum_ {j = M + 1} ^ {D} \boldsymbol {b} _ {j} ^ {\top} \underbrace {\left(\frac {1}{N} \sum_ {n = 1} ^ {N} \boldsymbol {x} _ {n} \boldsymbol {x} _ {n} ^ {\top}\right)} _ {=: S} \boldsymbol {b} _ {j} = \sum_ {j = M + 1} ^ {D} \boldsymbol {b} _ {j} ^ {\top} \boldsymbol {S} \boldsymbol {b} _ {j} \\ = \sum_ {j = M + 1} ^ {D} \operatorname{tr} (\boldsymbol {b} _ {j} ^ {\top} \boldsymbol {S} \boldsymbol {b} _ {j}) = \sum_ {j = M + 1} ^ {D} \operatorname{tr} (\boldsymbol {S} \boldsymbol {b} _ {j} \boldsymbol {b} _ {j} ^ {\top}) = \operatorname{tr} \Big (\underbrace {\left(\sum_ {j = M + 1} ^ {D} \boldsymbol {b} _ {j} \boldsymbol {b} _ {j} ^ {\top}\right) \boldsymbol {S}} _ {\text {projection matrix}} \Big), \end{array} \tag {10.43a}\tag{10.43b}
$$

where we exploited the property that the trace operator $\operatorname { t r } ( \cdot )$ (see (4.18)) is linear and invariant to cyclic permutations of its arguments. Since we assumed that our dataset is centered, i.e., $\mathbf { \mathbb { E } } [ \mathcal { X } ] = \mathbf { 0 }$ , we identify S as the data covariance matrix. Since the projection matrix in (10.43b) is constructed as a sum of rank-one matrices $\boldsymbol { b } _ { j } \boldsymbol { b } _ { j } ^ { \intercal }$ it itself is of rank $D - M$ 

Equation (10.43a) implies that we can formulate the average squared reconstruction error equivalently as the covariance matrix of the data, 

Figure 10.10 Embedding of MNIST digits 0 (blue) and 1 (orange) in a two-dimensional principal subspace using PCA. Four embeddings of the digits $^ { 6 } 0 ^ { \dag }$ and “1” in the principal subspace are highlighted in red with their corresponding original digit. 

projected onto the orthogonal complement of the principal subspace. Minimizing the average squared reconstruction error is therefore equivalent to minimizing the variance of the data when projected onto the subspace we ignore, i.e., the orthogonal complement of the principal subspace. Equivalently, we maximize the variance of the projection that we retain in the principal subspace, which links the projection loss immediately to the maximum-variance formulation of PCA discussed in Section 10.2. But this then also means that we will obtain the same solution that we obtained for the maximum-variance perspective. Therefore, we omit a derivation that is identical to the one presented in Section 10.2 and summarize the results from earlier in the light of the projection perspective. 

The average squared reconstruction error, when projecting onto the Mdimensional principal subspace, is 

$$
J _ {M} = \sum_ {j = M + 1} ^ {D} \lambda_ {j},\tag{10.44}
$$

where $\lambda _ { j }$ are the eigenvalues of the data covariance matrix. Therefore, to minimize (10.44) we need to select the smallest D M eigenvalues, which then implies that their corresponding eigenvectors are the basis of the orthogonal complement of the principal subspace. Consequently, this means that the basis of the principal subspace comprises the eigenvectors $b _ { 1 } , \dots , b _ { M }$ that are associated with the largest M eigenvalues of the data covariance matrix. 

## Example 10.3 (MNIST Digits Embedding)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/f55f11de9e399c8c0e5f30aba4a9b684dd23bf1539237e07f2d58cd9b15ebaf0.jpg)



Figure 10.10 visualizes the training data of the MMIST digits $" 0 "$ and $^ { \mathfrak { s } } \mathrm { 1 } ^ { \mathfrak { p } }$ embedded in the vector subspace spanned by the first two principal components. We observe a relatively clear separation between ${ } ^ { \mathfrak { s o } } { } ^ { \mathfrak { s } }$ (blue dots) and $^ { \mathfrak { \omega } } \mathfrak { I } ^ { \mathfrak { s } }$ (orange dots), and we see the variation within each individual


cluster. Four embeddings of the digits $" 0 "$ and $^ { \mathfrak { s } } \mathrm { 1 } ^ { \mathfrak { p } }$ in the principal subspace are highlighted in red with their corresponding original digit. The figure reveals that the variation within the set $\mathrm { o f } \ ^ { \mathrm { \infty } } 0 ^ { \mathrm { \prime } }$ is significantly greater than the variation within the set of “1”. 

## 10.4 Eigenvector Computation and Low-Rank Approximations

In the previous sections, we obtained the basis of the principal subspace as the eigenvectors that are associated with the largest eigenvalues of the data covariance matrix 

$$
\boldsymbol {S} = \frac {1}{N} \sum_ {n = 1} ^ {N} \boldsymbol {x} _ {n} \boldsymbol {x} _ {n} ^ {\top} = \frac {1}{N} \boldsymbol {X} \boldsymbol {X} ^ {\top},\tag{10.45}
$$

$$
\pmb {X} = [ \pmb {x} _ {1}, \dots , \pmb {x} _ {N} ] \in \mathbb {R} ^ {D \times N}.\tag{10.46}
$$

Note that X is a $D \times N$ matrix, i.e., it is the transpose of the “typical” data matrix (Bishop, 2006; Murphy, 2012). To get the eigenvalues (and the corresponding eigenvectors) of S, we can follow two approaches: 

We perform an eigendecomposition (see Section 4.2) and compute the eigenvalues and eigenvectors of S directly. 

We use a singular value decomposition (see Section 4.5). Since S is symmetric and factorizes into $X \bar { X } ^ { \top }$ (ignoring the factor $\textstyle { \frac { 1 } { N } } )$ , the eigenvalues of $_ { s }$ are the squared singular values of $\boldsymbol { X }$ 

More specifically, the SVD of X is given by 

$$
\underbrace {\boldsymbol {X}} _ {D \times N} = \underbrace {\boldsymbol {U}} _ {D \times D} \underbrace {\boldsymbol {\Sigma}} _ {D \times N} \underbrace {\boldsymbol {V} ^ {\top}} _ {N \times N},\tag{10.47}
$$

where $U \in \mathbb { R } ^ { D \times D }$ and $V ^ { \top } \in \mathbb { R } ^ { N \times N }$ are orthogonal matrices and $\pmb { \Sigma } \in$ $\mathbb { R } ^ { D \times N }$ is a matrix whose only nonzero entries are the singular values $\sigma _ { i i } \geqslant$ 0. It then follows that 

$$
\boldsymbol {S} = \frac {1}{N} \boldsymbol {X} \boldsymbol {X} ^ {\top} = \frac {1}{N} \boldsymbol {U} \boldsymbol {\Sigma} \underbrace {\boldsymbol {V} ^ {\top} \boldsymbol {V}} _ {= \boldsymbol {I} _ {N}} \boldsymbol {\Sigma} ^ {\top} \boldsymbol {U} ^ {\top} = \frac {1}{N} \boldsymbol {U} \boldsymbol {\Sigma} \boldsymbol {\Sigma} ^ {\top} \boldsymbol {U} ^ {\top}.\tag{10.48}
$$

With the results from Section 4.5, we get that the columns of $U$ are the eigenvectors of $\pmb { X } \pmb { X } ^ { \top }$ (and therefore S). Furthermore, the eigenvalues $\lambda _ { d }$ of $\pmb { S }$ are related to the singular values of $\boldsymbol { X }$ via 

$$
\lambda_ {d} = \frac {\sigma_ {d} ^ {2}}{N}.\tag{10.49}
$$

This relationship between the eigenvalues of $\pmb { S }$ and the singular values of X provides the connection between the maximum variance view (Section 10.2) and the singular value decomposition. 

## 10.4.1 PCA Using Low-Rank Matrix Approximations

To maximize the variance of the projected data (or minimize the average squared reconstruction error), PCA chooses the columns of U in (10.48) to be the eigenvectors that are associated with the M largest eigenvalues of the data covariance matrix S so that we identify U as the projection matrix B in (10.3), which projects the original data onto a lower-dimensional subspace of dimension M. The Eckart-Young theorem (Theorem 4.25 in Section 4.6) offers a direct way to estimate the low-dimensional representation. Consider the best rank-M approximation 

$$
\tilde {\boldsymbol {X}} _ {M} := \operatorname{argmin} _ {\mathrm{rk} (\boldsymbol {A}) \leqslant M} \| \boldsymbol {X} - \boldsymbol {A} \| _ {2} \in \mathbb {R} ^ {D \times N}\tag{10.50}
$$

of $X ,$ , where $\left\| \cdot \right\| _ { 2 }$ is the spectral norm defined in (4.93). The Eckart-Young theorem states that $\tilde { \pmb { X } } _ { M }$ is given by truncating the SVD at the top-M singular value. In other words, we obtain 

$$
\tilde {\boldsymbol {X}} _ {M} = \underbrace {\boldsymbol {U} _ {M}} _ {D \times M} \underbrace {\boldsymbol {\Sigma} _ {M}} _ {M \times M} \underbrace {\boldsymbol {V} _ {M} ^ {\top}} _ {M \times N} \in \mathbb {R} ^ {D \times N}\tag{10.51}
$$

with orthogonal matrices $\pmb { U } _ { M } : = [ \pmb { u } _ { 1 } , \dots , \pmb { u } _ { M } ] \in \mathbb { R } ^ { D \times M }$ and $V _ { M } : =$ $[ { \pmb v } _ { 1 } , \dots , { \pmb v } _ { M } ] \in \mathbb { R } ^ { N \times M }$ and a diagonal matrix $\boldsymbol { \Sigma } _ { M } ^ { \mathsf { ^ { - } } } \in \mathbb { R } ^ { M \times M }$ whose diagonal entries are the M largest singular values of X. 

## 10.4.2 Practical Aspects

Finding eigenvalues and eigenvectors is also important in other fundamental machine learning methods that require matrix decompositions. In theory, as we discussed in Section 4.2, we can solve for the eigenvalues as roots of the characteristic polynomial. However, for matrices larger than 4 4 this is not possible because we would need to find the roots of a polynomial of degree 5 or higher. However, the Abel-Ruffini theorem (Ruffini, 1799; Abel, 1826) states that there exists no algebraic solution to this problem for polynomials of degree 5 or more. Therefore, in practice, we solve for eigenvalues or singular values using iterative methods, which are implemented in all modern packages for linear algebra. 

In many applications (such as PCA presented in this chapter), we only require a few eigenvectors. It would be wasteful to compute the full decomposition, and then discard all eigenvectors with eigenvalues that are beyond the first few. It turns out that if we are interested in only the first few eigenvectors (with the largest eigenvalues), then iterative processes, which directly optimize these eigenvectors, are computationally more efficient than a full eigendecomposition (or SVD). In the extreme case of only needing the first eigenvector, a simple method called the power iteration is very efficient. Power iteration chooses a random vector $\scriptstyle { \pmb x } _ { 0 }$ that is not in the null space of S and follows the iteration 

$$
\boldsymbol {x} _ {k + 1} = \frac {\boldsymbol {S x} _ {k}}{\| \boldsymbol {S x} _ {k} \|}, \quad k = 0, 1, \dots .\tag{10.52}
$$

This means the vector $\scriptstyle { \mathbf { { \mathit { x } } } } _ { k }$ is multiplied by S in every iteration and then normalized, i.e., we always have $\| \pmb { x } _ { k } \| = 1$ . This sequence of vectors converges to the eigenvector associated with the largest eigenvalue of S. The original Google PageRank algorithm (Page et al., 1999) uses such an algorithm for ranking web pages based on their hyperlinks. 

## 10.5 PCA in High Dimensions

In order to do PCA, we need to compute the data covariance matrix. In D dimensions, the data covariance matrix is a $D \times D$ matrix. Computing the eigenvalues and eigenvectors of this matrix is computationally expensive as it scales cubically in D. Therefore, PCA, as we discussed earlier, will be infeasible in very high dimensions. For example, if our ${ \bf { x } } _ { n }$ are images with 10,000 pixels $( \mathrm { e } . g . , 1 0 0 \times 1 0 0$ pixel images), we would need to compute the eigendecomposition of a $1 0 , 0 0 0 \times 1 0 , 0 0 0$ covariance matrix. In the following, we provide a solution to this problem for the case that we have substantially fewer data points than dimensions, i.e., $N \ll D .$ 

Assume we have a centered dataset $\pmb { x } _ { 1 } , \dots , \pmb { x } _ { N } , \pmb { x } _ { n } \in \mathbb { R } ^ { D }$ . Then the data covariance matrix is given as 

$$
\pmb {S} = \frac {1}{N} \pmb {X} \pmb {X} ^ {\top} \in \mathbb {R} ^ {D \times D},\tag{10.53}
$$

where $\pmb { X } = [ \pmb { x } _ { 1 } , \ldots , \pmb { x } _ { N } ]$ is a $D \times N$ matrix whose columns are the data points. 

We now assume that $N \ll D , \mathrm { i . e . }$ , the number of data points is smaller than the dimensionality of the data. If there are no duplicate data points, the rank of the covariance matrix S is N , so it has $D - N + 1$ many eigenvalues that are 0. Intuitively, this means that there are some redundancies. In the following, we will exploit this and turn the $D \times D$ covariance matrix into an $N \times N$ covariance matrix whose eigenvalues are all positive. 

In PCA, we ended up with the eigenvector equation 

$$
\boldsymbol {S} \boldsymbol {b} _ {m} = \lambda_ {m} \boldsymbol {b} _ {m}, \quad m = 1, \dots , M,\tag{10.54}
$$

where $b _ { m }$ is a basis vector of the principal subspace. Let us rewrite this equation a bit: With $\pmb { S }$ defined in (10.53), we obtain 

$$
\pmb {S} \pmb {b} _ {m} = \frac {1}{N} \pmb {X} \pmb {X} ^ {\top} \pmb {b} _ {m} = \lambda_ {m} \pmb {b} _ {m}.\tag{10.55}
$$

We now multiply $\pmb { X } ^ { \top } \in \mathbb { R } ^ { N \times D }$ from the left-hand side, which yields 

$$
\frac {1}{N} \underbrace {\boldsymbol {X} ^ {\top} \boldsymbol {X}} _ {N \times N} \underbrace {\boldsymbol {X} ^ {\top} \boldsymbol {b} _ {m}} _ {=: \boldsymbol {c} _ {m}} = \lambda_ {m} \boldsymbol {X} ^ {\top} \boldsymbol {b} _ {m} \iff \frac {1}{N} \boldsymbol {X} ^ {\top} \boldsymbol {X} \boldsymbol {c} _ {m} = \lambda_ {m} \boldsymbol {c} _ {m}  ,\tag{10.56}
$$

and we get a new eigenvector/eigenvalue equation: $\lambda _ { m }$ remains eigenvalue, which confirms our results from Section 4.5.3 that the nonzero eigenvalues of $X X ^ { \top }$ equal the nonzero eigenvalues of $\boldsymbol { X } ^ { \top } \boldsymbol { X }$ . We obtain the eigenvector of the matrix $\mathbf { \Psi } _ { N } ^ { 1 } \mathbf { \boldsymbol { X } } ^ { \top } \mathbf { \boldsymbol { X } } \in \mathbf { \tilde { \Psi } } \mathbb { R } ^ { N \times N }$ associated with $\lambda _ { m }$ as $\pmb { c } _ { m } : = \pmb { X } ^ { \top } \pmb { b } _ { m }$ . Assuming we have no duplicate data points, this matrix has rank N and is invertible. This also implies that $\begin{array} { r } { \frac { 1 } { N } \bar { \pmb X } ^ { \top } \bar { \pmb X } } \end{array}$ has the same (nonzero) eigenvalues as the data covariance matrix $\pmb { S }$ . But this is now an $N \times N$ matrix, so that we can compute the eigenvalues and eigenvectors much more efficiently than for the original $D \times D$ data covariance matrix. 

Now that we have the eigenvectors of $\scriptstyle { \frac { 1 } { N } } X ^ { \intercal } X$ , we are going to recover the original eigenvectors, which we still need for PCA. Currently, we know the eigenvectors of $\scriptstyle { \frac { 1 } { N } } X ^ { \intercal }$ X. If we left-multiply our eigenvalue/ eigenvector equation with $\boldsymbol { X }$ , we get 

$$
\underbrace {\frac {1}{N} \boldsymbol {X} \boldsymbol {X} ^ {\top}} _ {s} \boldsymbol {X} \boldsymbol {c} _ {m} = \lambda_ {m} \boldsymbol {X} \boldsymbol {c} _ {m}\tag{10.57}
$$

and we recover the data covariance matrix again. This now also means that we recover $X c _ { m }$ as an eigenvector of S. 

Remark. If we want to apply the PCA algorithm that we discussed in Section 10.6, we need to normalize the eigenvectors $X c _ { m }$ of $\pmb { S }$ so that they have norm 1. ◇ 

## 10.6 Key Steps of PCA in Practice

In the following, we will go through the individual steps of PCA using a running example, which is summarized in Figure 10.11. We are given a two-dimensional dataset (Figure 10.11(a)), and we want to use PCA to project it onto a one-dimensional subspace. 

1. Mean subtraction We start by centering the data by computing the mean µ of the dataset and subtracting it from every single data point. This ensures that the dataset has mean 0 (Figure 10.11(b)). Mean subtraction is not strictly necessary but reduces the risk of numerical problems. 

2. Standardization Divide the data points by the standard deviation $\sigma _ { d }$ of the dataset for every dimension $d = 1 , \ldots , D$ . Now the data is unit free, and it has variance 1 along each axis, which is indicated by the two arrows in Figure 10.11(c). This step completes the standardization of the data. 

3. Eigendecomposition of the covariance matrix Compute the data covariance matrix and its eigenvalues and corresponding eigenvectors. Since the covariance matrix is symmetric, the spectral theorem (Theorem 4.15) states that we can find an ONB of eigenvectors. In Figure 10.11(d), the eigenvectors are scaled by the magnitude of the corresponding eigenvalue. The longer vector spans the principal subspace, which we denote by U. The data covariance matrix is represented by the ellipse. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/2294b27ad6affb241137203a266bf17fb50f22121103ded5d3d416aea8a28e54.jpg)



(a) Original dataset.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/82337eb77f240ddbd0db679d52249fee54bebb9dc102906a7a5f2f018bc2ae3f.jpg)



(b) Step 1: Centering by subtracting the mean from each data point.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/d8596ab0d5c703643bab5791aa09a439985cd065bb4cc2e08d6e831920951752.jpg)



(b) centering; (c) divide by standard deviation; (d) eigendecomposi tion; (e) projection; (f) mapping back to original data space.



(c) Step 2: Dividing by the standard deviation to make the data unit free. Data has variance 1 along each axis.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/aef70a8021a4db6c7406e634857f138097a7002920d82d7f195c8680bc554086.jpg)



(d) Step 3: Compute eigenvalues and eigenvectors (arrows) of the data covariance matrix (ellipse).


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/7baddcb0d3a9072eb7ee69e5066d6582def25b549fe8bd3ba08a56c96f054b7a.jpg)



(e) Step 4: Project data onto the principal subspace.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/d38e314d00a19108eb9c42d4f158e0bdae6737f1644c41cb7af6669d0019d40b.jpg)



(f) Undo the standardization and move projected data back into the original data space from (a).


4. Projection We can project any data point $\pmb { x } _ { * } \in \mathbb { R } ^ { D }$ onto the principal subspace: To get this right, we need to standardize $^ { \mathbf { x } } { } ^ { \mathrm { ~ } }$ using the mean $\mu _ { d }$ and standard deviation $\sigma _ { d }$ of the training data in the dth dimension, respectively, so that 

$$
x _ {*} ^ {(d)} \leftarrow \frac {x _ {*} ^ {(d)} - \mu_ {d}}{\sigma_ {d}}, d = 1, \ldots , D,\tag{10.58}
$$

where $x _ { * } ^ { ( d ) }$ is the dth component of ${ \pmb x } _ { \ast }$ . We obtain the projection as 

$$
\tilde {\boldsymbol {x}} _ {*} = \boldsymbol {B} \boldsymbol {B} ^ {\top} \boldsymbol {x} _ {*}\tag{10.59}
$$

with coordinates 

$$
\boldsymbol {z} _ {*} = \boldsymbol {B} ^ {\top} \boldsymbol {x} _ {*}\tag{10.60}
$$

with respect to the basis of the principal subspace. Here, B is the matrix that contains the eigenvectors that are associated with the largest eigenvalues of the data covariance matrix as columns. PCA returns the coordinates (10.60), not the projections $\mathbf { x } _ { \ast }$ 



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



Having standardized our dataset, (10.59) only yields the projections in the context of the standardized dataset. To obtain our projection in the original data space (i.e., before standardization), we need to undo the standardization (10.58) and multiply by the standard deviation before adding the mean so that we obtain 

$$
\tilde {x} _ {*} ^ {(d)} \leftarrow \tilde {x} _ {*} ^ {(d)} \sigma_ {d} + \mu_ {d}, d = 1, \ldots , D.\tag{10.61}
$$

Figure 10.11(f) illustrates the projection in the original data space. 

## Example 10.4 (MNIST Digits: Reconstruction)

In the following, we will apply PCA to the MNIST digits dataset, which contains 60,000 examples of handwritten digits 0 through 9. Each digit is an image of size $2 8 \times 2 8 ,$ , i.e., it contains 784 pixels so that we can interpret every image in this dataset as a vector $\pmb { x } \in \mathbb { R } ^ { 7 8 4 }$ . Examples of these digits are shown in Figure 10.3. 

Figure 10.12 Effect of increasing the number of principal components on reconstruction. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/a4844ac0b4ce8d260d5e20316d2b82a4f4a6e1ab68857704142f98a7c3e92e65.jpg)


For illustration purposes, we apply PCA to a subset of the MNIST digits, and we focus on the digit “8”. We used 5,389 training images of the digit “8” and determined the principal subspace as detailed in this chapter. We then used the learned projection matrix to reconstruct a set of test images, which is illustrated in Figure 10.12. The first row of Figure 10.12 shows a set of four original digits from the test set. The following rows show reconstructions of exactly these digits when using a principal subspace of dimensions 1, 10, 100, and 500, respectively. We see that even with a single-dimensional principal subspace we get a halfway decent reconstruction of the original digits, which, however, is blurry and generic. With an increasing number of principal components (PCs), the reconstructions become sharper and more details are accounted for. With 500 principal components, we effectively obtain a near-perfect reconstruction. If we were to choose 784 PCs, we would recover the exact digit without any compression loss. 

Figure 10.13 shows the average squared reconstruction error, which is 

$$
\frac {1}{N} \sum_ {n = 1} ^ {N} \| \pmb {x} _ {n} - \tilde {\pmb {x}} _ {n} \| ^ {2} = \sum_ {i = M + 1} ^ {D} \lambda_ {i},\tag{10.62}
$$

as a function of the number M of principal components. We can see that the importance of the principal components drops off rapidly, and only marginal gains can be achieved by adding more PCs. This matches exactly our observation in Figure 10.5, where we discovered that the most of the variance of the projected data is captured by only a few principal components. With about 550 $\mathrm { P C } s ,$ , we can essentially fully reconstruct the training data that contains the digit “8” (some pixels around the boundaries show no variation across the dataset as they are always black). 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/bc19e6a45265b356bfb6471ce5750d0eee7f1a23a4fad843cae1ef3d8517dc27.jpg)



Figure 10.13 Average squared reconstruction error as a function of the number of principal components. The average squared reconstruction error is the sum of the eigenvalues in the orthogonal complement of the principal subspace.


## 10.7 Latent Variable Perspective

In the previous sections, we derived PCA without any notion of a prob abilistic model using the maximum-variance and the projection perspectives. On the one hand, this approach may be appealing as it allows us to sidestep all the mathematical difficulties that come with probability theory, but on the other hand, a probabilistic model would offer us more flexibility and useful insights. More specifically, a probabilistic model would 

Come with a likelihood function, and we can explicitly deal with noisy observations (which we did not even discuss earlier) 

Allow us to do Bayesian model comparison via the marginal likelihood as discussed in Section 8.6 

View PCA as a generative model, which allows us to simulate new data 

Allow us to make straightforward connections to related algorithms 

Deal with data dimensions that are missing at random by applying Bayes’ theorem 

Give us a notion of the novelty of a new data point 

Give us a principled way to extend the model, e.g., to a mixture of PCA models 

Have the PCA we derived in earlier sections as a special case 

Allow for a fully Bayesian treatment by marginalizing out the model parameters 

By introducing a continuous-valued latent variable $z \in \mathbb { R } ^ { M }$ it is possible to phrase PCA as a probabilistic latent-variable model. Tipping and Bishop (1999) proposed this latent-variable model as probabilistic PCA (PPCA). PPCA addresses most of the aforementioned issues, and the PCA solution that we obtained by maximizing the variance in the projected space or by minimizing the reconstruction error is obtained as the special case of maximum likelihood estimation in a noise-free setting. 

## 10.7.1 Generative Process and Probabilistic Model

In PPCA, we explicitly write down the probabilistic model for linear dimensionality reduction. For this we assume a continuous latent variable $z \in \mathbb { R } ^ { M }$ with a standard-normal prior $p ( z ) = \mathcal { N } ( \mathbf { 0 } , I )$ and a linear relationship between the latent variables and the observed x data where 

$$
\boldsymbol {x} = \boldsymbol {B} \boldsymbol {z} + \boldsymbol {\mu} + \boldsymbol {\epsilon} \in \mathbb {R} ^ {D},\tag{10.63}
$$

where $\epsilon \sim \mathcal { N } ( \mathbf { 0 } , \sigma ^ { 2 } I )$ is Gaussian observation noise and $B \in \mathbb { R } ^ { D \times M }$ and $\pmb { \mu } \in \mathbb { R } ^ { D }$ describe the linear/affine mapping from latent to observed variables. Therefore, PPCA links latent and observed variables via 

$$
p (\boldsymbol {x} | \boldsymbol {z}, \boldsymbol {B}, \boldsymbol {\mu}, \sigma^ {2}) = \mathcal {N} \big (\boldsymbol {x} \mid B \boldsymbol {z} + \boldsymbol {\mu}, \sigma^ {2} \boldsymbol {I} \big).\tag{10.64}
$$

Overall, PPCA induces the following generative process: 

$$
\boldsymbol {z} _ {n} \sim \mathcal {N} (\boldsymbol {z} | \boldsymbol {0}, \boldsymbol {I})\tag{10.65}
$$

$$
\boldsymbol {x} _ {n} \mid \boldsymbol {z} _ {n} \sim \mathcal {N} (\boldsymbol {x} \mid B \boldsymbol {z} _ {n} + \boldsymbol {\mu}, \sigma^ {2} \boldsymbol {I})\tag{10.66}
$$

To generate a data point that is typical given the model parameters, we follow an ancestral sampling scheme: We first sample a latent variable $z _ { n }$ from $p ( z )$ . Then we use $z _ { n }$ in (10.64) to sample a data point conditioned on the sampled $z _ { n } , { \mathrm { i . e . , } } x _ { n } \sim p ( { \pmb x } \mid z _ { n } , B , \mu , \sigma ^ { 2 } )$ 

This generative process allows us to write down the probabilistic model (i.e., the joint distribution of all random variables; see Section 8.4) as 

$$
p (\pmb {x}, \pmb {z} | \pmb {B}, \pmb {\mu}, \sigma^ {2}) = p (\pmb {x} | \pmb {z}, \pmb {B}, \pmb {\mu}, \sigma^ {2}) p (\pmb {z}),\tag{10.67}
$$

which immediately gives rise to the graphical model in Figure 10.14 using the results from Section 8.5. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/720cd997403aef5521e5c57c97f44110b59d397b484257f42de10d259f83c0cf.jpg)



Figure 10.14 Graphical model for probabilistic PCA. The observations x<sub>n</sub> explicitly depend on corresponding latent variables $\begin{array} { r } { z _ { n } \sim \mathcal { N } ( \mathbf { 0 } , I ) } \end{array}$ . The model parameters B, µ and the likelihood parameter σ are shared across the dataset.


Remark. Note the direction of the arrow that connects the latent variables z and the observed data x: The arrow points from $\textit { z } \mathrm { t o } \textbf { \em x }$ , which means that the PPCA model assumes a lower-dimensional latent cause z for highdimensional observations x. In the end, we are obviously interested in finding something out $\mathrm { \ a b o u t { } } z \mathrm { \ g i v e n }$ some observations. To get there we will apply Bayesian inference to $\displaystyle { } ^ { \mathfrak { \omega } _ { \mathrm { i n v e r t } } , \mathfrak { \ d } }$ the arrow implicitly and go from observations to latent variables. 

## Example 10.5 (Generating New Data Using Latent Variables)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/716797b1f86f3c8e8a7c1184a87976145dd513790ca9efc5158e63e64e33dfde.jpg)



Figure 10.15 Generating new MNIST digits. The latent variables z can be used to generate new data $\tilde { \mathbf { x } } = B z$ . The closer we stay to the training data, the more realistic the generated data.


Figure 10.15 shows the latent coordinates of the MNIST digits $" 8 "$ found by PCA when using a two-dimensional principal subspace (blue dots). We can query any vector $z _ { * }$ in this latent space and generate an image $\tilde { \mathbf { { x } } } _ { * } = $ $\pmb { B z }$ that resembles the digit “8”. We show eight of such generated images with their corresponding latent space representation. Depending on where we query the latent space, the generated images look different (shape, rotation, size, etc.). If we query away from the training data, we see more and more artifacts, e.g., the top-left and top-right digits. Note that the intrinsic dimensionality of these generated images is only two. 



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



## 10.7.2 Likelihood and Joint Distribution

Using the results from Chapter $^ { 6 , }$ we obtain the likelihood of this probabilistic model by integrating out the latent variable z (see Section 8.4.3) so that 

$$
p (\boldsymbol {x} \mid \boldsymbol {B}, \boldsymbol {\mu}, \sigma^ {2}) = \int p (\boldsymbol {x} \mid \boldsymbol {z}, \boldsymbol {B}, \boldsymbol {\mu}, \sigma^ {2}) p (\boldsymbol {z}) \mathrm{d} \boldsymbol {z}\tag{10.68a}
$$

$$
= \int \mathcal {N} (\boldsymbol {x} | \boldsymbol {B z} + \boldsymbol {\mu}, \sigma^ {2} \boldsymbol {I}) \mathcal {N} (\boldsymbol {z} | \boldsymbol {0}, \boldsymbol {I}) \mathrm{d} \boldsymbol {z}.\tag{10.68b}
$$

From Section $6 . 5 _ { : }$ , we know that the solution to this integral is a Gaussian distribution with mean 

$$
\mathbb {E} _ {\boldsymbol {x}} [ \boldsymbol {x} ] = \mathbb {E} _ {\boldsymbol {z}} [ B \boldsymbol {z} + \boldsymbol {\mu} ] + \mathbb {E} _ {\epsilon} [ \epsilon ] = \boldsymbol {\mu}\tag{10.69}
$$

and with covariance matrix 

$$
\begin{array}{r} \mathbb {V} [ \boldsymbol {x} ] = \mathbb {V} _ {\boldsymbol {z}} [ \boldsymbol {B} \boldsymbol {z} + \boldsymbol {\mu} ] + \mathbb {V} _ {\boldsymbol {\epsilon}} [ \boldsymbol {\epsilon} ] = \mathbb {V} _ {\boldsymbol {z}} [ \boldsymbol {B} \boldsymbol {z} ] + \sigma^ {2} \boldsymbol {I} \\ = \boldsymbol {B} \mathbb {V} _ {\boldsymbol {z}} [ \boldsymbol {z} ] \boldsymbol {B} ^ {\top} + \sigma^ {2} \boldsymbol {I} = \boldsymbol {B} \boldsymbol {B} ^ {\top} + \sigma^ {2} \boldsymbol {I}. \end{array}\tag{10.70a}
$$

(10.70b) 

The likelihood in (10.68b) can be used for maximum likelihood or MAP estimation of the model parameters. 

Remark. We cannot use the conditional distribution in (10.64) for maximum likelihood estimation as it still depends on the latent variables. The likelihood function we require for maximum likelihood (or MAP) estimation should only be a function of the data x and the model parameters, but must not depend on the latent variables. ◇ 

From Section 6.5, we know that a Gaussian random variable z and a linear/affine transformation $\textbf { \em x } = \textbf { \em B z }$ of it are jointly Gaussian distributed. We already know the marginals $p ( z ) = \mathcal { N } \big ( z | \mathbf { 0 } , I \big )$ and $p ( { \pmb x } ) =$ $\mathcal { N } ( { \pmb x } | { \pmb \mu } , B B ^ { \top } + \sigma ^ { 2 } { \pmb I } )$ . The missing cross-covariance is given as 

$$
\operatorname{Cov} [ \boldsymbol {x}, \boldsymbol {z} ] = \operatorname{Cov} _ {\boldsymbol {z}} [ \boldsymbol {B} \boldsymbol {z} + \boldsymbol {\mu} ] = \boldsymbol {B} \operatorname{Cov} _ {\boldsymbol {z}} [ \boldsymbol {z}, \boldsymbol {z} ] = \boldsymbol {B}.\tag{10.71}
$$

Therefore, the probabilistic model of PPCA, i.e., the joint distribution of latent and observed random variables is explicitly given by 

$$
p (\boldsymbol {x}, \boldsymbol {z} \mid \boldsymbol {B}, \boldsymbol {\mu}, \sigma^ {2}) = \mathcal {N} \left(\left[ \begin{array}{c} \boldsymbol {x} \\ \boldsymbol {z} \end{array} \right]   \bigg |   \left[ \begin{array}{c} \boldsymbol {\mu} \\ \boldsymbol {0} \end{array} \right], \left[ \begin{array}{c c} \boldsymbol {B} \boldsymbol {B} ^ {\top} + \sigma^ {2} \boldsymbol {I} & \boldsymbol {B} \\ \boldsymbol {B} ^ {\top} & \boldsymbol {I} \end{array} \right]\right)  ,\tag{10.72}
$$

with a mean vector of length $D + M$ and a covariance matrix of size $( D + M ) \times ( D + M )$ 

## 10.7.3 Posterior Distribution

The joint Gaussian distribution $p ( \pmb { x } , z | \pmb { B } , \pmb { \mu } , \sigma ^ { 2 } )$ in (10.72) allows us to determine the posterior distribution $p ( \pmb { z } \vert \pmb { x } )$ immediately by applying the rules of Gaussian conditioning from Section 6.5.1. The posterior distribution of the latent variable given an observation x is then 

$$
p (\boldsymbol {z} \mid \boldsymbol {x}) = \mathcal {N} (\boldsymbol {z} \mid \boldsymbol {m}, \boldsymbol {C}),\tag{10.73}
$$

$$
\boldsymbol {m} = \boldsymbol {B} ^ {\top} (\boldsymbol {B B} ^ {\top} + \sigma^ {2} \boldsymbol {I}) ^ {- 1} (\boldsymbol {x} - \boldsymbol {\mu}),\tag{10.74}
$$

$$
\boldsymbol {C} = \boldsymbol {I} - \boldsymbol {B} ^ {\top} (\boldsymbol {B B} ^ {\top} + \sigma^ {2} \boldsymbol {I}) ^ {- 1} \boldsymbol {B}.\tag{10.75}
$$

Note that the posterior covariance does not depend on the observed data x. For a new observation ${ \bf { \delta } } _ { \bf { { x } } } ,$ <sub>∗</sub> in data space, we use (10.73) to determine the posterior distribution of the corresponding latent variable $\boldsymbol { z } _ { \ast }$ . The covariance matrix $C$ allows us to assess how confident the embedding is. A covariance matrix $C$ with a small determinant (which measures volumes) tells us that the latent embedding $z _ { * }$ is fairly certain. If we obtain a posterior distribution $p ( z _ { * } \mid x _ { * } )$ with much variance, we may be faced with an outlier. However, we can explore this posterior distribution to understand what other data points x are plausible under this posterior. To do this, we exploit the generative process underlying PPCA, which allows us to explore the posterior distribution on the latent variables by generating new data that is plausible under this posterior: 

1. Sample a latent variable $z _ { * } \sim p ( z | x _ { * } )$ from the posterior distribution over the latent variables (10.73). 

2. Sample a reconstructed vector $\tilde { \mathbf { \em x } } _ { * } \sim p ( \mathbf { \boldsymbol { x } } \mid z _ { * } , B , \mu , \sigma ^ { 2 } )$ from (10.64). 

If we repeat this process many times, we can explore the posterior distribution (10.73) on the latent variables $z _ { \ast }$ and its implications on the observed data. The sampling process effectively hypothesizes data, which is plausible under the posterior distribution. 

## 10.8 Further Reading

We derived PCA from two perspectives: (a) maximizing the variance in the projected space; (b) minimizing the average reconstruction error. However, PCA can also be interpreted from different perspectives. Let us recap what we have done: We took high-dimensional data $\pmb { x } \in \mathbb { R } ^ { D }$ and used a matrix $B ^ { \top }$ to find a lower-dimensional representation $z \in \mathbb { R } ^ { M }$ . The columns of B are the eigenvectors of the data covariance matrix S that are associated with the largest eigenvalues. Once we have a low-dimensional representation z, we can get a high-dimensional version of it (in the original data space) as $\pmb { x } \approx \tilde { \pmb { x } } = \pmb { B } z = \pmb { B } \pmb { B } ^ { \top } \pmb { x } \in \mathbb { R } ^ { D }$ , where $B B ^ { \top }$ is a projection matrix. 

We can also think of PCA as a linear auto-encoder as illustrated in Figure 10.16. An auto-encoder encodes the data ${ \pmb x } _ { n } \in \mathbb { R } ^ { D }$ to a code $\pmb { z } _ { n } \in \mathbb { R } ^ { M }$ and decodes it to a ${ \tilde { \mathbf { x } } } _ { n }$ similar to ${ \bf { x } } _ { n }$ . The mapping from the data to the code is called the encoder, and the mapping from the code back to the original data space is called the decoder. If we consider linear mappings where the code is given by $\pmb { z } _ { n } = \pmb { B } ^ { \top } \pmb { x } _ { n } \in \mathbb { R } ^ { M }$ and we are interested in minimizing the average squared error between the data ${ \bf { x } } _ { n }$ and its reconstruction $\tilde { \boldsymbol { x } } _ { n } = B z _ { n } , n = 1 , . . . , N$ , we obtain 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/8260dcecb908c9d36ce654c143882dc0491ceefcdf5f61a40ac9c7e4517c4633.jpg)


$$
\frac {1}{N} \sum_ {n = 1} ^ {N} \| \boldsymbol {x} _ {n} - \tilde {\boldsymbol {x}} _ {n} \| ^ {2} = \frac {1}{N} \sum_ {n = 1} ^ {N} \left\| \boldsymbol {x} _ {n} - \boldsymbol {B B} ^ {\top} \boldsymbol {x} _ {n} \right\| ^ {2}.\tag{10.76}
$$

This means we end up with the same objective function as in (10.29) that we discussed in Section 10.3 so that we obtain the PCA solution when we minimize the squared auto-encoding loss. If we replace the linear mapping of PCA with a nonlinear mapping, we get a nonlinear auto-encoder. A prominent example of this is a deep auto-encoder where the linear functions are replaced with deep neural networks. In this context, the encoder is also known as a recognition network or inference network, whereas the decoder is also called a generator. 

Another interpretation of PCA is related to information theory. We can think of the code as a smaller or compressed version of the original data point. When we reconstruct our original data using the code, we do not get the exact data point back, but a slightly distorted or noisy version of it. This means that our compression is “lossy”. Intuitively, we want to maximize the correlation between the original data and the lowerdimensional code. More formally, this is related to the mutual information. We would then get the same solution to PCA we discussed in Section 10.3 by maximizing the mutual information, a core concept in information theory (MacKay, 2003). 

In our discussion on PPCA, we assumed that the parameters of the model, i.e., B, µ, and the likelihood parameter $\sigma ^ { 2 } .$ , are known. Tipping and Bishop (1999) describe how to derive maximum likelihood estimates for these parameters in the PPCA setting (note that we use a different notation in this chapter). The maximum likelihood parameters, when projecting D-dimensional data onto an M-dimensional subspace, are 

$$
\pmb {\mu} _ {\mathrm{ML}} = \frac {1}{N} \sum_ {n = 1} ^ {N} \pmb {x} _ {n},\tag{10.77}
$$

$$
\pmb {B} _ {\mathrm{ML}} = \pmb {T} (\pmb {\Lambda} - \sigma^ {2} \pmb {I}) ^ {\frac {1}{2}} \pmb {R},\tag{10.78}
$$

$$
\sigma_ {\mathrm{ML}} ^ {2} = \frac {1}{D - M} \sum_ {j = M + 1} ^ {D} \lambda_ {j},\tag{10.79}
$$

where $\pmb { T } \in \mathbb { R } ^ { D \times M }$ contains M eigenvectors of the data covariance matrix, $\pmb { \Lambda } = \mathrm { d i a g } ( \lambda _ { 1 } , \dots , \lambda _ { M } ) \in \mathbb { R } ^ { M \times M }$ is a diagonal matrix with the eigenvalues associated with the principal axes on its diagonal, and $\pmb { R } \in \mathbf { \bar { \mathbb { R } } } ^ { M \times M }$ is an arbitrary orthogonal matrix. The maximum likelihood solution $B _ { \mathrm { M I } }$ is unique up to an arbitrary orthogonal transformation, e.g., we can rightmultiply $B _ { \mathrm { M L } }$ with any rotation matrix R so that (10.78) essentially is a singular value decomposition (see Section 4.5). An outline of the proof is given by Tipping and Bishop (1999). 

The matrix $\Lambda - \sigma ^ { 2 } I$ in (10.78) is guaranteed to be positive semidefinite as the smallest eigenvalue of the data covariance matrix is bounded from below by the noise variance $\sigma ^ { 2 } ,$ 

The maximum likelihood estimate for $\pmb { \mu }$ given in (10.77) is the sample mean of the data. The maximum likelihood estimator for the observation noise variance $\sigma ^ { 2 }$ given in (10.79) is the average variance in the orthogonal complement of the principal subspace, i.e., the average leftover variance that we cannot capture with the first M principal components is treated as observation noise. 

In the noise-free limit where $\sigma  0$ , PPCA and PCA provide identical solutions: Since the data covariance matrix S is symmetric, it can be diagonalized (see Section 4.4), i.e., there exists a matrix $_ { \pmb { T } }$ of eigenvectors of $_ { s }$ so that 

$$
\pmb {S} = \pmb {T} \pmb {\Lambda} \pmb {T} ^ {- 1}.\tag{10.80}
$$

In the PPCA model, the data covariance matrix is the covariance matrix of the Gaussian likelihood $p ( { \pmb x } | B , { \pmb \mu } , \sigma ^ { 2 } )$ , which is $B B ^ { \top } + \sigma ^ { 2 } I$ , see (10.70b). For $\sigma  0$ , we obtain $B B ^ { \intercal }$ so that this data covariance must equal the PCA data covariance (and its factorization given in (10.80)) so that 

$$
\operatorname{Cov} [ \mathcal {X} ] = \boldsymbol {T} \boldsymbol {\Lambda} \boldsymbol {T} ^ {- 1} = \boldsymbol {B} \boldsymbol {B} ^ {\top} \iff \boldsymbol {B} = \boldsymbol {T} \boldsymbol {\Lambda} ^ {\frac {1}{2}} \boldsymbol {R},\tag{10.81}
$$

i.e., we obtain the maximum likelihood estimate in (10.78) for $\sigma = 0$ From (10.78) and (10.80), it becomes clear that (P)PCA performs a decomposition of the data covariance matrix. 

In a streaming setting, where data arrives sequentially, it is recommended to use the iterative expectation maximization (EM) algorithm for maximum likelihood estimation (Roweis, 1998). 

To determine the dimensionality of the latent variables (the length of the code, the dimensionality of the lower-dimensional subspace onto which we project the data), Gavish and Donoho (2014) suggest the heuristic that, if we can estimate the noise variance $\sigma ^ { 2 }$ of the data, we should discard all singular values smaller than $\frac { 4 \sigma { \sqrt { D } } } { \sqrt { 3 } }$ . Alternatively, we can use (nested) cross-validation (Section 8.6.1) or Bayesian model selection criteria (discussed in Section 8.6.2) to determine a good estimate of the intrinsic dimensionality of the data (Minka, 2001b). 

Similar to our discussion on linear regression in Chapter 9, we can place a prior distribution on the parameters of the model and integrate them out. By doing so, we (a) avoid point estimates of the parameters and the issues that come with these point estimates (see Section 8.6) and (b) allow for an automatic selection of the appropriate dimensionality M of the latent space. In this Bayesian PCA, which was proposed by Bishop (1999), a prior $p ( { \boldsymbol { \mu } } , B , \sigma ^ { 2 } )$ is placed on the model parameters. The generative process allows us to integrate the model parameters out instead of conditioning on them, which addresses overfitting issues. Since this integration is analytically intractable, Bishop (1999) proposes to use approximate inference methods, such as MCMC or variational inference. We refer to the work by Gilks et al. (1996) and Blei et al. (2017) for more details on these approximate inference techniques. 

In PPCA, we considered the linear model $p ( \pmb { x } _ { n } \mid z _ { n } ) = { \mathcal N } \big ( \pmb { x } _ { n } \mid B z _ { n } +$ $\mu , \sigma ^ { 2 } I )$ with prior $p ( z _ { n } ) = \mathcal { N } ( \mathbf { 0 } , I )$ , where all observation dimensions are affected by the same amount of noise. If we allow each observation dimension d to have a different variance $\sigma _ { d } ^ { 2 } .$ , we obtain factor analysis (FA) (Spearman, 1904; Bartholomew et al., 2011). This means that FA gives the likelihood some more flexibility than PPCA, but still forces the data to be explained by the model parameters B, $\pmb { \mu }$ .However, FA no longer allows for a closed-form maximum likelihood solution so that we need to use an iterative scheme, such as the expectation maximization algorithm, to estimate the model parameters. While in PPCA all stationary points are global optima, this no longer holds for FA. Compared to PPCA, FA does not change if we scale the data, but it does return different solutions if we rotate the data. 

An algorithm that is also closely related to PCA is independent component analysis (ICA (Hyvarinen et al., 2001)). Starting again with the latent-variable perspective $p ( { \pmb x } _ { n } \mid z _ { n } ) = { \mathcal { N } } \big ( { \pmb x } _ { n } \mid B z _ { n } + \pmb \mu , \sigma ^ { 2 } { \pmb I } \big )$ we now change the prior on $z _ { n }$ to non-Gaussian distributions. ICA can be used for blind-source separation. Imagine you are in a busy train station with many people talking. Your ears play the role of microphones, and they linearly mix different speech signals in the train station. The goal of blindsource separation is to identify the constituent parts of the mixed signals. As discussed previously in the context of maximum likelihood estimation for PPCA, the original PCA solution is invariant to any rotation. Therefore, PCA can identify the best lower-dimensional subspace in which the signals live, but not the signals themselves (Murphy, 2012). ICA addresses this issue by modifying the prior distribution $p ( z )$ on the latent sources to require non-Gaussian priors $p ( z )$ . We refer to the books by Hyvarinen et al. (2001) and Murphy (2012) for more details on ICA. 

PCA, factor analysis, and ICA are three examples for dimensionality reduction with linear models. Cunningham and Ghahramani (2015) provide a broader survey of linear dimensionality reduction. 

The (P)PCA model we discussed here allows for several important extensions. In Section 10.5, we explained how to do PCA when the input dimensionality D is significantly greater than the number N of data points. By exploiting the insight that PCA can be performed by computing (many) inner products, this idea can be pushed to the extreme by considering infinite-dimensional features. The kernel trick is the basis of kernel PCA and allows us to implicitly compute inner products between infinitedimensional features (Scholkopf et al.¨ , 1998; Scholkopf and Smola¨ , 2002). 

There are nonlinear dimensionality reduction techniques that are derived from PCA (Burges (2010) provides a good overview). The autoencoder perspective of PCA that we discussed previously in this section can be used to render PCA as a special case of a deep auto-encoder. In the deep auto-encoder, both the encoder and the decoder are represented by multilayer feedforward neural networks, which themselves are nonlinear mappings. If we set the activation functions in these neural networks to be the identity, the model becomes equivalent to PCA. A different approach to nonlinear dimensionality reduction is the Gaussian process latent-variable model (GP-LVM) proposed by Lawrence (2005). The GP-LVM starts off with the latent-variable perspective that we used to derive PPCA and replaces the linear relationship between the latent variables z and the observations x with a Gaussian process (GP). Instead of estimating the parameters of the mapping (as we do in PPCA), the GP-LVM marginalizes out the model parameters and makes point estimates of the latent variables z. Similar to Bayesian PCA, the Bayesian GP-LVM proposed by Titsias and Lawrence (2010) maintains a distribution on the latent variables z and uses approximate inference to integrate them out as well. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/bae4bcb8543f13687d93d48ec92bb2ac2ae2f8a700a7c137669b21a2caa1e5e2.jpg)


## Density Estimation with Gaussian Mixture Models

In earlier chapters, we covered already two fundamental problems in machine learning: regression (Chapter 9) and dimensionality reduction (Chapter 10). In this chapter, we will have a look at a third pillar of machine learning: density estimation. On our journey, we introduce important concepts, such as the expectation maximization (EM) algorithm and a latent variable perspective of density estimation with mixture models. 

When we apply machine learning to data we often aim to represent data in some way. A straightforward way is to take the data points themselves as the representation of the data; see Figure 11.1 for an example. However, this approach may be unhelpful if the dataset is huge or if we are interested in representing characteristics of the data. In density estimation, we represent the data compactly using a density from a parametric family, e.g., a Gaussian or Beta distribution. For example, we may be looking for the mean and variance of a dataset in order to represent the data compactly using a Gaussian distribution. The mean and variance can be found using tools we discussed in Section 8.3: maximum likelihood or maximum a posteriori estimation. We can then use the mean and variance of this Gaussian to represent the distribution underlying the data, i.e., we think of the dataset to be a typical realization from this distribution if we were to sample from it. 

Figure 11.1 Two-dimensional dataset that cannot be meaningfully represented by a Gaussian. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/0422efbef503fe0ba8fde49eb86cb72fe973dc0bb412ef4c2e7d494715c7d191.jpg)



This material is published by Cambridge University Press as Mathematics for Machine Learning by Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong (2020). This version is free to view and download for personal use only. Not for re-distribution, re-sale, or use in derivative works. ©by M. P. Deisenroth, A. A. Faisal, and C. S. Ong, 2024. https://mml-book.com.


In practice, the Gaussian (or similarly all other distributions we encountered so far) have limited modeling capabilities. For example, a Gaussian approximation of the density that generated the data in Figure 11.1 would be a poor approximation. In the following, we will look at a more expressive family of distributions, which we can use for density estimation: mixture models. 

Mixture models can be used to describe a distribution $p ( { \pmb x } )$ by a convex combination of K simple (base) distributions 

$$
p (\boldsymbol {x}) = \sum_ {k = 1} ^ {K} \pi_ {k} p _ {k} (\boldsymbol {x})\tag{11.1}
$$

$$
0 \leqslant \pi_ {k} \leqslant 1, \quad \sum_ {k = 1} ^ {K} \pi_ {k} = 1,\tag{11.2}
$$

where the components $p _ { k }$ are members of a family of basic distributions, $\mathsf { e . g . }$ , Gaussians, Bernoullis, or Gammas, and the $\pi _ { k }$ are mixture weights. Mixture models are more expressive than the corresponding base distributions because they allow for multimodal data representations, i.e., they can describe datasets with multiple “clusters”, such as the example in Figure 11.1. 

We will focus on Gaussian mixture models (GMMs), where the basic distributions are Gaussians. For a given dataset, we aim to maximize the likelihood of the model parameters to train the GMM. For this purpose, we will use results from Chapter 5, Chapter 6, and Section 7.2. However, unlike other applications we discussed earlier (linear regression or PCA), we will not find a closed-form maximum likelihood solution. Instead, we will arrive at a set of dependent simultaneous equations, which we can only solve iteratively. 

## 11.1 Gaussian Mixture Model

A Gaussian mixture model is a density model where we combine a finite number of K Gaussian distributions $\mathcal { N } ( \pmb { x } | \pmb { \mu } _ { k } , \pmb { \Sigma } _ { k } )$ so that 

$$
p (\boldsymbol {x} \mid \boldsymbol {\theta}) = \sum_ {k = 1} ^ {K} \pi_ {k} \mathcal {N} \left(\boldsymbol {x} \mid \boldsymbol {\mu} _ {k}, \boldsymbol {\Sigma} _ {k}\right)\tag{11.3}
$$

$$
0 \leqslant \pi_ {k} \leqslant 1, \quad \sum_ {k = 1} ^ {K} \pi_ {k} = 1,\tag{11.4}
$$

where we defined $\pmb \theta : = \{ \pmb \mu _ { k } , \pmb \Sigma _ { k } , \pi _ { k } : k = 1 , \ldots , K \}$ as the collection of all parameters of the model. This convex combination of Gaussian distribution gives us significantly more flexibility for modeling complex densities than a simple Gaussian distribution (which we recover from (11.3) for $K = 1 )$ . An illustration is given in Figure 11.2, displaying the weighted 

Figure 11.2 Gaussian mixture model. The Gaussian mixture distribution (black) is composed of a convex combination of Gaussian distributions and is more expressive than any individual component. Dashed lines represent the weighted Gaussian components. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/eefeceedfc833679fdfb9646cae5c3a1a0bb81ad995ba6afb0b6ed2f364c8d76.jpg)



components and the mixture density, which is given as


$$
p (x \mid \boldsymbol {\theta}) = 0. 5 \mathcal {N} (x \mid - 2, \frac {1}{2}) + 0. 2 \mathcal {N} (x \mid 1, 2) + 0. 3 \mathcal {N} (x \mid 4, 1)\tag{11.5}
$$

## 11.2 Parameter Learning via Maximum Likelihood

Assume we are given a dataset $\mathcal { X } ~ = ~ \{ { \pmb x } _ { 1 } , \ldots , { \pmb x } _ { N } \}$ , where $x _ { n } , \ n \ =$ $1 , \ldots , N ,$ are drawn i.i.d. from an unknown distribution $p ( { \pmb x } )$ . Our objective is to find a good approximation/representation of this unknown distribution $p ( { \pmb x } )$ by means of a GMM with $K$ mixture components. The parameters of the GMM are the $K$ means $\mu _ { k } ,$ , the covariances $\Sigma _ { k } ,$ , and mixture weights $\pi _ { k } .$ . We summarize all these free parameters in $\pmb \theta : =$ $\{ \pi _ { k } , \pmb { \mu } _ { k } , \pmb { \Sigma } _ { k } : ~ k = 1 , \ldots , K \}$ 

## Example 11.1 (Initial Setting)

Figure 11.3 Initial setting: GMM (black) with mixture three mixture components (dashed) and seven data points (discs). 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/487dccad8062eaddf8bc990e1a14b5f8c1b3d59f914869abe7b0f608321be684.jpg)



Throughout this chapter, we will have a simple running example that helps us illustrate and visualize important concepts.


We consider a one-dimensional dataset $\mathcal { X } = \{ - 3 , - 2 . 5 , - 1 , 0 , 2 , 4 , 5 \}$ consisting of seven data points and wish to find a GMM with $K \ : = \ : 3$ components that models the density of the data. We initialize the mixture components as 

$$
p _ {1} (x) = \mathcal {N} (x \mid - 4, 1)\tag{11.6}
$$

$$
p _ {2} (x) = \mathcal {N} (x \mid 0, 0. 2)\tag{11.7}
$$

$$
p _ {3} (x) = \mathcal {N} (x \mid 8, 3)\tag{11.8}
$$

and assign them equal weights $\pi _ { 1 } = \pi _ { 2 } = \pi _ { 3 } = { \frac { 1 } { 3 } }$ . The corresponding model (and the data points) are shown in Figure 11.3. 

In the following, we detail how to obtain a maximum likelihood estimate $\pmb { \theta } _ { \mathrm { M I } }$ of the model parameters θ. We start by writing down the likelihood, i.e., the predictive distribution of the training data given the parameters. We exploit our i.i.d. assumption, which leads to the factorized likelihood 

$$
p (\mathcal {X} \mid \boldsymbol {\theta}) = \prod_ {n = 1} ^ {N} p (\boldsymbol {x} _ {n} \mid \boldsymbol {\theta}), \quad p (\boldsymbol {x} _ {n} \mid \boldsymbol {\theta}) = \sum_ {k = 1} ^ {K} \pi_ {k} \mathcal {N} \left(\boldsymbol {x} _ {n} \mid \boldsymbol {\mu} _ {k}, \boldsymbol {\Sigma} _ {k}\right),\tag{11.9}
$$

where every individual likelihood term $p ( \pmb { x } _ { n } \mid \pmb { \theta } )$ is a Gaussian mixture density. Then we obtain the log-likelihood as 

$$
\log p (\mathcal {X} \mid \boldsymbol {\theta}) = \sum_ {n = 1} ^ {N} \log p (\boldsymbol {x} _ {n} \mid \boldsymbol {\theta}) = \underbrace {\sum_ {n = 1} ^ {N} \log \sum_ {k = 1} ^ {K} \pi_ {k} \mathcal {N} \left(\boldsymbol {x} _ {n} \mid \boldsymbol {\mu} _ {k} , \boldsymbol {\Sigma} _ {k}\right)} _ {=: \mathcal {L}}.\tag{11.10}
$$

We aim to find parameters ${ \pmb \theta } _ { \mathrm { M I } } ^ { * }$ that maximize the log-likelihood $\mathcal { L }$ defined in (11.10). Our “normal” procedure would be to compute the gradient $\mathrm { d } { \mathcal { L } } / \mathrm { d } \theta$ of the log-likelihood with respect to the model parameters $\theta ,$ set it to $\mathbf { 0 , }$ and solve for θ. However, unlike our previous examples for maximum likelihood estimation (e.g., when we discussed linear regression in Section 9.2), we cannot obtain a closed-form solution. However, we can exploit an iterative scheme to find good model parameters $\pmb { \theta } _ { \mathrm { M L } }$ , which will turn out to be the EM algorithm for GMMs. The key idea is to update one model parameter at a time while keeping the others fixed. 

Remark. If we were to consider a single Gaussian as the desired density, the sum over $k$ in (11.10) vanishes, and the log can be applied directly to the Gaussian component, such that we get 

$$
\log \mathcal {N} (\boldsymbol {x} \mid \boldsymbol {\mu}, \boldsymbol {\Sigma}) = - \frac {D}{2} \log (2 \pi) - \frac {1}{2} \log \det (\boldsymbol {\Sigma}) - \frac {1}{2} (\boldsymbol {x} - \boldsymbol {\mu}) ^ {\top} \boldsymbol {\Sigma} ^ {- 1} (\boldsymbol {x} - \boldsymbol {\mu}).\tag{11.11}
$$

This simple form allows us to find closed-form maximum likelihood estimates of $\pmb { \mu }$ and $\Sigma ,$ as discussed in Chapter 8. In (11.10), we cannot move the log into the sum over k so that we cannot obtain a simple closed-form maximum likelihood solution. ◇ 

Any local optimum of a function exhibits the property that its gradient with respect to the parameters must vanish (necessary condition); see Chapter 7. In our case, we obtain the following necessary conditions when we optimize the log-likelihood in (11.10) with respect to the GMM parameters $\mu _ { k } , \Sigma _ { k } , \pi _ { k }$ 

$$
\frac {\partial \mathcal {L}}{\partial \boldsymbol {\mu} _ {k}} = \mathbf {0} ^ {\top} \iff \sum_ {n = 1} ^ {N} \frac {\partial \log p (\boldsymbol {x} _ {n} \mid \boldsymbol {\theta})}{\partial \boldsymbol {\mu} _ {k}} = \mathbf {0} ^ {\top},\tag{11.12}
$$

$$
\frac {\partial \mathcal {L}}{\partial \boldsymbol {\Sigma} _ {k}} = \mathbf {0} \iff \sum_ {n = 1} ^ {N} \frac {\partial \log p (\boldsymbol {x} _ {n} \mid \boldsymbol {\theta})}{\partial \boldsymbol {\Sigma} _ {k}} = \mathbf {0},\tag{11.13}
$$

$$
\frac {\partial \mathcal {L}}{\partial \pi_ {k}} = 0 \iff \sum_ {n = 1} ^ {N} \frac {\partial \log p (\boldsymbol {x} _ {n} \mid \boldsymbol {\theta})}{\partial \pi_ {k}} = 0.\tag{11.14}
$$

For all three necessary conditions, by applying the chain rule (see Section 5.2.2), we require partial derivatives of the form 

$$
\frac {\partial \log p (\boldsymbol {x} _ {n} \mid \boldsymbol {\theta})}{\partial \boldsymbol {\theta}} = \frac {1}{p (\boldsymbol {x} _ {n} \mid \boldsymbol {\theta})} \frac {\partial p (\boldsymbol {x} _ {n} \mid \boldsymbol {\theta})}{\partial \boldsymbol {\theta}},\tag{11.15}
$$

where $\pmb \theta = \{ \pmb \mu _ { k } , \pmb \Sigma _ { k } , \pi _ { k } , k = 1 , \ldots , K \}$ are the model parameters and 

$$
\frac {1}{p (\boldsymbol {x} _ {n} \mid \boldsymbol {\theta})} = \frac {1}{\sum_ {j = 1} ^ {K} \pi_ {j} \mathcal {N} (\boldsymbol {x} _ {n} \mid \boldsymbol {\mu} _ {j} , \boldsymbol {\Sigma} _ {j})}.\tag{11.16}
$$

In the following, we will compute the partial derivatives (11.12) through (11.14). But before we do this, we introduce a quantity that will play a central role in the remainder of this chapter: responsibilities. 

## 11.2.1 Responsibilities

We define the quantity 

$$
r _ {n k} := \frac {\pi_ {k} \mathcal {N} (\boldsymbol {x} _ {n} \mid \boldsymbol {\mu} _ {k} , \boldsymbol {\Sigma} _ {k})}{\sum_ {j = 1} ^ {K} \pi_ {j} \mathcal {N} (\boldsymbol {x} _ {n} \mid \boldsymbol {\mu} _ {j} , \boldsymbol {\Sigma} _ {j})}\tag{11.17}
$$

as the responsibility of the kth mixture component for the nth data point. The responsibility $r _ { n k }$ of the kth mixture component for data point ${ \bf { x } } _ { n }$ is proportional to the likelihood 

$$
p (\boldsymbol {x} _ {n} \mid \pi_ {k}, \boldsymbol {\mu} _ {k}, \boldsymbol {\Sigma} _ {k}) = \pi_ {k} \mathcal {N} \big (\boldsymbol {x} _ {n} \mid \boldsymbol {\mu} _ {k}, \boldsymbol {\Sigma} _ {k} \big)\tag{11.18}
$$

of the mixture component given the data point. Therefore, mixture components have a high responsibility for a data point when the data point could be a plausible sample from that mixture component. Note that $\pmb { r } _ { n } : = [ r _ { n 1 } , \ldots , r _ { n K } ] ^ { \top } \in \bar { \mathbb { R } } ^ { K }$ is a (normalized) probability vector, i.e., $\textstyle \sum _ { k } r _ { n k } = 1$ with $r _ { n k } \ \geqslant \ 0$ . This probability vector distributes probability mass among the K mixture components, and we can think of $\boldsymbol { r } _ { n }$ as a “soft assignment” of ${ \pmb x } _ { n }$ to the K mixture components. Therefore, the responsibility $r _ { n k }$ from (11.17) represents the probability that ${ \bf { x } } _ { n }$ has been generated by the kth mixture component. 

## Example 11.2 (Responsibilities)

For our example from Figure 11.3, we compute the responsibilities $r _ { n k }$ 

$$
\left[ \begin{array}{c c c} 1. 0 & 0. 0 & 0. 0 \\ 1. 0 & 0. 0 & 0. 0 \\ 0. 0 5 7 & 0. 9 4 3 & 0. 0 \\ 0. 0 0 1 & 0. 9 9 9 & 0. 0 \\ 0. 0 & 0. 0 6 6 & 0. 9 3 4 \\ 0. 0 & 0. 0 & 1. 0 \\ 0. 0 & 0. 0 & 1. 0 \end{array} \right] \in \mathbb {R} ^ {N \times K}.\tag{11.19}
$$

Here the nth row tells us the responsibilities of all mixture components for $x _ { n } .$ The sum of all K responsibilities for a data point (sum of every row) is 1. The kth column gives us an overview of the responsibility of the kth mixture component. We can see that the third mixture component (third column) is not responsible for any of the first four data points, but takes much responsibility of the remaining data points. The sum of all entries of a column gives us the values $N _ { k }$ , i.e., the total responsibility of the kth mixture component. In our example, we get $N _ { 1 } = 2 . 0 5 8$ $N _ { 2 } =$ $2 . 0 0 8 , N _ { 3 } = 2 . 9 3 4$ 

In the following, we determine the updates of the model parameters $\mu _ { k } , \Sigma _ { k } , \pi _ { k }$ for given responsibilities. We will see that the update equations all depend on the responsibilities, which makes a closed-form solution to the maximum likelihood estimation problem impossible. However, for given responsibilities we will be updating one model parameter at a time, while keeping the others fixed. After this, we will recompute the responsibilities. Iterating these two steps will eventually converge to a local optimum and is a specific instantiation of the EM algorithm. We will discuss this in some more detail in Section 11.3. 

## 11.2.2 Updating the Means

Theorem 11.1 (Update of the GMM Means). The update of the mean parameters $\begin{array} { r } { \mu _ { k } , k = 1 , \ldots , K , } \end{array}$ , of the GMM is given by 

$$
\pmb {\mu} _ {k} ^ {n e w} = \frac {\sum_ {n = 1} ^ {N} r _ {n k} \pmb {x} _ {n}}{\sum_ {n = 1} ^ {N} r _ {n k}},\tag{11.20}
$$

where the responsibilities $r _ { n k }$ are defined in (11.17). 

©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 

Remark. The update of the means $\pmb { \mu } _ { k }$ of the individual mixture components in (11.20) depends on all means, covariance matrices $\Sigma _ { k }$ , and mixture weights $\pi _ { k }$ via $r _ { n k }$ given in (11.17). Therefore, we cannot obtain a closed-form solution for all $\pmb { \mu } _ { k }$ at once. 

Proof From (11.15), we see that the gradient of the log-likelihood with respect to the mean parameters $\pmb { \mu } _ { k } , k = 1 , \ldots , K$ , requires us to compute the partial derivative 

$$
\begin{array}{l} \frac {\partial p (\boldsymbol {x} _ {n} \mid \boldsymbol {\theta})}{\partial \boldsymbol {\mu} _ {k}} = \sum_ {j = 1} ^ {K} \pi_ {j} \frac {\partial \mathcal {N} (\boldsymbol {x} _ {n} \mid \boldsymbol {\mu} _ {j} , \boldsymbol {\Sigma} _ {j})}{\partial \boldsymbol {\mu} _ {k}} = \pi_ {k} \frac {\partial \mathcal {N} (\boldsymbol {x} _ {n} \mid \boldsymbol {\mu} _ {k} , \boldsymbol {\Sigma} _ {k})}{\partial \boldsymbol {\mu} _ {k}} \\ = \pi_ {k} (\boldsymbol {x} _ {n} - \boldsymbol {\mu} _ {k}) ^ {\top} \boldsymbol {\Sigma} _ {k} ^ {- 1} \mathcal {N} (\boldsymbol {x} _ {n} \mid \boldsymbol {\mu} _ {k}, \boldsymbol {\Sigma} _ {k}), \end{array}\tag{11.21a}
$$

(11.21b) 

where we exploited that only the kth mixture component depends on $\pmb { \mu } _ { k }$ We use our result from (11.21b) in (11.15) and put everything together so that the desired partial derivative of with respect to $\pmb { \mu } _ { k }$ is given as 

$$
\frac {\partial \mathcal {L}}{\partial \boldsymbol {\mu} _ {k}} = \sum_ {n = 1} ^ {N} \frac {\partial \log p (\boldsymbol {x} _ {n} | \boldsymbol {\theta})}{\partial \boldsymbol {\mu} _ {k}} = \sum_ {n = 1} ^ {N} \frac {1}{p (\boldsymbol {x} _ {n} | \boldsymbol {\theta})} \frac {\partial p (\boldsymbol {x} _ {n} | \boldsymbol {\theta})}{\partial \boldsymbol {\mu} _ {k}}\tag{11.22a}
$$

(11.22b) 

$$
= \sum_ {n = 1} ^ {N} r _ {n k} (\pmb {x} _ {n} - \pmb {\mu} _ {k}) ^ {\top} \pmb {\Sigma} _ {k} ^ {- 1}.\tag{11.22c}
$$

Here we used the identity from (11.16) and the result of the partial derivative in (11.21b) to get to (11.22b). The values $r _ { n k }$ are the responsibilities we defined in (11.17). 

We now solve (11.22c) for $\mu _ { k } ^ { \mathrm { n e w } }$ so that $\begin{array} { r } { \frac { \partial \mathcal { L } ( \boldsymbol { \mu } _ { k } ^ { \mathrm { n e w } } ) } { \partial \boldsymbol { \mu } _ { k } } = \mathbf { 0 } ^ { \top } } \end{array}$ and obtain 

$$
\sum_ {n = 1} ^ {N} r _ {n k} \boldsymbol {x} _ {n} = \sum_ {n = 1} ^ {N} r _ {n k} \boldsymbol {\mu} _ {k} ^ {\text { new }} \iff \boldsymbol {\mu} _ {k} ^ {\text { new }} = \frac {\sum_ {n = 1} ^ {N} r _ {n k} \boldsymbol {x} _ {n}}{\boxed {\sum_ {n = 1} ^ {N} r _ {n k}}} = \frac {1}{\boxed {N _ {k}}} \sum_ {n = 1} ^ {N} r _ {n k} \boldsymbol {x} _ {n},\tag{11.23}
$$

where we defined 

$$
N _ {k} := \sum_ {n = 1} ^ {N} r _ {n k}\tag{11.24}
$$

as the total responsibility of the kth mixture component for the entire dataset. This concludes the proof of Theorem 11.1. □ 

Intuitively, (11.20) can be interpreted as an importance-weighted Monte Carlo estimate of the mean, where the importance weights of data point ${ \bf { x } } _ { n }$ are the responsibilities $r _ { n k }$ of the kth cluster for $\mathbf { { x } } _ { n } , k = 1 , \ldots , K$ 

Therefore, the mean $\pmb { \mu } _ { k }$ is pulled toward a data point ${ \bf { x } } _ { n }$ with strength given by $r _ { n k }$ . The means are pulled stronger toward data points for which the corresponding mixture component has a high responsibility, i.e., a high likelihood. Figure 11.4 illustrates this. We can also interpret the mean update in (11.20) as the expected value of all data points under the distribution given by 

$$
\boldsymbol {r} _ {k} := \left[ r _ {1 k}, \dots , r _ {N k} \right] ^ {\top} / N _ {k},\tag{11.25}
$$

which is a normalized probability vector, i.e., 

Figure 11.4 Update of the mean parameter of mixture component in a GMM. The mean µ is being pulled toward individual data points with the weights given by the corresponding responsibilities. 

$$
\boldsymbol {\mu} _ {k} \leftarrow \mathbb {E} _ {\boldsymbol {r} _ {k}} [ \mathcal {X} ].\tag{11.26}
$$

## Example 11.3 (Mean Updates)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/172b7917e387d04be6977f60926a1a44e2a2997ea4f4126d436a5f0978c5ca0d.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/46a39cb68c4bc548403bb13e47f7a1e32cea362f138aa736f587a5e1cf48e502.jpg)



(a) GMM density and individual components prior to updating the mean values.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/467f69d59ef74a4c41c1c48e9ea0591b91dd1b5146a87520582eff2b6f17d601.jpg)



(b) GMM density and individual components after updating the mean values.



Figure 11.5 Effect of updating the mean values in a GMM. (a) GMM before updating the mean values; (b) GMM after updating the mean values $\mu _ { k }$ while retaining the variances and mixture weights.


In our example from Figure 11.3, the mean values are updated as follows: 

$$
\mu_ {1}: - 4 \to - 2. 7\tag{11.27}
$$

$$
\mu_ {2}: 0 \to - 0. 4\tag{11.28}
$$

$$
\mu_ {3}: 8 \to 3. 7\tag{11.29}
$$

Here we see that the means of the first and third mixture component move toward the regime of the data, whereas the mean of the second component does not change so dramatically. Figure 11.5 illustrates this change, where Figure 11.5(a) shows the GMM density prior to updating the means and Figure 11.5(b) shows the GMM density after updating the mean values $\mu _ { k }$ 

The update of the mean parameters in (11.20) look fairly straight forward. However, note that the responsibilities $r _ { n k }$ are a function of $\pi _ { j } , \mu _ { j } , \Sigma _ { j }$ for all $j = 1 , \ldots , K$ , such that the updates in (11.20) depend on all parameters of the GMM, and a closed-form solution, which we obtained for linear regression in Section 9.2 or PCA in Chapter 10, cannot be obtained. 

©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 

## 11.2.3 Updating the Covariances

Theorem 11.2 (Updates of the GMM Covariances). The update of the covariance parameters $\Sigma _ { k } , k = 1 , \dots , K$ of the GMM is given by 

$$
\boldsymbol {\Sigma} _ {k} ^ {n e w} = \frac {1}{N _ {k}} \sum_ {n = 1} ^ {N} r _ {n k} (\boldsymbol {x} _ {n} - \boldsymbol {\mu} _ {k}) (\boldsymbol {x} _ {n} - \boldsymbol {\mu} _ {k}) ^ {\top},\tag{11.30}
$$

where $r _ { n k }$ and $N _ { k }$ are defined in (11.17) and (11.24), respectively. 

Proof To prove Theorem 11.2, our approach is to compute the partial derivatives of the log-likelihood $\mathcal { L }$ with respect to the covariances $\Sigma _ { k }$ , set them to 0, and solve for $\Sigma _ { k }$ . We start with our general approach 

$$
\frac {\partial \mathcal {L}}{\partial \boldsymbol {\Sigma} _ {k}} = \sum_ {n = 1} ^ {N} \frac {\partial \log p (\boldsymbol {x} _ {n} \mid \boldsymbol {\theta})}{\partial \boldsymbol {\Sigma} _ {k}} = \sum_ {n = 1} ^ {N} \frac {1}{p (\boldsymbol {x} _ {n} \mid \boldsymbol {\theta})} \frac {\partial p (\boldsymbol {x} _ {n} \mid \boldsymbol {\theta})}{\partial \boldsymbol {\Sigma} _ {k}}.\tag{11.31}
$$

We already know $1 / p ( { \pmb x } _ { n } \mid { \pmb \theta } )$ from (11.16). To obtain the remaining partial derivative $\partial p ( { \pmb x } _ { n } \mid \pmb \theta ) / \partial { \pmb \Sigma } _ { k }$ , we write down the definition of the Gaussian distribution $p ( \pmb { x } _ { n } \mid \pmb { \theta } )$ (see (11.9)) and drop all terms but the kth. We then obtain 

$$
\begin{array}{l} \frac {\partial p (\boldsymbol {x} _ {n} \mid \boldsymbol {\theta})}{\partial \boldsymbol {\Sigma} _ {k}} \\ = \frac {\partial}{\partial \boldsymbol {\Sigma} _ {k}} \left(\pi_ {k} (2 \pi) ^ {- \frac {D}{2}} \det (\boldsymbol {\Sigma} _ {k}) ^ {- \frac {1}{2}} \exp \left(- \frac {1}{2} (\boldsymbol {x} _ {n} - \boldsymbol {\mu} _ {k}) ^ {\top} \boldsymbol {\Sigma} _ {k} ^ {- 1} (\boldsymbol {x} _ {n} - \boldsymbol {\mu} _ {k})\right)\right) \end{array} \tag {11.32a}\tag{11.32b}
$$

$$
= \pi_ {k} (2 \pi) ^ {- \frac {D}{2}} \left[ \frac {\partial}{\partial \boldsymbol {\Sigma} _ {k}} \det (\boldsymbol {\Sigma} _ {k}) ^ {- \frac {1}{2}} \exp \left(- \frac {1}{2} (\boldsymbol {x} _ {n} - \boldsymbol {\mu} _ {k}) ^ {\top} \boldsymbol {\Sigma} _ {k} ^ {- 1} (\boldsymbol {x} _ {n} - \boldsymbol {\mu} _ {k})\right) \right.
$$

$$
\left. + \det (\pmb {\Sigma} _ {k}) ^ {- \frac {1}{2}} \frac {\partial}{\partial \pmb {\Sigma} _ {k}} \exp \left(- \frac {1}{2} (\pmb {x} _ {n} - \pmb {\mu} _ {k}) ^ {\top} \pmb {\Sigma} _ {k} ^ {- 1} (\pmb {x} _ {n} - \pmb {\mu} _ {k})\right) \right].\tag{11.32c}
$$

We now use the identities 

$$
\frac {\partial}{\partial \boldsymbol {\Sigma} _ {k}} \det (\boldsymbol {\Sigma} _ {k}) ^ {- \frac {1}{2} (5. 1 0 1)} - \frac {1}{2} \det (\boldsymbol {\Sigma} _ {k}) ^ {- \frac {1}{2}} \boldsymbol {\Sigma} _ {k} ^ {- 1},\tag{11.33}
$$

$$
\frac {\partial}{\partial \boldsymbol {\Sigma} _ {k}} (\boldsymbol {x} _ {n} - \boldsymbol {\mu} _ {k}) ^ {\top} \boldsymbol {\Sigma} _ {k} ^ {- 1} (\boldsymbol {x} _ {n} - \boldsymbol {\mu} _ {k}) \stackrel {(5. 1 0 3)} {=} - \boldsymbol {\Sigma} _ {k} ^ {- 1} (\boldsymbol {x} _ {n} - \boldsymbol {\mu} _ {k}) (\boldsymbol {x} _ {n} - \boldsymbol {\mu} _ {k}) ^ {\top} \boldsymbol {\Sigma} _ {k} ^ {- 1}\tag{11.34}
$$

and obtain (after some rearranging) the desired partial derivative required in (11.31) as 

$$
\begin{array}{l} \frac {\partial p (\boldsymbol {x} _ {n} \mid \boldsymbol {\theta})}{\partial \boldsymbol {\Sigma} _ {k}} = \pi_ {k} \mathcal {N} \big (\boldsymbol {x} _ {n} \mid \boldsymbol {\mu} _ {k}, \boldsymbol {\Sigma} _ {k} \big) \\ \qquad \cdot \left[ - \frac {1}{2} (\boldsymbol {\Sigma} _ {k} ^ {- 1} - \boldsymbol {\Sigma} _ {k} ^ {- 1} (\boldsymbol {x} _ {n} - \boldsymbol {\mu} _ {k}) (\boldsymbol {x} _ {n} - \boldsymbol {\mu} _ {k}) ^ {\top} \boldsymbol {\Sigma} _ {k} ^ {- 1}) \right]. \end{array}\tag{11.35}
$$

Putting everything together, the partial derivative of the log-likelihood 

with respect to $\Sigma _ { k }$ is given by 

$$
\begin{array}{r l} \frac {\partial \mathcal {L}}{\partial \boldsymbol {\Sigma} _ {k}} & = \sum_ {n = 1} ^ {N} \frac {\partial \log p (\boldsymbol {x} _ {n} \mid \boldsymbol {\theta})}{\partial \boldsymbol {\Sigma} _ {k}} = \sum_ {n = 1} ^ {N} \frac {1}{p (\boldsymbol {x} _ {n} \mid \boldsymbol {\theta})} \frac {\partial p (\boldsymbol {x} _ {n} \mid \boldsymbol {\theta})}{\partial \boldsymbol {\Sigma} _ {k}} \\ & = \sum_ {n = 1} ^ {N} \underbrace {\frac {\pi_ {k} \mathcal {N} (\boldsymbol {x} _ {n} \mid \boldsymbol {\mu} _ {k} , \boldsymbol {\Sigma} _ {k})}{\sum_ {j = 1} ^ {K} \pi_ {j} \mathcal {N} (\boldsymbol {x} _ {n} \mid \boldsymbol {\mu} _ {j} , \boldsymbol {\Sigma} _ {j})}} _ {= r _ {n k}} \\ & \cdot \left[ - \frac {1}{2} (\boldsymbol {\Sigma} _ {k} ^ {- 1} - \boldsymbol {\Sigma} _ {k} ^ {- 1} (\boldsymbol {x} _ {n} - \boldsymbol {\mu} _ {k}) (\boldsymbol {x} _ {n} - \boldsymbol {\mu} _ {k}) ^ {\top} \boldsymbol {\Sigma} _ {k} ^ {- 1}) \right] \\ & = - \frac {1}{2} \sum_ {n = 1} ^ {N} r _ {n k} (\boldsymbol {\Sigma} _ {k} ^ {- 1} - \boldsymbol {\Sigma} _ {k} ^ {- 1} (\boldsymbol {x} _ {n} - \boldsymbol {\mu} _ {k}) (\boldsymbol {x} _ {n} - \boldsymbol {\mu} _ {k}) ^ {\top} \boldsymbol {\Sigma} _ {k} ^ {- 1}) \\ & = - \frac {1}{2} \boldsymbol {\Sigma} _ {k} ^ {- 1} \underbrace {\sum_ {n = 1} ^ {N} r _ {n k}} _ {= N _ {k}} + \frac {1}{2} \boldsymbol {\Sigma} _ {k} ^ {- 1} \left(\sum_ {n = 1} ^ {N} r _ {n k} (\boldsymbol {x} _ {n} - \boldsymbol {\mu} _ {k}) (\boldsymbol {x} _ {n} - \boldsymbol {\mu} _ {k}) ^ {\top}\right) \boldsymbol {\Sigma} _ {k} ^ {- 1}. \end{array} \tag {11.36a}\tag{11.36d}
$$

We see that the responsibilities $r _ { n k }$ also appear in this partial derivative. Setting this partial derivative to 0, we obtain the necessary optimality condition 

$$
N _ {k} \boldsymbol {\Sigma} _ {k} ^ {- 1} = \boldsymbol {\Sigma} _ {k} ^ {- 1} \left(\sum_ {n = 1} ^ {N} r _ {n k} (\boldsymbol {x} _ {n} - \boldsymbol {\mu} _ {k}) (\boldsymbol {x} _ {n} - \boldsymbol {\mu} _ {k}) ^ {\top}\right) \boldsymbol {\Sigma} _ {k} ^ {- 1}\tag{11.37a}
$$

$$
\Longleftrightarrow N _ {k} \boldsymbol {I} = \left(\sum_ {n = 1} ^ {N} r _ {n k} (\boldsymbol {x} _ {n} - \boldsymbol {\mu} _ {k}) (\boldsymbol {x} _ {n} - \boldsymbol {\mu} _ {k}) ^ {\top}\right) \boldsymbol {\Sigma} _ {k} ^ {- 1}.\tag{11.37b}
$$

By solving for $\Sigma _ { k }$ , we obtain 

$$
\boldsymbol {\Sigma} _ {k} ^ {\text { new }} = \frac {1}{N _ {k}} \sum_ {n = 1} ^ {N} r _ {n k} (\boldsymbol {x} _ {n} - \boldsymbol {\mu} _ {k}) (\boldsymbol {x} _ {n} - \boldsymbol {\mu} _ {k}) ^ {\top},\tag{11.38}
$$

where $\boldsymbol { r } _ { k }$ is the probability vector defined in (11.25). This gives us a simple update rule for $\Sigma _ { k }$ for $k = 1 , \ldots , K$ and proves Theorem 11.2. 

Similar to the update of $\pmb { \mu } _ { k }$ in (11.20), we can interpret the update of the covariance in (11.30) as an importance-weighted expected value of the square of the centered data $\tilde { \mathcal { X } } _ { k } : = \{ \pmb { x } _ { 1 } - \pmb { \mu } _ { k } , \ldots , \pmb { x } _ { N } - \pmb { \mu } _ { k } \}$ 

## Example 11.4 (Variance Updates)

In our example from Figure 11.3, the variances are updated as follows: 

$$
\sigma_ {1} ^ {2}: 1 \to 0. 1 4\tag{11.39}
$$

$$
\sigma_ {2} ^ {2}: 0. 2 \to 0. 4 4
$$

$$
\sigma_ {3} ^ {2}: 3 \to 1. 5 3\tag{11.40}
$$

(11.41) 



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



Here we see that the variances of the first and third component shrink significantly, whereas the variance of the second component increases slightly. 

Figure 11.6 illustrates this setting. Figure 11.6(a) is identical (but zoomed in) to Figure 11.5(b) and shows the GMM density and its individual components prior to updating the variances. Figure 11.6(b) shows the GMM density after updating the variances. 

Figure 11.6 Effect of updating the variances in a GMM. (a) GMM before updating the variances; (b) GMM after updating the variances while retaining the means and mixture weights. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/64e1e0656631f603180af0b51104ad973d6f02cc480a6e072ad9bfb96ab07ba3.jpg)



(a) GMM density and individual components prior to updating the variances.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/b281089e218ed0ec79114043290141b478d7422ddbecb9e89cec64fa527bb9a3.jpg)



(b) GMM density and individual components after updating the variances.


Similar to the update of the mean parameters, we can interpret (11.30) as a Monte Carlo estimate of the weighted covariance of data points ${ \bf { x } } _ { n }$ associated with the kth mixture component, where the weights are the responsibilities $r _ { n k }$ . As with the updates of the mean parameters, this update depends on all $\pi _ { j } , \mu _ { j } , \Sigma _ { j } , \ j = 1 , \ldots , K$ , through the responsibilities $r _ { n k : }$ , which prohibits a closed-form solution. 

## 11.2.4 Updating the Mixture Weights

Theorem 11.3 (Update of the GMM Mixture Weights). The mixture weights of the GMM are updated as 

$$
\pi_ {k} ^ {\text {new}} = \frac {N _ {k}}{N}, k = 1, \dots , K,\tag{11.42}
$$

where N is the number of data points and $N _ { k }$ is defined in (11.24). 

Proof To find the partial derivative of the log-likelihood with respect to the weight parameters $\pi _ { k } , \ k \ = \ 1 , \ldots , K$ , we account for the constraint $\textstyle \sum _ { k } \pi _ { k } = 1$ by using Lagrange multipliers (see Section 7.2). The Lagrangian is 

$$
\mathfrak {L} = \mathcal {L} + \lambda \left(\sum_ {k = 1} ^ {K} \pi_ {k} - 1\right)\tag{11.43a}
$$

Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



$$
= \sum_ {n = 1} ^ {N} \log \sum_ {k = 1} ^ {K} \pi_ {k} \mathcal {N} \left(\boldsymbol {x} _ {n} \mid \boldsymbol {\mu} _ {k}, \boldsymbol {\Sigma} _ {k}\right) + \lambda \left(\sum_ {k = 1} ^ {K} \pi_ {k} - 1\right),\tag{11.43b}
$$

where $\mathcal { L }$ is the log-likelihood from (11.10) and the second term encodes for the equality constraint that all the mixture weights need to sum up to 1. We obtain the partial derivative with respect to $\pi _ { k }$ as 

$$
\frac {\partial \mathfrak {L}}{\partial \pi_ {k}} = \sum_ {n = 1} ^ {N} \frac {\mathcal {N} (\boldsymbol {x} _ {n} \mid \boldsymbol {\mu} _ {k} , \boldsymbol {\Sigma} _ {k})}{\sum_ {j = 1} ^ {K} \pi_ {j} \mathcal {N} (\boldsymbol {x} _ {n} \mid \boldsymbol {\mu} _ {j} , \boldsymbol {\Sigma} _ {j})} + \lambda\tag{11.44a}
$$

$$
= \frac {1}{\pi_ {k}} \underbrace {\sum_ {n = 1} ^ {N} \frac {\pi_ {k} \mathcal {N} \left(\boldsymbol {x} _ {n} \mid \boldsymbol {\mu} _ {k} , \boldsymbol {\Sigma} _ {k}\right)}{\sum_ {j = 1} ^ {K} \pi_ {j} \mathcal {N} \left(\boldsymbol {x} _ {n} \mid \boldsymbol {\mu} _ {j} , \boldsymbol {\Sigma} _ {j}\right)}} _ {= N _ {k}} + \lambda = \frac {N _ {k}}{\pi_ {k}} + \lambda ,\tag{11.44b}
$$

and the partial derivative with respect to the Lagrange multiplier λ as 

$$
\frac {\partial \mathfrak {L}}{\partial \lambda} = \sum_ {k = 1} ^ {K} \pi_ {k} - 1.\tag{11.45}
$$

Setting both partial derivatives to 0 (necessary condition for optimum) yields the system of equations 

$$
\pi_ {k} = - \frac {N _ {k}}{\lambda},\tag{11.46}
$$

$$
1 = \sum_ {k = 1} ^ {K} \pi_ {k}.\tag{11.47}
$$

Using (11.46) in (11.47) and solving for $\pi _ { k } ,$ , we obtain 

$$
\sum_ {k = 1} ^ {K} \pi_ {k} = 1 \iff - \sum_ {k = 1} ^ {K} \frac {N _ {k}}{\lambda} = 1 \iff - \frac {N}{\lambda} = 1 \iff \lambda = - N.\tag{11.48}
$$

This allows us to substitute N for λ in (11.46) to obtain 

$$
\pi_ {k} ^ {\mathrm{new}} = \frac {N _ {k}}{N},\tag{11.49}
$$

which gives us the update for the weight parameters $\pi _ { k }$ and proves Theorem 11.3. □ 

We can identify the mixture weight in (11.42) as the ratio of the total responsibility of the kth cluster and the number of data points. Since $\begin{array} { r } { N = \sum _ { k } N _ { k } } \end{array}$ , the number of data points can also be interpreted as the total responsibility of all mixture components together, such that $\pi _ { k }$ is the relative importance of the kth mixture component for the dataset. 

Remark. Since $\begin{array} { r } { N _ { k } = \sum _ { i = 1 } ^ { N } r _ { n k } } \end{array}$ , the update equation (11.42) for the mixture weights $\pi _ { k }$ also depends on all $\pi _ { j } , \mu _ { j } , \Sigma _ { j } , j = 1 , \ldots , K$ via the responsibilities $r _ { n k }$ 

Figure 11.7 Effect of updating the mixture weights in a GMM. (a) GMM before updating the mixture weights; (b) GMM after updating the mixture weights while retaining the means and variances. Note the different scales of the vertical axes. 

## Example 11.5 (Weight Parameter Updates)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/1e078bfb52a25091656868f3da979f5375952bf8fac204fb5332aa8e3724d19b.jpg)



(a) GMM density and individual components prior to updating the mixture weights.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/9873bb1f840a2336a7bb6e8c9408761b3fbbd359494eb698647944d89d0b3d83.jpg)



(b) GMM density and individual components after updating the mixture weights.


In our running example from Figure 11.3, the mixture weights are updated as follows: 

$$
\pi_ {1}: \frac {1}{3} \to 0. 2 9\tag{11.50}
$$

$$
\pi_ {2}: \frac {1}{3} \to 0. 2 9\tag{11.51}
$$

$$
\pi_ {3}: \frac {1}{3} \to 0. 4 2\tag{11.52}
$$

Here we see that the third component gets more weight/importance, while the other components become slightly less important. Figure 11.7 illustrates the effect of updating the mixture weights. Figure 11.7(a) is identical to Figure 11.6(b) and shows the GMM density and its individual components prior to updating the mixture weights. Figure 11.7(b) shows the GMM density after updating the mixture weights. 

Overall, having updated the means, the variances, and the weights once, we obtain the GMM shown in Figure 11.7(b). Compared with the initialization shown in Figure 11.3, we can see that the parameter updates caused the GMM density to shift some of its mass toward the data points. 

After updating the means, variances, and weights once, the GMM fit in Figure 11.7(b) is already remarkably better than its initialization from Figure 11.3. This is also evidenced by the log-likelihood values, which increased from 28.3 (initialization) to 14.4 after a full update cycle. 

## 11.3 EM Algorithm

Unfortunately, the updates in (11.20), (11.30), and (11.42) do not constitute a closed-form solution for the updates of the parameters $\mu _ { k } , \Sigma _ { k } , \pi _ { k }$ of the mixture model because the responsibilities $r _ { n k }$ depend on those parameters in a complex way. However, the results suggest a simple iterative scheme for finding a solution to the parameters estimation problem via maximum likelihood. The expectation maximization algorithm (EM algorithm) was proposed by Dempster et al. (1977) and is a general iterative scheme for learning parameters (maximum likelihood or MAP) in mixture models and, more generally, latent-variable models. 

In our example of the Gaussian mixture model, we choose initial values for $\mu _ { k } , \Sigma _ { k } , \pi _ { k }$ and alternate until convergence between 

E-step: Evaluate the responsibilities $r _ { n k }$ (posterior probability of data point n belonging to mixture component k). 

M-step: Use the updated responsibilities to reestimate the parameters $\mu _ { k } , \Sigma _ { k } , \pi _ { k }$ 

Every step in the EM algorithm increases the log-likelihood function (Neal and Hinton, 1999). For convergence, we can check the log-likelihood or the parameters directly. A concrete instantiation of the EM algorithm for estimating the parameters of a GMM is as follows: 

1. Initialize $\pmb { \mu } _ { k } , \pmb { \Sigma } _ { k } , \pi _ { k }$ 

2. E-step: Evaluate responsibilities $r _ { n k }$ for every data point ${ \bf { x } } _ { n }$ using current parameters $\pi _ { k } , \pmb { \mu } _ { k } , \pmb { \Sigma } _ { k }$ 

$$
r _ {n k} = \frac {\pi_ {k} \mathcal {N} (\pmb {x} _ {n} | \pmb {\mu} _ {k} , \pmb {\Sigma} _ {k})}{\sum_ {j} \pi_ {j} \mathcal {N} (\pmb {x} _ {n} | \pmb {\mu} _ {j} , \pmb {\Sigma} _ {j})}.\tag{11.53}
$$

3. M-step: Reestimate parameters $\pi _ { k } , \mu _ { k } , \Sigma _ { k }$ using the current responsibilities $r _ { n k }$ (from E-step): 

$$
\pmb {\mu} _ {k} = \frac {1}{N _ {k}} \sum_ {n = 1} ^ {N} r _ {n k} \pmb {x} _ {n},\tag{11.54}
$$

$$
\mu_ {k}
$$

$$
\boldsymbol {\Sigma} _ {k} = \frac {1}{N _ {k}} \sum_ {n = 1} ^ {N} r _ {n k} (\boldsymbol {x} _ {n} - \boldsymbol {\mu} _ {k}) (\boldsymbol {x} _ {n} - \boldsymbol {\mu} _ {k}) ^ {\top},\tag{11.55}
$$

$$
\pi_ {k} = \frac {N _ {k}}{N}.\tag{11.56}
$$

## Example 11.6 (GMM Fit)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/7574af57a35d8d3f4e310e7d50d5ec9d1b55b73a97cb4f6f38c6d18a7d936903.jpg)



(a) Final GMM fit. After five iterations, the EM algorithm converges and returns this GMM.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/bcb9d2ab9544738cad94d6a592c9ec871860cc50abaa4ae877d94321bf5e2e36.jpg)



Figure 11.8 EM algorithm applied to the GMM from Figure 11.2. (a) Final GMM fit; (b) negative log-likelihood as a function of the EM iteration.



(b) Negative log-likelihood as a function of the EM iterations.


©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 

Figure 11.9 Illustration of the EM algorithm for fitting a Gaussian mixture model with three components to a two-dimensional dataset. (a) Dataset; (b) negative log-likelihood (lower is better) as a function of the EM iterations. The red dots indicate the iterations for which the mixture components of the corresponding GMM fits are shown in (c) through (f). The yellow discs indicate the means of the Gaussian mixture components. Figure 11.10(a) shows the final GMM fit. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/c09e76ba622524e8945c3de15fb7dcddbddbf76ee34a13660e8cd3bf8da026a3.jpg)



(a) Dataset.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/33e75b782cc9e33dfbd3c21623f53370e38a43e1a5b96386af9afaed822065db.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/dfa9ed0a351fb0b6174b3bbfd6570483bc303eb468b28536d127f5ee061c8e07.jpg)



(b) Negative log-likelihood.



(c) EM initialization.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/aa11d676b23f0f55be91773f44da73894aaddfe1ba3237657584f4d4c94fe243.jpg)



(d) EM after one iteration.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/f0ea384ea8160cdbe5b3108ba63fc1963e4fc2837b174ec5d363b2a3d684a9dd.jpg)



(e) EM after 10 iterations.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/52d545c844bcad2c6a75c0592571698aefc865d10882770be4e0833703ab1446.jpg)



(f) EM after 62 iterations.


When we run EM on our example from Figure 11.3, we obtain the final result shown in Figure 11.8(a) after five iterations, and Figure 11.8(b) shows how the negative log-likelihood evolves as a function of the EM iterations. The final GMM is given as 

$$
\begin{array}{c} p (x) = 0. 2 9 \mathcal {N} \big (x \mid - 2. 7 5,   0. 0 6 \big) + 0. 2 8 \mathcal {N} \big (x \mid - 0. 5 0,   0. 2 5 \big) \\ + 0. 4 3 \mathcal {N} \big (x \mid 3. 6 4,   1. 6 3 \big)  . \end{array}\tag{11.57}
$$

We applied the EM algorithm to the two-dimensional dataset shown in Figure 11.1 with $K = 3$ mixture components. Figure 11.9 illustrates some steps of the EM algorithm and shows the negative log-likelihood as a function of the EM iteration (Figure 11.9(b)). Figure 11.10(a) shows 

## 11.4 Latent-Variable Perspective

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/59753f19867c11295850911dbab06379258013cca5ca6114bb8b99d553f0a865.jpg)



(a) GMM fit after 62 iterations.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/cf2d5a25a8824dfee9c472d8e29665d58ce6ddda8db6eaf0bbe037d04f5c78d1.jpg)



(b) Dataset colored according to the responsibilities of the mixture components.



Figure 11.10 GMM fit and responsibilities when EM converges. (a) GMM fit when EM converges; (b) each data point is colored according to the responsibilities of the mixture components.


the corresponding final GMM fit. Figure 11.10(b) visualizes the final responsibilities of the mixture components for the data points. The dataset is colored according to the responsibilities of the mixture components when EM converges. While a single mixture component is clearly responsible for the data on the left, the overlap of the two data clusters on the right could have been generated by two mixture components. It becomes clear that there are data points that cannot be uniquely assigned to a single component (either blue or yellow), such that the responsibilities of these two clusters for those points are around 0.5. 

## 11.4 Latent-Variable Perspective

We can look at the GMM from the perspective of a discrete latent-variable model, i.e., where the latent variable z can attain only a finite set of values. This is in contrast to PCA, where the latent variables were continuousvalued numbers in <sup>RM</sup>. 

The advantages of the probabilistic perspective are that (i) it will justify some ad hoc decisions we made in the previous sections, (ii) it allows for a concrete interpretation of the responsibilities as posterior probabil ities, and (iii) the iterative algorithm for updating the model parameters can be derived in a principled manner as the EM algorithm for maximum likelihood parameter estimation in latent-variable models. 

## 11.4.1 Generative Process and Probabilistic Model

To derive the probabilistic model for GMMs, it is useful to think about the generative process, i.e., the process that allows us to generate data, using a probabilistic model. 

We assume a mixture model with K components and that a data point x can be generated by exactly one mixture component. We introduce a binary indicator variable $z _ { k } \in \{ 0 , 1 \}$ with two states (see Section 6.2) that indicates whether the kth mixture component generated that data point 

so that 

$$
p (\boldsymbol {x} \mid z _ {k} = 1) = \mathcal {N} \bigl (\boldsymbol {x} \mid \boldsymbol {\mu} _ {k}, \boldsymbol {\Sigma} _ {k} \bigr).\tag{11.58}
$$

We define $z : = [ z _ { 1 } , \ldots , z _ { K } ] ^ { \top } \in \mathbb { R } ^ { K }$ as a probability vector consisting of $K - 1$ many 0s and exactly one 1. For example, for $K = 3 ,$ , a valid z would be $z = [ z _ { 1 } , z _ { 2 } , z _ { 3 } ] ^ { \top } = [ 0 , 1 , 0 ] ^ { \top }$ , which would select the second mixture component since $z _ { 2 } = 1$ 

Remark. Sometimes this kind of probability distribution is called “multinoulli”, a generalization of the Bernoulli distribution to more than two values (Murphy, 2012). ◇ 

The properties of z imply that $\textstyle \sum _ { k = 1 } ^ { K } z _ { k } = 1$ . Therefore, $_ z$ is a one-hot encoding (also: 1-of-K representation). 

Thus far, we assumed that the indicator variables $z _ { k }$ are known. However, in practice, this is not the case, and we place a prior distribution 

$$
p (\boldsymbol {z}) = \boldsymbol {\pi} = [ \pi_ {1}, \dots , \pi_ {K} ] ^ {\top}, \quad \sum_ {k = 1} ^ {K} \pi_ {k} = 1,\tag{11.59}
$$

on the latent variable z. Then the kth entry 

$$
\pi_ {k} = p (z _ {k} = 1)\tag{11.60}
$$

Figure 11.11 Graphical model for a GMM with a single data point. 

of this probability vector describes the probability that the kth mixture component generated data point x. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/315c1d4c5a2809bf5e12948365f196f0fa494dd922da9e29d445b0dcce7e20a7.jpg)


Remark (Sampling from a GMM). The construction of this latent-variable model (see the corresponding graphical model in Figure 11.11) lends itself to a very simple sampling procedure (generative process) to generate data: 

1. Sample $z ^ { ( i ) } \sim p ( z )$ 

2. Sample $\pmb { x } ^ { ( i ) } \sim p ( \pmb { x } | z ^ { ( i ) } = 1 )$ 

In the first step, we select a mixture component i (via the one-hot encoding z) at random according to $p ( z ) = \pi ;$ in the second step we draw a sample from the corresponding mixture component. When we discard the samples of the latent variable so that we are left with the $\pmb { x } ^ { ( i ) }$ , we have valid samples from the GMM. This kind of sampling, where samples of random variables depend on samples from the variable’s parents in the graphical model, is called ancestral sampling. $\diamondsuit$ 

Generally, a probabilistic model is defined by the joint distribution of the data and the latent variables (see Section 8.4). With the prior $p ( z )$ defined in (11.59) and (11.60) and the conditional $p ( { \pmb x } \mid z )$ from (11.58), we obtain all K components of this joint distribution via 

$$
p (\boldsymbol {x}, z _ {k} = 1) = p (\boldsymbol {x} \mid z _ {k} = 1) p (z _ {k} = 1) = \pi_ {k} \mathcal {N} \left(\boldsymbol {x} \mid \boldsymbol {\mu} _ {k}, \boldsymbol {\Sigma} _ {k}\right)\tag{11.61}
$$

Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 

for $k = 1 , \ldots , K$ , so that 

$$
p (\boldsymbol {x}, \boldsymbol {z}) = \left[ \begin{array}{c} p (\boldsymbol {x}, z _ {1} = 1) \\ \vdots \\ p (\boldsymbol {x}, z _ {K} = 1) \end{array} \right] = \left[ \begin{array}{c} \pi_ {1} \mathcal {N} \big (\boldsymbol {x} \mid \boldsymbol {\mu} _ {1},   \boldsymbol {\Sigma} _ {1} \big) \\ \vdots \\ \pi_ {K} \mathcal {N} \big (\boldsymbol {x} \mid \boldsymbol {\mu} _ {K},   \boldsymbol {\Sigma} _ {K} \big) \end{array} \right],\tag{11.62}
$$

which fully specifies the probabilistic model. 

## 11.4.2 Likelihood

To obtain the likelihood $p ( { \pmb x } | { \pmb \theta } )$ in a latent-variable model, we need to marginalize out the latent variables (see Section 8.4.3). In our case, this can be done by summing out all latent variables from the joint $p ( { \pmb x } , z )$ in (11.62) so that 

$$
p (\boldsymbol {x} \mid \boldsymbol {\theta}) = \sum_ {\boldsymbol {z}} p (\boldsymbol {x} \mid \boldsymbol {\theta}, \boldsymbol {z}) p (\boldsymbol {z} \mid \boldsymbol {\theta}), \quad \boldsymbol {\theta} := \left\{\boldsymbol {\mu} _ {k}, \boldsymbol {\Sigma} _ {k}, \pi_ {k}: k = 1, \dots , K \right\}.\tag{11.63}
$$

We now explicitly condition on the parameters θ of the probabilistic model, which we previously omitted. In (11.63), we sum over all K possible onehot encodings of $z ,$ which is denoted by $\textstyle \sum _ { z }$ . Since there is only a single nonzero single entry in each z there are only $K$ possible configurations/ settings of z. For example, if $K = 3 .$ , then z can have the configurations 

$$
\left[ \begin{array}{c} 1 \\ 0 \\ 0 \end{array} \right], \left[ \begin{array}{c} 0 \\ 1 \\ 0 \end{array} \right], \left[ \begin{array}{c} 0 \\ 0 \\ 1 \end{array} \right].\tag{11.64}
$$

Summing over all possible configurations of z in (11.63) is equivalent to looking at the nonzero entry of the z-vector and writing 

$$
\begin{array}{l} p (\boldsymbol {x} \mid \boldsymbol {\theta}) = \sum_ {\boldsymbol {z}} p (\boldsymbol {x} \mid \boldsymbol {\theta}, \boldsymbol {z}) p (\boldsymbol {z} \mid \boldsymbol {\theta}) \\ = \sum_ {k = 1} ^ {K} p (\boldsymbol {x} \mid \boldsymbol {\theta}, z _ {k} = 1) p (z _ {k} = 1 \mid \boldsymbol {\theta}) \end{array}\tag{11.65a}
$$

(11.65b) 

so that the desired marginal distribution is given as 

$$
\begin{array}{l} p (\boldsymbol {x} \mid \boldsymbol {\theta}) \stackrel {(1 1. 6 5 b)} {=} \sum_ {k = 1} ^ {K} p (\boldsymbol {x} \mid \boldsymbol {\theta}, z _ {k} = 1) p (z _ {k} = 1 | \boldsymbol {\theta}) \\ = \sum_ {k = 1} ^ {K} \pi_ {k} \mathcal {N} \big (\boldsymbol {x} \mid \boldsymbol {\mu} _ {k}, \boldsymbol {\Sigma} _ {k} \big), \end{array}\tag{11.66a}
$$

(11.66b) 

which we identify as the GMM model from (11.3). Given a dataset $\mathcal { X } ,$ , we immediately obtain the likelihood 

$$
p (\mathcal {X} \mid \boldsymbol {\theta}) = \prod_ {n = 1} ^ {N} p (\boldsymbol {x} _ {n} \mid \boldsymbol {\theta}) \stackrel {(1 1. 6 6 b)} {=} \prod_ {n = 1} ^ {N} \sum_ {k = 1} ^ {K} \pi_ {k} \mathcal {N} \left(\boldsymbol {x} _ {n} \mid \boldsymbol {\mu} _ {k}, \boldsymbol {\Sigma} _ {k}\right),\tag{11.67}
$$

©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 


Figure 11.12 Graphical model for a GMM with N data points.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/2e14df2c7a58bc9340f25c297c1ad32a1fc635b8e08a4d6465a0b2f594323611.jpg)


which is exactly the GMM likelihood from (11.9). Therefore, the latentvariable model with latent indicators $z _ { k }$ is an equivalent way of thinking about a Gaussian mixture model. 

## 11.4.3 Posterior Distribution

Let us have a brief look at the posterior distribution on the latent variable z. According to Bayes’ theorem, the posterior of the kth component having generated data point x 

$$
p (z _ {k} = 1 \mid \boldsymbol {x}) = \frac {p (z _ {k} = 1) p (\boldsymbol {x} \mid z _ {k} = 1)}{p (\boldsymbol {x})},\tag{11.68}
$$

where the marginal $p ( { \pmb x } )$ is given in (11.66b). This yields the posterior distribution for the kth indicator variable $z _ { k }$ 

$$
p (z _ {k} = 1 \mid \boldsymbol {x}) = \frac {p (z _ {k} = 1) p (\boldsymbol {x} \mid z _ {k} = 1)}{\sum_ {j = 1} ^ {K} p (z _ {j} = 1) p (\boldsymbol {x} \mid z _ {j} = 1)} = \frac {\pi_ {k} \mathcal {N} \big (\boldsymbol {x} \mid \boldsymbol {\mu} _ {k} , \boldsymbol {\Sigma} _ {k} \big)}{\sum_ {j = 1} ^ {K} \pi_ {j} \mathcal {N} \big (\boldsymbol {x} \mid \boldsymbol {\mu} _ {j} , \boldsymbol {\Sigma} _ {j} \big)},\tag{11.69}
$$

which we identify as the responsibility of the kth mixture component for data point x. Note that we omitted the explicit conditioning on the GMM parameters $\pi _ { k } , \mu _ { k } , \Sigma _ { k }$ where $k = 1 , \ldots , K$ 

## 11.4.4 Extension to a Full Dataset

Thus far, we have only discussed the case where the dataset consists only of a single data point x. However, the concepts of the prior and posterior can be directly extended to the case of N data points ${ \mathcal { X } } : = \{ { \pmb x } _ { 1 } , \ldots , { \pmb x } _ { N } \}$ 

In the probabilistic interpretation of the GMM, every data point ${ \bf { x } } _ { n }$ possesses its own latent variable 

$$
\boldsymbol {z} _ {n} = \left[ z _ {n 1}, \dots , z _ {n K} \right] ^ {\top} \in \mathbb {R} ^ {K}.\tag{11.70}
$$

Previously (when we only considered a single data point x), we omitted the index $n ,$ but now this becomes important. 

## 11.4 Latent-Variable Perspective

We share the same prior distribution π across all latent variables $z _ { n } .$ The corresponding graphical model is shown in Figure 11.12, where we use the plate notation. 

The conditional distribution $p ( \pmb { x } _ { 1 } , \dots , \pmb { x } _ { N } | z _ { 1 } , \dots , z _ { N } )$ factorizes over the data points and is given as 

$$
p \left(\boldsymbol {x} _ {1}, \dots , \boldsymbol {x} _ {N} \mid \boldsymbol {z} _ {1}, \dots , \boldsymbol {z} _ {N}\right) = \prod_ {n = 1} ^ {N} p \left(\boldsymbol {x} _ {n} \mid \boldsymbol {z} _ {n}\right).\tag{11.71}
$$

To obtain the posterior distribution $p ( z _ { n k } = 1 | \pmb { x } _ { n } )$ , we follow the same reasoning as in Section 11.4.3 and apply Bayes’ theorem to obtain 

$$
\begin{array}{c} p (z _ {n k} = 1 \mid \boldsymbol {x} _ {n}) = \frac {p (\boldsymbol {x} _ {n} \mid z _ {n k} = 1) p (z _ {n k} = 1)}{\sum_ {j = 1} ^ {K} p (\boldsymbol {x} _ {n} \mid z _ {n j} = 1) p (z _ {n j} = 1)} \\ = \frac {\pi_ {k} \mathcal {N} \big (\boldsymbol {x} _ {n} \mid \boldsymbol {\mu} _ {k} , \boldsymbol {\Sigma} _ {k} \big)}{\sum_ {j = 1} ^ {K} \pi_ {j} \mathcal {N} \big (\boldsymbol {x} _ {n} \mid \boldsymbol {\mu} _ {j} , \boldsymbol {\Sigma} _ {j} \big)} = r _ {n k}. \end{array}\tag{11.72a}
$$

(11.72b) 

This means that $p ( z _ { k } = 1 | \pmb { x } _ { n } )$ is the (posterior) probability that the kth mixture component generated data point ${ \bf { x } } _ { n }$ and corresponds to the responsibility $r _ { n k }$ we introduced in (11.17). Now the responsibilities also have not only an intuitive but also a mathematically justified interpretation as posterior probabilities. 

## 11.4.5 EM Algorithm Revisited

The EM algorithm that we introduced as an iterative scheme for maximum likelihood estimation can be derived in a principled way from the latentvariable perspective. Given a current setting ${ \pmb \theta } ^ { ( t ) }$ of model parameters, the E-step calculates the expected log-likelihood 

$$
Q (\pmb {\theta} | \pmb {\theta} ^ {(t)}) = \mathbb {E} _ {\pmb {z} | \pmb {x}, \pmb {\theta} ^ {(t)}} [ \log p (\pmb {x}, \pmb {z} | \pmb {\theta}) ]\tag{11.73a}
$$

$$
= \int \log p (\pmb {x}, \pmb {z} | \pmb {\theta}) p (\pmb {z} | \pmb {x}, \pmb {\theta} ^ {(t)}) \mathrm{d} \pmb {z},\tag{11.73b}
$$

where the expectation of log $p ( \pmb { x } , z | \pmb { \theta } )$ is taken with respect to the posterior $p ( \boldsymbol { z } | \boldsymbol { x } , \boldsymbol { \theta } ^ { ( t ) } )$ of the latent variables. The M-step selects an updated set of model parameters $\pmb { \theta } ^ { ( t + 1 ) }$ by maximizing (11.73b). 

Although an EM iteration does increase the log-likelihood, there are no guarantees that EM converges to the maximum likelihood solution. It is possible that the EM algorithm converges to a local maximum of the log-likelihood. Different initializations of the parameters θ could be used in multiple EM runs to reduce the risk of ending up in a bad local optimum. We do not go into further details here, but refer to the excellent expositions by Rogers and Girolami (2016) and Bishop (2006). 

## 11.5 Further Reading

The GMM can be considered a generative model in the sense that it is straightforward to generate new data using ancestral sampling (Bishop, 2006). For given GMM parameters $\pi _ { k } , \mu _ { k } , \Sigma _ { k } , k = 1 , \ldots , K$ , we sample an index k from the probability vector $[ \pi _ { 1 } , \ldots , \pi _ { K } ] ^ { \top }$ and then sample a data point $\pmb { x } \sim \mathcal { N } ( \pmb { \mu } _ { k } , \pmb { \Sigma } _ { k } )$ . If we repeat this N times, we obtain a dataset that has been generated by a GMM. Figure 11.1 was generated using this procedure. 

Throughout this chapter, we assumed that the number of components K is known. In practice, this is often not the case. However, we could use nested cross-validation, as discussed in Section 8.6.1, to find good models. 

Gaussian mixture models are closely related to the K-means clustering algorithm. K-means also uses the EM algorithm to assign data points to clusters. If we treat the means in the GMM as cluster centers and ignore the covariances (or set them to I), we arrive at K-means. As also nicely described by MacKay (2003), K-means makes a “hard” assignment of data points to cluster centers $\pmb { \mu } _ { k }$ , whereas a GMM makes a “soft” assignment via the responsibilities. 

We only touched upon the latent-variable perspective of GMMs and the EM algorithm. Note that EM can be used for parameter learning in general latent-variable models, e.g., nonlinear state-space models (Ghahramani and Roweis, 1999; Roweis and Ghahramani, 1999) and for reinforcement learning as discussed by Barber (2012). Therefore, the latent-variable perspective of a GMM is useful to derive the corresponding EM algorithm in a principled way (Bishop, 2006; Barber, 2012; Murphy, 2012). 

We only discussed maximum likelihood estimation (via the EM algorithm) for finding GMM parameters. The standard criticisms of maximum likelihood also apply here: 

As in linear regression, maximum likelihood can suffer from severe overfitting. In the GMM case, this happens when the mean of a mixture component is identical to a data point and the covariance tends to 0. Then, the likelihood approaches infinity. Bishop (2006) and Barber (2012) discuss this issue in detail. 

We only obtain a point estimate of the parameters $\pi _ { k } , \mu _ { k } , \Sigma _ { k }$ for k = $1 , \ldots , K$ , which does not give any indication of uncertainty in the parameter values. A Bayesian approach would place a prior on the parameters, which can be used to obtain a posterior distribution on the parameters. This posterior allows us to compute the model evidence (marginal likelihood), which can be used for model comparison, which gives us a principled way to determine the number of mixture components. Unfortunately, closed-form inference is not possible in this setting because there is no conjugate prior for this model. However, approximations, such as variational inference, can be used to obtain an approximate posterior (Bishop, 2006). 

Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 

## 11.5 Further Reading

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/29996969dff8097992c8fd2f59e2bcc54eacb6b9fe23cc8216217ad5a074e03e.jpg)



Figure 11.13 Histogram (orange bars) and kernel density estimation (blue line). The kernel density estimator produces a smooth estimate of the underlying density, whereas the histogram is an unsmoothed count measure of how many data points (black) fall into a single bin.


In this chapter, we discussed mixture models for density estimation. There is a plethora of density estimation techniques available. In practice, we often use histograms and kernel density estimation. 

Histograms provide a nonparametric way to represent continuous densities and have been proposed by Pearson (1895). A histogram is constructed by “binning” the data space and count, how many data points fall into each bin. Then a bar is drawn at the center of each bin, and the height of the bar is proportional to the number of data points within that bin. The bin size is a critical hyperparameter, and a bad choice can lead to overfitting and underfitting. Cross-validation, as discussed in Section 8.2.4, can be used to determine a good bin size. 

Kernel density estimation, independently proposed by Rosenblatt (1956) and Parzen (1962), is a nonparametric way for density estimation. Given N i.i.d. samples, the kernel density estimator represents the underlying distribution as 

$$
p (\boldsymbol {x}) = \frac {1}{N h} \sum_ {n = 1} ^ {N} k \left(\frac {\boldsymbol {x} - \boldsymbol {x} _ {n}}{h}\right),\tag{11.74}
$$

where k is a kernel function, i.e., a nonnegative function that integrates to 1 and $h > 0$ is a smoothing/bandwidth parameter, which plays a similar role as the bin size in histograms. Note that we place a kernel on every single data point ${ \bf { x } } _ { n }$ in the dataset. Commonly used kernel functions are the uniform distribution and the Gaussian distribution. Kernel density estimates are closely related to histograms, but by choosing a suitable kernel, we can guarantee smoothness of the density estimate. Figure 11.13 illustrates the difference between a histogram and a kernel density estimator (with a Gaussian-shaped kernel) for a given dataset of 250 data points. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/894152bbc64515d32b9474fda0a16c6ff9b9ecbb8337d54607b749239237868e.jpg)


Input example x<sub>n</sub> may also be referred to as inputs, data points, features, or instances. class 

For probabilistic models, it is mathematically convenient to use $\{ 0 , 1 \}$ as a binary representation; see the remark after Example 6.12. 

# Classification with Support Vector Machines

In many situations, we want our machine learning algorithm to predict one of a number of (discrete) outcomes. For example, an email client sorts mail into personal mail and junk mail, which has two outcomes. Another example is a telescope that identifies whether an object in the night sky is a galaxy, star, or planet. There are usually a small number of outcomes, and more importantly there is usually no additional structure on these outcomes. In this chapter, we consider predictors that output binary values, i.e., there are only two possible outcomes. This machine learning task is called binary classification. This is in contrast to Chapter 9, where we considered a prediction problem with continuous-valued outputs. 

For binary classification, the set of possible values that the label/output can attain is binary, and for this chapter we denote them by $\{ + 1 , - 1 \}$ . In other words, we consider predictors of the form 

$$
f: \mathbb {R} ^ {D} \to \{+ 1, - 1 \}.\tag{12.1}
$$

Recall from Chapter 8 that we represent each example (data point) ${ \bf { x } } _ { n }$ as a feature vector of D real numbers. The labels are often referred to as the positive and negative classes, respectively. One should be careful not to infer intuitive attributes of positiveness of the +1 class. For example, in a cancer detection task, a patient with cancer is often labeled +1. In principle, any two distinct values can be used, e.g., True, False , 0, 1 or red, blue . The problem of binary classification is well studied, and we defer a survey of other approaches to Section 12.6. 

We present an approach known as the support vector machine (SVM), which solves the binary classification task. As in regression, we have a supervised learning task, where we have a set of examples ${ \pmb x } _ { n } \in \mathbb { R } ^ { D }$ along with their corresponding (binary) labels $y _ { n } \in \{ + 1 , - 1 \}$ . Given a training data set consisting of example–label pairs $\{ ( \pmb { x } _ { 1 } , y _ { 1 } ) , \dotsc , ( \pmb { x } _ { N } , y _ { N } ) \}$ , we would like to estimate parameters of the model that will give the smallest classification error. Similar to Chapter 9, we consider a linear model, and hide away the nonlinearity in a transformation $\phi$ of the examples (9.13). We will revisit ϕ in Section 12.4. 

The SVM provides state-of-the-art results in many applications, with sound theoretical guarantees (Steinwart and Christmann, 2008). There are two main reasons why we chose to illustrate binary classification using 

## 370

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/a43f886f55e9ddef85d84037bc0b023eb60aea05377b4143886521f35c38a243.jpg)



Figure 12.1 Example 2D data, illustrating the intuition of data where we can find a linear classifier that separates orange crosses from blue discs.


SVMs. First, the SVM allows for a geometric way to think about supervised machine learning. While in Chapter 9 we considered the machine learning problem in terms of probabilistic models and attacked it using maximum likelihood estimation and Bayesian inference, here we will consider an alternative approach where we reason geometrically about the machine learning task. It relies heavily on concepts, such as inner products and projections, which we discussed in Chapter 3. The second reason why we find SVMs instructive is that in contrast to Chapter 9, the optimization problem for SVM does not admit an analytic solution so that we need to resort to a variety of optimization tools introduced in Chapter 7. 

The SVM view of machine learning is subtly different from the maximum likelihood view of Chapter 9. The maximum likelihood view proposes a model based on a probabilistic view of the data distribution, from which an optimization problem is derived. In contrast, the SVM view starts by designing a particular function that is to be optimized during training, based on geometric intuitions. We have seen something similar already in Chapter 10, where we derived PCA from geometric principles. In the SVM case, we start by designing a loss function that is to be minimized on training data, following the principles of empirical risk minimization (Section 8.2). 

Let us derive the optimization problem corresponding to training an SVM on example–label pairs. Intuitively, we imagine binary classification data, which can be separated by a hyperplane as illustrated in Figure 12.1. Here, every example ${ \bf { x } } _ { n }$ (a vector of dimension 2) is a two-dimensional location $( x _ { n } ^ { ( 1 ) }$ and $x _ { n } ^ { ( 2 ) } )$ , and the corresponding binary label $y _ { n }$ is one of two different symbols (orange cross or blue disc). “Hyperplane” is a word that is commonly used in machine learning, and we encountered hyperplanes already in Section 2.8. A hyperplane is an affine subspace of dimension $D - 1$ (if the corresponding vector space is of dimension D). The examples consist of two classes (there are two possible labels) that have features (the components of the vector representing the example) arranged in such a way as to allow us to separate/classify them by drawing a straight line. 

©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 

In the following, we formalize the idea of finding a linear separator of the two classes. We introduce the idea of the margin and then extend linear separators to allow for examples to fall on the “wrong” side, incurring a classification error. We present two equivalent ways of formalizing the SVM: the geometric view (Section 12.2.4) and the loss function view (Section 12.2.5). We derive the dual version of the SVM using Lagrange multipliers (Section 7.2). The dual SVM allows us to observe a third way of formalizing the SVM: in terms of the convex hulls of the examples of each class (Section 12.3.2). We conclude by briefly describing kernels and how to numerically solve the nonlinear kernel-SVM optimization problem. 

## 12.1 Separating Hyperplanes

Given two examples represented as vectors $\mathbf { \boldsymbol { x } } _ { i }$ and $\scriptstyle { \mathbf { { \mathit { x } } } } _ { j } .$ , one way to compute the similarity between them is using an inner product $\langle \pmb { x } _ { i } , \pmb { x } _ { j } \rangle$ . Recall from Section 3.2 that inner products are closely related to the angle between two vectors. The value of the inner product between two vectors depends on the length (norm) of each vector. Furthermore, inner products allow us to rigorously define geometric concepts such as orthogonality and projections. 

The main idea behind many classification algorithms is to represent data in $\mathbb { R } ^ { D }$ and then partition this space, ideally in a way that examples with the same label (and no other examples) are in the same partition. In the case of binary classification, the space would be divided into two parts corresponding to the positive and negative classes, respectively. We consider a particularly convenient partition, which is to (linearly) split the space into two halves using a hyperplane. Let example $\pmb { x } \in \mathbb { R } ^ { D }$ be an element of the data space. Consider a function 

$$
f: \mathbb {R} ^ {D} \to \mathbb {R}\tag{12.2a}
$$

$$
\boldsymbol {x} \mapsto f (\boldsymbol {x}) := \langle \boldsymbol {w}, \boldsymbol {x} \rangle + b,\tag{12.2b}
$$

parametrized by ${ \pmb w } \in \mathbb { R } ^ { D }$ and $b \in \mathbb { R }$ . Recall from Section 2.8 that hyperplanes are affine subspaces. Therefore, we define the hyperplane that separates the two classes in our binary classification problem as 

$$
\left\{\boldsymbol {x} \in \mathbb {R} ^ {D}: f (\boldsymbol {x}) = 0 \right\}.\tag{12.3}
$$

An illustration of the hyperplane is shown in Figure 12.2, where the vector w is a vector normal to the hyperplane and b the intercept. We can derive that w is a normal vector to the hyperplane in (12.3) by choosing any two examples ${ \pmb x } _ { a }$ and $\mathbf { \delta } _ { \mathbf { \mathcal { X } } _ { b } }$ on the hyperplane and showing that the vector between them is orthogonal to w. In the form of an equation, 

$$
f (\boldsymbol {x} _ {a}) - f (\boldsymbol {x} _ {b}) = \langle \boldsymbol {w}, \boldsymbol {x} _ {a} \rangle + b - (\langle \boldsymbol {w}, \boldsymbol {x} _ {b} \rangle + b)\tag{12.4a}
$$

$$
= \left\langle \boldsymbol {w}, \boldsymbol {x} _ {a} - \boldsymbol {x} _ {b} \right\rangle ,\tag{12.4b}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/d83f634d11e3b402d3692bba95594e9da5b501731c5db602d51907cd5da90e65.jpg)



12.1 Separating Hyperplanes



373



Figure 12.2 Equation of a separating hyperplane (12.3). (a) The standard way of representing the equation in 3D. (b) For ease of drawing, we look at the hyperplane edge on.


where the second line is obtained by the linearity of the inner product (Section 3.2). Since we have chosen ${ \pmb x } _ { a }$ and $\scriptstyle { \mathbf { \mathcal { x } } } _ { b }$ to be on the hyperplane, this implies that $f ( { \pmb x } _ { a } ) = 0$ and $f ( { \pmb x } _ { b } ) = 0$ and hence $\langle { \pmb w } , { \pmb x } _ { a } - { \pmb x } _ { b } \rangle = 0$ Recall that two vectors are orthogonal when their inner product is zero. Therefore, we obtain that w is orthogonal to any vector on the hyperplane. 

Remark. Recall from Chapter 2 that we can think of vectors in different ways. In this chapter, we think of the parameter vector w as an arrow indicating a direction, i.e., we consider w to be a geometric vector. In contrast, we think of the example vector x as a data point (as indicated by its coordinates), i.e., we consider x to be the coordinates of a vector with respect to the standard basis. 

When presented with a test example, we classify the example as positive or negative depending on the side of the hyperplane on which it occurs. Note that (12.3) not only defines a hyperplane; it additionally defines a direction. In other words, it defines the positive and negative side of the hyperplane. Therefore, to classify a test example $\scriptstyle { \pmb { x } } _ { \mathrm { t e s t } } .$ , we calculate the value of the function $f ( \pmb { x } _ { \mathrm { t e s t } } )$ and classify the example as +1 if $f ( \pmb { x } _ { \mathrm { t e s t } } ) \geq 0$ and 1 otherwise. Thinking geometrically, the positive examples lie “above” the hyperplane and the negative examples “below” the hyperplane. 

When training the classifier, we want to ensure that the examples with positive labels are on the positive side of the hyperplane, i.e., 

$$
\langle \boldsymbol {w}, \boldsymbol {x} _ {n} \rangle + b \geqslant 0 \quad \text { when } \quad y _ {n} = + 1\tag{12.5}
$$

and the examples with negative labels are on the negative side, i.e., 

$$
\langle \boldsymbol {w}, \boldsymbol {x} _ {n} \rangle + b <   0 \quad \text { when } \quad y _ {n} = - 1.\tag{12.6}
$$

Refer to Figure 12.2 for a geometric intuition of positive and negative examples. These two conditions are often presented in a single equation 

$$
y _ {n} (\langle \boldsymbol {w}, \boldsymbol {x} _ {n} \rangle + b) \geqslant 0.\tag{12.7}
$$

Equation (12.7) is equivalent to (12.5) and (12.6) when we multiply both sides of (12.5) and (12.6) with $y _ { n } = 1$ and $y _ { n } = - 1$ , respectively. 

Figure 12.3 Possible separating hyperplanes. There are many linear classifiers (green lines) that separate orange crosses from blue discs. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/d48043653129e713b13d4320212a0efc732899892b276859c77873e31b041e1f.jpg)


## 12.2 Primal Support Vector Machine

Based on the concept of distances from points to a hyperplane, we now are in a position to discuss the support vector machine. For a dataset $\left\{ ( \pmb { x } _ { 1 } , y _ { 1 } ) , \dotsc , ( \pmb { x } _ { N } , y _ { N } ) \right\}$ that is linearly separable, we have infinitely many candidate hyperplanes (refer to Figure 12.3), and therefore classifiers, that solve our classification problem without any (training) errors. To find a unique solution, one idea is to choose the separating hyperplane that maximizes the margin between the positive and negative examples. In other words, we want the positive and negative examples to be separated by a large margin (Section 12.2.1). In the following, we compute the distance between an example and a hyperplane to derive the margin. Recall that the closest point on the hyperplane to a given point (example $\scriptstyle { \pmb x } _ { n } )$ is obtained by the orthogonal projection (Section 3.8). 

## 12.2.1 Concept of the Margin

The concept of the margin is intuitively simple: It is the distance of the separating hyperplane to the closest examples in the dataset, assuming that the dataset is linearly separable. However, when trying to formalize this distance, there is a technical wrinkle that may be confusing. The technical wrinkle is that we need to define a scale at which to measure the distance. A potential scale is to consider the scale of the data, i.e., the raw values of ${ \pmb x } _ { n }$ . There are problems with this, as we could change the units of measurement of ${ \bf { x } } _ { n }$ and change the values in ${ \bf { x } } _ { n } .$ , and, hence, change the distance to the hyperplane. As we will see shortly, we define the scale based on the equation of the hyperplane (12.3) itself. 

Consider a hyperplane $\langle { \pmb w } , { \pmb x } \rangle + b ,$ and an example ${ \bf { \mathcal { x } } } _ { a }$ as illustrated in Figure 12.4. Without loss of generality, we can consider the example $\scriptstyle { \mathbf { { \vec { x } } } } _ { a }$ to be on the positive side of the hyperplane, i.e., $\langle { \pmb w } , { \pmb x } _ { a } \rangle + b > 0$ . We would like to compute the distance $r > 0$ of $\scriptstyle { \mathbf { { \vec { x } } } } _ { a }$ from the hyperplane. We do so by considering the orthogonal projection (Section 3.8) of $\scriptstyle { \mathbf { { \vec { x } } } } _ { a }$ onto the hyperplane, which we denote by $ { \boldsymbol { { x } } } _ { a } ^ { \prime }$ . Since w is orthogonal to the hyperplane, we know that the distance $r$ is just a scaling of this vector w. If the length of $\pmb { w }$ is known, then we can use this scaling factor r factor to work out the absolute distance between ${ \pmb x } _ { a }$ and $ { \boldsymbol { { x } } } _ { a } ^ { \prime }$ . For convenience, we choose to use a vector of unit length (its norm is 1) and obtain this by dividing w by its norm, $\frac { \pmb { w } } { \| \pmb { w } \| }$ . Using vector addition (Section 2.4), we obtain 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/358d1c1f74fa2a4ea78803e8298fb9281a85191bb8793b9dbf0b14ce5d1ddc13.jpg)



Figure 12.4 Vector addition to express distance to hyperplane: $\begin{array} { r } { \pmb { x } _ { a } = \pmb { x } _ { a } ^ { \prime } + r \frac { \pmb { w } } { \| \pmb { w } \| } . } \end{array}$


$$
\pmb {x} _ {a} = \pmb {x} _ {a} ^ {\prime} + r \frac {\pmb {w}}{\| \pmb {w} \|}.\tag{12.8}
$$

Another way of thinking about r is that it is the coordinate of ${ \pmb x } _ { a }$ in the subspace spanned by $w / | | w |$ . We have now expressed the distance of ${ \pmb x } _ { a }$ from the hyperplane as $r ,$ and if we choose ${ \pmb x } _ { a }$ to be the point closest to the hyperplane, this distance r is the margin. 

Recall that we would like the positive examples to be further than $r$ from the hyperplane, and the negative examples to be further than distance $r$ (in the negative direction) from the hyperplane. Analogously to the combination of (12.5) and (12.6) into (12.7), we formulate this objective as 

$$
y _ {n} (\langle \boldsymbol {w}, \boldsymbol {x} _ {n} \rangle + b) \geqslant r.\tag{12.9}
$$

In other words, we combine the requirements that examples are at least r away from the hyperplane (in the positive and negative direction) into one single inequality. 

Since we are interested only in the direction, we add an assumption to our model that the parameter vector w is of unit length, $\mathrm { i . e . , ~ } \| w \| = 1$ where we use the Euclidean norm $\| \mathbf { \boldsymbol { w } } \| = \sqrt { \mathbf { \boldsymbol { w } } ^ { \top } \mathbf { \boldsymbol { w } } }$ (Section 3.1). This assumption also allows a more intuitive interpretation of the distance r (12.8) since it is the scaling factor of a vector of length 1. 

Remark. A reader familiar with other presentations of the margin would notice that our definition of $\| \pmb { w } \| = 1$ is different from the standard presentation if the SVM was the one provided by Scholkopf and Smola¨ (2002), for example. In Section 12.2.3, we will show the equivalence of both approaches. 

Collecting the three requirements into a single constrained optimization 

Figure 12.5 Derivation of the margin: $\begin{array} { r } { r = \frac { 1 } { \| \pmb { w } \| } } \end{array}$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/9e63783aa93f3d9d9c49c8a51898a1db30ea0f9f3f47cdcbdf4eb2341cdc405e.jpg)


problem, we obtain the objective 

$$
\begin{array}{l l} \underset {\boldsymbol {w}, b, r} {\max} & \underbrace {r} _ {\text {margin}} \\ \text {subject to} & \underbrace {y _ {n} (\langle \boldsymbol {w} , \boldsymbol {x} _ {n} \rangle + b) \geqslant r} _ {\text {data fitting}}, \underbrace {\| \boldsymbol {w} \| = 1} _ {\text {normalization}}, \quad r > 0, \end{array}\tag{12.10}
$$

which says that we want to maximize the margin r while ensuring that the data lies on the correct side of the hyperplane. 

Remark. The concept of the margin turns out to be highly pervasive in machine learning. It was used by Vladimir Vapnik and Alexey Chervonenkis to show that when the margin is large, the “complexity” of the function class is low, and hence learning is possible (Vapnik, 2000). It turns out that the concept is useful for various different approaches for theoretically analyzing generalization error (Steinwart and Christmann, 2008; Shalev-Shwartz and Ben-David, 2014). ◇ 

## 12.2.2 Traditional Derivation of the Margin

In the previous section, we derived (12.10) by making the observation that we are only interested in the direction of w and not its length, leading to the assumption that $\lVert \mathbf { \boldsymbol { w } } \rVert = 1$ . In this section, we derive the margin maximization problem by making a different assumption. Instead of choosing that the parameter vector is normalized, we choose a scale for the data. We choose this scale such that the value of the predictor $\langle { \pmb w } , { \pmb x } \rangle + b$ is 1 at the closest example. Let us also denote the example in the dataset that is closest to the hyperplane by ${ \bf { \mathcal { x } } } _ { a }$ 

Figure 12.5 is identical to Figure 12.4, except that now we rescaled the axes, such that the example ${ \bf { \mathcal { x } } } _ { a }$ lies exactly on the margin, $\mathrm { i } . \mathrm { e } . , \left. w , x _ { a } \right.$ + $b = 1$ . Since $ { \boldsymbol { { x } } } _ { a } ^ { \prime }$ is the orthogonal projection of ${ \pmb x } _ { a }$ onto the hyperplane, it must by definition lie on the hyperplane, i.e., 

$$
\left\langle \boldsymbol {w}, \boldsymbol {x} _ {a} ^ {\prime} \right\rangle + b = 0.\tag{12.11}
$$

Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 

By substituting (12.8) into (12.11), we obtain 

$$
\left\langle \boldsymbol {w}, \boldsymbol {x} _ {a} - r \frac {\boldsymbol {w}}{\| \boldsymbol {w} \|} \right\rangle + b = 0.\tag{12.12}
$$

Exploiting the bilinearity of the inner product (see Section 3.2), we get 

$$
\langle \boldsymbol {w}, \boldsymbol {x} _ {a} \rangle + b - r \frac {\langle \boldsymbol {w} , \boldsymbol {w} \rangle}{\| \boldsymbol {w} \|} = 0.\tag{12.13}
$$

Observe that the first term is 1 by our assumption of scale, i. $\mathrm { e } . , \left. \pmb { w } , \pmb { x } _ { a } \right. +$ $b = 1$ . From (3.16) in Section 3.1, we know that $\langle { \pmb w } , { \pmb w } \rangle = \| { \pmb w } \| ^ { 2 }$ . Hence, the second term reduces to $r \| \boldsymbol { w } \|$ . Using these simplifications, we obtain 

$$
r = \frac {1}{\| \boldsymbol {w} \|}.\tag{12.14}
$$

This means we derived the distance $r$ in terms of the normal vector w of the hyperplane. At first glance, this equation is counterintuitive as we seem to have derived the distance from the hyperplane in terms of the length of the vector ${ \pmb w } _ { i }$ , but we do not yet know this vector. One way to think about it is to consider the distance $r$ to be a temporary variable that we only use for this derivation. Therefore, for the rest of this section we will denote the distance to the hyperplane by $\frac { 1 } { \| \pmb { w } \| }$ . In Section 12.2.3, we will see that the choice that the margin equals 1 is equivalent to our previous assumption of $\| \pmb { w } \| = 1$ in Section 12.2.1. 

Similar to the argument to obtain (12.9), we want the positive and negative examples to be at least 1 away from the hyperplane, which yields the condition 

$$
y _ {n} (\langle \boldsymbol {w}, x _ {n} \rangle + b) \geqslant 1.
$$

(12.15) 

Combining the margin maximization with the fact that examples need to be on the correct side of the hyperplane (based on their labels) gives us 

$$
\max _ {\boldsymbol {w}, b} \frac {1}{\| \boldsymbol {w} \|}\tag{12.16}
$$

$$
\text { subject   to } y _ {n} (\langle \boldsymbol {w}, \boldsymbol {x} _ {n} \rangle + b) \geqslant 1 \quad \text { for   all } \quad n = 1, \dots , N.\tag{12.17}
$$

Instead of maximizing the reciprocal of the norm as in (12.16), we often minimize the squared norm. We also often include a constant $\frac { 1 } { 2 }$ that does not affect the optimal w, b but yields a tidier form when we compute the gradient. Then, our objective becomes 

$$
\min _ {\boldsymbol {w}, b} \frac {1}{2} \| \boldsymbol {w} \| ^ {2}\tag{12.18}
$$

$$
\text { subject   to } y _ {n} (\langle \boldsymbol {w}, \boldsymbol {x} _ {n} \rangle + b) \geqslant 1 \quad \text { for   all } \quad n = 1, \dots , N.\tag{12.19}
$$

Equation (12.18) is known as the hard margin SVM. The reason for the expression “hard” is because the formulation does not allow for any vi olations of the margin condition. We will see in Section 12.2.4 that this “hard” condition can be relaxed to accommodate violations if the data is not linearly separable. 

## 12.2.3 Why We Can Set the Margin to 1

In Section 12.2.1, we argued that we would like to maximize some value $^ { r , }$ which represents the distance of the closest example to the hyperplane. In Section 12.2.2, we scaled the data such that the closest example is of distance 1 to the hyperplane. In this section, we relate the two derivations, and show that they are equivalent. 

Theorem 12.1. Maximizing the margin $^ { r , }$ where we consider normalized weights as in (12.10), 

$$
\begin{array}{l} \underset {\boldsymbol {w}, b, r} {\max} \underbrace {r} _ {m a r g i n} \\ \text {subject to} \quad \underbrace {y _ {n} (\langle \boldsymbol {w} , \boldsymbol {x} _ {n} \rangle + b) \geqslant r} _ {d a t a f i t t i n g}, \quad \underbrace {\| \boldsymbol {w} \| = 1} _ {n o r m a l i z a t i o n}, \quad r > 0, \end{array}\tag{12.20}
$$

is equivalent to scaling the data, such that the margin is unity: 

$$
\begin{array}{l l} \underset {\boldsymbol {w}, b} {\min} & \underbrace {\frac {1}{2} \left\| \boldsymbol {w} \right\| ^ {2}} _ {\text {margin}} \\ \text {subject to} & \underbrace {y _ {n} (\langle \boldsymbol {w} , \boldsymbol {x} _ {n} \rangle + b) \geqslant 1} _ {\text {data fitting}}. \end{array}\tag{12.21}
$$

Proof Consider (12.20). Since the square is a strictly monotonic transformation for non-negative arguments, the maximum stays the same if we consider $r ^ { 2 }$ in the objective. Since $\lVert \mathbf { \boldsymbol { w } } \rVert = 1$ we can reparametrize the equation with a new weight vector $\mathbf { \Delta } w ^ { \prime }$ that is not normalized by explicitly using $\frac { \pmb { w } ^ { \prime } } { \| \pmb { w } ^ { \prime } \| }$ . We obtain 

$$
\begin{array}{l l} \max _ {\boldsymbol {w} ^ {\prime}, b, r} & r ^ {2} \\ \text { subject   to } & y _ {n} \left(\left\langle \frac {\boldsymbol {w} ^ {\prime}}{\| \boldsymbol {w} ^ {\prime} \|}, \boldsymbol {x} _ {n} \right\rangle + b\right) \geqslant r, \quad r > 0. \end{array}\tag{12.22}
$$

Equation (12.22) explicitly states that the distance $r$ is positive. Therefore, we can divide the first constraint by $^ { r , }$ which yields 

$$
\begin{array}{l} \max _ {\boldsymbol {w} ^ {\prime}, b, r} \quad r ^ {2} \\ \text { subject   to } \quad y _ {n} \left(\left\langle \underbrace {\frac {\boldsymbol {w} ^ {\prime}}{\| \boldsymbol {w} ^ {\prime} \| r}} _ {\boldsymbol {w} ^ {\prime \prime}}, \boldsymbol {x} _ {n} \right\rangle + \underbrace {\frac {b}{r}} _ {b ^ {\prime \prime}}\right) \geqslant 1, \quad r > 0 \end{array}\tag{12.23}
$$

Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 

## 12.2 Primal Support Vector Machine

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/4a65c3da1a2de14a8514e43fd1748b8153951c7319d5fe06de5e54de9d417875.jpg)



(a) Linearly separable data, with a large margin


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/927a4d5a97234690c1b757523bda6d2f41dc1a7404a032357ae637d3c5ae2165.jpg)



Figure 12.6 (a) Linearly separable and (b) non-linearly separable data.



(b) Non-linearly separable data


renaming the parameters to ${ \pmb w } ^ { \prime \prime }$ and $b ^ { \prime \prime }$ . Since $\begin{array} { r } { \pmb { w } ^ { \prime \prime } = \frac { \pmb { w } ^ { \prime } } { \Vert \pmb { w } ^ { \prime } \Vert r } } \end{array}$ , rearranging for r gives 

$$
\left\| \boldsymbol {w} ^ {\prime \prime} \right\| = \left\| \frac {\boldsymbol {w} ^ {\prime}}{\left\| \boldsymbol {w} ^ {\prime} \right\| r} \right\| = \frac {1}{r} \cdot \left\| \frac {\boldsymbol {w} ^ {\prime}}{\left\| \boldsymbol {w} ^ {\prime} \right\|} \right\| = \frac {1}{r}.\tag{12.24}
$$

By substituting this result into (12.23), we obtain 

$$
\begin{array}{l l} \max _ {\boldsymbol {w} ^ {\prime \prime}, b ^ {\prime \prime}} & \frac {1}{\| \boldsymbol {w} ^ {\prime \prime} \| ^ {2}} \\ \text { subject   to } & y _ {n} \left(\langle \boldsymbol {w} ^ {\prime \prime}, \boldsymbol {x} _ {n} \rangle + b ^ {\prime \prime}\right) \geqslant 1. \end{array}\tag{12.25}
$$

The final step is to observe that maximizing $\frac { 1 } { \| \pmb { w } ^ { \prime \prime } \| ^ { 2 } }$ yields the same solution as minimizing $\scriptstyle { \frac { 1 } { 2 } } \left\| w ^ { \prime \prime } \right\| ^ { 2 }$ , which concludes the proof of Theorem 12.1. 

## 12.2.4 Soft Margin SVM: Geometric View

In the case where data is not linearly separable, we may wish to allow some examples to fall within the margin region, or even to be on the wrong side of the hyperplane as illustrated in Figure 12.6. 

The model that allows for some classification errors is called the soft margin SVM. In this section, we derive the resulting optimization problem using geometric arguments. In Section 12.2.5, we will derive an equivalent optimization problem using the idea of a loss function. Using Lagrange multipliers (Section 7.2), we will derive the dual optimization problem of the SVM in Section 12.3. This dual optimization problem allows us to observe a third interpretation of the SVM: as a hyperplane that bisects the line between convex hulls corresponding to the positive and negative data examples (Section 12.3.2). 

The key geometric idea is to introduce a slack variable $\xi _ { n }$ corresponding to each example–label pair $( x _ { n } , y _ { n } )$ that allows a particular example to be within the margin or even on the wrong side of the hyperplane (refer to 

Figure 12.7 Soft margin SVM allows examples to be within the margin or on the wrong side of the hyperplane. The slack variable $\xi$ measures the distance of a positive example $\mathbf { x } _ { + }$ to the positive margin hyperplane ⟨w, x⟩ + b = 1 when $^ { x _ { + } }$ is on the wrong side. 

regularizer 

There are alternative parametrizations of this regularization, which is why (12.26a) is also often referred to as the C-SVM. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/8efc63265b0b6a3cd8581bd0ce834da67544a76e5a251439ba7af9479abae2e4.jpg)


Figure 12.7). We subtract the value of $\xi _ { n }$ from the margin, constraining $\xi _ { n }$ to be non-negative. To encourage correct classification of the samples, we add $\xi _ { n }$ to the objective 

$$
\min _ {\boldsymbol {w}, b, \boldsymbol {\xi}} \frac {1}{2} \| \boldsymbol {w} \| ^ {2} + C \sum_ {n = 1} ^ {N} \xi_ {n}\tag{12.26a}
$$

$$
\text { subject   to } \quad y _ {n} (\langle \boldsymbol {w}, \boldsymbol {x} _ {n} \rangle + b) \geqslant 1 - \xi_ {n}\tag{12.26b}
$$

$$
\xi_ {n} \geqslant 0\tag{12.26c}
$$

for $n = 1 , \ldots , N$ . In contrast to the optimization problem (12.18) for the hard margin SVM, this one is called the soft margin SVM. The parameter $C > 0$ trades off the size of the margin and the total amount of slack that we have. This parameter is called the regularization parameter since, as we will see in the following section, the margin term in the objective function (12.26a) is a regularization term. The margin term $\| \pmb { w } \| ^ { 2 }$ is called the regularizer, and in many books on numerical optimization, the regularization parameter is multiplied with this term (Section 8.2.3). This is in contrast to our formulation in this section. Here a large value of C implies low regularization, as we give the slack variables larger weight, hence giving more priority to examples that do not lie on the correct side of the margin. 

Remark. In the formulation of the soft margin SVM (12.26a) w is regularized, but b is not regularized. We can see this by observing that the regularization term does not contain b. The unregularized term b complicates theoretical analysis (Steinwart and Christmann, 2008, chapter 1) and decreases computational efficiency (Fan et al., 2008). ◇ 

## 12.2.5 Soft Margin SVM: Loss Function View

Let us consider a different approach for deriving the SVM, following the principle of empirical risk minimization (Section 8.2). For the SVM, we choose hyperplanes as the hypothesis class, that is 

$$
f (\boldsymbol {x}) = \langle \boldsymbol {w}, \boldsymbol {x} \rangle + b.\tag{12.27}
$$

We will see in this section that the margin corresponds to the regularization term. The remaining question is, what is the loss function? In contrast to Chapter 9, where we consider regression problems (the output of the predictor is a real number), in this chapter, we consider binary classification problems (the output of the predictor is one of two labels $\{ + 1 , - 1 \} )$ ). Therefore, the error/loss function for each single example– label pair needs to be appropriate for binary classification. For example, the squared loss that is used for regression (9.10b) is not suitable for binary classification. 

Remark. The ideal loss function between binary labels is to count the number of mismatches between the prediction and the label. This means that for a predictor f applied to an example ${ \bf x } _ { n } ,$ we compare the output $f ( { \pmb x } _ { n } )$ with the label $y _ { n }$ . We define the loss to be zero if they match, and one if they do not match. This is denoted by $\mathbf { 1 } ( f ( \pmb { x } _ { n } ) \neq \ - \ - \ - \ - \ y _ { n } )$ and is called the zero-one loss. Unfortunately, the zero-one loss results in a combinatorial optimization problem for finding the best parameters w, b. Combinatorial optimization problems (in contrast to continuous optimization problems discussed in Chapter 7) are in general more challenging to solve. 

What is the loss function corresponding to the SVM? Consider the error between the output of a predictor $f ( { \pmb x } _ { n } )$ and the label $y _ { n }$ . The loss describes the error that is made on the training data. An equivalent way to derive (12.26a) is to use the hinge loss 

$$
\ell (t) = \max \{0, 1 - t \} \quad \text { where } \quad t = y f (\boldsymbol {x}) = y (\langle \boldsymbol {w}, \boldsymbol {x} \rangle + b)  .\tag{12.28}
$$

If $f ( { \pmb x } )$ is on the correct side (based on the corresponding label y) of the hyperplane, and further than distance 1, this means that $t \geqslant 1$ and the hinge loss returns a value of zero. If $f ( { \pmb x } )$ is on the correct side but too close to the hyperplane $( 0 < t < 1 )$ , the example x is within the margin, and the hinge loss returns a positive value. When the example is on the wrong side of the hyperplane $( t < 0 )$ , the hinge loss returns an even larger value, which increases linearly. In other words, we pay a penalty once we are closer than the margin to the hyperplane, even if the prediction is correct, and the penalty increases linearly. An alternative way to express the hinge loss is by considering it as two linear pieces 

$$
\ell (t) = \left\{ \begin{array}{l l} 0 & \text { if } \quad t \geqslant 1 \\ 1 - t & \text { if } \quad t <   1 \end{array} \right.,\tag{12.29}
$$

as illustrated in Figure 12.8. The loss corresponding to the hard margin SVM 12.18 is defined as 

$$
\ell (t) = \left\{ \begin{array}{l l} 0 & \text { if } \quad t \geqslant 1 \\ \infty & \text { if } \quad t <   1 \end{array} \right..\tag{12.30}
$$

©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 


Figure 12.8 The hinge loss is a convex upper bound of zero-one loss.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/1cf8369e222acad5417efab3615ee3cdf3f8dd222858b59b244dc45cd4a4d1e8.jpg)


This loss can be interpreted as never allowing any examples inside the margin. 

For a given training set $\left\{ ( \pmb { x } _ { 1 } , y _ { 1 } ) , \dotsc , ( \pmb { x } _ { N } , y _ { N } ) \right\}$ , we seek to minimize the total loss, while regularizing the objective with $\ell _ { 2 }$ -regularization (see Section 8.2.3). Using the hinge loss (12.28) gives us the unconstrained optimization problem 

$$
\min _ {\boldsymbol {w}, b} \underbrace {\frac {1}{2} \| \boldsymbol {w} \| ^ {2}} _ {\text { regularizer }} + \underbrace {C \sum_ {n = 1} ^ {N} \max \{0 , 1 - y _ {n} (\langle \boldsymbol {w} , \boldsymbol {x} _ {n} \rangle + b) \}} _ {\text { error   term }}.\tag{12.31}
$$

The first term in (12.31) is called the regularization term or the regularizer (see Section 8.2.3), and the second term is called the loss term or the error term. Recall from Section 12.2.4 that the term $\scriptstyle { \frac { 1 } { 2 } } \left\| w \right\| ^ { 2 }$ arises directly from the margin. In other words, margin maximization can be interpreted as regularization. 

In principle, the unconstrained optimization problem in (12.31) can be directly solved with (sub-)gradient descent methods as described in Section 7.1. To see that (12.31) and (12.26a) are equivalent, observe that the hinge loss (12.28) essentially consists of two linear parts, as expressed in (12.29). Consider the hinge loss for a single example-label pair (12.28). We can equivalently replace minimization of the hinge loss over t with a minimization of a slack variable $\xi$ with two constraints. In equation form, 

$$
\min _ {t} \max \{0, 1 - t \}\tag{12.32}
$$

is equivalent to 

$$
\begin{array}{c c} \min _ {\xi , t} & \xi \\ \text { subject   to } & \xi \geqslant 0, \quad \xi \geqslant 1 - t. \end{array}\tag{12.33}
$$

By substituting this expression into (12.31) and rearranging one of the constraints, we obtain exactly the soft margin SVM (12.26a). 

Remark. Let us contrast our choice of the loss function in this section to the loss function for linear regression in Chapter 9. Recall from Section 9.2.1 that for finding maximum likelihood estimators, we usually minimize the negative log-likelihood. Furthermore, since the likelihood term for linear regression with Gaussian noise is Gaussian, the negative log-likelihood for each example is a squared error function. The squared error function is the loss function that is minimized when looking for the maximum likelihood solution. $\diamondsuit$ 

## 12.3 Dual Support Vector Machine

The description of the SVM in the previous sections, in terms of the vari ables w and $b ,$ is known as the primal SVM. Recall that we consider inputs $\pmb { x } \in \mathbb { R } ^ { D }$ with D features. Since w is of the same dimension as x, this means that the number of parameters (the dimension of w) of the opti mization problem grows linearly with the number of features. 

In the following, we consider an equivalent optimization problem (the so-called dual view), which is independent of the number of features. Instead, the number of parameters increases with the number of examples in the training set. We saw a similar idea appear in Chapter 10, where we expressed the learning problem in a way that does not scale with the number of features. This is useful for problems where we have more features than the number of examples in the training dataset. The dual SVM also has the additional advantage that it easily allows kernels to be applied, as we shall see at the end of this chapter. The word “dual” appears often in mathematical literature, and in this particular case it refers to convex duality. The following subsections are essentially an application of convex duality, which we discussed in Section 7.2. 

## 12.3.1 Convex Duality via Lagrange Multipliers

Recall the primal soft margin SVM (12.26a). We call the variables $w , b ,$ and ξ corresponding to the primal SVM the primal variables. We use $\alpha _ { n } \geqslant$ 0 as the Lagrange multiplier corresponding to the constraint (12.26b) that the examples are classified correctly and $\gamma _ { n } \geqslant 0$ as the Lagrange multiplier corresponding to the non-negativity constraint of the slack variable; see (12.26c). The Lagrangian is then given by 

In Chapter 7, we used λ as Lagrange multipliers. In this section, we follow the notation commonly chosen in SVM literature, and use α and γ. 

$$
\begin{array}{l} \mathfrak {L} (\boldsymbol {w}, b, \xi , \alpha , \gamma) = \frac {1}{2} \| \boldsymbol {w} \| ^ {2} + C \sum_ {n = 1} ^ {N} \xi_ {n} \\ \underbrace {- \sum_ {n = 1} ^ {N} \alpha_ {n} (y _ {n} (\langle \boldsymbol {w} , \boldsymbol {x} _ {n} \rangle + b) - 1 + \xi_ {n})} _ {\text { constraint   (12.26b) }} \underbrace {- \sum_ {n = 1} ^ {N} \gamma_ {n} \xi_ {n}} _ {\text { constraint   (12.26c) }}. \end{array} \tag {12.34}
$$



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



representer theorem The representer theorem is actually a collection of theorems saying that the solution of minimizing empirical risk lies in the subspace (Section 2.4.3) defined by the examples. 

By differentiating the Lagrangian (12.34) with respect to the three primal variables $w , b ,$ and ξ respectively, we obtain 

$$
\frac {\partial \mathfrak {L}}{\partial \boldsymbol {w}} = \boldsymbol {w} ^ {\top} - \sum_ {n = 1} ^ {N} \alpha_ {n} y _ {n} \boldsymbol {x} _ {n} ^ {\top},\tag{12.35}
$$

$$
\frac {\partial \mathfrak {L}}{\partial b} = - \sum_ {n = 1} ^ {N} \alpha_ {n} y _ {n},\tag{12.36}
$$

$$
\frac {\partial \mathfrak {L}}{\partial \xi_ {n}} = C - \alpha_ {n} - \gamma_ {n}.\tag{12.37}
$$

We now find the maximum of the Lagrangian by setting each of these partial derivatives to zero. By setting (12.35) to zero, we find 

$$
\pmb {w} = \sum_ {n = 1} ^ {N} \alpha_ {n} y _ {n} \pmb {x} _ {n},\tag{12.38}
$$

which is a particular instance of the representer theorem (Kimeldorf and Wahba, 1970). Equation (12.38) states that the optimal weight vector in the primal is a linear combination of the examples ${ \bf { x } } _ { n }$ . Recall from Section 2.6.1 that this means that the solution of the optimization problem lies in the span of training data. Additionally, the constraint obtained by setting (12.36) to zero implies that the optimal weight vector is an affine combination of the examples. The representer theorem turns out to hold for very general settings of regularized empirical risk minimization (Hofmann et al., 2008; Argyriou and Dinuzzo, 2014). The theorem has more general versions (Scholkopf et¨ $\mathrm { a l . , }$ 2001), and necessary and sufficient conditions on its existence can be found in Yu et al. (2013). 

Remark. The representer theorem (12.38) also provides an explanation of the name “support vector machine.” The examples ${ \bf x } _ { n } ,$ for which the corresponding parameters $\alpha _ { n } = 0$ , do not contribute to the solution w at all. The other examples, where $\alpha _ { n } \ > \ 0$ , are called support vectors since they “support” the hyperplane. ◇ 

By substituting the expression for w into the Lagrangian (12.34), we obtain the dual 

$$
\begin{array}{l} \mathfrak {D} (\xi , \alpha , \gamma) = \frac {1}{2} \sum_ {i = 1} ^ {N} \sum_ {j = 1} ^ {N} y _ {i} y _ {j} \alpha_ {i} \alpha_ {j} \left\langle \boldsymbol {x} _ {i}, \boldsymbol {x} _ {j} \right\rangle - \sum_ {i = 1} ^ {N} y _ {i} \alpha_ {i} \left\langle \sum_ {j = 1} ^ {N} y _ {j} \alpha_ {j} \boldsymbol {x} _ {j}, \boldsymbol {x} _ {i} \right\rangle \\ \qquad + C \sum_ {i = 1} ^ {N} \xi_ {i} - b \sum_ {i = 1} ^ {N} y _ {i} \alpha_ {i} + \sum_ {i = 1} ^ {N} \alpha_ {i} - \sum_ {i = 1} ^ {N} \alpha_ {i} \xi_ {i} - \sum_ {i = 1} ^ {N} \gamma_ {i} \xi_ {i}. \end{array}\tag{12.39}
$$

Note that there are no longer any terms involving the primal variable w. By setting (12.36) to zero, we obtain $\textstyle \sum _ { n = 1 } ^ { N } y _ { n } \alpha _ { n } { \dot { = } } 0$ . Therefore, the term involving b also vanishes. Recall that inner products are symmetric and bilinear (see Section 3.2). Therefore, the first two terms in (12.39) are over the same objects. These terms (colored blue) can be simplified, and we obtain the Lagrangian 

$$
\mathfrak {D} (\xi , \alpha , \gamma) = - \frac {1}{2} \sum_ {i = 1} ^ {N} \sum_ {j = 1} ^ {N} y _ {i} y _ {j} \alpha_ {i} \alpha_ {j} \left\langle \boldsymbol {x} _ {i}, \boldsymbol {x} _ {j} \right\rangle + \sum_ {i = 1} ^ {N} \alpha_ {i} + \sum_ {i = 1} ^ {N} (C - \alpha_ {i} - \gamma_ {i}) \xi_ {i}.\tag{12.40}
$$

The last term in this equation is a collection of all terms that contain slack variables $\xi _ { i }$ . By setting (12.37) to zero, we see that the last term in (12.40) is also zero. Furthermore, by using the same equation and recalling that the Lagrange multiplers $\gamma _ { i }$ are non-negative, we conclude that $\alpha _ { i } \leqslant C$ We now obtain the dual optimization problem of the SVM, which is expressed exclusively in terms of the Lagrange multipliers $\alpha _ { i }$ . Recall from Lagrangian duality (Definition 7.1) that we maximize the dual problem. This is equivalent to minimizing the negative dual problem, such that we end up with the dual SVM 

$$
\begin{array}{l} \min _ {\boldsymbol {\alpha}} \quad \frac {1}{2} \sum_ {i = 1} ^ {N} \sum_ {j = 1} ^ {N} y _ {i} y _ {j} \alpha_ {i} \alpha_ {j} \left\langle \boldsymbol {x} _ {i}, \boldsymbol {x} _ {j} \right\rangle - \sum_ {i = 1} ^ {N} \alpha_ {i} \\ \text { subject   to } \quad \sum_ {i = 1} ^ {N} y _ {i} \alpha_ {i} = 0 \\ 0 \leqslant \alpha_ {i} \leqslant C \quad \text { for   all } \quad i = 1, \ldots , N. \end{array}\tag{12.41}
$$

The equality constraint in (12.41) is obtained from setting (12.36) to zero. The inequality constraint $\alpha _ { i } \ \geqslant \ 0$ is the condition imposed on Lagrange multipliers of inequality constraints (Section 7.2). The inequality constraint $\alpha _ { i } \leqslant C$ is discussed in the previous paragraph. 

The set of inequality constraints in the SVM are called “box constraints” because they limit the vector ${ \pmb { \alpha } } = [ \alpha _ { 1 } , \cdot \cdot \cdot , \alpha _ { N } ] ^ { \top } \in \mathbb { R } ^ { N }$ of Lagrange multipliers to be inside the box defined by 0 and $C$ on each axis. These axis-aligned boxes are particularly efficient to implement in numerical solvers (Dostal´ , 2009, chapter 5). 

Once we obtain the dual parameters $\alpha ,$ we can recover the primal parameters w by using the representer theorem (12.38). Let us call the optimal primal parameter $\boldsymbol { w } ^ { * }$ . However, there remains the question on how to obtain the parameter $b ^ { * }$ . Consider an example ${ \bf { x } } _ { n }$ that lies exactly on the margin’s boundary, i.e., $\langle { \pmb w } ^ { * } , { \pmb x } _ { n } \rangle + b = y _ { n }$ . Recall that $y _ { n }$ is either +1 or 1. Therefore, the only unknown is $b ,$ which can be computed by 

$$
b ^ {*} = y _ {n} - \left\langle \boldsymbol {w} ^ {*}, \boldsymbol {x} _ {n} \right\rangle .\tag{12.42}
$$

Remark. In principle, there may be no examples that lie exactly on the margin. In this case, we should compute $\left| y _ { n } - \left. \pmb { w } ^ { * } , \pmb { x } _ { n } \right. \right|$ for all support vectors and take the median value of this absolute value difference to be 

Figure 12.9 Convex hulls. (a) Convex hull of points, some of which lie within the boundary; (b) convex hulls around positive and negative examples. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/2469afd7822e0d8375bce7aa42dc8e713012119d3950fed30f52711805ccc496.jpg)



the value of $b ^ { * }$ . A derivation of this can be found in http: $/ / \mathtt { f }$ ouryears. eu/2012/06/07/the-svm-bias-term-conspiracy/. ◇


## 12.3.2 Dual SVM: Convex Hull View

Another approach to obtain the dual SVM is to consider an alternative geometric argument. Consider the set of examples ${ \bf { x } } _ { n }$ with the same label. We would like to build a convex set that contains all the examples such that it is the smallest possible set. This is called the convex hull and is illustrated in Figure 12.9. 

Let us first build some intuition about a convex combination of points. Consider two points $\mathbf { \delta x } _ { 1 }$ and $\mathbf { \boldsymbol { x } } _ { 2 }$ and corresponding non-negative weights $\alpha _ { 1 } , \alpha _ { 2 } \geqslant 0$ such that $\alpha _ { 1 } { + } \alpha _ { 2 } = 1$ . The equation $\alpha _ { 1 } { \pmb x } _ { 1 } \mathrm { + } \alpha _ { 2 } { \pmb x } _ { 2 }$ describes each point on a line between $\mathbf { \mathcal { x } } _ { 1 }$ and $\mathbf { \boldsymbol { x } } _ { 2 }$ . Consider what happens when we add a third point $\mathbf { x } _ { 3 }$ along with a weight $\alpha _ { 3 } \ \geqslant \ 0$ such that $\textstyle \sum _ { n = 1 } ^ { 3 } \alpha _ { n } \ = \ 1$ The convex combination of these three points ${ \pmb x } _ { 1 } , { \pmb x } _ { 2 } , { \pmb x } _ { 3 }$ spans a twodimensional area. The convex hull of this area is the triangle formed by the edges corresponding to each pair of of points. As we add more points, and the number of points becomes greater than the number of dimensions, some of the points will be inside the convex hull, as we can see in Figure 12.9(a). 

In general, building a convex convex hull can be done by introducing non-negative weights $\alpha _ { n } \geqslant 0$ corresponding to each example ${ \bf { x } } _ { n }$ . Then the convex hull can be described as the set 

$$
\operatorname{conv} (\boldsymbol {X}) = \left\{\sum_ {n = 1} ^ {N} \alpha_ {n} \boldsymbol {x} _ {n} \right\} \quad \text { with } \quad \sum_ {n = 1} ^ {N} \alpha_ {n} = 1 \quad \text { and } \quad \alpha_ {n} \geqslant 0,\tag{12.43}
$$

for all $n = 1 , \ldots , N$ . If the two clouds of points corresponding to the positive and negative classes are separated, then the convex hulls do not overlap. Given the training data $( { \pmb x } _ { 1 } , y _ { 1 } ) , \dots , ( { \pmb x } _ { N } , y _ { N } )$ , we form two convex hulls, corresponding to the positive and negative classes respectively. We pick a point $^ { c , }$ which is in the convex hull of the set of positive examples, and is closest to the negative class distribution. Similarly, we pick a point d in the convex hull of the set of negative examples and is closest to the positive class distribution; see Figure 12.9(b). We define a difference vector between d and c as 

$$
\boldsymbol {w} := \boldsymbol {c} - \boldsymbol {d}.\tag{12.44}
$$

Picking the points c and d as in the preceding cases, and requiring them to be closest to each other is equivalent to minimizing the length/norm of w, so that we end up with the corresponding optimization problem 

$$
\arg \min _ {\boldsymbol {w}} \| \boldsymbol {w} \| = \arg \min _ {\boldsymbol {w}} \frac {1}{2} \| \boldsymbol {w} \| ^ {2}.\tag{12.45}
$$

Since c must be in the positive convex hull, it can be expressed as a convex combination of the positive examples, i.e., for non-negative coefficients $\alpha _ { n } ^ { + }$ 

$$
\pmb {c} = \sum_ {n: y _ {n} = + 1} \alpha_ {n} ^ {+} \pmb {x} _ {n}.\tag{12.46}
$$

In (12.46), we use the notation $n : y _ { n } = + 1$ to indicate the set of indices n for which $y _ { n } = + 1$ . Similarly, for the examples with negative labels, we obtain 

$$
\boldsymbol {d} = \sum_ {n: y _ {n} = - 1} \alpha_ {n} ^ {-} \boldsymbol {x} _ {n}.\tag{12.47}
$$

By substituting (12.44), (12.46), and (12.47) into (12.45), we obtain the objective 

$$
\min _ {\boldsymbol {\alpha}} \frac {1}{2} \left\| \sum_ {n: y _ {n} = + 1} \alpha_ {n} ^ {+} \boldsymbol {x} _ {n} - \sum_ {n: y _ {n} = - 1} \alpha_ {n} ^ {-} \boldsymbol {x} _ {n} \right\| ^ {2}.\tag{12.48}
$$

Let α be the set of all coefficients, i.e., the concatenation of $\alpha ^ { + }$ and $\pmb { \alpha } ^ { - }$ Recall that we require that for each convex hull that their coefficients sum to one, 

$$
\sum_ {n: y _ {n} = + 1} \alpha_ {n} ^ {+} = 1 \quad \text { and } \quad \sum_ {n: y _ {n} = - 1} \alpha_ {n} ^ {-} = 1.\tag{12.49}
$$

This implies the constraint 

$$
\sum_ {n = 1} ^ {N} y _ {n} \alpha_ {n} = 0.\tag{12.50}
$$



©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 



This result can be seen by multiplying out the individual classes 

$$
\begin{array}{c} \sum_ {n = 1} ^ {N} y _ {n} \alpha_ {n} = \sum_ {n: y _ {n} = + 1} (+ 1) \alpha_ {n} ^ {+} + \sum_ {n: y _ {n} = - 1} (- 1) \alpha_ {n} ^ {-} \\ = \sum_ {n: y _ {n} = + 1} \alpha_ {n} ^ {+} - \sum_ {n: y _ {n} = - 1} \alpha_ {n} ^ {-} = 1 - 1 = 0. \end{array}\tag{12.51a}
$$

(12.51b) 

The objective function (12.48) and the constraint (12.50), along with the assumption that ${ \pmb { \alpha } } \geqslant { \bf 0 } .$ , give us a constrained (convex) optimization problem. This optimization problem can be shown to be the same as that of the dual hard margin SVM (Bennett and Bredensteiner, 2000a). 

Remark. To obtain the soft margin dual, we consider the reduced hull. The reduced hull is similar to the convex hull but has an upper bound to the size of the coefficients α. The maximum possible value of the elements of α restricts the size that the convex hull can take. In other words, the bound on α shrinks the convex hull to a smaller volume (Bennett and Bredensteiner, 2000b). ◇ 

## 12.4 Kernels

Consider the formulation of the dual SVM (12.41). Notice that the inner product in the objective occurs only between examples $\mathbf { \nabla } _ { \mathbf { x } _ { i } }$ and $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { j } }$ There are no inner products between the examples and the parameters. Therefore, if we consider a set of features $\phi ( { \pmb x } _ { i } )$ to represent ${ \mathbf { x } } _ { i } ,$ the only change in the dual SVM will be to replace the inner product. This modularity, where the choice of the classification method (the SVM) and the choice of the feature representation $\phi ( { \pmb x } )$ can be considered separately, provides flexibility for us to explore the two problems independently. In this section, we discuss the representation $\phi ( { \pmb x } )$ and briefly introduce the idea of kernels, but do not go into the technical details. 

Since $\phi ( { \pmb x } )$ could be a non-linear function, we can use the SVM (which assumes a linear classifier) to construct classifiers that are nonlinear in the examples ${ \bf { x } } _ { n }$ . This provides a second avenue, in addition to the soft margin, for users to deal with a dataset that is not linearly separable. It turns out that there are many algorithms and statistical methods that have this property that we observed in the dual SVM: the only inner products are those that occur between examples. Instead of explicitly defining a non-linear feature map $\phi ( \cdot )$ and computing the resulting inner product between examples $\mathbf { \nabla } _ { \mathbf { x } _ { i } }$ and $\boldsymbol { \mathscr { x } } _ { j }$ , we define a similarity function $k ( \pmb { x } _ { i } , \pmb { x } _ { j } )$ between $\mathbf { \nabla } _ { \mathbf { x } _ { i } }$ and $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { j } }$ . For a certain class of similarity functions, called kernels, the similarity function implicitly defines a non-linear feature map $\phi ( \cdot )$ Kernels are by definition functions $k : \mathcal { X } \times \mathcal { X } $ <sup>R</sup> for which there exists a Hilbert space and $\phi : \mathcal { X } \to \mathcal { H }$ a feature map such that 

$$
k (\pmb {x} _ {i}, \pmb {x} _ {j}) = \langle \phi (\pmb {x} _ {i}), \phi (\pmb {x} _ {j}) \rangle_ {\mathcal {H}}.\tag{12.52}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/651702a26809d584798027a5988b44ff4935a9e666f740298a966624771a5c04.jpg)



(a) SVM with linear kernel


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/89cb6003efa34ea7b37cf15ab8ef0b0fdf53ee4bb4aeb76bcef4265998bfa98b.jpg)



Figure 12.10 SVM with different kernels. Note that while the decision boundary is nonlinear, the underlying problem being solved is for a linear separating hyperplane (albeit with a nonlinear kernel).



(b) SVM with RBF kernel


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/cf4e719fa1d495750d3651b1bc1887f54dc39857949b6189f449bf32c569dfcc.jpg)



(c) SVM with polynomial (degree 2) kernel


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-28/51743cfc-c7bd-4178-ae8d-8a1239a7ed2b/b74654f31910a2415552c4ba1e8855c328431bea97b651cb8b9dc6fdc05e8ca6.jpg)



(d) SVM with polynomial (degree 3) kernel


There is a unique reproducing kernel Hilbert space associated with every kernel k (Aronszajn, 1950; Berlinet and Thomas-Agnan, 2004). In this unique association, $\phi ( { \pmb x } ) = k ( \cdot , { \pmb x } )$ is called the canonical feature map. The generalization from an inner product to a kernel function (12.52) is known as the kernel trick (Scholkopf and Smola¨ , 2002; Shawe-Taylor and Cristianini, 2004), as it hides away the explicit non-linear feature map. 

The matrix ${ \pmb K } \in \mathbb { R } ^ { N \times N }$ , resulting from the inner products or the appli cation of $k ( \cdot , \cdot )$ to a dataset, is called the Gram matrix, and is often just referred to as the kernel matrix. Kernels must be symmetric and positive semidefinite functions so that every kernel matrix K is symmetric and positive semidefinite (Section 3.2.3): 

$$
\forall \boldsymbol {z} \in \mathbb {R} ^ {N}: \boldsymbol {z} ^ {\top} \boldsymbol {K} \boldsymbol {z} \geqslant 0.\tag{12.53}
$$

Some popular examples of kernels for multivariate real-valued data $\pmb { x } _ { i } \in \mathbb { \Sigma }$ $\mathbb { R } ^ { D }$ are the polynomial kernel, the Gaussian radial basis function kernel, and the rational quadratic kernel (Scholkopf and Smola¨ , 2002; Rasmussen and Williams, 2006). Figure 12.10 illustrates the effect of different kernels on separating hyperplanes on an example dataset. Note that we are still solving for hyperplanes, that is, the hypothesis class of functions are still linear. The non-linear surfaces are due to the kernel function. 

Remark. Unfortunately for the fledgling machine learner, there are multiple meanings of the word “kernel.” In this chapter, the word “kernel” comes from the idea of the reproducing kernel Hilbert space (RKHS) (Aron szajn, 1950; Saitoh, 1988). We have discussed the idea of the kernel in linear algebra (Section 2.7.3), where the kernel is another word for the null space. The third common use of the word “kernel” in machine learning is the smoothing kernel in kernel density estimation (Section 11.5). ◇ 

Since the explicit representation $\phi ( { \pmb x } )$ is mathematically equivalent to the kernel representation $k ( \pmb { x } _ { i } , \pmb { x } _ { j } )$ , a practitioner will often design the kernel function such that it can be computed more efficiently than the inner product between explicit feature maps. For example, consider the polynomial kernel (Scholkopf and Smola¨ , 2002), where the number of terms in the explicit expansion grows very quickly (even for polynomials of low degree) when the input dimension is large. The kernel function only requires one multiplication per input dimension, which can provide significant computational savings. Another example is the Gaussian radial basis function kernel (Scholkopf and Smola¨ , 2002; Rasmussen and Williams, 2006), where the corresponding feature space is infinite dimensional. In this case, we cannot explicitly represent the feature space but can still compute similarities between a pair of examples using the kernel. 

Another useful aspect of the kernel trick is that there is no need for the original data to be already represented as multivariate real-valued data. Note that the inner product is defined on the output of the function $\phi ( \cdot )$ , but does not restrict the input to real numbers. Hence, the function $\phi ( \cdot )$ and the kernel function $k ( \cdot , \cdot )$ can be defined on any object, e.g., sets, sequences, strings, graphs, and distributions (Ben-Hur et al., 2008; Gartner¨ , 2008; Shi et al., 2009; Sriperumbudur et al., 2010; Vishwanathan et al., 2010). 

## 12.5 Numerical Solution

We conclude our discussion of SVMs by looking at how to express the problems derived in this chapter in terms of the concepts presented in Chapter 7. We consider two different approaches for finding the optimal solution for the SVM. First we consider the loss view of SVM 8.2.2 and express this as an unconstrained optimization problem. Then we express the constrained versions of the primal and dual SVMs as quadratic programs in standard form 7.3.2. 

Consider the loss function view of the SVM (12.31). This is a convex unconstrained optimization problem, but the hinge loss (12.28) is not differentiable. Therefore, we apply a subgradient approach for solving it. However, the hinge loss is differentiable almost everywhere, except for one single point at the hinge $t = 1$ . At this point, the gradient is a set of possible values that lie between 0 and 1. Therefore, the subgradient g of the hinge loss is given by 

$$
g (t) = \left\{ \begin{array}{l l} - 1 & t <   1 \\ [ - 1, 0 ] & t = 1 \\ 0 & t > 1 \end{array} \right..\tag{12.54}
$$

Using this subgradient, we can apply the optimization methods presented in Section 7.1. 

Both the primal and the dual SVM result in a convex quadratic programming problem (constrained optimization). Note that the primal SVM in (12.26a) has optimization variables that have the size of the dimension D of the input examples. The dual SVM in (12.41) has optimization variables that have the size of the number N of examples. 

To express the primal SVM in the standard form (7.45) for quadratic programming, let us assume that we use the dot product (3.5) as the inner product. We rearrange the equation for the primal SVM (12.26a), such that the optimization variables are all on the right and the inequality of the constraint matches the standard form. This yields the optimization 

$$
\begin{array}{l l} \min _ {\boldsymbol {w}, b, \boldsymbol {\xi}} & \frac {1}{2} \| \boldsymbol {w} \| ^ {2} + C \sum_ {n = 1} ^ {N} \xi_ {n} \\ \text { subject   to } & - y _ {n} \boldsymbol {x} _ {n} ^ {\top} \boldsymbol {w} - y _ {n} b - \xi_ {n} \leqslant - 1 \\ & - \xi_ {n} \leqslant 0 \end{array}\tag{12.55}
$$

$n = 1 , \ldots , N$ . By concatenating the variables ${ \pmb w } , b , { \pmb x } _ { n }$ into a single vector, and carefully collecting the terms, we obtain the following matrix form of the soft margin SVM: 

$$
\min _ {\boldsymbol {w}, b, \boldsymbol {\xi}} \quad \frac {1}{2} \left[ \begin{array}{c} \boldsymbol {w} \\ b \\ \boldsymbol {\xi} \end{array} \right] ^ {\top} \left[ \begin{array}{c c} \boldsymbol {I} _ {D} & \boldsymbol {0} _ {D, N + 1} \\ \boldsymbol {0} _ {N + 1, D} & \boldsymbol {0} _ {N + 1, N + 1} \end{array} \right] \left[ \begin{array}{c} \boldsymbol {w} \\ b \\ \boldsymbol {\xi} \end{array} \right] + \left[ \begin{array}{c c} \boldsymbol {0} _ {D + 1, 1} & C \boldsymbol {1} _ {N, 1} \end{array} \right] ^ {\top} \left[ \begin{array}{c} \boldsymbol {w} \\ b \\ \boldsymbol {\xi} \end{array} \right]
$$

$$
\text { subject   to } \left[ \begin{array}{c c c} - \boldsymbol {Y} \boldsymbol {X} & - \boldsymbol {y} & - \boldsymbol {I} _ {N} \\ \boldsymbol {0} _ {N, D + 1} & & - \boldsymbol {I} _ {N} \end{array} \right] \left[ \begin{array}{c} \boldsymbol {w} \\ b \\ \boldsymbol {\xi} \end{array} \right] \leqslant \left[ \begin{array}{c} - \boldsymbol {1} _ {N, 1} \\ \boldsymbol {0} _ {N, 1} \end{array} \right].\tag{12.56}
$$

In the preceding optimization problem, the minimization is over the parameters $[ { \pmb w } ^ { \top } , \bar { b , { \pmb \xi } } ^ { \bar { \top } } ] ^ { \top } \ \in \ \mathbb { R } ^ { D + \bar { 1 + } N }$ , and we use the notation: $\pmb { I _ { m } }$ to represent the identity matrix of size $m \times m , \mathbf { 0 } _ { m , n }$ to represent the matrix of zeros of size m $\times \ n ,$ and $\mathbf { 1 } _ { m , n }$ to represent the matrix of ones of size $m \times n$ . In addition, y is the vector of labels $\left[ y _ { 1 } , \cdot \cdot \cdot , y _ { N } \right] ^ { \top } , Y = \mathrm { d i a g } ( { \pmb y } )$ 

is an N by N matrix where the elements of the diagonal are from $^ { y , }$ and $\pmb { X } \in \mathbb { R } ^ { N \times D }$ is the matrix obtained by concatenating all the examples. 

We can similarly perform a collection of terms for the dual version of the SVM (12.41). To express the dual SVM in standard form, we first have to express the kernel matrix K such that each entry is $K _ { i j } = k ( \pmb { x } _ { i } , \pmb { x } _ { j } )$ . If we have an explicit feature representation $\mathbf { \nabla } _ { \mathbf { x } _ { i } }$ then we define $K _ { i j } = \langle { \pmb x } _ { i } , { \pmb x } _ { j } \rangle$ For convenience of notation we introduce a matrix with zeros everywhere except on the diagonal, where we store the labels, that is, $\pmb { Y } = \mathrm { d i a g } ( \pmb { y } )$ The dual SVM can be written as 

$$
\begin{array}{r l} \min _ {\boldsymbol {\alpha}} & \frac {1}{2} \boldsymbol {\alpha} ^ {\top} \boldsymbol {Y} \boldsymbol {K} \boldsymbol {Y} \boldsymbol {\alpha} - \mathbf {1} _ {N, 1} ^ {\top} \boldsymbol {\alpha} \\ \text {subject to} & \left[ \begin{array}{c} \boldsymbol {y} ^ {\top} \\ - \boldsymbol {y} ^ {\top} \\ - \boldsymbol {I} _ {N} \\ \boldsymbol {I} _ {N} \end{array} \right] \boldsymbol {\alpha} \leqslant \left[ \begin{array}{c} \mathbf {0} _ {N + 2, 1} \\ C \mathbf {1} _ {N, 1} \end{array} \right]. \end{array}\tag{12.57}
$$

Remark. In Sections 7.3.1 and 7.3.2, we introduced the standard forms of the constraints to be inequality constraints. We will express the dual SVM’s equality constraint as two inequality constraints, i.e., 

$$
\boldsymbol {A} \boldsymbol {x} = \boldsymbol {b} \quad \text { is   replaced   by } \quad \boldsymbol {A} \boldsymbol {x} \leqslant \boldsymbol {b} \quad \text { and } \quad \boldsymbol {A} \boldsymbol {x} \geqslant \boldsymbol {b}.\tag{12.58}
$$

Particular software implementations of convex optimization methods may provide the ability to express equality constraints. ◇ 

Since there are many different possible views of the SVM, there are many approaches for solving the resulting optimization problem. The approach presented here, expressing the SVM problem in standard convex optimization form, is not often used in practice. The two main implementations of SVM solvers are Chang and Lin (2011) (which is open source) and Joachims (1999). Since SVMs have a clear and well-defined optimization problem, many approaches based on numerical optimization techniques (Nocedal and Wright, 2006) can be applied (Shawe-Taylor and Sun, 2011). 

## 12.6 Further Reading

The SVM is one of many approaches for studying binary classification. Other approaches include the perceptron, logistic regression, Fisher discriminant, nearest neighbor, naive Bayes, and random forest (Bishop, 2006; Murphy, 2012). A short tutorial on SVMs and kernels on discrete sequences can be found in Ben-Hur et al. (2008). The development of SVMs is closely linked to empirical risk minimization, discussed in Section 8.2. Hence, the SVM has strong theoretical properties (Vapnik, 2000; Steinwart and Christmann, 2008). The book about kernel methods (Scholkopf¨ and Smola, 2002) includes many details of support vector machines and how to optimize them. A broader book about kernel methods (Shawe-Taylor and Cristianini, 2004) also includes many linear algebra approaches for different machine learning problems. 

An alternative derivation of the dual SVM can be obtained using the idea of the Legendre–Fenchel transform (Section 7.3.3). The derivation considers each term of the unconstrained formulation of the SVM (12.31) separately and calculates their convex conjugates (Rifkin and Lippert, 2007). Readers interested in the functional analysis view (also the regularization methods view) of SVMs are referred to the work by Wahba (1990). Theoretical exposition of kernels (Aronszajn, 1950; Schwartz, 1964; Saitoh, 1988; Manton and Amblard, 2015) requires a basic ground ing in linear operators (Akhiezer and Glazman, 1993). The idea of kernels have been generalized to Banach spaces (Zhang et al., 2009) and Kre˘ın spaces (Ong et al., 2004; Loosli et al., 2016). 

Observe that the hinge loss has three equivalent representations, as shown in (12.28) and (12.29), as well as the constrained optimization problem in (12.33). The formulation (12.28) is often used when comparing the SVM loss function with other loss functions (Steinwart, 2007). The two-piece formulation (12.29) is convenient for computing subgradients, as each piece is linear. The third formulation (12.33), as seen in Section 12.5, enables the use of convex quadratic programming (Section 7.3.2) tools. 

Since binary classification is a well-studied task in machine learning, other words are also sometimes used, such as discrimination, separation, and decision. Furthermore, there are three quantities that can be the out put of a binary classifier. First is the output of the linear function itself (often called the score), which can take any real value. This output can be used for ranking the examples, and binary classification can be thought of as picking a threshold on the ranked examples (Shawe-Taylor and Cristianini, 2004). The second quantity that is often considered the output of a binary classifier is the output determined after it is passed through a non-linear function to constrain its value to a bounded range, for example in the interval [0, 1]. A common non-linear function is the sigmoid function (Bishop, 2006). When the non-linearity results in well-calibrated probabilities (Gneiting and Raftery, 2007; Reid and Williamson, 2011), this is called class probability estimation. The third output of a binary classifier is the final binary decision +1, 1 , which is the one most commonly assumed to be the output of the classifier. 

The SVM is a binary classifier that does not naturally lend itself to a probabilistic interpretation. There are several approaches for converting the raw output of the linear function (the score) into a calibrated class probability estimate $( P ( Y = 1 | X = x ) )$ that involve an additional calibration step (Platt, 2000; Zadrozny and Elkan, 2001; Lin et al., 2007). From the training perspective, there are many related probabilistic approaches. We mentioned at the end of Section 12.2.5 that there is a relationship between loss function and the likelihood (also compare Sections 8.2 and 8.3). The maximum likelihood approach corresponding to a well-calibrated transformation during training is called logistic regression, which comes from a class of methods called generalized linear models. Details of logistic regression from this point of view can be found in Agresti (2002, chapter 5) and McCullagh and Nelder (1989, chapter 4). Naturally, one could take a more Bayesian view of the classifier output by estimating a posterior distribution using Bayesian logistic regression. The Bayesian view also includes the specification of the prior, which includes design choices such as conjugacy (Section 6.6.1) with the likelihood. Additionally, one could consider latent functions as priors, which results in Gaussian process classification (Rasmussen and Williams, 2006, chapter 3). 

## References



Abel, Niels H. 1826. D´emonstration de l’Impossibilit´e de la R´esolution Alg´ebrique des Equations G<sup>´</sup> ´en´erales qui Passent le Quatri`eme Degr´e. Grøndahl and Søn. 





Adhikari, Ani, and DeNero, John. 2018. Computational and Inferential Thinking: The Foundations of Data Science. Gitbooks. 





Agarwal, Arvind, and Daum´e III, Hal. 2010. A Geometric View of Conjugate Priors. Machine Learning, 81(1), 99–113. 





Agresti, A. 2002. Categorical Data Analysis. Wiley. 





Akaike, Hirotugu. 1974. A New Look at the Statistical Model Identification. IEEE Transactions on Automatic Control, 19(6), 716–723. 





Akhiezer, Naum I., and Glazman, Izrail M. 1993. Theory of Linear Operators in Hilbert Space. Dover Publications. 





Alpaydin, Ethem. 2010. Introduction to Machine Learning. MIT Press. 





Amari, Shun-ichi. 2016. Information Geometry and Its Applications. Springer. 





Argyriou, Andreas, and Dinuzzo, Francesco. 2014. A Unifying View of Representer Theorems. In: Proceedings of the International Conference on Machine Learning. 





Aronszajn, Nachman. 1950. Theory of Reproducing Kernels. Transactions of the American Mathematical Society, 68, 337–404. 





Axler, Sheldon. 2015. Linear Algebra Done Right. Springer. 





Bakir, Gokhan, Hofmann, Thomas, Sch¨ olkopf, Bernhard, Smola, Alexander J., Taskar,¨ Ben, and Vishwanathan, S. V. N. (eds). 2007. Predicting Structured Data. MIT Press. 





Barber, David. 2012. Bayesian Reasoning and Machine Learning. Cambridge University Press. 





Barndorff-Nielsen, Ole. 2014. Information and Exponential Families: In Statistical The ory. Wiley. 





Bartholomew, David, Knott, Martin, and Moustaki, Irini. 2011. Latent Variable Models and Factor Analysis: A Unified Approach. Wiley. 





Baydin, Atılım G., Pearlmutter, Barak A., Radul, Alexey A., and Siskind, Jeffrey M. 2018. Automatic Differentiation in Machine Learning: A Survey. Journal of Machine Learning Research, 18, 1–43. 





Beck, Amir, and Teboulle, Marc. 2003. Mirror Descent and Nonlinear Projected Subgradient Methods for Convex Optimization. Operations Research Letters, 31(3), 167– 175. 





Belabbas, Mohamed-Ali, and Wolfe, Patrick J. 2009. Spectral Methods in Machine Learning and New Strategies for Very Large Datasets. Proceedings of the National Academy of Sciences, 0810600105. 





Belkin, Mikhail, and Niyogi, Partha. 2003. Laplacian Eigenmaps for Dimensionality Reduction and Data Representation. Neural Computation, 15(6), 1373–1396. 





Ben-Hur, Asa, Ong, Cheng Soon, Sonnenburg, Soren, Sch¨ olkopf, Bernhard, and R¨ atsch,¨ Gunnar. 2008. Support Vector Machines and Kernels for Computational Biology. PLoS Computational Biology, 4(10), e1000173. 





Bennett, Kristin P., and Bredensteiner, Erin J. 2000a. Duality and Geometry in SVM Classifiers. In: Proceedings of the International Conference on Machine Learning. 





Bennett, Kristin P., and Bredensteiner, Erin J. 2000b. Geometry in Learning. Pages 132–145 of: Geometry at Work. Mathematical Association of America. 





Berlinet, Alain, and Thomas-Agnan, Christine. 2004. Reproducing Kernel Hilbert Spaces in Probability and Statistics. Springer. 





Bertsekas, Dimitri P. 1999. Nonlinear Programming. Athena Scientific. 





Bertsekas, Dimitri P. 2009. Convex Optimization Theory. Athena Scientific. 





Bickel, Peter J., and Doksum, Kjell. 2006. Mathematical Statistics, Basic Ideas and Selected Topics. Vol. 1. Prentice Hall. 





Bickson, Danny, Dolev, Danny, Shental, Ori, Siegel, Paul H., and Wolf, Jack K. 2007. Linear Detection via Belief Propagation. In: Proceedings of the Annual Allerton Con ference on Communication, Control, and Computing. 





Billingsley, Patrick. 1995. Probability and Measure. Wiley. 





Bishop, Christopher M. 1995. Neural Networks for Pattern Recognition. Clarendon Press. 





Bishop, Christopher M. 1999. Bayesian PCA. In: Advances in Neural Information Pro cessing Systems. 





Bishop, Christopher M. 2006. Pattern Recognition and Machine Learning. Springer. 





Blei, David M., Kucukelbir, Alp, and McAuliffe, Jon D. 2017. Variational Inference: A Review for Statisticians. Journal of the American Statistical Association, 112(518), 859–877. 





Blum, Arvim, and Hardt, Moritz. 2015. The Ladder: A Reliable Leaderboard for Machine Learning Competitions. In: International Conference on Machine Learning. 





Bonnans, J. Fr´ed´eric, Gilbert, J. Charles, Lemar´echal, Claude, and Sagastizabal, Clau-´ dia A. 2006. Numerical Optimization: Theoretical and Practical Aspects. Springer. 





Borwein, Jonathan M., and Lewis, Adrian S. 2006. Convex Analysis and Nonlinear Optimization. 2nd edn. Canadian Mathematical Society. 





Bottou, L´eon. 1998. Online Algorithms and Stochastic Approximations. Pages 9–42 of: Online Learning and Neural Networks. Cambridge University Press. 





Bottou, L´eon, Curtis, Frank E., and Nocedal, Jorge. 2018. Optimization Methods for Large-Scale Machine Learning. SIAM Review, 60(2), 223–311. 





Boucheron, Stephane, Lugosi, Gabor, and Massart, Pascal. 2013. Concentration In equalities: A Nonasymptotic Theory of Independence. Oxford University Press. 





Boyd, Stephen, and Vandenberghe, Lieven. 2004. Convex Optimization. Cambridge University Press. 





Boyd, Stephen, and Vandenberghe, Lieven. 2018. Introduction to Applied Linear Alge bra. Cambridge University Press. 





Brochu, Eric, Cora, Vlad M., and de Freitas, Nando. 2009. A Tutorial on Bayesian Optimization of Expensive Cost Functions, with Application to Active User Modeling and Hierarchical Reinforcement Learning. Tech. rept. TR-2009-023. Department of Computer Science, University of British Columbia. 





Brooks, Steve, Gelman, Andrew, Jones, Galin L., and Meng, Xiao-Li (eds). 2011. Hand book of Markov Chain Monte Carlo. Chapman and Hall/CRC. 





Brown, Lawrence D. 1986. Fundamentals of Statistical Exponential Families: With Ap plications in Statistical Decision Theory. Institute of Mathematical Statistics. 





Bryson, Arthur E. 1961. A Gradient Method for Optimizing Multi-Stage Allocation Processes. In: Proceedings of the Harvard University Symposium on Digital Computers and Their Applications. 





Bubeck, S´ebastien. 2015. Convex Optimization: Algorithms and Complexity. Founda tions and Trends in Machine Learning, 8(3-4), 231–357. 





Buhlmann, Peter, and Van De Geer, Sara. 2011.¨ Statistics for High-Dimensional Data. Springer. 





Burges, Christopher. 2010. Dimension Reduction: A Guided Tour. Foundations and Trends in Machine Learning, 2(4), 275–365. 





Carroll, J Douglas, and Chang, Jih-Jie. 1970. Analysis of Individual Differences in Multidimensional Scaling via an N-Way Generalization of “Eckart-Young” Decomposition. Psychometrika, 35(3), 283–319. 





Casella, George, and Berger, Roger L. 2002. Statistical Inference. Duxbury. 





C¸inlar, Erhan. 2011. Probability and Stochastics. Springer. 





Chang, Chih-Chung, and Lin, Chih-Jen. 2011. LIBSVM: A Library for Support Vector Machines. ACM Transactions on Intelligent Systems and Technology, 2, 27:1–27:27. 





Cheeseman, Peter. 1985. In Defense of Probability. In: Proceedings of the International Joint Conference on Artificial Intelligence. 





Chollet, Francois, and Allaire, J. J. 2018. Deep Learning with R. Manning Publications. 





Codd, Edgar F. 1990. The Relational Model for Database Management. Addison-Wesley Longman Publishing. 





Cunningham, John P., and Ghahramani, Zoubin. 2015. Linear Dimensionality Reduction: Survey, Insights, and Generalizations. Journal of Machine Learning Research, 16, 2859–2900. 





Datta, Biswa N. 2010. Numerical Linear Algebra and Applications. SIAM. 





Davidson, Anthony C., and Hinkley, David V. 1997. Bootstrap Methods and Their Appli cation. Cambridge University Press. 





Dean, Jeffrey, Corrado, Greg S., Monga, Rajat, and Chen, et al. 2012. Large Scale Distributed Deep Networks. In: Advances in Neural Information Processing Systems. 





Deisenroth, Marc P., and Mohamed, Shakir. 2012. Expectation Propagation in Gaussian Process Dynamical Systems. Pages 2618–2626 of: Advances in Neural Informa tion Processing Systems. 





Deisenroth, Marc P., and Ohlsson, Henrik. 2011. A General Perspective on Gaussian Filtering and Smoothing: Explaining Current and Deriving New Algorithms. In: Proceedings of the American Control Conference. 





Deisenroth, Marc P., Fox, Dieter, and Rasmussen, Carl E. 2015. Gaussian Processes for Data-Efficient Learning in Robotics and Control. IEEE Transactions on Pattern Analysis and Machine Intelligence, 37(2), 408–423. 





Dempster, Arthur P., Laird, Nan M., and Rubin, Donald B. 1977. Maximum Likelihood from Incomplete Data via the EM Algorithm. Journal of the Royal Statistical Society, 39(1), 1–38. 





Deng, Li, Seltzer, Michael L., Yu, Dong, Acero, Alex, Mohamed, Abdel-rahman, and Hinton, Geoffrey E. 2010. Binary Coding of Speech Spectrograms Using a Deep Auto-Encoder. In: Proceedings of Interspeech. 





Devroye, Luc. 1986. Non-Uniform Random Variate Generation. Springer. 





Donoho, David L., and Grimes, Carrie. 2003. Hessian Eigenmaps: Locally Linear Embedding Techniques for High-Dimensional Data. Proceedings of the National Academy of Sciences, 100(10), 5591–5596. 





Dostal, Zden ´ ˘ek. 2009. Optimal Quadratic Programming Algorithms: With Applications to Variational Inequalities. Springer. 





Douven, Igor. 2017. Abduction. In: The Stanford Encyclopedia of Philosophy. Meta physics Research Lab, Stanford University. 





Downey, Allen B. 2014. Think Stats: Exploratory Data Analysis. 2nd edn. O’Reilly Media. 





Dreyfus, Stuart. 1962. The Numerical Solution of Variational Problems. Journal of Mathematical Analysis and Applications, 5(1), 30–45. 





Drumm, Volker, and Weil, Wolfgang. 2001. Lineare Algebra und Analytische Geometrie. Lecture Notes, Universitat Karlsruhe (TH).¨ 





Dudley, Richard M. 2002. Real Analysis and Probability. Cambridge University Press. 





©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 





Eaton, Morris L. 2007. Multivariate Statistics: A Vector Space Approach. Institute of Mathematical Statistics Lecture Notes. 





Eckart, Carl, and Young, Gale. 1936. The Approximation of One Matrix by Another of Lower Rank. Psychometrika, 1(3), 211–218. 





Efron, Bradley, and Hastie, Trevor. 2016. Computer Age Statistical Inference: Algorithms, Evidence and Data Science. Cambridge University Press. 





Efron, Bradley, and Tibshirani, Robert J. 1993. An Introduction to the Bootstrap. Chapman and Hall/CRC. 





Elliott, Conal. 2009. Beautiful Differentiation. In: International Conference on Functional Programming. 





Evgeniou, Theodoros, Pontil, Massimiliano, and Poggio, Tomaso. 2000. Statistical Learning Theory: A Primer. International Journal of Computer Vision, 38(1), 9–13. 





Fan, Rong-En, Chang, Kai-Wei, Hsieh, Cho-Jui, Wang, Xiang-Rui, and Lin, Chih-Jen. 2008. LIBLINEAR: A Library for Large Linear Classification. Journal of Machine Learning Research, 9, 1871–1874. 





Gal, Yarin, van der Wilk, Mark, and Rasmussen, Carl E. 2014. Distributed Variational Inference in Sparse Gaussian Process Regression and Latent Variable Models. In: Advances in Neural Information Processing Systems. 





Gartner, Thomas. 2008.¨ Kernels for Structured Data. World Scientific. 





Gavish, Matan, and Donoho, David L. 2014. The Optimal Hard Threshold for Singular Values is 4 3. IEEE Transactions on Information Theory, 60(8), 5040–5053. 





Gelman, Andrew, Carlin, John B., Stern, Hal S., and Rubin, Donald B. 2004. Bayesian Data Analysis. Chapman and Hall/CRC. 





Gentle, James E. 2004. Random Number Generation and Monte Carlo Methods. Springer. 





Ghahramani, Zoubin. 2015. Probabilistic Machine Learning and Artificial Intelligence. Nature, 521, 452–459. 





Ghahramani, Zoubin, and Roweis, Sam T. 1999. Learning Nonlinear Dynamical Systems Using an EM Algorithm. In: Advances in Neural Information Processing Systems. MIT Press. 





Gilks, Walter R., Richardson, Sylvia, and Spiegelhalter, David J. 1996. Markov Chain Monte Carlo in Practice. Chapman and Hall/CRC. 





Gneiting, Tilmann, and Raftery, Adrian E. 2007. Strictly Proper Scoring Rules, Prediction, and Estimation. Journal of the American Statistical Association, 102(477), 359–378. 





Goh, Gabriel. 2017. Why Momentum Really Works. Distill. 





Gohberg, Israel, Goldberg, Seymour, and Krupnik, Nahum. 2012. Traces and Determi nants of Linear Operators. Birkhauser.¨ 





Golan, Jonathan S. 2007. The Linear Algebra a Beginning Graduate Student Ought to Know. Springer. 





Golub, Gene H., and Van Loan, Charles F. 2012. Matrix Computations. JHU Press. 





Goodfellow, Ian, Bengio, Yoshua, and Courville, Aaron. 2016. Deep Learning. MIT Press. 





Graepel, Thore, Candela, Joaquin Quinonero-Candela, Borchert, Thomas, and Her-˜ brich, Ralf. 2010. Web-Scale Bayesian Click-through Rate Prediction for Sponsored Search Advertising in Microsoft’s Bing Search Engine. In: Proceedings of the Interna tional Conference on Machine Learning. 





Griewank, Andreas, and Walther, Andrea. 2003. Introduction to Automatic Differenti ation. In: Proceedings in Applied Mathematics and Mechanics. 





Griewank, Andreas, and Walther, Andrea. 2008. Evaluating Derivatives, Principles and Techniques of Algorithmic Differentiation. SIAM. 





Grimmett, Geoffrey R., and Welsh, Dominic. 2014. Probability: An Introduction. Oxford University Press. 





Grinstead, Charles M., and Snell, J. Laurie. 1997. Introduction to Probability. American Mathematical Society. 





Hacking, Ian. 2001. Probability and Inductive Logic. Cambridge University Press. 





Hall, Peter. 1992. The Bootstrap and Edgeworth Expansion. Springer. 





Hallin, Marc, Paindaveine, Davy, and Siman, Miroslav. 2010. Multivariate Quan-<sup>ˇ</sup> tiles and Multiple-Output Regression Quantiles: From ℓ<sub>1</sub> Optimization to Halfspace Depth. Annals of Statistics, 38, 635–669. 





Hasselblatt, Boris, and Katok, Anatole. 2003. A First Course in Dynamics with a Panorama of Recent Developments. Cambridge University Press. 





Hastie, Trevor, Tibshirani, Robert, and Friedman, Jerome. 2001. The Elements of Sta tistical Learning – Data Mining, Inference, and Prediction. Springer. 





Hausman, Karol, Springenberg, Jost T., Wang, Ziyu, Heess, Nicolas, and Riedmiller, Martin. 2018. Learning an Embedding Space for Transferable Robot Skills. In: Proceedings of the International Conference on Learning Representations. 





Hazan, Elad. 2015. Introduction to Online Convex Optimization. Foundations and Trends in Optimization, 2(3–4), 157–325. 





Hensman, James, Fusi, Nicolo, and Lawrence, Neil D. 2013. Gaussian Processes for` Big Data. In: Proceedings of the Conference on Uncertainty in Artificial Intelligence. 





Herbrich, Ralf, Minka, Tom, and Graepel, Thore. 2007. TrueSkill(TM): A Bayesian Skill Rating System. In: Advances in Neural Information Processing Systems. 





Hiriart-Urruty, Jean-Baptiste, and Lemar´echal, Claude. 2001. Fundamentals of Convex Analysis. Springer. 





Hoffman, Matthew D., Blei, David M., and Bach, Francis. 2010. Online Learning for Latent Dirichlet Allocation. Advances in Neural Information Processing Systems. 





Hoffman, Matthew D., Blei, David M., Wang, Chong, and Paisley, John. 2013. Stochas tic Variational Inference. Journal of Machine Learning Research, 14(1), 1303–1347. 





Hofmann, Thomas, Scholkopf, Bernhard, and Smola, Alexander J. 2008. Kernel Meth-¨ ods in Machine Learning. Annals of Statistics, 36(3), 1171–1220. 





Hogben, Leslie. 2013. Handbook of Linear Algebra. Chapman and Hall/CRC. 





Horn, Roger A., and Johnson, Charles R. 2013. Matrix Analysis. Cambridge University Press. 





Hotelling, Harold. 1933. Analysis of a Complex of Statistical Variables into Principal Components. Journal of Educational Psychology, 24, 417–441. 





Hyvarinen, Aapo, Oja, Erkki, and Karhunen, Juha. 2001. Independent Component Anal ysis. Wiley. 





Imbens, Guido W., and Rubin, Donald B. 2015. Causal Inference for Statistics, Social and Biomedical Sciences. Cambridge University Press. 





Jacod, Jean, and Protter, Philip. 2004. Probability Essentials. Springer. 





Jaynes, Edwin T. 2003. Probability Theory: The Logic of Science. Cambridge University Press. 





Jefferys, William H., and Berger, James O. 1992. Ockham’s Razor and Bayesian Analysis. American Scientist, 80, 64–72. 





Jeffreys, Harold. 1961. Theory of Probability. Oxford University Press. 





Jimenez Rezende, Danilo, and Mohamed, Shakir. 2015. Variational Inference with Normalizing Flows. In: Proceedings of the International Conference on Machine Learning. 





Jimenez Rezende, Danilo, Mohamed, Shakir, and Wierstra, Daan. 2014. Stochastic Backpropagation and Approximate Inference in Deep Generative Models. In: Pro ceedings of the International Conference on Machine Learning. 





Joachims, Thorsten. 1999. Advances in Kernel Methods – Support Vector Learning. MIT Press. Chap. Making Large-Scale SVM Learning Practical, pages 169–184. 





Jordan, Michael I., Ghahramani, Zoubin, Jaakkola, Tommi S., and Saul, Lawrence K. 1999. An Introduction to Variational Methods for Graphical Models. Machine Learning, 37, 183–233. 





©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 





Julier, Simon J., and Uhlmann, Jeffrey K. 1997. A New Extension of the Kalman Filter to Nonlinear Systems. In: Proceedings of AeroSense Symposium on Aerospace/Defense Sensing, Simulation and Controls. 





Kaiser, Marcus, and Hilgetag, Claus C. 2006. Nonoptimal Component Placement, but Short Processing Paths, Due to Long-Distance Projections in Neural Systems. PLoS Computational Biology, 2(7), e95. 





Kalman, Dan. 1996. A Singularly Valuable Decomposition: The SVD of a Matrix. Col lege Mathematics Journal, 27(1), 2–23. 





Kalman, Rudolf E. 1960. A New Approach to Linear Filtering and Prediction Problems. Transactions of the ASME – Journal of Basic Engineering, 82(Series D), 35–45. 





Kamthe, Sanket, and Deisenroth, Marc P. 2018. Data-Efficient Reinforcement Learning with Probabilistic Model Predictive Control. In: Proceedings of the International Conference on Artificial Intelligence and Statistics. 





Katz, Victor J. 2004. A History of Mathematics. Pearson/Addison-Wesley. 





Kelley, Henry J. 1960. Gradient Theory of Optimal Flight Paths. Ars Journal, 30(10), 947–954. 





Kimeldorf, George S., and Wahba, Grace. 1970. A Correspondence between Bayesian Estimation on Stochastic Processes and Smoothing by Splines. Annals of Mathemat ical Statistics, 41(2), 495–502. 





Kingma, Diederik P., and Welling, Max. 2014. Auto-Encoding Variational Bayes. In: Proceedings of the International Conference on Learning Representations. 





Kittler, Josef, and Foglein, Janos. 1984. Contextual Classification of Multispectral Pixel¨ Data. Image and Vision Computing, 2(1), 13–29. 





Kolda, Tamara G., and Bader, Brett W. 2009. Tensor Decompositions and Applications. SIAM Review, 51(3), 455–500. 





Koller, Daphne, and Friedman, Nir. 2009. Probabilistic Graphical Models. MIT Press. 





Kong, Linglong, and Mizera, Ivan. 2012. Quantile Tomography: Using Quantiles with Multivariate Data. Statistica Sinica, 22, 1598–1610. 





Lang, Serge. 1987. Linear Algebra. Springer. 





Lawrence, Neil D. 2005. Probabilistic Non-Linear Principal Component Analysis with Gaussian Process Latent Variable Models. Journal of Machine Learning Research, 6(Nov.), 1783–1816. 





Leemis, Lawrence M., and McQueston, Jacquelyn T. 2008. Univariate Distribution Relationships. American Statistician, 62(1), 45–53. 





Lehmann, Erich L., and Romano, Joseph P. 2005. Testing Statistical Hypotheses. Springer. 





Lehmann, Erich Leo, and Casella, George. 1998. Theory of Point Estimation. Springer. Liesen, Jorg, and Mehrmann, Volker. 2015.¨ Linear Algebra. Springer. 





Lin, Hsuan-Tien, Lin, Chih-Jen, and Weng, Ruby C. 2007. A Note on Platt’s Probabilistic Outputs for Support Vector Machines. Machine Learning, 68, 267–276. 





Ljung, Lennart. 1999. System Identification: Theory for the User. Prentice Hall. 





Loosli, Ga¨elle, Canu, St´ephane, and Ong, Cheng Soon. 2016. Learning SVM in Kre˘ın Spaces. IEEE Transactions of Pattern Analysis and Machine Intelligence, 38(6), 1204– 1216. 





Luenberger, David G. 1969. Optimization by Vector Space Methods. Wiley. 





MacKay, David J. C. 1992. Bayesian Interpolation. Neural Computation, 4, 415–447. 





MacKay, David J. C. 1998. Introduction to Gaussian Processes. Pages 133–165 of: Bishop, C. M. (ed), Neural Networks and Machine Learning. Springer. 





MacKay, David J. C. 2003. Information Theory, Inference, and Learning Algorithms. Cambridge University Press. 





Magnus, Jan R., and Neudecker, Heinz. 2007. Matrix Differential Calculus with Appli cations in Statistics and Econometrics. Wiley. 





Manton, Jonathan H., and Amblard, Pierre-Olivier. 2015. A Primer on Reproducing Kernel Hilbert Spaces. Foundations and Trends in Signal Processing, 8(1–2), 1–126. 





Markovsky, Ivan. 2011. Low Rank Approximation: Algorithms, Implementation, Applications. Springer. 





Maybeck, Peter S. 1979. Stochastic Models, Estimation, and Control. Academic Press. McCullagh, Peter, and Nelder, John A. 1989. Generalized Linear Models. CRC Press. 





McEliece, Robert J., MacKay, David J. C., and Cheng, Jung-Fu. 1998. Turbo Decoding as an Instance of Pearl’s “Belief Propagation” Algorithm. IEEE Journal on Selected Areas in Communications, 16(2), 140–152. 





Mika, Sebastian, Ratsch, Gunnar, Weston, Jason, Sch¨ olkopf, Bernhard, and M¨ uller,¨ Klaus-Robert. 1999. Fisher Discriminant Analysis with Kernels. Pages 41–48 of: Proceedings of the Workshop on Neural Networks for Signal Processing. 





Minka, Thomas P. 2001a. A Family of Algorithms for Approximate Bayesian Inference. Ph.D. thesis, Massachusetts Institute of Technology. 





Minka, Tom. 2001b. Automatic Choice of Dimensionality of PCA. In: Advances in Neural Information Processing Systems. 





Mitchell, Tom. 1997. Machine Learning. McGraw-Hill. 





Mnih, Volodymyr, Kavukcuoglu, Koray, and Silver, David, et al. 2015. Human-Level Control through Deep Reinforcement Learning. Nature, 518, 529–533. 





Moonen, Marc, and De Moor, Bart. 1995. SVD and Signal Processing, III: Algorithms, Architectures and Applications. Elsevier. 





Moustaki, Irini, Knott, Martin, and Bartholomew, David J. 2015. Latent-Variable Modeling. American Cancer Society. Pages 1–10. 





Muller, Andreas C., and Guido, Sarah. 2016.¨ Introduction to Machine Learning with Python: A Guide for Data Scientists. O’Reilly Publishing. 





Murphy, Kevin P. 2012. Machine Learning: A Probabilistic Perspective. MIT Press. 





Neal, Radford M. 1996. Bayesian Learning for Neural Networks. Ph.D. thesis, Department of Computer Science, University of Toronto. 





Neal, Radford M., and Hinton, Geoffrey E. 1999. A View of the EM Algorithm that Justifies Incremental, Sparse, and Other Variants. Pages 355–368 of: Learning in Graphical Models. MIT Press. 





Nelsen, Roger. 2006. An Introduction to Copulas. Springer. 





Nesterov, Yuri. 2018. Lectures on Convex Optimization. Springer. 





Neumaier, Arnold. 1998. Solving Ill-Conditioned and Singular Linear Systems: A Tutorial on Regularization. SIAM Review, 40, 636–666. 





Nocedal, Jorge, and Wright, Stephen J. 2006. Numerical Optimization. Springer. 





Nowozin, Sebastian, Gehler, Peter V., Jancsary, Jeremy, and Lampert, Christoph H. (eds). 2014. Advanced Structured Prediction. MIT Press. 





O’Hagan, Anthony. 1991. Bayes-Hermite Quadrature. Journal of Statistical Planning and Inference, 29, 245–260. 





Ong, Cheng Soon, Mary, Xavier, Canu, St´ephane, and Smola, Alexander J. 2004. Learning with Non-Positive Kernels. In: Proceedings of the International Conference on Machine Learning. 





Ormoneit, Dirk, Sidenbladh, Hedvig, Black, Michael J., and Hastie, Trevor. 2001. Learning and Tracking Cyclic Human Motion. In: Advances in Neural Information Processing Systems. 





Page, Lawrence, Brin, Sergey, Motwani, Rajeev, and Winograd, Terry. 1999. The PageRank Citation Ranking: Bringing Order to the Web. Tech. rept. Stanford Info-Lab. 





Paquet, Ulrich. 2008. Bayesian Inference for Latent Variable Models. Ph.D. thesis, University of Cambridge. 





Parzen, Emanuel. 1962. On Estimation of a Probability Density Function and Mode. Annals of Mathematical Statistics, 33(3), 1065–1076. 





©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 





Pearl, Judea. 1988. Probabilistic Reasoning in Intelligent Systems: Networks of Plausible Inference. Morgan Kaufmann. 





Pearl, Judea. 2009. Causality: Models, Reasoning and Inference. 2nd edn. Cambridge University Press. 





Pearson, Karl. 1895. Contributions to the Mathematical Theory of Evolution. II. Skew Variation in Homogeneous Material. Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences, 186, 343–414. 





Pearson, Karl. 1901. On Lines and Planes of Closest Fit to Systems of Points in Space. Philosophical Magazine, 2(11), 559–572. 





Peters, Jonas, Janzing, Dominik, and Scholkopf, Bernhard. 2017.¨ Elements of Causal Inference: Foundations and Learning Algorithms. MIT Press. 





Petersen, Kaare B., and Pedersen, Michael S. 2012. The Matrix Cookbook. Tech. rept. Technical University of Denmark. 





Platt, John C. 2000. Probabilistic Outputs for Support Vector Machines and Compar isons to Regularized Likelihood Methods. In: Advances in Large Margin Classifiers. 





Pollard, David. 2002. A User’s Guide to Measure Theoretic Probability. Cambridge University Press. 





Polyak, Roman A. 2016. The Legendre Transformation in Modern Optimization. Pages 437–507 of: Goldengorin, B. (ed), Optimization and Its Applications in Control and Data Sciences. Springer. 





Press, William H., Teukolsky, Saul A., Vetterling, William T., and Flannery, Brian P. 2007. Numerical Recipes: The Art of Scientific Computing. Cambridge University Press. 





Proschan, Michael A., and Presnell, Brett. 1998. Expect the Unexpected from Condi tional Expectation. American Statistician, 52(3), 248–252. 





Raschka, Sebastian, and Mirjalili, Vahid. 2017. Python Machine Learning: Machine Learning and Deep Learning with Python, scikit-learn, and TensorFlow. Packt Publish ing. 





Rasmussen, Carl E., and Ghahramani, Zoubin. 2001. Occam’s Razor. In: Advances in Neural Information Processing Systems. 





Rasmussen, Carl E., and Ghahramani, Zoubin. 2003. Bayesian Monte Carlo. In: Ad vances in Neural Information Processing Systems. 





Rasmussen, Carl E., and Williams, Christopher K. I. 2006. Gaussian Processes for Ma chine Learning. MIT Press. 





Reid, Mark, and Williamson, Robert C. 2011. Information, Divergence and Risk for Binary Experiments. Journal of Machine Learning Research, 12, 731–817. 





Rifkin, Ryan M., and Lippert, Ross A. 2007. Value Regularization and Fenchel Duality. Journal of Machine Learning Research, 8, 441–479. 





Rockafellar, Ralph T. 1970. Convex Analysis. Princeton University Press. 





Rogers, Simon, and Girolami, Mark. 2016. A First Course in Machine Learning. Chap man and Hall/CRC. 





Rosenbaum, Paul R. 2017. Observation and Experiment: An Introduction to Causal Inference. Harvard University Press. 





Rosenblatt, Murray. 1956. Remarks on Some Nonparametric Estimates of a Density Function. Annals of Mathematical Statistics, 27(3), 832–837. 





Roweis, Sam T. 1998. EM Algorithms for PCA and SPCA. Pages 626–632 of: Advances in Neural Information Processing Systems. 





Roweis, Sam T., and Ghahramani, Zoubin. 1999. A Unifying Review of Linear Gaussian Models. Neural Computation, 11(2), 305–345. 





Roy, Anindya, and Banerjee, Sudipto. 2014. Linear Algebra and Matrix Analysis for Statistics. Chapman and Hall/CRC. 





Rubinstein, Reuven Y., and Kroese, Dirk P. 2016. Simulation and the Monte Carlo Method. Wiley. 





Ruffini, Paolo. 1799. Teoria Generale delle Equazioni, in cui si Dimostra Impossibile la Soluzione Algebraica delle Equazioni Generali di Grado Superiore al Quarto. Stamperia di S. Tommaso d’Aquino. 





Rumelhart, David E., Hinton, Geoffrey E., and Williams, Ronald J. 1986. Learning Representations by Back-Propagating Errors. Nature, 323(6088), 533–536. 





Sæmundsson, Steindor, Hofmann, Katja, and Deisenroth, Marc P. 2018. Meta Rein-´ forcement Learning with Latent Variable Gaussian Processes. In: Proceedings of the Conference on Uncertainty in Artificial Intelligence. 





Saitoh, Saburou. 1988. Theory of Reproducing Kernels and its Applications. Longman Scientific and Technical. 





Sarkk ¨ a, Simo. 2013. ¨ Bayesian Filtering and Smoothing. Cambridge University Press. 





Scholkopf, Bernhard, and Smola, Alexander J. 2002.¨ Learning with Kernels – Support Vector Machines, Regularization, Optimization, and Beyond. MIT Press. 





Scholkopf, Bernhard, Smola, Alexander J., and M¨ uller, Klaus-Robert. 1997. Kernel¨ Principal Component Analysis. In: Proceedings of the International Conference on Artificial Neural Networks. 





Scholkopf, Bernhard, Smola, Alexander J., and M¨ uller, Klaus-Robert. 1998. Nonlinear¨ Component Analysis as a Kernel Eigenvalue Problem. Neural Computation, 10(5), 1299–1319. 





Scholkopf, Bernhard, Herbrich, Ralf, and Smola, Alexander J. 2001. A Generalized¨ Representer Theorem. In: Proceedings of the International Conference on Computational Learning Theory. 





Schwartz, Laurent. 1964. Sous Espaces Hilbertiens d’Espaces Vectoriels Topologiques et Noyaux Associ´es. Journal d’Analyse Math´ematique, 13, 115–256. 





Schwarz, Gideon E. 1978. Estimating the Dimension of a Model. Annals of Statistics, 6(2), 461–464. 





Shahriari, Bobak, Swersky, Kevin, Wang, Ziyu, Adams, Ryan P., and De Freitas, Nando. 2016. Taking the Human out of the Loop: A Review of Bayesian Optimization. Proceedings of the IEEE, 104(1), 148–175. 





Shalev-Shwartz, Shai, and Ben-David, Shai. 2014. Understanding Machine Learning: From Theory to Algorithms. Cambridge University Press. 





Shawe-Taylor, John, and Cristianini, Nello. 2004. Kernel Methods for Pattern Analysis. Cambridge University Press. 





Shawe-Taylor, John, and Sun, Shiliang. 2011. A Review of Optimization Methodologies in Support Vector Machines. Neurocomputing, 74(17), 3609–3618. 





Shental, Ori, Siegel, Paul H., Wolf, Jack K., Bickson, Danny, and Dolev, Danny. 2008. Gaussian Belief Propagation Solver for Systems of Linear Equations. Pages 1863– 1867 of: Proceedings of the International Symposium on Information Theory. 





Shewchuk, Jonathan R. 1994. An Introduction to the Conjugate Gradient Method without the Agonizing Pain. 





Shi, Jianbo, and Malik, Jitendra. 2000. Normalized Cuts and Image Segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 22(8), 888–905. 





Shi, Qinfeng, Petterson, James, Dror, Gideon, Langford, John, Smola, Alexander J., and Vishwanathan, S. V. N. 2009. Hash Kernels for Structured Data. Journal of Machine Learning Research, 2615–2637. 





Shiryayev, Albert N. 1984. Probability. Springer. 





Shor, Naum Z. 1985. Minimization Methods for Non-Differentiable Functions. Springer. 





Shotton, Jamie, Winn, John, Rother, Carsten, and Criminisi, Antonio. 2006. Texton-Boost: Joint Appearance, Shape and Context Modeling for Multi-Class Object Recognition and Segmentation. In: Proceedings of the European Conference on Computer Vision. 





Smith, Adrian F. M., and Spiegelhalter, David. 1980. Bayes Factors and Choice Criteria for Linear Models. Journal of the Royal Statistical Society B, 42(2), 213–220. 





©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 





Snoek, Jasper, Larochelle, Hugo, and Adams, Ryan P. 2012. Practical Bayesian Optimization of Machine Learning Algorithms. In: Advances in Neural Information Processing Systems. 





Spearman, Charles. 1904. “General Intelligence,” Objectively Determined and Measured. American Journal of Psychology, 15(2), 201–292. 





Sriperumbudur, Bharath K., Gretton, Arthur, Fukumizu, Kenji, Scholkopf, Bernhard,¨ and Lanckriet, Gert R. G. 2010. Hilbert Space Embeddings and Metrics on Proba bility Measures. Journal of Machine Learning Research, 11, 1517–1561. 





Steinwart, Ingo. 2007. How to Compare Different Loss Functions and Their Risks. Constructive Approximation, 26, 225–287. 





Steinwart, Ingo, and Christmann, Andreas. 2008. Support Vector Machines. Springer. 





Stoer, Josef, and Burlirsch, Roland. 2002. Introduction to Numerical Analysis. Springer. 





Strang, Gilbert. 1993. The Fundamental Theorem of Linear Algebra. The American Mathematical Monthly, 100(9), 848–855. 





Strang, Gilbert. 2003. Introduction to Linear Algebra. Wellesley-Cambridge Press. 





Stray, Jonathan. 2016. The Curious Journalist’s Guide to Data. Tow Center for Digital Journalism at Columbia’s Graduate School of Journalism. 





Strogatz, Steven. 2014. Writing about Math for the Perplexed and the Traumatized. Notices of the American Mathematical Society, 61(3), 286–291. 





Sucar, Luis E., and Gillies, Duncan F. 1994. Probabilistic Reasoning in High-Level Vision. Image and Vision Computing, 12(1), 42–60. 





Szeliski, Richard, Zabih, Ramin, and Scharstein, Daniel, et al. 2008. A Comparative Study of Energy Minimization Methods for Markov Random Fields with Smoothness-Based Priors. IEEE Transactions on Pattern Analysis and Machine In telligence, 30(6), 1068–1080. 





Tandra, Haryono. 2014. The Relationship between the Change of Variable Theorem and the Fundamental Theorem of Calculus for the Lebesgue Integral. Teaching of Mathematics, 17(2), 76–83. 





Tenenbaum, Joshua B., De Silva, Vin, and Langford, John C. 2000. A Global Geometric Framework for Nonlinear Dimensionality Reduction. Science, 290(5500), 2319– 2323. 





Tibshirani, Robert. 1996. Regression Selection and Shrinkage via the Lasso. Journal of the Royal Statistical Society B, 58(1), 267–288. 





Tipping, Michael E., and Bishop, Christopher M. 1999. Probabilistic Principal Compo nent Analysis. Journal of the Royal Statistical Society: Series B, 61(3), 611–622. 





Titsias, Michalis K., and Lawrence, Neil D. 2010. Bayesian Gaussian Process Latent Variable Model. In: Proceedings of the International Conference on Artificial Intelligence and Statistics. 





Toussaint, Marc. 2012. Some Notes on Gradient Descent. https://ipvs.informatik.uni stuttgart.de/mlr/marc/notes/gradientDescent.pdf. 





Trefethen, Lloyd N., and Bau III, David. 1997. Numerical Linear Algebra. SIAM. 





Tucker, Ledyard R. 1966. Some Mathematical Notes on Three-Mode Factor Analysis. Psychometrika, 31(3), 279–311. 





Vapnik, Vladimir N. 1998. Statistical Learning Theory. Wiley. 





Vapnik, Vladimir N. 1999. An Overview of Statistical Learning Theory. IEEE Transactions on Neural Networks, 10(5), 988–999. 





Vapnik, Vladimir N. 2000. The Nature of Statistical Learning Theory. Springer. 





Vishwanathan, S. V. N., Schraudolph, Nicol N., Kondor, Risi, and Borgwardt, Karsten M. 2010. Graph Kernels. Journal of Machine Learning Research, 11, 1201– 1242. 





von Luxburg, Ulrike, and Scholkopf, Bernhard. 2011. Statistical Learning Theory:¨ Models, Concepts, and Results. Pages 651–706 of: D. M. Gabbay, S. Hartmann, J. Woods (ed), Handbook of the History of Logic, vol. 10. Elsevier. 





Wahba, Grace. 1990. Spline Models for Observational Data. Society for Industrial and Applied Mathematics. 





Walpole, Ronald E., Myers, Raymond H., Myers, Sharon L., and Ye, Keying. 2011. Probability and Statistics for Engineers and Scientists. Prentice Hall. 





Wasserman, Larry. 2004. All of Statistics. Springer. 





Wasserman, Larry. 2007. All of Nonparametric Statistics. Springer. 





Whittle, Peter. 2000. Probability via Expectation. Springer. 





Wickham, Hadley. 2014. Tidy Data. Journal of Statistical Software, 59, 1–23. 





Williams, Christopher K. I. 1997. Computing with Infinite Networks. In: Advances in Neural Information Processing Systems. 





Yu, Yaoliang, Cheng, Hao, Schuurmans, Dale, and Szepesvari, Csaba. 2013. Charac-´ terizing the Representer Theorem. In: Proceedings of the International Conference on Machine Learning. 





Zadrozny, Bianca, and Elkan, Charles. 2001. Obtaining Calibrated Probability Estimates from Decision Trees and Naive Bayesian Classifiers. In: Proceedings of the International Conference on Machine Learning. 





Zhang, Haizhang, Xu, Yuesheng, and Zhang, Jun. 2009. Reproducing Kernel Banach Spaces for Machine Learning. Journal of Machine Learning Research, 10, 2741–2775. 





Zia, Royce K. P., Redish, Edward F., and McKay, Susan R. 2009. Making Sense of the Legendre Transform. American Journal of Physics, 77(614), 614–622. 



## Index

1-of-K representation, 364 ℓ norm, 71 ℓ norm, 72 abduction, 258 Abel-Ruffini theorem, 334 Abelian group, 36 absolutely homogeneous, 71 activation function, 315 affine mapping, 63 affine subspace, 61 Akaike information criterion, 288 algebra, 17 algebraic multiplicity, 106 analytic, 143 ancestral sampling, 340, 364 angle, 76 associativity, 24, 26, 36 attribute, 253 augmented matrix, 29 auto-encoder, 343 automatic differentiation, 161 automorphism, 49 backpropagation, 159 basic variable, 30 basis, 44 basis vector, 45 Bayes factor, 287 Bayes’ law, 185 Bayes’ rule, 185 Bayes’ theorem, 185 Bayesian GP-LVM, 347 Bayesian inference, 274 Bayesian information criterion, 288 Bayesian linear regression, 303 Bayesian model selection, 286 Bayesian network, 278, 283 Bayesian PCA, 346 Bernoulli distribution, 205 Beta distribution, 206 bilinear mapping, 72 bijective, 48 binary classification, 370 Binomial distribution, 206 blind-source separation, 346 Borel σ-algebra, 180 

canonical basis, 45 canonical feature map, 389 canonical link function, 315 categorical variable, 180 Cauchy-Schwarz inequality, 75 change-of-variable technique, 219 characteristic polynomial, 104 Cholesky decomposition, 114 Cholesky factor, 114 Cholesky factorization, 114 class, 370 classification, 315 closure, 36 code, 343 codirected, 105 codomain, 58, 139 collinear, 105 column, 22 column space, 59 column vector, 22, 38 completing the squares, 307 concave function, 236 condition number, 230 conditional probability, 179 conditionally independent, 195 conjugate, 208 conjugate prior, 208 convex conjugate, 242 convex function, 236 convex hull, 386 convex optimization problem, 236, 239 convex set, 236 coordinate, 50 coordinate representation, 50 coordinate vector, 50 correlation, 191 covariance, 190 covariance matrix, 190, 198 covariate, 253 CP decomposition, 136 cross-covariance, 191 cross-validation, 258, 263 cumulative distribution function, 178, 181 d-separation, 281 

## 407

This material is published by Cambridge University Press as Mathematics for Machine Learning by Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong (2020). This version is free to view and download for personal use only. Not for re-distribution, re-sale, or use in derivative works. ©by M. P. Deisenroth, A. A. Faisal, and C. S. Ong, 2024. https://mml-book.com. 

Draft (2024-01-15) of “Mathematics for Machine Learning”. Feedback: https://mml-book.com. 

data covariance matrix, 318 data point, 253 data-fit term, 302 decoder, 343 deep auto-encoder, 347 defective, 111 denominator layout, 151 derivative, 141 design matrix, 294, 296 determinant, 99 diagonal matrix, 115 diagonalizable, 116 diagonalization, 116 difference quotient, 141 dimension, 45 dimensionality reduction, 317 directed graphical model, 278, 283 direction, 61 direction space, 61 distance, 75 distribution, 177 distributivity, 24, 26 domain, 58, 139 dot product, 72 dual SVM, 385 Eckart-Young theorem, 131, 334 eigendecomposition, 116 eigenspace, 106 eigenspectrum, 106 eigenvalue, 105 eigenvalue equation, 105 eigenvector, 105 elementary transformations, 28 EM algorithm, 360 embarrassingly parallel, 264 empirical covariance, 192 empirical mean, 192 empirical risk, 260 empirical risk minimization, 257, 260 encoder, 343 endomorphism, 49 epigraph, 236 equivalent, 56 error function, 294 error term, 382 Euclidean distance, 72, 75 Euclidean norm, 72 Euclidean vector space, 73 event space, 175 evidence, 186, 285, 306 example, 253 expected risk, 261 expected value, 187 exponential family, 205, 211 extended Kalman filter, 170 factor analysis, 346 factor graph, 283 

feature, 253 feature map, 254 feature matrix, 296 feature vector, 295 Fisher discriminant analysis, 136 Fisher-Neyman theorem, 210 forward mode, 161 free variable, 30 full rank, 47 full SVD, 128 fundamental theorem of linear mappings, 60 Gaussian elimination, 31 Gaussian mixture model, 349 Gaussian process, 316 Gaussian process latent-variable model, 347 general linear group, 37 general solution, 28, 30 generalized linear model, 272, 315 generating set, 44 generative process, 272, 286 generator, 344 geometric multiplicity, 108 Givens rotation, 94 global minimum, 225 GP-LVM, 347 gradient, 146 Gram matrix, 389 Gram-Schmidt orthogonalization, 89 graphical model, 278 group, 36 Hadamard product, 23 hard margin SVM, 377 Hessian, 164 Hessian eigenmaps, 136 Hessian matrix, 165 hinge loss, 381 histogram, 369 hyperparameter, 258 hyperplane, 61, 62 hyperprior, 281 i.i.d., 195 ICA, 346 identity automorphism, 49 identity mapping, 49 identity matrix, 23 image, 58, 139 independent and identically distributed, 195, 260, 266 independent component analysis, 346 inference network, 344 injective, 48 inner product, 73 inner product space, 73 intermediate variables, 162 inverse. 24 

## Index

inverse element, 36 invertible, 24 Isomap, 136 isomorphism, 49 Jacobian, 146, 150 Jacobian determinant, 152 Jeffreys-Lindley paradox, 287 Jensen’s inequality, 239 joint probability, 178 Karhunen-Lo`eve transform, 318 kernel, 33, 47, 58, 254, 388 kernel density estimation, 369 kernel matrix, 389 kernel PCA, 347 kernel trick, 316, 347, 389 label, 253 Lagrange multiplier, 234 Lagrangian, 234 Lagrangian dual problem, 234 Laplace approximation, 170 Laplace expansion, 102 Laplacian eigenmaps, 136 LASSO, 303, 316 latent variable, 275 law, 177, 181 law of total variance, 203 leading coefficient, 30 least-squares loss, 154 least-squares problem, 261 least-squares solution, 88 left-singular vectors, 119 Legendre transform, 242 Legendre-Fenchel transform, 242 length, 71 likelihood, 185, 265, 269, 291 line, 61, 82 linear combination, 40 linear manifold, 61 linear mapping, 48 linear program, 239 linear subspace, 39 linear transformation, 48 linearly dependent, 40 linearly independent, 40 link function, 272 loading, 322 local minimum. 225 log-partition function, 211 logistic regression, 315 logistic sigmoid, 315 loss function, 260, 381 loss term, 382 lower-triangular matrix, 101 Maclaurin series, 143 Manhattan norm, 71 MAP, 300 MAP estimation, 269 

margin, 374 marginal, 190 marginal likelihood, 186, 286, 306 marginal probability, 179 marginalization property, 184 Markov random field, 283 matrix, 22 matrix factorization, 98 maximum a posteriori, 300 maximum a posteriori estimation, 269 maximum likelihood, 257 maximum likelihood estimate, 296 maximum likelihood estimation, 265, 293 mean, 187 mean function, 309 mean vector, 198 measure, 180 median, 188 metric, 76 minimal, 44 minimax inequality, 234 misfit term, 302 mixture model, 349 mixture weight, 349 mode, 188 model, 251 model evidence, 286 model selection, 258 Moore-Penrose pseudo-inverse, 35 multidimensional scaling, 136 multiplication by scalars, 37 multivariate, 178 multivariate Gaussian distribution, 198 multivariate Taylor series, 166 natural parameters, 212 negative log-likelihood, 265 nested cross-validation, 258, 284 neutral element 36 noninvertible, 24 nonsingular, 24 norm, 71 normal distribution, 197 normal equation, 86 normal vector, 80 null space, 33, 47, 58 numerator layout, 150 Occam’s razor, 285 ONB, 79 one-hot encoding, 364 ordered basis, 50 orthogonal, 77 orthogonal basis, 79 orthogonal complement, 79 orthogonal matrix, 78 orthonormal, 77 orthonormal basis, 79 

©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020). 

outer product, 38 overfitting, 262, 271, 299 PageRank, 114 parameters, 61 parametric equation, 61 partial derivative, 146 particular solution, 27, 30 PCA, 317 pdf, 181 penalty term, 263 pivot, 30 plane, 62 plate, 281 population mean and covariance, 191 positive definite, 71, 73, 74, 76 posterior, 185, 269 posterior odds, 287 power iteration, 334 power series representation, 145 PPCA, 340 preconditioner, 230 predictor, 12, 255 primal problem, 234 principal component, 322 principal component analysis, 136, 317 principal subspace, 327 prior, 185, 269 prior odds, 287 probabilistic inverse, 186 probabilistic PCA, 340 probabilistic programming, 278 probability, 175 probability density function, 181 probability distribution, 172 probability integral transform, 217 probability mass function, 178 product rule, 184 projection, 82 projection error, 88 projection matrix, 82 pseudo-inverse, 86 random variable, 172, 175 range, 58 rank, 47 rank deficient, 47 rank-k approximation, 130 rank-nullity theorem, 60 raw-score formula for variance, 193 recognition network, 344 reconstruction error, 88, 327 reduced hull, 388 reduced row-echelon form, 31 reduced SVD, 129 REF, 30 regression, 289 regular, 24 regularization, 262, 302, 382 

regularization parameter, 263, 302, 380 regularized least squares, 302 regularizer, 263, 302, 380, 382 representer theorem, 384 responsibility, 352 reverse mode, 161 right-singular vectors, 119 RMSE, 298 root mean square error, 298 rotation, 91 rotation matrix, 92 row, 22 row vector, 22, 38 row-echelon form, 30 sample mean, 192 sample space, 175 scalar, 37 scalar product, 72 sigmoid, 213 similar, 56 singular, 24 singular value decomposition, 119 singular value equation, 124 singular value matrix, 119 singular values, 119 slack variable, 379 soft margin SVM, 379, 380 solution, 20 span, 44 special solution, 27 spectral clustering, 136 spectral norm, 131 spectral theorem, 111 spectrum, 106 square matrix, 25 standard basis, 45 standard deviation, 190 standard normal distribution, 198 standardization, 336 statistical independence, 194 statistical learning theory, 265 stochastic gradient descent, 231 strong duality, 236 sufficient statistics, 210 sum rule, 184 support point, 61 support vector, 384 supporting hyperplane, 242 surjective, 48 SVD, 119 SVD theorem, 119 symmetric, 73, 76 symmetric matrix, 25 symmetric, positive definite, 74 symmetric, positive semidefinite, 74 system of linear equations, 20 target space, 175 

## Index

Taylor polynomial, 142, 166 Taylor series, 142 test error, 300 test set, 262, 284 Tikhonov regularization, 265 trace, 103 training, 12 training error, 300 training set, 260, 292 transfer function, 315 transformation matrix, 51 translation vector, 63 transpose, 25, 38 triangle inequality, 71, 76 truncated SVD, 129 Tucker decomposition, 136 underfitting, 271 undirected graphical model, 283 uniform distribution, 182 univariate, 178 unscented transform, 170 upper-triangular matrix, 101 validation set, 263, 284 variable selection, 316 variance, 190 vector, 37 vector addition, 37 vector space, 37 vector space homomorphism, 48 vector space with inner product, 73 vector subspace, 39 weak duality, 235 zero-one loss, 381 