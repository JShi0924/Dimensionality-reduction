# Dimensionality reduction  
From Professor Stefan Bekiranov's lecture slides "Differential expression analysis". Scribed by Jiachen Shi.
## Curse of dimensionality  

<div align=center><img width="500" height="250" src="https://github.com/JShi0924/Dimensionality-reduction/blob/main/dimentionality%20reduction_image/hypersphere.png">  


> **<p align="left">Calculation of the volume of the hypersphere divided by the hypercube as $d → ∞$**
> 
> $V_{HC}=(2r)^d$ and $V_{HS}=\frac{\pi^{\frac{d}{2}}r^d}{\Gamma (\frac{d}{2}+1)}$ thus,  
	$\frac{V_{HS}}{V_{HC}}=\frac{\pi^{\frac{d}{2}}}{2^d \Gamma (\frac{d}{2}+1)}\rightarrow \frac{1}{\sqrt \pi d}(\frac{\pi e}{2d})^{\frac{d}{2}}$

> **<p align="left">So as $d → ∞$, $V_{HS}/V_{HC}→0$**  

> <p align="left">And the distance between the center and corners of the hypercube is $\sqrt dr→ ∞$


**<p align="left">Based on the calculation, we know that:**   

* <p align="left">Nearly all of high-dimensional space in a hypercube is **distant from the center and close to the border**.
* <p align="left">High dimensional datasets at risk of **being sparse**. The average distance between two random points:
	in a unit square is roughly 0.52.
	in a unit 3-d cube is roughly 0.66.
	in a unit 1,000,000-d hypercube is ∼408.25.
* **<p align="left">Distances from a random point to its nearest and farthest neighbor are similar.**
* <p align="left">Distance-based classification generalizes poorly unless # samples grows exponentially with d  

## <p align="left">The features of biological networks  

<div align=center><img width="500" height="400" src="https://github.com/JShi0924/Dimensionality-reduction/blob/main/dimentionality%20reduction_image/Biological%20networks.png">  

* <p align="left">Highly interconnected with modular structure.
* <p align="left">Weakly to strongly scale-free (fraction of nodes with degree k follows a power law k<sup>(-$\alpha$)</sup>.
* <p align="left">Subsets of genes, proteins or regulatory elements tend to form highly correlated modules.
* <p align="left">Functional genomics datasets tend to (not always!) occupy a low dimensional subpace of the feature space (e.g., genes, proteins, regulatory elements).
* <p align="left">Ideal for dimenstional reduction approaches to both visualize and analyze functional genomics data.  

## <p align="left">Principal components analysis (PCA)
<div align=center><img width="400" height="300" src="https://github.com/JShi0924/Dimensionality-reduction/blob/main/dimentionality%20reduction_image/pca.png">  

### <p align="left">The concept of PCA  
<p align="left">The principal components of a collection of points in a real coordinate space are a sequence of p unit vectors, where the i-th vector is the direction of a line that best fits the data while being orthogonal to the first i-1 vectors. Here, a best-fitting line is defined as one that minimizes the average squared distance from the points to the line. These directions constitute an orthonormal basis in which different individual dimensions of the data are linearly uncorrelated. Principal component analysis (PCA) is the process of computing the principal components and using them to perform a change of basis on the data, sometimes using only the first few principal components and ignoring the rest.

### <p align="left">The principle of PCA   
<p align="left">Assume we have $n$ samples and $p$ features which are in the form of a $n × p$ centered matrix Χ where we subtracted the mean across samples of each feature.  

> <p align="left">The unbiased sample covariance matrix is then 
<center>$\sum_{XX}= {\frac{1}{n-1}}X^TX$<center>

<p align="left">PCA finds a linear transformation $Z = XV$ that diagonalizes $\sum_{XX}$.  

**<p align="left">Singular value decomposition**  

> <p align="left">X can be decomposed as follows:  
<center>$X=UDV^T$</center >  

<p align="left">where U and V are $n×n$ and $p×p$ orthogonal matricies, respectively, and D is a $n×p$ diagonal matrix. The diagonal elements of D are the singular values of X. The columns of U and V are the left-singular vectors and right-singular vectors.

<p align="left">The left singular vectors and right singular vectors of X are the eigenvectors of $XX^T$  and $X^TX$.  

<p align="left">The nonzero singular values of X are the square roots of the eigenvalues of $XX^T$  and $X^TX$.  
	
**<p align="left">PCA**  

> <p align="left">The covariance matrix of Z = XV where the columns of V are the right-singular vectors of X is  

> <center>$\sum_{ZZ}= {\frac{1}{n-1}}Z^TZ={\frac{1}{n-1}}D^TD={\frac{1}{n-1}}\hat{D}^2$<center>  

> <p align="left">Where $D ̂^2$ is a a square diagonal matrix (0s truncated), and we have used the SVD of X  
> <center>$(UDV^T)^T=VD^TU^T$<center>
<center>$V^TV=I_P$<center>
<center>$U^TU=I_n$<br><center>  
	
**<p align="left">Non-negative matrix factorization</p>**  

<p align="left">Factorize a matrix V with all positive elements into a product of two matricies W (features matrix) and H (coefficients matrix) subject to the constraint that W and H have no negative elements. Formally,  
<center>V $\approx$ WH such that W $\geq$ 0 and H $\geq$ 0  
<p align="left">where V, W and H are $m \times n$, $m \times p$ and $p \times n$ matricies, m is the number of samples, n is the number of features, p << n, and possibly p << m.  

<p align="left">To approximate V $\approx$ WH, we need a cost function. We'll focus on a Euclidean distance based cost function between V and WH  
<center>$||V - WH||^2=\sum_{ij}(V_{ij} - (WH)_{ij})^2$<CENTER> 
<p align="left">Minimize cost function using Lee and Seung's multiplicative update rule:  
* Initialize H and W with non negative values  
* Update $H_{ij}$ and $W_{ij}$
	<CENTER>$H_{ij}\leftarrow H_{ij}\frac{(W^TV)_{ij}}{(W^TWH)_{ij}}$<CENTER>  
	<CENTER>$W_{ij}\leftarrow W_{ij}\frac{(VH^T)_{ij}}{(WHH^T)_{ij}}$<CENTER>  
<p align="left">* Stop when $H_{ij}$ and $W_{ij}$ don't change within a specified tolerance  

<div align=center><img width="600" height="450" src="https://github.com/JShi0924/Dimensionality-reduction/blob/main/dimentionality%20reduction_image/NMF.png">  

## <p align="left">T-distributed stochastic neighbor embedding (t-SNE)  

<div align=center><img width="600" height="450" src="https://github.com/JShi0924/Dimensionality-reduction/blob/main/dimentionality%20reduction_image/t-sne.png">  

<p align="left">A non-linear dimensional reduction approach that attempts to map a distribution of pairwise distances among nn high-dimensional samples from their high dimension to a distribution of pairwise distances of the nn samples in a low dimension.    

<div align=center><img width="650" height="350" src="https://github.com/JShi0924/Dimensionality-reduction/blob/main/dimentionality%20reduction_image/t-sne_1.png">  

<p align="left">The goal of t-SNE is to learn a map $X ↦ Y$ such that the $y^1$, ..., $y^n$ are mapped to a lower dimensional space (d = 2 or 3). In the lower dimensional space, the probability of $y^i$, $y^j$ being associated/near each other is assumed to follow a t-distribution with one degree of freedom where for i $\neq$ j  
<center>$q_{i,j}=\frac{(1+||y_j-y_j||^2)^{-1}}{\sum_{k,l;k\neq l}(1+||y_k-y_l||^2)^{-1}} $ and 0 otherwise  
<p align="left">The locations of the $y_i$ in the lower dimensional space are found by minimizing the Kullback-Leibler divergence between the $p_{ij}$ and $p_{ij}$ distributions  
<center>$KL(p||q)=\sum_{i,j;i\neq j}p_{ij}\log \frac{p_{ij}}{q_{ij}}$  
<p align="left">which is a measure of the difference between distributions p and q. The minimization of the Kullback-Leibler divergence with respect to $y_i$ is performed using gradient descent.  

## <p align="left">Uniform manifold approximation and projection (UMAP)  
<p align="left">Uniform manifold approximation and projection (UMAP) is a nonlinear dimensionality reduction technique. Visually, it is similar to t-SNE, but it assumes that the data is uniformly distributed on a locally connected Riemannian manifold and that the Riemannian metric is locally constant or approximately locally constant.  

<div align=center><img width="700" height="450" src="https://github.com/JShi0924/Dimensionality-reduction/blob/main/dimentionality%20reduction_image/UMAP.png">  
 
> <p align="left">Let $X_i$, ..., $X_n$ be the input data. For each $X_i$ compute the k nearest neughbors $X_{i1}$, ..., $X_{ik}$ and  
<center>$\rho=\min \{d(X_{i}, X_{ik})|1 \leq j \leq k, d > 0\}$  
and set $\sigma_i$ using $\sum^k_{j=1}e^{-max(0,d(X_i,X_{i,j})-\rho_i)/\sigma_i}=\log_2(k)$  

**<p align="left">Graph construction**  
> <p align="left">Define a weighted directed graph $\overline{G}=(V,E,w)$ where the verticies V of $\overline{G}$ are the set X. Then form the set of directed edges with weights $w_h$  
<CENTER>$E=\{(X_i,X_{ij})|1\leq j \leq k, 1 \leq i \leq n \}$  
<CENTER>$w_h(X_i,X_{ij})=e^{-max(0,d(X_i,X_{ij})-\rho_i)/\sigma_i}$  
<p align="left">Combine edges of $\overline{G}$ with adjacency matrix A into a unified undirected graph $G$ with adjacency matrix $B$  
<CENTER>$B=A+A^T-A\circ A^T$  
<p align="left">where $\circ$ is the Haddamard (or pointwise) product.  
<p align="left">If $A_{ij}$ is the probability that the directed edge from $X_i$ to $X_j$ exists, then $B_{ij}$ is the probability that at least one of the two directed edges (from $X_i$ to $X_j$ and from $X_j$ to $X_i$) exists.  

**<p align="left">Graph layout**
<p align="left">Learn a mapping $X↦Y$ by first initializing the low dimentional representation using spectral embedding. Spectral embedding is a dimensional reduction approach that maps a connected graph G to a low dimension vector space in such a way that two points that are "close" in the graph are "close" in the low dimensional vector space.  

<p align="left">Formally, this is done by calculating the eigenvectors of the Laplacian L = D - B of the adjacency matrix B where D is a diagonal matrix with $D_{ii}=\sum_j B_{ij}$.  

<p align="left">Sort the eigenvalues $\lambda_0, \lambda_1, \lambda_2, ...,$ and the eigenvectors corresponding to $\lambda_1$ and $\lambda_2$ represent the embedding in a 2-d vector space.  

<p align="left">Fit $w_l$, a differentiable form of the 2-d weights to exp. Estimate aa and bb by performing non-linear regression of $w_l(x,y)=(1+a(||x-y||_2^2)^b)_{-1}$ to $\psi(x,y)=e^{-(||x-y||_2-d_{min})}$ if $||x-y||_2$ is greater than $d_{min}$ and 1 otherwise.  

<p align="left">The locations of the $y_i$ in the lower dimensional space are found by minimizing the binary cross entropy loss $L$ using stochastic gradient descent where the loss is  
<center>$L=-\displaystyle \sum_{b \in B}w_h(b) \log w_l(b)+(1-w_h(b)) \log(1-w_l(b))$
