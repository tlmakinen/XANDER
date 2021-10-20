This code is part of an exploratory study for finding anomalies in the Chandra source catalog directly from time series data. Our goal is to automate otherwise serendipitous discovery with the help of deep learning architectures trained on existing preprocessed data. 

# xander: X-ay ANomaly DEtectoR

### experiment overview

Since Chandra source catalog event files are of variable length, they pose a challenge with regards to developing statistical tools like neural networks for classification and anomaly detection. To circumvent this, we first considered a structure function akin to Mahabal et al's approach for lightcurves, in which time series data are broken into grids of time $dt$ and energy $d\epsilon$ differences. For an event file of length $n$ photon events, this corresponds to a dataset of size $n(n-1)/2$ datapoints. Mahabal et al then explored a 2D histogramming scheme in $dt$ and $d\epsilon$, which could be fed into convolutional neural networks for anomaly detection. 

We first attempted a regression problem, in which 2D histograms of event files were fed into a regression neural network with the aim to predict event file variabilities. However, we found that this approach did 

We tried this experiment initially, but found such histogramming schemes to be hard to interpret, especially for event files with a very small number of counts. Furthermore, choice of binning significantly impacted event file image representation and network extraction. Rather than rely on a hand-picked binning for our dataset, we turn to a more clever network implementation

### deepsets implementation description
This choice of structure function, while not easily interpretable in the form of an image for regression or anomaly detection, *does* make event files permutation-invariant. We can equivalently think of our event files as *sets* (or graphs) of photon events and their corresponding times and positions. The raw event file is an order-dependent directed graph of these events. However, in the $(dt,d\epsilon)$ representation, the data are now order-invariant. DeepSets neural network architectures exploit this property and have been shown to be more efficient in regression and anomaly classification for unordered data.

##### DeepSets Notation
We adopt the unified notation introduced in Battaglia et al. We define each $i^{\rm th}$ structured event file as a set, or an unordered graph $G_i$, comprised of a set of nodes $V$ and an empty set of edges $E=\{0 \}$, such that $G_i=(V,E)_i$. Each of the $n_{\rm nodes} = n(n-1)/2$ nodes is a $S = (dt,d\epsilon,dx,dy)$ tuple. Each event file graph $G_i$ has a *global* attribute $\boldsymbol{u}$ assigned to it, in our case the set of $k=11$ quantities we wish to predict:
$$ \boldsymbol{u} = [\sigma_1, \sigma_2, \dots \sigma_k]^T $$.

Intuitively, the set $S$ of node features will have some relationship with the global properties of the event file. To relate these, we define a neural network function to embed node features
$$ \phi^v(\textbf{v}, \textbf{u}) = f^v(\textbf{v}, \textbf{u}) = NN_v([\textbf{v}, \textbf{u}]) $$
where $[\textbf{x}, \textbf{z}]$ indicates vector concatenation. Once nodes are embedded, the resulting features are then fed into a layer that combines nodes

$$ \rho^{v\rightarrow u}(V) = \sum_i \textbf{v}_i $$

where the summation can be any order-invariant operator, such as a maxpool or average over multiple inputs. Another decoder network can then be used to produce the predicted 



Expanding an event file into this format helps remove the dependence o


IDEA: for regression, make hardness ratios global inputs and variabilities global outputs 

for anomaly detection, make (hardness, variabilities) global inputs and anomalousness the output
