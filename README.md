The software designed and tested in this repository is a Python 3.10 implementation of an Echo State Network (ESN).
It is a precursor to the modern recurrent neural network in that it incorporates temporal-encodings into the learning of 
data. However, it circumvents the need for complicated and expensive training algorithms by leveraging the macroscopic 
performance capabilities of random initialisation (and albeit judicious selection of hyperparameters).

Besides the final readout weight matrix, the entirety of the network is randomly initialised, and its primary constituent
is the sparse and recurrently connected reservoir at the 'core' of the network. This encodes temporal information by way 
of an update equation, which sums a fraction of the previous state with an expanded vector of the input signal. This 
enriched feature space can then be learnt using simple ordinary-least-squares regression techniques.

This implementation focuses on delivering the core components of the Echo State Network, whilst providing flexibility on 
how it can be customised, with options such as: changing the parent distributions of the randomly generated weights; 
utilising a bias vector alongside the inputs, using feedback to allow the output to contribute to the feature expansion; 
noise-injection, amongst others. Due to being a pythonic, CPU-based class, it is not designed for large or highly complex
tasks, and is most appropriate for proof-of-concept academic investigation of the network's utility on different datasets.