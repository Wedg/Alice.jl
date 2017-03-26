# Alice.jl

A deep learning package built in Julia.

The package is not yet registered so to download, run the command:
```jlcon
Pkg.clone("https://github.com/Wedg/Alice.jl.git")
```

## Demos

There are 3 demos that display almost all the functionality. All the data for each one is contained in the repository.

#### MNIST
![](demo/mnist/mnist.jpg)  
Predict the digit label (0 to 9) of each image.  
Feedforward Neural Network and Convolutional Neural Network - [html file](http://htmlpreview.github.com/?https://github.com/Wedg/Alice.jl/blob/master/demo/mnist/Demo_MNIST_28x28.html) (view in browser) or 
[ipynb file](demo/mnist/Demo_MNIST_28x28.ipynb) (run in Jupyter)

#### STL10 (reduced)
![](demo/stl10/stl10_8.jpg)  
Predict the object in each image from 4 classes (airplane, car, cat, dog).  
Part 1 - Sparse Autoencoder - [html file](http://htmlpreview.github.com/?https://github.com/Wedg/Alice.jl/blob/master/demo/stl10/Demo_STL10_A_Sparse_Autoencoder.html) (view in browser) or 
[ipynb file](demo/stl10/Demo_STL10_A_Sparse_Autoencoder.ipynb) (run in Jupyter)  
Part 2 - Convolutional Neural Network - [html file](http://htmlpreview.github.com/?https://github.com/Wedg/Alice.jl/blob/master/demo/stl10/Demo_STL10_B_Convolution_and_Pooling.html) (view in browser) or 
[ipynb file](demo/stl10/Demo_STL10_B_Convolution_and_Pooling.ipynb) (run in Jupyter) 

#### Word Embedding
NLP (Natural Language Processing) example. Learn feature representations of words through learning to predict the next word in a given sequence of words.  
Word Embedding Network - [html file](http://htmlpreview.github.com/?https://github.com/Wedg/Alice.jl/blob/master/demo/ngrams/Demo_Word_Embedding.html) (view in browser) or 
[ipynb file](demo/ngrams/Demo_Word_Embedding.ipynb) (run in Jupyter)

## Documentation
There are a number of types/structs defined that are used to build a neural network. There is a data container to place all training, validation and test sets in. And there are a number of network layers (input, hidden and output). The current available list is:

Data container:
- Data

Input Layer:
- InputLayer

Hidden Layers:
- FullyConnectedLayer
- WordEmbeddingLayer
- SparseEncoderLayer
- ConvolutionLayer
- MeanPoolLayer
- MaxPoolLayer

Output Layers:
- LinearOutputLayer
- MultiLinearOutputLayer
- LogisticOutputLayer
- SoftmaxOutputLayer

## Building a Neural Network
#### Step 1 - Place data in a data container
There are 4 functions available:    
- `Data(X_train)`
- `Data(X_train, y_train)`
- `Data(X_train, y_train, X_val, y_val)`
- `Data(X_train, y_train, X_val, y_val, X_test, y_test)`

  (`X_train` is an array of training data, `y_train` is an array of target data for the training set, `X_val` is an array of validation data, `y_val` is an array of target data for the validation set, `X_test` is an array of test data, and `y_test` is an array of target data for the test set.)

Note that a reference to the original data is used as opposed to a copy for better memory management. So if the data is changed that will also change the data in this data container.

#### Step 2 - Create input layer
There is only 1 function:  
- `InputLayer(databox, batch_size)`  
(`databox` is a data container described above and `batch_size` is an integer giving the number of observations of the training set in each mini-batch.)

#### Step 3 - Create hidden layers
Each hidden layer has its own constructor. Each constructor starts with the same two arguments. The first one (`datatype` - a floating point data type) is optional (if excluded will default to `Float64`) and the second (`input_dims` - tuple of the size dimensions of the previous layer) is required.  

Following the first two (or one if `datatype` is left out) each hidden layer has it's own positional arguments (each shown below).

And following the positional arguments there are two optional keyword arguments - `activation` (for choosing the activation function) and `init` (for choosing the initialisation of the weights). These are described in more detail in the section below).

The constructor functions are (note that the square brackets around `datatype` are just indicating that it is optional):  


- `FullyConnectedLayer([datatype, ]input_dims, fan_out)`  
(`fan_out` is the number of neurons in the layer)

- `WordEmbeddingLayer([datatype, ]input_dims, vocab_size, num_feats)`  
(`vocab_size` is the number of words in the vocabulary and `num_feats` is the number of features / length of feature vector given to each word)

- `SparseEncoderLayer([datatype, ]input_dims, fan_out, ρ, β)`  
(`ρ` is the sparsity parameter and `β` is the parameter controlling the weight of the sparsity penalty term)

- `ConvolutionalLayer([datatype, ]input_dims, patch_dims)`  
(`patch_dims` is a tuple giving the size dimensions of the patch / filter used for the convolution operation)

- `MeanPoolLayer([datatype, ]input_dims, stride)`  
- `MaxPoolLayer([datatype, ]input_dims, stride)`  
(`stride` is the pooling stride used for the pooling operation - either max or mean)
