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

The constructor functions are (note that the square brackets are just indicating that the argument is optional):  


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

#### Hidden layer activation functions
The keyword argument `activation` is used to select the activation function (if the layer has an activation - WordEmbeddingLayer, MeanPool and MaxPool do not). The default (i.e. applied if no selection is made is `:logistic`) and the options are:
- `:logistic` - logistic
- `:tanh` - hyperbolic tangent
- `:relu` - rectified linear unit

#### Initialising weights in a hidden layer
The keyword argument `init` is used to select the distribution which is sampled from to initialise the layer weights. Note that the bias values are set to zero. The option is to provide either a distribution (any distribution from the Distributions package e.g. `Normal(0, 0.01)` or `Uniform(-0.5, 0.5)`) or one of the named options which are the following:  
- `:glorot_logistic_uniform`
- `:glorot_logistic_normal`
- `:glorot_tanh_uniform`
- `:glorot_tanh_normal`
- `:he_uniform`
- `:he_normal`
- `:lecun_uniform`
- `:lecun_normal`

The default selection (i.e. applied if no selection is made) is `:glorot_logistic_uniform` for layers with logistic activations, `:glorot_tanh_uniform` for layers with tanh activations, `:he_uniform` for layers with relu activations, and `Normal(0, 0.01)` for the WordEmbeddingLayer.

See here for my understanding of the merits of the different named options - [html file](http://htmlpreview.github.com/?https://github.com/Wedg/Alice.jl/blob/master/demo/Demo_Initialise_Weights.html) (view in browser) or 
[ipynb file](demo/Demo_Initialise_Weights.ipynb) (run in Jupyter)

#### Step 4 - Create output layers
These are constructed in a similar way to the hidden layers:  
- `LinearOutputLayer([datatype, ]databox, input_dims)`
- `MultiLinearOutputLayer([datatype, ]databox, input_dims)`
- `LogisticOutputLayer([datatype, ]databox, input_dims)`
- `SoftmaxOutputLayer([datatype, ]databox, input_dims, num_classes)`

  (`datatype`, `databox`, `input_dims` are as defined above and `num_classes` is an integer giving the number of classes / categories)
  
There is also the `init` keyword argument that can be used in the same way as initialising the hidden layers. The output layers are initialised to zero by default.

#### Step 5 - Create the neural network
There is 1 constructor function with 2 methods and 1 keyword argument:  
- `NeuralNet(databox, layers)`
- `NeuralNet(databox, layers, λ)`
- `NeuralNet(databox, layers, λ, regularisation = :L2)`

  (`databox` is as defined above, `layers` is a vector of layers (typically 1 input layer, many hidden layers, and 1 output layer), `λ` is the regularisation parameter and , `regularisation` is the keyword argument for the type of regularisation - options are `:L1` and `:L2` (the default))


#### An Example
In the MNIST demo the first feedforward neural network is created as follows:
```jlcon
# Data Box and Input Layer
databox = Data(train_images, train_labels, val_images, val_labels)
batch_size = 128
input = InputLayer(databox, batch_size)

# Fully connected hidden layers
dim = 30
fc1 = FullyConnectedLayer(size(input), dim, activation = :tanh)
fc2 = FullyConnectedLayer(size(fc1), dim, activation = :tanh)

# Softmax Output Layer
num_classes = 10
output = SoftmaxOutputLayer(databox, size(fc2), num_classes)

# Model
λ = 1e-3    # Regularisation
net = NeuralNet(databox, [input, fc1, fc2, output], λ, regularisation=:L2)
```

This creates the following:
```jlcon
Neural Network
Training Data Dimensions - (28,28,50000)
Layers:
Layer 1 - InputLayer{Float64}, Dimensions - (28,28,128)
Layer 2 - FullyConnectedLayer{Float64}, Activation - tanh, Dimensions - (30,128)
Layer 3 - FullyConnectedLayer{Float64}, Activation - tanh, Dimensions - (30,128)
Layer 4 - SoftmaxOutputLayer{Float64,Int64}, Dimensions - (10,128)
```

## Training a Neural Network
There are 2 broad options for training:
1. `train` function for stochastic mini-batch training by gradient descent with momentum
2. `train_nlopt` function for full batch training using the NLopt package that provides an interface to the open-source NLopt library for nonlinear optimisation

`train` function:

- `train(net, num_epochs, α, μ[, nesterov = true, shuffle = false, last_train_every = 1, full_train_every = num_epochs, val_every = num_epochs])`

  (Positional arguments - `net` is the neural network, `num_epochs` is the total number of epochs to run through, `α` is the learning rate, `μ` is the momentum parameter)  
  (Keyword arguments - `nesterov` is whether to use Nesterov's accelerated gradient method (default is `true`, if `false` uses standard momentum method), `shuffle` is whether to randomly shuffle the data before each epoch (default is `false`), `last_train_every` selects the epoch intervals to display the last batch training error (default is `1` i.e. every epoch), `full_train_every` selects the epoch intervals to display the loss on the full training set (default is `num_epochs` i.e. only at the end), and `val_every` selects the epoch intervals to display the loss on the validation set (default is `num_epochs` i.e. only at the end))

`train_nlopt` function:

- `train_nlopt(net[, maxiter, algorithm])`

  (Positional arguments - `net` is the neural network)  
  (Keyword arguments - `maxiter` is the maximum number of iterations through the training set (will stop before that if a tolerance is achieved) (default is `100`), `algorithm` is any of the NLopt provided algorithms (default is `:LD_LBFGS`))

#### Visualising the training loss
If the `train` function has been used there is a plotting function to display the training progress:  
- `plot_loss_history(net, last_train_every, full_train_every, val_every)`

  (`net` is the neural net, `last_train_every`, `full_train_every` and `val_every` are as defined above in the `train` function but here they are just integers i.e. not keywords)
  
This will produce something like this:
![](demo/Loss_history.png)


## Evaluating performance
The `train` function will display the results of training on the training and test sets (unless you've chosen not to display). But to manually evaluate performance there are a number of functions:
- `loss(net, X, y)` - provides training loss on inputs `X` and target `y` e.g. `X_train` and `y_train` as defined above
- `val_loss(net, X, y)` - provides validation loss (without the regularisation cost) on `X` and `y`
- `loss_and_accuracy(net, X, y)` - provides training loss and accuracy percentage on `X` and `y`
- `val_loss_and_accuracy(net, X, y)` - provides validation loss (without regularisation cost) and accuracy percentage on `X` and `y`
- `accuracy(net, X, y)` - provides the accuracy percentage on `X` and `y`

Note that the accuracy functions will only work for Logistic and Softmax output layers.
