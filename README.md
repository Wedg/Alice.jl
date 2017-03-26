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
Predict the digit label (1 to 9) of each image.  
[html file](http://htmlpreview.github.com/?https://github.com/Wedg/Alice.jl/blob/master/demo/mnist/Demo_MNIST_28x28.html) (view in browser) or 
[ipynb file](demo/mnist/Demo_MNIST_28x28.ipynb) (run in Jupyter)

#### STL10
1. 
2. 

#### Word Embedding
1. 
2. 

## Documentation

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
