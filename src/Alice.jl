module Alice

##############################################################################
##
## Dependencies and Reexports
##
##############################################################################

using Reexport
@reexport using Gadfly
@reexport using Distributions
@reexport using Images
using NLopt

using Base.Threads
import Base.show
import Base.size

##############################################################################
##
## Exported methods and types
##
##############################################################################

export Data, InputLayer, FullyConnectedLayer, WordEmbeddingLayer,
       SparseEncoderLayer, ConvolutionLayer, MeanPoolLayer, MaxPoolLayer,
       LinearOutputLayer, MultiLinearOutputLayer, LogisticOutputLayer,
       SoftmaxOutputLayer, NeuralNet, tinynet, check_gradients,
       load_ngrams, load_patches, load_train_subset, load_test_subset,
       load_features,
       fwd_prop!, bwd_prop!, train, train_nlopt,
       loss, val_loss, loss_and_accuracy, val_loss_and_accuracy, accuracy,
       predictions,
       size, viewbatch, display_nearest_words, predict_next_word,
       plot_loss_history, display_rgb_cols, display_rgb_weights

##############################################################################
##
## Load source files
##
##############################################################################

include("layers.jl")
include("init_params.jl")
include("activations.jl")
include("convolutions.jl")
include("pooling.jl")
include("cost_and_delta.jl")
include("regularisation.jl")
include("backprop.jl")
include("train.jl")
include("train_nlopt.jl")
include("performance.jl")
include("utils.jl")
include("check_gradients.jl")
include("plotting.jl");

end # module
