#=
This page defines a number of composite types:

1. Data
    - Contains all the training, validation and test data.
    - The training data is fed in batches into the network during learning.

2. Layer
    - An abstract type with a number of concrete layer types:

    Input Layers:
    i.    InputLayer

    Hidden Layers:
    ii.   FullyConnectedLayer
    iii.  WordEmbeddingLayer
    iv.   SparseEncoderLayer
    v.    ConvolutionLayer
    vi.   MeanPoolLayer
    v.    MaxPoolLayer

    Output Layers:
    vi.   LinearOutputLayer
    vii.  MultiLinearOutputLayer
    viii. LogisticOutputLayer
    ix.   SoftmaxOutputLayer

    TODO - MultiLogistic, MultiSoftmax

3. NeuralNet
    - The complete network architecture.
    - Contains the data, network layers, and regularisation.
=#

########################################
# Data Container
########################################

abstract Data

type UData{R1<:Real} <: Data
    X_train::Array{R1}
end

function Data(X_train)
    UData(X_train)
end

type TData{R1<:Real, R2<:Real} <: Data
    X_train::Array{R1}
    y_train::Array{R2}
end

function Data(X_train, y_train)
    TData(X_train, y_train)
end

type TVData{R1<:Real, R2<:Real} <: Data
    X_train::Array{R1}
    y_train::Array{R2}
    X_val::Array{R1}
    y_val::Array{R2}
end

function Data(X_train, y_train, X_val, y_val)
    TVData(X_train, y_train, X_val, y_val)
end

type TVTData{R1<:Real, R2<:Real} <: Data
    X_train::Array{R1}
    y_train::Array{R2}
    X_val::Array{R1}
    y_val::Array{R2}
    X_test::Array{R1}
    y_test::Array{R2}
end

function Data(X_train, y_train, X_val, y_val, X_test, y_test)
    TVTData(X_train, y_train, X_val, y_val, X_test, y_test)
end

########################################
# Layer
########################################

# Define an abstract type as a parent of the layers defined below
abstract NetLayer

# Define some type unions for the input values to the layer constructors
UTupInt = Union{Tuple{Vararg{Integer}}, Integer}
USymDistn = Union{Symbol, Distribution}

########################################
# Input Layer
########################################

# Define InputLayer composite type
type InputLayer{R<:Real} <: NetLayer
    A::Array{R}       # input data
    Ar::Matrix{R}     # shared memory matrix reshape for chaining with next layer
end

# InputLayer outer constructor
function InputLayer{D<:Data, T<:Integer}(databox::D, batch_size::T)

    # Input array - convert from SubArray to Array
    X = convert(Array, viewbatch(databox.X_train, 1, batch_size))

    # Dims for reshaped 2D matrix
    rows = prod(size(X)[1:end-1])
    cols = size(X)[end]

    InputLayer(
    X, reshape(X, rows, cols)
    )
end

########################################
# Fully Connected Layer
########################################

# Define FullyConnectedLayer composite type
type FullyConnectedLayer{F<:AbstractFloat} <: NetLayer
    b::Vector{F}       # bias vector
    W::Matrix{F}       # weight matrix
    Z::Matrix{F}       # weighted sum of inputs and params i.e. Z⁽³⁾ = A⁽²⁾ W⁽³⁾ + b⁽³⁾
    A::Matrix{F}       # post activation values i.e. A⁽³⁾ = f(Z⁽³⁾)
    Δ::Matrix{F}       # delta matrix
    ∇b::Vector{F}      # bias gradient
    ∇W::Matrix{F}      # weight gradient
    b_vel::Vector{F}   # bias velocity (for momentum learning)
    W_vel::Matrix{F}   # weight velocity (for momentum learning)
    f!::Function       # activation function
    ∇f!::Function      # activation function derivative and hademaard product
end

# FullyConnectedLayer outer constructor
function FullyConnectedLayer(F::DataType, input_dims::UTupInt, fan_out::Integer;
                             activation::Symbol = :logistic,
                             init::USymDistn = :default)

    # Dims
    fan_in = prod(input_dims[1:(end-1)])
    batch_size = input_dims[end]

    # Initialise weights
    distn = choose_distn(init, activation, fan_in, fan_out)
    W = F.(rand(distn, fan_out, fan_in))
    b = zeros(F, fan_out)

    # Build layer
    FullyConnectedLayer(
    b, W,
    zeros(F, fan_out, batch_size),
    zeros(F, fan_out, batch_size),
    zeros(F, fan_out, batch_size),
    zeros(b), zeros(W), zeros(b), zeros(W),
    activation_dict[activation], activation_deriv_dict[activation]
    )
end

# Default types are 64 bit
function FullyConnectedLayer(input_dims::UTupInt, fan_out::Integer;
                             activation::Symbol = :logistic,
                             init::USymDistn = :default)
    FullyConnectedLayer(Float64, input_dims, fan_out,
                        activation = activation, init = init)
end

########################################
# Word Embedding Layer
########################################

# Define WordEmbeddingLayer composite type
type WordEmbeddingLayer{F<:AbstractFloat} <: NetLayer
    W::Matrix{F}       # weight matrix
    A::Matrix{F}       # feature values of each input n-gram
    Δ::Matrix{F}       # delta matrix
    ∇W::Matrix{F}      # weight gradient
    W_vel::Matrix{F}   # weight velocity (for momentum learning)
end

# WordEmbeddingLayer outer constructor
function WordEmbeddingLayer(F::DataType, input_dims::UTupInt,
                            vocab_size::Integer, num_feats::Integer;
                            init::USymDistn = :default)

    # Dims
    num_words = input_dims[1]
    batch_size = input_dims[2]

    # Initialise weights
    if init == :default
        distn = Normal(0, 0.01)
    elseif init <: Symbol
        println("Error: Named initialisations are not used for Word Embedding Layer.
               Enter instead a distribution or leave out and the default
               distribution - Normal(0, 0.01) - will be used.")
        return
    elseif init <: Distribution
        distn = init
    end
    W = F.(rand(distn, num_feats, vocab_size))

    # Build layer
    WordEmbeddingLayer(
    W,
    zeros(F, num_words * num_feats, batch_size),
    zeros(F, num_words * num_feats, batch_size),
    zeros(W), zeros(W)
    )
end

# Default types are 64 bit
function WordEmbeddingLayer(input_dims::UTupInt, vocab_size::Integer,
                            num_feats::Integer; init::USymDistn = :default)
    WordEmbeddingLayer(Float64, input_dims, vocab_size, num_feats,
                       init = init)
end

########################################
# Sparse Encoder Layer
########################################

# Define SparseEncoderLayer composite type
type SparseEncoderLayer{F<:AbstractFloat} <: NetLayer
    b::Vector{F}       # bias vector
    W::Matrix{F}       # weight matrix
    Z::Matrix{F}       # weighted sum of inputs and params i.e. Z⁽³⁾ = A⁽²⁾ W⁽³⁾ + b⁽³⁾
    A::Matrix{F}       # post activation values i.e. A⁽³⁾ = f(Z⁽³⁾)
    Δ::Matrix{F}       # delta matrix
    ∇b::Vector{F}      # bias gradient
    ∇W::Matrix{F}      # weight gradient
    b_vel::Vector{F}   # bias velocity (for momentum learning)
    W_vel::Matrix{F}   # weight velocity (for momentum learning)
    p̂::Vector{F}       # average activation - "prob" of each neuron firing
    spgrad::Vector{F}  # gradient of the cost of sparsity
    p::F               # sparsity parameter
    β::F               # controls the weight of the sparsity penalty term
    f!::Function       # activation function
    ∇f!::Function      # activation function derivative and hademaard product
    δ!::Function       # delta function
    m::F               # batch_size
end

# SparseEncoderLayer outer constructor
function SparseEncoderLayer(F::DataType, input_dims::UTupInt, fan_out::Integer,
                            p::Real, β::Real;
                            activation::Symbol = :logistic,
                            init::USymDistn = :default)

    # Dims
    fan_in = prod(input_dims[1:(end-1)])
    batch_size = input_dims[end]

    # Initialise weights
    distn = choose_distn(init, activation, fan_in, fan_out)
    W = F.(rand(distn, fan_out, fan_in))
    b = zeros(F, fan_out)

    # Build layer
    SparseEncoderLayer(
    b, W,
    zeros(F, fan_out, batch_size),
    zeros(F, fan_out, batch_size),
    zeros(F, fan_out, batch_size),
    zeros(b), zeros(W), zeros(b), zeros(W),
    zeros(b), zeros(b), p, β,
    activation_dict[activation], activation_deriv_dict[activation],
    delta_sparse!,
    F(batch_size)
    )
end

# Default types are 64 bit
function SparseEncoderLayer(input_dims::UTupInt, fan_out::Integer,
                            p::Real, β::Real;
                            activation::Symbol = :logistic,
                            init::USymDistn = :default)
    SparseEncoderLayer(Float64, input_dims, fan_out, p, β,
                       activation = activation, init = init)
end

########################################
# Convolution Layer
########################################

# Define ConvolutionLayer composite type
type ConvolutionLayer{F<:AbstractFloat} <: NetLayer
    b::Vector{F}       # bias vector
    W::Array{F}        # weight array
    Z::Array{F}        # convolved sum of inputs and params plus bias i.e. Z⁽³⁾ = A⁽²⁾ ∗ W⁽³⁾ + b⁽³⁾
    Zr::Matrix{F}      # shared memory matrix reshape for chaining with next layer
    A::Array{F}        # post activation values i.e. A⁽³⁾ = f(Z⁽³⁾)
    Ar::Matrix{F}      # shared memory matrix reshape for chaining with next layer
    Δ::Array{F}        # delta array
    Δr::Matrix{F}      # shared memory matrix reshape for chaining with next layer
    ∇b::Vector{F}      # bias gradient
    ∇W::Array{F}       # weight gradient
    b_vel::Vector{F}   # bias velocity (for momentum learning)
    W_vel::Array{F}    # weight velocity (for momentum learning)
    f!::Function       # activation function
    ∇f!::Function      # activation function derivative and hademaard product
end

# ConvolutionLayer outer constructor
function ConvolutionLayer(F::DataType, input_dims::Tuple, patch_dims::Tuple;
                          activation::Symbol = :logistic,
                          init::USymDistn = :default)

    # Dims
    fan_in = prod(patch_dims[1:(end - 1)])
    fan_out = patch_dims[end]
    conv_dims = Array(Int, 4)   # dim x dim x num_patches x num_obs
    conv_dims[1] = input_dims[1] - patch_dims[1] + 1
    conv_dims[2] = input_dims[2] - patch_dims[2] + 1
    conv_dims[3] = fan_out
    conv_dims[4] = input_dims[end]

    # Initialise weights
    distn = choose_distn(init, activation, fan_in, fan_out)
    W = F.(rand(distn, patch_dims...))
    b = zeros(F, fan_out)

    # Z, A & Δ - Initialise for shared memory reshaping
    Z = zeros(F, conv_dims...)
    A = zeros(F, conv_dims...)
    Δ = zeros(F, conv_dims...)
    rows = prod(conv_dims[1:end-1])
    cols = conv_dims[end]

    # Build layer
    ConvolutionLayer(
    b, W,
    Z, reshape(Z, rows, cols),
    A, reshape(A, rows, cols),
    Δ, reshape(Δ, rows, cols),
    zeros(b), zeros(W), zeros(b), zeros(W),
    activation_dict[activation], activation_deriv_dict[activation]
    )
end

# Default types are 64 bit
function ConvolutionLayer(input_dims::Tuple, patch_dims::Tuple;
                          activation::Symbol = :logistic,
                          init::USymDistn = :default)
    ConvolutionLayer(Float64, input_dims, patch_dims,
                     activation = activation, init = init)
end

########################################
# Mean Pooling Layer
########################################

# Define MeanPoolLayer composite type
type MeanPoolLayer{F<:AbstractFloat} <: NetLayer
    A::Array{F}        # post activation values i.e. A⁽³⁾ = meanpool(A⁽²⁾)
    Ar::Matrix{F}      # shared memory matrix reshape for chaining with next layer
    Δ::Array{F}        # delta array
    Δr::Matrix{F}      # shared memory matrix reshape for chaining with next layer
end

# MeanPoolLayer outer constructor
function MeanPoolLayer(F::DataType, input_dims::UTupInt, stride::Integer)

    # Dims
    @assert rem(input_dims[1], stride) == 0
    pool_dim = div(input_dims[1], stride)
    pool_dims = (pool_dim, pool_dim, input_dims[end - 1], input_dims[end])

    # A & Δ - Initialise for shared memory reshaping
    A = zeros(F, pool_dims)
    Δ = zeros(F, pool_dims)
    rows = prod(pool_dims[1:(end - 1)])
    cols = pool_dims[end]

    # Build layer
    MeanPoolLayer(
    A, reshape(A, rows, cols),
    Δ, reshape(Δ, rows, cols)
    )
end

# Default types are 64 bit
function MeanPoolLayer(input_dims::UTupInt, stride::Integer)
    MeanPoolLayer(Float64, input_dims, stride)
end

########################################
# Max Pooling Layer
########################################

# Define MaxPoolLayer composite type
type MaxPoolLayer{F<:AbstractFloat, T<:Integer} <: NetLayer
    A::Array{F}        # post activation values i.e. A⁽³⁾ = maxpool(A⁽²⁾)
    Ar::Matrix{F}      # shared memory matrix reshape for chaining with next layer
    Δ::Array{F}        # delta array
    Δr::Matrix{F}      # shared memory matrix reshape for chaining with next layer
    maxinds::Array{T}  # linear index of the max value from each patch
end

# MaxPoolLayer outer constructor
function MaxPoolLayer(F::DataType, input_dims::UTupInt, stride::Integer)

    # Dims
    @assert rem(input_dims[1], stride) == 0
    pool_dim = div(input_dims[1], stride)
    pool_dims = (pool_dim, pool_dim, input_dims[end - 1], input_dims[end])

    # A & Δ - Initialise for shared memory reshaping
    A = zeros(F, pool_dims)
    Δ = zeros(F, pool_dims)
    rows = prod(pool_dims[1:end-1])
    cols = pool_dims[end]

    # Build layer
    MaxPoolLayer(
    A, reshape(A, rows, cols),
    Δ, reshape(Δ, rows, cols),
    zeros(Int, input_dims...)
    )
end

# Default types are 64 bit
function MaxPoolLayer(input_dims::UTupInt, stride::Integer)
    MaxPoolLayer(Float64, input_dims, stride)
end

########################################
# Linear Output Layer
########################################

# Define LinearOutputLayer composite type
type LinearOutputLayer{F<:AbstractFloat} <: NetLayer
    b::Vector{F}       # bias vector
    W::Matrix{F}       # weight matrix
    A::Matrix{F}       # weighted sum of inputs and weights plus bias - Z⁽ᴸ⁾ = A⁽ᴸ⁻¹⁾ W⁽ᴸ⁾ + b⁽ᴸ⁾
    Δ::Matrix{F}       # delta matrix
    ∇b::Vector{F}      # bias gradient
    ∇W::Matrix{F}      # weight gradient
    b_vel::Vector{F}   # bias velocity (for momentum learning)
    W_vel::Matrix{F}   # weight velocity (for momentum learning)
    δ!::Function       # delta function
    y::Vector{F}       # target vector / labels
    cost::Function     # cost function
    m::F               # batch_size
end

# LinearOutputLayer outer constructor
function LinearOutputLayer(F::DataType, databox::Data, input_dims::UTupInt;
                           init::Distribution = Bernoulli(0))

    # Dims
    dim_in = prod(input_dims[1:(end-1)])
    batch_size = input_dims[end]

    # Target vector - convert from SubArray to Array
    y = convert(Array, viewbatch(databox.y_train, 1, batch_size))

    # Check target is a vector of reals
    @assert typeof(y) <: Vector
    @assert eltype(y) <: AbstractFloat

    # Initialise weights
    W = F.(rand(init, 1, dim_in))
    b = zeros(F, 1)

    # Build layer
    LinearOutputLayer(
    b, W,
    zeros(F, 1, batch_size), zeros(F, 1, batch_size),
    zeros(b), zeros(W), zeros(b), zeros(W),
    delta_linear!, y, quadratic_cost,
    F(batch_size)
    )
end

# Default types are 64 bit
function LinearOutputLayer(databox::Data, input_dims::UTupInt;
                           init::Distribution = Bernoulli(0))
    LinearOutputLayer(Float64, databox, input_dims, init = init)
end

########################################
# Multi Linear Output Layer
########################################

# Define MultiLinearOutputLayer composite type
type MultiLinearOutputLayer{F<:AbstractFloat} <: NetLayer
    b::Vector{F}       # bias vector
    W::Matrix{F}       # weight matrix
    A::Matrix{F}       # weighted sum of inputs and weights plus bias - Z⁽ᴸ⁾ = A⁽ᴸ⁻¹⁾ W⁽ᴸ⁾ + b⁽ᴸ⁾
    Δ::Matrix{F}       # delta matrix
    ∇b::Vector{F}      # bias gradient
    ∇W::Matrix{F}      # weight gradient
    b_vel::Vector{F}   # bias velocity (for momentum learning)
    W_vel::Matrix{F}   # weight velocity (for momentum learning)
    δ!::Function       # delta function
    y::Matrix{F}       # target matrix - ℝ num_outputs x num_obs
    cost::Function     # cost function
    m::F               # batch_size
end

# MultiLinearOutputLayer outer constructor
function MultiLinearOutputLayer(F::DataType, databox::Data, input_dims::UTupInt;
                                init::Distribution = Bernoulli(0))

    # Dims
    dim_in = prod(input_dims[1:(end-1)])
    batch_size = input_dims[end]

    # Target vector - convert from SubArray to Array
    y = convert(Array, viewbatch(databox.y_train, 1, batch_size))

    # Check target is a matrix of reals
    @assert typeof(y) <: Matrix
    @assert eltype(y) <: AbstractFloat

    # Number of Outputs
    num_outputs = size(y, 1)

    # Initialise weights
    W = F.(rand(init, num_outputs, dim_in))
    b = zeros(F, num_outputs)

    # Build layer
    MultiLinearOutputLayer(
    b, W,
    zeros(F, num_outputs, batch_size), zeros(F, num_outputs, batch_size),
    zeros(b), zeros(W), zeros(b), zeros(W),
    delta_linear!, y, quadratic_cost,
    F(batch_size)
    )
end

# Default types are 64 bit
function MultiLinearOutputLayer(databox::Data, input_dims::UTupInt;
                                init::Distribution = Bernoulli(0))
    MultiLinearOutputLayer(Float64, databox, input_dims, init = init)
end

########################################
# Logistic Output Layer
########################################

# Define LogisticOutputLayer composite type
type LogisticOutputLayer{F<:AbstractFloat, T<:Integer} <: NetLayer
    b::Vector{F}       # bias vector
    W::Matrix{F}       # weight matrix
    Z::Matrix{F}       # weighted sum of inputs and weights plus bias i.e. Z⁽ᴸ⁾ = A⁽ᴸ⁻¹⁾ W⁽ᴸ⁾ + b⁽ᴸ⁾
    A::Matrix{F}       # post activation values i.e. A⁽ᴸ⁾ = f(Z⁽ᴸ⁾)
    Δ::Matrix{F}       # delta matrix
    ∇b::Vector{F}      # bias gradient
    ∇W::Matrix{F}      # weight gradient
    b_vel::Vector{F}   # bias velocity (for momentum learning)
    W_vel::Matrix{F}   # weight velocity (for momentum learning)
    f!::Function       # hypothesis function
    δ!::Function       # delta function
    y::Vector{T}       # target vector / labels
    cost::Function     # cost function
    m::F               # batch_size
end

# OutputLayer outer constructor
function LogisticOutputLayer(F::DataType, databox::Data, input_dims::UTupInt;
                             init::Distribution = Bernoulli(0))

    # Dims
    dim_in = prod(input_dims[1:(end-1)])
    batch_size = input_dims[end]

    # Target vector - convert from SubArray to Array
    y = convert(Array, viewbatch(databox.y_train, 1, batch_size))

    # Check target is a vector of 0 and 1 integers
    @assert typeof(y) <: Vector
    @assert eltype(y) <: Integer
    @assert sort(unique(y)) == [0, 1]

    # Initialise weights
    W = F.(rand(init, 1, dim_in))
    b = zeros(F, 1)

    # Build layer
    LogisticOutputLayer(
    b, W,
    zeros(F, 1, batch_size), zeros(F, 1, batch_size), zeros(F, 1, batch_size),
    zeros(b), zeros(W), zeros(b), zeros(W),
    logistic!, delta_logistic!, y, cross_entropy_cost,
    F(batch_size)
    )
end

# Default types are 64 bit
function LogisticOutputLayer(databox::Data, input_dims::UTupInt;
                             init::Distribution = Bernoulli(0))
    LogisticOutputLayer(Float64, databox, input_dims, init = init)
end

########################################
# Softmax Output Layer
########################################

# Define SoftmaxOutputLayer composite type
type SoftmaxOutputLayer{F<:AbstractFloat, T<:Integer} <: NetLayer
    b::Vector{F}       # bias vector
    W::Matrix{F}       # weight matrix
    Z::Matrix{F}       # weighted sum of inputs and weights plus bias i.e. Z⁽ᴸ⁾ = A⁽ᴸ⁻¹⁾ W⁽ᴸ⁾ + b⁽ᴸ⁾
    u::Matrix{F}       # Max row vector used in softmax function
    A::Matrix{F}       # post activation values i.e. A⁽ᴸ⁾ = f(Z⁽ᴸ⁾)
    Δ::Matrix{F}       # delta matrix
    ∇b::Vector{F}      # bias gradient
    ∇W::Matrix{F}      # weight gradient
    b_vel::Vector{F}   # bias velocity (for momentum learning)
    W_vel::Matrix{F}   # weight velocity (for momentum learning)
    f!::Function       # hypothesis function
    δ!::Function       # delta function
    y::Vector{T}       # target vector / labels
    cost::Function     # cost function
    m::F               # batch_size
end

# SoftmaxOutputLayer outer constructor
function SoftmaxOutputLayer(F::DataType, databox::Data, input_dims::UTupInt,
                            num_classes::Integer;
                            init::Distribution = Bernoulli(0))

    # Dims
    dim_in = prod(input_dims[1:(end-1)])
    batch_size = input_dims[end]

    # Target vector - convert from SubArray to Array
    y = convert(Array, viewbatch(databox.y_train, 1, batch_size))

    # Check target is a vector of integers
    @assert typeof(y) <: Vector
    @assert eltype(y) <: Integer

    # Initialise weights
    W = F.(rand(init, num_classes, dim_in))
    b = zeros(F, num_classes)

    # Build layer
    SoftmaxOutputLayer(
    b, W,
    zeros(F, num_classes, batch_size), zeros(F, 1, batch_size),
    zeros(F, num_classes, batch_size), zeros(F, num_classes, batch_size),
    zeros(b), zeros(W), zeros(b), zeros(W),
    softmax!, delta_softmax!, y, log_loss_cost,
    F(batch_size)
    )
end

# Default types are 64 bit
function SoftmaxOutputLayer(databox::Data, input_dims::UTupInt,
                            num_classes::Integer;
                            init::Distribution = Bernoulli(0))
    SoftmaxOutputLayer(Float64, databox, input_dims, num_classes, init = init)
end

########################################
# Layer Type Unions
########################################

# Output layers
LinOutputLayer = Union{LinearOutputLayer, MultiLinearOutputLayer}
OutputLayer = Union{LinOutputLayer, LogisticOutputLayer, SoftmaxOutputLayer}

# Pooling layers
PoolLayer = Union{MeanPoolLayer, MaxPoolLayer}

# High dimensional layers
HDL = Union{InputLayer, ConvolutionLayer, PoolLayer}

# Low dimensional layers
LDL = Union{FullyConnectedLayer, WordEmbeddingLayer, SparseEncoderLayer,
            OutputLayer}


########################################
# Whole Neural Network
########################################

# Define Network composite type that contains the layers
type NeuralNet{D<:Data, L<:NetLayer, F<:AbstractFloat, T<:Integer}
    dtype::DataType
    data::D
    layers::Vector{L}
    reg_cost::Function
    reg_grad!::Function
    λ::F
    batch_size::T
    last_train_loss::Vector{F}
    full_train_loss::Vector{F}
    val_loss::Vector{F}
end

# NeuralNet outer constructor
function NeuralNet{D<:Data, L<:NetLayer}(data::D, layers::Vector{L},
                                         λ=0.0; regularisation = :L2)

    # Check that the layer dimensions allow chaining
    checkdims(layers)

    # Check floating point data type is consistent
    F = eltype(layers[end].A)
    for ℓ in layers[2:end]
        @assert F == eltype(ℓ.A)
    end

    # Create Neural Network
    NeuralNet(
    F, data, layers,
    reg_cost_dict[regularisation], reg_grad_dict[regularisation], F(λ),
    size(layers[1])[end],
    Array(F, 0), Array(F, 0), Array(F, 0)
    )
end


########################################
# Check dimensions before building network
########################################

# Low -> Low
function checkdims(ℓ::LDL, prev_ℓ::LDL)
    @assert size(ℓ.W, 2) == size(prev_ℓ.A, 1)  # Number of features match weight matrix
    @assert size(ℓ.A, 2) == size(prev_ℓ.A, 2)  # Number of observations / batch size
end

# High -> Low
function checkdims(ℓ::LDL, prev_ℓ::HDL)
    @assert size(ℓ.W, 2) == size(prev_ℓ.Ar, 1)  # Number of features match weight matrix
    @assert size(ℓ.A, 2) == size(prev_ℓ.Ar, 2)  # Number of observations / batch size
end

# High -> High:

# High -> Word Embedding
function checkdims(ℓ::WordEmbeddingLayer, prev_ℓ::HDL)
    @assert size(ℓ.A, 1) / size(ℓ.W, 1) == size(prev_ℓ.A, 1)  # Number of words chains with weight matrix
    @assert size(ℓ.A)[end] == size(prev_ℓ.A)[end]  # Number of observations / batch size
end

# High -> Convolution
function checkdims(ℓ::ConvolutionLayer, prev_ℓ::HDL)
    @assert size(ℓ.A)[end] == size(prev_ℓ.A)[end]  # Number of observations / batch size
end

# High -> Pool
function checkdims(ℓ::PoolLayer, prev_ℓ::HDL)
    @assert size(ℓ.A)[end] == size(prev_ℓ.A)[end]  # Number of observations / batch size
    @assert size(prev_ℓ.A, 1) % size(ℓ.A, 1) == 0  # Pooling divides previous layer rows evenly
    @assert size(prev_ℓ.A, 2) % size(ℓ.A, 2) == 0  # Pooling divides previous layer cols evenly
end

# All
function checkdims(layers::Vector{NetLayer})
    for (ℓ, prev_ℓ) in zip(layers[2:end], layers[1:end-1])
        checkdims(ℓ, prev_ℓ)
    end
end
