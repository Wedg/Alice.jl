#=
This page defines the functions:
fwd_prop! - for feeding forward through the network
bwd_prop! - for feeding backward through the network to calculate gradients

Different methods are defined for different layer types and the network as
a whole.
=#

################################################################################
#
# Forward Propagate
#
################################################################################

########################################
# Layer -> OutputLayer
########################################

# LinearOutputLayer does not have an activation function
# Write weighted sum directly to A
function fwd_prop!(ℓ::LinOutputLayer, prev_ℓ::HDL)

    # A⁽²⁾ = b⁽²⁾ .+ W⁽²⁾ A⁽¹⁾
    A_mul_B!(ℓ.A, ℓ.W, prev_ℓ.Ar)
    broadcast!(+, ℓ.A, ℓ.b, ℓ.A)

    # No activation
end

function fwd_prop!(ℓ::LinOutputLayer, prev_ℓ::LDL)

    # Z⁽²⁾ = b⁽²⁾ .+ W⁽²⁾ A⁽¹⁾
    A_mul_B!(ℓ.A, ℓ.W, prev_ℓ.A)
    broadcast!(+, ℓ.A, ℓ.b, ℓ.A)

    # No activation
end

# LogisticOutputLayer
# fwd_prop! method covered by the general LD method defined below

# Softmax Layer activation function different to standard
# Vector `u` used for the max values of each row
function fwd_prop!(ℓ::SoftmaxOutputLayer, prev_ℓ::HDL)

    # Z⁽²⁾ = b⁽²⁾ .+ W⁽²⁾ A⁽¹⁾
    A_mul_B!(ℓ.Z, ℓ.W, prev_ℓ.Ar)
    broadcast!(+, ℓ.Z, ℓ.b, ℓ.Z)

    # A⁽²⁾ = f(Z⁽²⁾)
    ℓ.f!(ℓ.A, ℓ.Z, ℓ.u)
end

function fwd_prop!(ℓ::SoftmaxOutputLayer, prev_ℓ::LDL)

    # Z⁽²⁾ = b⁽²⁾ .+ W⁽²⁾ A⁽¹⁾
    A_mul_B!(ℓ.Z, ℓ.W, prev_ℓ.A)
    broadcast!(+, ℓ.Z, ℓ.b, ℓ.Z)

    # A⁽²⁾ = f(Z⁽²⁾)
    ℓ.f!(ℓ.A, ℓ.Z, ℓ.u)
end

########################################
# Layer -> Low Dimension Layer
########################################

# LD -> LD
function fwd_prop!(ℓ::LDL, prev_ℓ::LDL)

    # Z⁽²⁾ = b⁽²⁾ .+ W⁽²⁾ A⁽¹⁾
    A_mul_B!(ℓ.Z, ℓ.W, prev_ℓ.A)
    broadcast!(+, ℓ.Z, ℓ.b, ℓ.Z)

    # A⁽²⁾ = f(Z⁽²⁾)
    ℓ.f!(ℓ.A, ℓ.Z)
end

# HD -> LD
function fwd_prop!(ℓ::LDL, prev_ℓ::HDL)

    # Z⁽²⁾ = b⁽²⁾ .+ W⁽²⁾ A⁽¹⁾
    A_mul_B!(ℓ.Z, ℓ.W, prev_ℓ.Ar)
    broadcast!(+, ℓ.Z, ℓ.b, ℓ.Z)

    # A⁽²⁾ = f(Z⁽²⁾)
    ℓ.f!(ℓ.A, ℓ.Z)
end

########################################
# Layer -> Word Embedding Layer
########################################

# LD -> Word Embedding
function fwd_prop!(ℓ::WordEmbeddingLayer, prev_ℓ::LDL)

    # Embed word features
    embed!(ℓ.A, ℓ.W, prev_ℓ.A)
end

# HD -> Word Embedding
function fwd_prop!(ℓ::WordEmbeddingLayer, prev_ℓ::HDL)

    # Embed word features
    embed!(ℓ.A, ℓ.W, prev_ℓ.Ar)
end

########################################
# Layer -> Sparse Encoder Layer
########################################

# LD -> Sparse Encoder
function fwd_prop!(ℓ::SparseEncoderLayer, prev_ℓ::LDL)

    # Z⁽²⁾ = b⁽²⁾ .+ W⁽²⁾ A⁽¹⁾
    A_mul_B!(ℓ.Z, ℓ.W, prev_ℓ.A)
    broadcast!(+, ℓ.Z, ℓ.b, ℓ.Z)

    # A⁽²⁾ = f(Z⁽²⁾)
    ℓ.f!(ℓ.A, ℓ.Z)

    # Average of activations - "probability" of each neuron firing
    mean!(ℓ.p̂, ℓ.A)
end

# HD -> Sparse Encoder
function fwd_prop!(ℓ::SparseEncoderLayer, prev_ℓ::HDL)

    # Z⁽²⁾ = b⁽²⁾ .+ W⁽²⁾ A⁽¹⁾
    A_mul_B!(ℓ.Z, ℓ.W, prev_ℓ.A)
    broadcast!(+, ℓ.Z, ℓ.b, ℓ.Z)

    # A⁽²⁾ = f(Z⁽²⁾)
    ℓ.f!(ℓ.A, ℓ.Z)

    # Average of activations - "probability" of each neuron firing
    mean!(ℓ.p̂, ℓ.A)
end

########################################
# High Dimension Layer -> High Dimension Layer
########################################

# Convolution
function fwd_prop!(ℓ::ConvolutionLayer, prev_ℓ::HDL)

    # Z⁽²⁾ = b⁽²⁾ .+ W⁽²⁾ ∗ A⁽¹⁾
    conv!(ℓ.Z, ℓ.W, prev_ℓ.A)
    bcastbias!(ℓ.Z, ℓ.b)

    # A⁽²⁾ = f(Z⁽²⁾)
    ℓ.f!(ℓ.A, ℓ.Z)
end

# Mean Pooling
function fwd_prop!(ℓ::MeanPoolLayer, prev_ℓ::HDL)
    meanpool!(ℓ.A, prev_ℓ.A)
end

# Max Pooling
function fwd_prop!(ℓ::MaxPoolLayer, prev_ℓ::HDL)
    maxpool!(ℓ.A, prev_ℓ.A, ℓ.maxinds)
end

########################################
# Whole Network
########################################

function fwd_prop!(net::NeuralNet)

    # Feed forward each hidden layer and the output layer
    for (ℓ, prev_ℓ) in zip(net.layers[2:end], net.layers[1:end-1])
        fwd_prop!(ℓ, prev_ℓ)
    end
end


################################################################################
#
# Backward Propagate
#
################################################################################

########################################
# Layer -> OutputLayer
########################################

function bwd_prop!(ℓ::OutputLayer, prev_ℓ::HDL)

    # Delta of final layer
    # Δ⁽ᴸ⁾ = 1/m (A⁽ᴸ⁾ - Y)               (recall: A⁽ᴸ⁾ = H)
    ℓ.δ!(ℓ.Δ, ℓ.A, ℓ.y, ℓ.m)

    # Gradient of bias vector
    # ∇b⁽ᴸ⁾ = sum(Δ⁽ᴸ⁾, 2)
    sum!(ℓ.∇b, ℓ.Δ)

    # Gradient of weight matrix
    # ∇W⁽ᴸ⁾ = Δ⁽ᴸ⁾ A⁽ᴸ⁻¹⁾ᵀ
    A_mul_Bt!(ℓ.∇W, ℓ.Δ, prev_ℓ.Ar)
end

function bwd_prop!(ℓ::OutputLayer, prev_ℓ::LDL)

    # Delta of final layer
    # Δ⁽ᴸ⁾ = 1/m (A⁽ᴸ⁾ - Y)               (recall: A⁽ᴸ⁾ = H)
    ℓ.δ!(ℓ.Δ, ℓ.A, ℓ.y, ℓ.m)

    # Gradient of bias vector
    # ∇b⁽ᴸ⁾ = sum(Δ⁽ᴸ⁾, 2)
    sum!(ℓ.∇b, ℓ.Δ)

    # Gradient of weight matrix
    # ∇W⁽ᴸ⁾ = Δ⁽ᴸ⁾ A⁽ᴸ⁻¹⁾ᵀ
    A_mul_Bt!(ℓ.∇W, ℓ.Δ, prev_ℓ.A)
end

########################################
# Low -> Low -> Low
########################################

# Low -> Low -> Low
function bwd_prop!(ℓ::LDL, prev_ℓ::LDL, next_ℓ::LDL)

    # Δ⁽²⁾ = W⁽³⁾ᵀ Δ⁽³⁾ ⊙ f'(Z⁽²⁾)
    # Broken into 2 steps:
    # (a) - Δ⁽²⁾ := W⁽³⁾ᵀ Δ⁽³⁾
    # (b) - Δ⁽²⁾ := Δ⁽²⁾ ⊙ f'(Z⁽²⁾)
    At_mul_B!(ℓ.Δ, next_ℓ.W, next_ℓ.Δ)
    ℓ.∇f!(ℓ.Δ, ℓ.Z)

    # Gradient of bias vector
    # ∇b⁽²⁾ = sum(Δ⁽²⁾, 2)
    sum!(ℓ.∇b, ℓ.Δ)

    # Gradient of weight matrix
    # ∇W⁽²⁾ = Δ⁽²⁾ A⁽¹⁾
    A_mul_Bt!(ℓ.∇W, ℓ.Δ, prev_ℓ.A)
end

# Low -> Word embedding -> Low
function bwd_prop!(ℓ::WordEmbeddingLayer, prev_ℓ::LDL, next_ℓ::LDL)

    # Δ⁽²⁾ = W⁽³⁾ᵀ Δ⁽³⁾
    At_mul_B!(ℓ.Δ, next_ℓ.W, next_ℓ.Δ)

    # Gradient of weight matrix
    embed_grad!(ℓ.∇W, ℓ.Δ, prev_ℓ.A)
end

# Low -> Sparse -> Low
function bwd_prop!(ℓ::SparseEncoderLayer, prev_ℓ::LDL, next_ℓ::LDL)

    # Δ⁽²⁾ = (W⁽³⁾ᵀ Δ⁽³⁾ .+ spgrad⁽²⁾) ⊙ f'(Z⁽²⁾)
    # Broken into 3 steps:
    # (a) - Δ⁽²⁾ := W⁽³⁾ᵀ Δ⁽³⁾
    # (b) - Δ⁽²⁾ := Δ⁽²⁾ .+ spgrad⁽²⁾
    # (c) - Δ⁽²⁾ := Δ⁽²⁾ ⊙ f'(Z⁽²⁾)
    At_mul_B!(ℓ.Δ, next_ℓ.W, next_ℓ.Δ)
    ℓ.δ!(ℓ.Δ, ℓ.spgrad, ℓ.p̂, ℓ.p, ℓ.β, ℓ.m)
    ℓ.∇f!(ℓ.Δ, ℓ.Z)

    # Gradient of bias vector
    # ∇b⁽²⁾ = sum(Δ⁽²⁾, 2)
    sum!(ℓ.∇b, ℓ.Δ)

    # Gradient of weight matrix
    # ∇W⁽²⁾ = Δ⁽²⁾ A⁽¹⁾
    A_mul_Bt!(ℓ.∇W, ℓ.Δ, prev_ℓ.A)
end

########################################
# High -> Low -> Low
########################################

# High -> Low -> Low
function bwd_prop!(ℓ::LDL, prev_ℓ::HDL, next_ℓ::LDL)

    # Δ⁽²⁾ = W⁽³⁾ᵀ Δ⁽³⁾ ⊙ f'(Z⁽²⁾)
    # Broken into 2 steps:
    # (a) - Δ⁽²⁾ := W⁽³⁾ᵀ Δ⁽³⁾
    # (b) - Δ⁽²⁾ := Δ⁽²⁾ ⊙ f'(Z⁽²⁾)
    At_mul_B!(ℓ.Δ, next_ℓ.W, next_ℓ.Δ)
    ℓ.∇f!(ℓ.Δ, ℓ.Z)

    # Gradient of bias vector
    # ∇b⁽²⁾ = sum(Δ⁽²⁾, 2)
    sum!(ℓ.∇b, ℓ.Δ)

    # Gradient of weight matrix
    # ∇W⁽²⁾ = Δ⁽²⁾ A⁽¹⁾
    A_mul_Bt!(ℓ.∇W, ℓ.Δ, prev_ℓ.Ar)
end

# High -> Word embedding -> Low
function bwd_prop!(ℓ::WordEmbeddingLayer, prev_ℓ::HDL, next_ℓ::LDL)

    # Δ⁽²⁾ = W⁽³⁾ᵀ Δ⁽³⁾
    At_mul_B!(ℓ.Δ, next_ℓ.W, next_ℓ.Δ)

    # Gradient of weight matrix
    embed_grad!(ℓ.∇W, ℓ.Δ, prev_ℓ.Ar)
end

# High -> Sparse -> Low
function bwd_prop!(ℓ::SparseEncoderLayer, prev_ℓ::HDL, next_ℓ::LDL)

    # Δ⁽²⁾ = (W⁽³⁾ᵀ Δ⁽³⁾ .+ spgrad⁽²⁾) ⊙ f'(Z⁽²⁾)
    # Broken into 3 steps:
    # (a) - Δ⁽²⁾ := W⁽³⁾ᵀ Δ⁽³⁾
    # (b) - Δ⁽²⁾ := Δ⁽²⁾ .+ spgrad⁽²⁾
    # (c) - Δ⁽²⁾ := Δ⁽²⁾ ⊙ f'(Z⁽²⁾)
    At_mul_B!(ℓ.Δ, next_ℓ.W, next_ℓ.Δ)
    ℓ.δ!(ℓ.Δ, ℓ.spgrad, ℓ.p̂, ℓ.p, ℓ.β, ℓ.m)
    ℓ.∇f!(ℓ.Δ, ℓ.Z)

    # Gradient of bias vector
    # ∇b⁽²⁾ = sum(Δ⁽²⁾, 2)
    sum!(ℓ.∇b, ℓ.Δ)

    # Gradient of weight matrix
    # ∇W⁽²⁾ = Δ⁽²⁾ A⁽¹⁾
    A_mul_Bt!(ℓ.∇W, ℓ.Δ, prev_ℓ.Ar)
end

########################################
# High -> High -> Low
########################################

# High -> Convolution -> Low
function bwd_prop!(ℓ::ConvolutionLayer, prev_ℓ::HDL, next_ℓ::LDL)

    # Δ⁽²⁾ = W⁽³⁾ᵀ Δ⁽³⁾ ⊙ f'(Z⁽²⁾)
    # Broken into 2 steps:
    # (a) - Δ⁽²⁾ := W⁽³⁾ᵀ Δ⁽³⁾
    # (b) - Δ⁽²⁾ := Δ⁽²⁾ ⊙ f'(Z⁽²⁾)
    At_mul_B!(ℓ.Δr, next_ℓ.W, next_ℓ.Δ)
    ℓ.∇f!(ℓ.Δr, ℓ.Zr)

    # Gradient of bias vector
    # ∇bₖ⁽²⁾ = ∑ Δₖ⁽²⁾
    biasgrad!(ℓ.∇b, ℓ.Δ)

    # Gradient of weight matrices
    # ∇Wₖ⁽²⁾ = Δₖ⁽²⁾ ∗ A⁽¹⁾
    convgrad!(ℓ.∇W, ℓ.Δ, prev_ℓ.A)

    return
end

# High -> Pool -> Low
function bwd_prop!(ℓ::PoolLayer, prev_ℓ::HDL, next_ℓ::LDL)

    # Δ⁽²⁾ = W⁽³⁾ᵀ Δ⁽³⁾
    At_mul_B!(ℓ.Δr, next_ℓ.W, next_ℓ.Δ)
end

########################################
# High -> High -> High
########################################

# High -> Convolution -> MeanPool
function bwd_prop!(ℓ::ConvolutionLayer, prev_ℓ::HDL, next_ℓ::MeanPoolLayer)

    # Δ⁽²⁾ = upsample( Δ⁽³⁾ ) ⊙ f'(Z⁽²⁾)
    # Broken into 2 steps:
    # (a) - Δ⁽²⁾ := upsample( Δ⁽³⁾ )
    # (b) - Δ⁽²⁾ := Δ⁽²⁾ ⊙ f'(Z⁽²⁾)
    upsample_mean!(ℓ.Δ, next_ℓ.Δ)
    ℓ.∇f!(ℓ.Δr, ℓ.Zr)

    # Gradient of bias vector
    # ∇bₖ⁽²⁾ = ∑ Δₖ⁽²⁾
    biasgrad!(ℓ.∇b, ℓ.Δ)

    # Gradient of weight matrices
    # ∇Wₖ⁽²⁾ = Δₖ⁽²⁾ ∗ A⁽¹⁾
    convgrad!(ℓ.∇W, ℓ.Δ, prev_ℓ.A)
end

# High -> Convolution -> MaxPool
function bwd_prop!(ℓ::ConvolutionLayer, prev_ℓ::HDL, next_ℓ::MaxPoolLayer)

    # Δ⁽²⁾ = upsample( Δ⁽³⁾ ) ⊙ f'(Z⁽²⁾)
    # Broken into 2 steps:
    # (a) - Δ⁽²⁾ := upsample( Δ⁽³⁾ )
    # (b) - Δ⁽²⁾ := Δ⁽²⁾ ⊙ f'(Z⁽²⁾)
    upsample_max!(ℓ.Δ, next_ℓ.Δ, next_ℓ.maxinds)
    ℓ.∇f!(ℓ.Δr, ℓ.Zr)

    # Gradient of bias vector
    # ∇bₖ⁽²⁾ = ∑ Δₖ⁽²⁾
    biasgrad!(ℓ.∇b, ℓ.Δ)

    # Gradient of weight matrices
    # ∇Wₖ⁽²⁾ = Δₖ⁽²⁾ ∗ A⁽¹⁾
    convgrad!(ℓ.∇W, ℓ.Δ, prev_ℓ.A)
end

# High -> Pool -> High
function bwd_prop!(ℓ::PoolLayer, prev_ℓ::HDL, next_ℓ::ConvolutionLayer)

    # Delta from convolution layer
    # Δₖ⁽²⁾ = W⁽³⁾ ∗ Δₖ⁽³⁾
    conv!(ℓ.Δ, next_ℓ.W, next_ℓ.Δ)
end
# TODO - CHECK THIS !!!


########################################
# Whole Network
########################################

function bwd_prop!(net::NeuralNet)

    # Final layer
    ℓ = net.layers[end]
    prev_ℓ = net.layers[end-1]
    bwd_prop!(ℓ, prev_ℓ)

    # Hidden layers
    for (prev_ℓ, ℓ, next_ℓ) in zip(net.layers[end-2:-1:1], net.layers[end-1:-1:2], net.layers[end:-1:3])
        bwd_prop!(ℓ, prev_ℓ, next_ℓ)
    end

    # Regularisation
    net.reg_grad!(net)
end
