#=

=#

################################################################################
#
# Create tiny layers and tiny network
#
################################################################################

########################################
# Input Layer
########################################

# Slices of Arrays
tinyslice{R<:Real}(X::Array{R,1}, ld_out, hd_dim, num_channels, num_obs) = X[1:num_obs]
tinyslice{R<:Real}(X::Array{R,2}, ld_out, hd_dim, num_channels, num_obs) = X[1:ld_out, 1:num_obs]
tinyslice{R<:Real}(X::Array{R,3}, ld_out, hd_dim, num_channels, num_obs) = X[1:hd_dim, 1:hd_dim, 1:num_obs]
tinyslice{R<:Real}(X::Array{R,4}, ld_out, hd_dim, num_channels, num_obs) = X[1:hd_dim, 1:hd_dim, 1:num_channels, 1:num_obs]

# InputLayer
function tinylayer(ℓ::InputLayer, ld_out, hd_dim,
                   num_channels, batch_size)
    X = tinyslice(ℓ.A, ld_out, hd_dim, num_channels, batch_size)
    rows = prod(size(X)[1:end-1])
    cols = size(X)[end]
    InputLayer(
    X, reshape(X, rows, cols)
    )
end

########################################
# Hidden Layers
########################################

# FullyConnectedLayer
function tinylayer(ℓ::FullyConnectedLayer, ld_in, ld_out, hd_dim, num_channels,
                                         patch_dim, num_patches, pooling_stride,
                                         num_outputs, num_classes, batch_size)

    # Datatype
    F = eltype(ℓ.A)

    # Take small slices out of ℓ
    b = ℓ.b[1:ld_out]
    W = ℓ.W[1:ld_out, 1:ld_in]

    # Create layer
    FullyConnectedLayer(
    b, W,
    zeros(F, ld_out, batch_size),
    zeros(F, ld_out, batch_size),
    zeros(F, ld_out, batch_size),
    zeros(b), zeros(W), zeros(b), zeros(W),
    ℓ.f!, ℓ.∇f!
    )
end

# WordEmbeddingLayer
function tinylayer(ℓ::WordEmbeddingLayer, ld_in, ld_out, hd_dim, num_channels,
                                         patch_dim, num_patches, pooling_stride,
                                         num_outputs, num_classes, batch_size)

    # Datatype
    F = eltype(ℓ.A)

    # Small constants
    vocab_size = 20
    num_words = 2
    num_feats = ld_out / num_words

    # Take small slices out of ℓ
    W = ℓ.W[1:ld_out, 1:ld_in] # TODO

    # Create layer
    _, W = init_uniform(F, num_feats, vocab_size)
    A = zeros(F, num_words * num_feats, batch_size)
    Δ = zeros(F, num_words * num_feats, batch_size)
    WordEmbeddingLayer(
    W,
    A, Δ,
    zeros(W), zeros(W)
    )
end

# SparseEncoderLayer
function tinylayer(ℓ::SparseEncoderLayer, ld_in, ld_out, hd_dim, num_channels,
                                         patch_dim, num_patches, pooling_stride,
                                         num_outputs, num_classes, batch_size)

    # Datatype
    F = eltype(ℓ.A)

    # Take small slices out of ℓ
    b = ℓ.b[1:ld_out]
    W = ℓ.W[1:ld_out, 1:ld_in]

    # Create layer
    SparseEncoderLayer(
    b, W,
    zeros(F, ld_out, batch_size),
    zeros(F, ld_out, batch_size),
    zeros(F, ld_out, batch_size),
    zeros(b), zeros(W), zeros(b), zeros(W),
    zeros(b), zeros(b), ℓ.p, ℓ.β,
    ℓ.f!, ℓ.∇f!, ℓ.δ!, F(batch_size)
    )
end

# ConvolutionLayer
function tinylayer(ℓ::ConvolutionLayer, ld_in, ld_out, hd_dim, num_channels,
                                        patch_dim, num_patches, pooling_stride,
                                        num_outputs, num_classes, batch_size)

    # Datatype
    F = eltype(ℓ.A)

    # Patch slice indices
    num_patch_dims = length(size(ℓ.W))
    if num_patch_dims == 3
        patch_ids = (1:patch_dim, 1:patch_dim, 1:num_patches)
    elseif num_patch_dims == 4
        patch_ids = (1:patch_dim, 1:patch_dim, 1:num_channels, 1:num_patches)
    end

    # Take small slices out of b and W
    b = ℓ.b[patch_ids[end]]
    W = ℓ.W[patch_ids...]

    # Convolution dims
    num_conv_dims = length(size(ℓ.A))
    if num_conv_dims == 4
        conv_dims = (hd_dim, hd_dim, num_patches, batch_size)
    elseif num_conv_dims == 5
        conv_dims = (hd_dim, hd_dim, num_channels, num_patches, batch_size)
    end

    # Create Z, A and Δ
    Z = zeros(F, conv_dims)
    A = zeros(F, conv_dims)
    Δ = zeros(F, conv_dims)

    # For reshaping
    rows = prod(conv_dims[1:end-1])

    # Create layer
    ConvolutionLayer(
    b, W,
    Z, reshape(Z, rows, batch_size),
    A, reshape(A, rows, batch_size),
    Δ, reshape(Δ, rows, batch_size),
    zeros(b), zeros(W), zeros(b), zeros(W),
    ℓ.f!, ℓ.∇f!
    )
end

# MeanPoolLayer
function tinylayer(ℓ::MeanPoolLayer, ld_in, ld_out, hd_dim, num_channels,
                                     patch_dim, num_patches, pooling_stride,
                                     num_outputs, num_classes, batch_size)

    # Datatype
    F = eltype(ℓ.A)

    # Pool dims
    num_dims = length(size(ℓ.A))
    # TODO - Can there be 5 dims??
    if num_dims == 4
        dims = (hd_dim, hd_dim, num_patches, batch_size)
    elseif num_dims == 5
        dims = (hd_dim, hd_dim, num_channels, num_patches, batch_size)
    end

    # Create empty A and Δ
    A = zeros(dims)
    Δ = zeros(dims)

    # For reshaping
    rows = prod(dims[1:end-1])

    # Create layer
    MeanPoolLayer(
    A, reshape(A, rows, batch_size),
    Δ, reshape(Δ, rows, batch_size)
    )
end

# MaxPoolLayer
function tinylayer(ℓ::MaxPoolLayer, ld_in, ld_out, hd_dim, num_channels,
                                    patch_dim, num_patches, pooling_stride,
                                    num_outputs, num_classes, batch_size)

    # Datatype
    F = eltype(ℓ.A)

    # Pool dims
    num_dims = length(size(ℓ.A))
    prev_dim = hd_dim * pooling_stride
    # TODO - Can there be 5 dims??
    if num_dims == 4
        dims = (hd_dim, hd_dim, num_patches, batch_size)
        dims_in = (prev_dim, prev_dim, num_patches, batch_size)
    elseif num_dims == 5
        dims = (hd_dim, hd_dim, num_channels, num_patches, batch_size)
        dims_in = (prev_dim, prev_dim, num_channels, num_patches, batch_size)
    end

    # Create empty A and Δ
    A = zeros(dims)
    Δ = zeros(dims)

    # For reshaping
    rows = prod(dims[1:end-1])

    # Create layer
    MaxPoolLayer(
    A, reshape(A, rows, batch_size),
    Δ, reshape(Δ, rows, batch_size),
    zeros(Int, dims_in...)
    )
end


########################################
# Output Layers
########################################

# LinearOutputLayer
function tinylayer(ℓ::LinearOutputLayer, ld_in, ld_out, hd_dim, num_channels,
                                         patch_dim, num_patches, pooling_stride,
                                         num_outputs, num_classes, batch_size)

    # Datatypes
    F = eltype(ℓ.A)

    # Take small slices out of ℓ weights and target
    b = ℓ.b[:]
    W = ℓ.W[:, 1:ld_in]
    y = ℓ.y[1:batch_size]

    # Create layer
    LinearOutputLayer(
    b, W,
    zeros(F, 1, batch_size), zeros(F, 1, batch_size),
    zeros(b), zeros(W), zeros(b), zeros(W),
    ℓ.δ!, y, ℓ.cost, F(batch_size)
    )
end

# MultiLinearOutputLayer
function tinylayer(ℓ::MultiLinearOutputLayer, ld_in, ld_out, hd_dim, num_channels,
                                         patch_dim, num_patches, pooling_stride,
                                         num_outputs, num_classes, batch_size)

    # Datatype
    F = eltype(ℓ.A)

    # Take small slices out of ℓ weights and target
    b = ℓ.b[1:num_outputs]
    W = ℓ.W[1:num_outputs, 1:ld_in]
    y = ℓ.y[1:num_outputs, 1:batch_size]

    # Create layer
    MultiLinearOutputLayer(
    b, W,
    zeros(F, num_outputs, batch_size), zeros(F, num_outputs, batch_size),
    zeros(b), zeros(W), zeros(b), zeros(W),
    ℓ.δ!, y, ℓ.cost, F(batch_size)
    )
end

# LogisticOutputLayer
function tinylayer(ℓ::LogisticOutputLayer, ld_in, ld_out, hd_dim, num_channels,
                                         patch_dim, num_patches, pooling_stride,
                                         num_outputs, num_classes, batch_size)

    # Datatype
    F = eltype(ℓ.A)

    # Take small slices out of ℓ weights and target
    b = ℓ.b[:]
    W = ℓ.W[:, 1:ld_in]
    y = rand(0:1, batch_size)   #ℓ.y[1:batch_size]

    # Create layer
    LogisticOutputLayer(
    b, W,
    zeros(F, 1, batch_size), zeros(F, 1, batch_size), zeros(F, 1, batch_size),
    zeros(b), zeros(W), zeros(b), zeros(W),
    ℓ.f!, ℓ.δ!, y, ℓ.cost, F(batch_size)
    )
end

# SoftmaxOutputLayer
function tinylayer(ℓ::SoftmaxOutputLayer, ld_in, ld_out, hd_dim, num_channels,
                                         patch_dim, num_patches, pooling_stride,
                                         num_outputs, num_classes, batch_size)

    # Datatype
    F = eltype(ℓ.A)

    # Take small slices out of ℓ weights and target
    b = ℓ.b[1:num_classes]
    W = ℓ.W[1:num_classes, 1:ld_in]
    y = rand(1:num_classes, batch_size)   #ℓ.y[1:batch_size]

    # Create layer
    SoftmaxOutputLayer(
    b, W,
    zeros(F, num_classes, batch_size), zeros(F, 1, batch_size),
    zeros(F, num_classes, batch_size), zeros(F, num_classes, batch_size),
    zeros(b), zeros(W), zeros(b), zeros(W),
    ℓ.f!, ℓ.δ!, y, ℓ.cost, F(batch_size)
    )
end

########################################
# Network
########################################

# Network
function tinynet(net::NeuralNet)

    # Start values that gett changed as layers are built
    ld_in = 9
    ld_out = 9
    hd_dim = 3

    # Constant values - not changed
    num_channels = 3
    patch_dim = 3
    num_patches = 5
    pooling_stride = 2
    num_outputs = min(9, size(net.data.X_train, 1))
    num_classes = min(5, length(unique(net.data.y_train)))
    batch_size = min(40, net.batch_size)

    # Create tiny hidden layers from last to first
    tinylayers = NetLayer[]
    for (ℓ, prev_ℓ) in zip(net.layers[end:-1:2], net.layers[(end-1):-1:1])

        # Update ld_in var if chaining previous high dim with this low dim layer
        if typeof(ℓ) <: LDL && typeof(prev_ℓ) <: HDL
            prev_ℓ_dims = length(size(prev_ℓ.A))
            if prev_ℓ_dims == 4
                ld_in = hd_dim * hd_dim * num_patches
            elseif prev_ℓ_dims == 5
                ld_in = hd_dim * hd_dim * num_channels * num_patches
            end
        end

        # Create tiny layer and add to vector
        push!(tinylayers, tinylayer(ℓ, ld_in, ld_out, hd_dim, num_channels,
                                    patch_dim, num_patches, pooling_stride,
                                    num_outputs, num_classes, batch_size))

        # Update high dimension var for use in previous high dimension layer
        if typeof(ℓ) <: PoolLayer
            hd_dim = hd_dim * pooling_stride
        elseif typeof(ℓ) <: ConvolutionLayer
            hd_dim = hd_dim + patch_dim - 1
        end
    end

    # Create tiny input layer
    ℓ = net.layers[1]
    push!(tinylayers, tinylayer(ℓ, ld_out, hd_dim, num_channels, batch_size))

    # Reverse - i.e. flip to ordering from input o output
    tinylayers = reverse(tinylayers)

    # Check that the layer dimensions allow chaining
    checkdims(tinylayers)

    # Create tiny data container
    tdata = Data(tinylayers[1].A, tinylayers[end].y)

    # Create tiny network
    F = net.dtype
    NeuralNet(F, tdata, tinylayers, net.reg_cost, net.reg_grad!, net.λ,
              batch_size, Array(F, 0), Array(F, 0), Array(F, 0))
end

################################################################################
#
# Display functions
#
################################################################################

# Display detail - show difference for every parameter
function display_grad_check(numeric_grad, grad)
    # Display the difference between numeric and analytic gradient for each parameter
    println()
    @printf "  Numeric Gradient     Analytic Gradient         Difference\n"
    for i=1:length(numeric_grad)
        @printf "%18e %21e %18e \n" numeric_grad[i] grad[i] numeric_grad[i] - grad[i]
    end
    println()
end

# Display summary - show relative difference
function display_grad_reldiff(numeric_grad, grad)
    # Display the norm of the difference between two solutions.
    # The diff should be less than 1e-7
    diff = norm(view(grad, :) - view(numeric_grad, :))
    denom = max(norm(view(grad, :)), norm(view(numeric_grad, :)))
    rel_diff = diff == 0.0 ? 0.0 : diff / denom
    println("  $rel_diff")
    println()
end


################################################################################
#
# Check the gradients
#
################################################################################

function check_gradients(bignet::NeuralNet; detail=false)

    # Craete tinynet
    net = tinynet(bignet)

    # Calculate gradients analytically
    fwd_prop!(net)
    bwd_prop!(net)

    # Size of perturbation
    ϵ = 1e-4

    # Output Layer
    L = net.layers[end]

    # Print
    println("Check gradients of a tiny network with similar architecture:\n")
    ℓ = net.layers[1]
    str = "Layer 1, " * string(typeof(ℓ))[7:end] * ", Dimensions - $(size(ℓ.A)) :\n\n"
    print_with_color(:blue, str)

    for (n, ℓ) in enumerate(net.layers[2:end])

        # Print results
        print_with_color(:blue, "Layer $(n+1), " * displaystr(ℓ) * " :\n\n")

        # Skip pooling layers
        if typeof(ℓ) <: PoolLayer
            println("  No parameters in pooling layer.\n")
            continue
        end

        # Initialise arrays
        numeric_grad_b = zeros(ℓ.b)
        numeric_grad_W = zeros(ℓ.W)

        # Estimate gradient for each bias parameter separately
        for i in eachindex(ℓ.b)

            # Store value of parameter
            tmp = ℓ.b[i]

            # Perturb the parameter and calculate the cost
            ℓ.b[i] += ϵ
            fwd_prop!(net)
            cost1 = L.cost(L.A, L.y) / L.m +
                    sparse_cost(net)

            ℓ.b[i] -= 2ϵ
            fwd_prop!(net)
            cost2 = L.cost(L.A, L.y) / L.m +
                    sparse_cost(net)

            # Calculate numeric gradient using central difference
            numeric_grad_b[i] = (cost1 - cost2) ./ (2ϵ)

            # Reset the parameter
            ℓ.b[i] = tmp
        end

        # Estimate gradient for each weight parameter separately
        for i in eachindex(ℓ.W)

            # Store value of parameter
            tmp = ℓ.W[i]

            # Perturb the parameter and calculate the cost
            ℓ.W[i] += ϵ
            fwd_prop!(net)
            cost1 = L.cost(L.A, L.y) / L.m +
                    net.reg_cost(net) +
                    sparse_cost(net)

            ℓ.W[i] -= 2ϵ
            fwd_prop!(net)
            cost2 = L.cost(L.A, L.y) / L.m +
                    net.reg_cost(net) +
                    sparse_cost(net)

            # Calculate numeric gradient using central difference
            numeric_grad_W[i] = (cost1 - cost2) ./ (2ϵ)

            # Reset the parameter
            ℓ.W[i] = tmp
        end

        # Detail
        if detail
            println("  Bias parameters :")
            display_grad_check(numeric_grad_b, ℓ.∇b)
            println("  Weight parameters :")
            display_grad_check(numeric_grad_W, ℓ.∇W)
        end

        # Summary
        println("  Bias parameters relative difference :")
        display_grad_reldiff(numeric_grad_b, ℓ.∇b)
        println("  Weight parameters relative difference :")
        display_grad_reldiff(numeric_grad_W, ℓ.∇W)

    end
end
