#=
Train the model using gradient descent.

Training parameters
num_epochs - number of epochs
α - learning rate
μ - momentum
TODO - decay - function of epochs
TODO - early stopping

Store and display parameters
last_train_every - selected epoch intervals for last training batch
full_train_every - selected epoch intervals for full training set
val_every - selected epoch intervals for validation
display - choose whether to display progress
store - choose whether to store progress
=#

################################################################################
#
# Main Training Function
#
################################################################################

function train(net::NeuralNet, num_epochs::Integer, α::Real, μ::Real;
               nesterov::Bool = true, shuffle::Bool=false,
               last_train_every::Integer = 1,
               full_train_every::Integer = num_epochs,
               val_every::Integer = num_epochs)

    # Some counts
    num_obs = size(net.data.X_train)[end]
    num_batches = div(num_obs, net.batch_size)

    # Floating point data type
    F = net.dtype

    # Learning parameters - convert type
    α = F(α)
    μ = F(μ)

    # Nesterov parameters
    a = μ^2
    b = (μ + F(1)) * α

    # Choose momentum gradient descent or Nesterov accelerated gradient descent
    nesterov ? (update! = nag_update!) : (update! = mgd_update!)

    # Choose coffee break function - either regression or classification problem
    coffee_break = choose_coffee_break(net)

    # Permutation of observations - to be shuffled if selected
    perm = collect(1:num_obs)

    # Loop over epochs
    for epoch in 1:num_epochs

        # Shuffle permutation (if selected) and initialise batch unit range
        shuffle && shuffle!(perm)
        batch_range = 1:net.batch_size

        # Loop over mini batches
        for batch in 1:num_batches

            # Update inputs & target with next batch and increment range
            batch_ids = perm[batch_range]
            net.layers[1].A[:] = viewbatch(net.data.X_train, batch_ids)
            net.layers[end].y[:] = viewbatch(net.data.y_train, batch_ids)
            batch_range += net.batch_size

            # Forward propogation through each layer
            fwd_prop!(net)

            # Back propogation through each layer to calculate gradients
            bwd_prop!(net)

            # Update parameters using the gradients
            for ℓ in net.layers[2:end]
                update!(ℓ, μ, α, a, b)
            end

        end

        # Display and store progress updates
        # Last training loss
        last_batch_cost(net, epoch, last_train_every)
        # Coffee break - full training and validation loss
        coffee_break(net, epoch, num_epochs, full_train_every, val_every)

    end

    # Completed training - show final results
    show_completed(net)

    return
end


################################################################################
#
# Momentum Gradient Descent Update
#
################################################################################

function mgd_update!{F<:AbstractFloat}(W::Array{F}, W_vel::Array{F},
                                       ∇W::Array{F}, μ::F, α::F)
    for i in eachindex(W)
        @inbounds W_vel[i] = μ * W_vel[i] - α * ∇W[i]
        @inbounds W[i] += W_vel[i]
    end
end

function mgd_update!{F<:AbstractFloat}(ℓ::NetLayer, μ::F, α::F, a::F, b::F)
    mgd_update!(ℓ.b, ℓ.b_vel, ℓ.∇b, μ, α)
    mgd_update!(ℓ.W, ℓ.W_vel, ℓ.∇W, μ, α)
end

function mgd_update!{F<:AbstractFloat}(ℓ::WordEmbeddingLayer, μ::F, α::F, a::F, b::F)
    mgd_update!(ℓ.W, ℓ.W_vel, ℓ.∇W, μ, α)
end

function mgd_update!{F<:AbstractFloat}(ℓ::PoolLayer, μ::F, α::F, a::F, b::F)
end

################################################################################
#
# Nesterov's Accelerated Gradient Descent Update
#
################################################################################

function nag_update!{F<:AbstractFloat}(W::Array{F}, W_vel::Array{F},
                                       ∇W::Array{F}, μ::F, α::F, a::F, b::F)
    for i in eachindex(W)
        @inbounds W[i] += a * W_vel[i] - b * ∇W[i]
        @inbounds W_vel[i] = μ * W_vel[i] - α * ∇W[i]
    end
end

function nag_update!{F<:AbstractFloat}(ℓ::NetLayer, μ::F, α::F, a::F, b::F)
    nag_update!(ℓ.b, ℓ.b_vel, ℓ.∇b, μ, α, a, b)
    nag_update!(ℓ.W, ℓ.W_vel, ℓ.∇W, μ, α, a, b)
end

function nag_update!{F<:AbstractFloat}(ℓ::WordEmbeddingLayer, μ::F, α::F, a::F, b::F)
    nag_update!(ℓ.W, ℓ.W_vel, ℓ.∇W, μ, α, a, b)
end

function nag_update!{F<:AbstractFloat}(ℓ::PoolLayer, μ::F, α::F, a::F, b::F)
end


################################################################################
#
# Display and store progress
#
################################################################################

########################################
# Last training batch
########################################

# Time
t() = Dates.format(now(), "HH:MM:SS")

# Last batch training error - cost with regularisation
function last_batch_cost(net, epoch, last_train_every)

    if mod(epoch, last_train_every) == 0
        # Cost
        L = net.layers[end]
        J = L.cost(L.A, L.y) / net.batch_size

        # Add regularisation cost
        J += net.reg_cost(net)

        # Add sparsity cost
        J += sparse_cost(net)

        # Display
        print(t())
        @printf(" : Epoch %d, last batch training error (with regⁿ) - %.3f\n",
                epoch, J)

        # Store
        push!(net.last_train_loss, J)
    end

end

########################################
# Coffee Breaks
########################################

# Choos classification or regression
function choose_coffee_break(net)
    L = net.layers[end]
    typeof(L) <: LinOutputLayer ? coffee_break_r : coffee_break_c
end

# Coffee break for classification task - shows accuracy
function coffee_break_c(net, epoch, num_epochs, full_train_every, val_every)

    # Don't run coffee break after last epoch
    epoch == num_epochs && return

    # Both training and validation sets
    if mod(epoch, full_train_every) == 0 && mod(epoch, val_every) == 0
        @printf("\nCoffee break:\n")

        # Cost and display of training set
        J, acc = loss_and_accuracy(net, net.data.X_train, net.data.y_train)
        @printf("Training error (with regⁿ) - %.3f", J)
        @printf("  |  Training accuracy - %.1f\n", acc)
        push!(net.full_train_loss, J)

        # Cost and display of validation set
        J, acc = val_loss_and_accuracy(net, net.data.X_val, net.data.y_val)
        @printf("Validation error (without regⁿ) - %.3f", J)
        @printf("  |  Validation accuracy - %.1f\n\n", acc)
        push!(net.val_loss, J)

    # Training set only
    elseif mod(epoch, full_train_every) == 0
        @printf("\nCoffee break:\n")

        # Cost and display of training set
        J, acc = loss_and_accuracy(net, net.data.X_train, net.data.y_train)
        @printf("Training error (with regⁿ) - %.3f", J)
        @printf("  |  Training accuracy - %.1f\n\n", acc)
        push!(net.full_train_loss, J)

    # Validation set only
    elseif mod(epoch, val_every) == 0
        @printf("\nCoffee break:\n")

        # Cost and display of validation set
        J, acc = val_loss_and_accuracy(net, net.data.X_val, net.data.y_val)
        @printf("Validation error (without regⁿ) - %.3f", J)
        @printf("  |  Validation accuracy - %.1f\n\n", acc)
        push!(net.val_loss, J)

    end
end

# Coffee break for regression task - only loss (no accuracy)
function coffee_break_r(net, epoch, full_train_every, val_every)

    # Don't run coffee break after last epoch
    epoch == num_epochs && return

    # Both training and validation sets
    if mod(epoch, full_train_every) == 0 && mod(epoch, val_every) == 0
        @printf("\nCoffee break:\n")

        # Cost and display of training set
        J, acc = loss(net, net.data.X_train, net.data.y_train)
        @printf("Training error (with regⁿ) - %.3f\n", J)

        # Cost and display of validation set
        J, acc = val_loss_and_accuracy(net, net.data.X_val, net.data.y_val)
        @printf("Validation error (without regⁿ) - %.3f\n\n", J)

    # Training set only
    elseif mod(epoch, full_train_every) == 0
        @printf("\nCoffee break:\n")

        # Cost and display of training set
        J, acc = loss_and_accuracy(net, net.data.X_train, net.data.y_train)
        @printf("Training error (with regⁿ) - %.3f\n\n", J)

    # Validation set only
    elseif mod(epoch, val_every) == 0
        @printf("\nCoffee break:\n")

        # Cost and display of validation set
        J, acc = val_loss_and_accuracy(net, net.data.X_val, net.data.y_val)
        @printf("Validation error (without regⁿ) - %.3f\n\n", J)

    end
end


################################################################################
#
# Completed training - results
#
################################################################################

function show_completed(net)
    if typeof(net.layers[end]) <: LinOutputLayer
        show_completed_regression(net)
    else
        show_completed_classification(net)
    end
end

function show_completed_classification(net)

    @printf "\nCompleted Training:\n"

    if typeof(net.data) <: TData

        # Cost and display of training set
        J, acc = loss_and_accuracy(net, net.data.X_train, net.data.y_train)
        @printf("Training error (with regⁿ) - %.3f", J)
        @printf("  |  Training accuracy - %.1f\n", acc)
        push!(net.full_train_loss, J)

    else
        # Cost and display of training set
        J, acc = loss_and_accuracy(net, net.data.X_train, net.data.y_train)
        @printf("Training error (with regⁿ) - %.3f", J)
        @printf("  |  Training accuracy - %.1f\n", acc)
        push!(net.full_train_loss, J)

        # Cost and display of validation set
        J, acc = val_loss_and_accuracy(net, net.data.X_val, net.data.y_val)
        @printf("Validation error (without regⁿ) - %.3f", J)
        @printf("  |  Validation accuracy - %.1f\n", acc)
        push!(net.val_loss, J)
    end
end

function show_completed_regression(net)

    @printf "\nCompleted Training:\n"

    if typeof(net.data) <: TData

        # Cost and display of training set
        J = loss(net, net.data.X_train, net.data.y_train)
        @printf("Training error (with regⁿ) - %.3f\n", J)
        push!(net.full_train_loss, J)

    else
        # Cost and display of training set
        J = loss(net, net.data.X_train, net.data.y_train)
        @printf("Training error (with regⁿ) - %.3f\n", J)
        push!(net.full_train_loss, J)

        # Cost and display of validation set (without regularisation)
        J = val_loss(net, net.data.X_val, net.data.y_val)
        @printf("Validation error (without regⁿ) - %.3f\n", J)
        push!(net.val_loss, J)
    end
end
