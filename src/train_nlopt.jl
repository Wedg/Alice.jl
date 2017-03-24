################################################################################
#
# Full batch training using the nlopt.jl package
#
################################################################################

# Outer wrapper into NLopt - This is the function called by the user
function train_nlopt(net::NeuralNet; maxiter=100, algorithm = :LD_LBFGS)

    # Create vector of initial parameters from net
    params = zeros(Float64, num_params(net))
    unroll_param_vec!(params, net)

    # Train the neural net
    train_nlopt(net, params, maxiter=maxiter, algorithm=algorithm)
end

# Wrapper into NLopt
function train_nlopt(net::NeuralNet, params::Vector; maxiter=100, algorithm = :LD_LBFGS)

    # Create "short hand" for the cost function to be minimized
    function nlopt_cost_function(params, grad)
        cost_and_grad!(net, params, grad)
    end

    # Options
    options = Opt(algorithm, length(params))
    maxeval!(options, maxiter)

    # Objective
    min_objective!(options, nlopt_cost_function)

    # Train the neural net
    @printf "Training Loss:\n"
    cost, params, _ = optimize(options, params)
end

# Single function to return cost and gradient - input into convex optimiser
function cost_and_grad!{F<:AbstractFloat}(net::NeuralNet, params::Vector{F}, grad::Vector{F})

    # Copy params into net
    input_param_vec!(params, net)

    # Perform forward propogation through the layers
    fwd_prop!(net)

    # Cost / Loss / Error
    L = net.layers[end]
    J = L.cost(L.A, L.y) / L.m + net.reg_cost(net) + sparse_cost(net)
    @printf " -> %.3f" J

    # Perform back propogation to calculate gradients
    bwd_prop!(net)

    # Unroll and set gradient vector in-place
    unroll_grad_vec!(grad, net)

    # Return cost
    J
end

# Roll single vector of parameters into the network
function input_param_vec!(params, net)
    @assert length(params) == num_params(net)
    start = 1
    for ℓ in net.layers[2:end]

        # No parameters in pooling layers
        typeof(ℓ) <: PoolLayer && continue

        # Place bias
        idx = length(ℓ.b)
        stop = start + idx - 1
        ℓ.b[:] = view(params, start:stop)
        start += idx

        # Place weight
        idx = length(ℓ.W)
        stop = start + idx - 1
        ℓ.W[:] = view(params, start:stop)
        start += idx
    end
end

# Unroll parameters from the network into a single vector
function unroll_param_vec!(params, net)
    @assert length(params) == num_params(net)
    start = 1
    for ℓ in net.layers[2:end]

        # No parameters in pooling layers
        typeof(ℓ) <: PoolLayer && continue

        # Place bias
        idx = length(ℓ.b)
        stop = start + idx - 1
        params[start:stop] = ℓ.b[:]
        start += idx

        # Place weight
        idx = length(ℓ.W)
        stop = start + idx - 1
        params[start:stop] = ℓ.W[:]
        start += idx
    end
end

# Unroll gradient parameters from the network into a single vector
function unroll_grad_vec!(grad, net)
    @assert length(grad) == num_params(net)
    start = 1
    for ℓ in net.layers[2:end]

        # No parameters in pooling layers
        typeof(ℓ) <: PoolLayer && continue

        # Place bias
        idx = length(ℓ.∇b)
        stop = start + idx - 1
        grad[start:stop] = ℓ.∇b[:]
        start += idx

        # Place weight
        idx = length(ℓ.∇W)
        stop = start + idx - 1
        grad[start:stop] = ℓ.∇W[:]
        start += idx
    end
end
