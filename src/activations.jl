#=
Hidden layer activation functions and gradients.
Options are:
:logistic, :tanh, :rectifier
=#

################################################################################
#
# Activation Functions and Derivatives
#
################################################################################

########################################
# Logistic
########################################

# logistic function of element z
# f(z) = 1 / (1 + exp(-z))
function logistic{F<:AbstractFloat}(z::F)
    one(z) / (one(z) + exp(-z))
end

# logistic function of array Z
# replacing array A in-place
# A[i] = f(Z[i]) = 1 / (1 + exp(-Z[i])) ∀ i
function logistic!{F<:AbstractFloat}(A::Array{F}, Z::Array{F})
    size(A) == size(Z) || throw(DimensionMismatch("Inconsistent array sizes"))
    @threads for i in eachindex(Z)
        @inbounds A[i] = logistic(Z[i])
    end
end

# logistic function derivative of element z
# df/dz = f(z) (1 - f(z))
function logistic_deriv{F<:AbstractFloat}(z::F)
    s = logistic(z)
    s * (F(1) - s)
end

# logistic function derivative of array Z
# ∂f/∂Z[i] = f(Z[i]) (1 - f(Z[i]))
# replacing dest in-place with Hademaard product of dest and derivative i.e.
# dest[i] = dest[i] * ∂f/∂Z[i]
function logistic_deriv!{F<:AbstractFloat}(dest::Array{F}, Z::Array{F})
    size(dest) == size(Z) || throw(DimensionMismatch("Inconsistent array sizes"))
    @threads for i in eachindex(Z)
        @inbounds dest[i] *= logistic_deriv(Z[i])
    end
end

########################################
# Tanh
########################################

# Elementwise tanh function of array Z
# replacing array A in-place
function tanh!{F<:AbstractFloat}(A::Array{F}, Z::Array{F})
    size(A) == size(Z) || throw(DimensionMismatch("Inconsistent array sizes"))
    @threads for i in eachindex(Z)
        @inbounds A[i] = tanh(Z[i])
    end
end

# Elementwise tanh function derivative of array Z
# ∂f/∂Z[i] = 1 - f(Z[i])^2
# replacing dest in-place with Hademaard product of dest and derivative i.e.
# dest[i] = dest[i] * ∂f/∂Z[i]
function tanh_deriv!{F<:AbstractFloat}(dest::Array{F}, Z::Array{F})
    size(dest) == size(Z) || throw(DimensionMismatch("Inconsistent array sizes"))
    @threads for i in eachindex(Z)
        @inbounds dest[i] *= (F(1) - tanh(Z[i])^2)
    end
end

########################################
# Rectified linear units
########################################

# Elementwise rectifier function of array Z
# replacing array A in-place
function relu!{F<:AbstractFloat}(A::Array{F}, Z::Array{F})
    size(A) == size(Z) || throw(DimensionMismatch("Inconsistent array sizes"))
    @threads for i in eachindex(Z)
        @inbounds A[i] = max(F(0), Z[i])
    end
end

# Elementwise rectifier function derivative of array Z
# ∂f/∂Z[i] = 0 (if Z[i] <= 0) or 1 (if Z[i] > 0)
# replacing dest in-place with Hademaard product of dest and derivative i.e.
# dest[i] = dest[i] * ∂f/∂Z[i]
function relu_deriv!{F<:AbstractFloat}(dest::Array{F}, Z::Array{F})
    size(dest) == size(Z) || throw(DimensionMismatch("Inconsistent array sizes"))
    for i in eachindex(Z)
        @inbounds (Z[i] <= F(0)) && (dest[i] = F(0))
        #@inbounds dest[i] *= ifelse(Z[i] > F(0), F(1), F(0))
    end
end

########################################
# Softmax
########################################

# Softmax function of matrix Z replacing matrix A in-place
# Applies function to each column of Z i.e. each column will sum to 1
function softmax!{F<:AbstractFloat}(A::Matrix{F}, Z::Matrix{F}, u::Matrix{F})
    #size(A) == size(Z) || throw(DimensionMismatch("Inconsistent array sizes"))
    maximum!(u, Z)
    for j in indices(Z, 2)
        s = F(0)
        for i in indices(Z, 1)
            @inbounds s += (A[i, j] = exp(Z[i, j] - u[j]))
        end
        for i in indices(Z, 1)
            @inbounds A[i, j] /= s
        end
    end
end

################################################################################
#
# Activation Dictionaries
#
################################################################################

activation_dict = Dict{Symbol, Function}(:logistic => logistic!,
                                         :tanh => tanh!,
                                         :relu => relu!)

activation_deriv_dict = Dict{Symbol, Function}(:logistic => logistic_deriv!,
                                               :tanh => tanh_deriv!,
                                               :relu => relu_deriv!)
