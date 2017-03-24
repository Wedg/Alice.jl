#=
Regularisation of network.
Contains L1 and L2 regularisation as well as sparsity constraint for sparse
encoder.
=#


################################################################################
#
# L1
#
################################################################################

########################################
# L1 Cost                                ∈ ℝ
########################################

# For each layer weight matrix
# Jℓ₁ = λ ∑||Wᵢⱼ||    ∀ Wᵢⱼ
function ℓ1_reg_cost(net)
    Jᵣ = zero(net.dtype)
    for ℓ in net.layers[2:end]

        # No parameters in pooling layers
        typeof(ℓ) <: PoolLayer && continue

        # ∑||Wᵢⱼ||
        Jᵣ += sumabs(ℓ.W)
    end

    # λ ∑||Wᵢⱼ||
    Jᵣ *= net.λ
end

########################################
# L1 Gradient                            ∈ ℝ^ num_feats
########################################

#=
Note that an option to make the L1 gradient numerically stable is to make the
function √(x²+ϵ). This would make the function smooth around x=0.
=#
# ∂Jℓ₂/∂Wᵢⱼ = λ sign(Wᵢⱼ)    ∀ Wᵢⱼ
# Added to the gradient in-place
function ℓ1_reg_grad!(net)
    for ℓ in net.layers[2:end]

        # No parameters in pooling layers
        typeof(ℓ) <: PoolLayer && continue

        # ∇W := λ sign(W) + ∇W
        BLAS.axpy!(net.λ, sign(ℓ.W), ℓ.∇W)
    end
end


################################################################################
#
# L2 (Weight Decay)
#
################################################################################

########################################
# L2 Cost                                ∈ ℝ
########################################

# Jℓ₂ = (λ/2) ∑||Wᵢⱼ||²    ∀ Wᵢⱼ
function ℓ2_reg_cost(net)
    Jᵣ = zero(net.dtype)
    for ℓ in net.layers[2:end]

        # No parameters in pooling layers
        typeof(ℓ) <: PoolLayer && continue

        # ∑||Wᵢⱼ||²
        Jᵣ += sumabs2(ℓ.W)
    end

    # (λ/2) ∑||Wᵢⱼ||²
    Jᵣ *= net.λ * net.dtype(0.5)
end

########################################
# L2 Gradient                            ∈ ℝ^ num_feats
########################################

# ∂Jℓ₂/∂Wᵢⱼ = λ Wᵢⱼ    ∀ Wᵢⱼ
# Added to the gradient in-place
function ℓ2_reg_grad!(net)
    for ℓ in net.layers[2:end]

        # No parameters in pooling layers
        typeof(ℓ) <: PoolLayer && continue

        # ∇W := λ W + ∇W
        BLAS.axpy!(net.λ, ℓ.W, ℓ.∇W)
    end
end


################################################################################
#
# L1 and L2 Dictionaries (for user selections through api)
#
################################################################################

reg_cost_dict = Dict{Symbol, Function}(:L1 => ℓ1_reg_cost, :L2 => ℓ2_reg_cost)
reg_grad_dict = Dict{Symbol, Function}(:L1 => ℓ1_reg_grad!, :L2 => ℓ2_reg_grad!)



################################################################################
#
# Sparsity (KL Divergence of Bernoulli means)
#
################################################################################

########################################
# KL Divergence                          ∈ ℝ
########################################

# Sum of KL Divergences between a Bernoulli random variable with mean p and a
# vector of Bernoulli random variables with mean p̂
function kl_div_sum{F<:AbstractFloat}(p̂::Vector{F}, p::F)
    s = F(0)
    for j in eachindex(p̂)
        @inbounds s += (p * log(p / p̂[j]) +
                        (F(1) - p) * log((F(1) - p) / (F(1) - p̂[j])))
    end
    return s
end

# Sparsity cost function for the net
function sparse_cost(net)
    J = zero(net.dtype)
    for ℓ in net.layers[2:end-1]

        # Only for sparse encoder layers
        !(typeof(ℓ) <: SparseEncoderLayer) && continue

        # β ∑ˢ KL(p || p̂)
        J += ℓ.β * kl_div_sum(ℓ.p̂, ℓ.p)
    end
    return J
end

########################################
# Sparsity Gradient                      ∈ ℝ^ num_feats
########################################

function delta_sparse!{F<:AbstractFloat}(Δ::Array{F}, spgrad::Vector{F},
                                         p̂::Vector{F}, p::F, β::F, m::F)
    for j in eachindex(p̂)
        @inbounds spgrad[j] = (- p / p̂[j] + (F(1) - p) / (F(1) - p̂[j])) * β / m
    end
    broadcast!(+, Δ, spgrad, Δ)
end
