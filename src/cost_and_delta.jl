#=
Cost and final layer delta functions.
These are paired to each type of output layer.
=#

################################################################################
#
# LinearOutputLayer
#
################################################################################

########################################
# Quadratic / [Sum of Squares Diff]       # ℝ
########################################

# J = 1/2 ∑ᵐ ∑ᵏ (h - y)²
# h is a row vector and y is a column vector
function quadratic_cost{F<:AbstractFloat}(h::Array{F}, y::Vector{F})
    #length(y) == size(h, 2) || throw(DimensionMismatch("Inconsistent array sizes"))
    s = F(0)
    @simd for i in eachindex(h)
        @inbounds s += abs2(h[i] - y[i])
    end
    return s * F(0.5)
end

########################################
# Linear Delta                            # ℝ^ 1 x num_obs
########################################

# Final layer delta function for regression
function delta_linear!{F<:AbstractFloat}(Δ::Array{F}, h::Array{F}, y::Vector{F},
                                         num_obs::F)

    # Δ⁽ᴸ⁾ = 1/m (h - y)               (recall: A⁽ᴸ⁾ = h)
    @simd for j in eachindex(y)
        @inbounds Δ[j] = (h[j] - y[j]) / num_obs
    end
end


################################################################################
#
# LogisticOutputLayer
#
################################################################################

########################################
# Cross Entropy                           # ℝ
########################################

# J = ∑ᵐ - y * log(h) - (1-y) log(1-h)
# h is a row vector and y is a column vector
function cross_entropy_cost{F<:AbstractFloat, T<:Integer}(h::Array{F}, y::Vector{T})
    #length(y) == size(h, 2) || throw(DimensionMismatch("Inconsistent array sizes"))
    s = F(0)
    @simd for j in eachindex(h)
        @inbounds if y[j] == T(1)
                      s -= log(h[j])
                  else
                      s -= log1p(-h[j])
                  end
    end
    s
end

########################################
# Logistic Delta                          # ℝ^ 1 x num_obs
########################################

# Final layer delta function for regression
function delta_logistic!{F<:AbstractFloat, T<:Integer}(Δ::Array{F}, h::Array{F},
                                                       y::Vector{T}, num_obs::F)

    # Δ⁽ᴸ⁾ = h - y               (recall: A⁽ᴸ⁾ = h)
    @simd for j in eachindex(y)
        @inbounds Δ[j] = (h[j] - y[j]) / num_obs
    end
end


################################################################################
#
# SoftmaxOutputLayer
#
################################################################################

########################################
# Log Loss                                # ℝ
########################################

# J = ∑ᵐ ∑ᵏ - 1{y=k} log(h)
function log_loss_cost{F<:AbstractFloat, T<:Integer}(H::Array{F}, y::Vector{T})
    length(y) == size(H, 2) || throw(DimensionMismatch("Inconsistent array sizes"))
    s = F(0)
    for j in eachindex(y)
        @inbounds s -= log(H[y[j], j])
    end
    s
end

########################################
# Softmax Delta                          # ℝ^ num_classes x num_obs
########################################

# Final layer delta function for classification
function delta_softmax!{F<:AbstractFloat, T<:Integer}(Δ::Array{F}, H::Array{F},
                                                      y::Vector{T}, num_obs::F)

    # Δ⁽ᴸ⁾ = 1/m (H - Y)               (recall: A⁽ᴸ⁾ = H)
    copy!(Δ, H)
    for j in eachindex(y)
        @inbounds Δ[y[j], j] -= F(1)
    end

    # Divide by m
    broadcast!(/, Δ, Δ, num_obs)
end


################################################################################
#
# MultiLinearOutputLayer
#
################################################################################

########################################
# Quadratic                              # ℝ
########################################

# J = 1/2 ∑ᵐ ∑ᵏ (h - y)²
function quadratic_cost{F<:AbstractFloat}(H::Array{F}, Y::Array{F})
    size(H) == size(Y) || throw(DimensionMismatch("Inconsistent array sizes"))
    s = F(0)
    @simd for i in eachindex(H)
        @inbounds s += abs2(H[i] - Y[i])
    end
    return s * F(0.5)
end

########################################
# Multi Linear Delta                     # ℝ^ num_outputs x num_obs
########################################

# Final layer delta function for regression
function delta_linear!{F<:AbstractFloat}(Δ::Array{F}, H::Array{F}, Y::Array{F},
                                         num_obs::F)

    # Δ⁽ᴸ⁾ = 1/m (h - y)               (recall: A⁽ᴸ⁾ = h)

    @simd for j in eachindex(Y)
        @inbounds Δ[j] = (H[j] - Y[j]) / num_obs
    end
end
