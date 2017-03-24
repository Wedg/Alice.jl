#=

=#

# Convolution function
# 2D input (e.g. greyscale image) -> 2D convolved output for each image and patch
# Inputs:
# A - inputs from previous layer - dim x dim x num_obs - e.g. greyscale images
# W - weight patches - dim x dim x num_patches
# Output (updated in-place):
# Z - convolved features - dim x dim x num_patches x num_obs
function conv!{F<:AbstractFloat}(Z::Array{F,4}, W::Array{F,3}, A::Array{F,3})
    @threads for img in indices(Z, 4)
        for patch in indices(Z, 3)
            for z_col in indices(Z, 2)
                for z_row in indices(Z, 1)
                    tmp = F(0)
                    for p_col in indices(W, 2)
                        a_col = z_col + p_col - 1
                        for p_row in indices(W, 1)
                            a_row = z_row + p_row - 1
                            @inbounds @fastmath tmp += A[a_row, a_col, img] * W[p_row, p_col, patch]
                        end
                    end
                    @inbounds Z[z_row, z_col, patch, img] = tmp
                end
            end
        end
    end
end

# Convolution function
# 3D input (e.g. colour image) -> 2D convolved output for each image and patch
# Inputs:
# A - inputs from previous layer - dim x dim x channels x num_obs - e.g. colour images
# W - weight patches - dim x dim x channels x num_patches
# Output (updated in-place):
# Z - convolved features - dim x dim x num_patches x num_obs
function conv!{F<:AbstractFloat}(Z::Array{F,4}, W::Array{F,4}, A::Array{F,4})
    @threads for img in indices(Z, 4)
        for patch in indices(Z, 3)
            for z_col in indices(Z, 2)
                for z_row in indices(Z, 1)
                    tmp = F(0)
                    for chan in indices(W, 3)
                        for p_col in indices(W, 2)
                            a_col = z_col + p_col - 1
                            for p_row in indices(W, 1)
                                a_row = z_row + p_row - 1
                                @inbounds @fastmath tmp += A[a_row, a_col, chan, img] * W[p_row, p_col, chan, patch]
                            end
                        end
                    end
                    @inbounds Z[z_row, z_col, patch, img] = tmp
                end
            end
        end
    end
end

# Gradient of weights used in convolution
# 2D input (e.g. greyscale image) -> 2D convolved output for each image and patch
# Inputs:
# A - inputs from previous layer - dim x dim x num_obs - e.g. greyscale images
# Δ - delta i.e. ∂J/∂z of the convolved features - dim x dim x num_patches x num_obs
# Output (updated in-place):
# ∇W - gradients of the weight patches - dim x dim x num_patches
function convgrad!{F<:AbstractFloat}(∇W::Array{F,3}, Δ::Array{F,4}, A::Array{F,3})
    fill!(∇W, F(0))
    for img in indices(Δ, 4)
        for patch in indices(Δ, 3)
            for g_col in indices(∇W, 2)
                for g_row in indices(∇W, 1)
                    tmp = F(0)
                    for δ_col in indices(Δ, 2)
                        a_col = g_col + δ_col - 1
                        for δ_row in indices(Δ, 1)
                            a_row = g_row + δ_row - 1
                            @inbounds @fastmath tmp += A[a_row, a_col, img] * Δ[δ_row, δ_col, patch, img]
                        end
                    end
                    @inbounds ∇W[g_row, g_col, patch] += tmp
                end
            end
        end
    end
end

# Gradient of weights used in convolution
# 3D input (e.g. colour image) -> 2D convolved output for each image and patch
# Inputs:
# A - inputs from previous layer - dim x dim x channels x num_obs - e.g. colour images
# Δ - delta i.e. ∂J/∂z of the convolved features - dim x dim x num_patches x num_obs
# Output (updated in-place):
# ∇W - gradients of the weight patches - dim x dim x channels x num_patches
function convgrad!{F<:AbstractFloat}(∇W::Array{F,4}, Δ::Array{F,4}, A::Array{F,4})
    fill!(∇W, F(0))
    for img in indices(Δ, 4)
        for patch in indices(Δ, 3)
            for chan in indices(∇W, 3)
                for g_col in indices(∇W, 2)
                    for g_row in indices(∇W, 1)
                        tmp = F(0)
                        for δ_col in indices(Δ, 2)
                            a_col = g_col + δ_col - 1
                            for δ_row in indices(Δ, 1)
                                a_row = g_row + δ_row - 1
                                @inbounds @fastmath tmp += A[a_row, a_col, chan, img] * Δ[δ_row, δ_col, patch, img]
                            end
                        end
                        @inbounds ∇W[g_row, g_col, chan, patch] += tmp
                    end
                end
            end
        end
    end
end

# Broadcasts the bias vector throughout the convolved features
# Input:
# b - bias vector - num_patches
# Output (updated in-place):
# Z - convolved features - dim x dim x num_patches x num_obs
function bcastbias!{F<:AbstractFloat}(Z::AbstractArray{F, 4}, b::AbstractArray{F, 1})
    @threads for img in indices(Z, 4)
        for patch in indices(Z, 3)
            for col in indices(Z, 2)
                for row in indices(Z, 1)
                    @inbounds Z[row, col, patch, img] += b[patch]
                end
            end
        end
    end
end

# Gradient of bias used in convolution
# Input:
# Δ - delta i.e. ∂J/∂z of the convolved features - dim x dim x num_patches x num_obs
# Output (updated in-place):
# ∇b - gradients of the bias features - num_patches
function biasgrad!{F<:AbstractFloat}(∇b::AbstractArray{F, 1}, Δ::AbstractArray{F, 4})
    fill!(∇b, F(0))
    for img in indices(Δ, 4)
        for patch in indices(Δ, 3)
            for col in indices(Δ, 2)
                for row in indices(Δ, 1)
                    @inbounds ∇b[patch] += Δ[row, col, patch, img]
                end
            end
        end
    end
end


################################################################################
#
# Word Embedding
#
################################################################################

# Embedding of features in each word
function embed!{F<:AbstractFloat, T<:Integer}(A::Matrix{F}, W::Matrix{F},
                                              input::Matrix{T})
    num_words = size(input, 1)
    num_feats = size(W, 1)
    for col in indices(input, 2)
        for (r1, r2) in zip(1:num_words, 1:num_feats:(num_feats * (num_words - 1) + 1))
            col_w = input[r1, col]
            for (r3, r4) in zip(r2:(r2 + num_feats - 1), 1:num_feats)
                @inbounds A[r3, col] = W[r4, col_w]
            end
        end
    end
end

# Gradient of parameters in embedding layer
function embed_grad!{F<:AbstractFloat, T<:Integer}(∇W::Matrix{F}, Δ::Matrix{F},
                                                   input::Matrix{T})
    num_feats = size(∇W, 1)
    num_words = div(size(Δ, 1), num_feats)
    fill!(∇W, 0.)
    for m in indices(Δ, 2)
        for w in 1:num_words
            col = input[w, m]
            for (r1, r2) in zip(1:num_feats, ((w - 1) * num_feats + 1):(w * num_feats))
                @inbounds ∇W[r1, col] += Δ[r2, m]
            end
        end
    end
end
