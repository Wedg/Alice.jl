#=
Two pooling layers - MeanPool and MaxPool
=#

########################################
# MeanPool
########################################

#=
Mean pooling function
 2D input -> 2D pooled output for each patch of each image
 Input:
 A - inputs from previous layer - dim x dim x num_patches x num_obs
 Output (updated in-place):
 Z - pooled features - pool_dim x pool_dim x num_patches x num_obs
=#
function meanpool!{F<:AbstractFloat}(Z::Array{F,4}, A::Array{F,4})

    stride = div(size(A, 1), size(Z, 1))
    t = stride * stride

    for img in indices(Z, 4)
        for patch in indices(Z, 3)
            col_start = 1
            col_end = stride
            for pcol in indices(Z, 2)
                row_start = 1
                row_end = stride
                for prow in indices(Z, 1)
                    s = F(0)
                    for incol = col_start:col_end
                        for inrow = row_start:row_end
                            @inbounds s += A[inrow, incol, patch, img]
                        end
                    end
                    @inbounds Z[prow, pcol, patch, img] = s / t
                    row_start += stride
                    row_end += stride
                end
                col_start += stride
                col_end += stride
            end
        end
    end
end

#=
Upsample function for mean pooling - uniformly distributes the error for a
single pooling unit among the units which feed into it in the previous layer
 Input:
 Δpool - delta of pool layer
       - pool_dim x pool_dim x num_patches x num_obs
 Output (updated in-place):
 Δconv - delta of previous layer (usually a convolution layer)
       - conv_dim x conv_dim x num_patches x num_obs
=#
function upsample_mean!{F<:AbstractFloat}(Δconv::Array{F,4}, Δpool::Array{F,4})

    stride = div(size(Δconv, 1), size(Δpool, 1))
    m = inv(stride^2)

    for img in indices(Δpool, 4)
        for patch in indices(Δpool, 3)
            start_col = 1
            stop_col = stride
            for pcol in indices(Δpool, 2)
                start_row = 1
                stop_row = stride
                for prow in indices(Δpool, 1)
                    @inbounds tmp = Δpool[prow, pcol, patch, img] * m
                    for col in start_col:stop_col
                        for row in start_row:stop_row
                            @inbounds Δconv[row, col, patch, img] = tmp
                        end
                    end
                    start_row += stride
                    stop_row += stride
                end
                start_col += stride
                stop_col += stride
            end
        end
    end
end


########################################
# MaxPool
########################################

#=
Max pooling function
 2D input -> 2D pooled output for each patch of each image
 Inputs:
 A - inputs from previous layer - dim x dim x num_patches x num_obs
 Output (updated in-place):
 Z - pooled features - pool_dim x pool_dim x num_patches x num_obs
 maxinds - linear index of the max value from each patch
=#
function maxpool!{F<:AbstractFloat, T<:Integer}(Z::Array{F,4}, A::Array{F,4},
                                                maxinds::Array{T,4})

    dims = size(A)
    stride = div(size(A, 1), size(Z, 1))

    for img in indices(Z, 4)
        for patch in indices(Z, 3)
            col_start = 1
            col_end = stride
            for pcol in indices(Z, 2)
                row_start = 1
                row_end = stride
                for prow in indices(Z, 1)
                    maxval = F(-Inf)
                    maxind = T(0)
                    for incol = col_start:col_end
                        for inrow = row_start:row_end
                            @inbounds val = A[inrow, incol, patch, img]
                            if val > maxval
                                maxval = val
                                maxind = sub2ind(dims, inrow, incol, patch, img)
                            end
                        end
                    end
                    @inbounds Z[prow, pcol, patch, img] = maxval
                    @inbounds maxinds[prow, pcol, patch, img] = maxind
                    row_start += stride
                    row_end += stride
                end
                col_start += stride
                col_end += stride
            end
        end
    end
end


#=
Upsample function for max pooling - the unit which was chosen as the max
receives all the error
 Input:
 Δpool - delta of pool layer
       - pool_dim x pool_dim x num_patches x num_obs
 maxinds - linear index of the max value from each patch
 Output (updated in-place):
 Δconv - delta of previous layer (usually a convolution layer)
       - conv_dim x conv_dim x num_patches x num_obs
=#
function upsample_max!(Δconv, Δpool, maxinds)
    fill!(Δconv, 0)
    for img in indices(Δpool, 4)
        for patch in indices(Δpool, 3)
            for pcol in indices(Δpool, 2)
                for prow in indices(Δpool, 1)
                    @inbounds ind = maxinds[prow, pcol, patch, img]
                    @inbounds Δconv[ind] = Δpool[prow, pcol, patch, img]
                end
            end
        end
    end
end
