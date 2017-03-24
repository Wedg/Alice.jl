#=
This page contains a number of common methods used to select the distribution
which is sampled from to initialise the weights of the neural network.

The named methods are:
    Glorot (applied when using sigmoidal activation functions)
    http://www.jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
    - glorot_tanh_uniform
    - glorot_tanh_normal
    - glorot_logistic_uniform
    - glorot_logistic_normal
    He (applied when using rectified linear activation function)
    https://arxiv.org/abs/1502.01852
    - he_uniform
    - he_normal
    Le Cun
    http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
    - lecun_uniform
    - lecun_normal
=#

########################################
# Select method
########################################

function choose_distn(init::USymDistn, activation::Symbol,
                      fan_in::Integer, fan_out::Integer)

    # Default selections based on the activation function
    if init == :default
        if activation == :logistic
            init_method = glorot_logistic_uniform
        elseif activation == :tanh
            init_method = glorot_tanh_uniform
        elseif activation == :relu
            init_method = he_uniform
        end

        return init_method(fan_in, fan_out)

    # User chooses one of the named distributions
    elseif typeof(init) <: Symbol
        init_method = init_dict[init]

        return init_method(fan_in, fan_out)

    # User enters their own distribution
    elseif typeof(init) <: Distribution
        return init
    end
end

########################################
# Glorot
########################################
#=
http://www.jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
=#

function glorot_tanh_uniform{T<:Integer}(fan_in::T, fan_out::T)
    r = sqrt(6 / (fan_in + fan_out))
    return Uniform(-r, r)
end

function glorot_tanh_normal{T<:Integer}(fan_in::T, fan_out::T)
    σ = sqrt(2 / (fan_in + fan_out))
    return Normal(0, σ)
end

function glorot_logistic_uniform{T<:Integer}(fan_in::T, fan_out::T)
    r = 4 * sqrt(6 / (fan_in + fan_out))
    return Uniform(-r, r)
end

function glorot_logistic_normal{T<:Integer}(fan_in::T, fan_out::T)
    σ = 4 * sqrt(2 / (fan_in + fan_out))
    return Normal(0, σ)
end

########################################
# He
########################################
#=
https://arxiv.org/abs/1502.01852
=#

function he_uniform{T<:Integer}(fan_in::T, fan_out::T)
    r = sqrt(6 / fan_in)
    return Uniform(-r, r)
end

function he_normal{T<:Integer}(fan_in::T, fan_out::T)
    σ = sqrt(2 / fan_in)
    return Normal(0, σ)
end

########################################
# LeCun
########################################
#=
http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
=#

function lecun_uniform{T<:Integer}(fan_in::T, fan_out::T)
    r = sqrt(3 / fan_in)
    return Uniform(-r, r)
end

function lecun_normal{T<:Integer}(fan_in::T, fan_out::T)
    σ = sqrt(1 / fan_in)
    return Normal(0, σ)
end


########################################
# Initialising Dictionaries
########################################

init_dict = Dict{Symbol, Function}(:glorot_tanh_uniform => glorot_tanh_uniform,
                                   :glorot_tanh_normal => glorot_tanh_normal,
                                   :glorot_logistic_uniform => glorot_logistic_uniform,
                                   :glorot_logistic_normal => glorot_logistic_normal,
                                   :he_uniform => he_uniform,
                                   :he_normal => he_normal,
                                   :lecun_uniform => lecun_uniform,
                                   :lecun_normal => lecun_normal)
