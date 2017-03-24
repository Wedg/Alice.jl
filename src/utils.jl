#=
This page contains various utility functions.
=#

########################################
# Size of layer - overload Base.size
########################################

size(ℓ::NetLayer) = size(ℓ.A)


########################################
# Network Display
########################################

# Display of the NeuralNet type
function show(io::IO, net::NeuralNet)
    println(io, "Neural Network")
    println(io, "Training Data Dimensions - $(size(net.data.X_train))")
    println(io, "Layers:")
    num_layers = length(net.layers)
    for (n, ℓ) in enumerate(net.layers)
        println(io, "Layer $n - ", displaystr(ℓ))
    end
end

# Display of hidden layers
function show(io::IO, ℓ::NetLayer)
    println(io, displaystr(ℓ))
end

# Display string for layers with activation functions
function displaystr(ℓ::NetLayer)
    str = "$(typeof(ℓ))"
    activation = "$(ℓ.f!)"[1:end-1]
    str *= ", Activation - $activation"
    str *= ", Dimensions - $(size(ℓ))"
end

# Display string for layers without activation functions
function displaystr{L <: Union{InputLayer, WordEmbeddingLayer, PoolLayer, OutputLayer}}(ℓ::L)
    str = "$(typeof(ℓ))"
    str *= ", Dimensions - $(size(ℓ))"
end


########################################
# Batch Views
########################################

# View of a batch - outer dimension contains the observations (e.g. images)
viewbatch{R<:Real, T<:Integer}(X::Array{R,1}, ids::Vector{T}) = view(X, ids)
viewbatch{R<:Real, T<:Integer}(X::Array{R,2}, ids::Vector{T}) = view(X, :, ids)
viewbatch{R<:Real, T<:Integer}(X::Array{R,3}, ids::Vector{T}) = view(X, :, :, ids)
viewbatch{R<:Real, T<:Integer}(X::Array{R,4}, ids::Vector{T}) = view(X, :, :, :, ids)
viewbatch{R<:Real, T<:Integer}(X::Array{R,1}, start::T, stop::T) = view(X, start:stop)
viewbatch{R<:Real, T<:Integer}(X::Array{R,2}, start::T, stop::T) = view(X, :, start:stop)
viewbatch{R<:Real, T<:Integer}(X::Array{R,3}, start::T, stop::T) = view(X, :, :, start:stop)
viewbatch{R<:Real, T<:Integer}(X::Array{R,4}, start::T, stop::T) = view(X, :, :, :, start:stop)


########################################
# Count all params in net
########################################

function num_params(net::NeuralNet)
    count = 0
    for ℓ in net.layers[2:end]
        typeof(ℓ) <: PoolLayer && continue
        count += length(ℓ.b)
        count += length(ℓ.W)
    end
    count
end


########################################
# Word Embedding Layer functions
########################################

function string_id(word::String, vocab::Vector{String})
    for i = 1:length(vocab)
        vocab[i] == word && return i
    end
    "No match"
end

function display_nearest_words(embed::WordEmbeddingLayer, vocab::Vector{String},
                               word::String, k::Int)

    # Number of words in vocab
    vocab_size = length(vocab);

    # Find id of word
    id = string_id(word, vocab)

    # Compute distance to other words
    word_rep = embed.W[:, id]
    diff = embed.W .- word_rep
    dist = vec(sqrt(sum(diff.^2, 1)))

    # Sort and remove top item
    table = [dist 1:vocab_size]
    table = sortrows(table)[2:k+1, :]

    # Print list
    @printf("%-10s %s\n", "word", "distance")
    @printf("%s\n", "--------   --------")
    for w = 1:k
        @printf("%-10s %.2f\n", vocab[Int(table[w, 2])], table[w, 1])
    end
end

function predict_next_word(net::NeuralNet, vocab::Vector{String}, input::Tuple,
                           k::Int)

    # Find indices of the inputs words and place them in input layer
    num_inputs = length(input)
    ids = zeros(Int, num_inputs)
    for i in 1:num_inputs
        ids[i] = string_id(input[i], vocab)
    end
    net.layers[1].A[:, 1] = ids

    # Forward prop
    fwd_prop!(net)
    net.layers[end].A[:, 1]

    # Predicted next word
    vocab_size = length(vocab)
    output = [net.layers[end].A[:, 1] 1:vocab_size]
    output = sortrows(output, rev=true)[1:k, :]

    # Print completed sentences
    space = length(prod(input)) + length(input) + 9
    println("string" * " "^space * "probability")
    println("-"^space * " "^6 * "-"^11)
    for i in 1:k
        for w in input
            print("$w ")
            #@printf("%s ", w)
        end
        @printf("%-14s %.3f\n", vocab[Int(output[i, 2])], output[i, 1])
    end
end


########################################
# Demo Data Paths
########################################

# Root and dataset paths
rootpath = dirname(@__FILE__)[1:end-4]

# ngram demo data
ngrampath = joinpath(rootpath, "demo", "ngrams", "ngramdata.jld")
load_ngrams() = load(ngrampath, "train_data", "valid_data", "test_data", "vocab")

# stl10 demo data
stl10path = joinpath(rootpath, "demo", "stl10")
stl10patches = joinpath(stl10path, "stl_sampled_patches.jld")
load_patches() = load(stl10patches, "patches")
stl10trainsubset = joinpath(stl10path, "stl_train_subset.jld")
load_train_subset() = load(stl10trainsubset, "train_images", "train_labels")
stl10testsubset = joinpath(stl10path, "stl_test_subset.jld")
load_test_subset() = load(stl10testsubset, "test_images", "test_labels")
stl10features = joinpath(stl10path, "stl_features.jld")
load_features() = load(stl10features, "W", "b")
