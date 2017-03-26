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

function load_patches()
    patches1 = load(joinpath(stl10path, "stl_sampled_patches1.jld"), "patches1")
    patches2 = load(joinpath(stl10path, "stl_sampled_patches2.jld"), "patches1")
    patches3 = load(joinpath(stl10path, "stl_sampled_patches3.jld"), "patches1")
    patches4 = load(joinpath(stl10path, "stl_sampled_patches4.jld"), "patches1")
    patches5 = load(joinpath(stl10path, "stl_sampled_patches5.jld"), "patches1")
    patches6 = load(joinpath(stl10path, "stl_sampled_patches6.jld"), "patches1")
    patches7 = load(joinpath(stl10path, "stl_sampled_patches7.jld"), "patches1")
    patches8 = load(joinpath(stl10path, "stl_sampled_patches8.jld"), "patches1")
    return hcat(patches1, patches2, patches3, patches4, patches5, patches6,
                patches7, patches8)
end

function load_train_subset()
    train_images1, train_labels1 = load(joinpath(stl10path, "stl_train_subset1.jld"), "train_images1", "train_labels1")
    train_images2, train_labels2 = load(joinpath(stl10path, "stl_train_subset2.jld"), "train_images2", "train_labels2")
    train_images3, train_labels3 = load(joinpath(stl10path, "stl_train_subset3.jld"), "train_images3", "train_labels3")
    train_images4, train_labels4 = load(joinpath(stl10path, "stl_train_subset4.jld"), "train_images4", "train_labels4")
    train_images5, train_labels5 = load(joinpath(stl10path, "stl_train_subset5.jld"), "train_images5", "train_labels5")
    train_images6, train_labels6 = load(joinpath(stl10path, "stl_train_subset6.jld"), "train_images6", "train_labels6")
    train_images7, train_labels7 = load(joinpath(stl10path, "stl_train_subset7.jld"), "train_images7", "train_labels7")
    train_images8, train_labels8 = load(joinpath(stl10path, "stl_train_subset8.jld"), "train_images8", "train_labels8")
    train_images = cat(4, train_images1, train_images2, train_images3, train_images4,
                          train_images5, train_images6, train_images7, train_images8)
    train_labels = vcat(train_labels1, train_labels2, train_labels3, train_labels4,
                        train_labels5, train_labels6, train_labels7, train_labels8)
    return train_images, train_labels
end

function load_test_subset()
    test_images1, test_labels1 = load(joinpath(stl10path, "stl_test_subset1.jld"), "test_images1", "test_labels1")
    test_images2, test_labels2 = load(joinpath(stl10path, "stl_test_subset2.jld"), "test_images2", "test_labels2")
    test_images3, test_labels3 = load(joinpath(stl10path, "stl_test_subset3.jld"), "test_images3", "test_labels3")
    test_images4, test_labels4 = load(joinpath(stl10path, "stl_test_subset4.jld"), "test_images4", "test_labels4")
    test_images5, test_labels5 = load(joinpath(stl10path, "stl_test_subset5.jld"), "test_images5", "test_labels5")
    test_images6, test_labels6 = load(joinpath(stl10path, "stl_test_subset6.jld"), "test_images6", "test_labels6")
    test_images7, test_labels7 = load(joinpath(stl10path, "stl_test_subset7.jld"), "test_images7", "test_labels7")
    test_images8, test_labels8 = load(joinpath(stl10path, "stl_test_subset8.jld"), "test_images8", "test_labels8")
    test_images9, test_labels9 = load(joinpath(stl10path, "stl_test_subset9.jld"), "test_images9", "test_labels9")
    test_images10, test_labels10 = load(joinpath(stl10path, "stl_test_subset10.jld"), "test_images10", "test_labels10")
    test_images11, test_labels11 = load(joinpath(stl10path, "stl_test_subset11.jld"), "test_images11", "test_labels11")
    test_images12, test_labels12 = load(joinpath(stl10path, "stl_test_subset12.jld"), "test_images12", "test_labels12")
    test_images13, test_labels13 = load(joinpath(stl10path, "stl_test_subset13.jld"), "test_images13", "test_labels13")
    test_images = cat(4, test_images1, test_images2, test_images3, test_images4,
                         test_images5, test_images6, test_images7, test_images8,
                         test_images9, test_images10, test_images11, test_images12,
                         test_images13)
    test_labels = vcat(test_labels1, test_labels2, test_labels3, test_labels4,
                       test_labels5, test_labels6, test_labels7, test_labels8,
                       test_labels9, test_labels10, test_labels11, test_labels12,
                       test_labels13)
    return test_images, test_labels
end

load_features() = load(stl10features, "W", "b")
