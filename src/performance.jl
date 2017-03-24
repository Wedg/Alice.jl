#=

=#

########################################
# Model Cost / Error
########################################

# With regularisation
function loss(net, X, y)

    # Data type
    F = net.dtype

    # Number of batches
    num_obs = size(X)[end]
    num_batches = div(num_obs, net.batch_size)

    # Cost
    J = F(0)

    # ℓ1 = first layer, L = final layer
    ℓ1 = net.layers[1]
    L = net.layers[end]

    # Loop over mini batches
    for batch = 1:num_batches

        # Update network with next batch
        stop = batch * net.batch_size
        start = stop - net.batch_size + 1
        ℓ1.A[:] = viewbatch(X, start, stop)
        L.y[:] = viewbatch(y, start, stop)

        # Feed forward
        fwd_prop!(net)

        # Cost without regularisation
        J += L.cost(L.A, L.y)
    end

    # Divide by number of observations
    J /= (net.batch_size * num_batches)

    # Add regularisation cost
    J += net.reg_cost(net)

    # Add sparsity cost
    J += sparse_cost(net)

    return J
end

# Without regularisation
function val_loss(net, X, y)

    # Data type
    F = net.dtype

    # Number of batches
    num_obs = size(X)[end]
    num_batches = div(num_obs, net.batch_size)

    # Cost
    J = F(0)

    # ℓ1 = first layer, L = final layer
    ℓ1 = net.layers[1]
    L = net.layers[end]

    # Loop over mini batches
    for batch = 1:num_batches

        # Update network with next batch
        stop = batch * net.batch_size
        start = stop - net.batch_size + 1
        ℓ1.A[:] = viewbatch(X, start, stop)
        L.y[:] = viewbatch(y, start, stop)

        # Feed forward
        fwd_prop!(net)

        # Cost without regularisation
        J += L.cost(L.A, L.y)
    end

    # Divide by number of observations
    J /= (net.batch_size * num_batches)

    return J
end

#loss(net) = loss(net, net.data.X_train, net.data.y_train)

########################################
# Model Cost / Error and Accuracy
########################################

# With regularisation
function loss_and_accuracy(net, X, y)

    # Data type
    F = net.dtype

    # Number of batches
    num_obs = size(X)[end]
    num_batches = div(num_obs, net.batch_size)

    # Cost and Count of Correct Predictions
    J = F(0)
    num_correct = 0

    # ℓ1 = first layer, L = final layer
    ℓ1 = net.layers[1]
    L = net.layers[end]

    # Loop over mini batches
    for batch = 1:num_batches

        # Update network with next batch
        stop = batch * net.batch_size
        start = stop - net.batch_size + 1
        ℓ1.A[:] = viewbatch(X, start, stop)
        L.y[:] = viewbatch(y, start, stop)

        # Feed forward
        fwd_prop!(net)

        # Cost without regularisation
        J += L.cost(L.A, L.y)

        # Count correct predictions
        num_correct += count_correct(L)
    end

    # Divide by number of observations
    num_used = net.batch_size * num_batches
    J /= num_used

    # Add regularisation cost
    J += net.reg_cost(net)

    # Add sparsity cost
    J += sparse_cost(net)

    # Accuracy (%)
    accuracy = num_correct / num_used * 100

    return J, accuracy
end

# Without regularisation
function val_loss_and_accuracy(net, X, y)

    # Data type
    F = net.dtype

    # Number of batches
    num_obs = size(X)[end]
    num_batches = div(num_obs, net.batch_size)

    # Cost and Count of Correct Predictions
    J = F(0)
    num_correct = 0

    # ℓ1 = first layer, L = final layer
    ℓ1 = net.layers[1]
    L = net.layers[end]

    # Loop over mini batches
    for batch = 1:num_batches

        # Update network with next batch
        stop = batch * net.batch_size
        start = stop - net.batch_size + 1
        ℓ1.A[:] = viewbatch(X, start, stop)
        L.y[:] = viewbatch(y, start, stop)

        # Feed forward
        fwd_prop!(net)

        # Cost without regularisation
        J += L.cost(L.A, L.y)

        # Count correct predictions
        num_correct += count_correct(L)
    end

    # Divide by number of observations
    num_used = net.batch_size * num_batches
    J /= num_used

    # Accuracy (%)
    accuracy = num_correct / num_used * 100

    return J, accuracy
end

########################################
# Model Accuracy
########################################

function accuracy(net, X, y)

    # Number of batches
    num_obs = size(X)[end]
    num_batches = div(num_obs, net.batch_size)

    # Count of Correct Predictions
    num_correct = 0

    # ℓ1 = first layer, L = final layer
    ℓ1 = net.layers[1]
    L = net.layers[end]

    # Loop over mini batches
    for batch = 1:num_batches

        # Update network with next batch
        stop = batch * net.batch_size
        start = stop - net.batch_size + 1
        ℓ1.A[:] = viewbatch(X, start, stop)
        L.y[:] = viewbatch(y, start, stop)

        # Feed forward
        fwd_prop!(net)

        # Count correct predictions
        num_correct += count_correct(L)
    end

    # Accuracy (%)
    return num_correct / (net.batch_size * num_batches) * 100
end

########################################
# Count Correct Predictions
########################################

# Counts the correct predictions in Softmax Output Layer
function count_correct(ℓ::SoftmaxOutputLayer)

    num_correct = 0
    for i in indices(ℓ.A, 2)
        indmax(ℓ.A[:, i]) == ℓ.y[i] && (num_correct += 1)
    end

    return num_correct
end

# Counts the correct predictions in Logistic Output Layer
function count_correct(ℓ::LogisticOutputLayer)

    num_correct = 0
    for i in indices(ℓ.A, 2)
        round(ℓ.A[i]) == ℓ.y[i] && (num_correct += 1)
    end

    return num_correct
end

########################################
# Model Predictions
########################################

# Outputs the predicted label from Softmax Output Layer
function predictions(ℓ::SoftmaxOutputLayer)

    # Number of observations / size of vector
    num_obs = size(ℓ.A, 2)

    # Prediction vector
    pred = Array(Int, num_obs)
    for i in 1:num_obs
        pred[i] = indmax(ℓ.A[:, i])
    end

    return pred
end

# Predictions for Logistic Output Layer - hypothesis closest to 0 or 1
predictions(ℓ::LogisticOutputLayer) = round(Int, ℓ.A)

# Predicted value for Linear Output Layer = Hypothesis
predictions(ℓ::LinearOutputLayer) = ℓ.A
