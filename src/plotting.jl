
#Gadfly.set_default_plot_size(24cm, 12cm)
# Loss curves
function plot_loss_history(net, last_train_every, full_train_every, val_every)

    # Number of batches run
    num_batches = length(net.last_train_loss) * last_train_every

    # Last batch training data
    xids = last_train_every:last_train_every:num_batches
    layer3 = layer(x=xids, y=net.last_train_loss, Geom.point, Geom.line,
    Theme(default_color=colorant"lightblue"))

    # Full training data
    xids = full_train_every:full_train_every:(num_batches - 1)
    xids = [xids; num_batches]
    layer2 = layer(x=xids, y=net.full_train_loss, Geom.point, Geom.line,
    Theme(default_color=colorant"blue"))

    # Validation data
    xids = val_every:val_every:(num_batches - 1)
    xids = [xids; num_batches]
    layer1 = layer(x=xids, y=net.val_loss, Geom.point, Geom.line,
    Theme(default_color=colorant"red"))

    # Plot
    plot(layer1, layer2, layer3,
         Guide.xlabel("epoch"), Guide.ylabel("loss"),
         Coord.cartesian(ymin=0.0),
         Guide.manual_color_key("", ["last batch training loss    ",
                                     "full training loss    ",
                                     "validation loss"],
                                    ["lightblue", "blue", "red"]),
         Theme(key_position = :top)
         )
end

# Display colour images that are stored in columns
function display_rgb_cols(A; extrema = (0.0, 1.0), scale::Int = 1)

    # Size and number of images to display
    num_input, num_imgs = size(A)
    @assert num_input % 3 == 0
    channel_size = div(num_input, 3)

    # Error if too big - max 450
    num_imgs > 450 && throw("Maximum number of images is 450. Input a smaller subset.")

    # Dimensions of individual images - must be squares
    dim = Int(sqrt(channel_size))
    channel_size == dim^2 || throw("Individual images must be square")
    dimp1 = dim + 1    # Useful for padding i.e. adding a pixel strip between individual images

    # Shift to range [0, 1] using provided extrema
    minval, maxval = extrema
    diffval = maxval - minval
    A = (A .- minval) ./ diffval

    # Split the input into 3 arrays - one for each RGB channel
    R = A[1:channel_size, :]                             # 1st channel_size rows
    G = A[(channel_size + 1):(channel_size * 2), :]      # 2nd channel_size rows
    B = A[(channel_size * 2 + 1):(channel_size * 3), :]  # 3rd channel_size rows

    # Initialise image canvas rectangle
    cols = round(Int, sqrt(num_imgs * 2))   # Tried also φ
    rows = ceil(Int, num_imgs / cols)
    I = ones(RGB, dim * rows + rows - 1, dim * cols + cols - 1)

    # Transfer features to the canvas
    for x = 0:(rows - 1)
        for y = 0:(cols - 1)

            # Number of the image - count through the previous rows and cols
            img = x * cols + y + 1

            # Stop after through all images - will leave rest of canvas blank
            img > num_imgs && break

            # Indices of this image on the main canvas
            x_start, x_end = x * dimp1 + 1, x * dimp1 + dim
            y_start, y_end = y * dimp1 + 1, y * dimp1 + dim

            # Fill the RGB array
            I[x_start:x_end, y_start:y_end] = RGB.(R[:, img], G[:, img], B[:, img])

        end
    end

    # If not rescaling then we are done
    scale == 1 && return I

    # Optionally rescale the image up for bigger display
    Irows, Icols = size(I)
    ISrows, IScols = Irows * scale, Icols * scale
    IS = zeros(RGB, ISrows, IScols)
    for j = 1:IScols
        for i = 1:ISrows
            IS[i, j] = I[ceil(Int, i / scale), ceil(Int, j / scale)]
        end
    end

    # Return image
    return IS
end

# Function to display network
function display_rgb_weights(A; scale::Int = 1)

    # Size and number of images to display
    num_input, num_imgs = size(A)
    @assert num_input % 3 == 0
    channel_size = div(num_input, 3)

    # Error if too big - max 450
    num_imgs > 450 && throw("Maximum number of images is 450. Input a smaller subset.")

    # Dimensions of individual images - must be squares
    dim = Int(sqrt(channel_size))
    channel_size == dim^2 || throw("Individual images must be square")
    dimp1 = dim + 1    # Useful for padding i.e. adding a pixel strip between individual images

    # Split the input into 3 arrays - one for each RGB channel
    R = A[1:channel_size, :]                             # 1st channel_size rows
    G = A[(channel_size + 1):(channel_size * 2), :]      # 2nd channel_size rows
    B = A[(channel_size * 2 + 1):(channel_size * 3), :]  # 3rd channel_size rows

    # Max neuron firing of each channel and normalise
    R = (R ./ maximum(abs(R), 1) .+ 1) ./ 2
    G = (G ./ maximum(abs(G), 1) .+ 1) ./ 2
    B = (B ./ maximum(abs(B), 1) .+ 1) ./ 2

    # Initialise image canvas rectangle
    cols = round(Int, sqrt(num_imgs * 2))   # Tried also φ
    rows = ceil(Int, num_imgs / cols)
    I = ones(RGB, dim * rows + rows - 1, dim * cols + cols - 1)

    # Transfer features to the canvas
    for x = 0:(rows - 1)
        for y = 0:(cols - 1)

            # Number of the image - count through the previous rows and cols
            img = x * cols + y + 1

            # Stop after through all images - will leave rest of canvas blank
            img > num_imgs && break

            # Indices of this image on the main canvas
            x_start, x_end = x * dimp1 + 1, x * dimp1 + dim
            y_start, y_end = y * dimp1 + 1, y * dimp1 + dim

            # Fill the RGB array
            I[x_start:x_end, y_start:y_end] = RGB.(R[:, img], G[:, img], B[:, img])

        end
    end

    # If not rescaling then we are done
    scale == 1 && return I

    # Optionally rescale the image up for bigger display
    Irows, Icols = size(I)
    ISrows, IScols = Irows * scale, Icols * scale
    IS = zeros(RGB, ISrows, IScols)
    for j = 1:IScols
        for i = 1:ISrows
            IS[i, j] = I[ceil(Int, i / scale), ceil(Int, j / scale)]
        end
    end

    # Return image
    return IS
end
