import keras.backend as K


def get_activations(model, model_inputs, print_shape_only=False, layer_name=None):
    print('----- activations -----')
    activations = []
    inp = model.input

    model_multi_inputs_cond = True
    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        model_multi_inputs_cond = False

    outputs = [layer.output for layer in model.layers if
               layer.name == layer_name or layer_name is None]  # all layer outputs

    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    if model_multi_inputs_cond:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(1.)
    else:
        list_inputs = [model_inputs, 1.]

    # Learning phase. 1 = Test mode (no dropout or batch normalization)
    # layer_outputs = [func([model_inputs, 1.])[0] for func in funcs]
    layer_outputs = [func(list_inputs)[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations


def display_activations(activation_maps):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    """
    (1, 26, 26, 32)
    (1, 24, 24, 64)
    (1, 12, 12, 64)
    (1, 12, 12, 64)
    (1, 9216)
    (1, 128)
    (1, 128)
    (1, 10)
    """

    batch_size = activation_maps[0].shape[0]
    assert batch_size == 1, 'One image at a time to visualize.'
    X = np.zeros((25,25), dtype = float)
    A = np.array(activation_maps[7])
    plt.clf()


    fig, axes = plt.subplots(nrows = 8, ncols = 4,sharex = True, sharey = True)
    fig.patch.set_facecolor('white')

    i = 0
    #fig.subplots_adjust( wspace = 0)
    #fig.subplots_adjust(0,0,1,1,0,0)
    fig.subplots_adjust(wspace = 0, hspace = 0)
    for ax in axes.flat:
        #ax.label_outer()
        X = A[0,: ,:, i]
        i = i + 1
        im = ax.imshow(X, cmap = 'seismic')
        ax.set_adjustable('box-forced')
        ax.set_axis_off()

    #plt.tight_layout()
    plt.colorbar(im, ax=axes.ravel().tolist())
    plt.savefig('sigfilters.png', dpi =1200)
    plt.show()

    """
    fig = plt.figure(figsize=(8,4))
    ax = [plt.subplot(8,4,i+i) for i in range(32)]
    j = 0
    for a in ax.flat:
        a.set_xticklabels([])
        a.set_yticklabels([])
        im = a.imshow(A[0,:,:,j], cmap = 'seismic')
        j = j + 1

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.colorbar(im, a.axes.ravel().tolist())
    plt.show()



    #plt.clf()
    #plt.figure()
    #for i in xrange(32):
    #    plt.subplot(4,8,i)
    #    plt.imshow(A[0,:,:,i], cmap = 'seismic')

    #plt.xlabel('X label')
    #plt.ylabel('Y label')
    #plt.tight_layout()
    #plt.show()
     """


    # Hide x labels and tick labels for all but bottom plot.

    """
    for i, activation_map in enumerate(activation_maps):
        print('Displaying activation map {}'.format(i))

        shape = activation_map.shape
        print(shape)
        print(len(shape))
        if len(shape) == 4:
            print(activation_map.shape)


            activations = np.hstack(np.transpose(activation_map[0], (2, 0, 1)))

            print(activations.shape)

        if i == 0:
            for j in xrange(activation_map.shape[3]):
                plt.subplot(4, 8, j)
                print activations.shape[0][0,:][0,:][j]
                plt.imshow(activations[0][0:26][0:26][j], interpolation='None', cmap='jet')
            plt.show()
    """

def display_activations2(activation_maps):
    import numpy as np
    import matplotlib.pyplot as plt
    """
    (1, 26, 26, 32)
    (1, 24, 24, 64)
    (1, 12, 12, 64)
    (1, 12, 12, 64)
    (1, 9216)
    (1, 128)
    (1, 128)
    (1, 10)
    """
    batch_size = activation_maps[0].shape[0]
    assert batch_size == 1, 'One image at a time to visualize.'
    fig = plt.figure()
    for i, activation_map in enumerate(activation_maps):
        print('Displaying activation map {}'.format(i))
        shape = activation_map.shape
        if len(shape) == 4:
            activations = np.hstack(np.transpose(activation_map[0], (2, 0, 1)))
        elif len(shape) == 2:
            # try to make it square as much as possible. we can skip some activations.
            activations = activation_map[0]
            num_activations = len(activations)
            if num_activations > 1024:  # too hard to display it on the screen.
                square_param = int(np.floor(np.sqrt(num_activations)))
                activations = activations[0: square_param * square_param]
                activations = np.reshape(activations, (square_param, square_param))
            else:
                activations = np.expand_dims(activations, axis=0)
        else:
            raise Exception('len(shape) = 3 has not been implemented.')

        plt.subplot(4, 8, i)
        plt.imshow(activations, interpolation='None', cmap='jet')

    plt.show()