import numpy as np


class Sigmoid:
    def __init__(self):
        pass

    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))


    def forward(self, x):
        """
        :param x: Inputs, of any shape.

        :return out: Outputs, of the same shape as x.
        :return cache: Cache, stored for backward computation, of the same shape as x.
        """
        shape = x.shape
        out, cache = np.zeros(shape), np.zeros(shape)
        ########################################################################
        # TODO:                                                                #
        # Implement the forward pass of Sigmoid activation function            #
        ########################################################################

        cache = x
        out = Sigmoid.sigmoid(x) # le sigmoid

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return out, cache

    def backward(self, dout, cache):
        """
        :param dout: Upstream gradient from the computational graph, from the Loss function
                    and up to this layer. Has the shape of the output of forward().
        :param cache: The values that were stored during forward() to the memory,
                    to be used during the backpropogation.
        :return: dx: the gradient w.r.t. input X, of the same shape as X
        """
        dx = None
        ########################################################################
        # TODO:                                                                #
        # Implement the backward pass of Sigmoid activation function           #
        ########################################################################

        x = cache
        s = Sigmoid.sigmoid(x)
        dx = dout * (1 - s) * s

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return dx


class Relu:
    def __init__(self):
        pass

    def forward(self, x):
        """
        :param x: Inputs, of any shape.

        :return outputs: Outputs, of the same shape as x.
        :return cache: Cache, stored for backward computation, of the same shape as x.
        """
        out = None
        cache = None
        ########################################################################
        # TODO:                                                                #
        # Implement the forward pass of Relu activation function               #
        ########################################################################

        cache = x
        out = np.maximum(0, x) # element-wise comparisons, returns an array as opposed to np.max -> vector or a scalar

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return out, cache

    def backward(self, dout, cache):
        """
        :param dout: Upstream gradient from the computational graph, from the Loss function
                    and up to this layer. Has the shape of the output of forward().
        :param cache: The values that were stored during forward() to the memory,
                    to be used during the backpropogation.
        :return: dx: the gradient w.r.t. input X, of the same shape as X
        """
        dx = None
        ########################################################################
        # TODO:                                                                #
        # Implement the backward pass of Relu activation function              #
        ########################################################################

        x = cache
        # dx = np.zeros(x.shape)
        dx = np.zeros_like(x) # same shape
        dx[x > 0] = dout[x > 0] # where x > 0, we pass the gradient, analogous to multiplying thoes entries by 1

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return dx


def affine_forward(x: np.ndarray, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.
    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.
    Inputs:
    :param x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    :param w: A numpy array of weights, of shape (D, M)
    :param b: A numpy array of biases, of shape (M,)
    :return out: output, of shape (N, M)
    :return cache: (x, w, b)
    """
    N, M = x.shape[0], b.shape[0]
    out = np.zeros((N,M))
    ########################################################################
    # TODO: Implement the affine forward pass. Store the result in out.    #
    # You will need to reshape the input into rows.                        #
    ########################################################################

    out = np.dot(x.reshape(N, -1), w) + b  # -1 converts to D = d_1 * ... * d_k; all broadcasted by the number of minibatches

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.
    Inputs:
    :param dout: Upstream derivative, of shape (N, M)
    :param cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: A numpy array of biases, of shape (M,
    :return dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    :return dw: Gradient with respect to w, of shape (D, M)
    :return db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ########################################################################
    # TODO: Implement the affine backward pass.                            #
    ########################################################################

    N, M = dout.shape

    dx = np.dot(dout, w.T).reshape(x.shape)  # dout is (N, M), w.T is (M, D), so dx will be (N, D) and then reshaped to (N, d1, ..., dk)
    dw = np.dot(x.reshape(N, -1).T, dout)  # x is (N, d1, ..., dk), reshaped to (N, D), so dw will be (D, M)
    db = np.sum(dout, axis=0)  # sum over the first dimension (N), so db will be (M,)

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
    return dx, dw, db