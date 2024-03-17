import cupy as cp
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def ReLU(x):
    return cp.clip(x, 0, None)


def softmax(z):
    """
    Takes in an ndarray (..., ..., N) and applies softmax
        along the last dimension.
    """
    exp_z = cp.exp(z)
    return exp_z / cp.sum(exp_z, axis=-1, keepdims=True)


def cross_entropy_loss(outputs, y_true, from_logits=True):
    """
    Computes the average loss across these examples.
    outputs has shape (N, out_size)
    y_true has shape (out_size)
    """
    if from_logits:
        out_probs = softmax(outputs)
    else:
        out_probs = outputs

    # Terrible hack to select only the outputs we care about
    # https://stackoverflow.com/questions/70664524/numpy-use-all-rows-in-multidimensional-integer-indexing

    if len(out_probs.shape) == 1:
        assert cp.issubdtype(
            y_true, int), "Output and label dimensions are incompatible."
        return -cp.log(out_probs[y_true])

    return (-cp.sum(cp.log(cp.take_along_axis(out_probs, y_true[:, None], 1))) / len(y_true)).get()


class ArrayWithMomentum:
    """
    Ingests gradient vectors / matrices and spits out updated values.
    """

    def __init__(self, arr, beta, gamma, epsilon=1e-8):
        self.arr = arr
        self.arr_sq = cp.power(arr, 2)
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.t = 0

    def update(self, new_arr):
        self.arr = self.beta * self.arr + (1 - self.beta) * new_arr
        self.arr_sq = self.gamma * self.arr_sq + \
            (1 - self.gamma) * cp.power(new_arr, 2)
        self.t += 1

    def get(self):
        m = self.arr / (1 - cp.power(self.beta, self.t + 1))
        v = self.arr_sq / (1 - cp.power(self.gamma, self.t + 1))
        return m / (cp.sqrt(v) + self.epsilon)


class Layer:
    """
    Represents a layer in a deep neural network and its incoming weights.
    Takes in an ndarray as icput and returns outputs.
    Has utilities for computing gradients etc.
    """

    def __init__(self, in_size, out_size, act_func="relu"):
        """
        in_size: number of icput nodes
        out_size: number of output nodes
        act_func: activation function (string)
        Use He initialization for weights.
        """
        self.weights = cp.random.normal(
            loc=0,
            scale=np.sqrt(4 / (in_size + out_size)),
            size=(out_size, in_size)
        )
        self.biases = cp.zeros(out_size)
        self.act_func = act_func.lower()

        # Store values for forward pass
        self.in_acts = None
        self.out_acts = None

        self.weights_update = ArrayWithMomentum(
            cp.zeros_like(self.weights), 0.9, 0.999, 1e-8)
        self.biases_update = ArrayWithMomentum(
            cp.zeros_like(self.biases), 0.9, 0.999, 1e-8)

    def __call__(self, inputs):
        """
        Feed forward.
        inputs.shape: (batch_size, in_size)
        """
        self.in_acts = inputs[:]

        # Terrible hack to account for matrix order being weird
        self.out_preacts = (self.weights @ inputs.T).T

        # Add biases
        self.out_preacts += cp.broadcast_to(self.biases,
                                            self.out_preacts.shape)

        # Apply the activation function
        if self.act_func == "relu":
            self.out_acts = ReLU(self.out_preacts)
        else:
            self.out_acts = self.out_preacts
        return self.out_acts

    def compute_grads(self, out_preacts_grad, lambda_):
        """
        Compute the gradients of weights, biases, and icput activations (nodes)
            given gradient of output pre-activations.
        Assumes self.in_acts, self.out_preacts, and self.out_acts are all defined
            and have shape (batch_size, ...)

        out_preacts_grad: gradients of output activations
            shape: (batch_size, N_nodes)
        """
        # numpy.matmul documentation:
        # https://numpy.org/doc/stable/reference/generated/numpy.matmul
        self.biases_grad = out_preacts_grad[:]
        self.weights_grad = out_preacts_grad[:, :, None] \
            @ self.in_acts[:, None, :]

        # L2 regularization
        # self.weights_grad += 2 * lambda_ * self.weights

        in_preacts_grad = (self.in_acts > 0) * \
            (self.weights.T @ out_preacts_grad.T).T

        # Return gradients of previous layer's pre-activations
        # shape: (batch_size, N_nodes_prev_layer)
        return in_preacts_grad

    def update_params(self, alpha):
        """
        Update parameters with SGD and Adam.
        """
        self.weights_update.update(cp.mean(self.weights_grad, axis=0))
        self.biases_update.update(cp.mean(self.biases_grad, axis=0))

        self.biases -= self.biases_update.get() * alpha
        self.weights -= self.weights_update.get() * alpha


class Network:
    def __init__(self, in_size, out_size, hidden_sizes):
        """
        Initialize the deep network.
        """
        self.layers = []
        self.layers.append(Layer(in_size, hidden_sizes[0]))
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(Layer(hidden_sizes[i], hidden_sizes[i+1]))
        self.layers.append(Layer(hidden_sizes[-1], out_size))

    def __call__(self, inputs):
        """
        inputs must have shape (batch_size, in_size)
        Returns the output logits (pre-normalization outputs)
        """
        cur = inputs
        for layer in self.layers:
            cur = layer(cur)
        return cur

    def loss(self, X, y):
        return cross_entropy_loss(self.__call__(X), y, from_logits=True)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y.get())

    def predict(self, inputs):
        """
        Runs the inputs through the model, but argmax-es the last layer.
        """
        outputs = self.__call__(inputs)
        return np.argmax(outputs.get(), axis=1)

    def compute_grads(self, X, y, lambda_):
        """
        Compute gradients through all layers of the network.
        Should always be preceded by Network.__call__(X)

        X: inputs, of shape (batch_size, n_features)
        y: outputs, of shape (batch_size,)
        lambda_: L2 regularization term
        """
        batch_size, _ = X.shape

        outputs = self.__call__(X)  # shape: (batch_size, 10)
        exp = cp.exp(outputs)       # shape: (batch_size, 10)
        sum_exp = cp.sum(exp, axis=1)

        label_mask = cp.zeros_like(exp, dtype=bool)
        label_mask[cp.arange(batch_size), y] = True

        # Get (batch_size,) array of ONLY label outputs
        label_outputs = exp[cp.arange(batch_size), y]
        outputs_grad = cp.zeros_like(exp)
        outputs_grad -= (label_mask * exp) * (sum_exp - label_outputs)[:, None]
        outputs_grad += (~label_mask * exp) * label_outputs[:, None]
        outputs_grad /= (sum_exp**2)[:, None]

        cur_preact_grad = self.layers[-1].compute_grads(
            outputs_grad, lambda_=lambda_)
        for i in range(len(self.layers) - 2, -1, -1):
            cur_preact_grad = self.layers[i].compute_grads(
                cur_preact_grad, lambda_=lambda_)

    def update_params(self, alpha=1e-3):
        for layer in self.layers:
            layer.update_params(alpha)
