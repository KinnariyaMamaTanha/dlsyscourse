import struct
import numpy as np
import gzip
try:
    from simple_ml_ext import *
except:
    pass


def add(x, y) -> np.ndarray | int | float:
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    ### BEGIN YOUR CODE
    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        return x + y
    elif isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return x + y
    else:
        raise ValueError(f"Invalid input types. x: {type(x)} and y: {type(y)}")
    ### END YOUR CODE


def parse_mnist(image_filename: str, label_filename: str) -> tuple[np.ndarray, np.ndarray]:
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    with gzip.open(image_filename, "rb") as f:
        magic_number = struct.unpack('>I', f.read(4))[0]
        num_images = struct.unpack('>I', f.read(4))[0]
        num_rows = struct.unpack('>I', f.read(4))[0]
        num_cols = struct.unpack('>I', f.read(4))[0]
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, num_rows * num_cols)
        images = images.astype(np.float32) / 255.0
    
    with gzip.open(label_filename, "rb") as f:
        magic_number = struct.unpack('>I', f.read(4))[0]
        num_labels = struct.unpack('>I', f.read(4))[0]
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    
    return images, labels
    ### END YOUR CODE


def softmax_loss(Z: np.ndarray, y: np.ndarray) -> float:
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.uint8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE
    assert Z.ndim == 2 and y.ndim == 1
    assert Z.shape[0] == y.shape[0]
    
    shifted_logits = Z - np.max(Z, axis=1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(shifted_logits), axis=1))
    loss = np.mean(log_sum_exp - shifted_logits[np.arange(Z.shape[0]), y])
    return loss
    ### END YOUR CODE


def softmax_regression_epoch(X: np.ndarray, y: np.ndarray, theta: np.ndarray, lr: float = 0.1, batch: int = 100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    assert X.ndim == 2 and y.ndim == 1
    assert X.shape[0] == y.shape[0]
    assert theta.ndim == 2
    assert theta.shape[0] == X.shape[1]
    assert theta.shape[1] == y.max() + 1

    bs = X.shape[0]
    num_classes = y.max() + 1
    for i in range(0, bs, batch):
        X_batch = X[i:i+batch]
        y_batch = y[i:i+batch]

        Z = X_batch @ theta
        Z -= np.max(Z, axis=1, keepdims=True)
        Z = np.exp(Z)
        Z /= np.sum(Z, axis=1, keepdims=True)

        grad = X_batch.T @ (Z - np.eye(num_classes)[y_batch]) / batch
        theta -= lr * grad
    ### END YOUR CODE


def nn_epoch(X: np.ndarray, y: np.ndarray, W1: np.ndarray, W2: np.ndarray, lr: float = 0.1, batch: int = 100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    assert X.ndim == 2 and y.ndim == 1
    assert X.shape[0] == y.shape[0]
    assert W1.ndim == 2
    assert W2.ndim == 2
    assert W1.shape[0] == X.shape[1]
    assert W2.shape[0] == W1.shape[1]

    bs = X.shape[0]
    for i in range(0, bs, batch):
        X_batch = X[i:i+batch]
        y_batch = y[i:i+batch]

        Z1 = np.maximum(X_batch @ W1, 0)

        Z2 = Z1 @ W2
        Z2_max = np.max(Z2, axis=1, keepdims=True)
        Z2 = np.exp(Z2 - Z2_max)
        Z2 /= np.sum(Z2, axis=1, keepdims=True)

        G2 = Z2 - np.eye(W2.shape[1])[y_batch]
        G1 = (Z1 > 0) * (G2 @ W2.T)

        W2 -= lr * (Z1.T @ G2) / batch
        W1 -= lr * (X_batch.T @ G1) / batch
    ### END YOUR CODE



### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, y_te: np.ndarray, epochs: int = 10, lr: float = 0.5, batch: int = 100,
                  cpp: bool = False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, y_te: np.ndarray, hidden_dim: int = 500,
             epochs: int = 10, lr: float = 0.5, batch: int = 100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.1)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr = 0.2)
