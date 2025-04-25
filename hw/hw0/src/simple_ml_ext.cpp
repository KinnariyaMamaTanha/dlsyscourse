#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    // X: m * n
    // y: m
    // theta: n * k

    auto* Z = new float[batch * k];
    auto* grad = new float[n * k];

    for (size_t i = 0; i < m; i += batch) {
        size_t current_batch_size = std::min(batch, m - i);

        // Z = X_batch @ theta: current_batch_size * n @ n * k
        // Z: current_batch_size * k
        // Optimized loop order for potentially better cache locality accessing theta
        // Initialize Z for the current batch
        for (size_t j = 0; j < current_batch_size; ++j) {
            auto* Z_row = Z + j * k;
            for (size_t d = 0; d < k; ++d) {
                Z_row[d] = 0.0f;
            }
        }
        // Perform matrix multiplication
        for (size_t c = 0; c < n; ++c) { // Loop over input dimension
            for (size_t j = 0; j < current_batch_size; ++j) { // Loop over batch examples
                float X_val = X[(i + j) * n + c];
                // Precompute pointer to theta row
                const float* theta_row = theta + c * k;
                // Precompute pointer to Z row
                float* Z_row = Z + j * k;
                for (size_t d = 0; d < k; ++d) { // Loop over classes
                    Z_row[d] += X_val * theta_row[d];
                }
            }
        }


        // Compute Softmax and Z - Iy in a combined way
        for (size_t j = 0; j < current_batch_size; ++j) {
            float* Z_row = Z + j * k;

            // Find max for numerical stability
            float max_val = *std::max_element(Z_row, Z_row + k);

            float sum_exp = 0.0f;
            // Compute exp, sum, and subtract Iy (one-hot encoded y)
            unsigned char y_true = y[i + j];
            for (size_t d = 0; d < k; ++d) {
                // Z[j*k+d] = exp(Z[j*k+d] - max_val)
                Z_row[d] = std::exp(Z_row[d] - max_val);
                sum_exp += Z_row[d];
            }

            // Normalize (divide by sum) and subtract Iy
            // Iy is 1 if d == y_true, 0 otherwise
            for (size_t d = 0; d < k; ++d) {
                 // Z[j*k+d] = Z[j*k+d] / sum_exp - (d == y_true ? 1.0f : 0.0f)
                Z_row[d] = Z_row[d] / sum_exp - (d == y_true);
            }
        }
        // Now Z holds the (probabilities - Iy) values


        // grad = X_batch.T @ (Z) / current_batch_size
        // grad: n * k
        // Initialize grad to zero
        std::fill(grad, grad + n * k, 0.0f);

        // grad = X_batch.T @ Z
        // Loop order optimized for cache access (iterate through X and Z row-wise)
        for (size_t j = 0; j < current_batch_size; ++j) { // Loop over batch examples
             const float* X_row = X + (i + j) * n;
             const float* Z_row = Z + j * k;
             for (size_t d = 0; d < n; ++d) { // Loop over input features
                 float X_val = X_row[d];
                 // Precompute pointer to grad row
                 float* grad_row = grad + d * k;
                 for (size_t c = 0; c < k; ++c) { // Loop over classes
                     grad_row[c] += X_val * Z_row[c];
                 }
             }
        }

        // Update theta using the gradient
        // theta -= lr * (grad / current_batch_size)
        float lr_batch = lr / current_batch_size;
        for (size_t j = 0; j < n; ++j) {
            for (size_t d = 0; d < k; ++d) {
                theta[j * k + d] -= lr_batch * grad[j * k + d];
            }
        }
    }


    delete[] Z;
    delete[] grad;

    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
