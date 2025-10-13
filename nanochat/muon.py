"""
Muon optimizer from Keller et al. - MLX port
Simplified for single-device training on Apple Silicon.
"""
import mlx.core as mx
import mlx.nn as nn


def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    Quintic iteration with coefficients selected to maximize slope at zero.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.astype(mx.float32)

    if G.shape[-2] > G.shape[-1]:
        X = mx.transpose(X, list(range(X.ndim - 2)) + [-1, -2])

    # Ensure spectral norm is at most 1
    norm = mx.sqrt(mx.sum(mx.square(X), axis=(-2, -1), keepdims=True))
    X = X / (norm + 1e-7)

    # Perform the NS iterations
    for _ in range(steps):
        A = X @ mx.transpose(X, list(range(X.ndim - 2)) + [-1, -2])
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if G.shape[-2] > G.shape[-1]:
        X = mx.transpose(X, list(range(X.ndim - 2)) + [-1, -2])

    return X


class Muon:
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    MLX implementation for single-device training.

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization
    post-processing step, in which each 2D parameter's update is replaced with the
    nearest orthogonal matrix.

    Arguments:
        learning_rate: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD.
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """

    def __init__(self, learning_rate=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        self.state = {}

    def update(self, model, gradients):
        """
        Update model parameters given gradients.

        Args:
            model: The model (or parameter dict)
            gradients: Dict of gradients matching model structure

        Returns:
            Updated parameters
        """
        if not self.state:
            # Initialize momentum buffers
            self.state = {k: mx.zeros_like(g) for k, g in gradients.items()}

        updates = {}
        for key, grad in gradients.items():
            if grad is None:
                continue

            # Get or initialize momentum buffer
            buf = self.state.get(key, mx.zeros_like(grad))

            # Update momentum buffer
            buf = self.momentum * buf + (1 - self.momentum) * grad
            self.state[key] = buf

            # Choose between momentum buffer and Nesterov
            g = (1 - self.momentum) * grad + self.momentum * buf if self.nesterov else buf

            # Orthogonalize if 2D
            if g.ndim >= 2:
                g = zeropower_via_newtonschulz5(g, steps=self.ns_steps)

            # Compute aspect ratio scaling
            if g.ndim >= 2:
                aspect_ratio = max(1, g.shape[-2] / g.shape[-1]) ** 0.5
            else:
                aspect_ratio = 1.0

            updates[key] = -self.learning_rate * aspect_ratio * g

        return updates

    def apply_updates(self, model, updates):
        """Apply parameter updates to model"""
        # This will be called by the training loop
        pass


def value_and_grad_with_filter(fn, params, param_filter):
    """
    Compute value and gradients only for filtered parameters.

    Args:
        fn: Function to differentiate
        params: Model parameters
        param_filter: Set of parameter keys to compute gradients for
    """
    def filtered_fn(filtered_params, all_params):
        # Merge filtered params back into all params
        merged = {**all_params}
        merged.update(filtered_params)
        return fn(merged)

    # Extract filtered parameters
    filtered_params = {k: v for k, v in params.items() if k in param_filter}

    # Compute gradients only for filtered params
    grad_fn = mx.grad(filtered_fn)
    grads = grad_fn(filtered_params, params)
    value = fn(params)

    return value, grads
