"""
AdamW optimizer - MLX port
Simplified for single-device training on Apple Silicon.
MLX provides built-in AdamW, but we wrap it for compatibility.
"""
import mlx.core as mx
import mlx.optimizers as optim


class AdamW:
    """
    Wrapper around MLX's built-in AdamW optimizer.
    Provides interface compatible with the original nanochat code.
    """

    def __init__(self, learning_rate=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Create the MLX optimizer
        self.optimizer = optim.AdamW(
            learning_rate=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )

    def update(self, model, gradients):
        """
        Compute parameter updates given gradients.

        Args:
            model: The model (or parameter dict)
            gradients: Dict of gradients matching model structure

        Returns:
            Updated parameters
        """
        return self.optimizer.apply_gradients(gradients, model)

    def set_learning_rate(self, lr):
        """Update the learning rate"""
        self.learning_rate = lr
        self.optimizer.learning_rate = lr


# For backwards compatibility
DistAdamW = AdamW
