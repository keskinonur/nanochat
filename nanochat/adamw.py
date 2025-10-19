"""
AdamW optimizer - MLX port
Simplified for single-device training on Apple Silicon.
Implements AdamW with state management compatible with the nanochat API.
"""
import mlx.core as mx


class AdamW:
    """
    AdamW optimizer implementation for MLX.
    Provides interface compatible with the original nanochat code.
    """

    def __init__(self, learning_rate=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.state = {}  # Stores momentum buffers
        self.step_count = 0

    def update(self, model, gradients):
        """
        Compute parameter updates given gradients.

        Args:
            model: The model (or parameter dict)
            gradients: Dict of gradients matching model structure

        Returns:
            Updated parameters (dict with same structure as model)
        """
        self.step_count += 1
        updated = {}

        for key, grad in gradients.items():
            if key not in model:
                continue

            param = model[key]

            # Initialize state if needed
            if key not in self.state:
                self.state[key] = {
                    'm': mx.zeros_like(grad),  # First moment
                    'v': mx.zeros_like(grad),  # Second moment
                }

            state = self.state[key]
            m, v = state['m'], state['v']
            beta1, beta2 = self.betas

            # Update biased first and second moment estimates
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * mx.square(grad)

            # Store updated moments
            state['m'] = m
            state['v'] = v

            # Bias correction
            m_hat = m / (1 - beta1 ** self.step_count)
            v_hat = v / (1 - beta2 ** self.step_count)

            # Compute update with weight decay (AdamW style - decoupled)
            update = m_hat / (mx.sqrt(v_hat) + self.eps)
            updated[key] = param - self.learning_rate * (update + self.weight_decay * param)

        return updated

    def set_learning_rate(self, lr):
        """Update the learning rate"""
        self.learning_rate = lr


# For backwards compatibility
DistAdamW = AdamW
