"""Titans-style Neural Long-Term Memory Module for RL agents.

Based on: "Titans: Learning to Memorize at Test Time" (Behrouz et al., 2024)
https://arxiv.org/abs/2501.00663

Key concepts from the paper:
- Memory is an MLP that learns to memorize at TEST TIME via gradient descent
- Surprise = gradient magnitude of associative memory loss
- Momentum for past surprise (keeps attention through time)
- Weight decay for adaptive forgetting
- Loss: ℓ(M; x) = ||M(k) - v||² (associative memory)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import copy


class NeuralMemoryModule(nn.Module):
    """Neural Long-Term Memory from Titans paper.
    
    This is a meta-model that learns to memorize at test time.
    The memory is an MLP whose weights are updated via gradient descent
    on an associative memory loss.
    
    From the paper (Section 3.1):
    - M_t = (1 - α_t) * M_{t-1} + S_t  (memory update with forgetting)
    - S_t = η_t * S_{t-1} - θ_t * ∇ℓ(M; x)  (surprise with momentum)
    - ℓ(M; x) = ||M(k) - v||²  (associative memory loss)
    """
    
    def __init__(
        self, 
        input_dim: int, 
        memory_dim: int = 128,
        memory_depth: int = 2,  # Paper uses simple MLPs
        momentum: float = 0.9,  # η in paper
        forget_rate: float = 0.1,  # α in paper  
        lr: float = 0.1,  # θ in paper - learning rate for memory updates
    ):
        super().__init__()
        self.input_dim = input_dim
        self.memory_dim = memory_dim
        self.momentum = momentum
        self.forget_rate = forget_rate
        self.lr = lr
        
        # Key and Value projections (like in Transformers)
        self.W_K = nn.Linear(input_dim, memory_dim, bias=False)
        self.W_V = nn.Linear(input_dim, memory_dim, bias=False)
        self.W_Q = nn.Linear(input_dim, memory_dim, bias=False)
        
        # The Memory Network M - a simple MLP
        # This gets "trained" at test time via gradient descent!
        layers = []
        for i in range(memory_depth):
            if i == 0:
                layers.append(nn.Linear(memory_dim, memory_dim))
            else:
                layers.append(nn.Linear(memory_dim, memory_dim))
            if i < memory_depth - 1:
                layers.append(nn.SiLU())  # Paper uses SiLU
        self.memory_net = nn.Sequential(*layers)
        
        # Learnable gates for α, η, θ (data-dependent in paper)
        self.alpha_gate = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )
        self.eta_gate = nn.Sequential(
            nn.Linear(input_dim, 1), 
            nn.Sigmoid()
        )
        self.theta_gate = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )
        
        # Output projection with gating (Section 4.4)
        self.out_norm = nn.LayerNorm(memory_dim)
        self.out_gate = nn.Linear(memory_dim, memory_dim)
        
        # Momentum buffer for surprise
        self.register_buffer('surprise_momentum', torch.zeros(memory_dim))
        self.register_buffer('surprise_ema', torch.ones(1))
        
    def _compute_memory_gradient(
        self, 
        memory_net: nn.Module,
        key: torch.Tensor, 
        value: torch.Tensor
    ) -> torch.Tensor:
        """Compute gradient of associative memory loss.
        
        Loss: ℓ(M; x) = ||M(k) - v||²
        
        Returns the gradient magnitude (surprise).
        """
        # Forward through memory
        predicted = memory_net(key)
        
        # Associative memory loss
        loss = F.mse_loss(predicted, value, reduction='none')
        
        # Surprise = loss magnitude per sample
        surprise = loss.mean(dim=-1, keepdim=True)
        
        return surprise, loss.mean()
    
    def forward(
        self, 
        x: torch.Tensor, 
        prev_memory_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process input and update memory.
        
        The key insight from Titans: we UPDATE the memory network weights
        at test time based on surprise (gradient of associative loss).
        
        For RL, we simplify: instead of actually updating weights (expensive),
        we use the surprise signal to gate a GRU-style memory state.
        """
        batch_size = x.shape[0]
        
        # Project to key, value, query
        key = self.W_K(x)  # What to store
        value = self.W_V(x)  # Associated value
        query = self.W_Q(x)  # What to retrieve
        
        # Initialize memory state if needed
        if prev_memory_state is None:
            prev_memory_state = torch.zeros(batch_size, self.memory_dim, device=x.device)
        prev_memory_state = prev_memory_state.detach()
        
        # Compute surprise (gradient magnitude of associative loss)
        # This tells us how "unexpected" the current input is
        with torch.no_grad():
            predicted = self.memory_net(key)
            surprise_raw = F.mse_loss(predicted, value, reduction='none').mean(dim=-1, keepdim=True)
        
        # Data-dependent gates
        alpha = self.alpha_gate(x) * self.forget_rate  # Forgetting rate
        eta = self.eta_gate(x) * self.momentum  # Momentum decay
        theta = self.theta_gate(x) * self.lr  # Learning rate
        
        # Update surprise momentum (Eq. 10 in paper)
        # S_t = η * S_{t-1} - θ * ∇ℓ
        # Simplified: we use surprise_raw as proxy for gradient magnitude
        surprise = eta * surprise_raw + (1 - eta) * surprise_raw
        
        # Normalize surprise to [0, 1] for stability
        surprise_normalized = torch.sigmoid(surprise - 0.5)
        
        # Memory update (Eq. 13 in paper)
        # M_t = (1 - α) * M_{t-1} + S_t
        # We interpret this as: blend old memory with new encoding based on surprise
        
        # Encode current input
        encoded = self.memory_net(key)
        
        # Higher surprise = more memory update
        update_strength = surprise_normalized * theta
        
        # Forgetting + new information
        new_memory_state = (1 - alpha) * prev_memory_state + update_strength * encoded
        
        # Normalize to prevent explosion
        new_memory_state = self.out_norm(new_memory_state)
        
        # Retrieve from memory using query
        retrieved = self.memory_net(query)
        
        # Gated output (Section 4.4)
        gate = torch.sigmoid(self.out_gate(retrieved))
        output = gate * retrieved + (1 - gate) * encoded
        
        # Update EMA for logging
        with torch.no_grad():
            self.surprise_ema = 0.99 * self.surprise_ema + 0.01 * surprise.mean()
        
        return output, new_memory_state, surprise_normalized
    
    def get_prediction_loss(self) -> Optional[torch.Tensor]:
        """No auxiliary loss - surprise is computed inline."""
        return None


class TitansFeatureExtractor(nn.Module):
    """Feature extractor with Titans-style long-term memory.
    
    Combines:
    - CNN for spatial features (your existing architecture)
    - Neural memory module for temporal context
    - Surprise-based selective memorization
    """
    
    def __init__(
        self,
        cnn_output_dim: int,
        vector_dim: int,
        memory_dim: int = 128,
        memory_depth: int = 3,
        output_dim: int = 128
    ):
        super().__init__()
        
        self.input_dim = cnn_output_dim + vector_dim
        self.memory_dim = memory_dim
        self.output_dim = output_dim
        
        # Input projection
        self.input_proj = nn.Linear(self.input_dim, memory_dim)
        
        # Titans memory module
        self.memory = NeuralMemoryModule(
            input_dim=memory_dim,
            memory_dim=memory_dim,
            memory_depth=memory_depth,
            momentum=0.9,
            forget_rate=0.01
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(memory_dim, output_dim),
            nn.ReLU()
        )
        
        # Store memory state for recurrent processing
        self._memory_state = None
        
    def reset_memory(self, batch_size: int = 1, device: torch.device = None):
        """Reset memory state (call at episode start)."""
        if device is None:
            device = next(self.parameters()).device
        self._memory_state = torch.zeros(batch_size, self.memory_dim, device=device)
    
    def forward(
        self, 
        cnn_features: torch.Tensor, 
        vector_features: torch.Tensor,
        reset_memory: bool = False
    ) -> Tuple[torch.Tensor, dict]:
        """Forward pass with memory update.
        
        Args:
            cnn_features: CNN output [batch, cnn_dim]
            vector_features: Vector observations [batch, vector_dim]
            reset_memory: Whether to reset memory state
            
        Returns:
            features: Output features [batch, output_dim]
            info: Dict with surprise scores and memory state
        """
        batch_size = cnn_features.shape[0]
        device = cnn_features.device
        
        # Reset memory if needed
        if reset_memory or self._memory_state is None:
            self.reset_memory(batch_size, device)
        
        # Handle batch size changes
        if self._memory_state.shape[0] != batch_size:
            self.reset_memory(batch_size, device)
        
        # Combine CNN and vector features
        combined = torch.cat([cnn_features, vector_features], dim=-1)
        projected = self.input_proj(combined)
        
        # Process through memory
        memory_output, new_memory, surprise = self.memory(
            projected, self._memory_state
        )
        
        # Update stored memory state
        self._memory_state = new_memory.detach()
        
        # Project to output
        output = self.output_proj(memory_output)
        
        info = {
            'surprise': surprise.detach(),
            'memory_state': new_memory.detach(),
            'surprise_ema': self.memory.surprise_ema.item(),
            'prediction_loss': self.memory.get_prediction_loss()
        }
        
        return output, info


class TitansCNNExtractor(nn.Module):
    """Complete CNN + Titans Memory feature extractor for SB3.
    
    Drop-in replacement for CustomCNNExtractor with added long-term memory.
    """
    
    def __init__(
        self,
        observation_space,
        features_dim: int = 128,
        memory_dim: int = 128,
        memory_depth: int = 3
    ):
        super().__init__()
        
        image_space = observation_space.spaces["image"]
        vector_space = observation_space.spaces["vector"]
        
        n_channels = image_space.shape[0]
        grid_size = image_space.shape[1]
        
        # CNN for image processing (same as your existing architecture)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate CNN output size
        with torch.no_grad():
            sample = torch.zeros(1, n_channels, grid_size, grid_size)
            cnn_output_size = self.cnn(sample).shape[1]
        
        vector_size = vector_space.shape[0]
        
        # Titans memory module
        self.titans_memory = TitansFeatureExtractor(
            cnn_output_dim=cnn_output_size,
            vector_dim=vector_size,
            memory_dim=memory_dim,
            memory_depth=memory_depth,
            output_dim=features_dim
        )
        
        self._features_dim = features_dim
        self._last_info = {}
    
    @property
    def features_dim(self) -> int:
        return self._features_dim
    
    def reset_memory(self, batch_size: int = 1):
        """Reset memory state - call at episode boundaries."""
        self.titans_memory.reset_memory(batch_size)
    
    def forward(self, observations: dict) -> torch.Tensor:
        """Extract features with memory context.
        
        Args:
            observations: Dict with 'image' and 'vector' keys
            
        Returns:
            features: Memory-augmented features [batch, features_dim]
        """
        image = observations["image"]
        vector = observations["vector"]
        
        # Process image through CNN
        cnn_features = self.cnn(image)
        
        # Process through Titans memory
        features, info = self.titans_memory(cnn_features, vector)
        
        # Store info for logging/debugging
        self._last_info = info
        
        return features
    
    def get_surprise(self) -> float:
        """Get the last computed surprise score."""
        if 'surprise' in self._last_info:
            return self._last_info['surprise'].mean().item()
        return 0.0
    
    def get_prediction_loss(self) -> Optional[torch.Tensor]:
        """Get the auxiliary prediction loss for training."""
        if 'prediction_loss' in self._last_info:
            return self._last_info['prediction_loss']
        return None


# MIRAS Variants (from the paper)

class YAADMemory(NeuralMemoryModule):
    """YAAD variant: Robust to outliers using Huber loss.
    
    Uses a gentler penalty for large errors, making the model
    more robust when input data is messy or inconsistent.
    """
    
    def __init__(self, *args, huber_delta: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.huber_delta = huber_delta
    
    def _compute_surprise(
        self, 
        memory_state: torch.Tensor, 
        new_input: torch.Tensor
    ) -> torch.Tensor:
        """Compute surprise using Huber loss instead of MSE."""
        predicted = self.surprise.predictor(memory_state)
        error = new_input - predicted
        
        # Huber loss: quadratic for small errors, linear for large
        abs_error = torch.abs(error)
        quadratic = torch.clamp(abs_error, max=self.huber_delta)
        linear = abs_error - quadratic
        huber = 0.5 * quadratic ** 2 + self.huber_delta * linear
        
        surprise = huber.sum(dim=-1, keepdim=True)
        return surprise


class MONETAMemory(NeuralMemoryModule):
    """MONETA variant: Uses generalized norms for stricter penalties.
    
    Explores more disciplined rules for attention and forgetting,
    potentially leading to more stable long-term memory.
    """
    
    def __init__(self, *args, norm_p: float = 1.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm_p = norm_p
    
    def _compute_surprise(
        self, 
        memory_state: torch.Tensor, 
        new_input: torch.Tensor
    ) -> torch.Tensor:
        """Compute surprise using p-norm."""
        predicted = self.surprise.predictor(memory_state)
        error = new_input - predicted
        
        # Generalized p-norm
        surprise = torch.norm(error, p=self.norm_p, dim=-1, keepdim=True)
        return surprise


class MEMORAMemory(NeuralMemoryModule):
    """MEMORA variant: Probability-constrained memory updates.
    
    Forces memory to act like a probability distribution,
    ensuring controlled and balanced updates.
    """
    
    def forward(
        self, 
        x: torch.Tensor, 
        prev_memory: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process with softmax-normalized memory updates."""
        batch_size = x.shape[0]
        
        if prev_memory is None:
            prev_memory = torch.zeros(batch_size, self.memory_dim, device=x.device)
        
        encoded = self.memory_net(x)
        surprise = self.surprise(prev_memory, encoded)
        
        # Softmax normalization for probability-like updates
        update_weights = F.softmax(encoded, dim=-1)
        memory_weights = F.softmax(prev_memory, dim=-1)
        
        surprise_gate = torch.sigmoid(surprise - 1.0)
        forget_gate = self.forget_rate * (1 - surprise_gate)
        
        # Interpolate between probability distributions
        new_memory = (
            (1 - forget_gate) * memory_weights + 
            surprise_gate * update_weights
        )
        # Scale back to feature space
        new_memory = new_memory * self.memory_dim
        
        output = encoded + 0.5 * new_memory
        
        return output, new_memory, surprise


def create_titans_extractor(
    observation_space,
    features_dim: int = 128,
    memory_variant: str = "titans",  # "titans", "yaad", "moneta", "memora"
    memory_depth: int = 3
):
    """Factory function to create Titans-style feature extractors.
    
    Args:
        observation_space: Gymnasium observation space
        features_dim: Output feature dimension
        memory_variant: Which memory architecture to use
        memory_depth: Depth of memory network
        
    Returns:
        TitansCNNExtractor with specified memory variant
    """
    extractor = TitansCNNExtractor(
        observation_space,
        features_dim=features_dim,
        memory_dim=features_dim,
        memory_depth=memory_depth
    )
    
    # Swap memory module based on variant
    if memory_variant == "yaad":
        extractor.titans_memory.memory = YAADMemory(
            input_dim=features_dim,
            memory_dim=features_dim,
            memory_depth=memory_depth
        )
    elif memory_variant == "moneta":
        extractor.titans_memory.memory = MONETAMemory(
            input_dim=features_dim,
            memory_dim=features_dim,
            memory_depth=memory_depth
        )
    elif memory_variant == "memora":
        extractor.titans_memory.memory = MEMORAMemory(
            input_dim=features_dim,
            memory_dim=features_dim,
            memory_depth=memory_depth
        )
    
    return extractor
