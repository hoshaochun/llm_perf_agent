"""Model specification configurations for MoE models."""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List
import json
from pathlib import Path


class AttentionType(str, Enum):
    """Supported attention types."""
    FULL = "full_attention"
    SLIDING = "sliding_attention"


class QuantizationMethod(str, Enum):
    """Supported quantization methods."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    FP8 = "fp8"
    INT4 = "int4"
    FP4 = "fp4"
    MXFP4 = "mxfp4"

def get_quantization_bytes(method: str) -> float:
    """Get bytes per parameter for a quantization method."""
    if method == QuantizationMethod.FP32:
        return 4.0
    elif method in [QuantizationMethod.FP16, QuantizationMethod.BF16]:
        return 2.0
    elif method in [QuantizationMethod.INT8, QuantizationMethod.FP8]:
        return 1.0
    elif method in [QuantizationMethod.INT4, QuantizationMethod.FP4, QuantizationMethod.MXFP4]:
        return 0.5
    return 2.0


@dataclass
class ModelConfig:
    """Configuration for a Mixture of Experts model."""
    
    name: str
    model_orig_dtype: str
    ffn_weight_dtype: str
    attn_weight_dtype: str
    activation_dtype: str
    kv_cache_dtype: str
    n_layers: int
    hidden_size: int # model_d
    vocab_size: int
    n_attention_heads: int
    n_kv_heads: int
    intermediate_size: int # FFN intermediate dimension
    n_experts: int
    top_k: int # experts per token
    max_seq_len: int
    head_dim: int
    moe_intermediate_size: int
    
    # Sliding window attention support
    layer_types: Optional[List[str]] = None  # List of attention types per layer ("full_attention" or "sliding_attention")
    sliding_window: Optional[int] = None  # Window size for sliding attention layers
    
    # Mixed quantization support (for HF-style quantization_config)
    # If None, uses the main 'quantization' for everything
    attn_quantization: Optional[str] = None  # Quantization for attention weights
    embed_quantization: Optional[str] = None  # Quantization for embeddings
    
    # Legacy compatibility
    _precision_bytes: Optional[float] = None
    
    @property
    def precision_bytes(self) -> float:
        """Get bytes per parameter based on quantization (for experts/MLP)."""
        if self._precision_bytes is not None:
            return self._precision_bytes
        return get_quantization_bytes(self.quantization)
    
    @precision_bytes.setter
    def precision_bytes(self, value: float):
        """Set precision bytes directly (legacy support)."""
        self._precision_bytes = value
    
    @property
    def attn_precision_bytes(self) -> float:
        """Get bytes per parameter for attention weights."""
        if self.attn_quantization is not None:
            return get_quantization_bytes(self.attn_quantization)
        return self.precision_bytes
    
    @property
    def embed_precision_bytes(self) -> float:
        """Get bytes per parameter for embeddings."""
        if self.embed_quantization is not None:
            return get_quantization_bytes(self.embed_quantization)
        return self.precision_bytes
    
    # @property
    # def head_dim(self) -> int:
    #     return self.hidden_size // self.n_attention_heads
    
    @property
    def n_full_attention_layers(self) -> int:
        """Number of full attention layers."""
        if self.layer_types is None:
            return self.n_layers
        return sum(1 for lt in self.layer_types if lt == AttentionType.FULL or lt == "full_attention")
    
    @property
    def n_sliding_attention_layers(self) -> int:
        """Number of sliding attention layers."""
        if self.layer_types is None:
            return 0
        return sum(1 for lt in self.layer_types if lt == AttentionType.SLIDING or lt == "sliding_attention")
    
    def get_layer_attention_type(self, layer_idx: int) -> str:
        """Get the attention type for a specific layer.
        
        Args:
            layer_idx: Layer index (0-based)
            
        Returns:
            "full_attention" or "sliding_attention"
        """
        if self.layer_types is None or layer_idx >= len(self.layer_types):
            return "full_attention"
        return self.layer_types[layer_idx]
    
    def get_effective_kv_len(self, layer_idx: int, kv_cache_len: int) -> int:
        """Get the effective KV cache length for a layer.
        
        For sliding attention layers, the effective KV cache length is capped
        at the sliding window size. For full attention layers, it's the full
        KV cache length.
        
        Args:
            layer_idx: Layer index (0-based)
            kv_cache_len: Full KV cache length
            
        Returns:
            Effective KV cache length for this layer
        """
        attn_type = self.get_layer_attention_type(layer_idx)
        if attn_type == "sliding_attention" and self.sliding_window is not None:
            return min(kv_cache_len, self.sliding_window)
        return kv_cache_len
    
    @property
    def attn_param_bytes(self) -> float:
        """Total attention parameters in bytes (per layer)."""
        # wq, wk, wv, wo
        n_params = 2 * self.hidden_size * self.head_dim * (
            self.n_attention_heads + self.n_kv_heads
        )
        return n_params * self.attn_precision_bytes
    
    @property
    def attn_flops_per_token(self) -> int:
        """FLOPs for attention projection per token (per layer)."""
        return 2 * 2 * self.hidden_size * self.head_dim * (
            self.n_attention_heads + self.n_kv_heads
        )
    
    @property
    def expert_param_bytes(self) -> float:
        """Parameters for ONE expert in bytes (per layer)."""
        # w1 (gate), w2 (down), w3 (up) for SwiGLU
        n_params = 3 * self.hidden_size * self.intermediate_size
        return n_params * self.precision_bytes
    
    @property
    def expert_flops_per_token(self) -> int:
        """FLOPs for ONE expert per token (per layer)."""
        return 3 * 2 * self.hidden_size * self.intermediate_size
    
    @property
    def total_param_bytes(self) -> float:
        """Total model parameters in bytes."""
        # Attention params per layer (may use different quantization)
        attn_total = self.attn_param_bytes * self.n_layers
        # Expert params per layer (all experts) - uses main quantization
        expert_total = self.expert_param_bytes * self.n_experts * self.n_layers
        # Embedding and output projection (may use different quantization)
        embed_total = 2 * self.vocab_size * self.hidden_size * self.embed_precision_bytes
        return attn_total + expert_total + embed_total
    
    def kv_cache_bytes_per_token(self, n_layers: Optional[int] = None) -> float:
        """KV cache size per token in bytes.
        
        Note: KV cache is always stored in the attention precision (typically bf16),
        not the weight quantization precision. This is because:
        1. KV cache stores activations, not weights
        2. Lower precision would significantly impact attention accuracy
        """
        if n_layers is None:
            n_layers = self.n_layers
        # key + value for each kv_head
        # Always use bf16 (2 bytes) for KV cache, regardless of weight quantization
        kv_precision = 2.0  # bf16 for KV cache
        return (
            2 * n_layers * self.n_kv_heads * self.head_dim * kv_precision
        )
    
    def kv_cache_bytes_for_seq_len(self, seq_len: int) -> float:
        """Calculate total KV cache memory for a given sequence length.
        
        This accounts for sliding window attention where some layers only
        store a limited number of tokens in their KV cache.
        
        Args:
            seq_len: The full sequence length
            
        Returns:
            Total KV cache size in bytes
        """
        kv_precision = 2.0  # bf16 for KV cache
        kv_per_layer_per_token = 2 * self.n_kv_heads * self.head_dim * kv_precision
        
        total_bytes = 0.0
        for layer_idx in range(self.n_layers):
            effective_len = self.get_effective_kv_len(layer_idx, seq_len)
            total_bytes += kv_per_layer_per_token * effective_len
        
        return total_bytes
    
    @classmethod
    def from_json(cls, path: str) -> "ModelConfig":
        """Load model config from HuggingFace-style config.json."""
        with open(path, "r") as f:
            config = json.load(f)
        
        # For MoE models, prioritize moe_intermediate_size over intermediate_size
        # (moe_intermediate_size is the per-expert FFN dimension)
        intermediate_size = config.get("moe_intermediate_size") or config.get("intermediate_size", 14336)
        
        # Try different keys for number of experts (different model families use different keys)
        n_experts = (
            config.get("num_experts") or      # Qwen, some others
            config.get("num_local_experts") or # Mixtral, DeepSeek
            config.get("n_routed_experts") or  # DeepSeek-V2
            8  # default
        )
        
        # Parse quantization - support both simple "quantization" field and HF-style "quantization_config"
        quantization = config.get("quantization", "bf16")
        attn_quantization = None
        embed_quantization = None
        
        if "quantization_config" in config:
            quant_config = config["quantization_config"]
            # Get the main quantization method
            quantization = quant_config.get("quant_method", quantization)
            
            # Parse modules_to_not_convert to determine which components stay in bf16
            modules_not_convert = quant_config.get("modules_to_not_convert", [])
            
            # Check if attention modules should not be converted (stay in bf16)
            attn_not_converted = any(
                "self_attn" in m or "attention" in m.lower() 
                for m in modules_not_convert
            )
            if attn_not_converted:
                attn_quantization = "bf16"
            
            # Check if embedding modules should not be converted (stay in bf16)
            embed_not_converted = any(
                "embed" in m.lower() or "lm_head" in m.lower()
                for m in modules_not_convert
            )
            if embed_not_converted:
                embed_quantization = "bf16"
        
        return cls(
            name=config.get("model_type", "custom"),
            quantization=quantization,
            n_layers=config.get("num_hidden_layers", 32),
            hidden_size=config.get("hidden_size", 4096),
            vocab_size=config.get("vocab_size", 32000),
            n_attention_heads=config.get("num_attention_heads", 32),
            n_kv_heads=config.get("num_key_value_heads", 8),
            intermediate_size=intermediate_size,
            n_experts=n_experts,
            top_k=config.get("num_experts_per_tok", 2),
            max_seq_len=config.get("max_position_embeddings", 32768),
            layer_types=config.get("layer_types"),
            sliding_window=config.get("sliding_window"),
            attn_quantization=attn_quantization,
            embed_quantization=embed_quantization,
        )
    
    def to_dict(self) -> dict:
        """Export config to dictionary."""
        result = {
            "name": self.name,
            "quantization": self.quantization,
            "precision_bytes": self.precision_bytes,
            "n_layers": self.n_layers,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "n_attention_heads": self.n_attention_heads,
            "n_kv_heads": self.n_kv_heads,
            "intermediate_size": self.intermediate_size,
            "n_experts": self.n_experts,
            "top_k": self.top_k,
            "max_seq_len": self.max_seq_len,
        }
        # Include sliding window attention fields if present
        if self.layer_types is not None:
            result["layer_types"] = self.layer_types
        if self.sliding_window is not None:
            result["sliding_window"] = self.sliding_window
        # Include mixed quantization fields if they differ from default
        if self.attn_quantization is not None:
            result["attn_quantization"] = self.attn_quantization
            result["attn_precision_bytes"] = self.attn_precision_bytes
        if self.embed_quantization is not None:
            result["embed_quantization"] = self.embed_quantization
            result["embed_precision_bytes"] = self.embed_precision_bytes
        return result


# Preset model configurations
PRESET_MODELS = {
    "openai/gpt-oss-20b": ModelConfig(
        name="openai/gpt-oss-20b",
        model_orig_dtype="bf16",
        ffn_weight_dtype="mxfp4",
        attn_weight_dtype="bf16",
        activation_dtype="bf16",
        kv_cache_dtype="bf16",
        n_layers=24,
        hidden_size=2880,
        vocab_size=201088,
        n_attention_heads=64,
        n_kv_heads=8,
        head_dim=64,
        intermediate_size=2880,
        moe_intermediate_size=2880,
        n_experts=32,
        top_k=4,
        max_seq_len=131072,
        sliding_window=128
    ),
    "Qwen/Qwen3-Coder-30B-A3B-Instruct": ModelConfig(
        name="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        model_orig_dtype="bf16",
        ffn_weight_dtype="bf16",
        attn_weight_dtype="bf16",
        activation_dtype="bf16",
        kv_cache_dtype="bf16",
        n_layers=48,
        hidden_size=2048,
        vocab_size=151936,
        n_attention_heads=32,
        n_kv_heads=4,
        head_dim=128,
        intermediate_size=6144,
        moe_intermediate_size=768,
        n_experts=128,
        top_k=8,
        max_seq_len=262144,
    ),
    "cpatonn/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit": ModelConfig(
        name="cpatonn/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit",
        model_orig_dtype="bf16",
        ffn_weight_dtype="int4",
        attn_weight_dtype="int4",
        activation_dtype="bf16",
        kv_cache_dtype="bf16",
        n_layers=48,
        hidden_size=2048,
        vocab_size=151936,
        n_attention_heads=32,
        n_kv_heads=4,
        head_dim=128,
        intermediate_size=5472,
        moe_intermediate_size=768,
        n_experts=128,
        top_k=8,
        max_seq_len=262144,
    ),
}

