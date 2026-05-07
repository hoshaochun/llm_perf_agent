import argparse
from configs.model_specs import PRESET_MODELS, get_quantization_bytes


def calculate_kv_cache_size(
    model,
    concurrency: int,
    context_length: int
) -> float:
    """
    Estimates the KV cache size (GiB), accounting for models that interleave
    Sliding Window Attention and Full Attention layers.
    """
    n_layers = model.n_layers
    sliding_window = model.sliding_window
    max_seq_len = model.max_seq_len
    n_kv_heads = model.n_kv_heads
    head_dim = model.head_dim

    # 1. Enforce the model's absolute maximum context window
    requested_seq_len = context_length
    if requested_seq_len > max_seq_len:
        print(
            f"  [Warning] Requested sequence ({requested_seq_len}) exceeds model's max context ({max_seq_len}). Capping to max context."
        )
        requested_seq_len = max_seq_len

    # 2. Determine layer distribution
    # full: full attention
    # swa: sliding window attention
    num_swa_layers = 0
    if sliding_window is not None:
        num_swa_layers = n_layers
        if model.name == "openai/gpt-oss-20b":
            num_swa_layers = n_layers // 2

    num_full_layers = n_layers - num_swa_layers

    # 3. Calculate tokens cached per request
    # Full attention layers cache every token up to the requested sequence length
    tokens_in_full_layers = num_full_layers * requested_seq_len

    # SWA layers only cache up to the window size
    swa_capped_len = min(requested_seq_len, sliding_window) if sliding_window else 0
    tokens_in_swa_layers = num_swa_layers * swa_capped_len

    # Total effective tokens stored per request across all layers
    total_cached_tokens = concurrency * (tokens_in_full_layers + tokens_in_swa_layers)

    # 4. Determine bytes per parameter
    bytes_per_param = get_quantization_bytes(model.kv_cache_dtype)

    # 5. Calculate memory footprint
    # We calculate the cost per token for a SINGLE layer first
    bytes_per_token_per_layer = 2 * n_kv_heads * head_dim * bytes_per_param

    # Multiply by the sum of tokens across all layers
    total_bytes = bytes_per_token_per_layer * total_cached_tokens

    # Convert to GiB
    size_gb = total_bytes / (1024**3)

    return size_gb

def calculate_model_size(model) -> float:
    """
    Calculate the VRAM required to load the model weights.
    """
    # 1. Determine byte multipliers
    orig_bytes = get_quantization_bytes(model.model_orig_dtype)
    attn_bytes = get_quantization_bytes(model.attn_weight_dtype)
    ffn_bytes = get_quantization_bytes(model.ffn_weight_dtype)

    # 2. Embeddings
    embedding_vram = model.vocab_size * model.hidden_size * orig_bytes

    # 3. Attention (per layer)
    q_params = model.hidden_size * (model.n_attention_heads * model.head_dim)
    k_params = model.hidden_size * (model.n_kv_heads * model.head_dim)
    v_params = model.hidden_size * (model.n_kv_heads * model.head_dim)
    o_params = (model.n_attention_heads * model.head_dim) * model.hidden_size
    
    attn_params_per_layer = q_params + k_params + v_params + o_params
    attn_vram_per_layer = attn_params_per_layer * attn_bytes

    # 4. FFN / MoE (per layer)
    if model.n_experts > 0:
        # Router (usually kept in original dtype)
        router_params = model.hidden_size * model.n_experts
        router_vram = router_params * orig_bytes

        # Experts
        expert_params = 3 * model.hidden_size * model.moe_intermediate_size
        ffn_vram_per_layer = router_vram + (expert_params * model.n_experts * ffn_bytes)
    else:
        # Standard Dense Model
        ffn_params = 3 * model.hidden_size * model.intermediate_size
        ffn_vram_per_layer = ffn_params * ffn_bytes

    # Layer Norms (Attention Norm + FFN Norm)
    ln_vram_per_layer = (2 * model.hidden_size) * orig_bytes

    # 5. Total Layer VRAM
    total_layers_vram = model.n_layers * (attn_vram_per_layer + ffn_vram_per_layer + ln_vram_per_layer)

    # 6. Output LM Head & Final Norm
    final_norm_vram = model.hidden_size * orig_bytes
    lm_head_vram = model.vocab_size * model.hidden_size * orig_bytes

    # 7. Total Weights VRAM
    total_weights_bytes = embedding_vram + total_layers_vram + final_norm_vram + lm_head_vram
    total_weights_gb = total_weights_bytes / (1024**3)

    return total_weights_gb

# --- Example Usage ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimate VRAM requirements (Weights + KV Cache) for LLM deployment."
    )

    # Define command line arguments with default values
    parser.add_argument(
        "--concurrency",
        type=int,
        required=True,
        help="Target concurrency (number of parallel requests).",
    )
    parser.add_argument(
        "--context-length", 
        type=int, 
        required=True,
        help="Maximum context length per request."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model as defined in PRESET_MODELS.",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Load model specs
    try:
        model_spec = PRESET_MODELS[args.model]
    except KeyError:
        print(f"\n[!] Error: Model '{args.model}' not found in PRESET_MODELS.")
        exit(1)

    # --- Computations ---
    kv_dtype_bytes = get_quantization_bytes(model_spec.kv_cache_dtype)
    
    model_size_gb = calculate_model_size(model_spec)
    kv_cache_size_gb = calculate_kv_cache_size(
        model=model_spec,
        concurrency=args.concurrency,
        context_length=args.context_length
    )
    total_vram_gb = model_size_gb + kv_cache_size_gb

    # --- Structured Output ---
    print(f"\n{'='*55}")
    print(f"{'LLM VRAM ESTIMATION REPORT':^55}")
    print(f"{'='*55}")

    print("\n[ Deployment Parameters ]")
    print(f"{'-'*55}")
    print(f"{'Model Name':<22} : {args.model}")
    print(f"{'Target Concurrency':<22} : {args.concurrency} concurrent requests")
    print(f"{'Context Length':<22} : {args.context_length:,} tokens / request")
    print(f"{'KV Cache Precision':<22} : {model_spec.kv_cache_dtype} ({kv_dtype_bytes} bytes/param)")

    print("\n[ Memory Footprint ]")
    print(f"{'-'*55}")
    print(f"{'Model Size':<22} : {model_size_gb:>8.3f} GB")
    print(f"{'KV Cache Size':<22} : {kv_cache_size_gb:>8.3f} GB")
    print(f"{'-'*55}")
    print(f"{'Total Required VRAM':<22} : {total_vram_gb:>8.3f} GB")
    print(f"{'='*55}\n")