import random
from dataclasses import dataclass, field
from typing import Any, List, Tuple, Dict
import json
import numpy as np
from configs.model_specs import get_quantization_bytes

expert_latency_cache = {}


@dataclass
class OperationLatency:
    """Stores compute, memory, and the bound latency for a single operation."""

    c: float = 0.0
    m: float = 0.0
    bound: float = 0.0  # Equivalent to "c+m" in the original code

    def calculate_bound(self, p):
        """Calculates the bound (max of compute or memory)."""
        if p:
            self.bound = (self.c**p + self.m**p) ** (1 / p)
        else:
            self.bound = max(self.c, self.m)

    def __add__(self, other: "OperationLatency") -> "OperationLatency":
        if not isinstance(other, OperationLatency):
            return NotImplemented
        return OperationLatency(
            c=self.c + other.c, m=self.m + other.m, bound=self.bound + other.bound
        )


def default_latency_dict() -> Dict[str, OperationLatency]:
    """Provides a default dictionary for operation latencies."""
    return {
        "qkv": OperationLatency(),
        "attn_score": OperationLatency(),
        "o": OperationLatency(),
        "up_proj": OperationLatency(),
        "down_proj": OperationLatency(),
        "llm_head": OperationLatency(),
    }


def sum_latencies(latency_dict: Dict[str, OperationLatency]) -> OperationLatency:
    """Helper function to aggregate total compute, memory, and bound across a dictionary."""
    total = OperationLatency()
    for op in latency_dict.values():
        total += op
    return total


def add_latency_dicts(l1: Dict[str, OperationLatency], l2: Dict[str, OperationLatency]):
    """Adds l2 to l1"""
    for op in l1:
        l1[op] += l2[op]


@dataclass
class Latency:
    waiting: OperationLatency = field(default_factory=OperationLatency)
    prefill: Dict[str, OperationLatency] = field(default_factory=default_latency_dict)
    decode: Dict[str, OperationLatency] = field(default_factory=default_latency_dict)


@dataclass
class Request:
    start_time: float
    input_len: int
    output_len: int
    id: Any
    finish_time: float = 0
    kv_len: int = 0
    cur_input_len: int = 0
    latency: Latency = field(default_factory=Latency)


def num_tokens_in_batch(requests: List[Request]) -> int:
    return sum(r.cur_input_len for r in requests)


def get_achievable_bandwidth(data_size_bytes, mem_bw):
    """
    Estimates the achievable memory bandwidth for a given data size in bytes.

    :param data_size_bytes: Arbitrary data size in bytes
    :return: Estimated bandwidth in GB/s
    """
    bw_data = {}
    # convert the data size key to int
    for data_size in mem_bw:
        bw_data[int(data_size)] = mem_bw[data_size]

    # Convert arbitrary bytes to Megabytes (1 MB = 1024^2 bytes)
    data_size_mb = data_size_bytes / (1024**2)

    # Extract X (MB) and Y (GB/s) points, sorted by X
    points = sorted(bw_data.items())

    x_mb = np.array([p[0] for p in points])
    y_bw = np.array([p[1] for p in points])

    # Use log2 space for the X-axis because the benchmark grows exponentially
    # and bandwidth scales logarithmically with cache boundaries
    x_log = np.log2(x_mb)
    data_size_log = np.log2(data_size_mb)

    # Interpolate (numpy.interp automatically handles out-of-bounds by capping
    # to the minimum or maximum Y values)
    estimated_bw = np.interp(data_size_log, x_log, y_bw)
    return estimated_bw * 1e9


def estimate_operation_latency(
    total_flops, compute_precision, total_bytes, bench_data, gpu
) -> OperationLatency:
    # for now we only consider bf16 computation
    compute_flops = gpu.bf16_flops
    # if compute_precision == "fp8":
    #     compute_flops = gpu.fp8_flops

    mem_bandwidth = gpu.mem_bandwidth
    if bench_data:
        compute_flops = bench_data["compute"][compute_precision] * 1e12
        mem_bandwidth = get_achievable_bandwidth(total_bytes, bench_data["memory"])

    compute_time = total_flops / compute_flops
    memory_time = total_bytes / mem_bandwidth

    l = OperationLatency(c=compute_time, m=memory_time)

    # Calculate final bounds (max of compute/memory) per layer
    p = None
    if bench_data:
        p = bench_data["p"]
    l.calculate_bound(p)

    return l


def matmul_latency(
    n: int, h1: int, h2: int, bench_data: Dict, gpu, w_dtype, a_dtype
) -> OperationLatency:
    """
    Calculates compute and memory latency for a matrix multiplication.
    Compute: 2 * N * h1 * h2 FLOPs
    Memory: Precision * (Inputs + Weights + Outputs) bytes
    """
    w_byte = get_quantization_bytes(w_dtype)
    a_byte = get_quantization_bytes(a_dtype)

    total_flops = 2 * n * h1 * h2
    total_bytes = a_byte * (n * h1 + n * h2) + w_byte * h1 * h2

    return estimate_operation_latency(
        total_flops, a_dtype, total_bytes, bench_data, gpu
    )


def GGEMM_latency(
    shapes: List[Tuple[int, int, int]], bench_data: Dict, gpu, w_dtype, a_dtype
) -> OperationLatency:
    w_byte = get_quantization_bytes(w_dtype)
    a_byte = get_quantization_bytes(a_dtype)

    total_flops = 0
    total_bytes = 0
    for s in shapes:
        n, h1, h2 = s
        if n == 0:
            continue
        total_flops += 2 * n * h1 * h2
        total_bytes += a_byte * (n * h1 + n * h2) + w_byte * (h1 * h2)

    return estimate_operation_latency(
        total_flops, a_dtype, total_bytes, bench_data, gpu
    )


def attn_score_latency(
    requests: List[Request],
    h: int,
    g: int,
    max_window_len: int,
    bench_data,
    gpu,
    w_dtype,
    a_dtype,
) -> OperationLatency:
    a_byte = get_quantization_bytes(a_dtype)

    total_flops = 0
    total_bytes = 0

    for r in requests:
        n_token = r.cur_input_len
        n_kv = min(n_token + r.kv_len, max_window_len)

        # flops
        q_k_flops = 2 * n_token * n_kv * h
        score_v_flops = 2 * n_kv * n_token * h
        total_flops += q_k_flops + score_v_flops

        # bytes
        read_q = n_token * h
        read_kv = 2 * n_kv * h // g
        if n_token + r.kv_len > max_window_len:
            read_kv = 2 * (n_token + max_window_len) * h // g
        write_o = n_token * h
        total_bytes += a_byte * (read_q + read_kv + write_o)

    return estimate_operation_latency(
        total_flops, a_dtype, total_bytes, bench_data, gpu
    )


def transformer_latency(
    requests: List[Request], layer: int, bench_data: Dict, gpu, model
) -> Dict[str, OperationLatency]:
    latency = default_latency_dict()
    n_token = num_tokens_in_batch(requests)

    # ---------------------------------------------------------
    # GQA Attention
    # ---------------------------------------------------------
    g = model.n_attention_heads // model.n_kv_heads

    # 1. Linear Projections (Q, K, V)
    latency["qkv"] = matmul_latency(
        n_token,
        model.hidden_size,
        model.head_dim * (model.n_attention_heads + 2 * model.n_kv_heads),
        bench_data,
        gpu,
        model.attn_weight_dtype,
        model.activation_dtype,
    )

    # 2. Attention Core (QK^T and Score * V)
    window_size = model.max_seq_len
    if model.sliding_window and layer % 2 == 0:
        window_size = model.sliding_window

    latency["attn_score"] = attn_score_latency(
        requests,
        model.head_dim * model.n_attention_heads,
        g,
        window_size,
        bench_data,
        gpu,
        model.attn_weight_dtype,
        model.activation_dtype,
    )

    # 3. Output Projection
    latency["o"] = matmul_latency(
        n_token,
        model.head_dim * model.n_attention_heads,
        model.hidden_size,
        bench_data,
        gpu,
        model.attn_weight_dtype,
        model.activation_dtype,
    )

    # ---------------------------------------------------------
    # Mixture of Experts (FFN)
    # ---------------------------------------------------------
    if n_token in expert_latency_cache:
        latency["up_proj"] = expert_latency_cache[n_token]["up_proj"]
        latency["down_proj"] = expert_latency_cache[n_token]["down_proj"]

    else:
        # For now, we assume uniform distribution
        expert_list = [i for i in range(model.n_experts)]
        selected_experts = []
        for i in range(n_token):
            selected = random.sample(expert_list, model.top_k)
            selected_experts.append(list(selected))

        token_counts = [0 for i in range(model.n_experts)]
        for i in range(n_token):
            for e in selected_experts[i]:
                token_counts[e] += 1

        shapes = []
        for i in range(model.n_experts):
            shapes.append(
                (token_counts[i], model.hidden_size, model.moe_intermediate_size * 2)
            )

        # 1. Up & Gate Projection
        # h -> 2 * inter
        latency["up_proj"] = GGEMM_latency(
            shapes, bench_data, gpu, model.ffn_weight_dtype, model.activation_dtype
        )

        shapes = []
        for i in range(model.n_experts):
            shapes.append(
                (token_counts[i], model.moe_intermediate_size, model.hidden_size)
            )

        # 2. Down Projection
        # inter -> h
        latency["down_proj"] = GGEMM_latency(
            shapes, bench_data, gpu, model.ffn_weight_dtype, model.activation_dtype
        )

        expert_latency_cache[n_token] = {
            "up_proj": latency["up_proj"],
            "down_proj": latency["down_proj"],
        }

    return latency


def estimate_forward_latency(
    requests: List[Request], bench_data: Dict, gpu, model
) -> Dict[str, OperationLatency]:
    total_latency = default_latency_dict()

    for l in range(model.n_layers):
        layer_latency = transformer_latency(requests, l, bench_data, gpu, model)
        # Add the current layer's latencies to the running totals
        add_latency_dicts(total_latency, layer_latency)

    # LLM Head
    total_latency["llm_head"] = matmul_latency(
        len(requests),
        model.hidden_size,
        model.vocab_size,
        bench_data,
        gpu,
        model.model_orig_dtype,
        model.activation_dtype,
    )

    return total_latency
