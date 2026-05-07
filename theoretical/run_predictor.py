import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.hw_specs import PRESET_GPUS
from configs.model_specs import PRESET_MODELS
from theoretical.predictor import (
    Request,
    estimate_forward_latency,
    num_tokens_in_batch,
    default_latency_dict,
    add_latency_dicts,
    sum_latencies,
)




def print_batch(requests):
    print(f"number of requests: {len(requests)}")
    print(f"number of tokens: {num_tokens_in_batch(requests)}")
    for r in requests:
        print(f"id: {r.id} n_token: {r.cur_input_len} n_kv: {r.kv_len}")

    print("==============================")


def schedule_requests(request_pool, waiting_queue, running_queue, args):
    max_num_batched_tokens = args.max_num_batched_tokens
    num_concurrency = args.num_concurrency

    # empty the batch first
    for i in range(len(running_queue)):
        running_queue[i].cur_input_len = 0

    # collect new tokens by FCFS
    # top priority: decode request in running queue
    for i in range(len(running_queue)):
        if running_queue[i].input_len <= running_queue[i].kv_len:
            running_queue[i].cur_input_len = 1

    # second priority: prefill request in running queue
    for i in range(len(running_queue)):
        if running_queue[i].input_len > running_queue[i].kv_len:
            running_queue[i].cur_input_len = min(
                running_queue[i].input_len - running_queue[i].kv_len,
                max_num_batched_tokens - num_tokens_in_batch(running_queue),
            )
    # get new requests
    while (
        len(running_queue) + len(waiting_queue) < num_concurrency
        and len(request_pool) > 0
    ):
        request = request_pool.pop(0)
        waiting_queue.append(request)

    # third priority: prefill request in waiting queue
    while (
        num_tokens_in_batch(running_queue) < max_num_batched_tokens
        and len(waiting_queue) > 0
        and len(running_queue) < num_concurrency
    ):
        request = waiting_queue.pop(0)
        request.cur_input_len = min(
            request.input_len,
            max_num_batched_tokens - num_tokens_in_batch(running_queue),
        )
        running_queue.append(request)


import numpy as np

def analyze_performance(finished_queue, simulation_latency):
    simulation_latency = sum_latencies(simulation_latency)
    
    print("\n" + "="*76)
    print(f"{'PERFORMANCE ESTIMATION & TUNING GUIDE':^76}")
    print("="*76 + "\n")

    num_finished = len(finished_queue)

    if num_finished == 0:
        print("No requests finished to analyze.")
        return
    
    sum_total_tokens = 0
    sum_total_output_tokens = 0
    sum_waiting = 0.0
    sum_total_prefill = 0.0
    sum_total_decode = 0.0
    sum_decode_per_token = 0.0
    
    # Store tuples of (bound, compute, memory)
    sum_prefill_attn = np.zeros(3)
    sum_prefill_ffn = np.zeros(3)
    sum_prefill_llm_head = np.zeros(3) # Added llm_head
    
    sum_decode_attn = np.zeros(3)
    sum_decode_ffn = np.zeros(3)
    sum_decode_llm_head = np.zeros(3) # Added llm_head
    
    sum_total_latency = 0.0

    # Helper function to extract (bound, c, m) totals for specific operations
    def get_metrics(phase_dict, keys):
        # Added a check in case a key might be missing from some phases
        valid_keys = [k for k in keys if k in phase_dict]
        b = sum(phase_dict[k].bound for k in valid_keys)
        c = sum(phase_dict[k].c for k in valid_keys)
        m = sum(phase_dict[k].m for k in valid_keys)
        return np.array([b, c, m])

    for r in finished_queue:
        waiting_time = r.latency.waiting.bound
        sum_waiting += waiting_time
        
        # Prefill Aggregations
        p_attn = get_metrics(r.latency.prefill, ["qkv", "attn_score", "o"])
        p_ffn = get_metrics(r.latency.prefill, ["up_proj", "down_proj"])
        p_llm_head = get_metrics(r.latency.prefill, ["llm_head"]) # Added llm_head
        
        total_prefill = sum(op.bound for op in r.latency.prefill.values())
        sum_total_prefill += total_prefill

        # Decode Aggregations
        d_attn = get_metrics(r.latency.decode, ["qkv", "attn_score", "o"])
        d_ffn = get_metrics(r.latency.decode, ["up_proj", "down_proj"])
        d_llm_head = get_metrics(r.latency.decode, ["llm_head"]) # Added llm_head
        
        total_decode = sum(op.bound for op in r.latency.decode.values())
        sum_total_decode += total_decode
        
        # Protect against divide by zero if output_len is 1
        if r.output_len > 1:
            sum_decode_per_token += (total_decode / (r.output_len - 1))

        # Add to component totals
        sum_prefill_attn += p_attn
        sum_prefill_ffn += p_ffn
        sum_prefill_llm_head += p_llm_head # Added llm_head
        
        sum_decode_attn += d_attn
        sum_decode_ffn += d_ffn
        sum_decode_llm_head += d_llm_head # Added llm_head
        
        sum_total_latency += (waiting_time + total_prefill + total_decode)
        sum_total_tokens += (r.input_len + r.output_len)
        sum_total_output_tokens += r.output_len

    # Calculate final averages
    avg_waiting = sum_waiting / num_finished
    avg_total_prefill = sum_total_prefill / num_finished
    avg_total_decode = sum_total_decode / num_finished
    avg_total_decode_per_token = sum_decode_per_token / num_finished
    avg_total_latency = sum_total_latency / num_finished
    
    avg_p_attn = sum_prefill_attn / num_finished
    avg_p_ffn = sum_prefill_ffn / num_finished
    avg_p_llm_head = sum_prefill_llm_head / num_finished # Added llm_head
    
    avg_d_attn = sum_decode_attn / num_finished
    avg_d_ffn = sum_decode_ffn / num_finished
    avg_d_llm_head = sum_decode_llm_head / num_finished # Added llm_head

    def calc_pct(val, total):
        return (val / total * 100) if total > 0 else 0.0

    # 1. Top-level results
    print(f"Total Requests Processed: {num_finished}\n")
    
    print(f"Average Request Latency:  {avg_total_latency:.3f} s")
    print(f"Average TTFT:             {(avg_waiting + avg_total_prefill) * 1000:.3f} ms")
    print(f"Average TPOT:             {avg_total_decode_per_token * 1000:.3f} ms")
    
    if simulation_latency.bound > 0:
        print(f"Output Token Throughput:  {sum_total_output_tokens / simulation_latency.bound:.3f} token/s")
        print(f"Total Token Throughput:   {sum_total_tokens / simulation_latency.bound:.3f} token/s")

    # 2. Phase Breakdown
    pct_waiting = calc_pct(avg_waiting, avg_total_latency)
    pct_prefill = calc_pct(avg_total_prefill, avg_total_latency)
    pct_decode = calc_pct(avg_total_decode, avg_total_latency)

    print("\n" + "-"*76)
    print(f"{'PHASE BREAKDOWN':^76}")
    print("-"*76)
    print(f"Average Request Latency: {avg_total_latency:6.3f} s (100.00%)")
    print(f"  ├─ Waiting Latency:      {avg_waiting:6.3f} s ({pct_waiting:5.2f}%)")
    print(f"  ├─ Prefill Latency:      {avg_total_prefill:6.3f} s ({pct_prefill:5.2f}%)")
    print(f"  └─ Decode Latency:       {avg_total_decode:6.3f} s ({pct_decode:5.2f}%)")
    
    # 3. Hardware Bottleneck Analysis
    print("\n" + "-"*76)
    print(f"{'HARDWARE BOTTLENECK ANALYSIS':^76}")
    print("-"*76)
    
    print(f"{'Component':<18} | {'Bound(s)':<8} | {'% Total':<8} | {'Compute(s)':<10} | {'Memory(s)':<9} | {'Bottleneck':<10}")
    print("-" * 76)
    
    # Added LLM Head to components list
    components = [
        ("Prefill Attention", avg_p_attn),
        ("Prefill FFN", avg_p_ffn),
        ("Prefill LLM Head", avg_p_llm_head),
        ("Decode Attention", avg_d_attn),
        ("Decode FFN", avg_d_ffn),
        ("Decode LLM Head", avg_d_llm_head)
    ]
    
    total_compute_bound_time = 0.0
    total_memory_bound_time = 0.0
    bottleneck_insights = []

    for name, metrics in components:
        bound, comp, mem = metrics[0], metrics[1], metrics[2]
        pct = calc_pct(bound, avg_total_latency)
        
        # Determine bottleneck
        if comp > mem:
            bottleneck = "COMPUTE"
            total_compute_bound_time += bound
            if pct > 5.0: # Only highlight major bottlenecks
                bottleneck_insights.append(f"- {name} is strongly COMPUTE bound. Higher FLOPS/Tensor Cores will improve this.")
        elif mem > comp:
            bottleneck = "MEMORY"
            total_memory_bound_time += bound
            if pct > 5.0:
                bottleneck_insights.append(f"- {name} is strongly MEMORY bound. Higher memory bandwidth (HBM) will improve this.")
        else:
            bottleneck = "BALANCED"
            
        print(f"{name:<18} | {bound:<8.3f} | {pct:>5.2f}%  | {comp:<10.3f} | {mem:<9.3f} | {bottleneck:<10}")
    
    # 4. Tuning Recommendations
    print("\n" + "-"*76)
    print(f"{'TUNING RECOMMENDATIONS':^76}")
    print("-"*76)
    
    total_active_time = total_compute_bound_time + total_memory_bound_time
    if total_active_time > 0:
        pct_compute_bound = calc_pct(total_compute_bound_time, total_active_time)
        pct_memory_bound = calc_pct(total_memory_bound_time, total_active_time)
        
        print(f"Overall Active Execution Profile: {pct_compute_bound:.1f}% Compute Bound / {pct_memory_bound:.1f}% Memory Bound\n")
        
        if pct_memory_bound > 60.0:
            print("💡 Observation: Your workload is heavily MEMORY BOUND.")
            print("   Action: To optimize, consider hardware with higher memory bandwidth (e.g., HBM3),")
            print("   or increase your batch size/concurrency to increase arithmetic intensity.")
        elif pct_compute_bound > 60.0:
            print("💡 Observation: Your workload is heavily COMPUTE BOUND.")
            print("   Action: To optimize, you need a GPU with higher raw FLOPS (Compute). Memory bandwidth")
            print("   is currently not your primary limiting factor.")
        else:
            print("💡 Observation: Your workload is fairly BALANCED between compute and memory.")
            print("   Action: Upgrading either compute or memory alone may yield diminishing returns.")
            
        print("")
        for insight in bottleneck_insights:
            print(insight)
            
    print("============================================================================\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Configure hardware and workload for benchmarking/inference.")

    # Hardware configuration
    parser.add_argument(
        "--gpu", 
        type=str, 
        default="h100-sxm", 
        help="GPU preset name to use (default: h100-sxm)"
    )

    # Workload configuration
    parser.add_argument(
        "--model", 
        type=str, 
        default="qwen3-30b", 
        help="Model preset name to use (default: qwen3-30b)"
    )
    parser.add_argument(
        "--num-concurrency", 
        type=int, 
        default=256, 
        help="Number of concurrent requests (default: 256)"
    )
    parser.add_argument(
        "--input-len", 
        type=int, 
        default=128, 
        help="Input sequence length (default: 128)"
    )
    parser.add_argument(
        "--output-len", 
        type=int, 
        default=128, 
        help="Output sequence length (default: 128)"
    )
    parser.add_argument(
        "--random-range-ratio", 
        type=float, 
        default=0.1, 
        help="Random range ratio (default: 0.1)"
    )
    parser.add_argument(
        "--num-requests", 
        type=int, 
        default=12800, 
        help="Total number of requests (default: 12800)"
    )
    parser.add_argument(
        "--max-num-batched-tokens", 
        type=int, 
        default=8192, 
        help="Maximum number of batched tokens (default: 8192)"
    )
    parser.add_argument(
        "--bench-file", 
        type=str, 
        default=None, 
        help="Results from microbenchmarks"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    gpu = PRESET_GPUS[args.gpu]
    model = PRESET_MODELS[args.model]

    # read microbenchmark results
    bench_data = None
    if args.bench_file:
        with open(args.bench_file) as f:
            bench_data = json.load(f)
 
    request_pool = []
    waiting_queue = []
    running_queue = []
    finished_queue = []
    total_output_token = 0
    total_token = 0

    # initialize requests
    for i in range(args.num_requests):
        random_input_len = int(
            random.uniform(args.input_len * (1 - args.random_range_ratio), args.input_len * (1 + args.random_range_ratio))
        )
        random_output_len = int(
            random.uniform(args.output_len * (1 - args.random_range_ratio), args.output_len * (1 + args.random_range_ratio))
        )
        request_pool.append(Request(0, random_input_len, random_output_len, i))
        total_output_token += random_output_len
        total_token += random_output_len + random_input_len

    print("running workload simulation...")
    simulation_latency = default_latency_dict()
    pbar = tqdm(total=args.num_requests)
    while len(request_pool) > 0 or len(running_queue) > 0:
        # schedule requests
        schedule_requests(request_pool, waiting_queue, running_queue, args)

        # forward pass
        forward_latency = estimate_forward_latency(running_queue, bench_data, gpu, model)
        add_latency_dicts(simulation_latency, forward_latency)

        # update request status
        unfinished_id = []
        for i in range(len(running_queue)):
            # update kv cache length
            running_queue[i].kv_len += running_queue[i].cur_input_len

            # update latency
            if running_queue[i].input_len >= running_queue[i].kv_len:
                add_latency_dicts(running_queue[i].latency.prefill, forward_latency)

            else:
                add_latency_dicts(running_queue[i].latency.decode, forward_latency)

            # delete finished request
            if (
                running_queue[i].kv_len
                != running_queue[i].input_len + running_queue[i].output_len - 1
            ):
                unfinished_id.append(i)

            else:
                finished_queue.append(running_queue[i])
                pbar.update(1)

        running_queue = [running_queue[i] for i in unfinished_id]

        
        for i in range(len(waiting_queue)):
            waiting_queue[i].latency.waiting += sum_latencies(forward_latency)

    pbar.close()
    analyze_performance(finished_queue, simulation_latency)