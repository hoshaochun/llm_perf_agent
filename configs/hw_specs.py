"""Hardware and cluster configuration specifications."""

from dataclasses import dataclass, field
from typing import Optional
import json


@dataclass
class CommunicationSpec:
    """Communication bandwidth and overhead specifications."""
    
    collective_bandwidth: float  # bytes/second (e.g., all-reduce)
    p2p_bandwidth: float  # bytes/second (point-to-point)
    collective_overhead: float = 0.0  # seconds (latency overhead)
    p2p_overhead: float = 0.0  # seconds (latency overhead)
    
    def collective_time(self, data_bytes: int) -> float:
        """Estimate collective communication time in seconds."""
        if self.collective_bandwidth == 0:
            print("Warning: collective_bandwidth is 0, returning 0.0 time.")
            return 0.0  # No inter-node communication (single node)
        return self.collective_overhead + data_bytes / self.collective_bandwidth
    
    def p2p_time(self, data_bytes: int) -> float:
        """Estimate point-to-point communication time in seconds."""
        if self.p2p_bandwidth == 0:
            print("Warning: p2p_bandwidth is 0, returning 0.0 time.")
            return 0.0  # No inter-node communication (single node)
        return self.p2p_overhead + data_bytes / self.p2p_bandwidth


@dataclass
class HardwareConfig:
    """Configuration for a single GPU."""
    
    name: str
    bf16_flops: float  # FLOPs per second for bf16
    mem_bandwidth: float  # bytes per second
    mem_capacity: float  # bytes
    fp8_flops: float = 0.0  # FLOPs per second for fp8
    fp4_flops: float = 0.0  # FLOPs per second for fp4
    mxfp4_flops: float = 0.0 # FLOPs per second for mxfp4


@dataclass
class NodeConfig:
    """Configuration for a single node (server)."""
    
    gpu: HardwareConfig
    n_gpus: int
    intra_comm: CommunicationSpec
    node_id: int = 0
    
    @property
    def total_mem_capacity(self) -> float:
        """Total GPU memory capacity on this node."""
        return self.gpu.mem_capacity * self.n_gpus


@dataclass 
class ClusterConfig:
    """Configuration for a multi-node cluster."""
    
    nodes: list[NodeConfig]
    inter_comm: CommunicationSpec
    
    @property
    def total_gpus(self) -> int:
        """Total number of GPUs across all nodes."""
        return sum(node.n_gpus for node in self.nodes)
    
    @property
    def n_nodes(self) -> int:
        """Number of nodes in the cluster."""
        return len(self.nodes)
    
    @property
    def total_mem_capacity(self) -> float:
        """Total GPU memory capacity across all nodes."""
        return sum(node.total_mem_capacity for node in self.nodes)
    
    def gpus_per_node(self) -> list[int]:
        """List of GPU counts per node."""
        return [node.n_gpus for node in self.nodes]
    
    @classmethod
    def from_homogeneous(
        cls,
        gpu: HardwareConfig,
        n_nodes: int,
        gpus_per_node: int,
        intra_comm: CommunicationSpec,
        inter_comm: CommunicationSpec,
    ) -> "ClusterConfig":
        """Create a cluster with identical nodes."""
        nodes = [
            NodeConfig(
                gpu=gpu,
                n_gpus=gpus_per_node,
                intra_comm=intra_comm,
                node_id=i,
            )
            for i in range(n_nodes)
        ]
        return cls(nodes=nodes, inter_comm=inter_comm)
    
    def to_dict(self) -> dict:
        """Export cluster config to dictionary."""
        return {
            "n_nodes": self.n_nodes,
            "total_gpus": self.total_gpus,
            "nodes": [
                {
                    "node_id": node.node_id,
                    "gpu": node.gpu.name,
                    "n_gpus": node.n_gpus,
                }
                for node in self.nodes
            ],
        }


# Preset GPU configurations
PRESET_GPUS = {
    "4090": HardwareConfig(
        name="4090",
        bf16_flops=165.2e12,  # 165.2 TFLOPS
        mem_bandwidth=1008e9,  # 1008 GB/s
        mem_capacity=24e9,  # 24 GB
        fp8_flops=330.3e12,  # ~2x BF16 (Ada Lovelace Tensor Cores)
        fp4_flops=0,  # Not supported natively
    ),
    "h100-sxm": HardwareConfig(
        name="h100-sxm",
        bf16_flops=989e12,  # 989 TFLOPS (with sparsity: 1979)
        mem_bandwidth=3350e9,  # 3.35 TB/s
        mem_capacity=80e9,  # 80 GB
        fp8_flops=3958e12,  # 3958 TFLOPS (dense)
        fp4_flops=0,  # Blackwell feature
    ),
    "h200-nvl": HardwareConfig(
        name="h200-nvl",
        bf16_flops=835e12,
        mem_bandwidth=4800e9,
        mem_capacity=141e9,
        fp8_flops=1671e12,
        fp4_flops=0,
    ),
    "h100-pcie": HardwareConfig(
        name="h100-pcie",
        bf16_flops=756e12,
        mem_bandwidth=2000e9,
        mem_capacity=80e9,
        fp8_flops=1513e12,
        fp4_flops=0,
    ),
    "a100-sxm": HardwareConfig(
        name="a100-sxm",
        bf16_flops=624e12,
        mem_bandwidth=2039e9,
        mem_capacity=80e9,
        fp8_flops=624e12,  # No native FP8, fallback to BF16 speed or emulated
        fp4_flops=0,
    ),
    "a100-pcie": HardwareConfig(
        name="a100-pcie",
        bf16_flops=624e12,
        mem_bandwidth=1935e9,
        mem_capacity=80e9,
        fp8_flops=624e12,
        fp4_flops=0,
    ),
    "b200": HardwareConfig(
        name="b200",
        bf16_flops=2.25e15,
        mem_bandwidth=8e12,
        mem_capacity=180e9,
        fp8_flops=4.5e15,
        fp4_flops=9e15,
    ),
}

# Preset communication specs
PRESET_INTRA_COMM = {
    "nvlink-4": CommunicationSpec(  # NVLink 4.0 (H100)
        collective_bandwidth=900e9,  # ~900 GB/s
        p2p_bandwidth=900e9,
        collective_overhead=5e-6,  # 5 microseconds
        p2p_overhead=2e-6,
    ),
    "nvlink-3": CommunicationSpec(  # NVLink 3.0 (A100)
        collective_bandwidth=600e9,  # ~600 GB/s
        p2p_bandwidth=600e9,
        collective_overhead=5e-6,
        p2p_overhead=2e-6,
    ),
    "pcie-4": CommunicationSpec(  # PCIe 4.0
        collective_bandwidth=32e9,  # ~32 GB/s
        p2p_bandwidth=32e9,
        collective_overhead=10e-6,
        p2p_overhead=5e-6,
    ),
    "pcie-5": CommunicationSpec(  # PCIe 5.0
        collective_bandwidth=64e9,  # ~64 GB/s
        p2p_bandwidth=64e9,
        collective_overhead=8e-6,
        p2p_overhead=4e-6,
    ),
}

PRESET_INTER_COMM = {
    "infiniband-hdr": CommunicationSpec(  # InfiniBand HDR (200 Gbps)
        collective_bandwidth=25e9,  # ~25 GB/s
        p2p_bandwidth=25e9,
        collective_overhead=50e-6,  # 50 microseconds
        p2p_overhead=20e-6,
    ),
    "infiniband-ndr": CommunicationSpec(  # InfiniBand NDR (400 Gbps)
        collective_bandwidth=50e9,  # ~50 GB/s
        p2p_bandwidth=50e9,
        collective_overhead=40e-6,
        p2p_overhead=15e-6,
    ),
    "ethernet-100g": CommunicationSpec(  # 100GbE
        collective_bandwidth=12.5e9,  # ~12.5 GB/s
        p2p_bandwidth=12.5e9,
        collective_overhead=100e-6,
        p2p_overhead=50e-6,
    ),
}

