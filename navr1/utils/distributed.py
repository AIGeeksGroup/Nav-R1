"""
Distributed training utilities for Nav-R1
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from typing import Dict, Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)


def setup_distributed(rank: int, world_size: int, backend: str = "nccl"):
    """
    Initialize the distributed environment
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        backend: Communication backend (nccl for GPU, gloo for CPU)
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    # Set the device for this process
    torch.cuda.set_device(rank)
    
    logger.info(f"Process {rank} initialized with device cuda:{rank}")


def cleanup_distributed():
    """Clean up the distributed environment"""
    dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if this is the main process"""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank() -> int:
    """Get the rank of the current process"""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Get the total number of processes"""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def reduce_tensor(tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
    """
    Reduce tensor across all processes
    
    Args:
        tensor: Tensor to reduce
        op: Reduction operation
        
    Returns:
        Reduced tensor
    """
    if not dist.is_initialized():
        return tensor
        
    rt = tensor.clone()
    dist.all_reduce(rt, op=op)
    return rt


def gather_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Gather tensors from all processes
    
    Args:
        tensor: Tensor to gather
        
    Returns:
        Gathered tensor from all processes
    """
    if not dist.is_initialized():
        return tensor
        
    world_size = get_world_size()
    if world_size == 1:
        return tensor
        
    # Gather tensors
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    
    # Concatenate along batch dimension
    return torch.cat(tensor_list, dim=0)


def create_ddp_model(model: torch.nn.Module, device_ids: Optional[list] = None) -> DDP:
    """
    Wrap model with DistributedDataParallel
    
    Args:
        model: Model to wrap
        device_ids: List of device IDs to use
        
    Returns:
        DDP-wrapped model
    """
    if device_ids is None:
        device_ids = [get_rank()]
        
    return DDP(model, device_ids=device_ids, find_unused_parameters=True)


def create_distributed_sampler(dataset, shuffle: bool = True) -> DistributedSampler:
    """
    Create a distributed sampler for the dataset
    
    Args:
        dataset: Dataset to sample from
        shuffle: Whether to shuffle the data
        
    Returns:
        Distributed sampler
    """
    return DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=shuffle
    )


def save_checkpoint_on_main_process(
    checkpoint: Dict[str, Any],
    filepath: str,
    is_main_process: bool = None
):
    """
    Save checkpoint only on the main process
    
    Args:
        checkpoint: Checkpoint dictionary
        filepath: Path to save the checkpoint
        is_main_process: Whether this is the main process (auto-detect if None)
    """
    if is_main_process is None:
        is_main_process = is_main_process()
        
    if is_main_process:
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")


def load_checkpoint_on_main_process(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Load checkpoint only on the main process
    
    Args:
        filepath: Path to the checkpoint file
        
    Returns:
        Checkpoint dictionary or None if not main process
    """
    if is_main_process():
        checkpoint = torch.load(filepath, map_location='cpu')
        logger.info(f"Checkpoint loaded from {filepath}")
        return checkpoint
    return None


def broadcast_checkpoint(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """
    Broadcast checkpoint from main process to all processes
    
    Args:
        checkpoint: Checkpoint dictionary (only valid on main process)
        
    Returns:
        Checkpoint dictionary on all processes
    """
    if not dist.is_initialized():
        return checkpoint
        
    # Create a temporary file to share the checkpoint
    if is_main_process():
        temp_path = f"/tmp/checkpoint_rank_{get_rank()}.pt"
        torch.save(checkpoint, temp_path)
        
    # Wait for main process to save
    dist.barrier()
    
    # Load checkpoint on all processes
    if not is_main_process():
        temp_path = f"/tmp/checkpoint_rank_0.pt"
        checkpoint = torch.load(temp_path, map_location='cpu')
        
    # Clean up temporary file
    if is_main_process():
        os.remove(temp_path)
        
    return checkpoint


def run_distributed_training(
    train_fn: Callable,
    world_size: int,
    backend: str = "nccl",
    **kwargs
):
    """
    Run distributed training across multiple processes
    
    Args:
        train_fn: Training function to run
        world_size: Number of processes
        backend: Communication backend
        **kwargs: Additional arguments to pass to train_fn
    """
    if world_size == 1:
        # Single GPU training
        train_fn(rank=0, world_size=1, **kwargs)
    else:
        # Multi-GPU training
        mp.spawn(
            train_fn,
            args=(world_size, backend, kwargs),
            nprocs=world_size,
            join=True
        )


class DistributedMetrics:
    """Utility class for handling metrics in distributed training"""
    
    def __init__(self):
        self.metrics = {}
        
    def update(self, **kwargs):
        """Update metrics"""
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
            
    def compute(self) -> Dict[str, float]:
        """Compute average metrics across all processes"""
        if not dist.is_initialized():
            return {k: sum(v) / len(v) for k, v in self.metrics.items()}
            
        # Gather metrics from all processes
        gathered_metrics = {}
        for key, values in self.metrics.items():
            # Convert to tensor
            tensor = torch.tensor(sum(values), dtype=torch.float32)
            if torch.cuda.is_available():
                tensor = tensor.cuda()
                
            # Gather from all processes
            gathered_tensor = gather_tensor(tensor)
            gathered_metrics[key] = gathered_tensor.mean().item()
            
        return gathered_metrics
        
    def reset(self):
        """Reset metrics"""
        self.metrics = {}


def setup_logging(rank: int, log_level: str = "INFO"):
    """
    Setup logging for distributed training
    
    Args:
        rank: Process rank
        log_level: Logging level
    """
    if rank == 0:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    else:
        # Suppress logging for non-main processes
        logging.basicConfig(level=logging.ERROR)


def get_optimizer_state_dict(optimizer):
    """Get optimizer state dict for distributed training"""
    if hasattr(optimizer, 'module'):
        # DDP wrapped optimizer
        return optimizer.module.state_dict()
    return optimizer.state_dict()


def load_optimizer_state_dict(optimizer, state_dict):
    """Load optimizer state dict for distributed training"""
    if hasattr(optimizer, 'module'):
        # DDP wrapped optimizer
        optimizer.module.load_state_dict(state_dict)
    else:
        optimizer.load_state_dict(state_dict)
