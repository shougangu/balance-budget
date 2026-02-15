import gc
import torch


def cleanup_gpu(destroy_vllm=False):
    """Run garbage collection and clear CUDA cache to free GPU memory.

    Callers must `del` their own references before calling this,
    since Python's `del` only removes references in the current scope.

    Args:
        destroy_vllm: If True, also destroy vLLM model parallel state.
    """
    if destroy_vllm:
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
