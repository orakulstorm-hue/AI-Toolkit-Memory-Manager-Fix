import torch
from .manager_modules import LinearLayerMemoryManager, ConvLayerMemoryManager

LINEAR_MODULES = ["Linear", "LoRACompatibleLinear", "QLinear"]
CONV_MODULES = ["Conv2d", "LoRACompatibleConv", "QConv2d"]
UNMANAGED_MODULES = ["LayerNorm", "BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "GroupNorm", "InstanceNorm1d", "InstanceNorm2d", "InstanceNorm3d", "Embedding", "EmbeddingBag", "RNNBase", "LSTM", "GRU", "RNN", "Conv3d"]
UNMANAGED_MODULES_INCLUDES = ["RotaryEmbedding", "Norm", "RotaryPosEmbed"]

class MemoryManager:
    def __init__(self, module: torch.nn.Module, process_device: torch.device = torch.device("cpu")):
        self.module = module
        self.process_device = process_device
        self.unmanaged_modules = []

    def memory_managed_to(self, *args, **kwargs):
        for m in self.unmanaged_modules:
            if isinstance(m, torch.nn.Parameter):
                m.data = m.data.to(*args, **kwargs)
            else:
                m.to(*args, **kwargs)
        dtype = kwargs.get("dtype")
        if not dtype and args:
            for a in args:
                if isinstance(a, torch.dtype):
                    dtype = a
                    break
        if dtype:
            return self.module._mm_to(dtype=dtype)
        return self.module

    @classmethod
    def attach(cls, module, device, offload_percent=1.0, ignore_modules=[]):
        if hasattr(module, "_memory_manager"): return
        module._memory_manager = cls(module, device)
        module._mm_to = module.to
        module.to = module._memory_manager.memory_managed_to
        
        proc = [x for x in ignore_modules]
        all_m = []
        for n, m in module.named_modules():
            if (m.__class__.__name__ in LINEAR_MODULES or m.__class__.__name__ in CONV_MODULES) and m not in ignore_modules:
                if m not in all_m: all_m.append(m)
        
        total = len(all_m)
        curr = 0
        for n, m in module.named_modules():
            if m in proc: continue
            name = m.__class__.__name__
            if m in all_m:
                skip = False
                if offload_percent < 1.0 and (curr / max(1, total)) > offload_percent:
                    skip = True
                if skip:
                    module._memory_manager.unmanaged_modules.append(m)
                elif name in LINEAR_MODULES:
                    LinearLayerMemoryManager.attach(m, module._memory_manager)
                else:
                    ConvLayerMemoryManager.attach(m, module._memory_manager)
                proc.append(m)
                curr += 1
            elif name in UNMANAGED_MODULES or any(inc in name for inc in UNMANAGED_MODULES_INCLUDES):
                module._memory_manager.unmanaged_modules.append(m)
                proc.append(m)