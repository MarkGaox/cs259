import torch
import torch.nn.utils.prune as prune
from column_combine import *

class GroupPruneMethod(prune.BasePruningMethod):
    PRUNING_TYPE = 'structured'
    dim = -1

    def compute_mask(self, t, default_mask):
        matrix = t.reshape([t.shape[0], -1]).detach().numpy()
        groups = columnCombine(matrix)
        mask = structuredPruneMask(matrix, groups)
        mask = torch.tensor(mask.reshape(t.shape))
        return mask


def group_prune_structured(module, name):
    GroupPruneMethod.apply(module, name)
    return module


# Test how to correctly use custom prune method

class TrivialPruneMethod(prune.BasePruningMethod):
    PRUNING_TYPE = 'structured'
    dim = -1
    def compute_mask(self, t, default_mask):
        matrix = t.reshape([t.shape[0], -1]).detach().numpy()
        mask = np.zeros_like(matrix)
        mask[:, t.shape[1]**2:t.shape[1]**2 + 3] = 1
        #print("matrix shape mask:\n", mask)
        mask = torch.tensor(mask.reshape(t.shape))
        #print("tensor shape mask:\n", mask)
        return mask

def trivial_prune_structured(module, name):
    TrivialPruneMethod.apply(module, name)
    return module
