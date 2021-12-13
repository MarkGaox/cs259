import torch
import torch.nn.utils.prune as prune
import numpy as np
from column_combine import *
import math

class GroupPruneMethod(prune.BasePruningMethod):
    PRUNING_TYPE = 'structured'
    dim = -1
    array_size = 8

    def compute_mask(self, t, default_mask):
        matrix = t.reshape([t.shape[0], -1]).detach().numpy()
        if matrix.shape[1] <= self.array_size:
            matrixes = [matrix]
        else:
            remain = matrix.shape[1] % self.array_size
            matrixes = np.hsplit(matrix[:, :matrix.shape[1] - remain], matrix.shape[1] // self.array_size)
            if not remain == 0:
                matrixes.append(matrix[:, matrix.shape[1] - remain:])
        masks = []
        for matrix in matrixes:
            groups = columnCombine(matrix)
            mask = structuredPruneMask(matrix, groups)
            masks.append(mask)
        mask = np.hstack(masks)
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
