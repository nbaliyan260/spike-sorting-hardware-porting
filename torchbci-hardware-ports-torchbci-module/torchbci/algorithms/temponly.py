from torchbci.block import Filter, Detection, Alignment, Block, TemplateMatching, Clustering
import torch
import numpy as np
import torch.nn.functional as F
from typing import List, Tuple

class TemplateOnlyAlgorithm(Block):
    pass