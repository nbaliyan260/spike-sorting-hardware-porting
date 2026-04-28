from __future__ import annotations

from typing import Optional

from torchbci.block.base.base_block import BaseBlock


class MountainSort5Block(BaseBlock):
    """
    Base block for MountainSort5.
    Keeps the same chain-of-responsibility style used in the other ported algorithms.
    """

    def __init__(self) -> None:
        super().__init__()
        self.next_block: Optional["MountainSort5Block"] = None

    def set_next(self, next_block: "MountainSort5Block") -> "MountainSort5Block":
        self.next_block = next_block
        return next_block

    def run_block(self, batch):
        raise NotImplementedError("Subclasses must implement run_block().")

    def forward(self, batch):
        batch = self.run_block(batch)
        if self.next_block is not None:
            return self.next_block(batch)
        return batch