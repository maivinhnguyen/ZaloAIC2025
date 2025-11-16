# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
Siamese neural network modules for one-shot object detection.

This module contains components for the SiamYOLOv8 model, including the Matching Module
which implements the fusion logic from the SiamYOLOv8 paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MatchingModule(nn.Module):
    """
    Parameter-free Matching Module for SiamYOLOv8.

    This module implements the fusion logic from Equation 6 of the SiamYOLOv8 paper:
    Output = Q + s(Q × S) × S

    Where:
    - Q is the query feature map
    - S is the support feature map
    - × denotes element-wise multiplication
    - s denotes a softmax activation
    - + denotes element-wise addition

    This module has no learnable parameters and operates on feature maps of arbitrary dimensions.

    Args:
        None

    Shape:
        - Input: Two tensors of shape (B, C, H, W) for query and support
        - Output: Tensor of shape (B, C, H, W) representing fused features

    Examples:
        >>> mm = MatchingModule()
        >>> query = torch.randn(4, 256, 40, 40)
        >>> support = torch.randn(4, 256, 40, 40)
        >>> fused = mm(query, support)
        >>> fused.shape
        torch.Size([4, 256, 40, 40])
    """

    def __init__(self):
        """Initialize the MatchingModule (parameter-free)."""
        super().__init__()

    def forward(self, query: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementing the fusion logic: Q + s(Q × S) × S.

        Args:
            query (torch.Tensor): Query feature map of shape (B, C, H, W)
            support (torch.Tensor): Support feature map of shape (B, C, H, W)

        Returns:
            torch.Tensor: Fused feature map of shape (B, C, H, W)

        Notes:
            The formula is: Output = Query + Softmax(Query * Support) * Support
            Softmax is applied over the spatial dimension (H*W) for each channel independently.
        """
        # Element-wise multiplication between query and support
        element_mult = query * support

        # Apply softmax activation over spatial dimensions (H*W)
        # For each batch and channel, compute softmax across spatial locations
        B, C, H, W = element_mult.shape
        element_mult_flat = element_mult.view(B, C, -1)  # (B, C, H*W)
        attention_flat = F.softmax(element_mult_flat, dim=2)  # Softmax over spatial dimension (H*W)
        attention = attention_flat.view(B, C, H, W)  # Reshape back

        # Weight the support features by the attention map
        weighted_support = attention * support

        # Add the weighted support to the original query
        output = query + weighted_support

        return output
