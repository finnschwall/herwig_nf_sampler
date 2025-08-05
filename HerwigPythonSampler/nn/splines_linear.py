import math

import torch
import torch.nn.functional as F


# TODO: replace with torch.searchsorted
def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1




def linear_spline(
    inputs: torch.Tensor,
    unnormalized_widths: torch.Tensor,
    unnormalized_heights: torch.Tensor,
    inverse: bool = False,
    left: float = 0.0,
    right: float = 1.0,
    bottom: float = 0.0,
    top: float = 1.0,
    min_bin_width: float = 1e-3, # Not strictly needed for linear, but kept for signature consistency
    min_bin_height: float = 1e-3, # Not strictly needed for linear, but kept for signature consistency
    # min_derivative is not relevant for linear splines
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """
    Constrained linear spline transformation.
    The input points have to be within the spline boundaries [left, right].

    Note: This implementation assumes widths and heights are positive and sum to (right-left)
    and (top-bottom) respectively, which should be ensured by the preprocessing in
    unconstrained_linear_spline.

    Args:
        inputs: input tensor, shape (..., )
        unnormalized_widths: unnormalized spline bin widths, shape (..., n_bins)
        unnormalized_heights: unnormalized spline bin heights, shape (..., n_bins)
        inverse: if True, perform inverse transformation
        left: lower bound of inputs
        right: upper bound of inputs
        bottom: lower bound of outputs
        top: upper bound of outputs
        min_bin_width: (ignored for linear, kept for consistency)
        min_bin_height: (ignored for linear, kept for consistency)

    Returns:
        tuple containing:
        - output tensor with shape (..., )
        - log-Jacobian determinants with shape (..., )
        - dummy max_derivative (0.0 for linear)
        - dummy mean_derivative (0.0 for linear)
    """
    num_bins = unnormalized_widths.shape[-1]

    # Normalize widths and heights to sum to the total range
    widths = F.softmax(unnormalized_widths, dim=-1) * (right - left)
    heights = F.softmax(unnormalized_heights, dim=-1) * (top - bottom)

    # Ensure minimum width/height if strictly necessary (though softmax should prevent zeros)
    # widths = torch.clamp(widths, min=min_bin_width)
    # heights = torch.clamp(heights, min=min_bin_height)

    # Calculate cumulative sums for bin edges
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    cumwidths = cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right

    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    cumheights = cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top

    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    # Gather values for the identified bin
    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]
    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    input_bin_heights = heights.gather(-1, bin_idx)[..., 0]

    # Prevent division by zero if a bin width somehow becomes zero
    input_bin_widths = torch.clamp(input_bin_widths, min=1e-12)

    if inverse:
        # y -> x
        # y = y0 + (x - x0) * (h / w)
        # => x = x0 + (y - y0) * (w / h)
        slope = input_bin_widths / input_bin_heights
        outputs = input_cumwidths + (inputs - input_cumheights) * slope
        # Clamp outputs to be strictly within [left, right] if needed
        outputs = torch.clamp(outputs, min=left, max=right)
    else:
        # x -> y
        # y = y0 + (x - x0) * (h / w)
        slope = input_bin_heights / input_bin_widths
        outputs = input_cumheights + (inputs - input_cumwidths) * slope
        # Clamp outputs to be strictly within [bottom, top] if needed
        outputs = torch.clamp(outputs, min=bottom, max=top)

    # Log absolute determinant of Jacobian is log(|slope|)
    logabsdet = torch.log(torch.abs(slope))

    # Return dummy derivative values for monitoring (linear splines have constant slope per bin)
    dummy_max_derivative = 0.0
    dummy_mean_derivative = 0.0

    return outputs, logabsdet, dummy_max_derivative, dummy_mean_derivative


def unconstrained_linear_spline(
    inputs: torch.Tensor,
    unnormalized_widths: torch.Tensor,
    unnormalized_heights: torch.Tensor,
    inverse: bool = False,
    left: float = 0.0,
    right: float = 1.0,
    bottom: float = 0.0,
    top: float = 1.0,
    min_bin_width: float = 1e-3, # Kept for consistency
    min_bin_height: float = 1e-3, # Kept for consistency
    # min_derivative is not relevant
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """
    Unconstrained linear spline transformations.
    Points outside the bounds [left, right] are mapped onto themselves with logabsdet=0.
    """
    if not inverse:
        inside_interval_mask = (inputs >= left) & (inputs <= right)
    else:
        inside_interval_mask = (inputs >= bottom) & (inputs <= top)

    outside_interval_mask = ~inside_interval_mask.squeeze(-1) # Assuming inputs is (..., 1)
    # Handle multi-dim inputs if necessary, adjust mask creation accordingly
    # For now, assuming it works like the RQ case or inputs are processed dim-wise

    # Initialize outputs and logabsdet
    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    # Pass through points outside the interval unchanged
    outputs[outside_interval_mask] = inputs[outside_interval_mask]
    logabsdet[outside_interval_mask] = 0.0

    # Transform points inside the interval
    if inside_interval_mask.any():
        (
            outputs[inside_interval_mask],
            logabsdet[inside_interval_mask],
            max_derivative, # Will be dummy 0.0
            mean_derivative, # Will be dummy 0.0
        ) = linear_spline(
            inputs=inputs[inside_interval_mask],
            unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
            unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
            inverse=inverse,
            left=left,
            right=right,
            bottom=bottom,
            top=top,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
        )
    else:
        # If nothing is inside, dummy values
        max_derivative = 0.0
        mean_derivative = 0.0

    return outputs, logabsdet, max_derivative, mean_derivative