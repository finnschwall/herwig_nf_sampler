# Add these functions, e.g., after the rational_quadratic_spline function or at the end of the file

import torch
import torch.nn.functional as F
import math # Make sure math is imported

# Ensure searchsorted is defined or imported
# If not already present in your file:
def searchsorted(bin_locations: torch.Tensor, inputs: torch.Tensor, eps: float = 1e-6):
    """
    Finds the indices of the bins where the inputs belong.
    Assumes bin_locations is sorted along the last dimension.
    Inputs and bin_locations should broadcast correctly.
    """
    # Add a small amount for numerical stability
    bin_locations = bin_locations + eps * torch.sign(bin_locations[..., -1] - bin_locations[..., 0])
    # Use torch.searchsorted if available (PyTorch >= 1.8), otherwise manual implementation
    try:
        # torch.searchsorted expects sorted bins along the last dimension
        # and inputs to be broadcastable with bins [..., N] and [...]
        indices = torch.searchsorted(bin_locations, inputs, right=False) # right=False means find the leftmost bin
        # Clamp to valid range [0, num_bins - 1] because searchsorted can return num_bins
        return torch.clamp(indices, 0, bin_locations.shape[-1] - 2) # -2 because we have num_bins+1 edges for num_bins bins
    except (AttributeError, RuntimeError): # Fallback if torch.searchsorted not available or fails
        # Manual searchsorted implementation (simpler version)
        # Expand inputs to match bin_locations shape for comparison
        # This might not be the most efficient but should work for smaller dims
        expanded_inputs = inputs.unsqueeze(-1) # [..., 1]
        expanded_bins = bin_locations # [..., num_bins_edges]
        # Compare: [..., 1] >= [..., num_bins_edges-1] (we compare against left edges of bins)
        mask = expanded_inputs >= expanded_bins[..., :-1] # [..., num_bins]
        # Sum along the last dimension to count how many left edges are <= input
        indices = mask.sum(dim=-1) # [...]
        # Clamp to valid range [0, num_bins - 1]
        return torch.clamp(indices, 0, bin_locations.shape[-1] - 2)


def unconstrained_cubic_spline(
    inputs: torch.Tensor,
    unnorm_derivatives_left: torch.Tensor, # Shape (..., 1) - derivative at x=0
    unnorm_derivatives_right: torch.Tensor, # Shape (..., 1) - derivative at x=1
    unnorm_thetas: torch.Tensor,           # Shape (..., num_bins) - parameters for x-bin widths
    inverse: bool = False,
    left: float = 0.0,
    right: float = 1.0,
    bottom: float = 0.0,
    top: float = 1.0,
    min_derivative: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """
    Wrapper for cubic spline that handles inputs outside [left, right] by mapping them to themselves.
    Assumes left=bottom, right=top for the uniform case.
    """
    # For unconstrained, we assume inputs outside [0,1] map to themselves
    inside_interval_mask = (inputs >= left) & (inputs <= right)

    outputs = inputs.clone()
    logabsdet = torch.zeros_like(inputs)

    if inside_interval_mask.any():
        outputs_inside, logabsdet_inside, max_deriv, mean_deriv = cubic_spline(
            inputs=inputs[inside_interval_mask],
            unnorm_derivatives_left=unnorm_derivatives_left[inside_interval_mask],
            unnorm_derivatives_right=unnorm_derivatives_right[inside_interval_mask],
            unnorm_thetas=unnorm_thetas[inside_interval_mask],
            inverse=inverse,
            left=left,
            right=right,
            bottom=bottom,
            top=top,
            min_derivative=min_derivative,
        )
        outputs[inside_interval_mask] = outputs_inside
        logabsdet[inside_interval_mask] = logabsdet_inside

    return outputs, logabsdet, max_deriv, mean_deriv

def cubic_spline(
    inputs: torch.Tensor,              # Shape (...,)
    unnorm_derivatives_left: torch.Tensor, # Shape (..., 1)
    unnorm_derivatives_right: torch.Tensor, # Shape (..., 1)
    unnorm_thetas: torch.Tensor,       # Shape (..., num_bins)
    inverse: bool = False,
    left: float = 0.0,                 # Input lower bound (typically 0)
    right: float = 1.0,                # Input upper bound (typically 1)
    bottom: float = 0.0,               # Output lower bound (typically 0)
    top: float = 1.0,                  # Output upper bound (typically 1)
    min_derivative: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """
    Implements a monotonic cubic spline transformation for the [0,1] -> [0,1] case.
    The transformation is divided into 'num_bins' segments.
    The derivative at the left (0) and right (1) boundaries are parameterized.
    The positions of the internal bin boundaries (thetas) are parameterized.
    Heights are determined by the constraint that the total output range is [0,1].
    """
    if not inputs.is_floating_point():
        raise TypeError("Inputs must be a floating point tensor.")

    # --- Shape Assertions ---
    original_shape = inputs.shape
    if unnorm_derivatives_left.shape[-1] != 1 or unnorm_derivatives_right.shape[-1] != 1:
        raise ValueError("Derivatives must have shape (..., 1)")
    if len(unnorm_derivatives_left.shape) != len(inputs.shape) + 1 or len(unnorm_derivatives_right.shape) != len(inputs.shape) + 1:
         # Handle broadcasting if needed, but shapes should ideally match
         # Reshape if inputs is missing a trailing dimension
         if len(inputs.shape) < len(unnorm_derivatives_left.shape[:-1]):
             inputs = inputs.unsqueeze(-1) # Add dimension if needed for consistency in processing

    num_bins = unnorm_thetas.shape[-1]

    if num_bins < 1:
        raise ValueError("Number of bins must be at least 1.")

    # Flatten all dimensions for easier processing, keep last dim separate if needed
    # But since inputs is (...,) and params are (..., 1) or (..., num_bins),
    # it's better to process them as is, ensuring broadcasting works.
    # Let's assume inputs is (B, D) and params are (B, D, 1) or (B, D, num_bins)
    # after reshape in transform. So we work with (B*D, ...) shapes.
    # But to simplify, let's assume inputs is (N,) and params are (N, 1) or (N, num_bins)
    # after masking and reshaping in `transform`.

    # --- Parameterization ---
    # Bound derivatives to ensure monotonicity and minimum slope
    derivatives_left = min_derivative + F.softplus(unnorm_derivatives_left.squeeze(-1))  # (N,)
    derivatives_right = min_derivative + F.softplus(unnorm_derivatives_right.squeeze(-1)) # (N,)

    # Determine bin widths (x-coordinates) via softmax
    # thetas_raw represent the *relative* widths
    thetas_raw = F.softmax(unnorm_thetas, dim=-1) # (N, num_bins)
    thetas_cumsum = torch.cumsum(thetas_raw, dim=-1) # (N, num_bins)
    # Pad to get cumulative x positions [0, theta_1, ..., theta_{num_bins-1}, 1]
    thetas_cumsum_padded = F.pad(thetas_cumsum, (1, 0), value=0.0) # (N, num_bins + 1)
    thetas = torch.cat([thetas_cumsum_padded, torch.ones_like(thetas_cumsum_padded[:, :1])], dim=-1) # (N, num_bins + 2)
    # Ensure last element is exactly 1.0
    thetas[:, -1] = 1.0
    widths = thetas[:, 1:] - thetas[:, :-1] # (N, num_bins + 1)

    # --- Determine y-positions (etas) ---
    # For simplicity and to decouple shape from derivatives (as discussed),
    # we use the same binning structure for y as for x.
    # This means heights h_k = y_{k+1} - y_k are proportional to widths w_k.
    # To ensure sum(h_k) = 1, we normalize.
    # h_k = w_{k+1} / sum_{j=1}^{num_bins} w_j. // Check indexing.
    # Actually, widths w_k = x_{k+1} - x_k. Heights h_k = y_{k+1} - y_k.
    # If we want h_k = w_k (for shape), then sum h_k = sum w_k = 1 (if x in [0,1]).
    # So if x positions are determined by softmax, y positions can be identical.
    # This simplifies the implementation significantly.
    etas = thetas # (N, num_bins + 2)
    heights = etas[:, 1:] - etas[:, :-1] # (N, num_bins + 1)

    # --- Map input to bin index ---
    if inverse:
        bin_idx = searchsorted(etas, inputs.unsqueeze(-1)) # Find y-bin index for each input y
    else:
        bin_idx = searchsorted(thetas, inputs.unsqueeze(-1)) # Find x-bin index for each input x
    bin_idx = bin_idx.squeeze(-1) # (N,)

    # --- Gather parameters for the specific bin ---
    # bin_idx is (N,)
    bin_idx_expanded = bin_idx.unsqueeze(-1) # (N, 1)

    input_thetas = torch.gather(thetas, -1, bin_idx_expanded).squeeze(-1)          # (N,)
    input_widths = torch.gather(widths, -1, bin_idx_expanded).squeeze(-1)          # (N,)
    next_thetas = torch.gather(thetas, -1, bin_idx_expanded + 1).squeeze(-1)       # (N,)
    # next_widths not directly needed for cubic spline calc with this approach

    input_heights = torch.gather(heights, -1, bin_idx_expanded).squeeze(-1)      # (N,)
    input_etas = torch.gather(etas, -1, bin_idx_expanded).squeeze(-1)            # (N,)
    next_etas = torch.gather(etas, -1, bin_idx_expanded + 1).squeeze(-1)         # (N,)

    # --- Derive intermediate derivatives ---
    # Simplification: Use boundary derivatives to define all.
    # Average derivative for intermediate points.
    avg_derivative = (derivatives_left + derivatives_right) / 2.0 # (N,)
    # Create full derivative array (simplified)
    # We need derivatives at num_bins+1 points: 0 to num_bins.
    # We have d_0 and d_num_bins. Interpolate or fix internal ones.
    # Quick fix: Assume internal derivatives are the average.
    all_derivatives = torch.full((inputs.shape[0], num_bins + 1), avg_derivative.mean().item(), device=inputs.device, dtype=inputs.dtype) # (N, num_bins+1)
    # Override boundary derivatives
    all_derivatives[:, 0] = derivatives_left # (N,)
    all_derivatives[:, -1] = derivatives_right # (N,)
    # Interpolate linearly between boundaries for internal points
    if num_bins > 1:
        t_interp = torch.linspace(0.0, 1.0, num_bins + 1, device=inputs.device) # (num_bins+1,)
        all_derivatives = derivatives_left.unsqueeze(1) * (1 - t_interp) + derivatives_right.unsqueeze(1) * t_interp # (N, num_bins+1)

    input_derivatives = torch.gather(all_derivatives, -1, bin_idx_expanded).squeeze(-1)      # (N,)
    next_derivatives = torch.gather(all_derivatives, -1, bin_idx_expanded + 1).squeeze(-1)   # (N,)

    # --- Normalize derivatives by segment width for numerical stability ---
    delta_approx = (next_etas - input_etas) / (next_thetas - input_thetas + 1e-12) # dy/dx approx
    # Avoid division by zero or very small widths
    delta_approx = torch.clamp(delta_approx, min=1e-6)
    input_derivatives_scaled = input_derivatives / delta_approx # (N,)
    next_derivatives_scaled = next_derivatives / delta_approx   # (N,)

    # --- Cubic Hermite Spline Calculation ---
    if inverse:
        # --- Inverse Transformation (y -> x) ---
        y_val = inputs # (N,)
        y0 = input_etas # (N,)
        y1 = next_etas # (N,)
        m0 = input_derivatives_scaled * input_widths # (N,)
        m1 = next_derivatives_scaled * input_widths # (N,)

        # Initial guess for t (normalized x within the bin)
        t_guess = (y_val - y0) / (y1 - y0 + 1e-12) # (N,)
        t_guess = torch.clamp(t_guess, 0.0, 1.0) # (N,)

        # Newton-Raphson iterations
        t = t_guess # (N,)
        for _ in range(5):
            t_clamp = torch.clamp(t, 0.0, 1.0) # (N,)
            t2 = t_clamp * t_clamp
            t3 = t2 * t_clamp
            # Cubic Hermite polynomial for y(t)
            h_val = (2 * t3 - 3 * t2 + 1) * y0 + (t3 - 2 * t2 + t_clamp) * m0 + (-2 * t3 + 3 * t2) * y1 + (t3 - t2) * m1
            # Derivative dy/dt
            dh_dt = (6 * t2 - 6 * t_clamp) * y0 + (3 * t2 - 4 * t_clamp + 1) * m0 + (-6 * t2 + 6 * t_clamp) * y1 + (3 * t2 - 2 * t_clamp) * m1

            # Update t
            # Avoid division by zero in derivative
            dh_dt_safe = torch.where(dh_dt.abs() > 1e-10, dh_dt, torch.sign(dh_dt) * 1e-10)
            t = t - (h_val - y_val) / dh_dt_safe
            t = torch.clamp(t, 0.0, 1.0) # Ensure t stays within [0, 1]

        t_final = torch.clamp(t, 0.0, 1.0) # (N,)
        x_val = input_thetas + t_final * input_widths # (N,)
        outputs = x_val * (right - left) + left # Map back (though left=0, right=1)

        # --- Calculate logabsdet for inverse ---
        dy_dx = dh_dt / (input_widths + 1e-12) # (N,)
        dy_dx = torch.clamp(dy_dx.abs(), min=min_derivative) # Ensure positive and min value
        logabsdet = -torch.log(dy_dx) # (N,)

        max_deriv = max(derivatives_left.max().item(), derivatives_right.max().item())
        mean_deriv = (derivatives_left.mean() + derivatives_right.mean()).item() / 2

        return outputs.view(original_shape), logabsdet.view(original_shape), max_deriv, mean_deriv

    else:
        # --- Forward Transformation (x -> y) ---
        x_val = inputs # (N,)
        t = (x_val - input_thetas) / (input_widths + 1e-12) # (N,)
        t = torch.clamp(t, 0.0, 1.0) # (N,)

        t2 = t * t
        t3 = t2 * t
        # Cubic Hermite basis functions
        h00 = 2 * t3 - 3 * t2 + 1 # (N,)
        h10 = t3 - 2 * t2 + t     # (N,)
        h01 = -2 * t3 + 3 * t2    # (N,)
        h11 = t3 - t2             # (N,)

        # Boundary values and scaled derivatives
        h0 = input_etas # (N,)
        h1 = next_etas # (N,)
        m0 = input_derivatives_scaled * input_widths # (N,)
        m1 = next_derivatives_scaled * input_widths # (N,)

        # Calculate output y
        y_val = h00 * h0 + h10 * m0 + h01 * h1 + h11 * m1 # (N,)
        outputs = y_val * (top - bottom) + bottom # Map (though bottom=0, top=1)

        # --- Calculate logabsdet for forward ---
        # dy/dt
        dh_dt = (6 * t2 - 6 * t) * h0 + (3 * t2 - 4 * t + 1) * m0 + (-6 * t2 + 6 * t) * h1 + (3 * t2 - 2 * t) * m1 # (N,)
        dx_dt = input_widths # (N,) # dt/dx = 1 / w_k => dx/dt = w_k
        dy_dx = dh_dt / (input_widths + 1e-12) # (N,)
        dy_dx = torch.clamp(dy_dx.abs(), min=min_derivative) # Ensure positive and min value
        logabsdet = torch.log(dy_dx) # (N,)

        max_deriv = max(derivatives_left.max().item(), derivatives_right.max().item())
        mean_deriv = (derivatives_left.mean() + derivatives_right.mean()).item() / 2

        return outputs.view(original_shape), logabsdet.view(original_shape), max_deriv, mean_deriv
