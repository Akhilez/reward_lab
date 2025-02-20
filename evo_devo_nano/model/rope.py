import torch


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    """
    freqs is of shape (dim // 2,)
    It starts at 1 and decreases exponentially to 1 / theta.
    For example, when dim = 256,
    freqs = [1, 0.93, 0.86, ... , ~1 / theta]
    """

    t = torch.arange(end, device=freqs.device)  # type: ignore  # (end,)
    freqs = torch.outer(t, freqs).float()  # type: ignore  # (end, dim // 2)
    """
    freqs is of shape (end, dim // 2)
    It is the outer product of t and freqs.
    t = [0, 1, 2, ..., end - 1]
    freqs (before) = [1, 0.93, 0.86, ... , ~1 / theta]
    freqs (after) = [
        0 * [1, 0.93, 0.86, ... , ~1 / theta],
        1 * [1, 0.93, 0.86, ... , ~1 / theta],
        2 * [1, 0.93, 0.86, ... , ~1 / theta],
        ...,
        (end - 1) * [1, 0.93, 0.86, ... , ~1 / theta],
    ]
    """

    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    """
    freqs_cis is of shape (end, dim // 2), but is a complex tensor.
    freqs_cis.real is of shape (end, dim // 2) and of dtype float32.
    freqs_cis.imag is of shape (end, dim // 2) and of dtype float32.

    freqs_cis.real = cos(freqs)
    freqs_cis.imag = sin(freqs)
    """
    return freqs_cis


def apply_rotary_emb(
    x: torch.Tensor,  # (b, nh, l, hd)
    freqs_cis: torch.Tensor,  # (l, hd // 2)
) -> torch.Tensor:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        x (torch.Tensor): Tensor to apply rotary embeddings. (b, nh, l, hd)
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.  (l, hd // 2)

    Returns:
        torch.Tensor: Modified tensor with rotary embeddings.
        shape: (b, nh, l, hd)

    """
    # Split into consecutive pairs
    x_ = x.float().reshape(*x.shape[:-1], -1, 2)  # (b, nh, l, hd // 2, 2)

    # Convert to complex numbers
    x_ = torch.view_as_complex(x_)  # (b, nh, l, hd // 2)
    """
    This will split the last dimension into two tensors -- real and imag parts of a complex tensor.
    (b, nh, l, hd // 2, 2) -> (b, nh, l, hd // 2)
    First of the two is the real part, second is the imaginary part.
    To get the original back, we can do
        torch.cat([xq_.real.unsqueeze(-1), xq_.imag.unsqueeze(-1)], dim=-1)
        or torch.view_as_real(xq_)
    """

    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)  # (l, hd // 2) -> (1, 1, l, hd // 2)

    # Complex number multiplication: (a + ib) (c + id) = (ac - bd) + i(ad + bc)
    x_rotated = x_ * freqs_cis  # (b, nh, l, hd // 2)
    """
    xq_rotated = (
        (xq_.real * freqs_cis.real - xq_.imag * freqs_cis.imag)
        + i(xq_.real * freqs_cis.imag + xq_.imag * freqs_cis.real)
    )
    Since cis.real is cos and cis.imag is sin,
    xq_rotated = (
        (xq_.real * cos(radians) - xq_.imag * sin(radians))
        + i(xq_.real * sin(radians) + xq_.imag * cos(radians))
    )
    xq is rotated by radians.
    """

    x_out = torch.view_as_real(x_rotated).flatten(3)  # (b, nh, l, hd // 2, 2) -> (b, nh, l, hd)

    return x_out.type_as(x)


def try_rope():
    dim = 256
    n_heads = 4
    head_dim = dim // n_heads
    seq_len = 16
    end = 1024
    theta = 10000.0

    freqs_cis = precompute_freqs_cis(head_dim, end, theta)

    x = torch.randn(2, n_heads, seq_len, head_dim)

    x_rotated = apply_rotary_emb(x, freqs_cis[:seq_len])

    print(x_rotated.shape)


if __name__ == "__main__":
    try_rope()
