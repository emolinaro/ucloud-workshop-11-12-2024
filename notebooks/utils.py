def estimate_gpu_memory(P, Q, L, H, B, N, overhead_factor=1.2):
    """
    Estimate GPU memory required for serving an LLM.

    Parameters:
    - P (int): Number of parameters in the model.
    - Q (int): Model precision in bits (e.g., 16, 8, 4).
    - L (int): Context window length (number of tokens).
    - H (int): Hidden size (dimensionality of hidden layers).
    - B (int): Batch size.
    - N (int): Number of transformer layers.
    - overhead_factor (float): Additional memory overhead factor.

    Returns:
    - float: Estimated GPU memory in Gigabytes.
    """
    # Memory for model (in bytes)
    M_model = P * (Q / 8)  # P parameters * bytes per parameter based on precision
    M_model_GB = M_model / 1e9  # Convert to GB

    # Memory for context (in bytes)
    D = Q / 8  # Bytes per element
    M_context = L * H * D * N  # Tokens * Hidden size * Bytes per element * Layers
    M_context_GB = M_context / 1e9  # Convert to GB

    # Memory for batch (in bytes)
    M_batch = M_context * B  # Batch size
    M_batch_GB = M_batch / 1e9  # Convert to GB

    # Total memory before overhead
    M_total = M_model_GB + M_batch_GB

    # Apply overhead factor once
    M_total *= overhead_factor

    return M_total
