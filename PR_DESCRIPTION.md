# Add Chunked Softmax Implementation for Large Tensors

## Summary

This PR adds a chunked softmax implementation that processes rows in chunks, relaxing the `n_cols < TILE_SIZE` assumption. Since tiles are stored in registers/shared memory (which is limited), this chunked approach allows handling very large column dimensions.

As someone new to kernel programming, I wanted to experiment with writing a kernel that could handle tensors where `n_cols` exceeds a reasonable tile size. The existing implementations assume the entire row fits in a tile, but with limited register/shared memory, we need to process data in chunks. I'm not sure if there is some limit on what an optimal tile_size is, but I imagine at some point data will leak from registers or shared memory into GMEM. I'm really guessing here and intend on playing around with this more with NSight Compute.

## Implementation

The `softmax_kernel_chunked` uses a 3-pass algorithm:
1. **Pass 1**: Find maximum across all chunks
2. **Pass 2**: Compute denominator (sum of exp values) across all chunks  
3. **Pass 3**: Compute final softmax output

**⚠️ This implementation is NOT numerically stable. I'm requesting a pair of eyes to review the numerical stability of this approach.**

## Performance

Chunked softmax performs best at N=4096 (1.89x faster than PyTorch). At very large sizes (N=32768, 65536), it's the best CuTile option, though still slower than PyTorch.

### Raw Benchmark Numbers (GB/s)

| N     | Baseline CuTile | TMA CuTile | Chunked CuTile | PyTorch | Winner   |
|-------|-----------------|------------|----------------|---------|----------|
| 1024  | 4193.72         | 2359.79    | 3849.69        | 4902.53 | PyTorch  |
| 2048  | 5028.66         | 3236.89    | 4908.20        | 4221.67 | Baseline |
| 4096  | 4305.97         | 3493.09    | **5123.44**    | 2707.91 | **Chunked** |
| 8192  | 5111.81         | 3400.27    | 3078.89        | 2225.20 | Baseline |
| 16384 | 3691.26         | 3570.30    | 2643.45        | 4002.35 | PyTorch  |
| 32768 | 1171.42         | 1062.48    | **2495.97**    | 4396.83 | PyTorch  |
| 65536 | 737.44          | 702.78     | **2393.96**    | 3617.53 | PyTorch  |

### Speedup vs PyTorch

| N     | Baseline | TMA    | Chunked |
|-------|----------|--------|---------|
| 1024  | 0.86x    | 0.48x  | 0.79x   |
| 2048  | 1.19x    | 0.77x  | 1.16x   |
| 4096  | 1.59x    | 1.29x  | **1.89x** |
| 8192  | 2.30x    | 1.53x  | 1.38x   |
| 16384 | 0.92x    | 0.89x  | 0.66x   |
| 32768 | 0.27x    | 0.24x  | **0.57x** |
| 65536 | 0.20x    | 0.19x  | **0.66x** |

## Usage

```python
output = tilegym.ops.softmax(x, use_chunked=True, backend="cutile")
```

## Request for Review

I'm very new to kernel programming and would appreciate any tips or feedback on:
- The chunked processing approach
- Numerical stability considerations
- Performance optimizations
- Best practices for handling large tensors

Any suggestions for improvement would be greatly appreciated!
