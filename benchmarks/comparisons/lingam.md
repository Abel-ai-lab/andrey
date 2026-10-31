# DirectLiNGAM and ICA-LiNGAM

`lingam_cpu.py` compares DirectLiNGAM and ICA-LiNGAM with causal-learn on CPU, and
`lingam_gpu.py` compares DirectLiNGAM on CPU and GPU, on non-Gaussian data.

DirectLiNGAM and ICA-LiNGAM need the `lingam_sf` values in `SLICE` because Gaussian data cannot
identify the model. Use `andrey.direct_lingam.torch-cuda` for GPU runs. It selects CUDA inside
`fit`, so `ANDREY_DEVICE` has no effect. Both ICA-LiNGAM solutions pin FastICA's seed to 0.
