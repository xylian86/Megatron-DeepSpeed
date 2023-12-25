# Run with torch.compile (Experimental)

This folder contains an example that enables `torch.compile`.

## Enable `torch.compile`

To enable `torch.compile`, set the `--compile` flag.
`run.sh` in this folder shows an example.

## Verification

The following chart shows TFLOPS and loss curves resulting from No ZeRO and ZeRO 1/2/3.

- Sequence length: 2048
- Global batch size: 4
- Model: GPT-1.3B
- GPUS: 4x A100 (80GB)

![verification](loss_verification.png)
