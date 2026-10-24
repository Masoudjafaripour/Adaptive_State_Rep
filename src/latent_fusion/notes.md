Hand-written log; embedded into README.md on every run.

- **2026-10-23, smoke test:** all 5 variants run end to end. Without self-prediction, the latent drifted and the innovation NLL diverged (loss about 1e6 at iteration 2). Fixed by soft-bounding the latents to 3·tanh(·/3) and raising the minimum log-noise from −8 to −5.
- **Not yet trained at full length:** waiting for the fusion_e2e 3-seed runs to free the CPU.
