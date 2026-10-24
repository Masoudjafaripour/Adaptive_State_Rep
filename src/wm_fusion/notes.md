Hand-written log; embedded into README.md on every run.

- **2026-10-23, smoke test:** every variant runs end to end on `cartpole_swingup` (dm_control backend), including GIFs, records and README.
- **2026-10-23, compute:** on the M5 CPU, one world-model update (batch 16 × 50, 64×64 images) takes about 7 s while other jobs share the CPU, and collecting an episode takes about 3 s. The default budget of 100 episodes × 100 updates is about 20 h per method on this Mac, so full benchmark runs need a CUDA GPU (about 50 ms per update). The Mac is for debugging and short pilots (e.g. `--updates-per-episode 25`).
