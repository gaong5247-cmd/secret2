# Research & TPS verification (2026-11-04)

| Research | Practical idea reused | What is *not* implemented |
|---|---|---|
| [Open Pixel2Play / Scaling Behavior Cloning Improves Causal Reasoning (2026)](https://elefant-ai.github.io/open-p2p/) | Observe -> action label, behavioral cloning, lightweight action decoding. P2P's smallest public checkpoint is **150M parameters**, uses 192×192 frames, a transformer, and annotation data from 8,300+ hours of gameplay. Original repo code and model are MIT licensed. | No P2P pretrained weights, no P2P inference backend; this repository trains a separate tiny MLP from labels. Training a smaller student demands new paired samples/distillation. |
| [Lumine (2025)](https://arxiv.org/abs/2511.08892) | Fast/slow control, run basic actions frequently and reasoning less frequently; Lumine uses 5 Hz image perception and 30 Hz action generation. | No Qwen2-VL-7B reasoning model. Here a CPU tactical rule module substitutes. |
| [TD-MPC2 (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/hash/cf73d57b6dcda32b293df7c2d5341f49-Abstract-Conference.html) | Use short-horizon model-based trajectory predictions for interception. | No learned latent world model or MPC optimizer. |
| [DreamerV3 (Nature 2025)](https://www.nature.com/articles/s41586-025-08744-2) | Learn how actions alter future observations, eventually use simulation and rewards. | No self-play reinforcement learning engine or guaranteed rewards. |
| [Diffusion Policy (2024/2025)](https://diffusion-policy.cs.columbia.edu/) | Action chunking and receding-horizon ideas; control-delay sensitivity. | No diffusion model (too heavy for the initial low-latency target). |
| [PyTorch Intel XPU](https://docs.pytorch.org/docs/main/notes/get_start_xpu.html) | `torch.xpu.is_available()` and `model.to('xpu')` with CPU fallback. | No benchmark on the user's exact Arc GPU. |

## Identity and controls

- **Correct target game:** [TPS: Ultimate Soccer](https://www.roblox.com/games/5783581/TPS-Ultimate-Soccer), developer TAYFUN7, Roblox place ID **5783581**.
- [Creator's TPS Ultimate Soccer update log](https://devforum.roblox.com/t/tps-ultimate-soccer-version-log/2094793): tool behavior and Back Pass were modified; the update points players to the **in-game Help menu**. Camera-angle kick and ball collision options have also been added.
- **Different game:** [TPS: Street Soccer](https://www.roblox.com/games/335760407/TPS-Street-Soccer), with a [separate community controls guide](https://tps-street-soccer.fandom.com/wiki/Controls). Its control mappings **must not** be assumed to apply to TPS Ultimate.
- **Another different game:** [Ultimate Soccer](https://www.roblox.com/games/15131312477/Ultimate-Soccer) from Untitled Soccer Thing, which explicitly documents LMB/RMB/Q/E controls. **Not our target.**
- Published TPS Ultimate pages do **not** provide a fully verified current PC keymap; read the in-game Help menu before enabling input.

## Roadmap (unimplemented)

1. Record paired frames + key/mouse actions in a permitted practice setting; build labeled TPS data with player teams, own avatar, ball, feet and goal locations.
2. Train YOLO-nano/ONNX on TPS footage and a multi-object tracker; estimate camera motion by background optical flow/field lines and approximate ground-plane homography.
3. Capture expert skill demonstrations; distill P2P teacher decisions where supported or train on directly recorded humans with class-balanced loss and a time-windowed state.
4. Identify ball flight friction, launch power, curvature and bounce using controlled trials with fixed camera; calibrate explicit 3D/field geometry where possible.
5. Add hierarchical plays: interception, lane denial, possession, ball shielding, pass options, shot selection. Use short-horizon actions with safety gating.
6. Evaluate latency on Intel Arc XPU and CPU separately and log ball IoU/recall, player recall, decision latency, steal rate, pass success, and shot accuracy. **No benchmark results currently claimed.**
