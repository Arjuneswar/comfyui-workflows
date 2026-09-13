# Character Replacement (Wan 2.1 VACE)

**One line:** Replaces the performer in a source video with a different character, keeping the
original motion, timing and audio intact.

## The problem

Reshooting a performance with a different character means another shoot. The alternative —
frame-by-frame roto and comp — costs days per shot and still breaks on fast motion.

The naive generative approach fails in a specific way: text-to-video gives you a new character
but invents its own motion, so it no longer matches the plate, the cut or the audio. What's
needed is motion locked to the source and identity taken from a reference.

This workflow separates those two signals. Pose comes from the source video; identity comes
from a single reference image. Neither is left to the model to guess.

## How it works

**Models**

| Role | Checkpoint |
|---|---|
| Base video model | `Wan2.1/Wan14BT2VFusioniX_fp16_.safetensors` (14B) |
| Control module | `Wan2.1/Wan2_1-VACE_module_14B_bf16.safetensors` |
| VAE | `wan_2.1_vae.safetensors` (bf16) |
| Text encoder | `t5/umt5_xxl_fp16.safetensors` (bf16, offloaded) |

**Graph**

1. **Source video** — `VHS_LoadVideo` at a forced 16 fps, so frame timing is deterministic
   regardless of the input's native rate.
2. **Resize** — `ImageResizeKJv2` to 576×1024 portrait, centre crop.
3. **Pose extraction** — `DWPreprocessor` (`yolox_l` detector + `dw-ll_ucoco_384` pose model,
   1024px) with body, hand and face all enabled. This is the motion signal.
4. **Reference character** — `LoadImage` → `ImageResizeKJv2`. The reference resize takes its
   width and height as *inputs* from the video resize node rather than from its own widgets,
   so the reference always matches the plate dimensions and never needs manual adjustment
   when the source aspect changes.
5. **VACE encode** — pose sequence as control, reference image as identity, strength 1.0.
   Frame count comes from the loader, so the workflow adapts to clip length automatically.
6. **Sample** — `WanVideoSampler`, 9 steps, CFG 1.0, `flowmatch_causvid` scheduler.
7. **Decode** — tiled decode (272/272 tiles, 144/128 overlap) to keep VRAM bounded.
8. **Output** — `VHS_VideoCombine`, h264, 16 fps, CRF 19, with the **original audio passed
   through** from the loader so the result stays in sync with the source.

A second branch composites reference image, control pose and result side by side into a
labelled comparison strip — a review artifact for the client, not part of the generation.

**Memory management** (this is what makes a 14B model run at all on one GPU)

- `WanVideoBlockSwap` — 5 blocks swapped to system RAM
- SageAttention
- `torch.compile` via inductor, dynamic shapes off
- T5 text encoder offloaded after encoding
- Tiled VAE decode

## What was hard

**Nine steps at CFG 1.0 is not a typo.** FusioniX is a distilled/CausVid-style variant, so it
denoises in single-digit steps with guidance effectively off. Coming from standard diffusion
settings this reads like a broken config — the instinct is to raise steps and CFG, which
makes output *worse* and roughly 5× slower. Matching the scheduler (`flowmatch_causvid`) to
the checkpoint is what makes it work.

**Fitting 14B + VACE + T5 on one GPU.** Block swapping, offloading and tiled decode each cost
speed; the combination is the difference between running and OOM. Block-swap count is the
knob to tune first when moving to a different card.

**Clip length is the real constraint.** VACE encodes the whole sequence at once, so VRAM
scales with frame count. At 16 fps this is a short-clip tool — longer shots need to be split
and stitched, and pose continuity across the seam is the open problem.

**Identity vs. motion tug-of-war.** At high VACE strength the reference dominates and motion
softens; too low and the original performer bleeds through. Strength 1.0 with a clean,
well-lit, full-body reference was more reliable than tuning strength against a weak reference.

<!-- TODO: fill these in and uncomment. Hidden for now so the rendered page
     shows nothing unfinished.

## Where it's used / results

Production or experimental? How many shots? Who ran it — and did anyone other
than you run it unattended? That last one is the reliability signal.

## My role

Built the graph from scratch, or adapted an existing VACE template? Which parts
are yours? If the published result sits on a collaborator's profile, say so here
and credit them.

-->

## Files

- `workflow.json` — ComfyUI UI format. Drag onto the canvas, or **Workflow → Open**.
- `samples/output-with-audio.mp4` — example output, with the source audio passed through.

## Requirements

- ComfyUI-WanVideoWrapper
- ComfyUI-VideoHelperSuite
- ComfyUI-KJNodes
- comfyui_controlnet_aux (DWPose)
- SageAttention
