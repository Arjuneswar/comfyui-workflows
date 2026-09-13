# ComfyUI Workflows

Video workflows for [ComfyUI](https://github.com/comfyanonymous/ComfyUI), each with notes on
*why* the graph is built the way it is — the settings that look like mistakes but aren't, and
the constraints that shaped the design.

## Workflows

| Workflow | What it does | Base model |
|---|---|---|
| [Character Replacement](workflows/character-replacement) | Replaces the performer in a source video with a different character, keeping the original motion, timing and audio | Wan 2.1 VACE (14B) |

## Layout

```
workflows/<name>/
├── README.md      how it works, what was hard, requirements
├── workflow.json  ComfyUI UI format — drag onto the canvas to import
└── samples/       example output
```

## Using these

Open the workflow folder's README first: it lists the checkpoints, the custom node packs, and
the memory-management settings the graph depends on. Missing custom nodes show up as red
nodes on import; missing checkpoints as empty model dropdowns.

Everything here targets a single consumer GPU, so the graphs lean on block swapping, model
offloading and tiled VAE decode. Those are the first knobs to change when moving to different
hardware.
