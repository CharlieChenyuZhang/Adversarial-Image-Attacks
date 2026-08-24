# Adversarial Image Attacks

A lightweight Python package for combining a base image with a guide image to produce a bounded adversarial candidate. The candidate is kept within a configurable L-infinity distance of the base image.

## Installation

From the repository root, install the package in editable mode:

```bash
python -m pip install -e .
```

This installs both the Python package and the `advmerge` command.

## Command line usage

```bash
advmerge BASE GUIDE OUTPUT --strength 1 --epsilon 8/255
```

For example:

```bash
advmerge cat.jpg texture.png cat-adversarial.png \
  --strength 1 \
  --epsilon 8/255
```

`OUTPUT` must have a `.png` extension. Existing files are not replaced by default. Pass `--force` when replacement is intentional.

The main options are:

| Option | Default | Meaning |
| --- | --- | --- |
| `--strength` | `1` | Scales the direction from the base image toward the guide image. |
| `--epsilon` | `8/255` | Maximum per-channel absolute change from the base image. A fraction such as `8/255` or a decimal value may be used. |
| `--resize-mode` | `cover` | Resizes the guide with either `cover` or `stretch`. |
| `--background` | `255,255,255` | RGB background used when an input contains transparency. |
| `--force` | off | Allows an existing output file to be replaced. |

With `cover`, the guide keeps its aspect ratio, is scaled to fill the base image dimensions, and is center-cropped. With `stretch`, the guide is resized directly to the base image width and height. The base image always determines the output dimensions.

## Python API

```python
from PIL import Image

from adversarial_image_attacks import merge_images

with Image.open("cat.jpg") as base, Image.open("texture.png") as guide:
    candidate = merge_images(
        base,
        guide,
        strength=1.0,
        epsilon=8 / 255,
        resize_mode="cover",
        background=(255, 255, 255),
    )

candidate.save("cat-adversarial.png", format="PNG")
```

`merge_images` returns an RGB `PIL.Image.Image` with the same dimensions as the base. The optional arguments use the same defaults as the command line interface. The path-based `merge_files` helper used by the CLI writes a PNG and returns a `MergeResult`. Its perturbation statistics are normalized to `[0, 1]`, not expressed as integer values from 0 to 255.

## How the merge works

Images are converted to eight-bit RGB and the merge is calculated with floating-point pixel levels in the range `[0, 255]`. The normalized epsilon value is multiplied by 255 during this calculation. After the guide is resized to match the base, the equivalent normalized formula is:

```text
candidate = base + strength * (guide - base)
```

The change from the base is then projected into the L-infinity epsilon ball and the result is clipped to the valid image range:

```text
delta  = clip(candidate - base, -epsilon, epsilon)
output = clip(base + delta, 0, 1)
```

The final integer quantization is clamped to the same epsilon interval. This guarantees that no RGB channel in the saved PNG changes by more than `epsilon` in normalized pixel units. When epsilon is not an exact multiple of `1/255`, the maximum representable change is rounded down to a whole pixel level. Budgets smaller than `1/255` therefore leave the base unchanged.

RGB input is used directly. RGBA and other images with transparency are alpha-composited onto the configured background before the merge. The background defaults to white. The saved result is an RGB PNG so that perturbation values are not changed by lossy encoding.

## What this does and does not guarantee

Merging two images this way produces a bounded adversarial candidate. It does not guarantee that a classifier will make an error. Attack success depends on the target model, preprocessing, label, objective, epsilon budget, and evaluation procedure. Verify the output against the specific model and threat model relevant to your work.

Use this package only on models, data, and systems you own or are explicitly authorized to test. Follow applicable laws, policies, licenses, and research ethics requirements.
