# Adversarial Image Attacks

A lightweight Python package for combining a base image with a guide image to produce a bounded adversarial candidate. The candidate is kept within a configurable L-infinity distance of the base image.

## 中文快速开始：本地网页 UI

项目现在包含一个本地网页界面，可以完成两件事：

1. 上传 base image 和 guide image，生成有界的 adversarial candidate。
2. 使用 OpenAI `gpt-5.6-sol` 分别分析原图和候选图，并比较标签是否变化。

UI 依赖当前版本的 Gradio 和 OpenAI Python SDK，因此请使用 Python 3.10 或更高版本。下面以 Python 3.11 为例：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[ui]'
```

如果要运行 OpenAI 模型对比，请在 [OpenAI API Keys](https://platform.openai.com/api-keys) 创建自己的 Key，然后在当前终端设置环境变量：

```bash
export OPENAI_API_KEY="你的_API_Key"
```

不要把真实 Key 写进代码、README、截图或 Git commit。也可以把 Key 临时填入 UI 的密码输入框，它只用于当前请求，不会由本项目写入磁盘。

启动 UI：

```bash
advmerge-ui
```

浏览器会打开 [http://127.0.0.1:7860](http://127.0.0.1:7860)。如果没有自动打开，请手动访问这个地址。服务只监听本机回环地址，不会自动创建公网链接。

### UI 操作顺序

1. 在 `Base image` 上传希望保持视觉内容的原图。
2. 在 `Guide image` 上传用于引导扰动的第二张图片。
3. 设置 `Strength` 和 `Epsilon`。UI 中 epsilon 使用像素级别，例如 `8` 表示 `8/255`。
4. 点击“生成候选图”。这一步完全在本地运行，不调用 OpenAI。
5. 可选填写目标标签，例如 `dog`。点击“用 GPT-5.6 Sol 独立对比”后，原图和候选图会通过两个独立请求发送给 OpenAI。

模型固定为 OpenAI 的 [`gpt-5.6-sol`](https://developers.openai.com/api/docs/models/gpt-5.6-sol)，图片使用原始细节模式。模型返回主要标签、描述、置信度和目标匹配判断。提供目标标签时，只有“原图不匹配目标，而候选图匹配目标”才会显示定向攻击成功信号。

OpenAI API 会产生费用，图片也会按照你的 OpenAI 项目数据设置进行处理。只有点击模型对比按钮时才会上传图片。仓库本身不包含 API Key，当前自动化测试使用 mock client，不会发送图片或消耗 API 额度。

单次输出发生变化并不能自动证明攻击稳定有效。大语言模型的回答可能有随机性，严谨实验应固定提示词和成功标准，重复多次，并记录成功率。

## Installation

From the repository root, install the package in editable mode:

```bash
python -m pip install -e .
```

This installs both the Python package and the `advmerge` command.

The local browser UI and OpenAI evaluator are optional. Install them with:

```bash
python -m pip install -e '.[ui]'
```

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

The UI evaluates multimodal vision behavior, not a text-only language model. It uses two independent OpenAI Responses API calls so that the candidate is not analyzed alongside the base image. A changed answer is an experimental signal, not proof of a robust attack. For targeted evaluation, define the target label before looking at the candidate result.

Use this package only on models, data, and systems you own or are explicitly authorized to test. Follow applicable laws, policies, licenses, and research ethics requirements.

## Generated experiment

See the reproducible [cat-to-dog bounded merge experiment](examples/generated/cat-to-dog/README.md) for generated source images, four epsilon levels, pixel measurements, and a blind visual-model inspection.
