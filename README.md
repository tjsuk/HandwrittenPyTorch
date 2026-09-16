# Handwritten Digit Recognition with PyTorch

A beginner-friendly Jupyter notebook that trains a neural network to recognise handwritten
digits (0-9) using the [MNIST dataset](https://systemds.apache.org/datasets/mnist) and
[PyTorch](https://pytorch.org/). It's written as a self-contained **learning exercise**, not
just a script to run, and includes a drawing canvas so you can test (and even correct) the
model on your own handwriting.

This is the PyTorch counterpart to
[HandwrittenTensorflow](https://github.com/tjsuk/HandwrittenTensorflow), a
separate project solving the same problem with TensorFlow/Keras. The two aren't required
reading for each other, but the notebook does call out how PyTorch's approach differs from
Keras' as those differences come up, and finishes with a side-by-side comparison table — so if
you're comparing the two frameworks, they're a good pair to read together.

## What's inside

1. **Import libraries** — torch, torchvision, matplotlib, numpy
2. **Load the MNIST dataset** — via `torchvision.datasets.MNIST`, auto-normalised by `ToTensor()`
3. **View 20 random handwritten digits**
4. **Prepare the data** — batching with `DataLoader` (PyTorch has no automatic batching like Keras' `.fit()`)
5. **Build the model** — a class-based `nn.Module`, with an explanation of every layer and why the output is raw logits rather than softmax probabilities
6. **Define the loss function and optimizer** — PyTorch's equivalent of Keras' `.compile()`, explained in plain English
7. **Train the model** — a training loop written by hand (`zero_grad` / forward / `backward` / `step`), with progress printed per epoch and an on-screen reminder to wait for it to finish
8. **Evaluate the model** — understand the test accuracy score, plus PyTorch-specific pitfalls (forgetting `model.eval()`, forgetting `zero_grad()`, etc.)
9. **Make predictions** — converting logits to probabilities with `softmax`, plus a bar-chart view of the model's confidence across several random examples
10. **Visualise predictions** — compare predicted vs actual digits at a glance
11. **Draw your own digit** — the same interactive canvas as the TensorFlow notebook, with live prediction, correctness feedback, and manual online-learning "teach the model" steps
12. **Save the trained model** — with an explanation of what a `state_dict` actually is, and how that differs fundamentally from a Keras `.keras` file

It finishes with a **Summary** (including a Keras vs PyTorch comparison table), followed by two
bonus sections:

- **Watch backpropagation happen** — Step 7 already calls `loss.backward()` explicitly, so this
  section instead makes the otherwise-invisible *gradient tensors* it produces visible. A minimal
  one-weight toy example computes a gradient with autograd and checks it against a hand-worked
  calculus derivative (they match exactly), then the same technique is applied to the real trained
  model — inspecting actual gradient values on a real batch (`.grad` is `None` before `backward()`,
  populated after), and applying one genuine optimizer step to watch a specific weight change by a
  real, measurable amount (its original weights are restored immediately afterwards, so the demo
  has no side effects on the rest of the notebook).
- **Try a CNN yourself** — exactly which cells you'd change to use a Convolutional Neural Network
  instead, and a runnable CNN cell to compare its accuracy directly against the original model.

## Requirements

- **Python 3.9–3.12**
- pip

## Setup

1. **Create a virtual environment** in the project folder:

   ```bash
   python -m venv .venv
   ```

2. **Activate it.**

   Windows (PowerShell/cmd):
   ```bash
   .venv\Scripts\activate
   ```

   macOS/Linux:
   ```bash
   source .venv/bin/activate
   ```

3. **Install PyTorch (CPU build) and torchvision.** Installing from PyPI directly can pull in a
   very large CUDA-enabled build even on machines without a GPU — the command below uses
   PyTorch's own CPU-only index instead, which is much smaller and all this project needs:

   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

4. **Install the remaining dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

5. **Register the environment as a Jupyter kernel:**

   ```bash
   python -m ipykernel install --user --name=handwritten-digit-pytorch --display-name "Python (handwritten-digit-pytorch)"
   ```

## Running the notebook

Launch Jupyter Lab using this venv's own executable, rather than a plain `jupyter lab` command —
if another Python install is earlier on your `PATH`, or a server from a different environment is
already running, that command can silently connect you to the wrong one, which shows up as
widgets rendering as plain text instead of the interactive canvas (see Troubleshooting below):

```bash
.venv\Scripts\jupyter-lab.exe
```

Open `handwritten_digit_recognition_pytorch.ipynb`, select the **"Python
(handwritten-digit-pytorch)"** kernel, and run the cells from top to bottom.

The first time you run Step 2, torchvision will download MNIST automatically into a local `data/`
folder (a few tens of MB) — this only happens once. Training (Step 7) should take well under a
minute on a typical CPU and reach around 97–98% test accuracy.

## Troubleshooting

**Widgets render as plain text** (e.g. `Canvas(height=200, ...)` instead of an actual canvas, or
`HBox(children=(...))` instead of real buttons):

This means the Jupyter **server** you're connected to isn't the one with `ipywidgets`/`ipycanvas`
installed — the widget-rendering extension is loaded by whichever Python environment started the
server, not by the kernel you select afterwards.

- Make sure you launched Jupyter using this venv's own executable (`.venv\Scripts\jupyter-lab.exe`),
  not a plain `jupyter lab` command that might resolve to a different Python install on your `PATH`.
- If a server was already running (check with `jupyter server list`), re-running `jupyter lab`
  just opens a new browser tab connected to that *existing* server rather than starting a fresh
  one. Fully stop any stale servers (`Ctrl+C` in their terminal) before starting a new one from
  this venv.

**The drawing area needs scrollbars to see fully:**

Jupyter Lab boxes very tall cell outputs into a small scrollable region by default. Right-click
the output and choose **"Disable Scrolling for Outputs"**, or click the thin vertical bar running
down the output's left edge to toggle it between boxed and full-height.

## Notes

- Like the TensorFlow project, this runs on CPU by default. The training loop does check for a
  GPU (`torch.cuda.is_available()`) and will use one automatically if present — PyTorch's GPU
  support on Windows works normally (unlike TensorFlow, which dropped native Windows GPU support
  after 2.10), so if you do have an NVIDIA GPU and CUDA-enabled PyTorch installed, this notebook
  will pick it up without any code changes.
- The saved model file (`handwritten_digit_model_pytorch.pt`) is created when you run Step 12 —
  it isn't included in this repository. Note it only contains the learned weights (a
  `state_dict`), not the architecture — see Step 12 for why, and how to load it back.
- The downloaded `data/` folder is also excluded from the repository via `.gitignore`.

## Ideas to extend

- Train for more epochs, or try a different optimizer (e.g. plain SGD with momentum) or learning
  rate
- Try the Convolutional Neural Network in the Bonus section, or make it the notebook's main model
  instead — only Step 5 needs to change
- Test the model against unusual or messy handwriting using the drawing canvas's "teach the
  model" feature
- Explore the backpropagation bonus section further by inspecting gradients at different pixel
  indices, or across the whole image at once
- Compare this notebook side-by-side with the companion TensorFlow version

Each of these has a detailed, step-by-step walkthrough with full explanations and runnable code
in [`ideas_to_extend/`](ideas_to_extend/README.md).
