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

### Train for more epochs, or try a different optimizer or learning rate

In the Step 7 code cell, change `EPOCHS = 5` to a higher number, e.g. `EPOCHS = 15`. Since Step 5
already built `model` and Step 6 already created `optimizer`, just re-run the Step 7 cell directly
— it'll keep training the *same* model for the extra epochs. If you'd rather train a fresh model
from scratch for a fair comparison, re-run Step 5 (rebuilds `model` with new random weights) and
Step 6 (recreates `optimizer` to match) first.

To try a different optimizer, edit the Step 6 cell:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

Plain SGD generally needs a higher learning rate and some `momentum` to train at a comparable
speed to Adam — the values above are a reasonable starting point. Since this replaces `optimizer`
entirely, re-run Step 5 first to reset `model` to its untrained state, then Step 6, then Step 7,
so you're comparing a fair, freshly-trained run.

### Try the Convolutional Neural Network in the Bonus section

No setup needed — scroll to the **Bonus: Try a CNN yourself** section at the end of the notebook
and run its cell directly. It builds, trains, and evaluates a small CNN independently (using
`cnn_model` rather than `model`), and prints its test accuracy directly next to the original
model's from Step 8, so you can compare them immediately.

If you'd rather make the CNN the notebook's *main* model instead of a separate comparison, only
**Step 5** actually needs to change — replace its `HandwrittenDigitClassifier` class with the
`CNNClassifier` class shown in the Bonus section. Every other step (2 and 6 through 12) works
unchanged, since they only ever call `model(images)` without caring what's inside it — a nice
contrast with the TensorFlow notebook, which also needs its data-loading step changed for its CNN
bonus. The Bonus section's "Where in the code above you'd need to change things" part explains why.

### Test the model against unusual or messy handwriting

Run the Step 11a and 11b cells, then in Step 11's canvas:
1. Draw a digit in an unusual style — very thin, off-centre, rotated, or an unconventional way of
   forming a digit (e.g. a 7 with a crossbar, a closed-top 4)
2. Click **Predict** and see what the model guesses, and how confident it is
3. Click **No, wrong** if it got it wrong, pick the actual digit from the dropdown, and click
   **Teach the model**
4. Draw the same digit again and click **Predict** — it should now be more likely to get it right

If you correct the model on many examples and want to reset it back to its originally-trained
state, re-run Step 5 (rebuilds `model` from scratch), Step 6 (recreates `optimizer` to match),
and Step 7 (retrains on the full MNIST training set) — this discards any canvas-based corrections.

### Explore the backpropagation bonus section further

In the **Bonus: Watch backpropagation happen** section, find the cell containing
`first_layer_gradients[:5, 400]` and change `400` to a different pixel index between 0 and 783
(remember pixels are the *second* index of this weight tensor — see the note in that cell). Re-run
that cell (and the following one, which also references `400` when picking which weight to
update) and see how the gradient values and the resulting weight change differ:
- Indices near the image's edges/corners (e.g. `0`, `27`, `755`) tend to give gradients of exactly
  `0`, since MNIST digits rarely or never touch those pixels
- Indices nearer the centre (e.g. `350`-`450`) tend to give the largest, most varied gradients,
  since that's where digit strokes usually pass through

### Compare this notebook side-by-side with the TensorFlow version

Open both notebooks in separate tabs and step through them in parallel, section by section. A few
concrete things worth comparing directly:
- Run `model.summary()` (TensorFlow, Step 5) next to `print(model)` plus the parameter count
  (PyTorch, Step 5) — both report 101,770 parameters for the identical architecture, just formatted
  very differently
- Compare Step 6 in each: one `model.compile(...)` call vs. two separate `criterion`/`optimizer`
  objects, configuring the same underlying ideas
- Draw the *same* digit on both notebooks' canvases (Step 11 in each) and compare the predicted
  digit and confidence percentage each model gives it
- Read the "How this compares to Keras, at a glance" table in this notebook's Summary section for
  a full concept-by-concept mapping between the two
