# Extending Step 11: Testing the Model Against Unusual or Messy Handwriting

## What you're extending

[Step 11](../handwritten_digit_recognition_pytorch.ipynb)'s drawing canvas lets you draw a digit,
get a prediction, and — via Step 11b's **Teach the model** button — correct the model when it's
wrong. The notebook demonstrates this once with a normal-looking digit; this guide pushes it
deliberately with handwriting styles the model has never seen anything like during training.

## Why this matters

The model was trained exclusively on MNIST, a dataset of digits that are all centred, similarly
sized, and written in fairly conventional ways. Real handwriting is messier than that. Seeing
*where exactly* the model breaks — and then using the notebook's own online-learning feature to
fix it live — makes the gap between "trained on a benchmark" and "works in the real world"
concrete, rather than theoretical.

## How to do it

### 1. Run the setup cells

Run Step 11a (sets up the canvas) and Step 11b (defines the prediction/correction logic) if you
haven't already, then draw in the canvas Step 11 displays.

### 2. Try progressively unusual styles

Work through these, clicking **Predict** after each and noting the predicted digit and its
confidence (the bar chart from Step 9 shows this same kind of confidence breakdown):

- A digit drawn very small in one corner of the canvas (preprocessing crops and recentres it, but
  extreme cases can still confuse the model)
- A very thin, faint stroke vs. an extremely thick one
- A digit rotated 30-45 degrees
- Unconventional ways of forming a digit — a `7` with a crossbar through the middle, a `4` with a
  closed top, a `1` with a serif base

### 3. Correct the ones it gets wrong

For each mistake:
1. Click **No, wrong**
2. Pick the actual digit from the dropdown
3. Click **Teach the model**
4. Redraw the *same* digit in the *same* style and click **Predict** again

### 4. Check whether the correction generalises

Draw a *different* digit in the same unusual style you just corrected (e.g. if you taught it one
rotated `7`, draw a differently-rotated `7`). A single correction often only nudges the model
toward that one exact drawing, not the general pattern behind it.

## What to observe / think about

- **Confidence vs. correctness**: a wrong prediction with very high confidence (e.g. 95%+) tells
  you something different from a wrong prediction the model was already unsure about (e.g. 40%
  for the top guess) — the first is a genuine blind spot, the second is closer to a coin-flip the
  model got unlucky on.
- **Catastrophic forgetting**: Step 11b's "teach the model" runs a few real training steps on
  *just* your one drawing. If you correct the model on many very different examples in a row,
  watch for it becoming *worse* on ordinary digits it previously got right — fine-tuning on a
  handful of examples can overwrite more general patterns the original MNIST training established.
  If this happens, it's expected, not a bug.
- **Resetting**: if you want to undo accumulated corrections and get back to the originally
  trained model, re-run Step 5 (fresh `model`), Step 6 (fresh `optimizer`), and Step 7 (retrains
  on the full MNIST training set) — this discards every canvas-based correction.

## Related ideas worth knowing about

- **Data augmentation**: instead of correcting individual mistakes after the fact, you can make
  the *original* training more robust by randomly rotating, shifting, or thickening MNIST's
  training images before training even starts (via `torchvision.transforms`) — exposing the model
  to this kind of variation up front, rather than patching blind spots one drawing at a time.
- **Collecting a small custom dataset**: save several of your own drawings (via
  `canvas.get_image_data()`, already used in Step 11b's preprocessing) with their correct labels,
  and fine-tune on that whole batch at once rather than one correction at a time — a batch of
  corrections tends to generalise better than single-example updates.
