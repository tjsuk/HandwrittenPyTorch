# Extending the Summary: Comparing Side-by-Side with the TensorFlow Notebook

## What you're extending

This notebook's Summary ends with a "How this compares to Keras, at a glance" table, mapping
each PyTorch concept used here onto its TensorFlow/Keras equivalent from the companion
[HandwrittenTensorflow](https://github.com/tjsuk/HandwrittenTensorflow) project. This guide turns
that table from something you read into something you actually run and check, section by section,
against the real TensorFlow notebook.

## Why this matters

Reading a comparison table tells you the two frameworks solve the same problems differently;
running both notebooks side by side lets you *see* it — the same 101,770-parameter architecture
described in two very different styles, the same digit predicted (hopefully) the same way by two
independently-trained models, and the real difference in code volume between `model.fit(...)` and
a hand-written training loop.

## How to do it

### 1. Get both notebooks side by side

Clone the companion repository alongside this one (adjust the path as needed):

```bash
git clone https://github.com/tjsuk/HandwrittenTensorflow.git
```

Follow its own README to set up its virtual environment — it's a separate project with its own
`requirements.txt`, independent of this one. Open both notebooks in separate tabs or windows.

### 2. Step through both in parallel, section by section

A few concrete comparisons worth making directly, rather than just reading about:

- **Model definition and size**: run `model.summary()` in the TensorFlow notebook's Step 5, and
  `print(model)` plus the parameter-counting code in this notebook's Step 5. Both should report
  **101,770 total parameters** for the identical architecture — confirm the two layer-by-layer
  breakdowns actually agree (784×128 + 128 biases in the hidden layer, 128×10 + 10 biases in the
  output layer), just formatted completely differently.
- **Configuring training**: compare the TensorFlow notebook's single `model.compile(...)` call
  (Step 6) against this notebook's two separate `criterion`/`optimizer` objects (also Step 6) —
  note they configure the exact same two decisions (how to measure error, how to update weights).
- **Training code volume**: count the lines involved in actually training in each notebook — one
  `model.fit(x_train, y_train, epochs=5)` call vs. this notebook's explicit `for` loop over
  `train_loader` with manual `zero_grad()`/`backward()`/`step()` calls. Same result, very
  different amount of code to get there.
- **Drawing the same digit on both canvases**: Step 11 in each notebook has its own independent
  drawing canvas. Draw the *same* digit, in the *same* style, on both, and compare the predicted
  digit and confidence percentage each model reports. They were trained independently (different
  random initial weights), so don't expect identical confidence numbers — but they should usually
  agree on the predicted digit itself for a normal, clearly-drawn one.
- **Saving format**: compare the TensorFlow notebook's single self-describing `.keras` file
  (Step 12) against this notebook's `state_dict` (also Step 12, which only stores weights and
  needs the `HandwrittenDigitClassifier` class definition kept separately to reload).

### 3. Try the CNN bonus in both

Both notebooks include a CNN bonus section. Compare test accuracy from each — but note the
TensorFlow notebook's CNN bonus needs its data-loading step changed too, while this notebook's
CNN swap only touches Step 5 (see [`02_try_the_cnn.md`](02_try_the_cnn.md) for why). Reading both
bonus sections' explanations of *why* that difference exists is a good test of whether the
underlying architectural reasoning has actually sunk in, not just the code.

## What to observe / think about

- Both models should land in a similar test-accuracy ballpark (roughly 97-98%) despite being
  trained completely independently in two different frameworks — a good sanity check that neither
  notebook has a subtle bug making its numbers implausible.
- Notice *where* each framework hides complexity from you and where it doesn't. Keras hides
  batching and the training loop entirely; PyTorch makes you write both out, but in exchange never
  hides what `.backward()` and `.step()` are actually doing (see
  [`04_explore_backpropagation.md`](04_explore_backpropagation.md), which has no real equivalent
  you could write as easily against a `model.fit()` call).
- Neither framework is "better" in the abstract — form your own opinion on which style of
  visibility-vs-convenience trade-off you'd personally reach for on a real project, and *why*,
  rather than treating this as a boxes-ticked exercise.

## Related ideas worth knowing about

- **PyTorch Lightning**: a popular library built on top of PyTorch that reintroduces some of
  Keras' convenience (a `Trainer.fit()`-style call) while keeping PyTorch's model-definition
  style — worth a look once you're comfortable with the raw training loop this notebook writes
  by hand.
- **JAX/Flax**: a third major deep learning framework with yet another design philosophy
  (functional, explicit about randomness and state) — implementing this same MNIST classifier
  there is a good way to see a third point on the visibility-vs-convenience spectrum.
