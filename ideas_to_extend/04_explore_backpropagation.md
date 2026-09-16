# Extending the Bonus "Watch Backpropagation Happen" Section

## What you're extending

The **Bonus: Watch backpropagation happen** section inspects
`model.hidden.weight.grad[:5, 400]` — the gradients connecting 5 neurons to a single pixel
(index 400, roughly the centre of the image) — and explains that centre pixels tend to have
large, varied gradients while corner pixels (which MNIST digits never touch) have gradients of
exactly 0. This guide turns that single-pixel spot-check into a full picture across every pixel.

## Why this matters

Reading one pixel at a time makes the point, but it's easy to look at a single number and not
really *feel* the pattern. Visualising the gradient magnitude across the entire 28x28 image at
once turns "corners are 0, the centre is bigger" from an assertion into something you can see
directly — and reveals shape detail (a soft ring where digit strokes typically pass) that
spot-checking a handful of indices would never show.

## How to do it

### 1. Re-run the setup

Run the cells up through the one that computes `first_layer_gradients = model.hidden.weight.grad`
(the real-batch gradient inspection cell), so that tensor exists.

### 2. Average across all 128 neurons, and reshape into an image

`first_layer_gradients` has shape `(128, 784)` — one row per hidden neuron, one column per input
pixel. Averaging the *magnitude* of gradients down each column (across all 128 neurons) gives one
number per pixel, which reshapes neatly back into a 28x28 image:

```python
import matplotlib.pyplot as plt

# Average the absolute gradient across all 128 neurons, for each of the 784 pixels
avg_abs_gradient = first_layer_gradients.abs().mean(dim=0)   # shape: (784,)
gradient_image = avg_abs_gradient.reshape(28, 28).cpu().numpy()

plt.figure(figsize=(5, 5))
plt.imshow(gradient_image, cmap="hot")
plt.colorbar(label="Average |gradient|")
plt.title("Which pixels the first layer's gradient cares about")
plt.show()
```

### 3. Compare it against an actual average digit

To confirm the gradient map lines up with where digits actually have ink, compare it side by side
with the average of several real training images:

```python
sample_images, _ = next(iter(train_loader))
average_digit = sample_images.mean(dim=0).squeeze().cpu().numpy()   # average over the batch

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(average_digit, cmap="gray")
axes[0].set_title("Average pixel brightness\n(across a batch of real digits)")
axes[1].imshow(gradient_image, cmap="hot")
axes[1].set_title("Average |gradient|\n(first layer, same batch)")
plt.show()
```

## What to observe / think about

- **The two images should look similar in outline**: both should be brightest in a rough
  ring/blob covering the centre of the canvas, and close to zero at all four corners — pixels
  that never contain ink can't affect the loss, so they can't have a nonzero gradient either,
  exactly as the notebook's single-pixel example showed for index 400 (centre) vs. corner indices.
- **They won't match exactly.** The gradient depends on the *current* batch's labels and the
  model's current (imperfect) predictions, not just on where ink typically is — so expect a
  noisier, less symmetric pattern than the clean average-brightness image.
- **Try this again after training for more epochs** (see
  [`01_more_epochs_and_optimizers.md`](01_more_epochs_and_optimizers.md)) — as the model gets
  better, gradients for a well-classified image should generally shrink in magnitude, since
  there's less room left to improve on an already-confident, correct prediction.
- **Try it on `model.output.weight.grad` instead** (the second, 10x128 layer). It reshapes
  differently (rows are the 10 output digits, not pixels), but is worth comparing against the
  first layer's gradients — they answer different questions ("which pixels matter" vs. "which
  hidden-layer features matter for each digit").

## Related ideas worth knowing about

- **Saliency maps**: instead of the *weight* gradients used here, computing the gradient of the
  loss with respect to the *input image itself* (`images.requires_grad_(True)`) produces a map of
  which pixels of one specific drawing most influenced its prediction — a common real-world
  explainability technique, and a natural next step after this exercise.
- **Grad-CAM**: a more sophisticated version of the same idea, designed for CNNs (see
  [`02_try_the_cnn.md`](02_try_the_cnn.md)), that highlights which *regions* of an image most
  drove a particular prediction.
