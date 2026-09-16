# Extending the Bonus "Try a CNN Yourself" Section

## What you're extending

The **Bonus: Try a CNN yourself** section at the end of
[the notebook](../handwritten_digit_recognition_pytorch.ipynb) already builds, trains, and
evaluates a `CNNClassifier` independently of the main `model`, printing its test accuracy next
to Step 8's for comparison. This guide goes further: tweaking the CNN's architecture, and making
it the notebook's *main* model instead of a side comparison.

## Why this matters

The original model (Step 5) flattens the image to 784 numbers immediately, throwing away all
information about which pixels are *near* each other. A Convolutional Neural Network (CNN) keeps
the image in its 2D grid shape for longer, using small filters that slide across it looking for
local patterns (edges, curves, loops) regardless of where in the image they appear. This is
usually a better match for image data, and typically reaches higher accuracy on MNIST than a
plain fully-connected network of a similar size.

## How to do it

### 1. Run the bonus cell as-is first

No setup needed beyond having already run Steps 1-4 (so `train_loader`/`test_loader`/`device`
exist) and Step 8 (so `test_accuracy` exists for the printed comparison). Just scroll down and
run the CNN bonus cell directly.

### 2. Tweak the architecture

The bonus `CNNClassifier` uses 32 and 64 filters in its two convolutional layers. Try widening it:

```python
class WiderCNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 5 * 5, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

Only `fc1`'s output channel count needs recalculating if you change `conv2`'s `out_channels` —
the spatial size (5x5) stays the same regardless of channel counts, since that depends only on
the image size and kernel/pooling sizes, not how many filters you use.

### 3. Make the CNN the notebook's main model

Only **Step 5** needs to change. Replace its `HandwrittenDigitClassifier` class definition with
the bonus section's `CNNClassifier` class, and re-run Step 5. Steps 2 and 6 through 12 all work
completely unchanged afterwards, because they only ever call `model(images)` — none of them care
what's inside the model, as long as it accepts a batch of images and returns 10 logits per image.

This is a nice contrast with the companion TensorFlow notebook, where switching to a CNN also
requires changing the data-loading step. Here, `torchvision.datasets.MNIST` (Step 2) already
hands back images shaped `(1, 28, 28)` — one channel, 28 pixels by 28 pixels — which is exactly
the shape `Conv2d` layers expect. The original Linear-based model only worked with this same data
because its own `nn.Flatten()` collapsed it down to a flat list first.

## What to observe / think about

- **Parameter count**: print `sum(p.numel() for p in cnn_model.parameters())` and compare it to
  the original model's 101,770. A well-designed CNN is often *more* parameter-efficient per unit
  of accuracy than a fully-connected network, despite doing more computation.
- **Accuracy**: the bonus section already prints both models' test accuracy side by side — the
  CNN should come out at least a little ahead, and often by a clearer margin once you widen it
  as in step 2 above.
- **Training time**: CNNs do more floating-point work per image than a Linear-based model of a
  similar parameter count. Time both training loops (e.g. with Python's `time.time()` around the
  epoch loop) and compare accuracy-per-second, not just final accuracy.
- **Where accuracy plateaus**: MNIST is a fairly easy dataset — both the original model and a
  small CNN comfortably exceed 97%. The gap between them tends to widen much more on harder image
  datasets (e.g. `torchvision.datasets.FashionMNIST` or `CIFAR10`), which is worth trying if you
  want to see a CNN's advantage more dramatically.

## Related ideas worth knowing about

- **Batch normalization** (`nn.BatchNorm2d`, inserted after each convolution): often speeds up
  and stabilises CNN training, and is standard practice in almost every modern CNN architecture.
- **Data augmentation**: randomly rotating, shifting, or zooming training images slightly (via
  `torchvision.transforms`) exposes the model to more variation than the raw dataset provides,
  which tends to help most on harder datasets than MNIST.
- **A third convolutional block**: adding a `conv3`/`pool` pair before flattening lets the network
  build up even more abstract features, though on an image as small as 28x28 you'll run out of
  spatial room to pool further fairly quickly.
