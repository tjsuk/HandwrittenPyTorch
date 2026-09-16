# Extending Steps 6-7: Training for Longer, or Trying a Different Optimizer

## What you're extending

[Step 7](../handwritten_digit_recognition_pytorch.ipynb) trains for a fixed `EPOCHS = 5` using
the `optimizer` built in Step 6 (`torch.optim.Adam(model.parameters(), lr=0.001)`), and never
questions either choice. This guide reruns training with both changed, so you can see the effect
directly instead of taking the defaults on faith.

## Why this matters

Two of the most impactful choices in any training setup are "for how long" and "with what
update rule" — and both are cheap to experiment with here, since the whole notebook trains in
well under a minute:

- **More epochs** generally keeps reducing the training loss, but test accuracy (Step 8)
  eventually plateaus or even gets slightly worse once the model starts fitting quirks of the
  training set rather than genuinely useful patterns.
- **Adam vs. plain SGD** is one of the most common practical choices in deep learning. Adam
  adapts its effective step size per-weight automatically, which is why it works well "out of the
  box" at `lr=0.001`; plain SGD has no such adaptation, so it typically needs a higher learning
  rate and some `momentum` to train at a comparable speed.

## How to do it

### 1. Train for more epochs

In the Step 7 code cell, change:

```python
EPOCHS = 5
```

to:

```python
EPOCHS = 15
```

If you just re-run the Step 7 cell as-is, it *continues* training the same `model` for 15 more
epochs on top of the 5 it already did. For a fair, from-scratch comparison instead, re-run Step 5
first (rebuilds `model` with fresh random weights), then Step 6 (recreates `optimizer` to match),
then the edited Step 7.

### 2. Record the loss and accuracy history so you can compare runs

Wrap the training loop in a small function that returns its history, instead of only printing it:

```python
def train_model(model, optimizer, criterion, epochs):
    history = {"loss": [], "accuracy": []}
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        history["loss"].append(running_loss / total)
        history["accuracy"].append(correct / total)
        print(f"Epoch {epoch + 1}/{epochs} - loss: {history['loss'][-1]:.4f} - accuracy: {history['accuracy'][-1]:.4f}")
    return history
```

### 3. Try plain SGD instead of Adam

Edit the Step 6 cell:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

Re-run Step 5 (fresh `model`), then this edited Step 6, then Step 7 (or `train_model` above) to
get a fair, freshly-trained comparison against the original Adam run.

### 4. Plot both runs together

```python
import matplotlib.pyplot as plt

model_adam = HandwrittenDigitClassifier().to(device)
history_adam = train_model(model_adam, torch.optim.Adam(model_adam.parameters(), lr=0.001), nn.CrossEntropyLoss(), epochs=15)

model_sgd = HandwrittenDigitClassifier().to(device)
history_sgd = train_model(model_sgd, torch.optim.SGD(model_sgd.parameters(), lr=0.01, momentum=0.9), nn.CrossEntropyLoss(), epochs=15)

plt.plot(history_adam["loss"], label="Adam")
plt.plot(history_sgd["loss"], label="SGD (momentum=0.9)")
plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.legend()
plt.title("Adam vs. SGD over 15 epochs")
plt.show()
```

## What to observe / think about

- **Diminishing returns**: the loss curve should fall quickly in the first few epochs, then
  flatten out. Compare Step 8's test accuracy after 5 epochs vs. after 15 — the gain is usually
  much smaller than the jump from 1 epoch to 5.
- **Adam should converge faster** (fewer epochs to reach a given loss) than the SGD settings
  above, at least initially — that's *why* Adam has become such a common default. Try lowering
  SGD's learning rate to `0.001` (matching Adam's) and watch it barely train at all, to see why
  SGD needed the higher rate and momentum in the first place.
- This notebook doesn't hold out a separate validation set — Step 8's test set is only ever
  checked once, after training is finished. If you plan to compare many configurations like this,
  consider splitting off a validation subset instead of repeatedly checking the real test set,
  so your final test accuracy stays an honest, unseen-data measurement.

## Related ideas worth knowing about

- **Learning rate schedules**: gradually reducing the learning rate as training progresses
  (`torch.optim.lr_scheduler`) often reaches a better final result than a single fixed value for
  the whole run.
- **Other adaptive optimizers**: `torch.optim.RMSprop` and `torch.optim.AdamW` are close relatives
  of Adam worth comparing against — `AdamW` in particular is now the more common default in a lot
  of modern training code.
