from mnist import MNIST

import minitorch

mndata = MNIST("project/data/")
images, labels = mndata.load_training()

BACKEND = minitorch.TensorBackend(minitorch.FastOps)
BATCH = 16

# Number of classes (10 digits)
C = 10

# Size of images (height and width)
H, W = 28, 28


def RParam(*shape):
    r = 0.1 * (minitorch.rand(shape, backend=BACKEND) - 0.5)
    return minitorch.Parameter(r)


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size

    def forward(self, x):
        batch, in_size = x.shape
        return (
            x.view(batch, in_size) @ self.weights.value.view(in_size, self.out_size)
        ).view(batch, self.out_size) + self.bias.value


class Conv2d(minitorch.Module):
    def __init__(self, in_channels, out_channels, kh, kw):
        super().__init__()
        self.weights = RParam(out_channels, in_channels, kh, kw)
        self.bias = RParam(out_channels)

    def forward(self, input):
        """
        Apply 2D convolution operation

        Args:
            input: Tensor of shape (batch, in_channels, height, width)

        Returns:
            Output tensor of shape (batch, out_channels, out_height, out_width)
        """
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = self.weights.value.shape

        assert in_channels == in_channels2, "Input channels must match weight channels"

        # Calculate output dimensions
        out_h = h - kh + 1
        out_w = w - kw + 1

        # Use the built-in conv2d operation
        out = minitorch.conv2d(input, self.weights.value)

        # Add bias - reshape it to match broadcasting
        return out + self.bias.value.view(out_channels, 1, 1)





class Network(minitorch.Module):
    """
    Implement a CNN for MNist classification based on LeNet.

    This model should implement the following procedure:

    1. Apply a convolution with 4 output channels and a 3x3 kernel followed by a ReLU (save to self.mid)
    2. Apply a convolution with 8 output channels and a 3x3 kernel followed by a ReLU (save to self.out)
    3. Apply 2D pooling (either Avg or Max) with 4x4 kernel.
    4. Flatten channels, height, and width. (Should be size BATCHx392)
    5. Apply a Linear to size 64 followed by a ReLU and Dropout with rate 25%
    6. Apply a Linear to size C (number of classes).
    7. Apply a logsoftmax over the class dimension.
    """
    def __init__(self):
        super().__init__()

        # First convolution layer: 1->4 channels, 3x3 kernel
        self.conv1 = Conv2d(1, 4, 3, 3)

        # Second convolution layer: 4->8 channels, 3x3 kernel
        self.conv2 = Conv2d(4, 8, 3, 3)

        # Recalculate size after convolutions and pooling
        # Input: 28x28
        # After conv1: 26x26 (28 - 3 + 1)
        # After conv2: 24x24 (26 - 3 + 1)
        # After 4x4 pooling: 6x6 (24/4 = 6)
        # 8 channels * 6 * 6 = 288 features
        self.flattened_size = 8 * 7 * 7

        # Linear layers
        self.fc1 = Linear(self.flattened_size, 64)
        self.fc2 = Linear(64, C)

        # For visualization
        self.mid = None
        self.out = None

    def forward(self, x):

        # First conv + ReLU
        self.mid = self.conv1.forward(x).relu()

        # Second conv + ReLU
        self.out = self.conv2.forward(self.mid).relu()

        # Max pooling with 4x4 kernel
        pooled = minitorch.maxpool2d(self.out, (4, 4))

        # Flatten: combine all dimensions except batch
        batch_size = pooled.shape[0]
        flattened = pooled.view(batch_size, self.flattened_size)
        flattened = pooled.view(batch_size, self.flattened_size)

        # First fully connected + ReLU + Dropout
        fc1_out = self.fc1.forward(flattened).relu()
        fc1_dropped = minitorch.dropout(fc1_out, 0.25)

        # Second fully connected
        logits = self.fc2.forward(fc1_dropped)

        # Apply log softmax over class dimension
        return minitorch.softmax(logits, dim=1).log()


def make_mnist(start, stop):
    ys = []
    X = []
    for i in range(start, stop):
        y = labels[i]
        vals = [0.0] * 10
        vals[y] = 1.0
        ys.append(vals)
        X.append([[images[i][h * W + w] for w in range(W)] for h in range(H)])
    return X, ys


def default_log_fn(epoch, total_loss, correct, total, losses, model):
    print(f"Epoch {epoch} loss {total_loss} valid acc {correct}/{total}")


class ImageTrain:
    def __init__(self):
        self.model = Network()

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x], backend=BACKEND))

    def train(
        self, data_train, data_val, learning_rate, max_epochs=25, log_fn=default_log_fn
    ):
        (X_train, y_train) = data_train
        (X_val, y_val) = data_val
        self.model = Network()
        model = self.model
        n_training_samples = len(X_train)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)
        losses = []
        for epoch in range(1, max_epochs + 1):
            total_loss = 0.0

            model.train()
            for batch_num, example_num in enumerate(
                range(0, n_training_samples, BATCH)
            ):
                if n_training_samples - example_num <= BATCH:
                    continue
                y = minitorch.tensor(
                    y_train[example_num : example_num + BATCH], backend=BACKEND
                )
                x = minitorch.tensor(
                    X_train[example_num : example_num + BATCH], backend=BACKEND
                )
                x.requires_grad_(True)
                y.requires_grad_(True)
                # Forward
                out = model.forward(x.view(BATCH, 1, H, W)).view(BATCH, C)
                prob = (out * y).sum(1)
                loss = -(prob / y.shape[0]).sum()

                assert loss.backend == BACKEND
                loss.view(1).backward()

                total_loss += loss[0]
                losses.append(total_loss)

                # Update
                optim.step()

                if batch_num % 5 == 0:
                    model.eval()
                    # Evaluate on 5 held-out batches

                    correct = 0
                    for val_example_num in range(0, 1 * BATCH, BATCH):
                        y = minitorch.tensor(
                            y_val[val_example_num : val_example_num + BATCH],
                            backend=BACKEND,
                        )
                        x = minitorch.tensor(
                            X_val[val_example_num : val_example_num + BATCH],
                            backend=BACKEND,
                        )
                        out = model.forward(x.view(BATCH, 1, H, W)).view(BATCH, C)
                        for i in range(BATCH):
                            m = -1000
                            ind = -1
                            for j in range(C):
                                if out[i, j] > m:
                                    ind = j
                                    m = out[i, j]
                            if y[i, ind] == 1.0:
                                correct += 1
                    log_fn(epoch, total_loss, correct, BATCH, losses, model)

                    total_loss = 0.0
                    model.train()


if __name__ == "__main__":
    data_train, data_val = (make_mnist(0, 5000), make_mnist(10000, 10500))
    ImageTrain().train(data_train, data_val, learning_rate=0.01)
