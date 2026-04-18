
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def linear_link(x):
    return x


def cargar_npz(ruta):
    data = np.load(ruta)
    if isinstance(data, np.lib.npyio.NpzFile):
        if len(data.files) == 0:
            raise ValueError(f"El archivo {ruta} no contiene arrays")
        array = data[data.files[0]]
    else:
        array = data
    return array


def infer_image_shape(X):
    """
    Infer the image shape from the current data.
    Uses the shape already implied in test.ipynb (80, 70) if data are flattened to 5600.
    Falls back to (28, 28) if data are flattened to 784.
    """
    if X.ndim == 3:
        return X.shape[1], X.shape[2]

    if X.ndim != 2:
        raise ValueError(f"No se puede inferir la forma de imagen a partir de X.shape = {X.shape}")

    flat_dim = X.shape[1]
    if flat_dim == 80 * 70:
        return (80, 70)
    elif flat_dim == 28 * 28:
        return (28, 28)
    else:
        raise ValueError(
            f"No reconozco la dimensión aplanada {flat_dim}. "
            "Actualmente este código detecta automáticamente 80x70 y 28x28."
        )


class ReshapeToTensor:
    def __init__(self, image_shape):
        self.image_shape = image_shape

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float32).reshape(self.image_shape)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
        return x


class RandomSmallRotation:
    """
    Small random rotation transform, motivated by the augmentation section
    of the convolution notebook, where small plausible transformations are encouraged.
    """
    def __init__(self, degrees=10.0):
        self.degrees = degrees

    def __call__(self, x):
        angle = float(np.random.uniform(-self.degrees, self.degrees))
        return F.rotate(x, angle)


class NumpyImageDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, transform=None):
        super().__init__()
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]

        if self.transform is not None:
            x = self.transform(x)

        y = torch.tensor(y, dtype=torch.long)
        return x, y


def compute_mean_std(X, image_shape):
    reshape_to_tensor = ReshapeToTensor(image_shape)
    images = [reshape_to_tensor(x) for x in X]
    X_tensor = torch.stack(images, dim=0)  # [N, 1, H, W]
    mean = X_tensor.mean().item()
    std = X_tensor.std().item()

    if std == 0.0:
        std = 1.0

    return mean, std


class SmallConvNet(nn.Module):
    def __init__(
        self,
        dim_out,
        input_shape,
        activation,
        apply_bn,
        drop_prob,
        link_function,
        loss_function
    ):
        super().__init__()

        self.activation = activation
        self.apply_bn = apply_bn
        self.drop = nn.Dropout2d(drop_prob)
        self.link = link_function
        self.loss = loss_function

        # Block 1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # Block 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)

        # Block 3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        # Block 4
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)

        # Obtain the final spatial map size and define AvgPool2d accordingly
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_shape[0], input_shape[1])
            dummy = self._forward_features(dummy)
            k_h, k_w = dummy.shape[-2], dummy.shape[-1]

        self.avg_pool = nn.AvgPool2d(kernel_size=(k_h, k_w))
        self.linear = nn.Linear(128, dim_out)

    def _bn_or_identity(self, x, bn_layer):
        if self.apply_bn:
            return bn_layer(x)
        return x

    def _forward_features(self, x):
        x = self.conv1(x)
        x = self._bn_or_identity(x, self.bn1)
        x = self.activation(x)
        x = self.drop(x)

        x = self.conv2(x)
        x = self._bn_or_identity(x, self.bn2)
        x = self.activation(x)
        x = self.drop(x)

        x = self.conv3(x)
        x = self._bn_or_identity(x, self.bn3)
        x = self.activation(x)
        x = self.drop(x)

        x = self.conv4(x)
        x = self._bn_or_identity(x, self.bn4)
        x = self.activation(x)
        x = self.drop(x)

        return x

    def operator(self, x):
        x = self._forward_features(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x

    def forward_train(self, x, apply_link):
        self.train()
        x = self.operator(x)
        if apply_link:
            x = self.link(x)
        return x

    def forward_eval(self, x, apply_link):
        self.eval()
        x = self.operator(x)
        if apply_link:
            x = self.link(x)
        return x

    def compute_loss(self, t, y):
        return self.loss(y, t)


def compute_metric(dataloader, model):
    acc = 0.0
    tot_samples = 0
    for x, t in dataloader:
        x, t = x.to(device), t.to(device)
        y = model.forward_eval(x, apply_link=False)
        acc += (t == torch.argmax(y, dim=1)).sum()
        tot_samples += len(t)

    return acc / tot_samples


def test_model(
    model: torch.nn.Module,
    epochs: int,
    train_batch_size: int,
    eval_each: int,
    train_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset
):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    train_loader_eval = torch.utils.data.DataLoader(train_dataset, batch_size=1000, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    N_training = len(train_dataset)

    loss_epochs = []
    train_acc_epochs = []
    test_acc_epochs = []

    with torch.no_grad():
        loss_acc = 0.0
        for batch_idx, (x, t) in enumerate(train_loader_eval):
            x, t = x.to(device), t.to(device)
            y = model.forward_train(x, apply_link=False)
            L = model.compute_loss(t, y)
            loss_acc += len(x) * L.item()

        train_acc = compute_metric(train_loader_eval, model)
        test_acc = compute_metric(test_loader, model)

    loss_epochs.append(loss_acc / N_training)
    train_acc_epochs.append(train_acc)
    test_acc_epochs.append(test_acc)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for e in range(epochs):
        loss_acc = 0.0
        for batch_idx, (x, t) in enumerate(train_loader):
            x, t = x.to(device), t.to(device)

            y = model.forward_train(x, apply_link=False)
            L = model.compute_loss(t, y)
            loss_acc += len(x) * L.item()

            L.backward()
            optimizer.step()
            optimizer.zero_grad()

            print(f"On epoch {e+1} batch_idx {batch_idx + 1} got loss {L.item():.5f}", end="\r")

        scheduler.step()

        with torch.no_grad():
            train_acc = compute_metric(train_loader_eval, model)
            test_acc = compute_metric(test_loader, model)

        print(" " * 200, end="\r")
        print(
            f"On epoch {e+1} got loss {loss_acc / N_training:.5f} "
            f"with train accuracy {train_acc:.5f} and test accuracy {test_acc:.5f}"
        )

        loss_epochs.append(loss_acc / N_training)
        train_acc_epochs.append(train_acc)
        test_acc_epochs.append(test_acc)

    return loss_epochs, train_acc_epochs, test_acc_epochs


# =========================== ##
# Data pipeline configuration ##
# =========================== ##

dataset_dir = Path("datasets")
x_path = dataset_dir / "X_train.npz"
y_path = dataset_dir / "Y_train.npz"

X = cargar_npz(x_path)
Y = cargar_npz(y_path)

X = np.asarray(X)
Y = np.asarray(Y).reshape(-1)

image_shape = infer_image_shape(X)

# Encode labels as 0, 1, ..., K-1
classes, Y_encoded = np.unique(Y, return_inverse=True)
num_classes = len(classes)

# Train / validation split because the current notebook only points to training npz files
perm = torch.randperm(len(X)).numpy()
n_train = int(0.8 * len(X))

idx_train = perm[:n_train]
idx_test = perm[n_train:]

X_train = X[idx_train]
Y_train = Y_encoded[idx_train]

X_test = X[idx_test]
Y_test = Y_encoded[idx_test]

# Compute normalization stats only on training split
mean, std = compute_mean_std(X_train, image_shape)

transform_train = transforms.Compose([
    ReshapeToTensor(image_shape),
    RandomSmallRotation(degrees=10.0),
    transforms.Normalize((mean,), (std,))
])

transform_test = transforms.Compose([
    ReshapeToTensor(image_shape),
    transforms.Normalize((mean,), (std,))
])

train_dataset = NumpyImageDataset(X_train, Y_train, transform=transform_train)
test_dataset = NumpyImageDataset(X_test, Y_test, transform=transform_test)

print("Image shape:", image_shape)
print("Training samples:", len(train_dataset))
print("Validation/Test samples:", len(test_dataset))
print("Number of classes:", num_classes)
print("Normalization mean:", mean)
print("Normalization std:", std)

model = SmallConvNet(
    dim_out=num_classes,
    input_shape=image_shape,
    activation=torch.relu,
    apply_bn=True,
    drop_prob=0.15,
    link_function=linear_link,
    loss_function=nn.CrossEntropyLoss()
).to(device)

loss_epochs, train_acc_epochs, test_acc_epochs = test_model(
    model=model,
    epochs=40,
    train_batch_size=16,
    eval_each=1,
    train_dataset=train_dataset,
    test_dataset=test_dataset
)

torch.save(model.state_dict(), 'model_weights_custom_cnn.pth')

# ========= ##
# Plotting  ##
# ========= ##
epochs_axis = np.arange(len(loss_epochs))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(epochs_axis, loss_epochs)
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Cross entropy")

axes[1].plot(epochs_axis, [x.item() if torch.is_tensor(x) else x for x in train_acc_epochs])
axes[1].set_title("Train accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")

axes[2].plot(epochs_axis, [x.item() if torch.is_tensor(x) else x for x in test_acc_epochs])
axes[2].set_title("Validation/Test accuracy")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Accuracy")

plt.tight_layout()
plt.show()
