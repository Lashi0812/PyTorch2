# python inbuilt-libraries
from typing import Tuple, List, Dict, Optional
from collections import defaultdict
from pathlib import Path
from datetime import datetime


# torch
import torch
from torch import nn
from torchmetrics import Accuracy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# numpy
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D

from tqdm.auto import tqdm


def add_to_class(Class):
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)

    return wrapper

def set_figsize(figsize=(3.5, 2.5)):
    """Set the figure size for matplotlib."""
    plt.rcParams['figure.figsize'] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib."""
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim), axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

def plot(X, Y=None, xlabel=None, ylabel=None, legend=[], xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """Plot data points."""

    def has_one_axis(X):  # True if `X` (tensor or list) has 1 axis
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X): X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)

    set_figsize(figsize)
    if axes is None: axes = plt.gca()
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        axes.plot(x, y, fmt) if len(x) else axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


# Base class for Data Module
class DataModule:
    def __init__(self, root="../data", num_workers=2) -> None:
        self.root = root
        self.num_workers = num_workers

    def get_dataloader(self, train: bool):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)
    
    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        tensors = tuple(a[indices] for a in tensors)
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size,
                                           shuffle=train)


# Base class for Module
class Module(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def loss(self, y_logits, y):
        raise NotImplementedError

    def forward(self, x: torch.Tensor):
        assert hasattr(self, "net"), "save the model in the variable 'net'"
        return self.net(x)

    def configure_optimizer(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)

    def training_step(self, batch) -> Dict:
        X, y = batch
        # forward pass
        y_logits = self(X)
        loss = self.loss(y_logits, y)
        return dict(loss=loss)

    def validation_step(self, batch) -> Dict:
        X, y = batch
        # forward pass
        y_logits = self(X)
        loss = self.loss(y_logits, y)
        return dict(loss=loss)

    def layer_summary(self, input_shape):
        """Print the layer by doing the forward pass"""
        X = torch.rand(size=input_shape)
        assert hasattr(self, "net"), "save the model in the variable 'net'"
        for layer in self.net.children():
            X = layer(X)
            print(f"{layer.__class__.__name__:<15s} output shape :{tuple(X.shape)}")

    def apply_init(self, inputs, init=None):
        self.forward(*inputs)
        if init is not None:
            self.net.apply(init)


# animate class
class Animate:
    def __init__(self, plot_acc: bool, max_epochs: int, verbose: bool = False):
        num_cols = 2 if plot_acc else 1
        self.verbose = verbose
        self.fig, self.axes = plt.subplots(
            nrows=1, ncols=num_cols, figsize=(4 * num_cols, 3), constrained_layout=True
        )
        self.line_dict = defaultdict(Line2D)
        # creating the line plot
        for ax, name in zip(
            np.array(self.axes).ravel(), ["loss"] + (["acc"] if True else [])
        ):
            self.line_dict[f"train_{name}"] = ax.plot(
                [], [], "o-", markevery=[-1], lw=2, label=f"train_{name}"
            )[-1]
            self.line_dict[f"val_{name}"] = ax.plot(
                [], [], "o-", markevery=[-1], lw=2, label=f"val_{name}"
            )[-1]
            ax.set_xlim(0, max_epochs)
            ax.set_xlabel("epochs")
            ax.set_ylabel(f"{name}".title())
            ax.set_title(f"{name} Curve".title())
            ax.legend()

    def update(self, i):
        if self.verbose:
            print(f"[INFO] Entering Animate Update fit")
        if self.trainer.show_ani:
            self.trainer.fit_epoch()
        if self.verbose:
            print(f"[INFO] Finished {i} epoch")

        for key, line in self.line_dict.items():
            if self.verbose:
                print(f"[INFO] Drawing the {key} Line")
            line.set_data(
                list(range(len(self.trainer.history[key]))), self.trainer.history[key]
            )

        for ax in np.array(self.axes).ravel():
            if self.verbose:
                print(f"[INFO] Updating the axes limit")
            ax.autoscale()
            ax.relim()
            ax.autoscale_view()

        return self.line_dict.values()

    def ani_init(self):
        for line in self.line_dict.values():
            line.set_data([], [])
        return self.line_dict.values()

    def animate(self, save: bool = False):
        if self.verbose:
            print("[INFO] Entering Animate ")
        # print(hasattr(self, "trainer"))
        assert hasattr(self, "trainer"), "Plot is not prepared"
        self.anim = animation.FuncAnimation(
            fig=self.fig,
            func=self.update,
            # init_func=self.ani_init,
            frames=tqdm(range(self.trainer.max_epochs)),
            repeat=False,
            blit=True,
        )
        if save:
            print("Saving the animation")
            Writer = animation.writers["ffmpeg"]
            writer = Writer(fps=5, bitrate=1800)
            self.anim.save(
                f"{self.trainer.model.__class__.__name__}.mp4", writer=writer
            )
        plt.show()


# Trainer Class
class Trainer:
    def __init__(
        self,
        max_epochs,
        show_ani=False,
        save_ani=False,
        model_save_path: Optional[Path] = None,
        verbose: bool = False,
    ) -> None:
        self.max_epochs = max_epochs
        self.save_ani = save_ani
        self.show_ani = show_ani
        self.verbose = verbose
        self.model_save_path = model_save_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Training the model in {self.device}")

    def prepare_data(self, data: DataModule):
        if self.verbose:
            print("[INFO] Entering Data Preparation")
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (
            len(self.val_dataloader) if self.val_dataloader is not None else 0
        )

    def prepare_model(self, model: Module):
        if self.verbose:
            print("[INFO] Entering Model Preparation")
        model.trainer = self
        self.model: Module = model.to(self.device)
        self._has_accuracy: bool = (
            True if "accuracy" in dir(self.model.__class__) else False
        )

    def prepare_batch(self, batch):
        batch = [a.to(self.device) for a in batch]
        return batch

    def prepare_plot(self):
        if self.verbose:
            print("[INFO] Entering Plot Preparation")
        self.ani_plot = Animate(
            plot_acc=self._has_accuracy,
            max_epochs=self.max_epochs,
            verbose=self.verbose,
        )
        self.ani_plot.trainer = self

    def fit(self, model: Module, data: DataModule):
        if self.verbose:
            print("[INFO] Entering FIT")
        self.prepare_data(data=data)
        self.prepare_model(model)
        self.prepare_plot()
        self.optim = model.configure_optimizer()
        # all the training and animation is happening here
        # to store the history
        self.history = defaultdict(list)
        if self.show_ani:
            self.ani_plot.animate(self.save_ani)
        else:
            for epoch in tqdm(range(self.max_epochs)):
                self.fit_epoch()
                if self.verbose:
                    print(f'[INFO] At end Epoch {epoch}/{self.max_epochs}  - Training Loss {self.history["train_loss"][-1]}| Validation Loss {self.history["val_loss"][-1]}')
                    
            self.ani_plot.update(self.max_epochs)

            plt.show()

        if self.model_save_path:
            self.save_model(self.model_save_path, self.model.__class__.__qualname__)

    def fit_epoch(self):
        if self.verbose:
            print(f"[INFO] Start Fitting per epoch")
        # do the training step
        self.train_step()
        if self.val_dataloader is None:
            return
        # do the val step
        self.val_step()

    def train_step(self):
        batch_dict = defaultdict(list)
        # put the model in training mode
        self.model.train()
        # Loop through the train dataloader
        for batch_idx,batch in enumerate(self.train_dataloader):
            # do forward and find the loss
            result = self.model.training_step(self.prepare_batch(batch))
            loss = result["loss"]
            batch_dict["loss"].append(loss)
            if self._has_accuracy:
                batch_dict["acc"].append( result["acc"])
            # set the zero grad
            self.optim.zero_grad()
            with torch.no_grad():
                # back propagate the loss
                loss.backward()
                self.optim.step()
            if self.verbose and batch_idx % (self.num_train_batches//5) == 0:
                print(f'\t[INFO] Training Batch  {batch_idx}/{self.num_train_batches} - Training Loss {batch_dict["loss"][-1]}')
        # print(batch_dict.keys())
        for key in batch_dict.keys():
            # print(key)
            self.history[f"train_{key}"].append(
                (sum(batch_dict[key]) / self.num_train_batches).cpu().item()
            )

    def val_step(self):
        # put the model in eval mode
        batch_dict = defaultdict(list)
        self.model.eval()
        # Loop through the val dataloader
        for batch_idx,batch in enumerate(self.val_dataloader):
            with torch.inference_mode():
                result = self.model.validation_step(self.prepare_batch(batch))
                batch_dict["loss"].append( result["loss"])
                if self._has_accuracy:
                    batch_dict["acc"].append( result["acc"])
            if self.verbose and batch_idx % (self.num_val_batches//5) == 0:
                print(f'\t[INFO] Validation Batch {batch_idx}/{self.num_val_batches} - Validation Loss {batch_dict["loss"][-1]}')

        # print("test",batch_dict.keys())
        for key in batch_dict.keys():
            # print("test",key)
            self.history[f"val_{key}"].append(
                (sum(batch_dict[key]) / self.num_val_batches).cpu().item()
            )

    def save_model(self, model_save_path: Path, model_name):
        """Save the model state dict to reuse it."""
        model_name_time = get_model_name(model_name)
        # make dir for current data
        cur_date = datetime.utcnow().date().isoformat()
        Path(model_save_path /cur_date ).mkdir(exist_ok=True, parents=True)
        path = model_save_path/cur_date/ model_name_time
        print(f"[Info] Saving the model at {path}")
        self.save_path = str(path)
        torch.save(self.model.state_dict(), path)


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5,cmap="gray"):
    """
    Take the image in format fo [h,w,c]
    """
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(
        nrows=num_rows, ncols=num_cols, figsize=figsize, constrained_layout=True
    )
    for i, (ax, img) in enumerate(zip(axes.ravel(), imgs)):
        try:
            img = img.cpu().detach().numpy()
        except:
            pass
        ax.imshow(img,cmap=cmap)
        ax.axis("off")
        if titles:
            ax.set_title(titles[i])
    return axes

class LinearRegression(nn.Module):    

    def __init__(self, lr):
        super().__init__()
        self.lr= lr
        self.net = nn.LazyLinear(1)
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)

    def forward(self, X):
        """The linear regression model."""
        return self.net(X)

    def loss(self, y_hat, y):
        fn = nn.MSELoss()
        return fn(y_hat, y)
    
    def step(self, batch: List) -> Dict:
        X, y = batch
        # forward pass
        y_logits = self(X)
        loss = self.loss(y_logits, y)
        return dict(loss=loss)

    def validation_step(self, batch: List):
        return self.step(batch)

    def training_step(self, batch: List):
        return self.step(batch)

    def configure_optimizer(self):
        return torch.optim.SGD(self.parameters(), self.lr)

    def get_w_b(self):
        return (self.net.weight.data, self.net.bias.data)


class Classifier(Module):
    def accuracy(self, y_logits, y):
        acc_fn = Accuracy(
            task="multiclass", num_classes=y_logits.shape[-1]
        ).to(self.trainer.device)
        return acc_fn(y_logits, y)

    def step(self, batch: List) -> Dict:
        X, y = batch
        # forward pass
        y_logits = self(X).squeeze(1)
        loss = self.loss(y_logits, y)
        acc = self.accuracy(y_logits, y)
        return dict(loss=loss, acc=acc)

    def validation_step(self, batch: List):
        return self.step(batch)

    def training_step(self, batch: List):
        return self.step(batch)

    def loss(self, y_logits, y):
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(y_logits, y)


class FashionMNIST(DataModule):
    def __init__(self, batch_size: int = 64, resize=(28, 28)) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.resize = resize

        transform = transforms.Compose(
            [transforms.Resize(resize), transforms.ToTensor()]
        )

        self.train = datasets.FashionMNIST(
            root=self.root, train=True, transform=transform, download=True
        )
        self.val = datasets.FashionMNIST(
            root=self.root, train=False, transform=transform, download=True
        )
        self.classes = self.train.classes
        self.class_to_idx = self.train.class_to_idx

    def text_labels(self, indices: List):
        return [self.classes[a] for a in indices]

    def get_dataloader(self, train: bool):
        data = self.train if train else self.val
        return DataLoader(
            dataset=data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=train,
        )

    def visualize(self, batch: Tuple, num_rows=1, num_cols=8):
        X, y = batch
        labels = self.text_labels(y)
        show_images(X.squeeze(1), num_rows=num_rows, num_cols=num_cols, titles=labels)


def get_model_name(name: str):
    """Add the iso time to model name and the extension at end"""
    return f'{name}-{datetime.utcnow().time().replace(microsecond=0).isoformat().replace(":","")}.pth'


def view_channel(img, kernel,*args):
    intermediate_layers = [l.squeeze() for l in args]
    imgs = [
        img.squeeze(),
        kernel.squeeze(),
        *intermediate_layers
    ]
    title = [tuple(img.shape) for img in imgs]
    show_images(imgs, 1, len(imgs),titles=title)
    plt.show()

def visual_block(img, model, block_start, block_end, kernel_index,trainer,num_kernel_show=5,num_channel_show=5 ):
    """
    Help to view the the transformation the weighted layer
    """
    kernel = list(model.parameters())[kernel_index]
    layers = [kernel.detach().cpu().numpy()]
    for i in range(block_start, block_end):
        layers.append(model.net[:i](img.to(trainer.device)).detach().cpu().numpy())
    # for l in layers:
    #     print(l.shape)
    for filter_ in np.random.randint(0,kernel.shape[0],min(kernel.shape[0],num_kernel_show)):
        for channel in np.random.randint(0,kernel.shape[1],min(kernel.shape[1],num_channel_show)):
            kernel_slice = np.s_[filter_, channel, :, :]
            img_slice = np.s_[filter_, :, :]
            view_channel(
                img,
                layers[0][kernel_slice],
                *[l[img_slice]for l in layers[1:]]               
            )
