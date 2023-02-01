# python inbuilt-libraries
from typing import Tuple, List, Dict
from collections import defaultdict
import os

# torch
import torch
from torch import nn
from torchmetrics import Accuracy

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


# Base class for Module
class Module(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def loss(self, y_logits, y):
        raise NotImplementedError

    def forward(self, x: torch.Tensor):
        assert hasattr(self, "net") ,"save the model in the variable 'net'"
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

    def layer_summary(self,input_shape):
        """Print the layer by doing the forward pass"""
        X = torch.rand(size=input_shape)
        assert hasattr(self,"net") ,"save the model in the variable 'net'"
        for layer in self.net.children():
            X = layer(X)
            print(f'{layer.__class__.__name__:<15s} output shape :{tuple(X.shape)}')
        
    def apply_init(self,inputs,init=None):
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
        self, max_epochs, show_ani=False, save_ani=False, verbose: bool = False
    ) -> None:
        self.max_epochs = max_epochs
        self.save_ani = save_ani
        self.show_ani = show_ani
        self.verbose = verbose
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
            self.ani_plot.update(self.max_epochs)

            plt.show()

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
        if self.verbose:
            print("[INFO] Entered into the training step")
        batch_dict = defaultdict(int)
        # put the model in training mode
        self.model.train()
        # Loop through the train dataloader
        for batch in self.train_dataloader:
            # do forward and find the loss
            result = self.model.training_step(self.prepare_batch(batch))
            loss = result["loss"]
            batch_dict["loss"] += loss
            if self._has_accuracy:
                batch_dict["acc"] += result["acc"]
            # set the zero grad
            self.optim.zero_grad()
            with torch.no_grad():
                # back propagate the loss
                loss.backward()
                self.optim.step()
        # print(batch_dict.keys())
        for key in batch_dict.keys():
            # print(key)
            self.history[f"train_{key}"].append(
                (batch_dict[key] / self.num_train_batches).cpu().item()
            )

    def val_step(self):
        if self.verbose:
            print("[INFO] Entered into the Val step")
        # put the model in eval mode
        batch_dict = defaultdict(int)
        self.model.eval()
        # Loop through the val dataloader
        for batch in self.val_dataloader:
            with torch.inference_mode():
                result = self.model.validation_step(self.prepare_batch(batch))
                batch_dict["loss"] += result["loss"]
                if self._has_accuracy:
                    batch_dict["acc"] += result["acc"]

        # print("test",batch_dict.keys())
        for key in batch_dict.keys():
            # print("test",key)
            self.history[f"val_{key}"].append(
                (batch_dict[key] / self.num_val_batches).cpu().item()
            )


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """
    Take the image in format fo [h,w,c]
    """
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(
        nrows=num_rows, ncols=num_cols, figsize=figsize, layout="constrained"
    )
    for i, (ax, img) in enumerate(zip(axes.ravel(), imgs)):
        ax.imshow(img)
        ax.axis("off")
        if titles:
            ax.set_title(titles[i])
    return axes


class Classifier(Module):
    def accuracy(self,y_logits, y):
        acc_fn = Accuracy(task="multiclass",num_classes=list(self.modules())[-1].out_features).to(self.trainer.device)
        return acc_fn(y_logits, y)

    def step(self, batch: List)->Dict:
        X, y = batch
        # forward pass
        y_logits = self(X).squeeze(1)
        loss = self.loss(y_logits, y)
        acc = self.accuracy(y_logits, y)
        return dict(loss=loss,acc=acc)

    def validation_step(self, batch: List):
        return self.step(batch)

    def training_step(self, batch: List):
        return self.step(batch)

    def loss(self, y_logits, y):
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(y_logits, y)
