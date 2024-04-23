import os
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .utils import AverageMeter, compute_eig_lapl_torch_batch


class Trainer(object):
    def __init__(
        self,
        config: Mapping,
        model: nn.Module,
        dataloaders: tuple[DataLoader, DataLoader],
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.config = config
        output_dir = list(Path(config["trainer"]["output_dir"]).glob("run*"))
        run_num = 1 if not output_dir else max([int(str(x).split("-")[-1]) for x in output_dir]) + 1
        self.output_dir = Path(config["trainer"]["output_dir"]) / f"run-{run_num:03d}"
        self.ckpt_dir = self.output_dir / "ckpts"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = config["trainer"]["save_ckpt_every"]
        self.writer = SummaryWriter(log_dir=self.output_dir)
        self.writer.add_hparams({**self.config["model"], **self.config["optimizer"]}, {})
        ### datasets
        self.train_loader, self.val_loader = dataloaders
        ### trainings params
        self.max_iter = config["optimizer"]["max_iter"]
        self.init_lr = config["optimizer"]["lr"]
        self.exp_decay = config["optimizer"]["exp_decay"]
        self.lr_warmup = torch.linspace(0.0, self.init_lr, steps=(self.max_iter // 50) + 1)[1:]
        self.lr_decay = self.max_iter // 5
        self.optimizer = optim.Adam(list(self.model.parameters()), lr=0)

    def set_lr(self) -> Tensor:
        if self.curr_iter < len(self.lr_warmup):
            lr = self.lr_warmup[self.curr_iter]
        else:
            lr = self.init_lr * self.exp_decay ** (
                (self.curr_iter - len(self.lr_warmup)) / self.lr_decay
            )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        return lr

    def train(self) -> None:
        self.curr_iter = 0
        epoch = 0
        print(f"Start training for {self.max_iter // len(self.train_loader)} epochs...")
        while self.curr_iter < self.max_iter:
            # Run one epoch.
            self._train_epoch(epoch)
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
            epoch += 1

        self.writer.close()

    def _train_epoch(self, epoch: int) -> None:
        self.model.train()
        losses = AverageMeter()
        for _, data in enumerate(self.train_loader, 0):
            f1, f2, a1, a2 = [x.float().to(self.device, non_blocking=True) for x in data]
            n = a1.shape[0]

            l1 = compute_eig_lapl_torch_batch(a1)
            l2 = compute_eig_lapl_torch_batch(a2)

            self.lr = self.set_lr()
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.model(f1, f2, a1, a2, l1, l2)
            # optimize
            loss.sum().backward()
            self.optimizer.step()
            self.writer.add_scalar("loss", loss, epoch, new_style=True)
            self.writer.add_scalar("lr", self.lr, epoch, new_style=True)

            # update teacher weights
            self.model.update_moving_average()

            losses.update(loss.detach(), n)
            self.curr_iter += 1

        print("Epoch {} | Loss {:.4f}".format(epoch, losses.avg))

    def _save_checkpoint(self, epoch: int) -> None:
        filename = f"ckpt_{epoch:006d}.pt"
        PATH = os.path.join(self.ckpt_dir, filename)
        torch.save(self.model.state_dict(), PATH)
        print("Save model after epoch {} as {}.".format(epoch, filename))
