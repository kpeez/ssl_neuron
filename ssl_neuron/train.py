import os

import torch
from torch import optim
from utils import AverageMeter, compute_eig_lapl_torch_batch


class Trainer(object):
    def __init__(self, config, model, dataloaders):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.config = config
        self.ckpt_dir = config["trainer"]["ckpt_dir"]
        self.save_every = config["trainer"]["save_ckpt_every"]

        ### datasets
        self.train_loader = dataloaders[0]
        self.val_loader = dataloaders[1]

        ### trainings params
        self.max_iter = config["optimizer"]["max_iter"]
        self.init_lr = config["optimizer"]["lr"]
        self.exp_decay = config["optimizer"]["exp_decay"]
        self.lr_warmup = torch.linspace(0.0, self.init_lr, steps=(self.max_iter // 50) + 1)[1:]
        self.lr_decay = self.max_iter // 5

        self.optimizer = optim.Adam(list(self.model.parameters()), lr=0)

    def set_lr(self):
        if self.curr_iter < len(self.lr_warmup):
            lr = self.lr_warmup[self.curr_iter]
        else:
            lr = self.init_lr * self.exp_decay ** (
                (self.curr_iter - len(self.lr_warmup)) / self.lr_decay
            )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        return lr

    def train(self):
        self.curr_iter = 0
        epoch = 0
        while self.curr_iter < self.max_iter:
            # Run one epoch.
            self._train_epoch(epoch)

            if epoch % self.save_every == 0:
                # Save checkpoint.
                self._save_checkpoint(epoch)

            epoch += 1

    def _train_epoch(self, epoch):
        self.model.train()
        losses = AverageMeter()
        for _, data in enumerate(self.train_loader, 0):
            f1, f2, a1, a2 = [x.float().to(self.device, non_blocking=True) for x in data]
            n = a1.shape[0]

            # compute positional encoding
            l1 = compute_eig_lapl_torch_batch(a1)
            l2 = compute_eig_lapl_torch_batch(a2)

            self.lr = self.set_lr()
            self.optimizer.zero_grad(set_to_none=True)

            loss = self.model(f1, f2, a1, a2, l1, l2)

            # optimize
            loss.sum().backward()
            self.optimizer.step()

            # update teacher weights
            self.model.update_moving_average()

            losses.update(loss.detach(), n)
            self.curr_iter += 1

        print("Epoch {} | Loss {:.4f}".format(epoch, losses.avg))

    def _save_checkpoint(self, epoch):
        filename = "ckpt_{}.pt".format(epoch)
        PATH = os.path.join(self.ckpt_dir, filename)
        torch.save(self.model.state_dict(), PATH)
        print("Save model after epoch {} as {}.".format(epoch, filename))
