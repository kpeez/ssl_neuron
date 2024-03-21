import argparse
import json

from ssl_neuron.datasets import build_dataloader
from ssl_neuron.graphdino import create_model
from ssl_neuron.train import Trainer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", help="Path to config file.", type=str, default="./configs/config.json"
)


def main(args):
    # load config
    with open(args.config, "r") as f:
        config = json.load(f)

    # load data
    print("Loading dataset: {}".format(config["data"]["class"]))
    train_loader, val_loader = build_dataloader(config)

    # build model
    model = create_model(config)
    trainer = Trainer(config, model, [train_loader, val_loader])

    print("Start training.")
    trainer.train()
    print("Done.")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
