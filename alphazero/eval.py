# -*- coding: utf-8 -*-
"""
Script use to evaluate an agent.
"""
import argparse

from alphazero.addons.config import ENCODAGE_SIZE
from alphazero.game.config import config_2048
from alphazero.models.network import FinalNetwork
from alphazero.train.self_play import run_eval

# ##: Get arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--model", help="Which type of model tu use", required=True, type=str)
args = parser.parse_args()

# ##: Get configuration.
config = config_2048()
config.self_play.evaluation = 50

# ##: Prepare necessary
network = FinalNetwork(size=ENCODAGE_SIZE, path=args.model)

print("General evaluation ->  score: ", run_eval(config, network))
