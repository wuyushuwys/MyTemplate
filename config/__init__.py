import argparse
import importlib
import yaml
import os


def update_params(params: argparse.Namespace):
    config = getattr(importlib.import_module(f'config.{params.config_file}'), 'config')
    for n, v in config.items():
        params.__setattr__(n, v)


if __name__ == "__main__":
    pass