from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import models.discriminator

from .model import Model
from .discriminator import UNetDiscriminatorSN


def create_discrinimator(args:argparse.Namespace):

    d_arch = getattr(args, 'd_arch')
    discriminator_module = models.discriminator
    assert hasattr(discriminator_module, d_arch), f"{d_arch} not implemented or not found"
    d_model = getattr(discriminator_module, d_arch)(args)

    return d_model
