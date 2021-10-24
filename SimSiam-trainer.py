import os
import argparse
import pytorch_lightning as pl

from src.models.sslmodels.simsiam import SimSiamModel


parser = argparse.ArgumentParser(description='BYOL training command line interface')



args = parser.parse_args()