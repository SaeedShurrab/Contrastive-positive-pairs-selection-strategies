import os
import argparse
import pytorch_lightning as pl

from pytorch_lightning.loggers import mlflow
from src.models.sslmodels.byol import ByolModel


parser = argparse.ArgumentParser(description='BYOL training command line interface')

parser.add_argument('--strategy',type=str,default='unrestricted')

args = parser.parse_args()


print(args)