import os
import argparse
import pytorch_lightning as pl

from pytorch_lightning.loggers import MLFlowLogger
from torchvision.models import resnet
from src.models.sslmodels.simsiam import SimSiamModel, NegativeCosineSimilarity
from src.models.sslmodels.datasets import UnrestrictedDataLoader
from src.models.sslmodels.datasets import XYRetinaDataLoader
from src.models.sslmodels.datasets import ConsicutiveSessionsDataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
import src.models.basemodels as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))





parser = argparse.ArgumentParser(description='BYOL training command line interface')

parser.add_argument('-s','--strategy', type=str, default='unrestricted', 
                    choices=['unrestricted', 'xyscans', 'consecutive'],
                    metavar='TRAINING-PLAN', help='positive pairs selection strategy | \
                    strategies: (\'unrestricted\', \'xyscans\', \'consecutive\') \
                    | default: (unrestricted)' 
                   )

# data loading arguments
parser.add_argument('-d','--data-dir', type=str, 
                    default=os.path.join(os.curdir,'data','pretext'),
                    metavar='DIR', help='path to the training data | default: (./data/pretext)'
                   )
parser.add_argument('-b','--batch-size', type=int, default=256,
                    help='total number of gpus used during training | default: (256)'
                   )
parser.add_argument('-w', '--num-workers', type=int, default=8, metavar='N',
                    help='number of data loading workers | default: (8)'
                   )
parser.add_argument('-m', '--pin-memory', type=bool, default=True, metavar='N',
                    help='number of data loading workers | default: (8)'
                   )

# model arguments
parser.add_argument('--backbone', '--bb',type=str, default='resnet50', 
                    choices=model_names,
                    metavar='BACKBONE', help='model backbone | ' + \
                    'backbones: ('+  ', '.join(model_names) +') \
                    | default: (resnet50)' 
                   )
parser.add_argument('-o','--optimizer', type=str, default='sgd', 
                    choices=['Adam','adam', 'SGD', 'sgd'],
                    metavar='OPT', help='model optimizer | ' + \
                    'optimizers: (adam, sgd)| default: (sgd)' 
                   )
parser.add_argument('--learning-rate','--lr', type=float, default=0.01, metavar='LR',
                    help='model learning rate | default: (0.01)'
                   )
parser.add_argument('--weight-decay','--wd', type=float, default=0.0, metavar='WD',
                    help='L2 weight decay | default: (0.0)'
                   )
parser.add_argument('--scheduler','--sc', type=str, default='step', 
                    choices=['step','exponential'],
                    metavar='SCHED', help=' learning rate schduler  | ' + \
                    'schedulers: (step, exponential) | default: (step)' 
                   )
parser.add_argument('--scheduler-step','--ss', type=int, default=5, metavar='SS',
                    help='StepLR scheduler step size | default: (5)'
                   )
parser.add_argument('--scheduler-gamma','--sg', type=float, default=0.5, metavar='SG',
                    help='learrning rate reduction factor | default: (0.5)'
                   )                    

# trainer arguments 
parser.add_argument('-g','--ngpus', type=int, default=-1, metavar='N',
                    help='total number of gpus used during training  | default: (-1)'
                   )
parser.add_argument('-e','--epochs', type=int, default=100, metavar='N',
                    help='maximum number of training epochs | default: (100)'
                   )
parser.add_argument('-p','--precision', type=int, default=16, metavar='N',
                    help='auotomatic mexed precesion mode applied during the training  | default: (16)'
                   )

args = parser.parse_args()


if args.strategy == 'unrestricted':
    data_module = UnrestrictedDataLoader(data_dir=args.data_dir,
                                         transforms=None,
                                         batch_size=args.batch_size,
                                         num_workers=args.num_workers,
                                         pin_memory=args.pin_memory
                                        )

elif args.strategy == 'xyscans':
    data_module = XYRetinaDataLoader(data_dir=args.data_dir,
                                     transforms=None,
                                     batch_size=args.batch_size,
                                     num_workers=args.num_workers,
                                     pin_memory=args.pin_memory
                                    )


elif args.strategy == 'consecutive':
    data_module = ConsicutiveSessionsDataLoader(data_dir=args.data_dir,
                                                transforms=None,
                                                batch_size=args.batch_size,
                                                num_workers=args.num_workers,
                                                pin_memory=args.pin_memory
                                               )


model = SimSiamModel(backbone=models.__dict__[args.backbone],
                     criterion=NegativeCosineSimilarity,
                     optimizer=args.optimizer,
                     learning_rate=args.learning_rate,
                     weight_decay=args.weight_decay,
                     scheduler=args.scheduler,
                     sched_step_size=args.scheduler_step,
                     sched_gamma=args.scheduler_gamma,
                    )


checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")
mlflow_logger = MLFlowLogger(experiment_name='test',save_dir='logs')
lr_logger = LearningRateMonitor(logging_interval='epoch',)
early_stop = EarlyStopping(monitor="train_loss", min_delta=0.001, mode="min", patience=5,)

trainer = pl.Trainer(gpus=args.ngpus, logger=mlflow_logger, max_epochs=args.epochs,
                     callbacks=[checkpoint_callback, lr_logger, early_stop],
                     precision=args.precision)

#if __name__ == '__main__':
#    print(args)

print(args)
