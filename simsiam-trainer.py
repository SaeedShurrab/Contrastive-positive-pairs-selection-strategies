import os
import json
import argparse
import mlflow
import pytorch_lightning as pl

#from pytorch_lightning.loggers import MLFlowLogger
from src.modules.utils import MLFlowLoggerCheckpointer
from torchvision.models import resnet
from src.models.sslmodels.simsiam import NegativeCosineSimilarity
from src.modules.pretext.simsiam import SimSiamModel
from src.data.pretext.datasets import UnrestrictedDataModule
from src.data.pretext.datasets import XYScansDataModule
from src.data.pretext.datasets import ConsicutiveSessionsDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
#import src.models.basemodels as models
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))




parser = argparse.ArgumentParser(description='BYOL training command line interface')

# training mode option
parser.add_argument('-s','--strategy', type=str, default='unrestricted', 
                    choices=['unrestricted', 'xyscans', 'consecutive'],
                    metavar='TRAINING-PLAN', help='positive pairs selection strategy | \
                    strategies: (\'unrestricted\', \'xyscans\', \'consecutive\') \
                    | default: (unrestricted)' 
                   )


# data loading options
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

# model options
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
parser.add_argument('--scheduler','--sc', type=str, default='cosine', 
                    choices=['step','exponential','cosine'],
                    metavar='SCHED', help=' learning rate schduler  | ' + \
                    'schedulers: (step, exponential) | default: (step)' 
                   )
parser.add_argument('--scheduler-step','--ss', type=int, default=5, metavar='SS',
                    help='StepLR scheduler step size | default: (5)'
                   )
parser.add_argument('--scheduler-gamma','--sg', type=float, default=0.5, metavar='SG',
                    help='learrning rate reduction factor | default: (0.5)'
                   )                    

# trainer options
parser.add_argument('-g','--ngpus', type=int, default=-1, metavar='N',
                    help='total number of gpus used during training  | default: (-1)'
                   )
parser.add_argument('-e','--epochs', type=int, default=100, metavar='N',
                    help='maximum number of training epochs | default: (100)'
                   )
parser.add_argument('-p','--precision', type=int, default=16, metavar='N',
                    help='auotomatic mexed precesion mode applied during the training  | default: (16)'
                   )
parser.add_argument('--log-every-n','--le', type=int, default=1, metavar='FREQUENCY',
                    help='logging frequency every n steps  | default: (1)'
                   )

# logger options

parser.add_argument('-t','--tracking-uri',type=str, default='http://ec2-13-59-105-139.us-east-2.compute.amazonaws.com', metavar='URI',
                    help='Mlflow tracking uri directory | default: (file:///src/logs)'
                   )

# callbacks options
parser.add_argument('--monitor-quantity','--mq', type=str, default='train_loss',metavar='MODE',
                   help='quantity to monitor for checkpoint saving default :| (train_loss)'
                   )
parser.add_argument('--monitor-mode','--mm', type=str, default='min',metavar='MODE',
                   help='checkpoints saving mode based on the monitored quantity | default: (min)'
                   )
parser.add_argument('--es-delta', '--esd', type=float, default=0.001, metavar='DELTA',
                   help='minimum change in the monitoring quantity to early stop training | defualt: (0.001)'
                   )
parser.add_argument('--es-patience','--esp', type=int, default=5,metavar='PATIENCE',
                   help='minimum number of epochs without change in the monitor quantity before ealy stopping | defualt (5)'
                   )


args = parser.parse_args()




if args.strategy == 'unrestricted':
    data_module = UnrestrictedDataModule(data_dir=args.data_dir,
                                         transforms=None,
                                         batch_size=args.batch_size,
                                         num_workers=args.num_workers,
                                         pin_memory=args.pin_memory
                                        )

elif args.strategy == 'xyscans':
    data_module = XYScansDataModule(data_dir=args.data_dir,
                                     transforms=None,
                                     batch_size=args.batch_size,
                                     num_workers=args.num_workers,
                                     pin_memory=args.pin_memory
                                    )


elif args.strategy == 'consecutive':
    data_module = ConsicutiveSessionsDataModule(data_dir=args.data_dir,
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
                     max_epochs=args.epochs
                    )


version  = input(f'please specfiy the the current version of SimSiamS expriment and {args.strategy} run: ')
os.environ['MLFLOW_TRACKING_URI'] = args.tracking_uri
mlflow_logger = MLFlowLoggerCheckpointer(experiment_name='SimSiam', 
                                         tracking_uri=os.environ['MLFLOW_TRACKING_URI'],
                                         run_name=args.strategy,
                                         tags={'Version': version}
                                        )
checkpoint_callback = ModelCheckpoint(monitor=args.monitor_quantity, 
                                      mode= args.monitor_mode
                                     )
early_stop = EarlyStopping(monitor=args.monitor_quantity, 
                           min_delta=args.es_delta,
                           mode=args.monitor_mode, 
                           patience=args.es_patience
                          )
lr_logger = LearningRateMonitor(logging_interval='epoch')
trainer = pl.Trainer(gpus=args.ngpus,
                     logger=mlflow_logger, 
                     max_epochs=args.epochs,
                     precision=args.precision, 
                     log_every_n_steps=args.log_every_n, 
                     progress_bar_refresh_rate=1,
                     callbacks=[checkpoint_callback, lr_logger, early_stop],
                     auto_lr_find=True
                     )


if __name__ == '__main__':
    with open('args.json', 'w') as fp:
        json.dump(vars(args), fp)
    lr_finder = trainer.tune(model,datamodule=data_module)
    trainer.fit(model=model, datamodule=data_module)
    os.remove('./args.json')






'''
python simsiam-trainer.py --strategy unrestricted --data-dir /datastores/pretext --batch-size 128 --num-workers 8 --pin-memory True --backbone resnet18 --optimizer sgd --learning-rate 0.001 --weight-decay 0.0001 --scheduler cosine --ngpus -1 --epochs 100 --precision 16 --es-delta 0.001 --es-patience 5
'''


