import os
import json
import argparse
import mlflow
import pytorch_lightning as pl
#from pytorch_lightning.loggers import MLFlowLogger
from src.modules.utils import MLFlowLoggerCheckpointer
from src.models.sslmodels.byol import NormalizedMSELoss
from src.modules.pretext.byol import ByolModel
from src.data.pretext.datasets import UnrestrictedDataModule
from src.data.pretext.datasets import XYScansDataModule
from src.data.pretext.datasets import ConsicutiveSessionsDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
import src.models.basemodels as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))





parser = argparse.ArgumentParser(description='BYOL training command line interface')
# genral options
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
parser.add_argument('--target-decay','--td', type=float, default=0.996, metavar='TD',
                    help='target network decay factor  | default: (0.996)'
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
parser.add_argument('--scheduler','--sc', default='step', 
                    choices=['','step','exponential'],
                    metavar='SCHED', help=' learning rate schduler  | ' + \
                    'schedulers: (step, exponential) | default: (step)' 
                   )
parser.add_argument('--scheduler-step','--ss', type=int, default=1, metavar='SS',
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

parser.add_argument('-t','--tracking-uri',type=str, default='http://ec2-18-224-29-40.us-east-2.compute.amazonaws.com', metavar='URI',
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




model = ByolModel(backbone=models.__dict__[args.backbone],
                  criterion=NormalizedMSELoss,
                  target_decay=args.target_decay,
                  optimizer=args.optimizer,
                  learning_rate=args.learning_rate,
                  weight_decay=args.weight_decay,
                  scheduler=args.scheduler,
                  sched_step_size=args.scheduler_step,
                  sched_gamma=args.scheduler_gamma,
                 )


version  = input(f'please specfiy the the current version of BYOL expriment and {args.strategy} run: ')


mlflow_logger = MLFlowLoggerCheckpointer(experiment_name='BYOL1', 
                             tracking_uri=args.tracking_uri,
                             run_name=args.strategy,
                             tags={'Version': version}
                             )
checkpoint_callback = ModelCheckpoint(monitor=args.monitor_quantity, 
                                      mode= args.monitor_mode,filename='checkpoint'
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
                     )


if __name__ == '__main__':


    with open('args.json', 'w') as fp:
        json.dump(vars(args), fp)
    trainer.fit(model=model, datamodule=data_module)







'''
python byol-trainer.py --strategy unrestricted \
--data-dir ./data/pretext \
--batch-size 128 \
--num-workers 8 \
--pin-memory True \
--target-decay 0.996 \
--backbone resnet50 \
--optimizer adam \
--learning-rate 0.01 \
--weight-decay 0.0 \
--scheduler step \
--scheduler-step 5 \
--scheduler-gamma 0.5 \
--ngpus -1 \
--epochs 100 \
--precision 16 \
--log-every-n 1 \
--tracking-uri file:///src/logs \
--monitor-quantity train_loss \
--monitor-mode min \
--es-delta 0.01 \
--es-patience 5 
'''