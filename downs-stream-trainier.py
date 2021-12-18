import os
import json
import argparse
import mlflow
import torch
import pytorch_lightning as pl
#import src.models.basemodels as models
import torch.nn as nn
from src.modules.downstream.classification import ClassificationModel
from src.data.downstream.datasets import DownStreamDataModule
#from pytorch_lightning.loggers import MLFlowLogger
from src.modules.utils import MLFlowLoggerCheckpointer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from src.modules.utils import parse_weights
from torch.nn import functional as F

import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='down-stream training command line interface')


# General-info

parser.add_argument('--training-scheme', '--ts', type=str, default='linear', metavar='TRAINING-PROTOCOL',
                    choices=['linear','fine-tune','from-scratch','transfer-learning'],
                    help='scheme ate which the training will be performed | \
                          schemes: (linear, fine-tune, from-scratch, transfer-learning)'
)

parser.add_argument('--ssl-model','--ssl', type=str,default='SimSiam',metavar='MODEL',
                    choices=['BYOL','SimSiam'],
                    help='self-supervised learning model used as pretraining method'
                   )

parser.add_argument('-s','--strategy', type=str, default='unrestricted', metavar='TRAINING-PLAN',
                    choices=['unrestricted', 'xyscans', 'consecutive'],
                    help='positive pairs selection strategy | \
                    strategies: (unrestricted, xyscans, consecutive) \
                    | default: (unrestricted)' 
                   )

parser.add_argument('--weights-path', '--ckpt', type=str,default=os.path.join(os.curdir),metavar='WEIGHTS',
                    help=' pathe to the base pretrained weights to load'
                   )

parser.add_argument('--classification-problem','--cp', type=str,default='binary',metavar='TASK',
                    choices=['binary','multi-class','grading'],
                    help='OCT classification problem to be accomplished | problems: \
                        (binary, multi-class, grading) | default: binary'
                   )

# data loading options

parser.add_argument('-d','--data-dir', type=str, 
                    default=os.path.join(os.curdir,'data','down-stream'),
                    metavar='DIR', help='path to the training data | default: (./data/down-stream/)'
                   )
parser.add_argument('-b','--batch-size', type=int, default=32,
                    help='total number of gpus used during training | default: (256)'
                   )
parser.add_argument('-w', '--num-workers', type=int, default=8, metavar='N',
                    help='number of data loading workers | default: (8)'
                   )
parser.add_argument('-m', '--pin-memory', type=bool, default=True, metavar='N',
                    help='number of data loading workers | default: (8)'
                   )

# model options
parser.add_argument('--backbone', '--bb',type=str, default='resnet34', 
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
parser.add_argument('--output-dim', '--od', type=int, default=3, metavar='DIM',
                    help='number classes in classifciation problem'
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

parser.add_argument('-t','--tracking-uri',type=str, default='file:///src/logs', metavar='URI',
                    help='Mlflow tracking uri directory | default: (file:///src/logs)'
                   )
#http://ec2-13-59-105-139.us-east-2.compute.amazonaws.com/
# callbacks options
parser.add_argument('--monitor-quantity','--mq', type=str, default='val_loss',metavar='MODE',
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




if args.classification_problem == 'binary':
    data_dir = os.path.join(args.data_dir,'binary')

elif args.classification_problem ==' multi-class':
    data_dir = os.path.join(args.data_dir,'multi-class')

elif args.classification_problem == 'grading':
    disease = input('please enter disease name from (CSR, MRO, GA, CNV, FMH, PMH, VMT): ')
    data_dir = os.path.join(args.data_dir,'grading',disease)   

data_module = DownStreamDataModule(data_dir=data_dir,
                                   form=args.classification_problem,
                                   training_transforms=None,
                                   val_test_transforms=None,
                                   batch_size=args.batch_size,
                                   num_workers=args.num_workers,
                                   pin_memory=args.pin_memory
                                  )


if args.training_scheme in ['linear', 'transfer-learning']:
    freeze = True
else:
    freeze = False



model = ClassificationModel(model=models.__dict__[args.backbone],
                            criterion=F.cross_entropy,
                            optimizer=args.optimizer,
                            learning_rate=args.learning_rate,
                            weight_decay=args.weight_decay,
                            scheduler=args.scheduler,
                            sched_step_size=args.scheduler_step,
                            sched_gamma=args.scheduler_gamma,
                            output_dim=args.output_dim,
                            freeze= freeze,
                            max_epochs=args.epochs
                            )

all_weights = torch.load('./epoch=64-step=26974.ckpt',map_location=torch.device('cpu'))['state_dict']
ultimate_weights = parse_weights(all_weights)


if args.training_scheme in ['linear', 'fine-tune']:
    model.model.load_state_dict(ultimate_weights,strict = False)



version  = input(f'please specfiy the the current version of SimSiamS expriment and {args.strategy} run: ')



mlflow_logger = MLFlowLoggerCheckpointer(experiment_name=args.training_scheme, 
                                         tracking_uri=args.tracking_uri,
                                         run_name=' '.join([args.ssl_model, args.strategy, args.classification_problem]),
                                         tags={'training-scheme': args.training_scheme,
                                               'ssl-model':args.ssl_model,
                                               'strategy':args.strategy,
                                               'classification-problem': args.classification_problem,
                                               'Version': version,
                                              }
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
                     )


if __name__ == '__main__':
    with open('args.json', 'w') as fp:
        json.dump(vars(args), fp)
    trainer.fit(model=model, datamodule=data_module)
    os.remove('./args.json')
    



# python downs-stream-trainer.py --training-scheme linear --ssl-model SimSiam --strategy unrestricted --weights-path ./epoch=64-step=26974.ckpt --classification-problem binary --data-dir ./data/down-stream --batch-size 128 --pin-memory True --backbone resnet34 --optimizer adam --learning-rate 0.01 --weight-decay 0.0 --scheduler cosine --ngpus -1 --epochs 100 --precision 16 --es-delta 0.01 --es-patience 5 --output-dim 3

# python downs-stream-trainier.py --training-scheme from-scratch --ssl-model SimSiam --strategy unrestricted --weights-path ./epoch=64-step=26974.ckpt --classification-problem grading --data-dir ./data/down-stream --batch-size 16 --pin-memory False --num-workers 0 --backbone resnet18 --optimizer adam --learning-rate 0.0001 --weight-decay 0.001 --scheduler cosine --ngpus 0 --epochs 10 --precision 32 --es-delta 0.01 --es-patience 5 --output-dim 3
