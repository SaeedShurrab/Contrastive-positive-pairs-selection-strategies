import boto3
import argparse


s3 = boto3.client('s3')

BUCKET_NAME = 'trainingoutput2021'

checkpoint_name = 'epoch=93-step=25285.ckpt'

checkpoint_uri = 'logs/1/8d7861ac619c48debd23b75e5102f9b8/artifacts/epoch=93-step=25285.ckpt'

print('download started')
s3.download_file(BUCKET_NAME,checkpoint_uri,checkpoint_name)  

print('download completed')

