import boto3
import argparse


s3 = boto3.client('s3')

BUCKET_NAME = 'trainingoutput2021'

checkpoint_name = 'epoch=86-step=36104.ckpt'

checkpoint_uri = 'logs/1/f42b37e4bbcd442cb17326d2548b3938/artifacts/epoch=86-step=36104.ckpt'

print('download started')
s3.download_file(BUCKET_NAME,checkpoint_uri,checkpoint_name)  

print('download completed')

