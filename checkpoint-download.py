import boto3
import argparse


s3 = boto3.client('s3')

BUCKET_NAME = 'trainingoutput2021'

checkpoint_name = 'epoch=64-step=26974.ckpt'

checkpoint_uri = f'logs/1/a9994e6ca42447df8c71a59bba326665/artifacts/{checkpoint_name}'


print('download started')
s3.download_file(BUCKET_NAME,checkpoint_uri,checkpoint_name)  

print('download completed')

