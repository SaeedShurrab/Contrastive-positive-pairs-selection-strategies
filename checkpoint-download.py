import boto3
import argparse


s3 = boto3.client('s3')

BUCKET_NAME = 'trainingoutput2021'

checkpoint_name = 'checkpoint.ckpt'

checkpoint_uri = 'logs/7/f38c13ba160f4ce19f82b49e7a4ef054/artifacts/checkpoint.ckpt'

print('download started')
s3.download_file(BUCKET_NAME,checkpoint_uri,checkpoint_name)  

print('download completed')

