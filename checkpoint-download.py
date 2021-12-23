import boto3
import argparse


s3 = boto3.client('s3')

BUCKET_NAME = 'trainingoutput2021'

checkpoint_name = 'epoch=50-step=14024.ckpt'

checkpoint_uri = 'logs/5/b92b17862fb84ebfb2ed025695019e53/artifacts/epoch=50-step=14024.ckpt'

print('download started')
s3.download_file(BUCKET_NAME,checkpoint_uri,checkpoint_name)  

print('download completed')

