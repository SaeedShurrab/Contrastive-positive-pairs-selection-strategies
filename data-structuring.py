import os
import shutil
import tarfile
import argparse
import boto3
import tarfile
import pandas as pd
import random
from src.data.downstream.utils import keep_number

from tqdm import tqdm

splits = ['train', 'val', 'test']
random.seed(24)
try:
    os.mkdir('data')
except:
    print('directory exists')

data_dir = os.path.join(os.curdir,'data')

datasets = ['pretext', 'down-stream', 'none']

parser =  argparse.ArgumentParser(prog='Research data preperation')
parser.add_argument('-d','--dataset', type=str,default='none',metavar='NAME',
                    choices=datasets, help='type of research dataset to be prepared '
                   )
args = parser.parse_args()


if args.dataset == 'none':
    print(f'please specfy the the dataset to be prepared as follow: python {__file__} --dataset (dataset name)')


if args.dataset == 'pretext':
    pass


if args.dataset == 'down-stream':

    # data download


    # data extraction
    try: 
        assert 'down-stream' in os.listdir(data_dir)
    except AssertionError:
        print('down-stream dataset is being extracted')
        with tarfile.open(os.path.join(data_dir,'down-stream.tar.xz')) as archive:
            archive.extractall(path= data_dir)
            
    else:
        print('down-stream data is already extracted')

    try:
        os.remove(os.path.join(data_dir,'down-stream.tar.xz'))
    except FileNotFoundError:
        pass

    down_stream_dir = os.path.join(data_dir,'down-stream')
    raw_images_dir = os.path.join(down_stream_dir,'images')
    labels_path = os.path.join(down_stream_dir, 'diagnosis.csv')
    labels = pd.read_csv(labels_path)

    labels['Image_Path'] = labels['Image_Path'].apply(lambda x: keep_number(x))
    
    labels = labels.loc[labels.Diagnose != 'None']
    

    while True:
        down_stream_form = input('select downstream data form which can be (binary, multi-class, grading): ')
        down_stream_form = down_stream_form.lower()

        if down_stream_form in ['binary', 'multi-class', 'grading']:
            break
        else:
            continue
    
    if down_stream_form == 'binary':
        
        try:
            os.mkdir(os.path.join(down_stream_dir,'binary'))
        except:
            print('directory exisits')

        binary_dir = os.path.join(down_stream_dir,'binary')

        for split in splits:
            try:
                os.mkdir(os.path.join(binary_dir, split))
            except:
                print('directory exisits')



        for split in splits:
            try:
                os.mkdir(os.path.join(binary_dir,split,'Normal'))
                os.mkdir(os.path.join(binary_dir,split,'Abnormal'))
            except:
                print('directories exisit')

        try:
            assert len(os.listdir(os.path.join(binary_dir,'train','Normal'))) != 0
        except AssertionError: 
            print('binary training data preparation')
            for idx in tqdm(list(labels.Image_Path)):
    
                image = str(idx) + '.jpg'
                label = list(labels.loc[labels.Image_Path == idx]['Diagnose'])[0]
    
                if label == 'Normal':
                    shutil.copy(os.path.join(raw_images_dir,image),
                                os.path.join(binary_dir,'train','Normal')
                                )
                else:
                    shutil.copy(os.path.join(raw_images_dir,image),
                                os.path.join(binary_dir,'train','Abnormal')
                                )

        else:
            print('binary training data is already prepared')
        
        VAL_RATIO = 0.2

        val_normal_n = int(len(os.listdir(os.path.join(binary_dir,'train','Normal'))) * VAL_RATIO) 
        val_abnormal_n = int(len(os.listdir(os.path.join(binary_dir,'train','Abnormal'))) * VAL_RATIO)
    
        val_normal_images = random.sample(os.listdir(os.path.join(binary_dir,'train','Normal')),k=val_normal_n)
        val_abnormal_images = random.sample(os.listdir(os.path.join(binary_dir,'train','Abnormal')),k=val_abnormal_n,)
        

        try:
            assert len(os.listdir(os.path.join(binary_dir,'val','Normal'))) != 0
        except AssertionError: 
            print('preparaing binary validation data')
            for image in tqdm(val_normal_images):
                shutil.move(os.path.join(binary_dir,'train','Normal',image), 
                            os.path.join(binary_dir,'val','Normal',image)
                            )

            for image in tqdm(val_abnormal_images):
                shutil.move(os.path.join(binary_dir,'train','Abnormal',image), 
                            os.path.join(binary_dir,'val','Abnormal',image)
                            )
        else:
            print('binary validation data is already prepared')
        
        TEST_RATIO = 0.05 # out of all images

        TEST_RATIO = (TEST_RATIO/(VAL_RATIO))
        test_normal_n = int(len(os.listdir(os.path.join(binary_dir,'val','Normal'))) * TEST_RATIO) 
        test_abnormal_n = int(len(os.listdir(os.path.join(binary_dir,'val','Abnormal'))) * TEST_RATIO)
    
        test_normal_images = random.sample(os.listdir(os.path.join(binary_dir,'val','Normal')),k=test_normal_n)
        test_abnormal_images = random.sample(os.listdir(os.path.join(binary_dir,'val','Abnormal')),k=test_abnormal_n,)


        try:
            assert len(os.listdir(os.path.join(binary_dir,'test','Normal'))) != 0
        except AssertionError: 
            print('preparaing binary testing data')
            for image in tqdm(test_normal_images):
                shutil.move(os.path.join(binary_dir,'val','Normal',image), 
                            os.path.join(binary_dir,'test','Normal',image)
                            )

            for image in tqdm(test_abnormal_images):
                shutil.move(os.path.join(binary_dir,'val','Abnormal',image), 
                            os.path.join(binary_dir,'test','Abnormal',image)
                            )
        else:
            print('binary test data is already prepared')



    if down_stream_form == 'multi-class':
        
        try:
            os.mkdir(os.path.join(down_stream_dir,'multi-class'))
        except:
            print('directory exisits')

        multi_class_dir = os.path.join(down_stream_dir,'multi-class')

        for split in splits:
            try:
                os.mkdir(os.path.join(multi_class_dir, split))
            except:
                print('directory exisits')



        for split in splits:
            try:
                os.mkdir(os.path.join(multi_class_dir,split,'Normal'))
                os.mkdir(os.path.join(multi_class_dir,split,'MRO'))
                os.mkdir(os.path.join(multi_class_dir,split,'VMT'))
                os.mkdir(os.path.join(multi_class_dir,split,'CNV'))
                os.mkdir(os.path.join(multi_class_dir,split,'GA'))
                os.mkdir(os.path.join(multi_class_dir,split,'CSR'))
                os.mkdir(os.path.join(multi_class_dir,split,'PMH'))
                os.mkdir(os.path.join(multi_class_dir,split,'FMH'))
            except:
                print('directories exisit')

        
        try:
            assert len(os.listdir(os.path.join(multi_class_dir,'train','Normal'))) != 0
        except AssertionError: 
            print('multi-class training data preparation')
            for idx in tqdm(list(labels.Image_Path)):
    
                image = str(idx) + '.jpg'
                label = list(labels.loc[labels.Image_Path == idx]['Diagnose'])[0]

                if label == 'Normal':
                    shutil.copy(os.path.join(raw_images_dir,image),os.path.join(multi_class_dir,'train','Normal'))

                elif label == 'CSR':
                    shutil.copy(os.path.join(raw_images_dir,image),os.path.join(multi_class_dir,'train','CSR'))
            
                elif label == 'MRO':
                    shutil.copy(os.path.join(raw_images_dir,image),os.path.join(multi_class_dir,'train','MRO'))
            
                elif label == 'Geographic atrophy':
                    shutil.copy(os.path.join(raw_images_dir,image),os.path.join(multi_class_dir,'train','GA'))
            
                elif label == 'CNV ( Choroidal neovascularization )':
                    shutil.copy(os.path.join(raw_images_dir,image),os.path.join(multi_class_dir,'train','CNV'))
            
                elif label == 'Full Macular Hole ( Full Thickness )':
                    shutil.copy(os.path.join(raw_images_dir,image),os.path.join(multi_class_dir,'train','FMH'))
            
                elif label == 'Partial Macular Hole':
                    shutil.copy(os.path.join(raw_images_dir,image),os.path.join(multi_class_dir,'train','PMH'))
            
                elif label == 'VMT':
                    shutil.copy(os.path.join(raw_images_dir,image),os.path.join(multi_class_dir,'train','VMT'))

        else:
            print('multi-class training data is already prepared')

        VAL_RATIO = 0.2

        val_normal_n = int(len(os.listdir(os.path.join(multi_class_dir,'train','Normal'))) * VAL_RATIO) 
        val_CSR_n = int(len(os.listdir(os.path.join(multi_class_dir,'train','CSR'))) * VAL_RATIO)
        val_MRO_n = int(len(os.listdir(os.path.join(multi_class_dir,'train','MRO'))) * VAL_RATIO)
        val_GA_n = int(len(os.listdir(os.path.join(multi_class_dir,'train','GA'))) * VAL_RATIO)
        val_CNV_n = int(len(os.listdir(os.path.join(multi_class_dir,'train','CNV'))) * VAL_RATIO)
        val_FMH_n = int(len(os.listdir(os.path.join(multi_class_dir,'train','FMH'))) * VAL_RATIO)
        val_PMH_n = int(len(os.listdir(os.path.join(multi_class_dir,'train','PMH'))) * VAL_RATIO)
        val_VMT_n = int(len(os.listdir(os.path.join(multi_class_dir,'train','VMT'))) * VAL_RATIO)
    
        val_normal_images = random.sample(os.listdir(os.path.join(multi_class_dir,'train','Normal')),k=val_normal_n)
        val_CSR_images = random.sample(os.listdir(os.path.join(multi_class_dir,'train','CSR')),k=val_CSR_n)
        val_MRO_images = random.sample(os.listdir(os.path.join(multi_class_dir,'train','MRO')),k=val_MRO_n)
        val_GA_images = random.sample(os.listdir(os.path.join(multi_class_dir,'train','GA')),k=val_GA_n)
        val_CNV_images = random.sample(os.listdir(os.path.join(multi_class_dir,'train','CNV')),k=val_CNV_n)
        val_FMH_images = random.sample(os.listdir(os.path.join(multi_class_dir,'train','FMH')),k=val_FMH_n)
        val_PMH_images = random.sample(os.listdir(os.path.join(multi_class_dir,'train','PMH')),k=val_PMH_n)
        val_VMT_images = random.sample(os.listdir(os.path.join(multi_class_dir,'train','VMT')),k=val_VMT_n)
        

        try:
            assert len(os.listdir(os.path.join(multi_class_dir,'val','Normal'))) != 0
        except AssertionError: 
            print('preparaing multi-class validation data')
            for image in tqdm(val_normal_images):
                shutil.move(os.path.join(multi_class_dir,'train','Normal',image), 
                            os.path.join(multi_class_dir,'val','Normal',image)
                            )

            for image in tqdm(val_CSR_images):
                shutil.move(os.path.join(multi_class_dir,'train','CSR',image), 
                            os.path.join(multi_class_dir,'val','CSR',image)
                            )

            for image in tqdm(val_MRO_images):
                shutil.move(os.path.join(multi_class_dir,'train','MRO',image), 
                            os.path.join(multi_class_dir,'val','MRO',image)
                            )

            for image in tqdm(val_GA_images):
                shutil.move(os.path.join(multi_class_dir,'train','GA',image), 
                            os.path.join(multi_class_dir,'val','GA',image)
                            )

            for image in tqdm(val_CNV_images):
                shutil.move(os.path.join(multi_class_dir,'train','CNV',image), 
                            os.path.join(multi_class_dir,'val','CNV',image)
                            )

            for image in tqdm(val_FMH_images):
                shutil.move(os.path.join(multi_class_dir,'train','FMH',image), 
                            os.path.join(multi_class_dir,'val','FMH',image)
                            )

            for image in tqdm(val_PMH_images):
                shutil.move(os.path.join(multi_class_dir,'train','PMH',image), 
                            os.path.join(multi_class_dir,'val','PMH',image)
                            )

            for image in tqdm(val_VMT_images):
                shutil.move(os.path.join(multi_class_dir,'train','VMT',image), 
                            os.path.join(multi_class_dir,'val','VMT',image)
                            )

        else:
            print('multi-class validation data is already prepared')        

        
        TEST_RATIO = 0.05 # out of all images
        TEST_RATIO = (TEST_RATIO/(VAL_RATIO))

        test_normal_n = int(len(os.listdir(os.path.join(multi_class_dir,'val','Normal'))) * TEST_RATIO) 
        test_CSR_n = int(len(os.listdir(os.path.join(multi_class_dir,'val','CSR'))) * TEST_RATIO)
        test_MRO_n = int(len(os.listdir(os.path.join(multi_class_dir,'val','MRO'))) * TEST_RATIO)
        test_GA_n = int(len(os.listdir(os.path.join(multi_class_dir,'val','GA'))) * TEST_RATIO)
        test_CNV_n = int(len(os.listdir(os.path.join(multi_class_dir,'val','CNV'))) * TEST_RATIO)
        test_FMH_n = int(len(os.listdir(os.path.join(multi_class_dir,'val','FMH'))) * TEST_RATIO)
        test_PMH_n = int(len(os.listdir(os.path.join(multi_class_dir,'val','PMH'))) * TEST_RATIO)
        test_VMT_n = int(len(os.listdir(os.path.join(multi_class_dir,'val','VMT'))) * TEST_RATIO)
    
        test_normal_images = random.sample(os.listdir(os.path.join(multi_class_dir,'val','Normal')),k=test_normal_n)
        test_CSR_images = random.sample(os.listdir(os.path.join(multi_class_dir,'val','CSR')),k=test_CSR_n,)
        test_MRO_images = random.sample(os.listdir(os.path.join(multi_class_dir,'val','MRO')),k=test_MRO_n,)
        test_GA_images = random.sample(os.listdir(os.path.join(multi_class_dir,'val','GA')),k=test_GA_n,)
        test_CNV_images = random.sample(os.listdir(os.path.join(multi_class_dir,'val','CNV')),k=test_CNV_n,)
        test_FMH_images = random.sample(os.listdir(os.path.join(multi_class_dir,'val','FMH')),k=test_FMH_n,)
        test_PMH_images = random.sample(os.listdir(os.path.join(multi_class_dir,'val','PMH')),k=test_PMH_n,)
        test_VMT_images = random.sample(os.listdir(os.path.join(multi_class_dir,'val','VMT')),k=test_VMT_n,)


        try:
            assert len(os.listdir(os.path.join(multi_class_dir,'test','Normal'))) != 0
        except AssertionError: 
            print('preparaing multi-class testing data')
            for image in tqdm(test_normal_images):
                shutil.move(os.path.join(multi_class_dir,'val','Normal',image), 
                            os.path.join(multi_class_dir,'test','Normal',image)
                            )

            for image in tqdm(test_CSR_images):
                shutil.move(os.path.join(multi_class_dir,'val','CSR',image), 
                            os.path.join(multi_class_dir,'test','CSR',image)
                            )

            for image in tqdm(test_MRO_images):
                shutil.move(os.path.join(multi_class_dir,'val','MRO',image), 
                            os.path.join(multi_class_dir,'test','MRO',image)
                            )

            for image in tqdm(test_GA_images):
                shutil.move(os.path.join(multi_class_dir,'val','GA',image), 
                            os.path.join(multi_class_dir,'test','GA',image)
                            )

            for image in tqdm(test_CNV_images):
                shutil.move(os.path.join(multi_class_dir,'val','CNV',image), 
                            os.path.join(multi_class_dir,'test','CNV',image)
                            )

            for image in tqdm(test_FMH_images):
                shutil.move(os.path.join(multi_class_dir,'val','FMH',image), 
                            os.path.join(multi_class_dir,'test','FMH',image)
                            )

            for image in tqdm(test_PMH_images):
                shutil.move(os.path.join(multi_class_dir,'val','PMH',image), 
                            os.path.join(multi_class_dir,'test','PMH',image)
                            )

            for image in tqdm(test_VMT_images):
                shutil.move(os.path.join(multi_class_dir,'val','VMT',image), 
                            os.path.join(multi_class_dir,'test','VMT',image)
                            )
        else:
            print('multi-class test data is already prepared')