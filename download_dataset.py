import argparse
import os
import shutil
import subprocess
import sys
import zipfile

import pandas as pd
from tqdm import tqdm

'''
Download and extract the L3DAS22 dataset into a user-defined directory.
Command line arguments define which dataset partition to download and where to
save the unzipped folders. This script automatically merges the 2 parts of task1
train360 parts and prepares all folders for the preprocessing stage.
'''


def download_l3das22_dataset(output_path, unzip=True):
    dataset_name = "l3dasteam/l3das22"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    st = "kaggle datasets download " + dataset_name + " -p " + output_path + " --force"
    if unzip:
        st += " --unzip"
    dwn = subprocess.Popen(st, shell=True)
    dwn.communicate()

def merge_csv(path1, path2, output_path):
    "concatenate the csv files of train360 part 1 and part2  "
    df = pd.concat(map(pd.read_csv, [path1, path2]),
        ignore_index=True)
    print (df)
    df.to_csv(output_path, index=False)

def merge_train360(part1_path, part2_path):
    '''
    Merge the part1 and 2 extracted folders in a single one, as required by
    the preprocessing script.
    '''
    data1_p = os.path.join(part1_path, "data")
    data2_p = os.path.join(part2_path, "data")
    labels1_p = os.path.join(part1_path, "labels")
    labels2_p = os.path.join(part2_path, "labels")

    data1_c = os.listdir(data1_p)
    data1_c = list(filter(lambda x: "DS_Store" not in x, data1_c))
    data1_c =[os.path.join(data1_p, i) for i in data1_c]

    labels1_c = os.listdir(labels1_p)
    labels1_c = list(filter(lambda x: "DS_Store" not in x, labels1_c))
    labels1_c =[os.path.join(labels1_p, i) for i in labels1_c]

    print ("Merging task1 train360 sound data")
    with tqdm(total=len(data1_c)) as pbar:
        for source in data1_c:
            target = source.replace("L3DAS22_Task1_train360_1", "L3DAS22_Task1_train360_2")
            os.rename(source, target)
            pbar.update(1)

    print ("Merging task1 train360 labels data")
    with tqdm(total=len(data1_c)) as pbar:
        for source in labels1_c:
            target = source.replace("L3DAS22_Task1_train360_1", "L3DAS22_Task1_train360_2")
            os.rename(source, target)
            pbar.update(1)

    merged_name = part2_path.replace("_2", "")
    if not os.path.exists(merged_name):
        os.makedirs(merged_name)

    info_1_path = os.path.join(part1_path, "info.csv")
    info_2_path = os.path.join(part2_path, "info.csv")
    out_info_path = os.path.join(merged_name, "info.csv")

    print ("Merging task1 train360 info.csv")
    merge_csv(info_1_path, info_2_path, info_2_path)

    #os.rename(part2_path, merged_name)
    shutil.rmtree(part1_path)

    #print ("Merged!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_path', type=str, default="./DATASETS",
                        help='where to download the dataset')
    parser.add_argument('--unzip', type=str, default="True",
                        help='unzip the downloaded file')

    args = parser.parse_args()

    download_l3das22_dataset(args.output_path, eval(args.unzip))

    #merge train360 parts
    part1_path = os.path.join(args.output_path, "L3DAS22_Task1_train360_1","L3DAS22_Task1_train360_1")
    part2_path = os.path.join(args.output_path, "L3DAS22_Task1_train360_2","L3DAS22_Task1_train360_2")
    merge_train360(part1_path, part2_path)

    #create dir tree
    task1_path = os.path.join(args.output_path, "Task1")
    task2_path = os.path.join(args.output_path, "Task2")
    if not os.path.exists(task1_path):
        os.makedirs(task1_path)
    if not os.path.exists(task2_path):
        os.makedirs(task2_path)

    os.rename(os.path.join(args.output_path, "L3DAS22_Task1_train360_2","L3DAS22_Task1_train360_2"),
              os.path.join(task1_path, "L3DAS22_Task1_train360")
              )
    os.rename(os.path.join(args.output_path, "L3DAS22_Task1_train100","L3DAS22_Task1_train100"),
              os.path.join(task1_path, "L3DAS22_Task1_train100")
              )
    os.rename(os.path.join(args.output_path, "L3DAS22_Task1_dev","L3DAS22_Task1_dev"),
              os.path.join(task1_path, "L3DAS22_Task1_dev")
              )
    os.rename(os.path.join(args.output_path, "L3DAS22_Task2_train","L3DAS22_Task2_train"),
              os.path.join(task2_path, "L3DAS22_Task2_train")
              )
    os.rename(os.path.join(args.output_path, "L3DAS22_Task2_dev","L3DAS22_Task2_dev"),
              os.path.join(task2_path, "L3DAS22_Task2_dev")
              )
    f = os.listdir(args.output_path)
    f.remove("Task1")
    f.remove("Task2")
    f = [os.path.join(args.output_path, i) for i in f]
    for i in f:
        shutil.rmtree(i)

    print ("Download completed!")
