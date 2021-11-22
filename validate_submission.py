import argparse
import os
import sys

import librosa
import numpy as np
import pandas as pd

'''
Check if the the submssion folders are valid: all files must have the
correct format, shape and naming.
'''

def validate_task1_submission(submission_folder, test_folder):
    '''
    Args:
    - submission_folder: folder containing the model's output for task 1 (non zipped).
    - test_folder: folder containing the released test data (non zipped).
    '''
    #this is just a draft

    #read folders
    contents_submitted = sorted(os.listdir(submission_folder))
    contents_test = sorted(os.listdir(test_folder))
    contents_submitted = [i for i in contents_submitted if 'DS_Store' not in i]
    contents_test = [i for i in contents_test if 'DS_Store' not in i]
    contents_test = [i for i in contents_test if '_B' not in i]
    contents_test = [i.split('_')[0]+'.wav' for i in contents_test]

    #check if non.npy files are present
    non_npy = [x for x in contents_submitted if x[-4:] != '.npy']  #non .npy files
    if len(non_npy) > 0:
        raise AssertionError ('Non-.npy files present. Please include only .npy files '
                              'in the submission folder.')

    #check total number of files
    num_files = len(contents_submitted)
    target_num_files = len(contents_test)
    if not num_files == target_num_files:
        raise AssertionError ('Wrong amount of files. Target:' + str(target_num_files) +
                             ', detected:' + str(len(contents_submitted)))

    #check files naming
    names_submitted = [i.split('.')[0] for i in contents_submitted]
    names_test = [i.split('.')[0] for i in contents_test]
    names_submitted.sort()
    names_test.sort()
    if not names_submitted == names_test:
        raise AssertionError ('Wrong file naming. Please name each output file '
                               'exactly as its input .wav file, but with .npy extension')

    #check shape file-by-file
    for i in contents_test:
        submitted_path = os.path.join(submission_folder, i.split('.')[0]+'.npy')
        test_path = os.path.join(test_folder, i.split('.')[0]+'_A.wav')
        s = np.load(submitted_path, allow_pickle=True)
        t, _ = librosa.load(test_path, 16000, mono=False)
        target_shape = t.shape[-1]
        if not s.shape[-1] == target_shape:
            raise AssertionError ('Wrong shape for: ' + str(i) + '. Target: ' + str(target_shape) +
                                 ', detected:' + str(s.shape))

    print ('The shape of your submission for Task 1 is valid!')



def validate_task2_submission(submission_folder, test_folder):
    '''
    Args:
    - submission_folder: folder containing the model's output for task 1 (non zipped).
    - test_folder: folder containing the released test data (non zipped).
    '''
    #this is just a draft

    #read folders
    contents_submitted = sorted(os.listdir(submission_folder))
    contents_test = sorted(os.listdir(test_folder))
    contents_submitted = [i for i in contents_submitted if 'DS_Store' not in i]
    contents_test = [i for i in contents_test if 'DS_Store' not in i]
    contents_test = [i for i in contents_test if '_B' not in i]
    contents_test = [i.split('_')[0]+'.wav' for i in contents_test]

    #check if non .csv files are present
    non_npy = [x for x in contents_submitted if x[-4:] != '.csv']  #non .csv files
    if len(non_npy) > 0:
        raise AssertionError ('Non-.csv files present. Please include only .csv files '
                              'in the submission folder.')

    #check total number of files
    num_files = len(contents_submitted)
    target_num_files = len(contents_test)
    if not num_files == target_num_files:
        raise AssertionError ('Wrong amount of files. Target:' + str(target_num_files) +
                             ', detected:' + str(len(contents_submitted)))

    #check files naming
    names_submitted = [i.split('.')[0] for i in contents_submitted]
    names_test = [i.split('.')[0] for i in contents_test]
    names_submitted.sort()
    names_test.sort()
    if not names_submitted == names_test:
        raise AssertionError ('Wrong file naming. Please name each output file '
                               'exactly as its input .wav file, but with .csv extension')

    #check shape file-by-file
    for i in contents_submitted:
        submitted_path = os.path.join(submission_folder, i)
        #s = np.genfromtxt(submitted_path,delimiter=',',names=True, dtype=None, encoding=None)
        s = pd.read_csv(submitted_path, delimiter=',',sep='')
        if not s.shape[-1] == 5:
            raise AssertionError ('Wrong shape for: ' + str(i) + '. Target: ' + str(5) +
                                 ', detected:' + str(s.shape))
        #check if each column contains the right data type
        for i in range(len(s)):
            line = s.iloc[0]
            frame = line[0]
            class_name = line[1]
            x = line[2]
            y = line[3]
            z = line[4]
            try:
                int(frame)
                float(frame)
            except:
                raise AssertionError ('The element 0 of a row should be an integer')
            try:
                str(class_name)
            except:
                raise AssertionError ('The element 1 of a row should be a string')
            try:
                str(x)
            except:
                raise AssertionError ('The element 2 of a row should be a float')
            try:
                str(y)
            except:
                raise AssertionError ('The element 3 of a row should be a float')
            try:
                str(z)
            except:
                raise AssertionError ('The element 4 of a row should be a float')

    print ('The shape of your submission for Task 2 is valid!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #i/o parameters
    parser.add_argument('--submission_path', type=str,
                        help='Path to folder containing your submission (specific to each task)')
    parser.add_argument('--test_path', type=str,
                        help='Path to test set folder (specific to each task)')
    parser.add_argument('--task', type=int,
                        help='Task number to validate')
    args = parser.parse_args()
    #dataset parameters
    if args.task == 1:
        validate_task1_submission(args.submission_path, args.test_path)
    elif args.task == 2:
        validate_task2_submission(args.submission_path, args.test_path)
