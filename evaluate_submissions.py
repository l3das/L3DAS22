import argparse
import os
import pickle
import sys

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

import utility_functions as uf
from metrics import location_sensitive_detection, task1_metric


def main(args):

    contents_submitted = sorted(os.listdir(args.submission_path))
    contents_labels = sorted(os.listdir(args.truth_path))
    contents_submitted = [i for i in contents_submitted if 'DS_Store' not in i]
    contents_labels = [i for i in contents_labels if 'DS_Store' not in i]

    #######TASK1#######
    if args.task == 1:
        sr_task1 = 16000
        contents_labels = [i for i in contents_labels if '.wav' in i]
        contents_submitted = [i for i in contents_submitted if '.npy' in i]

        #contents_labels = contents_labels[:2]
        #contents_submitted = contents_submitted[:2]

        print (contents_submitted, 'cazzo')

        WER = 0.
        STOI = 0.
        METRIC = 0.
        with tqdm(total=len(contents_labels) // 1) as pbar:
            for example_num, (s,l) in enumerate(zip(contents_submitted, contents_labels)):
                #load vectors
                #s,l = data
                s_path = os.path.join(args.submission_path, s)
                l_path = os.path.join(args.truth_path, l)
                #s, _ = librosa.load(s_path, sr_task1, mono=True)
                s = np.load(s_path, allow_pickle=True)
                l, _ = librosa.load(l_path, sr_task1, mono=True)

                #squeeze arrays if needed
                s = np.squeeze(s)
                l = np.squeeze(l)

                #compute metrics
                metric, wer, stoi = task1_metric(l, s)

                if metric is not None:
                    METRIC += (1. / float(example_num + 1)) * (metric - METRIC)
                    WER += (1. / float(example_num + 1)) * (wer - WER)
                    STOI += (1. / float(example_num + 1)) * (stoi - STOI)

                pbar.set_description('M:' +  str(np.round(METRIC,decimals=3)) +
                       ', W:' + str(np.round(WER,decimals=3)) + ', S: ' + str(np.round(STOI,decimals=3)))
                pbar.update(1)

        #print and save results
        base_dir = os.path.join(args.output_path, args.team_name)
        results_name = os.path.join(base_dir, 'results_task1.npy')
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        results_dict = {'name': args.team_name,
                        'task': 1,
                        'metric': METRIC,
                        'wer': WER,
                        'stoi': STOI}

        np.save(results_name, results_dict)
        print ('*******************************')
        print('RESULTS')
        print (results_dict)

    #######TASK2#######
    elif args.task == 2:
        sound_classes_dict_task2 = {'Chink_and_clink':0,
                                   'Computer_keyboard':1,
                                   'Cupboard_open_or_close':2,
                                   'Drawer_open_or_close':3,
                                   'Female_speech_and_woman_speaking':4,
                                   'Finger_snapping':5,
                                   'Keys_jangling':6,
                                   'Knock':7,
                                   'Laughter':8,
                                   'Male_speech_and_man_speaking':9,
                                   'Printer':10,
                                   'Scissors':11,
                                   'Telephone':12,
                                   'Writing':13}

        contents_labels = [i for i in contents_labels if '.csv' in i]
        contents_submitted = [i for i in contents_submitted if '.csv' in i]

        TP = 0
        FP = 0
        FN = 0
        with tqdm(total=len(contents_labels) // 1) as pbar:
            for example_num, (s,l) in enumerate(zip(contents_submitted, contents_labels)):
                #load
                s_path = os.path.join(args.submission_path, s)
                l_path = os.path.join(args.truth_path, l)

                #compute sed and doa
                '''
                s = uf.csv_to_matrix_task2(s_path, sound_classes_dict_task2,
                                               dur=60, step=100/1000., max_loc_value=2.,
                                               no_overlaps=False)  #eric func
                sed_s = s[:,:14*3]
                doa_s = s[:,14*3:]
                s = uf.gen_submission_list_task2(sed_s, doa_s, max_overlaps=3, max_loc_value=2.)
                '''

                #s =

                l = uf.csv_to_matrix_task2(l_path, sound_classes_dict_task2,
                                               dur=60, step=100/1000., max_loc_value=2.,
                                               no_overlaps=False)  #eric func
                sed_l = l[:,:14*3]
                doa_l = l[:,14*3:]
                l = uf.gen_submission_list_task2(sed_l, doa_l, max_overlaps=3, max_loc_value=2.)



                #compute tp, fp, fn per file
                tp, fp, fn, _ = location_sensitive_detection(s, l, 600, 2., False)


                TP += tp
                FP += fp
                FN += fn

                pbar.update(1)

        #compute total F score
        precision = TP / (TP + FP + sys.float_info.epsilon)
        recall = TP / (TP + FN + sys.float_info.epsilon)
        F_score = 2 * ((precision * recall) / (precision + recall + sys.float_info.epsilon))

        #print and save results
        base_dir = os.path.join(args.output_path, args.team_name)
        results_name = os.path.join(base_dir, 'results_task2.npy')
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        #visualize and save results
        results_dict = {'name': args.team_name,
                        'task': 2,
                        'metric': F_score,
                        'precision': precision,
                        'recall': recall}

        np.save(results_name, results_dict)
        print ('*******************************')
        print ('RESULTS')
        print (results_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #i/o parameters
    parser.add_argument('--task', type=int, default=1)
    parser.add_argument('--submission_path', type=str, default='submissions_evaluation/submissions/EPUSPL_Task1/task1')
    parser.add_argument('--truth_path', type=str, default='submissions_evaluation/labels/task1')
    parser.add_argument('--output_path', type=str, default='submissions_evaluation/results/EPUSPL')

    parser.add_argument('--team_name', type=str, default='EPUSPL')
    parser.add_argument('--use_cuda', type=str, default='True')

    args = parser.parse_args()
    #eval string args
    args.use_cuda = eval(args.use_cuda)

    main(args)
