import csv
import os
import sys
import warnings

import jiwer
import librosa
import numpy as np
import pandas as pd
import torch
import transformers
from pystoi import stoi
from transformers import Wav2Vec2ForMaskedLM, Wav2Vec2Tokenizer

'''
Functions to compute the metrics for the 2 tasks of the L3DAS21 challenge.
- task1_metric returns the metric for task 1.
- location_sensitive_detection returns the metric for task 1.
Both functions require numpy matrices as input and can compute only 1 batch at time.
Please, have a look at the "evaluation_baseline_taskX.py" scripts for detailed examples
on the use of these functions.
'''

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

#TASK 1 METRICS
warnings.filterwarnings("ignore", category=FutureWarning)
transformers.logging.set_verbosity_error()
wer_tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h");
wer_model = Wav2Vec2ForMaskedLM.from_pretrained("facebook/wav2vec2-base-960h");

def wer(clean_speech, denoised_speech):
    """
    computes the word error rate(WER) score for 1 single data point
    """
    def _transcription(clean_speech, denoised_speech):

        # transcribe clean audio
        input_values = wer_tokenizer(clean_speech, return_tensors="pt").input_values;
        logits = wer_model(input_values).logits;
        predicted_ids = torch.argmax(logits, dim=-1);
        transcript_clean = wer_tokenizer.batch_decode(predicted_ids)[0];

        # transcribe
        input_values = wer_tokenizer(denoised_speech, return_tensors="pt").input_values;
        logits = wer_model(input_values).logits;
        predicted_ids = torch.argmax(logits, dim=-1);
        transcript_estimate = wer_tokenizer.batch_decode(predicted_ids)[0];

        return [transcript_clean, transcript_estimate]

    transcript = _transcription(clean_speech, denoised_speech);
    try:   #if no words are predicted
        wer_val = jiwer.wer(transcript[0], transcript[1])
    except ValueError:
        wer_val = None

    return wer_val


def task1_metric(clean_speech, denoised_speech, sr=16000):
    '''
    Compute evaluation metric for task 1 as (stoi+(1-word error rate)/2)
    This function computes such measure for 1 single datapoint
    '''
    WER = wer(clean_speech, denoised_speech)
    if WER is not None:  #if there is no speech in the segment
        STOI = stoi(clean_speech, denoised_speech, sr, extended=False)
        WER = np.clip(WER, 0., 1.)
        STOI = np.clip(STOI, 0., 1.)
        metric = (STOI + (1. - WER)) / 2.
    else:
        metric = None
        STOI = None
    return metric, WER, STOI

def compute_se_metrics(predicted_folder, truth_folder, fs=16000):
    '''
    Load all submitted sounds for task 1 and compute the average metric
    '''
    METRIC = []
    WER = []
    STOI = []
    predicted_list = [s for s in os.listdir(predicted_folder) if '.wav' in s]
    truth_list = [s for s in os.listdir(truth_folder) if '.wav' in s]
    n_sounds = len(predicted_list)
    for i in range(n_sounds):
        name = str(i) + '.wav'
        predicted_temp_path = os.path.join(predicted_folder, name)
        truth_temp_path = os.path.join(truth_folder, name)
        predicted = librosa.load(predicted_temp_path, sr=fs)
        truth = librosa.load(truth_temp_path, sr=fs)
        metric, wer, stoi = task1_metric(truth, predicted)
        METRIC.append(metric)
        WER.append(wer)
        STOI.append(stoi)

    average_metric = np.mean(METRIC)
    average_wer = np.mean(WER)
    average_stoi = np.mean(STOI)

    print ('*******************************')
    print ('Task 1 metric: ', average_metric)
    print ('Word error rate: ', average_wer)
    print ('Stoi: ', average_stoi)

    return average_metric


#TASK 2 METRICS
def location_sensitive_detection(pred, true, n_frames=100, spatial_threshold=2.,
                                 from_csv=False, verbose=False):
    '''
    Compute TP, FP, FN of a single data point using
    location sensitive detection
    '''
    TP = 0   #true positives
    FP = 0   #false positives
    FN = 0   #false negatives
    #read csv files into numpy matrices if required
    if from_csv:
        pred = pd.read_csv(pred, sep=',',header=None)
        true = pd.read_csv(true, sep=',',header=None)
        pred = pred.values
        true = true.values
    #build empty dict with a key for each time frame
    frames = {}
    for i in range(n_frames):
        frames[i] = {'p':[], 't':[]}
    #fill each time frame key with predicted and true entries for that frame
    for i in pred:
        frames[i[0]]['p'].append(i)
    for i in true:
        frames[i[0]]['t'].append(i)
    #iterate each time frame:
    for frame in range(n_frames):
        t = frames[frame]['t']  #all true events for frame i
        p = frames[frame]['p']  #all predicted events for frame i

        num_true_items = len(t)
        num_pred_items = len(p)
        matched = 0
        match_ids = []       #all pred ids that matched
        match_ids_t = []     #all truth ids that matched

        if num_true_items == 0:         #if there are PREDICTED but not TRUE events
            FP += num_pred_items        #all predicted are false positive
        elif num_pred_items == 0:       #if there are TRUE but not PREDICTED events
            FN += num_true_items        #all predicted are false negative
        elif num_true_items == 0 and num_pred_items == 0:
            pass
        else:
            for i_t in range(len(t)):           #iterate all true events
                match = False       #flag for matching events
                #count if in each true event there is or not a matching predicted event
                true_class = t[i_t][1]          #true class
                true_coord = t[i_t][-3:]        #true coordinates
                for i_p in range(len(p)):       #compare each true event with all predicted events
                    pred_class = p[i_p][1]      #predicted class
                    pred_coord = p[i_p][-3:]    #predicted coordinates
                    spat_error = np.linalg.norm(true_coord-pred_coord)  #cartesian distance between spatial coords
                    if true_class == pred_class and spat_error < spatial_threshold:  #if predicton is correct (same label + not exceeding spatial error threshold)
                        match_ids.append(i_p)   #append to pred matched ids
                        match_ids_t.append(i_t) #append to truth matched ids

            unique_ids = np.unique(match_ids)  #remove duplicates from matches ids lists
            unique_ids_t = np.unique(match_ids_t)
            matched = min(len(unique_ids), len(unique_ids_t))   #compute the number of actual matches without duplicates

            fn =  num_true_items - matched
            fp = num_pred_items - matched

            #add to counts
            TP += matched          #number of matches are directly true positives
            FN += fn
            FP += fp


    precision = TP / (TP + FP + sys.float_info.epsilon)
    recall = TP / (TP + FN + sys.float_info.epsilon)
    F_score = 2 * ((precision * recall) / (precision + recall + sys.float_info.epsilon))

    results = {'precision': precision,
               'recall': recall,
               'F score': F_score
               }


    if verbose:
        print ('true positives: ', TP)
        print ('false positives: ', FP)
        print ('false negatives: ', FN)
        print ('---------------------')


        print ('*******************************')
        print ('F score: ', F_score)
        print ('Precision: ', precision)
        print ('Recall: ', recall)
        print  ('TP: ' , TP)
        print  ('FP: ' , FP)
        print  ('FN: ' , FN)

    return TP, FP, FN, F_score

def compute_seld_metrics(predicted_folder, truth_folder, n_frames=100, spatial_threshold=0.3):
    '''
    compute F1 score from results folder of submitted results based on the
    location sensitive detection metric
    '''
    TP = 0
    FP = 0
    FN = 0
    predicted_list = [s for s in os.listdir(predicted_folder) if '.csv' in s]
    truth_list = [s for s in os.listdir(truth_folder) if '.csv' in s]
    n_files = len(predicted_list)
    #iterrate each submitted file
    for i in range(n_files):
        name = predicted_list[i]
        predicted_temp_path = os.path.join(predicted_folder, name)
        truth_temp_path = os.path.join(truth_folder, name)
        #compute tp,fp,fn for each file
        tp, fp, fn = location_sensitive_detection(predicted_temp_path,
                                                  truth_temp_path,
                                                  n_frames,
                                                  spatial_threshold)
        TP += tp
        FP += fp
        FN += fn

    #compute total F score
    precision = TP / (TP + FP + sys.float_info.epsilon)
    recall = TP / (TP + FN + sys.float_info.epsilon)

    print ('*******************************')
    F_score = (2 * precision * recall) / (precision + recall + sys.float_info.epsilon)
    print ('F score: ', F_score)
    print ('Precision: ', precision)
    print ('Recall: ', recall)

    return F_score


#gen_dummy_seld_results('./prova')
#compute_seld_metric('./prova/pred', './prova/truth')
