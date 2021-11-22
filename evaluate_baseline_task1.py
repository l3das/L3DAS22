import argparse
import os
import pickle
import sys

import numpy as np
import soundfile as sf
import torch
import torch.utils.data as utils
from tqdm import tqdm

from metrics import task1_metric
from models.FaSNet import FaSNet_origin, FaSNet_TAC
from models.MMUB import MIMO_UNet_Beamforming
from utility_functions import load_model, save_model

'''
Load pretrained model and compute the metrics for Task 1
of the L3DAS22 challenge. The metric is: (STOI+(1-WER))/2
Command line arguments define the model parameters, the dataset to use and
where to save the obtained results.
'''

def enhance_sound(predictors, model, device, length, overlap):
    '''
    Compute enhanced waveform using a trained model,
    applying a sliding crossfading window
    '''

    def pad(x, d):
        #zeropad to desired length
        pad = torch.zeros((x.shape[0], x.shape[1], d))
        pad[:,:,:x.shape[-1]] = x
        return pad

    def xfade(x1, x2, fade_samps, exp=1.):
        #simple linear/exponential crossfade and concatenation
        out = []
        fadein = np.arange(fade_samps) / fade_samps
        fadeout = np.arange(fade_samps, 0, -1) / fade_samps
        fade_in = fadein * exp
        fade_out = fadeout * exp
        x1[:,:,-fade_samps:] = x1[:,:,-fade_samps:] * fadeout
        x2[:,:,:fade_samps] = x2[:,:,:fade_samps] * fadein
        left = x1[:,:,:-fade_samps]
        center = x1[:,:,-fade_samps:] + x2[:,:,:fade_samps]
        end = x2[:,:,fade_samps:]
        return np.concatenate((left,center,end), axis=-1)

    overlap_len = int(length*overlap)  #in samples
    total_len = predictors.shape[-1]
    starts = np.arange(0,total_len, overlap_len)  #points to cut
    #iterate the sliding frames
    for i in range(len(starts)):
        start = starts[i]
        end = starts[i] + length
        if end < total_len:
            cut_x = predictors[:,:,start:end]
        else:
            #zeropad the last frame
            end = total_len
            cut_x = pad(predictors[:,:,start:end], length)

        #compute model's output
        cut_x = cut_x.to(device)
        predicted_x = model(cut_x, device)
        predicted_x = predicted_x.cpu().numpy()

        #reconstruct sound crossfading segments
        if i == 0:
            recon = predicted_x
        else:
            recon = xfade(recon, predicted_x, overlap_len)

    #undo final pad
    recon = recon[:,:,:total_len]

    return recon


def main(args):
    if args.use_cuda:
        device = 'cuda:' + str(args.gpu_id)
    else:
        device = 'cpu'

    print ('\nLoading dataset')
    #LOAD DATASET
    with open(args.predictors_path, 'rb') as f:
        predictors = pickle.load(f)
    with open(args.target_path, 'rb') as f:
        target = pickle.load(f)
    predictors = np.array(predictors)
    target = np.array(target)

    print ('\nShapes:')
    print ('Predictors: ', predictors.shape)
    print ('Target: ', target.shape)

    #convert to tensor
    predictors = torch.tensor(predictors).float()
    target = torch.tensor(target).float()
    #build dataset from tensors
    dataset_ = utils.TensorDataset(predictors, target)
    #build data loader from dataset
    dataloader = utils.DataLoader(dataset_, 1, shuffle=False, pin_memory=True)

    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    #LOAD MODEL
    if args.architecture == 'fasnet':
        model = FaSNet_origin(enc_dim=args.enc_dim, feature_dim=args.feature_dim,
                              hidden_dim=args.hidden_dim, layer=args.layer,
                              segment_size=args.segment_size, nspk=args.nspk,
                              win_len=args.win_len, context_len=args.context_len,
                              sr=args.sr)
    elif args.architecture == 'tac':
        model = FaSNet_TAC(enc_dim=args.enc_dim, feature_dim=args.feature_dim,
                              hidden_dim=args.hidden_dim, layer=args.layer,
                              segment_size=args.segment_size, nspk=args.nspk,
                              win_len=args.win_len, context_len=args.context_len,
                              sr=args.sr)
    elif args.architecture == 'MIMO_UNet_Beamforming':
        model = MIMO_UNet_Beamforming(fft_size=args.fft_size,
                                      hop_size=args.hop_size,
                                      input_channel=args.input_channel)
    if args.use_cuda:
        print("Moving model to gpu")
    model = model.to(device)

    #load checkpoint
    state = load_model(model, None, args.model_path, args.use_cuda)

    #COMPUTING METRICS
    print("COMPUTING TASK 1 METRICS")
    print ('M: Final Task 1 metric')
    print ('W: Word Error Rate')
    print ('S: Stoi')

    WER = 0.
    STOI = 0.
    METRIC = 0.
    count = 0
    model.eval()
    with tqdm(total=len(dataloader) // 1) as pbar, torch.no_grad():
        for example_num, (x, target) in enumerate(dataloader):

            outputs = enhance_sound(x, model, device, args.segment_length, args.segment_overlap)

            outputs = np.squeeze(outputs)
            target = np.squeeze(target)

            # outputs = outputs / np.max(outputs) * 0.9  #normalize prediction
            metric, wer, stoi = task1_metric(target, outputs)


            if metric is not None:

                METRIC += (1. / float(example_num + 1)) * (metric - METRIC)
                WER += (1. / float(example_num + 1)) * (wer - WER)
                STOI += (1. / float(example_num + 1)) * (stoi - STOI)

                #save sounds
                if args.save_sounds_freq is not None:
                    sounds_dir = os.path.join(args.results_path, 'sounds')
                    if not os.path.exists(sounds_dir):
                        os.makedirs(sounds_dir)

                    if count % args.save_sounds_freq == 0:
                        sf.write(os.path.join(sounds_dir, str(example_num)+'.wav'), outputs, 16000, 'PCM_16')
                        #print ('metric: ', metric, 'wer: ', wer, 'stoi: ', stoi)
            else:
                print ('No voice activity on this frame')
            pbar.set_description('M:' +  str(np.round(METRIC,decimals=3)) +
                   ', W:' + str(np.round(WER,decimals=3)) + ', S: ' + str(np.round(STOI,decimals=3)))
            pbar.update(1)
            count += 1

    #visualize and save results
    results = {'word error rate': WER,
               'stoi': STOI,
               'task 1 metric': METRIC
               }
    print ('*******************************')
    print ('RESULTS')
    for i in results:
        print (i, results[i])
    out_path = os.path.join(args.results_path, 'task1_metrics_dict.json')
    np.save(out_path, results)

    '''
    baseline results
    word error rate 0.46
    stoi 0.72
    task 1 metric 0.62
    '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #i/o parameters
    parser.add_argument('--model_path', type=str, default='RESULTS/Task1/checkpoint')
    parser.add_argument('--results_path', type=str, default='RESULTS/Task1/metrics')
    parser.add_argument('--save_sounds_freq', type=int, default=None)
    #dataset parameters
    parser.add_argument('--predictors_path', type=str, default='DATASETS/processed/task1_predictors_test_uncut.pkl')
    parser.add_argument('--target_path', type=str, default='DATASETS/processed/task1_target_test_uncut.pkl')
    parser.add_argument('--sr', type=int, default=16000)
    #reconstruction parameters
    parser.add_argument('--segment_length', type=int, default=76672)
    parser.add_argument('--segment_overlap', type=float, default=0.5)
    #model parameters
    parser.add_argument('--architecture', type=str, default='MIMO_UNet_Beamforming',
                        help="model name")
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--use_cuda', type=str, default='True')
    parser.add_argument('--enc_dim', type=int, default=64)
    parser.add_argument('--feature_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--layer', type=int, default=6)
    parser.add_argument('--segment_size', type=int, default=24)
    parser.add_argument('--nspk', type=int, default=1)
    parser.add_argument('--win_len', type=int, default=16)
    parser.add_argument('--context_len', type=int, default=16)
    parser.add_argument('--fft_size', type=int, default=512)
    parser.add_argument('--hop_size', type=int, default=128)
    parser.add_argument('--input_channel', type=int, default=4)

    args = parser.parse_args()
    #eval string args
    args.use_cuda = eval(args.use_cuda)

    main(args)
