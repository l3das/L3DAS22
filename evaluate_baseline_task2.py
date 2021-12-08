import argparse
import os
import pickle
import sys

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.utils.data as utils
from tqdm import tqdm

from metrics import location_sensitive_detection
from models.SELDNet import Seldnet_augmented, Seldnet_vanilla
from utility_functions import gen_submission_list_task2, load_model, save_model

'''
Load pretrained model and compute the metrics for Task 2
of the L3DAS22 challenge. The metric is F score computed with the
location sensitive detection: https://ieeexplore.ieee.org/document/8937220.
Command line arguments define the model parameters, the dataset to use and
where to save the obtained results.
'''

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
    n_time_frames = predictors.shape[-1]
    if args.architecture == 'seldnet_vanilla':
        model = Seldnet_vanilla(time_dim=n_time_frames, freq_dim=args.freq_dim, input_channels=args.input_channels,
                    output_classes=args.output_classes, pool_size=args.pool_size,
                    pool_time=args.pool_time, rnn_size=args.rnn_size, n_rnn=args.n_rnn,
                    fc_size=args.fc_size, dropout_perc=args.dropout_perc,
                    n_cnn_filters=args.n_cnn_filters, class_overlaps=args.class_overlaps,
                    verbose=args.verbose)
    if args.architecture == 'seldnet_augmented':
        model = Seldnet_augmented(time_dim=n_time_frames, freq_dim=args.freq_dim, input_channels=args.input_channels,
                    output_classes=args.output_classes, pool_size=args.pool_size,
                    pool_time=args.pool_time, rnn_size=args.rnn_size, n_rnn=args.n_rnn,
                    fc_size=args.fc_size, dropout_perc=args.dropout_perc,
                    cnn_filters=args.cnn_filters, class_overlaps=args.class_overlaps,
                    verbose=args.verbose)

    if args.use_cuda:
        print("Moving model to gpu")
    model = model.to(device)

    #load checkpoint
    state = load_model(model, None, args.model_path, args.use_cuda)

    #COMPUTING METRICS
    print("COMPUTING TASK 2 METRICS")
    TP = 0
    FP = 0
    FN = 0
    count = 0
    model.eval()
    with tqdm(total=len(dataloader) // 1) as pbar, torch.no_grad():
        for example_num, (x, target) in enumerate(dataloader):
            x = x.to(device)
            sed, doa = model(x)
            sed = sed.cpu().numpy().squeeze()
            doa = doa.cpu().numpy().squeeze()
            target = target.numpy().squeeze()
            #in the target matrices sed and doa are joint
            sed_target = target[:,:args.output_classes*args.class_overlaps]
            doa_target = target[:,args.output_classes*args.class_overlaps:]

            prediction = gen_submission_list_task2(sed, doa,
                                                   max_overlaps=args.class_overlaps,
                                                   max_loc_value=args.max_loc_value)

            target = gen_submission_list_task2(sed_target, doa_target,
                                               max_overlaps=args.class_overlaps,
                                               max_loc_value=args.max_loc_value)



            tp, fp, fn, _ = location_sensitive_detection(prediction, target, args.num_frames,
                                                      args.spatial_threshold, False)

            TP += tp
            FP += fp
            FN += fn

            count += 1
            pbar.update(1)

    #compute total F score
    precision = TP / (TP + FP + sys.float_info.epsilon)
    recall = TP / (TP + FN + sys.float_info.epsilon)
    F_score = 2 * ((precision * recall) / (precision + recall + sys.float_info.epsilon))

    #visualize and save results
    results = {'precision': precision,
               'recall': recall,
               'F score': F_score
               }
    print ('*******************************')
    print ('RESULTS')
    print ('F score: ', F_score)
    print ('Precision: ', precision)
    print ('Recall: ', recall)
    print  ('TP: ' , TP)
    print  ('FP: ' , FP)
    print  ('FN: ' , FN)

    out_path = os.path.join(args.results_path, 'task2_metrics_dict.json')
    np.save(out_path, results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #i/o parameters
    parser.add_argument('--model_path', type=str, default='RESULTS/Task2/checkpoint')
    parser.add_argument('--results_path', type=str, default='RESULTS/Task2/metrics')
    #dataset parameters
    parser.add_argument('--predictors_path', type=str, default='DATASETS/processed/task2_predictors_test.pkl')
    parser.add_argument('--target_path', type=str, default='DATASETS/processed/task2_target_test.pkl')
    parser.add_argument('--sr', type=int, default=32000)
    #eval parameters
    parser.add_argument('--max_loc_value', type=float, default=2,
                         help='max value of target loc labels (to rescale model\'s output since the models has tanh in the output loc layer)')
    parser.add_argument('--num_frames', type=int, default=300,
                        help='total number of time frames in the predicted seld matrices. (300 for 30-seconds sounds with 100msecs frames)')
    parser.add_argument('--spatial_threshold', type=float, default=2,
                        help='max cartesian distance for considering a true positive')
    #model parameters
    #the following parameters produce a prediction for each 100-msecs frame
    parser.add_argument('--architecture', type=str, default='seldnet_augmented',
                        help="model's architecture, can be seldnet_vanilla or seldnet_augmented")
    parser.add_argument('--input_channels', type=int, default=4,
                        help="4/8 for 1/2 mics, multiply x2 if using also phase information")
    parser.add_argument('--class_overlaps', type=int, default=3,
                        help= 'max number of simultaneous sounds of the same class')
    parser.add_argument('--use_cuda', type=str, default='True')
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--time_dim', type=int, default=4800)
    parser.add_argument('--freq_dim', type=int, default=256)
    parser.add_argument('--output_classes', type=int, default=14)
    parser.add_argument('--pool_size', type=str, default='[[8,2],[8,2],[2,2],[1,1]]')
    parser.add_argument('--cnn_filters', type=str, default='[64,128,256,512]',
                        help= 'only for seldnet augmented')
    parser.add_argument('--pool_time', type=str, default='True')
    parser.add_argument('--rnn_size', type=int, default=256)
    parser.add_argument('--n_rnn', type=int, default=3)
    parser.add_argument('--fc_size', type=int, default=1024)
    parser.add_argument('--dropout_perc', type=float, default=0.3)
    parser.add_argument('--n_cnn_filters', type=float, default=64,
                        help= 'only for seldnet vanilla')
    parser.add_argument('--verbose', type=str, default='False')
    parser.add_argument('--sed_loss_weight', type=float, default=1.)
    parser.add_argument('--doa_loss_weight', type=float, default=5.)


    args = parser.parse_args()
    #eval string args
    args.use_cuda = eval(args.use_cuda)
    args.pool_size= eval(args.pool_size)
    args.cnn_filters = eval(args.cnn_filters)
    args.verbose = eval(args.verbose)

    main(args)
