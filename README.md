# L3DAS22 challenge supporting API
This repository supports the L3DAS22 IEEE ICASSP Grand Challenge and it is aimed at downloading the dataset, pre-processing the sound files and the metadata, training and evaluating the baseline models and validating the final results.
We provide easy-to-use instruction to produce the results included in our paper.
Moreover, we extensively commented our code for easy customization.

For further information please refer to the challenge [website](https://www.l3das.com/icassp2022/index.html) and to the challenge [documentation](https://www.l3das.com/assets/file/L3DAS22_ICASSP_documentation.pdf).

## Installation
Our code is based on Python 3.7.

To install all required dependencies run:
```bash
pip install -r requirements.txt
```

Follow [these](https://www.kaggle.com/docs/api) instructions to properly create and place the kaggle.json file.

## Dataset download
It is possible to download the entire dataset through the script **download_dataset.py**. This script downloads the data, extracts the archives, merges the 2 parts of task1 train360 files and prepares all folders for the preprocessing stage.

To download run this command:
```bash
python download_dataset.py --output_path ./DATASETS --unzip True
```

This script may take long, especially the unzipping stage.

Alternatively, it is possible to manually download the dataset from [Kaggle](https://www.kaggle.com/l3dasteam/l3das22).

The *train360* section of task 1 is split in 2 downloadable files. If you manually download the dataset, you should manually merge the content of the 2 folders. You can use the function download_dataset.merge_train360().
Example:

```python
import download_dataset

train360_path = "path_that_contains_both_train360_parts"
download_dataset.merge_train360(train360_path)
```

## Pre-processing
The file **preprocessing.py** provides automated routines that load the raw audio waveforms and their correspondent metadata, apply custom pre-processing functions and save numpy arrays (.pkl files) containing the separate predictors and target matrices.

Run these commands to obtain the matrices needed for our baseline models:
```bash
python preprocessing.py --task 1 --input_path DATASETS/Task1 --training_set train100 --num_mics 1
python preprocessing.py --task 2 --input_path DATASETS/Task2 --num_mics 1 --frame_len 100
```
The two tasks of the challenge require different pre-processing.

For **Task1** the function returns 2 numpy arrays contatining:
* Input multichannel audio spectrograms (3d noise+speech scenarios) - Shape: [n_data, n_channels, n_freq_bins, n_time_steps].
* Output monoaural audio waveforms (clean speech) - Shape [n_data, 1, n_samples].

For **Task2** the function returns 2 numpy arrays contatining:
* Input multichannel audio spectra (3d acoustic scenarios): Shape: [n_data, n_channels, n_fft_bins, n_time_frames].
* Output seld matrices containing the class ids of all sounds present in each 100-milliseconds frame alongside with their location coordinates - Shape: [n_data, n_frames, ((n_classes * n_class_overlaps) + (n_classes * n_class_overlaps * n_coordinates))], where n_class_overlaps is the maximum amount of possible simultaneous sounds of the same class (3) and n_coordinates refers to the spatial dimensions (3).


## Baseline models
We provide baseline models for both tasks, implemented in PyTorch. For task 1 we use a Beamforming U-Net and for task 2 an augmented variant of the SELDNet architecture. Both models are based on the single-microphone dataset configuration. Moreover, for task 1 we used only Train100 as training set.

To train our baseline models with the default parameters run:
```bash
python train_baseline_task1.py
python train_baseline_task2.py
```
These models will produce the baseline results mentioned in the paper.

GPU is strongly recommended to avoid very long training times.

Alternatively, it is possible to download our pre-trained models with these commands:
```bash
python download_baseline_models.py --task 1 --output_path RESULTS/Task1/pretrained
python download_baseline_models.py --task 2 --output_path RESULTS/Task2/pretrained
```
These models are also available for manual download [here](https://drive.google.com/drive/u/1/folders/1rTvlzoQM6ZxVTZe6PSJ_-yHx-uHa5z4z).

We also provide a Replicate [interactive demo](https://replicate.ai/l3das/l3das22_challenge) of both baseline models.


## Evaluaton metrics
Our evaluation metrics for both tasks are included in the **metrics.py** script.
The functions **task1_metric** and **location_sensitive_detection** compute the evaluation metrics for task 1 and task 2, respectively. The default arguments reflect the challenge requirements. Please refer to the above-linked challenge paper for additional information about the metrics and how to format the prediction and target vectors.

Example:
```python
import metrics

task1_metric = metrics.task1_metric(prediction_vector, target_vector)
_,_,_,task2_metric = metrics.location_sensitive_detection(prediction_vector, target_vector)
```

To compute the challenge metrics for our basiline models run:
```bash
python evaluate_baseline_task1.py
python evaluate_baseline_task2.py
```

In case you want to evaluate our pre-trained models, please add
`
--model_path path/to/model
`
to the above commands.

## Submission shape validation
The script **validate_submission.py** can be used to assess the validity of the submission files shape. Instructions about how to format the submission can be found in the L3das [website](https://www.l3das.com/icassp2022/submission.html)
Use these commands to validate your submissions:
```bash
python validate_submission.py --task 1 --submission_path path/to/task1_submission_folder --test_path path/to/task1_test_dataset_folder

python validate_submission.py --task 2 --submission_path path/to/task2_submission_folder --test_path path/to/task2_test_dataset_folder
```

For each task, this script asserts if:
* The number of submitted files is correct
* The naming of the submitted files is correct
* Only the files to be submitted are present in the folder
* The shape of each submission file is as expected

Once you have valid submission folders, please follow the instructions on the link above to proceed with the submission.
