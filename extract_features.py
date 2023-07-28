import os
import numpy as np
import utils
import librosa
import pandas as pd
from sklearn import preprocessing
import config
#from gammatone.fftweight import fft_gtgram
import torch
from spafe.features.lfcc import lfcc
from spafe.utils.vis import show_spectrogram
from spafe.utils.preprocessing import SlidingWindow
from spafe.features.gfcc import erb_spectrogram
from spafe.features.gfcc import gfcc

# -----------------------------------------------------------------------
# Annotation extraction
# -----------------------------------------------------------------------
def load_labels(file_name, nframes):
    annotations = []
    for l in open(file_name):
        words = l.strip().split('\t')
        annotations.append([float(words[0]), float(words[1]), config.class_labels_soft[words[2]], float(words[3])])

    # Initialize label matrix
    label = np.zeros((nframes, len(config.class_labels_soft)))
    tmp_data = np.array(annotations)
    
    frame_start = np.floor(tmp_data[:, 0] * config.sample_rate / config.hop_size).astype(int)
    frame_end = np.ceil(tmp_data[:, 1] * config.sample_rate / config.hop_size).astype(int)
    se_class = tmp_data[:, 2].astype(int)
    for ind, val in enumerate(se_class):
        label[frame_start[ind]:frame_end[ind], val] = tmp_data[:, 3][ind]

    return label

# -----------------------------------------------------------------------
# Feature extraction
# -----------------------------------------------------------------------

def power_to_db(input):
    r"""Power to db, this function is the pytorch implementation of
    librosa.power_to_lb
    """
    ref_value = 1.0
    log_gamma = 10.0 * torch.log10(torch.clamp(input, min=1e-10, max=np.inf))
    log_gamma -= 10.0 * np.log10(np.maximum(1e-10, ref_value))
    top_db=80.0
    if top_db is not None:
        if top_db < 0:
            raise librosa.util.exceptions.ParameterError('top_db must be non-negative')
        log_gamma = torch.clamp(log_gamma, min=log_gamma.max().item() - top_db, max=np.inf)

    return log_gamma
    
def extract_mbe(_y, _sr, _nfft, _hop, _nb_mel, _fmin, _fmax):
    spec, _ = librosa.core.spectrum._spectrogram(y=_y, n_fft=_nfft, hop_length=_hop, power=1)
    mel_basis = librosa.filters.mel(sr=_sr, n_fft=_nfft, n_mels=_nb_mel, fmin=_fmin, fmax=_fmax)

    return np.dot(mel_basis, spec)

def extract_mfcc(_y, _sr, _nfft, _hop, _nb_mel, _fmin, _fmax):
    return librosa.feature.mfcc(y=_y, sr=_sr, n_fft=_nfft, hop_length=_hop, n_mfcc=_nb_mel, fmin =_fmin, fmax = _fmax,  htk=True)

def extract_gfcc(_y, _sr, _nfft, _hop, _nb_mel, _fmin, _fmax):
    window_time = _nfft/_sr
    hop_time = window_time/2
    gfccs  = gfcc(_y,
              fs=_sr,
              num_ceps = _nb_mel,
              pre_emph=1,
              pre_emph_coeff=0.97,
              window=SlidingWindow(window_time, hop_time, "hamming"),
              nfilts=_nb_mel,
              nfft=_nfft,
              low_freq=_fmin,
              high_freq=_fmin,
              normalize="mvn")
    return gfccs.T

def extract_erb(_y, _sr, _nfft, _hop, _nb_mel, _fmin, _fmax):
    window_time = _nfft/_sr
    hop_time = window_time/2
    gSpec, gfreqs = erb_spectrogram(_y,
                                fs=_sr,
                                pre_emph=0,
                                pre_emph_coeff=0.97,
                                window=SlidingWindow(window_time, hop_time, "hamming"),
                                nfilts=_nb_mel,
                                nfft=_nfft,
                                low_freq=_fmin,
                                high_freq=_fmax)
    return gSpec.T
    

def extract_cqt(_y, _sr, _nfft, _hop, _nb_mel, _fmin, _fmax):
    return np.abs(librosa.cqt(y=_y, sr=_sr, hop_length=_hop, n_bins=_nb_mel, fmin =_fmin))

def extract_vqt(_y, _sr, _nfft, _hop, _nb_mel, _fmin, _fmax):
    return np.abs(librosa.vqt(y=_y, sr=_sr, hop_length=_hop, n_bins=_nb_mel, fmin =_fmin))

def extract_lfcc(_y, _sr, _nfft, _hop, _nb_mel, _fmin, _fmax):
    window_time = _nfft/_sr
    hop_time = window_time/2
    lfccs  = lfcc(_y,
              fs=_sr,
              num_ceps = _nb_mel,
              pre_emph=1,
              pre_emph_coeff=0.97,
              window=SlidingWindow(window_time, hop_time, "hamming"),
              nfilts=_nb_mel,
              nfft=_nfft,
              low_freq=_fmin,
              high_freq=_fmax,
              normalize="mvn")
    return lfccs.T


def extract_data(dev_file, audio_path, annotation_path, feat_folder):
# Extract features for all audio files
    # User set parameters
    hop_len = config.hop_size
    fs = config.sample_rate
    
    nfft = int(hop_len*2)
    nb_mel_bands = 64
    is_mono = True
    fmin = 50
    fmax = 14000
    
    files = pd.read_csv(dev_file)['filename']
    for file in files:
        audio_name = file.split(os.path.sep)[-1]
        # MEL features
        y, sr = utils.load_audio(os.path.join(audio_path, file+'.wav'), mono=is_mono, fs=fs)
        mbe = extract_mfcc(y, sr, nfft, hop_len, nb_mel_bands, fmin, fmax).T
        tmp_feat_file = os.path.join(feat_folder, '{}.npz'.format(audio_name))
        np.savez(tmp_feat_file, mbe)

        nframes = mbe.shape[0]
               
        # Extraction SOFT Annotation
        annotation_file_soft = os.path.join(annotation_path, 'soft_labels_' + file + '.txt')
        annotations_soft = load_labels(annotation_file_soft, nframes)
        tmp_lab_file = os.path.join(feat_folder, '{}_soft.npz'.format(audio_name))
        np.savez(tmp_lab_file, annotations_soft)



# -----------------------------------------------------------------------
# Feature Normalization
# -----------------------------------------------------------------------
def fold_normalization(feat_folder, output_folder):
    for fold in np.arange(1, 6):

        name = str(fold)
        # Load data
        train_files = pd.read_csv('development_folds/fold{}_train.csv'.format(name))['filename'].tolist()
        val_files = pd.read_csv('development_folds/fold{}_val.csv'.format(name))['filename'].tolist()
        test_files = pd.read_csv('development_folds/fold{}_test.csv'.format(name))['filename'].tolist()

        X_train, X_val = None, None
        for file in train_files:
            audio_name = file.split('/')[-1]
            
            tmp_feat_file = os.path.join(feat_folder, '{}.npz'.format(audio_name))
            dmp = np.load(tmp_feat_file)
            tmp_mbe = dmp['arr_0']
            if X_train is None:
                X_train = tmp_mbe
            else:
                X_train = np.concatenate((X_train, tmp_mbe), 0)

        for file in val_files:
            audio_name = file.split('/')[-1]

            tmp_feat_file = os.path.join(feat_folder, '{}.npz'.format(audio_name))
            dmp = np.load(tmp_feat_file)
            tmp_mbe = dmp['arr_0']
            if X_val is None:
                X_val = tmp_mbe
            else:
                X_val = np.concatenate((X_val, tmp_mbe), 0)

        # Normalize the training data, and scale the testing data using the training data weights
        scaler = preprocessing.StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        normalized_feat_file = os.path.join(output_folder, 'merged_mbe_fold{}.npz'.format(fold))
        np.savez(normalized_feat_file, X_train, X_val)

        # For the test data save individually
        for file in test_files:
            audio_name = file.split('/')[-1]

            tmp_feat_file = os.path.join(feat_folder, '{}.npz'.format(audio_name))
            dmp = np.load(tmp_feat_file)
            tmp_mbe = dmp['arr_0']
            X_test = scaler.transform(tmp_mbe)
            normalized_test_file = os.path.join(output_folder, 'test_{}_fold{}.npz'.format(audio_name, fold))
            np.savez(normalized_test_file, X_test)
        
        print(f'\t{normalized_feat_file}')
        print(f'\ttrain {X_train.shape} val {X_val.shape}')



def merge_annotations_into_folds(feat_folder, labeltype, output_folder):
    category_dict = {'cafe_restaurant': 0, 'city_center':1, 'grocery_store':2, 'metro_station': 3, 'residential_area': 4}
    place_dict = {'cafe_restaurant': 1, 'city_center':0, 'grocery_store':1, 'metro_station': 1, 'residential_area': 0}
    for fold in np.arange(1, 6):
        name = str(fold)

        # Load data
        train_files = pd.read_csv('development_folds/fold{}_train.csv'.format(name))['filename'].tolist()
        val_files = pd.read_csv('development_folds/fold{}_val.csv'.format(name))['filename'].tolist()
        test_files = pd.read_csv('development_folds/fold{}_test.csv'.format(name))['filename'].tolist()

        Y_train,  Y_val = None, None
        for file in train_files:
            audio_name = file.split('/')[-1]
            category = file.split('/')[0]

            tmp_lab_file = os.path.join(feat_folder, '{}_{}.npz'.format(audio_name, labeltype))
            dmp = np.load(tmp_lab_file)
            label_mat = dmp['arr_0']
            nframes = label_mat.shape[0]
            category_label = np.zeros((nframes, 5))
            row_number = category_dict.get(category)
            category_label[:,row_number] = 1
            
            place_label = np.zeros((nframes, 2))
            row_number = place_dict.get(category)
            place_label[:,row_number] = 1
            
            if Y_train is None:
                Y_train = label_mat
                Y_train_c = category_label
                Y_train_i = place_label
            else:
                Y_train = np.concatenate((Y_train, label_mat), 0)
                Y_train_c = np.concatenate((Y_train_c, category_label), 0)
                Y_train_i = np.concatenate((Y_train_i, place_label), 0)

        for file in val_files:
            audio_name = file.split('/')[-1]
            category = file.split('/')[0]

            tmp_lab_file = os.path.join(feat_folder, '{}_{}.npz'.format(audio_name, labeltype))
            dmp = np.load(tmp_lab_file)
            label_mat = dmp['arr_0']
            nframes = label_mat.shape[0]
            category_label = np.zeros((nframes, 5))
            row_number = category_dict.get(category)
            category_label[:,row_number] = 1
            
            place_label = np.zeros((nframes, 2))
            row_number = place_dict.get(category)
            place_label[:,row_number] = 1
            
            if Y_val is None:
                Y_val = label_mat
                Y_val_c = category_label
                Y_val_i = place_label
            else:
                Y_val = np.concatenate((Y_val, label_mat), 0)
                Y_val_c = np.concatenate((Y_val_c, category_label), 0)
                Y_val_i = np.concatenate((Y_val_i, place_label), 0)

        lab_file = os.path.join(output_folder, 'merged_lab_{}_fold{}.npz'.format(labeltype, fold))
        np.savez(lab_file, Y_train, Y_val, Y_train_c, Y_val_c, Y_train_i, Y_val_i)
        
        for file in test_files:
            audio_name = file.split('/')[-1]
            category = file.split('/')[0]

            tmp_lab_file = os.path.join(feat_folder,'{}_{}.npz'.format(audio_name, labeltype))
            dmp = np.load(tmp_lab_file)
            label_mat = dmp['arr_0']
            nframes = label_mat.shape[0]
            category_label = np.zeros((nframes, 5))
            row_number = category_dict.get(category)
            category_label[:,row_number] = 1
            
            place_label = np.zeros((nframes, 2))
            row_number = place_dict.get(category)
            place_label[:,row_number] = 1
            
            lab_file = os.path.join(output_folder, 'lab_{}_{}_fold{}.npz'.format(labeltype, audio_name, fold ))
            np.savez(lab_file, label_mat, category_label, place_label)



        print(f'\t{lab_file}')
        print(f'\ttrain {Y_train.shape} val {Y_val.shape} ')



# ########################################
#              Main script starts here
# ########################################

if __name__ == '__main__':
    # path to all the data
    audio_path = '/notebooks/ntu/Task4b/data/audio'
    annotation_path = '/notebooks/ntu/Task4b/data/annotation'
    dev_file = 'development_split.csv'
    
    # Output
    feat_folder = '/notebooks/ntu/Task4b/dcase2023_task4b_baseline/features_mbe/'
    utils.create_folder(feat_folder)


    # Extract mel features for all the development data
    extract_data(dev_file, audio_path, annotation_path, feat_folder)

    # Normalize data into folds
    output_folder = 'development/features'
    utils.create_folder(output_folder)
    fold_normalization(feat_folder, output_folder)
    
    # Merge Soft Labels annotations
    output_folder = 'development/soft_labels'
    utils.create_folder(output_folder)
    merge_annotations_into_folds(feat_folder, 'soft', output_folder)
    
