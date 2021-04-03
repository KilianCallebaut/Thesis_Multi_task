import json
import math
import os
import pathlib

from DataReaders.ExtractionMethod import MelSpectrogram, Mfcc, LogbankSummary

extract_options = {
    MelSpectrogram().name: MelSpectrogram(),
    Mfcc().name: Mfcc(),
    LogbankSummary().name: LogbankSummary()
}

dataset_options = [
    'asvspoof',
    'chen',
    'dcaseScene',
    'fsdkaggle',
    'ravdess',
    'speechcommands'
]

network_options = [
    'cnn',
    'dnn'
]


def write_config(name: str, params: dict):
    if 'extraction_method' in params.keys():
        params['extraction_method'] = params['extraction_method'].name
    with open('configs/{}.json'.format(name), 'w') as outfile:
        json.dump(params, outfile)


def read_config(name: str):
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    with open(os.path.join(dir_path,
                           'configs/{}.json'.format(name))) as outfile:
        data = json.load(outfile)
        if 'extraction_method' in data.keys():
            data['extraction_method'] = extract_options[data['extraction_method']]
        if 'split' in data.keys():
            data['test_size'] = read_config('test_size_split') if data.pop('split') else read_config('test_size_val')
            data['test_size'] = data['test_size']['test_size']
        if 'window' in data.keys():
            window_feat = read_config('preparation_params_general_window') if data.pop('window') else read_config(
                'preparation_params_general_no_window')
            data = {**data, **window_feat}
        return data


def write_preparation_params(name: str, split: bool, window: bool, dic_of_labels_limits: dict):
    name_comp = 'preparation_params_{}'.format(name)
    param = dict(
        split=split,
        window=window,
        dic_of_labels_limits=dic_of_labels_limits
    )
    write_config(name_comp, param)

def calculate_window_size(extraction_params):
    # Number of seconds for window size
    sec = 1
    winlen = extraction_params['winlen']
    winstep = extraction_params['winstep']
    win_size = 1 + math.ceil((sec-winlen)/winstep)
    win_hop = math.floor(win_size/4)
    write_config('preparation_params_general_window', dict(window_size=win_size, window_hop=win_hop))

def set_windowed(win: bool, network: str):
    for d in dataset_options:
        param = read_config('preparation_params_{}_{}'.format(d, network))
        write_preparation_params('{}_{}'.format(d, network), param['split'], win, param['dic_of_labels_limits'])

