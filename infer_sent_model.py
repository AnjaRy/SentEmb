import nltk
import torch
from InferSent.models import InferSent
nltk.download('punkt')
from configparser import ConfigParser
import numpy as np

def read_config_file(file):
    config = ConfigParser()
    config.read(file)

    model_path = config['infersent']['model_path']
    w2v_path = config['infersent']['w2v_path']

    param_dict = {}
    for key in config['infersent_parameters']:
        value = config['infersent_parameters'][key]
        try:
            value = int(value)
        except ValueError:
            value = str(value)
        param_dict[key] = value

    return model_path, w2v_path, param_dict

def infer_sent(input, outputfile):
    model_path, w2v_path, param_dict = read_config_file("config.ini")
    infersent = InferSent(param_dict)
    infersent.load_state_dict(torch.load(model_path))

    infersent.set_w2v_path(w2v_path)

    infersent.build_vocab(input, tokenize=True)
    embeddings = infersent.encode(input, tokenize=True)

    np.savetxt(outputfile, embeddings, delimiter=',')
