import tensorflow_hub as hub
from configparser import ConfigParser
import numpy as np

def read_config_file(file):
    config = ConfigParser()
    config.read(file)

    model_path = config['universal_sentence_encoder']['model_path']

    return model_path

def universal_sentence_encoder(input, outputfile):
    model_path = read_config_file('config.ini')
    print('Model is loaded. This can take some time')
    embed = hub.load(model_path)
    print('Model loading complete')
    embeddings = embed(input)

    np.savetxt(outputfile, embeddings, delimiter=',')
