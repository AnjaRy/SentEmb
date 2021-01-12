from sentence_transformers import SentenceTransformer
from configparser import ConfigParser
import numpy as np

def read_config_file(file):
    config = ConfigParser()
    config.read(file)

    model_path = config['sentence_transformers']['model_path']

    return model_path

def sentence_transformers(input, outputfile):
    model_path = read_config_file('config.ini')
    model = SentenceTransformer(model_path)

    print('embedding in process')

    embeddings = model.encode(input)

    print('file writing in process')

    np.savetxt(outputfile, embeddings, delimiter=',')