import sent2vec
from configparser import ConfigParser
import numpy as np

def read_config_file(file):
    config = ConfigParser()
    config.read(file)

    model_path = config['sent2vec']['model_path']

    return model_path

def sent_2_vec(input, outputfile):
    model_path = read_config_file('config.ini')

    model = sent2vec.Sent2vecModel()
    model.load_model(model_path)
    embeddings = model.embed_sentences(input)

    np.savetxt(outputfile, embeddings, delimiter=',')
