from sentence_transformers import SentenceTransformer
from configparser import ConfigParser
import numpy as np

def read_config_file(file):
    """
    reads config file and gives back all relevant parameters
    :param file: config-file to read
    :return: strings of parameters taken from config-file
    """

    config = ConfigParser()
    config.read(file)

    model_path = config['sentence_transformers']['model_path']

    return model_path

def sentence_transformers(input, outputfile):
    """
    Embedds sentences with sentence-transformers
    :param input: list of sentences
    :param outputfile: string, name of inputfile to write embeddings in
    """
    model_path = read_config_file('config.ini')
    model = SentenceTransformer(model_path)

    print('embedding in process')

    embeddings = model.encode(input)

    print('file writing in process')

    np.savetxt(outputfile, embeddings, delimiter=',')