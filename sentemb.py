# command: python3 sentemb.py -m model -i input -o output

import sys
import getopt
from sentence_transformers_model import sentence_transformers
from sent2vec_model import sent_2_vec
from infer_sent_model import infer_sent
from universal_sentence_encoder_model import universal_sentence_encoder


def prepare_sentences(input_file):
    """
    Read an prepare sentences for embedding
    :param input_file: string, name of inputfile
    :return list of sentences
    """

    with open(input_file) as i:
        data = i.readlines()
    sentences = [line.strip() for line in data]
    return sentences


def embed_sentences(model, input, outputfile):
    """
    Calls requested embedding
    :param model: string, which model to call
    :param input: list of sentences
    :param outputfile: name of outputfile to write embedding in
    """

    if model == 'infersent':
        infer_sent(input, outputfile)
    elif model == 'sentence_transformers':
        sentence_transformers(input, outputfile)
    elif model == 'sent2vec':
        sent_2_vec(input, outputfile)
    elif model == 'universal_sentence_encoder':
        universal_sentence_encoder(input, outputfile)


def main(argv):
    inputfile = ''
    outputfile = ''
    model = ''

    # Handles input from command line
    try:
        opts, args = getopt.getopt(argv, "hm:i:o:")
    except getopt.GetoptError:
        print('Usage: sentemb.py -m <model> -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('Usage: sentemb.py -m <model> -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt == '-m':
            if arg in ['infersent', 'sentence_transformers', 'sent2vec', 'universal_sentence_encoder']:
                model = arg
            else:
                print('Model not known.  Available options are: infersent, sentence_transformers, sent2vec',
                      'universal_sentence_encoder')
        elif opt == "-i":
            inputfile = arg
        elif opt == "-o":
            outputfile = arg

    input = prepare_sentences(inputfile)

    embed_sentences(model, input, outputfile)


if __name__ == "__main__":
    main(sys.argv[1:])
