# that generates output with semantic textual similarity
# and paraphrase detection for sentence from queries to evaluate


# command: python3 evaluate.py -m model -i inputfile -o outputfile

# what about paraphrase detection?

import sys
import getopt
import numpy as np
from sentence_transformers_model import sentence_transformers
from sent2vec_model import sent_2_vec
from infer_sent_model import infer_sent
from universal_sentence_encoder_model import universal_sentence_encoder


def prepare_sentences(input_file):
    sentences =[]
    with open(input_file) as i:
        data = i.readlines()
    for line in data:
        line = line.strip()
        line = line.split('\t')
        sentences = sentences + line

    return sentences


def embed_sentences(model, input, outputfile):
    if model == 'infersent':
        infer_sent(input, outputfile)
    elif model == 'sentence_transformers':
        sentence_transformers(input, outputfile)
    elif model == 'sent2vec':
        sent_2_vec(input, outputfile)
    elif model == 'universal_sentence_encoder':
        universal_sentence_encoder(input, outputfile)


def calculate_distances(inputfile):
    embedding = np.loadtxt(inputfile, delimiter=',')
    distances = []

    for i in range(0, len(embedding)-1, 2):
        sentence1 = embedding[i]
        sentence2 = embedding[i+1]

        cos_sim = np.dot(sentence1, sentence2) / (np.linalg.norm(sentence1) * np.linalg.norm(sentence2))
        distances.append((float(cos_sim), i, i+1))
    return distances


def main(argv):
    inputfile = ''
    model = ''
    tmp_outputfile = 'tmp_output.txt'
    outputfile = ''
    paraphrase_detection = False
    top_k = 100

    try:
        opts, args = getopt.getopt(argv, "hm:i:o:k:")
    except getopt.GetoptError:
        print('Usage: evaluate.py -m <model to embed> -i <inputfile> -o <outputfile> -k <show only top_k pairs>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('Usage: evaluate.py -m <model to embed> -i <inputfile> -o <outputfile> -k <show only top_k pairs>')
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
        elif opt == '-k':
            paraphrase_detection = True
            top_k = int(arg)

    input = prepare_sentences(inputfile)

    embed_sentences(model, input, tmp_outputfile)

    distances = calculate_distances(tmp_outputfile)

    if paraphrase_detection:
        distances.sort(reverse=True)
        distances = distances[:top_k]
    else:
        pass

    # c=0
    with open(outputfile, 'w') as o:
         for distance, id1, id2 in distances:
            sentence1 = input[id1]
    #         c += 1
            sentence2 = input[id2]
    #         c += 1
    #
            o.write(str(distance) + '\t' + sentence1 + '\t' + sentence2 + '\n')

if __name__ == "__main__":
    main(sys.argv[1:])
