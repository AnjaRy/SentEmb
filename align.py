# command: python3 align.py -a <kind_of_alignment> -o <outputfile>
# Please make sure to check the config file as all 6 filenames are taken from there

import numpy as np
import sys
import getopt
from configparser import ConfigParser

def read_config_file(file):
    config = ConfigParser()
    config.read(file)

    s1 = config['alignment']['sentences1']
    s2 = config['alignment']['sentences2']
    e1 = config['alignment']['embeddings1']
    e2 = config['alignment']['embeddings2']
    c1 = config['alignment']['clustering1']
    c2 = config['alignment']['clustering2']
    n = config['alignment']['n_best']
    t = config['alignment']['threshold']

    return s1, s2, e1, e2, c1, c2, int(n), float(t)

def simple_alignment(s1, s2, e1, e2, n, t):
    pairs = []
    sentence_pairs = []
    with open(s1) as f:
        sentences1 = f.readlines()
    with open(s2) as f:
        sentences2 = f.readlines()
    embedding1 = np.loadtxt(e1, delimiter=',')
    embedding2 = np.loadtxt(e2, delimiter=',')

    for id1, emb1 in enumerate(embedding1):
        c = 0
        n_best = []

        for id2, emb2 in enumerate(embedding2):
            cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            cos_sim = float(cos_sim)

            if c < n:
                n_best.append((cos_sim, id1, id2))
            else:
                n_best.sort()
                if cos_sim > n_best[0][0]:
                    n_best[0] = (cos_sim, id1, id2)
            c+=1
        if n == 1:
            if n_best[0][0] > t:
                pairs = pairs + n_best
        else:
            pairs = pairs + n_best

    for cos_sim, id1, id2 in pairs:
        sentence_pairs.append((cos_sim, sentences1[id1].strip(), sentences2[id2].strip()))

    # sentence_pairs.sort(reverse=True)

    return sentence_pairs


def cluster_alignment(s1, s2, e1, e2, c1, c2, n, t):
    pairs = []
    sentence_pairs = []
    with open(s1) as f:
        sentences1 = f.readlines()
    with open(s2) as f:
        sentences2 = f.readlines()
    embedding1 = np.loadtxt(e1, delimiter=',')
    embedding2 = np.loadtxt(e2, delimiter=',')
    with open(c1) as f:
        cluster1 = f.readlines()
    with open(c2) as f:
        cluster2 = f.readlines()

    dict_cluster1 = {}
    dict_cluster2 = {}
    for id, cluster in enumerate(cluster1):
        cluster = cluster.strip()
        if cluster in dict_cluster1:
            dict_cluster1[cluster].append(id)
        else:
            dict_cluster1[cluster]=[]
            dict_cluster1[cluster].append(id)

    for id, cluster in enumerate(cluster2):
        cluster = cluster.strip()
        if cluster in dict_cluster2:
            dict_cluster2[cluster].append(id)
        else:
            dict_cluster2[cluster]=[]
            dict_cluster2[cluster].append(id)

    for id1, emb1 in enumerate(embedding1):
        c = 0
        n_best = []
        label = None
        for l, ids in dict_cluster1.items():
            if id1 in ids:
                label = l

        for id2 in dict_cluster2[label]:
            emb2 = embedding2[id2]

            cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

            if c < n:
                n_best.append((cos_sim, id1, id2))
            else:
                n_best.sort()
                if cos_sim > n_best[0][0]:
                    n_best[0] = (cos_sim, id1, id2)
            c+=1

        if n==1:
            if n_best[0][0] > t:
                pairs = pairs + n_best

        else:
            pairs = pairs + n_best

    for cos_sim, id1, id2 in pairs:
        sentence_pairs.append((cos_sim, sentences1[id1].strip(), sentences2[id2].strip()))

    # sentence_pairs.sort(reverse=True)

    return sentence_pairs


def main(argv):
    outputfile = ''
    kind_of_alignment = ''

    try:
        opts, args = getopt.getopt(argv, "ha:o:")
    except getopt.GetoptError:
        print('Usage: align.py -a <kind_of_alignment> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('Usage: align.py -a <kind_of_alignment> -o <outputfile>')
            sys.exit()
        elif opt == '-a':
            if arg in ['simple', 'cluster']:
                kind_of_alignment = arg
            else:
                print('Alignment not known. Available options are: simple, cluster')
        elif opt == "-o":
            outputfile = arg


    if kind_of_alignment == 'cluster':
        s1, s2, e1, e2, c1, c2, n_best, threshold = read_config_file('config.ini')
        pairs = cluster_alignment(s1, s2, e1, e2, c1, c2, n_best, threshold)

    if kind_of_alignment == 'simple':
        s1, s2, e1, e2, c1, c2, n_best, threshold = read_config_file('config.ini')
        pairs = simple_alignment(s1, s2, e1, e2, n_best, threshold)

    with open(outputfile, 'w') as o:
        for similarity, sentence1, sentence2 in pairs:
            o.write(str(similarity) + '\t' + sentence1 + '\t' + sentence2 + '\n')


if __name__ == "__main__":
    main(sys.argv[1:])
