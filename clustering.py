# that generates output with clustered sentences by similarities (using k-mean clustering &
# hierarchical clustering) generated respectively for the original & simplify text

import numpy as np
import sys
from configparser import ConfigParser
import getopt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

# command: python3 clustering.py -c kind_of_clustering -i1 embedding_1 -i2 embedding_2


def read_config_file(file):
    """
    reads config file and gives back all relevant parameters
    :param file: config-file to read
    :return: strings of parameters taken from config-file
    """
    config = ConfigParser()
    config.read(file)

    n = config['clustering']['n_clusters']

    param_dict = {}
    for key in config['clustering_parameters']:
        value = config['clustering_parameters'][key]
        if value == 'None':
            value = None
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    value = str(value)
        param_dict[key] = value

    return n, param_dict


def k_mean_clustering(input1, input2, n):
    """
    Clusters embeddings via k_mean clustering
    :param input1: list of np.arrays of embedding1
    :param input2: list of np.arrays of embedding2
    :param n: int how many clusters to form
    :return: 2 lists of int, labels for each sentence
    """

    # Merge the two datasets
    input = np.concatenate((input1, input2))

    # code adapted from sentence-transformers
    # Perform kmean clustering
    clustering_model = KMeans(n_clusters=int(n))
    clustering_model.fit(input)
    cluster_assignment = clustering_model.labels_

    # Separate the two sets
    cluster1 = cluster_assignment[:len(input1)]
    cluster2 = cluster_assignment[len(input1):]

    return cluster1, cluster2


def hierarchical_clustering(input1, input2, param_dict):
    """
    Clusters embeddings via hierarchical clustering
    :param input1: list of np.arrays of embedding1
    :param input2: list of np.arrays of embedding2
    :param param_dict: dictionary woth all parameters for clustering
    :return: 2 lists of int, labels for each sentence
    """

    # merge inputs
    input = np.concatenate((input1, input2))

    # Code adapted from sentence-transformers
    # Normalize the embeddings to unit length
    input = input / np.linalg.norm(input, axis=1, keepdims=True)

    # Perform kmean clustering
    clustering_model = AgglomerativeClustering(**param_dict)
    clustering_model.fit(input)
    cluster_assignment = clustering_model.labels_

    # Separate the two sets
    cluster1 = cluster_assignment[:len(input1)]
    cluster2 = cluster_assignment[len(input1):]

    return cluster1, cluster2


def write_cluster_to_file(cluster, file_sentences, outputfile):
    """
    Writes the sentences ordered by cluster in a file
    :param cluster: list of labels for each sentence
    :param file_sentences: string of filename of sentence-file
    :param outputfile: string, name of outputfile
    """

    # Read sentences
    with open(file_sentences) as fs:
        sentences = fs.readlines()

    cluster_dict = {}

    # Form a lookup dict for all clusters
    for n, label in enumerate(cluster):
        if label in cluster_dict:
            cluster_dict[label].append(n)
        else:
            cluster_dict[label] = []
            cluster_dict[label].append(n)

    # For each cluster look up all Ids and write the sentences with that id in a paragraph in the file
    with open(outputfile, 'w') as o:
        for label in cluster_dict.keys():
            o.write('Cluster: ' + str(label) + '\n')
            for id in cluster_dict[label]:
                o.write(sentences[id])
            o.write('\n\n')


def main(argv):
    inputfile_1 = ''
    inputfile_2 = ''
    kind_of_clustering = ''

    # handles input of command line and checks if input is valid
    try:
        opts, args = getopt.getopt(argv, "hc:i:y:")
    except getopt.GetoptError:
        print('Usage: clustering.py -c <kind_of_clustering> -i <embedding_1> -y <embedding_2>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('Usage: clustering.py -c <kind_of_clustering> -i <embedding_1> -y <embedding_2>')
            sys.exit()
        elif opt == '-c':
            if arg in ['k_mean', 'hierarchical']:
                kind_of_clustering = arg
            else:
                print('Clustering not known. Available options are: k_mean, hierarchical')
        elif opt == "-i":
            inputfile_1 = arg
        elif opt == "-y":
            inputfile_2 = arg

    outputfile_1 = 'cluster_' + inputfile_1
    outputfile_2 = 'cluster_' + inputfile_2

    # Read embedding files
    embedding1 = np.loadtxt(inputfile_1, delimiter=',')
    embedding2 = np.loadtxt(inputfile_2, delimiter=',')

    n, param_dict = read_config_file('config.ini')

    # Call the right method of clustering
    if kind_of_clustering == 'k_mean':
        cluster1, cluster2 = k_mean_clustering(embedding1, embedding2, n)
    elif kind_of_clustering == 'hierarchical':
        cluster1, cluster2 = hierarchical_clustering(embedding1, embedding2, param_dict)

    # Write output in outputfile
    with open(outputfile_1, 'w') as o1:
        for label in cluster1:
            o1.write(str(label) + '\n')

    with open(outputfile_2, 'w') as o2:
        for label in cluster2:
            o2.write(str(label) + '\n')

    # write_cluster_to_file(cluster1, 'plain_mini.txt', 'sentence_clustered_plain.txt')
    # write_cluster_to_file(cluster2, 'original_mini.txt', 'sentence_clustered_original.txt')


if __name__ == "__main__":
    main(sys.argv[1:])