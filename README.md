# SentEmb
A command line tool for aligning with different models

## Setup

### Virtual Environment
As this project contains many dependencies, it is best run in a virtual environment. To set up an environment with some model already downloaded, use

```bash
bash setup.sh
```

or the requirements.txt.

Before creating the virtual environment please check the file and make sure that everything which is downloaded 
automatically is wanted and needed.
the entire virtual environment needs 8.3 GB

### Models
In the setup.sh file some models are automatically downloaded.
Other alternatives can be found at:

Sent2Vec: https://github.com/epfml/sent2vec#downloading-sent2vec-pre-trained-models

Sentence-Transformers: https://www.sbert.net/docs/pretrained_models.html

Universal Sentence-Encoder: https://tfhub.dev/google/collections/universal-sentence-encoder/1


Models for sent2vec have to be downloaded manually and are expected to be in the sent2vec folder

### CONFIG
In all programs most of the parameters for the models are taken from the config file.
Run 'get_default_config.py' to get a default config-file. 'Align.py' also takes the filenames from the config file, 
so make sure they are correctly listed in the config file. 


## Embedding
With the command
```bash
python3 sentemb.py -m <model> -i <inputfile> -o <outputfile>
```
Where the inputfile is a file with one sentence per line. The output is a file with the saved embeddings.

### Options
-m: possible models are: 
- infersent (https://github.com/facebookresearch/InferSent)
  
- sentence_transformers (https://github.com/UKPLab/sentence-transformers)
  
- sent2vec (https://github.com/epfml/sent2vec)
  
- universal_sentence_encoder (https://tfhub.dev/google/universal-sentence-encoder/4)



## Clustering
This is an optional preprocessing step to make the alignment more efficient. It generates for 2 files a common vector space and attributes each sentence a label.

It is run with the command:
```bash
python3 clustering.py -c <kind_of_clustering> -i <embedding_1> -y <embedding_2>'
```
The two inputfiles are 2 files with embeddings generated with 'sentemb.py'.
The outputfiles are 2 files with one label per line for each sentence in the inputfile.
With the function 'write_cluster_to_file' the clusters can be written to a file, where the sentences are sorted by cluster.


### Options
-c gives the options:

- k_mean

- hierarchical

both methods are adapted from sentence transformers (https://www.sbert.net/examples/applications/clustering/README.html)

-Threshold and n_cluster:
Via the config-file the user can decide, how many clusters are generated with n_clusters. If the program should decide how many clusters should be generated, 
n_cluster has to be set to 1 and a threshold can be given.


## Aligning
The alignment can be run via:
```bash
python3 align.py -a <kind_of_alignment> -o <outputfile>
```
The inputfiles for this program are handled via the config-file as there are 4, respectively 6 inputfiles
needed. Further parameters are also handled via the config-file.

The inputfiles contain 2 files with the sentences, 2 files with the embeddings, and if needed
2 files with the clustering. Make sure to get the order right in the files and to not switch them accidentally.

The outputfile is a file with a pair of aligned sentences per line and their cosine-similarity.
It is possible to sort the files for the highest similarities.

### Options
-a takes two options:

- simple: The distance is calculated for each sentence for every sentence.

- cluster: The distances are only calculated within the assigned cluster.

## Evaluation
With the command:
```bash
python3 evaluate.py -m <model to embed> -i <inputfile> -o <outputfile> -k <show only top_k pairs>
```
the cosine similarity for given pairs is calculated.
The inputfile is a file with a sentence pair per line, separated with a tab.
The outputfile contains the sentence-pairs and their cosine-similarity.
The model to use to embed the sentences can be chosen via '-m'. For available options see
MODELS.

With a given -k, the outputfile is sorted and contains only the k highest similarities.
