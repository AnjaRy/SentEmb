# add more parameters for other models

import configparser

config = configparser.ConfigParser()

config['infersent'] = {}
config['infersent']['model_path'] = 'InferSent/encoder/infersent2.pkl'
config['infersent']['w2v_path'] = 'InferSent/fastText/crawl-300d-2M.vec'
# all the model parameters
config['infersent_parameters'] = {}
config['infersent_parameters']['bsize'] = '64'
config['infersent_parameters']['word_emb_dim'] = '300'
config['infersent_parameters']['enc_lstm_dim'] = '2048'
config['infersent_parameters']['pool_type'] = 'max'
config['infersent_parameters']['dpout_model'] = '0'
config['infersent_parameters']['version'] = '2'

config['sent2vec'] = {}
config['sent2vec']['model_path'] = 'sent2vec/torontobooks_unigrams.bin'

config['sentence_transformers'] = {}
config['sentence_transformers']['model_path'] = 'distilbert-base-nli-mean-tokens'

config['universal_sentence_encoder'] = {}
config['universal_sentence_encoder']['model_path'] = 'https://tfhub.dev/google/universal-sentence-encoder/4'

config['clustering'] = {}
# If hierarchical, None lets the system decide, how many clusters are used
config['clustering']['n_clusters'] = '5'

# Used for hierarchical clustering
config['clustering_parameters'] = {}
config['clustering_parameters']['n_clusters'] = 'None'
config['clustering_parameters']['distance_threshold'] = '1.5'
config['clustering_parameters']['affinity'] = 'cosine'
config['clustering_parameters']['linkage'] = 'average'

# All file names for Alignment, standard filenames
config['alignment'] = {}
config['alignment']['sentences1'] = 'sentences1.txt'
config['alignment']['sentences2'] = 'sentences2.txt'
config['alignment']['embeddings1'] = 'embeddings1.txt'
config['alignment']['embeddings2'] = 'embeddings2.txt'
config['alignment']['clustering1'] = 'clustering1.txt'
config['alignment']['clustering2'] = 'clustering2.txt'
config['alignment']['n_best'] = '1'
config['alignment']['threshold'] = '0.8'

with open('config.ini', 'w') as configfile:
    config.write(configfile)
