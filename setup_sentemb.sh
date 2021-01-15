# add all urls to different models
# add versions needed
# generates requirements.txt


#!/bin/bash

# Authors: Nicolas Spring, Marek Kostrzewa
# Adapted to sentemb: Anja Ryser

# creating a virtual environment
virtualenv sentemb_env
cd sentemb_env || exit 1
cd ..
source ./sentemb_env/bin/activate

pip3 install --upgrade cython
sudo apt install curl


# Install tensorflow_hub for sentence_transformer
pip3 install torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install tensorflow
pip3 install tensorflow_hub

# Install numpy and sklearn
pip3 install numpy
pip3 install sklearn

# Install InferSent
git clone https://github.com/facebookresearch/InferSent
cd InferSent || exit 1
pip3 install .

# Load models for InferSent

#mkdir GloVe
#if [ ! -f "GloVE/glove.840B.300d.zip" ]; then
#  echo 'download GloVe-Model'
#  curl -Lo GloVe/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
#fi
#unzip GloVe/glove.840B.300d.zip -d GloVe/

mkdir fastText
if [ ! -f "fastText/crawl-300d-2M.vec.zip" ]; then
  echo 'download Fasttext Model'
  curl -Lo fastText/crawl-300d-2M.vec.zip https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
fi
unzip fastText/crawl-300d-2M.vec.zip -d fastText/

mkdir encoder
if [ ! -f "encoder/infersent1.pkl" ]; then
  echo 'download infersent1 model'
  curl -Lo encoder/infersent1.pkl https://dl.fbaipublicfiles.com/infersent/infersent1.pkl
fi

if [ ! -f "encoder/infersent2.pkl" ]; then
  echo 'download Infersent 2 model'
  curl -Lo encoder/infersent2.pkl https://dl.fbaipublicfiles.com/infersent/infersent2.pkl
fi

cd ..

# Install sentence-transformers
git clone https://github.com/UKPLab/sentence-transformers
cd sentence-transformers || exit 1
pip3 install .
cd ..
mv sentence-transformers sentence_transformers
# possible models for sentence transformers: https://www.sbert.net/docs/pretrained_models.html

# Install sent2vec
git clone https://github.com/epfml/sent2vec
cd sent2vec || exit 1
pip3 install .
cd ..

# pretrained model for sent2vec: https://github.com/epfml/sent2vec#downloading-sent2vec-pre-trained-models
# have to be downloaded manually from Google Drive and saved in sent2vec folder

deactivate