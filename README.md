# SentEmb
A command line tool for aligning with different models

## Setup

### Virtual Environment
As this project contains many dependencies, it is best run in a virtual environment. To set up an environment with some model already downloaded, use

COMMAND

Before using this file, make sure you want everything which is downloaded. Check if the models you need are selected. Note that this virtual environment need XXX space.

### Models
As a standard some pretrained models are downloaded. There are many other available under:

LINK
LINK
LINK
LINK

### CONFIG


## Embedding
With the command:
COMMAND

A text file is taken and the embeddings for these sentences are generated. The text file should contain one sentence per line.

Options:
OPTIONS

Also Config



## Clustering
This is an optional preprocessing step to make the alignment more efficient. It generates for 2 files a common vector space and attributes each sentence a cluster.

it is  used via:
COMMAND

Options:
OPTIONS

Can be written and seen in file

ALso COnfig


## Aligning
With the command:
COMMAND
ACHTUNG CONFIG

A file with alignments is generated with  the cosinus-similarity and a sentence pair per line. It is possible to sort the file by similarity.

Options:
OPTIONS

ALSO CONFIG



## Evaluation

With the command:
COMMAND

for a file with generated sentence pairs the cosinus-similarity for each pair is calculated. The Model used to embed the sentences can be chosen.

Options:
OPTIONS
NO CONFIG FILE

BUUUUUH, CREEPY STUFF
