#!/bin/bash

wget bitbucket.org/irshadbhat/wikihowtoimprove-corpus/raw/e76ebb974beb5ec859ebb9f5c78037b80c45e42c/wikiHow_revisions_corpus.txt.bz2
mkdir /tmp/misunderstanding
mv wikiHow_revisions_corpus.txt.bz2 /tmp/misunderstanding/
touch /tmp/misunderstanding/typo_filtered_revisions.txt
touch /tmp/misunderstanding/typo_revisions.txt
python filter_typos.py /tmp/misunderstanding/wikiHow_revisions_corpus.txt.bz2 > /tmp/misunderstanding/typo_filtered_revisions.txt 2> /tmp/misunderstanding/typo_revisions.txt
