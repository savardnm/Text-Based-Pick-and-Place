import torch
import clip
from PIL import Image
from nltk.corpus import wordnet as wn
import nltk
import os
from nltk.chunk import RegexpParser
import spacy
from spacy import displacy
from time import time, sleep

# === SPACY ===
# Uses Spacy Linguistic parser to create a tree of the parts of speech (https://spacy.io/usage/linguistic-features#dependency-parse)
nlp = spacy.load("en_core_web_sm")

sentence = "Move the red screwdriver into the blue bin."

doc = nlp(sentence)

print(type(doc))

# Code to create a parse tree html file and view in browser (could help debug)
html = displacy.render(doc, style='dep', options={'compact':True})
with open('parse_tree.html', 'w', encoding='utf-8') as f:
    f.write(html)
displacy.serve(doc, style='dep', options={'compact': True}, port=8001)
