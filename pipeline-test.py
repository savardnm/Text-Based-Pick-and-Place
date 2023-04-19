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

sentence = "move the large red wrench onto wooden paintbrush"

doc = nlp(sentence)

print(type(doc))

# Code to create a parse tree html file and view in browser (could help debug)
# html = displacy.render(doc, style='dep', options={'compact':True})
# with open('parse_tree.html', 'w', encoding='utf-8') as f:
#     f.write(html)
# displacy.serve(doc, style='dep', options={'compact': True}, port=8000)

# Build a tree of word dependencies
def build_tree(token):
    candidate = {'text': token.text, 'dependency': token.dep_, }

    # if token.n_lefts + token.n_rights > 0:
    candidate['children'] = [build_tree(child) for child in token.children]

    return candidate

# Find the root node (main verb) and construct a tree using it as the base
root = [token for token in doc if token.dep_ == 'ROOT'][0]
tree = build_tree(root)


def find_token(tree, deps):
    # Searches a tree for a given dep (word type)
    if tree['dependency'] in deps:
        return (True, tree)
    
    for child in tree['children']:
        found, sub_tree = find_token(child, deps)

        if found:
            return (True, sub_tree)
        
    return (False, None)
        
success, pick = find_token(tree, ["dobj"])
success2, place = find_token(tree, ["pobj"])

if not (success and success2):
    print("failure")

print(pick['text'], "-->",place['text'])

descriptor_dependencies = ['amod', 'compound']
def get_descriptors(token):
    descriptors = []
    for child in token['children']:
        if child['dependency'] in descriptor_dependencies:
            descriptors += [child['text']]
    return descriptors

pick_descriptors = get_descriptors(pick)
place_descriptors = get_descriptors(place)

print(pick['text'],": ", pick_descriptors)
print(place['text'],": ", place_descriptors)

# === Detection simulation ===

file_list = os.listdir('./img/tool_segments/')

candidate_list = [file for file in file_list if pick['text'] in file]

# === CLIP ===
# https://github.com/openai/CLIP

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

logit_list = []
for candidate in candidate_list:
    print(candidate)

    space_separated = [word + " " for word in pick_descriptors] + [pick['text']]
    sentence = ''.join(space_separated)
    image = preprocess(Image.open("./img/tool_segments/" + candidate)).unsqueeze(0).to(device)
    text = clip.tokenize([sentence]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        logits_per_image, logits_per_text = model(image, text)

        logit_list.append(logits_per_image)
        print(logit_list)

        # print(logits_per_image, logits_per_text)
        # probs = logits_per_image.softmax(dim=-1).cpu().numpy()

logit_tup = tuple(logit_list)

logits_accross = torch.cat(logit_tup,1)

print(logits_accross)

probs = logits_accross.softmax(dim=-1).cpu().numpy()

print("'" + sentence + "'" + " Likelihood:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
