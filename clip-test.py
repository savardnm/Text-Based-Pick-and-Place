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

files = os.listdir('./img/tool_segments/')

# === SPACY ===
# Uses Spacy Linguistic parser to create a tree of the parts of speech (https://spacy.io/usage/linguistic-features#dependency-parse)
nlp = spacy.load("en_core_web_sm")

sentence = "move the large red screwdrive into the blue bin on the right"

doc = nlp(sentence)

print(type(doc))

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

for obj in tree:
    print(obj, tree[obj])


def find_token(tree, deps):
    # Searches a tree for a given dep (word type)
    if tree['dependency'] in deps:
        print("found ", tree['dependency'], "in", deps, "of", tree)
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

print(pick['children'])

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



exit(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("./img/tool_segments/wrench-01.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["red", "blue", "grey"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
