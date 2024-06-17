import re
import json



def read_json(path):
    print("Chargement 3")
    with open(path, "r") as output_file:
        liste = json.load(output_file)
    return liste

def remove_punctuation(text: str):
    punct = re.compile(r"[\.,;—:\?!’'«»“/\-]")
    cleaned_text = re.sub(punct, "", text)
    return cleaned_text

def write_json(path, json_object):
    with open(path, "w") as output_file:
        json.dump(json_object, output_file)