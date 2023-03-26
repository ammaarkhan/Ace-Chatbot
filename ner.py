from pathlib import Path
import spacy

output_dir=Path("NerModels")

# print("Loading from", output_dir)
nlp2 = spacy.load(output_dir)

doc = nlp2("cosc 121 is computer science")
# print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
dict = {}
for ent in doc.ents:
    dict.update({ent.label_: ent.text})

direct = {"computer science": ["cosc 121", "cosc 240", "cosc 341"]}

# if dict['COR'] in ["cosc 121", "cosc 240", "cosc 341"]:

if dict['COR'] in direct[dict['MAJOR']]:
    print("lets gooo")
else: 
    print("not there loser")

# print(dict['COR'], dict['MAJOR'])