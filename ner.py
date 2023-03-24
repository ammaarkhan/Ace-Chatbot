from pathlib import Path
import spacy

model = None
output_dir=Path("NerModels")

print("Loading from", output_dir)
nlp2 = spacy.load(output_dir)

doc = nlp2("cosc 120 is my fav subject")
print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
print(doc.ents[0].text)