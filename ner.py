from pathlib import Path
import spacy

output_dir=Path("NerModels")

# print("Loading from", output_dir)
nlp2 = spacy.load(output_dir)

doc = nlp2("cosc 111 is computer science")
# print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
user = {}
for ent in doc.ents:
    user.update({ent.label_: ent.text})

direct = {"computer science": ["cosc 121", "math 100", "math 101", "cosc 211", "cosc 221", "cosc 222", "math 221", "stat 230", "cosc 320", "cosc 304", "cosc 310" "cosc 341" "cosc 499", "phil 331"]}
optional = {"computer science": [["cosc 111", "cosc 123"], {"engl 109": ["2", "engl 112", "engl 113", "engl 114", "engl 150", "engl 151", "engl 153", "engl 154","engl 155", "engl 156", "engl 203", "corh 203", "corh 205", "apsc 176", "apsc 201"]}, ["phys 111", "phys 112"]]}
reply = ""

if user['COR'] in direct[user['MAJOR']]:
    reply = "Yes " + user['COR'].upper() + " is a requirement for " + user['MAJOR'].upper() + "." 
else:
    for type in optional[user['MAJOR']]:
        if isinstance(type, dict):
            l=[]
            [l.extend([k,v]) for k,v in type.items()]
            y = [l[0]]
            [y.extend(l[1])]
            if user['COR'] in y:
                reply = "Yes, take one of "+ y[0].upper() + " or " + y[1] + " of "  
                del y[0]
                del y[0]
                for i in y:
                    reply += i.upper() + ", "
        else:
            if user['COR'] in type:
                reply = "Yes, take one of "+ type[0].upper() 
                del type[0]
                for i in type:
                    reply += " or " + i.upper()  
        
if reply == "":
    reply = user['COR'] + " is not a requirement for " + user['MAJOR'] + " but might be used as an elective, speak with an Academic & Career Advisor for more clarity."
    
print(reply)