import spacy

nlp = spacy.load("en_core_web_sm")


def posParse(sentence):
    doc = nlp(sentence)
    doc_pos = [token.tag_ for token in doc]
    # doc_dep = [token.dep_ for token in doc]
    # doc_heads = [token.head.i for token in doc]
    return doc_pos

def edited(pair):
    pad = ' <PAD>'
    pair[1] = pair[1] + (len(pair[2].split()) - len(pair[1].split())) * (len(pair[2].split()) > len(pair[1].split())) * pad
    pair[2] = pair[2] + (len(pair[1].split()) - len(pair[2].split())) * (len(pair[1].split()) > len(pair[2].split())) * pad
    pair = [pair[0], pair[1], pair[2]]
    return pair



if __name__ == '__main__':
    print(posParse('There are only so many was we are allowed to live'))
    print(edited(['grow.txt', 'A man is a very fragile creature', 'A man is a creature']))
