import random
from collections import counter, defaultdict 
from pathlib import Path
BOOKS={"alice":"1661-0.txt", "sherlock":"1661-0.txt"}
USE_TRIGRAMS=True
def load_words(path):
    text=(Path(__file__).parent/path).read_text(encoding="utf-8").lower()
    text=text.replace('\u2019',"'").replace('\u2018',"'")
    chars=[ch if ch.isalpha() or ch=="'" else " " for ch in text]
    words=[]                                      
    for word in "".join(chars).split():
        word=word.strip("'")
        if word:
            words.append(word)
    return words