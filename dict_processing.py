with open ("dict_1.txt", encoding = 'utf-8',mode = "r") as dict_1:
    dict_1_words = [word.strip() for word in dict_1.readlines()]
with open ("dict_1_ext.txt", encoding = 'utf-8',mode = "r") as dict_1_ext:
    dict_1_ext_words = [word.strip() for word in dict_1_ext.readlines()]
print(len(dict_1_words))
print(len(set(dict_1_words)))
print(len(dict_1_ext_words))
print(len(set(dict_1_ext_words)))
print(set(dict_1_words))
print(set(dict_1_ext_words))
from nltk.corpus import wordnet as wn

def find_synonyms(word):
    synonyms = []
    for synset in wn.synsets(word):
        for lemma in synset.lemmas():
            synonyms.append(lemma.name())
    return synonyms

print(find_synonyms('happy'))

from sentence_transformers import SentenceTransformer
import numpy as np

# Загрузка предобученной модели
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
class labse:
    def __init__(self,model_path):
        self.model_path=model_path
        self.model = SentenceTransformer(self.model_path)

lb = labse('sentence-transformers/LaBSE')

# Функция для получения векторного представления слова
def get_word_vector(word):
    return model.encode([word])[0]
def labse_get_word_vector(word):
    return lb.model.encode([word])[0]
mydict_1 = dict.fromkeys(dict_1_words,[])
mydict_1_ext = dict.fromkeys(dict_1_ext_words,[])
import json

# Исходное слово
word = "happy"
word_vector = get_word_vector(word)

# Список слов для поиска слов с похожим значением
words_list = ["sad", "joyful", "unhappy", "content", "ecstatic"]

# Сравнение векторов для поиска слов с похожим значением
def find_similar_words(word_vector, words_list):
    similar_words = []
    for word in words_list:
        word_vector_candidate = get_word_vector(word)
        similarity = np.dot(word_vector, word_vector_candidate)
        #print(similarity)
        if similarity > 0.70: # Выберите порог сходства по вашему усмотрению
            similar_words.append(word)
    return similar_words
def labse_find_similar_words(word_vector, words_list):
    similar_words = []
    for word in words_list:
        word_vector_candidate = labse_get_word_vector(word)
        similarity = np.dot(word_vector, word_vector_candidate)
        #print(similarity)
        if similarity > 0.70: # Выберите порог сходства по вашему усмотрению
            similar_words.append(word)
    return similar_words
similar = []
#sim = find_similar_words(get_word_vector(key), mydict_1.keys())

#similar_words = find_similar_words(word_vector, words_list)
#print(f"Words similar to '{word}': {similar_words}")
remove_keys = []

for key in mydict_1.keys():
    sim = find_similar_words(get_word_vector(key), mydict_1_ext.keys())
    for s in sim:
        if key != s:
            if key not in similar:
               mydict_1[key] = sim
               similar.append(s)
               print(mydict_1[key],sim,key,s)
            else:
                remove_keys.append(key)
    with open('dictionary.json', encoding='utf-8', mode='w') as filehandler:
        json.dump({key: [item.lower() for item in value] if isinstance(value, list) else value.lower() for key, value in mydict_1.items()}, filehandler, ensure_ascii=False)

for key in remove_keys:
    mydict_1.pop(key,None)
for key in mydict_1.keys():
    if mydict_1[key] == []:
        mydict_1[key] = [key.lower()]
with open('dictionary_copy.json', encoding='utf-8', mode='w') as filehandler:
    json.dump(mydict_1, filehandler, ensure_ascii=False)

# print the topics learned by the model
