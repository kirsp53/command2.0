import json

json_data = []
with open("dictionary_copy.json", encoding='utf-8', mode='r') as json_file:
    json_data = json.load(json_file)



def get_dict(dict_path):
    keys = []
    values = []
    for category in json_data:
        keys.append(category)
        #print("///////////", category)
        topics = []
        for trendingtopic in json_data[category]:
            topics.append(trendingtopic)
        values.append(topics)
    dict_0 = dict.fromkeys(keys)
    for i in range(len(keys)):
        dict_0[keys[i]] = values[i]
    return dict_0
print(get_dict("dictionary_copy.json"))