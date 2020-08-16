import json
import re
import collections
import nltk
from bs4 import BeautifulSoup


def clean_str(text):
    # Clean the text
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"that's", "that is ", text)
    text = re.sub(r"there's", "there is ", text)
    text = re.sub(r"it's", "it is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"mightn't", "might not ", text)
    text = re.sub(r"mustn't", "must not ", text)
    text = re.sub(r"needn't", "need not", text)
    text = re.sub("shan't", "shall not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"'", " ", text)
    return text.strip()


def format2json(path, path_json):
    
    with open(path, 'r') as f:
        lines = f.readlines()
        f.close()
    
    text = []
    tags = []
    entity1 = []
    entity2 = []
    for i in range(0, len(lines), 4):
        line = lines[i].lower()
        line = line.strip().split('\t')
        soup = BeautifulSoup(line[1], 'html')
        e1_word = soup.find('e1').text
        e1 = e1_word.split(' ')
        e2_word = soup.find('e2').text
        e2 = e2_word.split(' ')

        if len(e1) != 1:
            e1 = '_'.join(e1)
            soup.find(text=e1_word).replaceWith(e1)
        if len(e2) != 1:
            e2 = '_'.join(e2)
            soup.find(text=e2_word).replaceWith(e2)
        line = soup.text
        line = clean_str(line)
        tokens = nltk.word_tokenize(line)
        tokens = tokens[1:len(tokens) - 1] # 去掉句子前后的引号
        line = ' '.join(tokens)
        text.append(line)
        if type(e1) == list:
            entity1.append(e1[0])
        else:
            entity1.append(e1)
        if type(e2) == list:
            entity2.append(e2[0])
        else:
            entity2.append(e2)
        # POS tag
        tag = nltk.pos_tag(tokens)
        tag = [item[1] for item in tag]
        tag = ' '.join(tag)
        tags.append(tag)

    #max length
    length = [len(sen.split(' ')) for sen in text]
    print("max_length: ", max(length))


    labels = []
    for i in range(1, len(lines), 4):
        line = lines[i]
        line = line.strip().split("(")[0]
        labels.append(line)
    labels_set = set(labels)
    print("The number of labels: ", len(labels_set))
    print("The number of every kind of label: ", collections.Counter(labels))
    
#     labels_dict = collections.OrderedDict()
#     j = 1
#     for item in labels_set:
#         if item == 'Other':
#             labels_dict[item] = 0
#         else:
#             labels_dict[item] = j
#             j += 1

#     labels_dict = sorted(labels_dict, key=labels_dict.__getitem__)
#     print(labels_dict)
#     path_relation = 'data_json/relation2id.json'
#     with open(path_relation, 'w') as f:
#         json.dump(labels_dict, f, indent=4)
#         f.close()

    file = []
    for i in range(len(labels)):
        dict = collections.OrderedDict()
        dict['sentence'] = text[i]
        dict['tags'] = tags[i]
        dict['head'] = {'word': entity1[i]}
        dict['tail'] = {'word': entity2[i]}
        dict['relation'] = labels[i]
        file.append(dict)

    with open(path_json, 'w') as f:
        json.dump(file, f, indent=4)
        f.close()
    
    tags_list = []
    for item in tags:
        tags_list.extend(item.split(' '))
    
    tags_set = list(set(tags_list))
    return tags_set
    
    
def main():
    path_train = 'TRAIN_FILE.TXT'
    path_test = 'TEST_FILE_FULL.TXT'
    path_train_json = 'data_json/train.json'
    path_test_json = 'data_json/test.json'
    
    path_tags = 'data_json/tags.json'
    
    print("Dealing with the train file...")
    train_tags_set = format2json(path_train, path_train_json)
    print("Finish cope with.")
    print("Coping with the test file...")
    test_tags_set = format2json(path_test, path_test_json)
    print("Finish cope with.")
    train_tags_set.extend(test_tags_set)
    tags_set = list(set(train_tags_set))
    
    with open(path_tags, 'w') as f:
        json.dump(tags_set, f, indent=4)
        f.close()
    
    
if __name__ == "__main__":
    main()