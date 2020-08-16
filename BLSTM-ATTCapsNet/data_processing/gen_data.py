import numpy as np
import os
import json

UNK_word2vec = np.random.normal(loc=0.0, scale=0.05, size=300).tolist()
in_path = "./path_json/"


test_file_name = 'test.json'
word_file_name = 'myvec-nostopwords.json'
rel_file_name = 'relation2id.json'
tags_file_name = 'tags.json'


def init(file_name, word_vec_file_name, rel2id_file_name, tags_file_name, max_length = 87, is_training = 'train'):
    if file_name is None or not os.path.isfile(file_name):
        raise Exception("[ERROR] Data file doesn't exist")
    if word_vec_file_name is None or not os.path.isfile(word_vec_file_name):
        raise Exception("[ERROR] Word vector file doesn't exist")
    if rel2id_file_name is None or not os.path.isfile(rel2id_file_name):
        raise Exception("[ERROR] rel2id file doesn't exist")
        
    print("Loading data file...")
    ori_data = json.load(open(file_name, "r"))
    print("Finish loading")
    print("Loading word_vec file...")
    ori_word_vec = json.load(open(word_vec_file_name, "r"))
    print("Finish loading")
    print("Loading rel2id file...")
    rel2id = json.load(open(rel2id_file_name, "r"))
    print("Finish loading")
    print("Loading tags file...")
    tags = json.load(open(tags_file_name, "r"))
    print("Finish loading")
        
    print("Building word vector matrix and mapping...")
    word2id = {}
    word_vec_mat = []
    word_size = len(ori_word_vec[0]['vec'])
    print("Got {} words of {} dims".format(len(ori_word_vec), word_size))
    word2id['BLANK'] = len(word2id)
    word_vec_mat.append(np.zeros(word_size, dtype = np.float32))
    
    for i in ori_word_vec:
        word2id[i['word']] = len(word2id)
        word_vec_mat.append(i['vec'])
        
    word2id['UNK'] = len(word2id)
    word_vec_mat.append(np.array(UNK_word2vec))
    word_vec_mat = np.array(word_vec_mat, dtype = np.float32)
    print("Finish building")
    
    # 词性的映射
    print("Building pos_tags mapping...")
    tags2id = {}
    tags2id['BLANK'] = len(tags2id)
    for item in tags:
        tags2id[item] = len(tags2id)
    tags2id['UNK'] = len(tags2id)
    print("Finish building")
    
    sen_tot = len(ori_data)
    sen_word = np.zeros((sen_tot, max_length), dtype = np.int64)
    sen_tag = np.zeros((sen_tot, max_length), dtype = np.int64)
    sen_pos1 = np.zeros((sen_tot, max_length), dtype = np.int64)
    sen_pos2 = np.zeros((sen_tot, max_length), dtype = np.int64)
    sen_mask = np.zeros((sen_tot, max_length, 3), dtype = np.float32)
    sen_label = np.zeros((sen_tot), dtype = np.int64)
    sen_len = np.zeros((sen_tot), dtype = np.int64)
    for i in range(len(ori_data)):
        if  i%1000 == 0:
            print(i)
        sen = ori_data[i]
        # sen_label 
        sen_label[i] = rel2id[sen['relation']]

        words = sen['sentence'].split()
        tags = sen['tags'].split()
        # sen_len
        sen_len[i] = min(len(words), max_length)
        # sen_word
        for k, word in enumerate(words):
            if k < max_length:
                if word in word2id:
                    sen_word[i][k] = word2id[word]
                else:
                    sen_word[i][k] = word2id['UNK']
        for k in range(k + 1, max_length):
            sen_word[i][k] = word2id['BLANK']
        
        # sen_tag
        for j, tag in enumerate(tags):
            if j < max_length:
                if tag in tags2id:
                    sen_tag[i][j] = tags2id[tag]
                else:
                    sen_tag[i][j] = tags2id['UNK']
        for j in range(j+1, max_length):
            sen_tag[i][j] = tags2id['BLANK']
            
        try:
            pos1, pos2 = words.index(sen['head']['word'].split()[0]), words.index(sen['tail']['word'].split()[0])
        except ValueError as e:
            print(e)
            print(i)
        # 两个实体相同的情况
        if pos1 == pos2:
            try:
                pos2 = pos1 + words[pos1 + 1 : ].index(sen['tail']['word'].split()[0]) + 1
            except ValueError as e:
                print(i)
        if pos1 == -1 or pos2 == -1:
            raise Exception("[ERROR] Position error, index = {}, sentence = {}, head = {}, tail = {}".format(i, sen['sentence'], sen['head']['word'], sen['tail']['word']))
        if pos1 >= max_length:
            pos1 = max_length - 1
        if pos2 >= max_length:
            pos2 = max_length - 1
        pos_min = min(pos1, pos2)
        pos_max = max(pos1, pos2)
        for j in range(max_length):
            # sen_pos1, sen_pos2
            if j < sen_len[i]: # 这里该加多少，取决于不加的时候的最小值的绝对值加一, 应该用小于号
                sen_pos1[i][j] = j - pos1 + 68
                sen_pos2[i][j] = j - pos2 + 68
            else:
                sen_pos1[i][j] = 0
                sen_pos2[i][j] = 0
            # sen_mask
            if j >= sen_len[i]:
                sen_mask[i][j] = [0, 0, 0]
            elif j - pos_min <= 0:
                sen_mask[i][j] = [100, 0, 0]
            elif j - pos_max <= 0:
                sen_mask[i][j] = [0, 100, 0]
            else:
                sen_mask[i][j] = [0, 0, 100]
        
    # sent_scope
    sent_scope = np.stack([list(range(len(ori_data))), list(range(len(ori_data)))], axis = 1)
    print("Processing instance label...")
    # sent_label
    if is_training:
        sent_label = sen_label
    else:
        sent_label = []
        for i in sen_label:
            one_hot = np.zeros(len(rel2id), dtype = np.int64)
            one_hot[i] = 1
            sent_label.append(one_hot)
        sent_label = np.array(sent_label, dtype = np.int64)
    print("Finishing processing")
    sent_scope = np.array(sent_scope, dtype = np.int64)
    sent_label = np.array(sent_label, dtype = np.int64)
    
    # saving
    print("Saving files")
    if is_training == 'train':
        name_prefix = "train"
    elif is_training == 'valid':
        name_prefix = 'valid'
    else:
        name_prefix = "test"
    np.save(os.path.join(out_path, 'vec.npy'), word_vec_mat)
    np.save(os.path.join(out_path, name_prefix + '_word.npy'), sen_word)
    np.save(os.path.join(out_path, name_prefix + '_tag.npy'), sen_tag)
    np.save(os.path.join(out_path, name_prefix + '_pos1.npy'), sen_pos1)
    np.save(os.path.join(out_path, name_prefix + '_pos2.npy'), sen_pos2)
    np.save(os.path.join(out_path, name_prefix + '_mask.npy'), sen_mask)
    np.save(os.path.join(out_path, name_prefix + '_sen_label.npy'), sent_label)
    np.save(os.path.join(out_path, name_prefix + '_sen_scope.npy'), sent_scope)
    print("Finish saving")
    

for i in range(10):
    out_path = './datas-key1/'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    train_file_name = os.path.join(in_path, str(i+1) + '/train.json')
    valid_file_name = os.path.join(in_path, str(i+1) + '/valid.json')
    out_path = os.path.join(out_path, str(i+1))
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    init(train_file_name, word_file_name, rel_file_name, tags_file_name, max_length=87, is_training='train')
    init(valid_file_name, word_file_name, rel_file_name, tags_file_name, max_length=87, is_training='valid')
    init(test_file_name, word_file_name, rel_file_name,tags_file_name, max_length=87, is_training='test')
