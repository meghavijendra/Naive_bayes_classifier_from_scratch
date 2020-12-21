#!/usr/bin/env python
# coding: utf-8
"""
Author: Megha Vijendra
Student ID: 1001736938
"""


import os
import random
import math
import timeit


def train_test_split():
    train_groups = {}
    test_groups = {}
    random.seed(1)
    for x in os.listdir('20_newsgroups'):
        y = random.sample(os.listdir('20_newsgroups/'+x),500)
        train_groups[x] = y
        files = []
        for z in os.listdir('20_newsgroups/'+x):
            if z not in y:
                files.append(z)
        test_groups[x] = files
    return train_groups, test_groups


def clean_data(contents):
    contents = contents.lower()
    contents = contents.replace("\n",' ')
    contents = contents.replace("\t",' ')
    chars = "\\`*_{}[]()>#+-.!$^'!/=,:\"<?|-#*_@"
    for c in chars:
        if c in contents:
            contents = contents.replace(c, " ")
    f = open('stopwords.txt',"r")
    stop_words = f.read()
    stop_words = stop_words.split("\n")
    stop_words = [' {0} '.format(elem) for elem in stop_words]
    for word in stop_words: 
        if word in contents:
            contents = contents.replace(word,' ')
    return contents


def bag_of_words(train_groups):
    words_dict = {}
    total_count_dict = {}
    for key, values in train_groups.items():
        group = {}
        for value in values:
            f = open('20_newsgroups/'+ key+'/'+value,"r")
            contents = f.read()
            contents = clean_data(contents)
            contents = contents.split(" ")
            contents = [i for i in contents if i != '']
            for word in contents:
                if word in group:
                    group[word] += 1
                else:
                    group[word] = 1
                if word in total_count_dict:
                    total_count_dict[word] += 1
                else:
                    total_count_dict[word] = 1
        words_dict[key]= group
    return words_dict, total_count_dict


def naive_bayers(test_groups,words_dict,total_count_dict):
    loop = 0
    prediction = 0
    for target_class, values in test_groups.items():
        sub_folders = list(test_groups.keys())
        len_classes = len(test_groups.keys())
        for value in values:
            f = open('20_newsgroups/'+ target_class+'/'+value,"r")
            contents = f.read()
            contents = clean_data(contents)
            contents = contents.split(" ")
            contents  = [i for i in contents if i != '']
            if contents =='NULL':
                break
            loop+= 1
            confusion_matrix = []
            for key in test_groups.keys():
                confusion_matrix.append(calc_props(words_dict,key,contents,len_classes))
            predicted=max(confusion_matrix)
            if target_class == sub_folders[confusion_matrix.index(predicted)]:
                prediction +=1
    return prediction,loop-1


def calc_props(words_dict, class_name, contents, len_classes):
    probability = 0
    total_words = sum(words_dict[class_name].values())
    prior_probability = 1 / len_classes
    count = 0
    for word in contents:
        likelihood = words_dict[class_name].get(word, 0.01)
        probability += math.log((float(likelihood)*prior_probability)/float(total_words))
    return probability


def main():
    start = timeit.default_timer()
    train_groups,test_groups = train_test_split()
    print("Data split into Training and Testing set")
    words_dict, total_count_dict = bag_of_words(train_groups)
    print("\nModel is trained and the bag of words  has ", len(total_count_dict), "words" )
    print("\nTesting the trained model using naive bayers on the test data")
    predict, loop = naive_bayers(test_groups, words_dict, total_count_dict)
    print('Prediction accuracy = {}'.format(predict/loop*100))
    stop = timeit.default_timer()
    print('Time elapsed: ', int(stop - start),"secs")


if __name__== "__main__":
    main()

