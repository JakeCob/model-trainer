import string
import ibm1_em
from nltk.tokenize import word_tokenize
import numpy as np

def sen_tokenizer(plist_sentence, pint_max_trans):
    lst_final = list()
    # (value, key)
    word_dict = {}  # will keep the word and order in its language
    # (key, value)
    reverse_dict = {}
    int_order = 0
    int_count = 0

    # creating a table (33: None, ...) where 33 is the unicode of the characters
    tbl_translate = dict((ord(char), None) for char in string.punctuation)

    for lst_indiv_sen in plist_sentence[:pint_max_trans]:
        if int_count == 0:
            # The \ufeff is only found in the first line. It’s the beginning of the file.
            lst_indiv_sen = lst_indiv_sen.replace(u'\ufeff', '')
            int_count += 1

        # returns a string where some specified characters are replaced with the character described in a dictionary
        temp = lst_indiv_sen.translate(tbl_translate)  # remove punctuation
        lst_tokens = word_tokenize(lst_indiv_sen.lower())

        str_output = ""

        # loop for storing the tokens in the two dictionaries and in the output string
        for str_token in lst_tokens:
            # storing in the dictionaries
            if str_token not in word_dict:
                word_dict[str_token] = int_order
                reverse_dict[int_order] = str_token
                int_order += 1
            # adding the token to the stored string
            str_output = str_output + str_token + " "

        str_output = str_output[:(len(str_output) - 1)]

        # storing the output in the final list at the end
        lst_final.append(str_output)

    # replacing the ufeff character from document start with empty string
    lst_final[0] = lst_final[0].replace(u'\ufeff', '')

    return lst_final, word_dict, reverse_dict


# Main
# Opening the data sets and putting it in an object
with open("english_corpus.txt", encoding="utf8") as text_file:
    obj_data_en = text_file.readlines()

with open("filipino_corpus.txt", encoding="utf8") as text_file:
    obj_data_fil = text_file.readlines()

# Lists for the new data
lst_new_en = list()
lst_new_fil = list()

# int_count = 0

for sen_counter in range(len(obj_data_en)):
    if sen_counter > 500000:
        break

    # Splits the sentence word by  word
    current_en_sen = obj_data_en[sen_counter].split()
    current_fil_sen = obj_data_fil[sen_counter].split()

    # checking the tokens
    # print(current_en_sen)
    # print(current_fil_sen)
    # int_count += 1

    # Adds the sentence at the end of the list
    lst_new_en.append(obj_data_en[sen_counter])
    lst_new_fil.append(obj_data_fil[sen_counter])

# putting the tokenized data to the data variables
# list of sentences where each sentence is a list of words
obj_data_en = lst_new_en.copy()
obj_data_fil = lst_new_fil.copy()

# Checking the list of the separated sentences
# print(obj_data_en)
# print(obj_data_fil)

# checking the number of sentences after tokenizing
# print(int_count)

max_trans = 30000

# for parsing the filipino sentences and tokenizing the words
lst_fil_sen, fil_word_dict, reverse_fil_dict = sen_tokenizer(obj_data_fil, max_trans)

# for parsing the english sentences and tokenizing the words
lst_en_sen, en_word_dict, reverse_en_dict = sen_tokenizer(obj_data_en, max_trans)

# IBM Model 1 expectation maximization
translate_eng_fil = ibm1_em.expect_max(fil_word_dict, en_word_dict, lst_fil_sen, lst_en_sen)

print(translate_eng_fil)
