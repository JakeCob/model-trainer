import numpy as np
import math
import utils


# expectation maximization
def expect_max(pfil_word_dict, pen_word_dict, plst_fil_sen, plst_en_sen):
    fil_occur = len(pfil_word_dict)
    en_occur = len(pen_word_dict)

    # IBM1 Expectation Maximization Algorithm
    # np.full() Return a new array of given shape and type, filled with fill_value.
    trans_en_fil_matrix = np.full((len(pfil_word_dict), len(pen_word_dict)), 1 / len(pen_word_dict), dtype=float)
    trans_en_fil_matrix_prev = np.full((len(pfil_word_dict), len(pen_word_dict)), 1, dtype=float)

    int_count = 0
    # Loop until
    while not utils.is_converged(trans_en_fil_matrix, trans_en_fil_matrix_prev, int_count):
        int_count += 1

        # making the current matrix as the old one
        trans_en_fil_matrix_prev = trans_en_fil_matrix.copy

        # initializing the enfil's value as 0
        total_enfil = np.full((len(pfil_word_dict), len(pen_word_dict)), 0, dtype=float)

        # initializing the final total value
        total_fin = np.full((len(pen_word_dict)), 0, dtype=float)

        for int_index, lst_fil_sen in enumerate(plst_fil_sen):  # for all sentence pairs (e,f)
            # computing for the normalization
            lst_fil_words = lst_fil_sen.split(" ")
            total_sen = np.full((len(lst_fil_words)), 0, dtype=float)

            # for all words in the filipino list of words
            for int_index2 in range(len(lst_fil_words)):
                str_fil_word = lst_fil_words[int_index2]

                # di ko pa gets bakit ginawang 0
                total_sen[int_index2] = 0
                lst_en_words = plst_en_sen[int_index].split(" ")

                # for all string words in the list of words
                for str_en_word in lst_en_words:
                    # continue even if the string is empty
                    if str_en_word == '':
                        continue

                    int_index_fildict = pfil_word_dict[str_fil_word]
                    int_index_endict = pen_word_dict[str_en_word]
                    total_sen[int_index2] += trans_en_fil_matrix[int_index_fildict][int_index_endict]

            # collecting the counts
            lst_fil_words = lst_fil_sen.split(" ")

            for int_index2 in range(len(lst_fil_words)):
                str_fil_word = lst_fil_words[int_index2]
                lst_en_words = plst_en_sen[int_index].split(" ")

                # for all string words in the list of words
                for str_en_word in lst_en_words:
                    if str_en_word == '':
                        continue

                    int_index_fildict = pfil_word_dict[str_fil_word]
                    int_index_endict = pen_word_dict[str_en_word]
                    total_enfil[int_index_fildict][int_index_endict] += trans_en_fil_matrix[int_index_fildict][int_index_endict] / total_sen[int_index2]
                    total_fin[int_index_fildict] += trans_en_fil_matrix[int_index_fildict][int_index_endict] / total_sen[int_index2]

        # estimate probabilities
        for int_en_index in range(en_occur):
            for int_fil_index in range(fil_occur):
                if total_enfil[int_fil_index][int_en_index] != 0:
                    trans_en_fil_matrix[int_fil_index][int_en_index] = total_enfil[int_fil_index][int_en_index] / total_fin[int_en_index]

    print("EM Algorithm Converged in ", (int_count-1), " iterations")
    return trans_en_fil_matrix
