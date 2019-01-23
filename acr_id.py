## @package docstring
# Acronym Expansion and their definitions from unknown documents
# @author: Denay Kirby

import sys
import re
import nltk
import numpy as np
import copy
import os


DIAG = 9
UP = 8
LEFT = 7

# common acronyms
dict = {
    "DOE" : "Department of Energy",
    "DOD" : "Department of Defense",
    "EPA" : "Environmental Protection Agency",
    "NRC" : "Nuclear Regulatory Commission",
    "GAO" : "Government Accountability Office"
    }

## splits file into word list
# @param file_obj file object to parse
# @return words list of words from file
def word_list(file_obj):
    words = []
    for line in file_obj:
        words.extend( line.split() )
    return words

## gets acronyms and corresponding indices
# @param words list of words from text file
# @return list of acronyms found
# @return acr_loc - list containing location of acronyms
def get_acronyms(words):
    acr = []
    acr_loc = []
    for index, w in enumerate(words):
        match = re.search('[A-Z]{3,10}', w)
        if match and match.group(0) not in acr:
            acr.append(match.group(0))
            acr_loc.append(index)
    return acr, acr_loc

## breaks list into subwindows around acronym
# @param words list of words
# @param acronym the acronym around window
# @param acronym_loc the location of the acronym in words
# @return prewindow the prewindow portion
# @return postwindow the postwindow portion
def get_subwindows(words, acronym, acronym_loc):
    subwin_len = len(acronym)*2
    if acronym_loc-subwin_len < 0:
        prewindow = words[:acronym_loc]
    else:
        prewindow = words[acronym_loc-subwin_len:acronym_loc]
    if acronym_loc+subwin_len > len(words):
        postwindow = words[acronym_loc:]
    else:
        postwindow = words[acronym_loc:acronym_loc+subwin_len]
    
    return prewindow, postwindow[1:]

## generates leader and type arrays from given window
# @param subwindow the subwindow to parse (pre or post)
# @return leaders array containing leading letters of words
# @return types array containing word types
# (s - stopword, H-initial part of hyphenated
# words, h-hypenated word part, a-acronym, w - regular word)
def parse_subwindow(subwindow):
    leaders = []
    types = []
    for w in subwindow:
        if w in stopwords:
            leaders.append(w[0].lower())
            types.append('s')
        elif '-' in w and len(w) > 1:
            hyphenated_w = w.split('-')
            leaders.append(hyphenated_w[0][0].lower())
            types.append('H')
            if not hyphenated_w[1:]:
                for h in hyphenated_w[1:]:
                    leaders.append(h[0].lower())
                    types.append('h')
        elif re.search('[A-Z]{3,10}', w):
            m = re.search('[A-Z]{3,10}', w)
            leaders.append(m.group(0)[0].lower())
            types.append('a')
        else:
            leaders.append(w[0].lower())
            types.append('w')
    return leaders, types


## builds longest common subsequence (LCS) matrix
# @param acronym acronym in question
# @param leader first letters of words in window
# @return lcs_len c matrix, contains length of LCS
# @return lcs_path b matrix, contains path of LCS
def build_LCS_matrix(acronym, leader):
    m = len(acronym)
    n = len(leader)
    lcs_len = np.zeros((m+1,n+1), dtype = int)
    lcs_path = np.zeros((m+1,n+1), dtype = int)

    for i in range(1,m+1):
        for j in range(1,n+1):
            if acronym[i-1].lower() == leader[j-1]:
                lcs_len[i,j] = lcs_len[i-1,j-1] + 1
                lcs_path[i,j] = DIAG
            elif lcs_len[i-1,j] >= lcs_len[i,j-1]:
                lcs_len[i,j] = lcs_len[i-1,j]
                lcs_path[i,j] = UP
            else:
                lcs_len[i,j] = lcs_len[i,j-1]
                lcs_path[i,j] = LEFT

    return lcs_len, lcs_path

## utility method to print LCS from build_LCS_matrix()
# @param path_matrix LCS path matrix
# @param seq sequence to print from
# @param i start position (lower right)
# @param j start position (lower right)
# example: print_LCS(b, acr[1].lower(), len(acr[1]), len(leaders))
def print_LCS(path_matrix, seq, i, j):    
    if i == 0 or j == 0:
        return
    if path_matrix[i,j] == DIAG:
        print_LCS(path_matrix, seq, i-1, j-1)
        print(seq[i-1])
    elif path_matrix[i,j] == UP:
        print_LCS(path_matrix, seq, i-1, j)
    else:
        print_LCS(path_matrix, seq, i, j-1)

## utility method to build vector from stack in parse_LCS_matrix()
# @param stack stack to build vector from
# @param n length of sequence
# @return v vector made from stack
def build_vector(stack, n):
    v = [0] * n

    for i in range(0, len(stack)):
        v[stack[i][1]-1] = stack[i][0]

    return v

## finds all acronym definition candidates from LCS matrix
# @param path_m path matrix (b) from build_LCS_matrix()
# @param start_i upper right corner (row) of path matrix
# @param start_j upper right corner (col) of path matrix
# @param m row dimension of path matrix
# @param n col dimension of path matrix
# @param lcs_len length of LCS
# @param stack stack to hold index positions of possible definition
# @param vectorlist list to hold possible definition candidates
# @return vectorlist - list of all possible definition candidates
def parse_LCS_matrix(path_m, start_i, start_j, m, n, lcs_len,
                     stack, vectorlist):    
    for i in range(start_i, m):
        for j in range(start_j, n):
            if path_m[i,j] == DIAG:
                stack.append([i,j])
                if lcs_len == 1:
                    vectorlist.append(build_vector(stack, n-1))                    
                else:
                    parse_LCS_matrix(path_m, i+1, j+1, m, n, lcs_len-1,
                                     stack,vectorlist)
                stack.pop()
    return vectorlist

## calculates statistics for a definition candidate
# @param vector vector in question
# @param types array of word types from subwindow
# @return misses the number of misses in the vector
# @return stopcount the nubmer of stopwords in definition
# @return distance the distance from the acronym
# @return size the number of entries excepting leading and trailing zeroes
def vector_values(vector, types):
    misses = 0
    stopcount = 0

    # calculate size
    i = 0
    while i < len(vector) and vector[i] == 0:
        i += 1
    first = i    
    i = len(vector)-1
    while i >= 0 and vector[i] == 0:
        i -= 1
    last = i
    size = last - first + 1

    #calculate distance
    distance = (len(vector)-1) - last

    # calculate misses and stopcount
    for i in range(first, last+1):
        if vector[i] > 0 and types[i] == 's':
            stopcount += 1
        elif vector[i] == 0 and types[i] != 's' and types[i] != 'h':
            misses += 1

    return misses, stopcount, distance, size

## compares two vectors of definition candidates
# @param vec_a first vector to compare
# @param vec_b second vector to compare
# @return the vector with better likelihood of definition
def compare_vectors(vec_a, vec_b, types):
    miss = 0
    stopct = 1
    dist = 2
    size = 3
    stats_a = vector_values(vec_a, types)
    stats_b = vector_values(vec_b, types)

    if stats_a[miss] > stats_b[miss]:
        return vec_b
    elif stats_a[miss] < stats_b[miss]:
        return vec_a
    if stats_a[stopct] > stats_b[stopct]:
        return vec_b
    elif stats_a[stopct] < stats_b[stopct]:
        return vec_a
    if stats_a[dist] > stats_b[dist]:
        return vec_b
    elif stats_a[dist] < stats_b[dist]:
        return vec_a
    if stats_a[size] > stats_b[size]:
        return vec_b
    elif stats_a[size] < stats_b[size]:
        return vec_a
    return vec_a

## prints probable definition of acronym
# @param def_vector vector containing possible definition
# @param subwindow the subwindow corresponding to the def_vector
# @return definition
def print_definition(def_vector, subwindow):
    definition = []
    subwin = []
    
    # break up hyphenated words if they exist
    for w in subwindow:
        if '-' in w:
            hyphen = w.split('-')
            for h in hyphen:
                subwin.append(h)
        else:
            subwin.append(w)

    # print definition from vector
    for i in range(0, len(def_vector)):
        if def_vector[i] > 0:
            definition.append(subwin[i])
    print(definition)

## compares all vectors generated from parse_LCS_matrix()
# @param vectorlist the vectorlist from parse_LCS_matrix
# @param types the type array corresponding to the subwindow being searched
# @return best the best one found in the vectorlist
def compare_vectors_driver(vectorlist, types):
    for i in range(0, len(vectorlist)-1):
        best = compare_vectors(vectorlist[i], vectorlist[i+1], types)
        
    return best

# MAIN PROGRAM
# description: runs methods to search text for acronyms

## given text as list of words prints acronyms and their definitions
# @param word_list text to be searched
# @param error default is .2
def find_acronym_definitions(words, error = .2):
    acr, acr_loc = get_acronyms(words)
    
    for index, item in enumerate(acr):
        print(item)
        window = get_subwindows(words, item, acr_loc[index])
        confidence_lvl = []
        for i, sub in enumerate(window):
            leaders, types = parse_subwindow(sub)
            c, b = build_LCS_matrix(item, leaders)
            lcs_len = c[c.shape[0]-1, c.shape[1]-1]
            confidence_lvl.append((lcs_len/len(item)*1.0) + error)

        # pick window with greater confidence_lvl
        if confidence_lvl[0] > confidence_lvl[1] and confidence_lvl[0] > 1.0:
            subwindow = 0
            leaders, types = parse_subwindow(window[subwindow])
            c, b = build_LCS_matrix(item, leaders)
            lcs_len = c[c.shape[0]-1, c.shape[1]-1]
        elif confidence_lvl[0] < confidence_lvl[0] and confidence_lvl[1] > 1.0:
            subwindow = 1
        else:
            if item in dict:
                print(dict[item])
            else:
                print("no matching definition found")
            continue
            

        possible_defs = parse_LCS_matrix(b, 0, 0, b.shape[0], b.shape[1], lcs_len, [], [])
        if len(possible_defs) > 1:
            best_match = compare_vectors_driver(possible_defs, types)
        else:
            best_match = possible_defs[0]

        print_definition(best_match, window[subwindow])

# read in file and run program
filename = "crs-2015-aml-0292_from_1_to_30.txt"
# filename = "d1072.txt"

try:
    file_object = open(filename, 'r')
except:
    IOError

stopwords = nltk.corpus.stopwords.words('english')
words = word_list(file_object)
find_acronym_definitions(words, .2)
