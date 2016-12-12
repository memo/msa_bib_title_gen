# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import argparse
import random
import numpy as np
import string
from collections import Counter

import bibtexparser
#import nltk #nltk does better job of tokening actually, 
#nltk.download() # only need to download once



def get_args():
    """check and return command line arguments"""
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--phrase', default='Junkhead', help='phrase to unabbreviate')
    parser.add_argument('--bib_path', default='msa.bib', help='path to bibtex file')
    parser.add_argument('--count_thresh', type=int, default=3, help='ignore words occuring less than this many times')
    parser.add_argument('--sample_count', type=int, default=30, help='how many to sample')
    args = parser.parse_args()
    args.bib_path = os.path.expandvars(args.bib_path)
    args.phrase = args.phrase.upper()
    return args



def get_bib_titles(bib_path):
    """load bibtex file and return all titles)"""
    with open(bib_path, 'r') as f:
        bibtex_str = f.read()
    bib_db = bibtexparser.loads(bibtex_str)
    return [x['title'] for x in bib_db.entries]


    
def tokenise(text):
    #nltk does better tokenisation, but not using it for now to avoid dependency
    # remove punctuation
    text = text.encode('ascii','ignore')
    text = text.translate(None, string.punctuation)
    return text.split()

    

def filter_words(words, word_counts, count_thresh):
    """return words with first letter capitalised, ignore non-alphanumeric and word freq < count_thresh"""
    return [x.capitalize() for x in words if x[0].isalpha() and word_counts[x] >= count_thresh]


            
def update_vocab_info(vocab, word_counts):
    """ return dict of { word : {'count', 'index'} } """
    return { word: {'count': word_counts[word], 'index':i} for i,word in enumerate(vocab)}

    

def get_words_stats(titles, count_thresh):
    """given a text, get all words and transition probabilities"""
    words = tokenise(' '.join(titles)) # full list of all words in titles
    words = [x.capitalize() for x in words] # capitalise first letter of each word
    word_counts = Counter(words) # count number of occurrences of each word
    words = filter_words(words, word_counts, count_thresh) # filter out rare and non-words
    vocab = sorted(list(set(words))) # vocabulary
 
    # construct word index map for reverse index lookup
    vocab_info = update_vocab_info(vocab, word_counts)

    
    # calculate transition probabilities
    # row : prob distribution over words (cols) for next word
    # two extra rows for [start of phrase] and [end of phrase]
    trans_probs_matrix = np.zeros([len(vocab)+2, len(vocab)], dtype=np.float64)
    word_start_index = len(vocab) # row index for start probabilities
    word_end_index = word_start_index+1 # row index for end probabilities
    print("TRAINING...")
    for title in titles: # loop all titles
        title_words = filter_words(tokenise(title), word_counts, count_thresh)  
        if len(title_words) > 0:
            print("processing title: ", title, title_words)
            prev_word_index = word_start_index # begin loop on sentence start probability row
            for cur_word in title_words: # loop all words in title
                cur_word_index = vocab_info[cur_word]['index']
                trans_probs_matrix[prev_word_index][cur_word_index] += 1
                print("  ", cur_word, cur_word_index, prev_word_index, trans_probs_matrix[prev_word_index][cur_word_index])
                prev_word_index = cur_word_index
                
            # increase end probability
            last_word_index = vocab_info[ title_words[-1] ]['index']
            trans_probs_matrix[word_end_index][last_word_index] += 1

    trans_probs_matrix[:] += 0.1 # add low equal probs (TODO: make this an arg parameter?)

    # normalise all rows
#    for row in trans_probs_matrix:
#        row /= np.sum(row)
        
    # add start token to info
    vocab_info[''] = {'count':0, 'index':word_start_index}
    
    return vocab, vocab_info, trans_probs_matrix


        
def keep_indices_zero_rest(probs, keep_indices, normalise=True):
    """
    probs : np.array of probability distribution
    keep_indices : indices to keep, zero out rest
    """
    probs = probs.copy()
    mask = np.ones(probs.shape,dtype=bool)
    mask[keep_indices] = False
    probs[mask] = 0
    if normalise:
        probs /= np.sum(probs)
    return probs
    
    
    
def sample_word_rand_uni(p):
    """sample word randomly with uniform distribution"""
    vocab_subset = p[0]
    return random.choice(vocab_subset)
        
    
    
def sample_word_max_prob_joint(p):
    """sample word with highest probability in joint distribution"""
    vocab_subset, vocab_subset_indices, word, trans_probs_matrix, vocab_info = p
    word_index = vocab_info[word]['index']
    probs = trans_probs_matrix[word_index] # probability distribution for following word
    probs = keep_indices_zero_rest(probs, vocab_subset_indices, True)
 #   new_word_index = np.argmax(np.random.multinomial(1, probs, 1))
    new_word_index = np.argmax(probs)
    return vocab[new_word_index]
    

    
def sample_word_prob_joint(p):
    """sample word from joint distribution"""
    vocab_subset, vocab_subset_indices, word, trans_probs_matrix, vocab_info = p
    word_index = vocab_info[word]['index']
    probs = trans_probs_matrix[word_index] # probability distribution for following word
    probs = keep_indices_zero_rest(probs, vocab_subset_indices, True)
    new_word_index = np.argmax(np.random.multinomial(1, probs, 1))
    return vocab[new_word_index]
    
    
    
def generate_title(phrase, vocab, vocab_info, trans_probs_matrix, sample_word_fn):
    title = []
    word = ''
    for c in phrase:
        u = c.upper()
        vocab_subset = [x for x in vocab if x[0] == u]
        if len(vocab_subset) > 0:
            vocab_subset_indices = [vocab_info[x]['index'] for x in vocab_subset]
            word = sample_word_fn( (vocab_subset, vocab_subset_indices, word, trans_probs_matrix, vocab_info) )
            title.append(word)
    title = ' '.join(title)
    print(title)
   
    
    
def dump_word_options(phrase, vocab):
    """dump options for each letter"""
    print("Word options for phrase:", phrase)
    for c in phrase:
        u = c.upper()
        vocab_subset = [x for x in vocab if x[0] == u]
        print(vocab_subset)


        
if __name__ == '__main__':
    args = get_args()
    titles = get_bib_titles(args.bib_path)
    vocab, vocab_info, trans_probs_matrix = get_words_stats(titles, args.count_thresh)
    
    dump_word_options(args.phrase, vocab)
    
    print("\nGenerating with random:")
    generate_title(args.phrase, vocab, vocab_info, trans_probs_matrix, sample_word_rand_uni)
    
    print("\nGenerating with maximum likelihood:")
    generate_title(args.phrase, vocab, vocab_info, trans_probs_matrix, sample_word_max_prob_joint)
    
    print("\nGenerating with probability:")
    for i in xrange(args.sample_count):
        generate_title(args.phrase, vocab, vocab_info, trans_probs_matrix, sample_word_prob_joint)
