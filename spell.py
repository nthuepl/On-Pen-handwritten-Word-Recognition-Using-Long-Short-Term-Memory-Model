import re
import math
from Levenshtein import levenshtein
from collections import Counter

def words(text): return re.findall(r'\w+', text.lower())

WORDS = Counter(words(open('20k.txt').read()))
default_MSR = {'i':0.2, 'n':0.1, 'other':0.01}
default_dw = 0.133

def P(word, N=sum(WORDS.values())): 
    "Probability of `word`."
    return WORDS[word] / N

def MSP(src, word, MSR=default_MSR, distance_weight=default_dw):
    p = 1
    p *= math.pow(distance_weight, levenshtein(src, word, len_weight=1))
    for ch in set(word)-(set(word) & set(src)):
        if ch in MSR.keys():
            p *= MSR[ch]
        else:
            p *= MSR['other']
    return p

def correction(word, MSR=default_MSR, distance_weight=default_dw): 
    "Most probable spelling correction for word."
    # return max(candidates(word), key=P)
    p_contribution = {}
    # p_contribution = [MSP(word, candidate) for candidate in candidates(word)]
    for candidate in candidates(word):
        p_contribution[candidate] = MSP(word, candidate, MSR=MSR, distance_weight=distance_weight)
    for candidate, p in p_contribution.items():
        if p is max(p_contribution.values()):
            return candidate

def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) | known(edits1(word)) | known(edits2(word)))

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    similars = {'m':'n', 'j':'i'}
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    # transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + similars[R[0]] + R[1:]           for L, R in splits if R and R[0] in similars.keys()]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    # return (e2 for e1 in edits1(word) for e2 in edits1(e1))
    tmp = set()
    for e1 in edits1(word):
        for e2 in edits1(e1):
            tmp.add(e2)
    return tmp

if __name__ == '__main__':
    print(candidates("aooayttes"))