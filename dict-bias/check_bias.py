#!/usr/bin/python2.7

import sys

ENLIST1000 = []
ENLIST3000 = []
ENSET_FB = set()

# taken from https://www.ef.com/english-resources/english-vocabulary/top-1000-words/
with open("top-1000-en.txt") as f:
    ENLIST1000 = [line.split()[0] for line in f]

# taken from https://www.ef.com/english-resources/english-vocabulary/top-3000-words/
with open("top-3000-en.txt") as f:
    ENLIST3000 = [line.split()[0] for line in f]

with open("en-de.txt") as f:
    ENSET_FB = set(line.split()[0] for line in f)

def print_words(list):
    for word in list:
        sys.stdout.write(word + " ") # no new line

# print len(ENSET_FB) # prints 74655

cnt = 0
ABSENT_WORDS = []
for i in range(len(ENLIST1000)):
    if ENLIST1000[i] in ENSET_FB:
        cnt += 1
    else:
        ABSENT_WORDS.append(ENLIST1000[i])

    if not (i + 1) % 1000:
        print "From top %d English words there are %d in FB dictionary." % (i + 1, cnt)
        print "Missing since last print:"
        print_words (ABSENT_WORDS)
        print ''
        ABSENT_WORDS = []

cnt = 0
ABSENT_WORDS = []
for i in range(len(ENLIST3000)):
    if ENLIST3000[i] in ENSET_FB:
        cnt += 1
    else:
        ABSENT_WORDS.append(ENLIST3000[i])

    if not (i + 1) % 3000:
        print "From top %d English words there are %d in FB dictionary." % (i + 1, cnt)
        print "Missing since last print:"
        print_words (ABSENT_WORDS)
        print ''
        ABSENT_WORDS = []