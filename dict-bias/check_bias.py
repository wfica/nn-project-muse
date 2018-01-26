#!/usr/bin/python2.7

ENLIST = []
ENSET_FB = set()

# taken from https://github.com/first20hours/google-10000-english/blob/master/google-10000-english.txt
# but I'll need to find another source of top English words as here we have "x", "g", "ul", "vg", ...
with open("top-10000-en.txt") as f:
    ENLIST = [line.split()[0] for line in f]

with open("en-de.txt") as f:
    ENSET_FB = set(line.split()[0] for line in f)

# print len(ENLIST), len(ENSET_FB) # prints 10000 74655

cnt = 0
ABSENT_WORDS = []
for i in range(len(ENLIST)):
    if ENLIST[i] in ENSET_FB:
        cnt += 1
    else:
        ABSENT_WORDS.append(ENLIST[i])

    if not (i + 1) % 1000:
        print "From top %d English words there are %d in FB dictionary." % (i + 1, cnt)
        print "Missing since last print:", ABSENT_WORDS
        print ''
        ABSENT_WORDS = []