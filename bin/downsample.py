#!/usr/bin/env python2
""" TODO: write a config-finding tool that finds a
sentence length that optimizes equal training set
balance (windows with breaks vs windows w/o). Right
now a simple heruistic of some simple multiple of
the window and sentence len works (i'm using 3x).
"""
from __future__ import print_function

import fileinput


# keep maximum chars from end/start of sentences
MAX_SENTENCE_LEN = 80


for line in fileinput.input():
    line = line.replace('\n', '').replace('\r', '').strip()

    if not line:
        continue

    if len(line) > MAX_SENTENCE_LEN:
        part = int(MAX_SENTENCE_LEN / 4.0)

        # print('LEN:', len(line))
        # if s_part > MAX_SIZE:
        #     s_part = MAX_SIZE

        line_start = line[:part]
        line_cntr = line[len(line):len(line)+part*2]
        line_end = line[-part:]
        line = line_start + line_end

    print(line)
