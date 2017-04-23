#!/usr/bin/env python2
""" TODO: write a config-finding tool that finds a
sentence length that optimizes equal training set
balance (windows with breaks vs windows w/o)
"""
from __future__ import print_function

import fileinput


# keep maximum chars from end/start of sentences
MAX_SENTENCE_LEN = 40


for line in fileinput.input():
    print('-'*50)
    line = line.replace('\n', '').replace('\r', '')

    if len(line) > MAX_SENTENCE_LEN:
        part = int(MAX_SENTENCE_LEN / 2)

        # print('LEN:', len(line))
        # if s_part > MAX_SIZE:
        #     s_part = MAX_SIZE

        line_start = line[:part]

        line_end = line[part:]
        line = line_start + line_end

    print(line)
