#!/usr/bin/env python2
from __future__ import print_function

import fileinput


# only keep this pct of the sentences (keeping the edges)
DOWNSAMPLE_PCT = 0.25
# keep maximum chars from end/start of sentences
MAX_SENTENCE_LEN = 80


for line in fileinput.input():
    print('-'*50)
    line = line.replace('\n', '').replace('\r', '')
    print('WHOLE LINE:', line)

    if len(line) > MAX_SENTENCE_LEN:
        part = int(len(line)*DOWNSAMPLE_PCT)
        s_part = int(part / 2)

        # print('LEN:', len(line))
        # if s_part > MAX_SIZE:
        #     s_part = MAX_SIZE

        e_part = s_part

        line_start = line[:s_part]
        print('LINE START:', s_part, line_start)

        line_end = line[-e_part:]
        print('LINE END:', e_part, line_end)

        line = line_start + line_end

    print('OUTPUT:', line)
