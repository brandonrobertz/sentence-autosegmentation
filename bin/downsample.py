#!/usr/bin/env python2
from __future__ import print_function

import fileinput


# only keep this pct of the sentences (keeping the edges)
DOWNSAMPLE_PCT = 0.15
# keep maximum chars from end/start of sentences
MAX_SIZE = 15


for line in fileinput.input():
    if len(line) > 20:
        part = int(len(line)*DOWNSAMPLE_PCT)
        if part > MAX_SIZE:
            part = MAX_SIZE

        line_start = line[:part]
        line_end = line[-part:]
        line = line_start + line_start

    print( line)
