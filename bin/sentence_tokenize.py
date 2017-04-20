#!/usr/bin/env python2
from __future__ import print_function

import fileinput

from nltk import sent_tokenize, download


remainder = None
for rawline in fileinput.input():
    try:
        line = rawline.strip().decode('utf-8')
    except LookupError:
        download('punkt')

    if not line:
        continue

    if remainder:
        line = "%s %s" % (remainder, line )
        remainder = None

    sents = sent_tokenize(line)

    if line[-1] not in "!?.;":
        remainder = sents.pop()

    for sent in sents:
        try:
            print( sent)
        except UnicodeEncodeError:
            pass
