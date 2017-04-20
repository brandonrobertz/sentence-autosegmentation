
gutenberg:
	find data/Gutenberg/txt/ -type f -exec cat {} \; \
		| grep -v '^\s*$$' \
		| sed 's/\[.*\]//g' \
		| tr [:upper:] [:lower:] \
		| tr '-' ' ' \
		| sed "s/[^a-z0-9\s'\.\?\!]/ /g" \
		| sed 's/\s\+/ /g' \
		| ./bin/sentence_tokenize.py \
		| tr -d \. \
		> data/gutenberg.cleaned
