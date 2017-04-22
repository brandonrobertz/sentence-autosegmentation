# Both are pre-set for using the gutenberg cleaned dataset
# feel free to change them to use your own datasets
GUTENBERG_ZIP=data/Gutenberg.zip
DATA_FILES_DIR=data/Gutenberg/txt/

default:gutenberg_unzip build_trainingset downsample train_test

dirs:
	mkdir -p data
	mkdir -p models
	mkdir -p graph

gutenberg_unzip: dirs
	if ! [ -f ${GUTENBERG_ZIP} ]; then \
		echo "You need to download the Gutenberg.zip file first";
		echo "see this URL for more information & download link";
		echo "http://web.eecs.umich.edu/~lahiri/gutenberg_dataset.html";
	fi
	unzip -d data ${GUTENBERG_ZIP}

build_trainingset:
	find ${DATA_FILES_DIR} -type f -exec cat {} \; \
		| grep -v '^\s*$$' \
		| sed 's/\[.*\]//g' \
		| tr [:upper:] [:lower:] \
		| tr '-' ' ' \
		| sed "s/[^a-z0-9\s'\.\?\!]/ /g" \
		| sed 's/\s\+/ /g' \
		| ./bin/sentence_tokenize.py \
		| tr -d '.!?' \
		> data/dataset.sentences

downsample:
	cat data/dataset.sentences \
		| ./bin/downsample.py \
		> data/dataset.downsampled

train_test:
	./classifier.py data/dataset.downsampled

tensorboard:
	tensorboard --logdir ./graph
