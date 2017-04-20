# Sentence Auto-Segmentation

Work-in-progress. Deep learning based sentence
segmentation from totally unstructured text.

## Dataset Creation

Training set comes from a pre-cleaned subset of the Gutenberg corpus.
You can download it here: http://web.eecs.umich.edu/~lahiri/gutenberg_dataset.html

The script will load the data, split it into
windows by steps and convert into numpy arrays.

Assuming you've downloaded the gutenberg dataset
and placed it in the `data/` directory, all you
need to do to get the initial formatted dataset
is to run:

    make gutenberg_unzip build_trainingset

This will place a training corpus in `data/dataset.sentences`.
