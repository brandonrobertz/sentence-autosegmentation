# Sentence Auto-Segmentation

Work-in-progress. Deep learning based sentence
segmentation from totally unstructured and unpunctuated
text like you'd get from autotranslation or speech-to-text.

Take raw text like this:

    the place is an english sea port the time is night and the business of the moment is dancing

And turn it into this:

    the place is an english sea port
    the time is night
    and the business of the moment is dancing

## Model Training

Just run `classification.py [dataset/location]` and it will
load the specified dataset, vectorize it (character-model),
train, test, and save a model.

## Dataset Creation

Training set comes from a pre-cleaned subset of the Gutenberg corpus.
You can download it from the
[University of Michigan](http://web.eecs.umich.edu/~lahiri/gutenberg_dataset.html).

The `Makefile` contains a pre-processing command that
pipes the data through sed/grep/tr which turns input text
into a similar format as that output by speech autotranslation
models with the exception that it is output as one sentence
per line for training purposes.

Assuming you've downloaded the gutenberg dataset
and placed it in the `data/` directory, all you
need to do to get the initial formatted dataset
is to run:

    make gutenberg_unzip build_trainingset

This will place a training corpus in `data/dataset.sentences`.

Since the dataset is heavily skewed, there is
a downsampling script which removes the center of
long sentences. `make downsample` will do this and
output it to `data/dataset.downsampled`
