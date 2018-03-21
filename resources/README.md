# TODO:

-   Pytorch models, [CNN text class](https://github.com/Shawn1993/cnn-text-classification-pytorch)
-   Char-level RNN/CNN, [kernel](https://www.kaggle.com/kmader/character-level-cnn-classification-with-dilations)
-   Train model for each class and use binary features for most important words, [kernel](https://www.kaggle.com/iezepov/the-worst-comment-ever-via-linear-model-lb-0-052)
-   Hierarchical Attention Networks, [keras](https://github.com/richliao/textClassifier)
-   Pyramid Networks, [Pytorch](https://github.com/dyhan0920/PyramidNet-PyTorch)
-   Twitter glove and preprocessing function, [disc](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/50350)
-   DeepMoji attention layer, [keras](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/49950)

## Working:

-   12.03: FastText Crawl seems to perform well.
-   12.03: Use Adam with 5e-4 & batch 256 or 1e-3 & 128 or **5e-4 & 128**. 5-fold is good enough.
-   12.03: Also LSTMdeep & GRUdeep v. good.
-   11.03: LSTMconcat2 and GRUconcat2 best so far + LSTMHierarchical. LSTM & GRUconcat 2nd place.
-   10.03: 200dim Glove, 320 seq len, 200k features.
-   [Keras attention layer](https://gist.github.com/cbaziotis/7ef97ccf71cbc14366835198c09809d2). Works with LSTM best.
-   Glove embedding works best (Glove vs FastText vs Google vs mine. Glove first transformed to gensim by their script).
-   BasicClean dataset gives the best results so far.
-   Nadam working best, 5e-4.
-   Use CuDNN cells, almost 2x speedup.

## In the middle:

-   2nd Attention type, [kernel](https://www.kaggle.com/qqgeogor/keras-lstm-attention-glove840b-lb-0-043). Worse than AttentionWithContext.

## Not working:

-   Spacy and Textacy cleaning gives worse results


# Resources:

## Various:

-   [Text Classification](https://github.com/brightmart/text_classification)
-   [Awesome NLP](https://github.com/keon/awesome-nlp)
-   [Tips for training RNNs](https://danijar.com/tips-for-training-recurrent-neural-networks/)
-   [NLP Best practices](http://ruder.io/deep-learning-nlp-best-practices/index.html)

## Features:

-   Train [W2V](https://radimrehurek.com/gensim/models/word2vec.html) for each class and use
    probabilities from model as feature

## Keras:

-   [Keras attention](https://github.com/philipperemy/keras-attention-mechanism)

### Quora:

-   [Quora 1st DL models](https://www.kaggle.com/lamdang/dl-models)
-   [1D CNN Quora model](https://www.kaggle.com/rethfro/1d-cnn-single-model-score-0-14-0-16-or-0-23)
