import gc

import numpy as np
import pandas as pd
import spacy
import textacy
import utils
import utils_clean
import utils_text
from gensim.models import KeyedVectors
from keras.preprocessing import sequence, text
from nltk.corpus import stopwords
from tqdm import tqdm
