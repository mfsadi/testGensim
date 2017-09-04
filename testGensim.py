'''
#-------------------------------------------------------------------------------------------------#
                                In the name of God
This script shows how to use gensim package in python for implementing word2vec models.
There is english and persian examples, but for achieving high quality word vectors it is strictly
recommended to use big corpora as training data, like google news.
 ------M.F.Saadi, Zanjan University, Computer Department, 04/09/2017, mfsadi@znu.ac.ir----
 Requirements:
  -gensim
  -matplotlib
  -sklearn
  -Google questions-phrases (Just for test, you can use any raw text data that is sentence separated)
#-------------------------------------------------------------------------------------------------#
'''

import gensim, logging
from gensim.models import word2vec
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from nltk.corpus import brown

# Showing log of training vectors
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = [['اول', 'جمله'], ['دوم', 'جمله']]
# Building Models
model1 = gensim.models.Word2Vec(word2vec.LineSentence("questions-phrases.txt"), min_count=1)
model2 = gensim.models.Word2Vec(sentences, min_count=1)
# model3=gensim.models.Word2Vec(brown.sents(), min_count=1)
# Saving models
model1.save("test.model")
model2.save("persian.model")
# model3.save("brown.model")

# Loading models
model11 = gensim.models.Word2Vec.load("test.model")
model12 = gensim.models.Word2Vec.load("persian.model")
# model13=gensim.models.Word2Vec.load("brown.model")
# Finding similarities
print(model11.most_similar("Bill_Gates"))
print(model12.most_similar("جمله"))
# print(model13.most_similar("queen"))
# for word, sim in model3.most_similar(positive=['italy', 'washington'], negative=['italy']):
#   print('\"%s\"\t- similarity: %g' % (word, sim))
# print('')
# Visualising word vectors
X = model1[model1.wv.vocab]
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
plt.show()
