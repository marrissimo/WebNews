import gensim
from gensim import utils
from gensim.models.doc2vec import TaggedDocument

class TaggedLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])
    #esporta vettore documento
    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences
    #permuta x addestramento epoche
    def sentences_perm(self):
        shuffle(self.sentences)
	return self.sentences

from gensim.models import Doc2Vec
from gensim.parsing import PorterStemmer  # indagare sul modello

global_stemmer = PorterStemmer()
class StemmingHelper(object):
    """
    Class to aid the stemming process - from word to stemmed form,
    and vice versa.
    The 'original' form of a stemmed word will be returned as the
    form in which its been used the most number of times in the text.
    """

    # This reverse lookup will remember the original forms of the stemmed
    # words
    word_lookup = {}

    @classmethod
    def stem(cls, word):
        """
        Stems a word and updates the reverse lookup.
        """

        # Stem the word
        stemmed = global_stemmer.stem(word)

        # Update the word lookup
        if stemmed not in cls.word_lookup:
            cls.word_lookup[stemmed] = {}
        cls.word_lookup[stemmed][word] = (
            cls.word_lookup[stemmed].get(word, 0) + 1)

        return stemmed

    @classmethod
    def original_form(cls, word):
        """
        Returns original form of a word given the stemmed version,
        as stored in the word lookup.
        """

        if word in cls.word_lookup:
            return max(cls.word_lookup[word].keys(),
                       key=lambda x: cls.word_lookup[word][x])
        else:
            return word
# ___________________________________________________________________

import random
from random import shuffle
import numpy
# ___________________________________________________________________
from sklearn.linear_model import LogisticRegression

# MULTICORE - FAST VERSION ___________________________________________________________________
import multiprocessing

cores = multiprocessing.cpu_count()
assert gensim.models.doc2vec.FAST_VERSION > -1

# LOG OPTION__________________________________________________________________________________
import logging
import sys

log = logging.getLogger()
log.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)
#________________________________________________________________________________________

# INIZIALIZZAZIONE :

# SELEZIONA MODALITA -  scegliere tra classificare 'titoli' o 'contenuti' degli articoli:


#mod = 'titoli'

mod = 'titoli'
path = './'+mod+'/'

mod2 = 'articoli'
path2 = './'+mod2+'/'



score = []

#!!!!!!!!!!!!!!!!!!!!!!!!!!INPUT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Carica Modello : load_model = 1  ( altrimenti lo crea dal train)
load_model = 1
size = 100
#!!!!!!!!!!!!!!!!!!!!!!!!!!INPUT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# TRAIN :__________________________________________________________________________________

# il file 'KFoldCrossValidation.py' divide il dataset inziale in k=3 parti, una usata per il test  e le altre due per il  train.

for k in range(0,3) :

    # Carico i file sorgenti del dataset : TRAIN POS , TRAIN NEG ,TEST POS e TEST NEG
    sources_title = {path+'Test_'+str(k)+'_FALSE.txt':'TEST_NEG',
               path+'Test_'+str(k)+'_TRUE.txt':'TEST_POS',
               path+'Train_'+str(k)+'_FALSE.txt':'TRAIN_NEG',
               path+'Train_'+str(k)+'_TRUE.txt':'TRAIN_POS'}
    sources_article = {path2+'Test_'+str(k)+'_FALSE.txt':'TEST_NEG',
               path2+'Test_'+str(k)+'_TRUE.txt':'TEST_POS',
               path2+'Train_'+str(k)+'_FALSE.txt':'TRAIN_NEG',
               path2+'Train_'+str(k)+'_TRUE.txt':'TRAIN_POS'}

    # Conto il numero degli articoli , uno per  ogni newline '\n' ( prendo quella piu piccola tra TRAIN POS e TRAIN NEG : voglio che abbiano lo stesso numero di articoli)
    lunghezza = min(len( open(path+'Train_'+str(k)+'_FALSE.txt').read().split('\n') ), len( open(path+'Train_'+str(k)+'_TRUE.txt').read().split('\n') )) - 1

    sentences_title = TaggedLineSentence(sources_title)
    sentences_article = TaggedLineSentence(sources_article)


    # Se voglio creare il modello e non caricarne uno precedente:
    if load_model == 0:
    #    log.info('TaggedDocument')


        # log.info('D2V')
        model_title = Doc2Vec(min_count=1, window=10, size=size, sample=1e-4, negative=5, workers=cores)
        model_title.build_vocab(sentences_title.to_array())

        model_article = Doc2Vec(min_count=1, window=10, size=size, sample=1e-4, negative=5, workers=cores)
        model_article.build_vocab(sentences_article.to_array())

        log.info('Epoch')
        for epoch in range(10):
            log.info('EPOCH: {}'.format(epoch))

            model_title.train(sentences_title.sentences_perm())
            #model_article.train(sentences_article.sentences_perm())

        # log.info('Model Save')
        model_title.save('./models/FakeNews_model_'+mod+'_'+str(k)+'_'+str(size)+'.d2v')
        model_article.save('./models/FakeNews_model_' + mod2 + '_' + str(k) + '_' + str(size) + '.d2v')

    model_title = Doc2Vec.load('./models/FakeNews_model_'+mod+'_'+str(k)+'_'+str(size)+'.d2v')
    log.info('Sentiment')
    model_article = Doc2Vec.load('./models/FakeNews_model_'+mod2+'_'+str(k)+'_'+str(size)+'.d2v')
    log.info('Sentiment')

    train_arrays = numpy.zeros((lunghezza*2, size*2))
    train_labels = numpy.zeros(lunghezza*2)

    for i in range(lunghezza):
        prefix_train_pos = 'TRAIN_POS_' + str(i)
        prefix_train_neg = 'TRAIN_NEG_' + str(i)

        t_vector= model_title.docvecs[prefix_train_pos]
        a_vector= model_article.docvecs[prefix_train_pos]

        train_arrays[i] =  numpy.concatenate( (t_vector, a_vector), axis=0)

        train_arrays[ lunghezza + i] = numpy.concatenate( (model_title.docvecs[prefix_train_neg], model_article.docvecs[prefix_train_neg]),axis=0)

        train_labels[i] = 1
        train_labels[lunghezza + i] = 0

    #print train_labels

# TEST: __________________________________________________________________________________

    lunghezza2 = min(len( open(path+'Test_'+str(k)+'_FALSE.txt').read().split('\n') ),
                     len( open(path+'Test_'+str(k)+'_TRUE.txt').read().split('\n') )) -1
   # print lunghezza2
    test_arrays = numpy.zeros((lunghezza2 * 2, size*2))
    test_labels = numpy.zeros(lunghezza2 * 2)

    for j in range(lunghezza2):
        prefix_test_pos = 'TEST_POS_' + str(j)
        prefix_test_neg = 'TEST_NEG_' + str(j)

        t_vector= model_title.docvecs[prefix_test_pos]
        a_vector= model_article.docvecs[prefix_test_pos]

        test_arrays[j] = numpy.concatenate(( t_vector , a_vector), axis=0)
        test_arrays[lunghezza2 + j] = numpy.concatenate((model_title.docvecs[prefix_test_neg] , model_article.docvecs[prefix_test_neg]), axis=0)

        test_labels[j] = 1
        test_labels[ lunghezza2 + j] = 0


# CLASSIFICATORE

    classifier = LogisticRegression( C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)
    classifier.fit ( train_arrays, train_labels )

# RESULTS :__________________________________________________________________________________
    score.append( classifier.score(test_arrays, test_labels) )
    print '\nPrecisione test nr.'+str(k+1)+'/3  : ' + str( score[-1])+'\n'

print score
print sum(score)/3


# Similarity Queries: ________________________________________________________________________

#  Dammi i documenti piu simili al Train neg 12
print
#print model.docvecs.most_similar('TRAIN_NEG_12')

# Guarda nel modello le parole  che hai imparato simili a ' clinton' o 'black'
print
#print model.most_similar(StemmingHelper.stem('clinton'))
print
#print model.most_similar(StemmingHelper.stem('black'))


doc_id = numpy.random.randint( lunghezza *2)   # pick random doc, re-run cell for more examples
sims = model_article.docvecs.most_similar(doc_id, topn=model_article.docvecs.count)  # get *all* similar documents

print('Random Target: %s\n' %(str(sentences_article.to_array()[doc_id]).split("', u'"))[-1].split("], [")[-1][:-2] )
testo=""
for s in (str(sentences_article.to_array()[doc_id]).split("TaggedDocument(")[-1]).split("', u'"):
    testo += str(s)+' '
print testo


print('\n\nSIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model_article)
for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    p = sims[index][0].split('_')

    print
    print p

    sentiment = p[1]
    indice = int(p[-1])

    if p[0] == 'TRAIN':
        if sentiment =='POS':
            indice = len( open(path2+'Train_'+str(k)+'_FALSE.txt').read().split('\n')) + indice - 1
    else:
        indice += lunghezza*2
        if sentiment =='POS':
            indice = len( open(path2+'Test_'+str(k)+'_FALSE.txt').read().split('\n')) + indice - 1

    #print indice
    testo=""
    prova = str(sentences_article.to_array()[indice]).split("TaggedDocument(")[-1]
    check=prova[0:2]
    if  check == '[]':
        print "ERROREEEEEEEE"
        #frasi = str(sentences_article.to_array())
        #print frasi
    for s in prova.split("', u'"):

        testo += str(s)+' '

    print testo


    #print('%s %s: %s\n' % (label, sims[index], testo ) )


