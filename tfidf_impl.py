
import numpy as np
import operator

class TfIdfOperator:

    def __init__(self, docs, max_docs_toread = 20000):
        print('__init__ number of documents: ' + str(len(docs)))
        self.documents = docs
    
    def dump(self, doc):
        print(doc['text'])
        with open('tfidf_result.log', 'a+') as f:
            print("", file=f)
            print(doc['text'], file=f)
            print('-----', file=f)
            listofitems = doc['words_tfidf_list'].items()
            sorted_list = sorted(listofitems, key=lambda x: (x[1]['tfidf'], x[1]['tf']))
            ctr = int(len(sorted_list) / 10)
            for i in range(ctr):
                print(sorted_list[(i+1) * -1][0] + " [" + str(sorted_list[(i+1) * -1][1]['tfidf']) + "]" , file=f)

    def get_allwords_tf(self):
        print('get_allwords_tf...')
        self.all_words = dict()
        for doc in self.documents:
            tokens = doc['text'].split(' ')
            for token in tokens:
                if not self.all_words.__contains__(token):
                    self.all_words[token] = {'tf': 1, 'df': 1, 'flag': doc['id']}
                else:
                    self.all_words[token]['tf'] += 1
                    if(self.all_words[token]['flag'] != doc['id']): 
                        self.all_words[token]['df'] += 1
                        self.all_words[token]['flag'] = doc['id']                                       
        

    def compute_tfidf(self):
        print('compute_tfidf...')
        num_of_docs = len(self.documents)    
        for doc in self.documents:
            # get distinct list of words in current document
            words = doc['text'].split(' ')
            for word in words:
                if not doc['words_tfidf_list'].__contains__(word):
                    doc['words_tfidf_list'][word] = {'tf': 1, 'df': self.all_words[word]['df'] , 'tfidf': 0.0}                
                else:
                    doc['words_tfidf_list'][word]['tf'] += 1                
                    doc['words_tfidf_list'][word]['tfidf'] = doc['words_tfidf_list'][word]['tf'] * np.log(num_of_docs / doc['words_tfidf_list'][word]['df'])


if(__name__ == "__main__"):
    docs = list()
    i = 1
    with open('Datasets\\r8-train-all-terms.txt') as inn:    
        for line in inn.readlines():
            cd = line.split('\t')
            dclass = cd[0]
            docs.append({'text': cd[1], 'class': cd[0], 'words_tfidf_list': {}, 'id': i })
            i+=1
            #if i == 3: break
    
    op = TfIdfOperator(docs)
    op.get_allwords_tf()
    op.compute_tfidf()
           
    for d in op.documents: 
        op.dump(d)
        print('--------------------------------')
    #print(docs)

#print( get_allwords_tf([['Hello', 'world'], ['hello', 'family'], ['world', 'sucks']]) )