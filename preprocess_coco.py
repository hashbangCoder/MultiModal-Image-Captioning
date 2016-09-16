import json
from nltk import word_tokenize as wt
from itertools import izip

# Not optimized in any way. Have to run only once
with open('../RealTimeNeuralStyle/Data/annotations/captions_train2014.json','r') as f:
    data = json.load(f)
print 'Re-tokenizing captions with space delimiter...'
full_str = []
for ind,item in enumerate(data['annotations']):
    full_str.append(item['caption'])
    item['caption'] = ' '.join(wt(item['caption']))
    if ind%10000 == 0:
        print '%f %% done...'%(ind*100/float(len(data['annotations'])))

full_str = ' '.join(full_str)
vocab = list(set(wt(full_str.lower())))
data['vocab_size'] = len(vocab)
_iter = iter(vocab)
b = dict(izip(_iter, _iter))

with open('../RealTimeNeuralStyle/Data/annotations/captions_train2014_pp.json','w') as f:
    json.dump(data,f)

with open('../RealTimeNeuralStyle/Data/annotations/captions_train2014_dict.json','w') as f:
    json.dump(b,f)
