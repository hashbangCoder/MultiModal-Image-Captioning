require('rnn')
require('cutorch')
require('cunn')
require('nn')
local model = require('models.lua')
local utils = require('utils.lua')


print('Loading Image Filenames and Captions... ')
local cnn = model.loadvgg('cudnn')
local data_iter = DataLoader.new()

data_iter.batch_size = 1
local batch = data_iter:getBatch()
local vocab

local vocab = data_iter:get_vocab()
local vocab_size = data_iter.vocab_size
local embed_len = 100

cnn = models.load_vgg(backend)
print('Building RNN model...')
rnn = models.rnn_model(vocab_size,embed_len)

batch = data_iter:getBatch()

image = batch[1][1]
caption = batch[1][2]


