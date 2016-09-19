-- Test Code to check correctness of model by overfitting on  a mini-batch
-- used globals to debug. Until I figure out mobdebug over ssh :(

local cutorch = require('cutorch')
local cudnn = require('cudnn')
local cunn = require('cunn')
local nn = require('nn')
local pl = require('pl')
local optim = require 'optim'

print('Using cuDNN as backend..')
cutorch.setDevice(2)
cudnn.fastest = true

local model = require('models.lua')
local utils = require('utils.lua')

print('Loading Image Filenames and Captions... ')
local data_iter = DataLoader.new()

collectgarbage('count')
data_iter.batch_size = 2
print('Loading the vocabulary set from file...')
local vocab = data_iter:get_vocab()
local vocab_size = data_iter.vocab_size
local embed_len = 100

collectgarbage('count')
print('Building RNN model...')
rnn = model.rnn_model(vocab_size,embed_len)
print('Loading pretrained CNN...')
cnn = model.load_vgg('cudnn')
cnn = cnn:cuda()
rnn = rnn:cuda()

local batchCap,batchIm = data_iter:getBatch()
batchCap = utils.appendTokens(batchCap)
rnn.modules[1].modules[2].modules[1].nfeatures = #batchCap
capInts = {}
targetInts = {}
local vis_out = {}
for i=1,#batchCap do
	capInts[i] = utils.captionToInts(batchCap[i],vocab)
end
capInts = utils.padZero(capInts)
for i=1,#batchCap do
	capInts[i],targetInts[i] = utils.getTarget(capInts[i])
	capInts[i] = torch.CudaTensor(capInts[i])
	targetInts[i] = torch.CudaTensor(targetInts[i])
	vis_out[i] = cnn:forward(batchIm[i]:cuda())
end
local batch_mod = nn.JoinTable(1):cuda()
-- TODO : Check batchsize = 1 compatibility in the above loop 

batched_vis_inp = batch_mod:forward(vis_out)
batched_cap_ints=  batch_mod:forward(capInts)
print(#batched_cap_ints)
batched_tgt_ints = batch_mod:forward(targetInts)
-- In this example, it is a mini-batch of size 2

crit = nn.SequencerCriterion(nn.MaskZeroCriterion(nn.CrossEntropyCriterion(),1))
crit = crit:cuda()
--Main loop
for i=1,1000 do
	rnn:zeroGradParameters()
	local rnn_out = rnn:forward(batched_cap_ints)
	local err = crit:forward(rnn_out,batched_tgt_ints)
	print('Error :',err)
	local gradOut = crit:backward(rnn_out,batched_tgt_ints)
	rnn:backward(batched_cap_ints,gradOut)
	rnn:updateParameters(0.00005)
	break
end
