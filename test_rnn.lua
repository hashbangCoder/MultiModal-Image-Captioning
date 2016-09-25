-- Test Code to check correctness of model by overfitting on  a mini-batch. Use simple SGD
-- used globals to debug. Until I figure out mobdebug over ssh :(

local cutorch = require('cutorch')
local cudnn = require('cudnn')
local cunn = require('cunn')
local nn = require('nn')
local pl = require('pl')
local optim = require 'optim'
local eval = require('evaluate.lua')

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
local reverse_vocab={}
for k,v in pairs(vocab) do
	reverse_vocab[v]=k
end

local vocab_size = data_iter.vocab_size
local embed_len = 100

print('Building RNN model...')
rnn = model.rnn_model(vocab_size,embed_len)
print('Loading pretrained CNN...')
local cnn = model.load_vgg('cudnn')
cnn = cnn:cuda()
rnn = rnn:cuda()

local batchCap,batchIm = data_iter:getBatch()
batchCap = utils.appendTokens(batchCap)
local capInts = {}
local targetInts = {}
local vis_out = {}
for i=1,#batchCap do
	capInts[i] = utils.captionToInts(batchCap[i],vocab)
	capInts[i],targetInts[i] = utils.getTarget(capInts[i])
	vis_out[i] = cnn:forward(batchIm[i]:cuda())
end
local capIntsPadded,maxLen = utils.padZero(capInts)
local targetIntsPadded,maxLen = utils.padZero(targetInts)
rnn.modules[1].modules[2].modules[1].nfeatures = maxLen
for i = 1,#batchCap do
	capIntsPadded[i] = torch.CudaTensor(capIntsPadded[i]):view(1,-1)
	targetIntsPadded[i] = torch.CudaTensor(targetIntsPadded[i]):view(1,-1)
end
local batch_mod = nn.JoinTable(1):cuda()
-- TODO : Check batchsize = 1 compatibility in the above loop 

local batched_vis_inp = batch_mod:forward(vis_out):clone()
local batched_cap_ints=  batch_mod:forward(capIntsPadded):clone()
local batched_tgt_ints = batch_mod:forward(targetIntsPadded):clone()

local crit = nn.SequencerCriterion(nn.MaskZeroCriterion(nn.CrossEntropyCriterion(),1))
crit = crit:cuda()
--Main loop
print('Start Training...')
for i=1,100000 do
	rnn:zeroGradParameters()
	local rnn_out = rnn:forward({batched_cap_ints,batched_vis_inp})
	local err = crit:forward(rnn_out,batched_tgt_ints)
	local gradOut = crit:backward(rnn_out,batched_tgt_ints)
	rnn:backward(batched_cap_ints,gradOut)
	rnn:updateParameters(0.00005)
	if i%200 == 0 then
		print('\nIter: ',i,'Error:',err)
		eval.model_eval(rnn,vocab,reverse_vocab,vis_out[1])
		print('Ground Truth : ',table.concat(batchCap[1],' '))
		print('Ground Truth(2): ',table.concat(batchCap[2],' '))
		rnn:training()
		rnn.modules[1].modules[2].modules[1].nfeatures = maxLen
	end
	collectgarbage()
end
