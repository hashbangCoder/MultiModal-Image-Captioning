local nn = require('nn')
local image =  require('image')
local paths = require('paths')
local optim = require('optim')
local xlua = require('xlua')
local utils = require('utils.lua')
local models = require('models.lua') 
------------------- CommandLine options ---------------------
local cmd = torch.CmdLine()
cmd:text()
cmd:text('Image Captioning using Multi-Modal RNN-CNN in Torch')
cmd:text()
cmd:text('Options : ')

cmd:option('-test','test_image.jpg','Test Image path')
cmd:option('-im_dir','COCOData/Data/train2014/','Path to train images')
cmd:option('-caption_dir','COCOData/Data/annotations/captions_train2014.json','Path to Train Captions')
cmd:option('-size',256,'Size of output')

cmd:option('-epochs',10,'Number of iteration to train')
cmd:option('-batch_size',20,'#Images per batch')
cmd:option('-embed_dims',300,'Dimensions of word embeddings')
cmd:option('-lr',1e-3,'Learning rate for Adam optimizer')
cmd:option('-beta',0.5,'Beta value for Adam optim')

cmd:option('-output','Output/mRNN.t7','Save output to')
cmd:option('-save_freq',20000,'How frequently to save output ')
cmd:option('-sample_freq',2000,'How frequently to sample output ')
cmd:option('-gpu',1,'GPU ID, Has to be >= 0 (CPU not supported now)')
cmd:option('-log','Logs/','File to log results')
-----------------------------------------------------------------------------------
cmd:text()
local opt = cmd:parse(arg)

print('Loading Image Filenames and Captions... ')
local data_iter = DataLoader.new(opt.epochs,opt.batch_size)
local vocab_size = data_iter.vocab_size
local embed_len = opt.embed_dims
print('Loading the vocabulary set from file and creating reverse hash table...')
local vocab = data_iter:get_vocab()
local reverse_vocab={}
for k,v in pairs(vocab) do
	reverse_vocab[v]=k
end

local backend = ''
if opt.gpu>=0  then
	local cudnn = require('cudnn')
	print('Using cuDNN as backend..')
	local cutorch = require( 'cutorch')
	local cunn = require('cunn')
	backend = 'cudnn'
	cutorch.setDevice(opt.gpu+1)
	cudnn.fastest = true
else 
	error('GPU ID must be >= 0. Wayyyyy tooo slow on CPU')
end
print('Building RNN model...')
local rnn = models.rnn_model(vocab_size,embed_len)
print('Loading pretrained CNN...')
local cnn = models.load_vgg('cudnn')
cnn = cnn:cuda()
rnn = rnn:cuda()
local batch_mod = nn.JoinTable(1):cuda()
--Combine all batch utility funcs into one big func
local function format_batch(captions,images) 
	local batchCap = utils.appendTokens(captions)
	local capInts = {}
	local targetInts = {}
	local vis_out = {}
	for i=1,#batchCap do
		capInts[i] = utils.captionToInts(batchCap[i],vocab)
		capInts[i],targetInts[i] = utils.getTarget(capInts[i])
		vis_out[i] = cnn:forward(images[i]:cuda())
	end
	local capIntsPadded,maxLen = utils.padZero(capInts)
	local targetIntsPadded,maxLen = utils.padZero(targetInts)
	rnn.modules[1].modules[2].modules[1].nfeatures = maxLen

	for i = 1,#batchCap do
		capIntsPadded[i] = torch.CudaTensor(capIntsPadded[i]):view(1,-1)
		targetIntsPadded[i] = torch.CudaTensor(targetIntsPadded[i]):view(1,-1)
	end
	-- TODO : Check batchsize = 1 compatibility in the above loop 
	local batched_vis_inp = batch_mod:forward(vis_out):clone()
	local batched_cap_ints=  batch_mod:forward(capIntsPadded):clone()
	local batched_tgt_ints = batch_mod:forward(targetIntsPadded):clone()
	return batched_vis_inp,batched_cap_ints,batched_tgt_ints
end

local crit = nn.SequencerCriterion(nn.MaskZeroCriterion(nn.CrossEntropyCriterion(),1)):cuda()
local optimMethod = optim.adam
local optimState = {learningRate = opt.lr,beta1 = opt.beta}
local params,gradParameters = rnn:getParameters()

print('Start Training...')
while data_iter.flag == false do
	local batchCap,batchIm = data_iter:getBatch()
	local iter = data_iter:getGlobalIter()
	local cnn_feats,captions,targets = format_batch(batchCap,batchIm)
	local function feval(params)
		collectgarbage()
		gradParameters:zero()
		local rnn_out = rnn:forward({captions,cnn_feats});
		local loss = crit:forward(rnn_out,targets)
		local gradOut = crit:backward(rnn_out,targets)
		rnn:backward(captions,gradOut);
		return loss,gradParameters
	end
	local _,loss = optimMethod(feval,params,optimState)
	if (iter%opt.sample_freq <opt.batch_size) and (iter>opt.batch_size) then
		--eval.model_eval(rnn,vocab,reverse_vocab,vis_out[1])
		--print('Ground Truth : ',table.concat(batchCap[1],' '))
		--rnn:training()
		--rnn.modules[1].modules[2].modules[1].nfeatures = maxLen
	end
	if (iter%200 <opt.batch_size) and (iter>opt.batch_size) then
		print('Iter: ',iter,'Error:',loss[1])
	end
	if (iter%opt.save_freq <opt.batch_size) and (iter>opt.batch_size) then
		torch.save(opt.output..'_'..tostring(iter),rnn:clearState())
	end
end
torch.save(opt.output..'end',transformNet:clearState())

