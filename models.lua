local models = {}
require 'dpnn'
require 'nn'
require 'rnn'
local loadcaffe = require('loadcaffe')

-- Load pretrained 16-layer VGG model and freeze layers
function models.load_vgg(backend)
	--local model =  loadcaffe.load('VGG/VGG_ILSVRC_19_layers_deploy.prototxt','VGG/vgg_normalised.caffemodel',backend)
	local base_path = '../RealTimeNeuralStyle/VGG/'
	local model =  loadcaffe.load(paths.concat(base_path,'VGG_ILSVRC_16_layers_deploy.prototxt'),paths.concat(base_path,'VGG_ILSVRC_16_layers.caffemodel'),backend)
	for i=38,#model do
		model:remove()
	end
	--assert(model:get(#model).name == 'relu4_2','VGG Model is loaded incorrectly')
	for i=1,#model do
		model:get(i).accGradParameters = function() end
	end
	return model
end

function models.rnn_model(vocabSize,embedLen)
	local fastlstm = nn.FastLSTM(embedLen,1024)
	fastlstm:maskZero(1)
	local rnn_cnn = nn.ParallelTable()
	local lookup = nn.LookupTableMaskZero(vocabSize,embedLen)
	local rnn = nn.Sequential():add(lookup):add(nn.Dropout(0.5)):add(fastlstm):add(nn.Linear(1024,2048)):add(nn.ReLU(true))
	
	rnn_cnn:add(rnn)
	--Use fc6 layer
	rnn_cnn:add(nn.ReLU(true)):add(nn.Linear(4096,2048))
	--Share weights and grads. use dpnn extension to save memory as compared to just share
	local shared_lin = nn.Linear(embedLen,vocabSize)
	shared_lin.weight:set(lookup.weight:t())
	shared_lin.gradWeight:set(lookup.gradWeight:t())
	local model = nn.Sequential():add(rnn_cnn):add(nn.CAddTable()):add(nn.Dropout(0.25)):add(shared_lin)
	
	collectgarbage()
	collectgarbage()
	return nn.Sequencer(model)


end

return models
