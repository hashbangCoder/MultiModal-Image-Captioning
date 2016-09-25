local models = {}
local dpnn = require( 'dpnn')
local nn = require( 'nn')
local rnn = require( 'rnn')
local loadcaffe = require('loadcaffe')

-- Load pretrained 16-layer VGG model and freeze layers
function models.load_vgg(backend)
	--local model =  loadcaffe.load('VGG/VGG_ILSVRC_19_layers_deploy.prototxt','VGG/vgg_normalised.caffemodel',backend)
	local base_path = 'VGG/'
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
	local lstm = nn.SeqLSTM(embedLen,1024)
	lstm:maskZero(1)	
	lstm.batchfirst=true
	local lookup = nn.LookupTableMaskZero(vocabSize,embedLen)
	--local rnn = nn.Sequential():add(lookup):add(nn.Dropout(0.5)):add(nn.Transpose({1,2})):add(fastlstm)
	--						   :add(nn.MaskZero(nn.Linear(1024,2048),1)):add(nn.ReLU(true))
	local rnn = nn.Sequential():add(lookup):add(nn.Dropout(0.5)):add(lstm)
							   :add(nn.Sequencer(nn.MaskZero(nn.Linear(1024,2048),1))):add(nn.Sequencer(nn.ReLU(true)))
	
	local rnn_cnn = nn.ParallelTable()
	-- Add the RNN to parallel table
	rnn_cnn:add(rnn)
	--Add the Visual input to parallel table after embedding in multimodal space. Use fc6 layer
	rnn_cnn:add(nn.Sequential():add(nn.Replicate(1))
							   :add(nn.Sequencer(nn.Sequential():add(nn.Linear(4096,2048)):add(nn.ReLU(true)))))
	--Share weights and grads. use dpnn extension to save memory as compared to just share
	local shared_lin = nn.Linear(embedLen,vocabSize)
	shared_lin.bias = false
	shared_lin.weight:set(lookup.weight);
	shared_lin.gradWeight:set(lookup.gradWeight);
	--local model = nn.Sequential():add(rnn_cnn):add(nn.CAddTable()):add(nn.Sequencer(nn.Sequential():add(nn.Dropout(0.25))
	--															  					:add(nn.MaskZero(nn.Linear(2048,embedLen),1))
	--															  					:add(nn.MaskZero(shared_lin,1))))
	local model = nn.Sequential():add(rnn_cnn):add(nn.CAddTable()):add(nn.Sequencer(nn.Sequential():add(nn.Dropout(0.25))
																  					:add(nn.Linear(2048,embedLen))
																  					:add(shared_lin)))


	collectgarbage()
	collectgarbage()
	--print( shared_lin.weight:size(),lookup.weight:size())
	return model
end
return models
