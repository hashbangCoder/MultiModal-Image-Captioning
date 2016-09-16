require 'nn'
local image =  require 'image'
local cudnn = require 'cudnn'
require 'paths'
require 'optim'
require 'xlua'
local utils = require('utils.lua')
local models = require('model.lua') 
------------------- CommandLine options ---------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Real Time Neural Style Transfer in Torch')
cmd:text()
cmd:text('Options : ')

cmd:option('-test','test_image.jpg','Test Image path')
cmd:option('-im_dir','~/RealTimeNeuralStyle/Data/train2014/','Path to train images')
cmd:option('-caption_dir','~/RealTimeNeuralStyle/Data/annotations/captions_train2014.json','Path to Train Captions')
cmd:option('-size',256,'Size of output')

cmd:option('-iter',160000,'Number of iteration to train')
cmd:option('-batch_size',4,'#Images per batch')
cmd:option('-save_freq',20000,'How frequently to save output ')
cmd:option('-saved_params','transformNet.t7','Save output to')
cmd:option('-output','Output/','Save output to')

cmd:option('-lr',1e-3,'Learning rate for optimizer')
cmd:option('-beta',0.5,'Beta value for Adam optim')

cmd:option('-gpu',1,'GPU ID, -1 for CPU (not fully supported now)')
cmd:option('-log','Logs/','File to log results')
-----------------------------------------------------------------------------------

cmd:text()
local opt = cmd:parse(arg)
cmd:log(opt.log .. 'main.log',opt)


if opt.gpu>=0  then
	print('Using cuDNN as backend..')
	require 'cutorch'
	require 'cunn'
	backend = 'cudnn'
	cutorch.setDevice(opt.gpu+1)
	cudnn.fastest = true
else 
	print('GPU ID must be >= 0. Cannot run on CPU')
	sys.exit()

-- Load VGG-16
cnn = models.load_vgg(backend)

print('Loading Image Filenames and Captions... ')
local images,captions = utils.load_data(opt.caption_dir,opt.im_dir)







