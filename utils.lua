--A simple mini-batch iterator for image,caption pairs
--Implements other utils for image handling

local utils = {}
local cjson = require('cjson')
local DataLoader = torch.class('DataLoader')
local image = require('image')
local tablex = require('pl.tablex')
function utils.pp(im)
	-- Take in RGB and subtract means
	-- If you're wondering what these floats are for, like I did, they're the mean pixel (Blue-Green-Red) for the pretrained VGG net in order to zero-center the input image.
	local means = torch.DoubleTensor({-103.939, -116.779, -123.68})
	--Convert RGB --> BGR and pixel range to 0-255
	im = im:mul(255.0)
	means = means:view(3, 1, 1):expandAs(im)
	-- Subtract means and resize
	im:add(1, means)
	return im
end

function utils.toBGR(im)
	local perm = torch.LongTensor{3, 2, 1}
	--Convert RGB --> BGR and pixel range to 0-255
	im = im:index(1, perm)
	return im

end

function utils.scale_pp(im,size)
	-- Take in grayscale/RGB return RGB-type scaled image
	if im:size(1) ==1 then
		im = torch.cat(im,torch.cat(im,im,1),1)
	elseif im:size(1) == 3 then
	else
		error('Input image is not an RGB/Grayscale image')
	end
	im = image.scale(im,size,size,'bilinear')
	return im
	
end

function utils.dp(im)
	-- Exact inverse of above
	local perm = torch.LongTensor{3, 2, 1}
	im = im:index(1, perm)
	return im:double()
end



function DataLoader:__init(epochs,batch_size)
	local captions = 'COCOData/Data/annotations/captions_train2014_pp.json'
	local im_dir = 'COCOData/Data/train2014'
	-- Cap-data is a map from image_id to table of captions
	-- image_data is a map from image_id to filename
	local _file = io.open(captions,'r')
	local data_files = cjson.decode(_file:read("*a"))
	local aData = data_files.annotations
	local iData = data_files.images
	local cap_data,image_data = {},{}
	local multiple_caption = 0
	for i=1,#aData do
		if cap_data[aData[i].image_id] then
			cap_data[aData[i].image_id][#cap_data[aData[i].image_id] + 1] = aData[i].caption
			multiple_caption = multiple_caption+1
		else 
			cap_data[aData[i].image_id] = {aData[i].caption}
		end
	end
	print('Total Number of multiple captions : ',multiple_caption)
	for i=1,#iData do
		if image_data[iData[i].id] then error('Duplicate IDs found') end
		image_data[iData[i].id] = iData[i].file_name
	end
	self.vocab_file = 'COCOData/Data/annotations/captions_train2014_dict.json'
	self.dir_path = 'COCOData/Data/train2014/'
	self.image_data = image_data
	self.cap_data = cap_data
	self.vocab_size = data_files.vocab_size + 2	--start and end tokens
	self.maxEpochs = epochs
	self.epochs = epochs
	self.flag = false
	self.batch_size = batch_size
	self.iterInd = 1
	self.global_iter = 1
	self:getFullData()
	self:shuffleInds()
end

function DataLoader:get_vocab()
	local _file =io.open(self.vocab_file,'r')
	local vocab = {}
	local _vocab = cjson.decode(_file:read("*a"))
	local cnt = 1
	--for i,j in pairs(_vocab) do
	--	vocab[j] = cnt
	--	cnt = cnt + 1
	--	vocab[i] = cnt 
	--	cnt = cnt+1
	--end
	---- Add start and end tokens
	_vocab['<go>']=tablex.size(_vocab) + 1
	_vocab['<end>'] = tablex.size(_vocab) + 1
	--_vocab = nil
	--collectgarbage();
	return _vocab
end

function DataLoader:getFullData()
	--fullData is a table where each entry is {id,caption}. There are multiple entries for same id but with different caption
	local fullData = {}
	for i,j in pairs(self.cap_data) do
		if #j > 1 then
			for k=1,#j do
				table.insert(fullData,{i,j[k]})
			end
		else
			table.insert(fullData,{i,j})
		end
	end
	self.cap_data = nil
	collectgarbage()
	self.fullData = fullData
end


function DataLoader:getBatch()
	collectgarbage()
	if self.flag == true then
		error('Maximum Epoch Limit reached')
		return 
	end
	local batchIm, batchCap = {},{}
	if self.iterInd + self.batch_size > self.totalSamples then
		for i=self.iterInd,self.totalSamples-1 do
			local image_id = self.fullData[self.shuffle[self.iterInd]][1]
			local caption =  self.fullData[self.shuffle[self.iterInd]][2]
			local im = image.loadJPG(paths.concat(self.dir_path,self.image_data[image_id]))
			im = utils.pp(utils.scale_pp(im,224))
			table.insert(batchIm,im)
			table.insert(batchCap,caption:lower():split(' '))
		end
		self.iterInd = self.totalSamples
		if self.epochs > 0 then
			self:shuffleInds()
			self.epochs = self.epochs - 1
			if self.epochs ==0 then self.flag = true end
			self.iterInd = 1
		end
	else
		for i=self.iterInd,self.iterInd+self.batch_size-1 do
			local image_id = self.fullData[self.shuffle[self.iterInd]][1]
			local caption = self.fullData[self.shuffle[self.iterInd]][2]
			local im = image.loadJPG(paths.concat(self.dir_path,self.image_data[image_id]))
			im = utils.pp(utils.scale_pp(im,224))
			im = im:view(1,3,224,224)
			table.insert(batchIm,im)
			table.insert(batchCap,caption:lower():split(' '))
		end
		self.iterInd = self.iterInd + self.batch_size
	end
	return batchCap,batchIm
end

function DataLoader:shuffleInds()
	self.shuffle = torch.randperm(#self.fullData)
	self.totalSamples = self.shuffle:nElement()
end

function DataLoader:getGlobalIter()
	return ((self.maxEpochs - self.epochs)*self.totalSamples + self.iterInd)
end

------------------- Other Utility Functions -----------------
-- Input is table minibatch of integers
function utils.padZero(inputTable)
	local maxLen = 0
	for _,j in ipairs(inputTable) do
		if #j >= maxLen then
			maxLen = #j
		end
	end
	for _,j in ipairs(inputTable) do
		while #j < maxLen do
			table.insert(j,0)
		end
	end
	return inputTable,maxLen
end

function utils.appendTokens(inputTable)
	for _,j in pairs(inputTable) do
		j[#j + 1] = '<end>'
		table.insert(j,1,'<go>')
	end
	return inputTable
end

--input is a table of integers
function utils.getTarget(input)
	target = tablex.copy(input)
	table.remove(target,1)
	table.remove(input,#input)
	return input,target
end

function utils.captionToInts(caption,vocab)
	local captionInteger = {}
	for i=1,#caption do
		local index = vocab[caption[i]]
		if index ~= nil then
			table.insert(captionInteger,index)
		else 
			print(caption[i],vocab[caption[i]])
			error('Unknown word')
		end
	end
	return captionInteger
end

return utils
