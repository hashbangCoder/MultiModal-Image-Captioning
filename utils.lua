--A simple mini-batch iterator for image,caption pairs
--Implements other utils for image handling

local utils = {}
require('pl')
local cjson = require('cjson')
local DataLoader = torch.class('DataLoader')
local image = require('image')

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



function DataLoader:__init(epcohs,batch_size)
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
	self.vocab_size = data_files.vocab_size
	self.epochs = 5--epochs
	self.batch_size = 1-- batch_size
	self.iterInd = 1
	self:getFullData()
	self:shuffleInds()
end

function DataLoader:get_vocab()
	local _file =io.open(self.vocab_file,'r')
	local vocab = {}
	local _vocab = cjson.decode(_file:read("*a"))
	for i,j in pairs(_vocab) do
		table.insert(vocab,i)
		table.insert(vocab,j)
	end
	_vocab=nil
	collectgarbage()
	return vocab
	--self.vocab = vocab
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
	local batch = {}
	if self.iterInd + self.batch_size > self.shuffle:nElement() then
		for i=self.iterInd,self.shuffle:nElement()-1 do
			local image_id = self.fullData[self.shuffle[self.iterInd]][1]
			local caption =  self.fullData[self.shuffle[self.iterInd]][2]
			local im = image.loadJPG(paths.concat(self.dir_path,self.image_data[image_id]))
			im = utils.pp(utils.scale_pp(im,256))
			--local batch_sample = {im,caption}	
			table.insert(batch,{im,caption})
		end
		if self.epochs > 0 then
			self:shuffleInds()
			self.epochs = self.epochs - 1
			self.iterInd = 1
		else
			print('End of epochs')
		end
	else
		for i=self.iterInd,self.iterInd+self.batch_size-1 do
			local image_id = self.fullData[self.shuffle[self.iterInd]][1]
			local caption = self.fullData[self.shuffle[self.iterInd]][2]
			local im = image.loadJPG(paths.concat(self.dir_path,self.image_data[image_id]))
			im = utils.pp(utils.scale_pp(im,256))
			--local batch_sample = {self.image_data[image_id],caption}	
			self.iterInd = self.iterInd + self.batch_size
			table.insert(batch,{im,caption})
		end
	end
	return batch


end

function DataLoader:shuffleInds()
	self.shuffle = torch.randperm(#self.fullData)

end

function DataLoader:getEpochs()
	return self.epochs
end

function DataLoader:getIterInd()
	return self.iterInd
end

function DataLoader:resetIterInd()
	self.iterInd = 1
end


return utils
