local eval = {}
local c = require 'trepl.colorize'
local tablex = require('pl.tablex')

function eval.model_eval(rnn,vocab,reverse_vocab,vis_input)
	local max_tokens = 15
	rnn:evaluate()
	rnn.modules[1].modules[2].modules[1].nfeatures = 1
	local lstm = rnn.modules[1].modules[1].modules[3]
	lstm:forget()
	
	local input = vocab['<go>']
	input = torch.CudaTensor({input}):view(1,-1)
	local softmax = cudnn.SoftMax():cuda()
	local output = {}
	lstm:remember('both')
	while max_tokens > 0 do
		local rnn_out = rnn:forward({input,vis_input}):squeeze():clone()
		local _,ind = softmax:forward(rnn_out):sort(1,true)
		local output_word = reverse_vocab[ind[1]]
		--print(ind[1],output_word,tablex.size(reverse_vocab))
		if output_word ~= nil then
			table.insert(output,output_word)
			if output_word == '<end>' then	break	end
		else
			print(ind[1],output_word,tablex.size(reverse_vocab))
			error('Reverse Table incorrect. Most likely need to fix the zero-padding while doing reverse-vocab matching...')
		end
		input = torch.CudaTensor({ind[1]}):view(1,-1)
		max_tokens  = max_tokens - 1
	end
	lstm:forget()
	table.remove(output,#output)
	print('Model Output : ',table.concat(output,' '))
	collectgarbage()
	return

end

return eval
