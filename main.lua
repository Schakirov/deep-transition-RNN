--[[
Trains a word-level or character-level (for inputs) lstm language model
Predictions are still made at the word-level.

Much of the code is borrowed from the following implementations
https://github.com/karpathy/char-rnn
https://github.com/wojzaremba/lstm
]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'util.Squeeze'
require 'util.misc'

BatchLoader = require 'util.BatchLoaderUnk'
model_utils = require 'util.model_utils'

local stringx = require('pl.stringx')
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a word+character-level language model')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data_dir','data/ptb','data directory. Should contain train.txt/valid.txt/test.txt with input data')
-- model params
cmd:option('-ndt', 1, 'deep-transition depth (1=commonLSTM)')
cmd:option('-ndt2', 1, 'deep-transition depth in seq2seq (1=common LSTM)')
cmd:option('-model','lstm', 'lstm or rnn')
cmd:option('-deepOut', 0, 'number of extra deepOutput layers')
cmd:option('-deepOutdim', '{}', 'dimensions of deepOutput')
cmd:option('-deepOutDropout',0.5,'dropout of DO')
cmd:option('-deepOutNonLin','','nonlinearity used in deepOut')
cmd:option('-up2down',0,'connection from uppest num_layer to lowest num_layer')
cmd:option('-validationFirst',0,'to begin evaluation on validation test immediately')
cmd:option('-unk','ъъъъъ','<unk> (ъъъъъ)')
cmd:option('-forgive',0,'if target word guessed within _forgive_ tries then it is placed in guessed_phrase as if guessed')
cmd:option('-showSentences',0,'show sentences each iteration')
cmd:option('-showUnk',1,'if 0 then we replace <unk> in showSentences with next probable')
-- cmd:option('-phrase_initial','. . . . . . . . . i go to see if you want to see me , i want you to see me , we can have a good day if you','nc')
-- cmd:option('-phrase_initial','it is summer . a boat sails on the river . i go to the river to swim . i see blue sky . the weather is hot . water is warm . i want','nc') -- 59mb10k
cmd:option('-phrase_initial','на улице лето . по реке плывет лодка . я иду к реке , чтобы купаться . на улице светит яркое солнце . поют птицы . на улице жарко . вода теплая . я хочу','nc') -- krapivin10k
-- cmd:option('-phrase_initial','a boeing spokesman explained that the company has been in constant communication with all of its customers and that it was impossible to predict what further disruptions might be triggered by the strike meanwhile supervisors','nc') -- ptb
cmd:option('-genText',0,'to generate text based on -phrase_initial and -checkpoint')
cmd:option('-genTextNoise',0,'noise when generating text to get out of repeating word cycles')
cmd:option('-ignoreTrainSet',0,'not load/reshape trainSet when genText/validationFirst')
cmd:option('-ignoreTestSet',0,'not check test performance')
cmd:option('-rnn_size', 650, 'size of LSTM internal state')
cmd:option('-use_words', 0, 'use words (1=yes)')
cmd:option('-use_chars', 1, 'use characters (1=yes)')
cmd:option('-seq3seq', 0, 'hierarchical deep Sentence')
cmd:option('-mbp',-1,'first linear module to backprop gradients in seq3seq; default -1 is supposed to cause error')
cmd:option('-highway_layers', 2, 'number of highway layers')
cmd:option('-word_vec_size', 650, 'dimensionality of word embeddings')
cmd:option('-char_vec_size', 15, 'dimensionality of character embeddings')
cmd:option('-feature_maps', '{50,100,150,200,200,200,200}', 'number of feature maps in the CNN')
cmd:option('-kernels', '{1,2,3,4,5,6,7}', 'conv net kernel widths')
cmd:option('-num_layers', 2, 'number of layers in the LSTM')
cmd:option('-dropout',0.5,'dropout. 0 = no dropout')
-- optimization
cmd:option('-hsm',0,'number of clusters to use for hsm. 0 = normal softmax, -1 = use sqrt(|V|)')
cmd:option('-learning_rate',1,'starting learning rate')
cmd:option('-learning_rate_decay',0.5,'learning rate decay')
cmd:option('-lr_decay_continuous',1,'decay after each print_every iterations')
cmd:option('-weight_decay',1,'weight decay (1 - no weight decay)')
cmd:option('-lr_min',0.001,'minimal learning rate')
cmd:option('-decay_when',1,'decay if validation perplexity does not improve by more than this much')
cmd:option('-param_init', 0.05, 'initialize parameters at')
cmd:option('-use_param_init',1,'whether to initialize all params with the same pdf')
cmd:option('-batch_norm', 0, 'use batch normalization over input embeddings (1=yes)')
cmd:option('-seq_length',35,'number of timesteps to unroll for')
cmd:option('-batch_size',20,'number of sequences to train on in parallel')
cmd:option('-max_epochs',250,'number of full passes through the training data')
cmd:option('-max_grad_norm',5,'normalize gradients at') -- is it the same as grad_clip?
cmd:option('-min_grad_norm',0,'normalize gradients at') -- is it the same as grad_clip?
cmd:option('-max_word_l',50,'maximum word length')
cmd:option('-logFile','log.txt','name of log file (log.txt by default)')
cmd:option('-threads', 16, 'number of threads') 
-- bookkeeping
cmd:option('-seed',3435,'torch manual random number generator seed')
cmd:option('-print_every',100,'how many steps/minibatches between printing out the loss')
cmd:option('-save_every', 1, 'save every n epochs')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-savefile','char','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-checkpoint', 'checkpoint.t7', 'start from a checkpoint if a valid checkpoint.t7 file is given')
cmd:option('-EOS', '', '<EOS> symbol. should be a single unused character (like +) for PTB and blank for others')
-- GPU/CPU
cmd:option('-gpuid', -1,'which gpu to use. -1 = use CPU')
cmd:option('-cudnn', 0,'use cudnn (1=yes). this should greatly speed up convolutions')
cmd:option('-time', 1, 'print batch times')
cmd:option('-padding',0,'padding')
cmd:text()
-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

assert(opt.use_words == 1 or opt.use_words == 0, '-use_words has to be 0 or 1')
assert(opt.use_chars == 1 or opt.use_chars == 0, '-use_chars has to be 0 or 1')
assert((opt.use_chars + opt.use_words) > 0, 'has to use at least one of words or chars')

--if opt.threads > 0 then
--    torch.setnumthreads(opt.threads)
--end

-- some housekeeping
loadstring('opt.kernels = ' .. opt.kernels)() -- get kernel sizes
loadstring('opt.feature_maps = ' .. opt.feature_maps)() -- get feature map sizes
loadstring('opt.deepOutdim = ' .. opt.deepOutdim)() -- get deepOut dimensions

opt.padding = 0 

-- global constants for certain tokens
opt.tokens = {}
opt.tokens.EOS = opt.EOS
opt.tokens.UNK = '|' -- unk word token
opt.tokens.START = '{' -- start-of-word token
opt.tokens.END = '}' -- end-of-word token
opt.tokens.ZEROPAD = ' ' -- zero-pad token 

-- load necessary packages depending on config options
if opt.gpuid >= 0 then
  print('using CUDA on GPU ' .. opt.gpuid .. '...')
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(opt.gpuid + 1)
end

if opt.cudnn == 1 then
  assert(opt.gpuid >= 0, 'GPU must be used if using cudnn')
  print('using cudnn...')
  require 'cudnn'
end

-- create the data loader class
loader = BatchLoader.create(opt.data_dir, opt.batch_size, opt.seq_length, opt.padding, opt.max_word_l, opt.ignoreTrainSet)
print('Word vocab size: ' .. #loader.idx2word .. ', Char vocab size: ' .. #loader.idx2char
  .. ', Max word length (incl. padding): ', loader.max_word_l)
opt.max_word_l = loader.max_word_l

-- if number of clusters is not explicitly provided
if opt.hsm == -1 then
  opt.hsm = torch.round(torch.sqrt(#loader.idx2word))
end

if opt.hsm > 0 then
  -- partition into opt.hsm clusters
  -- we want roughly equal number of words in each cluster
  HSMClass = require 'util.HSMClass'
  require 'util.HLogSoftMax'
  mapping = torch.LongTensor(#loader.idx2word, 2):zero()
  local n_in_each_cluster = #loader.idx2word / opt.hsm
  local _, idx = torch.sort(torch.randn(#loader.idx2word), 1, true)   
  local n_in_cluster = {} --number of tokens in each cluster
  local c = 1
  for i = 1, idx:size(1) do
    local word_idx = idx[i] 
    if n_in_cluster[c] == nil then
      n_in_cluster[c] = 1
    else
      n_in_cluster[c] = n_in_cluster[c] + 1
    end
    mapping[word_idx][1] = c
    mapping[word_idx][2] = n_in_cluster[c]        
    if n_in_cluster[c] >= n_in_each_cluster then
      c = c+1
    end
    if c > opt.hsm then --take care of some corner cases
      c = opt.hsm
    end
  end
  print(string.format('using hierarchical softmax with %d classes', opt.hsm))
end


-- load model objects. we do this here because of cudnn and hsm options
TDNN = require 'model.TDNN'
TDNN2 = require 'model.TDNN2'
LSTMTDNNdt = require 'model.LSTMTDNNdt'
RNNdt = require 'model.RNNdt'
HighwayMLP = require 'model.HighwayMLP'

-- make sure output directory exists
if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end

if path.exists(opt.checkpoint) then -- start re-training from a checkpoint
  print('loading ' .. opt.checkpoint .. ' for retraining')
  checkpoint = torch.load(opt.checkpoint)
--  local opt2 = opt
---  opt = checkpoint.opt
---  print(opt)
  opt = cmd:parse(arg)
  for k, v in pairs(checkpoint.opt) do
   opt[k] = v
  end
  for k, v in pairs(arg) do
    if k%2 == 1 and k > 0 then 
      local m = string.sub(arg[k],2,-1)
      local g = arg[k+1]
      if tonumber(g) == nil then g = '\'' .. g .. '\'' end
      loadstring('opt.' .. m .. ' = ' .. g)()
    end
  end
  print('some of opt are loaded from checkpoint, some others from command line; check main.lua for more clarity')
  --[[
  opt.data_dir = opt2.data_dir
  opt.dropout = opt2.dropout
  opt.learning_rate = opt2.learning_rate
  opt.learning_rate_decay = opt2.learning_rate_decay
  opt.decay_when = opt2.decay_when
  opt.batch_norm = opt2.batch_norm
  opt.seq_length = opt2.seq_length
  opt.batch_size = opt2.batch_size
  opt.max_epochs = opt2.max_epochs
  opt.max_grad_norm = opt2.max_grad_norm
  opt.threads = opt2.threads
  opt.seed = opt2.seed
  opt.print_every = opt2.print_every
  opt.save_every = opt2.save_every
  opt.gpuid = opt2.gpuid
  opt.cudnn = opt2.cudnn
  opt.lr_decay_continuous = opt2.lr_decay_continuous
  opt.showSentences = opt2.showSentences
  opt.validationFirst = opt2.validationFirst
  opt.unk = opt2.unk
  opt.forgive = opt2.forgive
  opt.phrase_initial = opt2.phrase_initial
  opt.genText = opt2.genText --]]
  retrain = true
end

-- define the model: prototypes for one timestep, then clone them in time
protos = {}
print('creating an LSTM-CNN with ' .. opt.num_layers .. ' layers')
if retrain then
  protos = checkpoint.protos
else
  if opt.model == 'rnn' then
    protos.rnn = RNNdt.rnndt(opt.rnn_size, opt.num_layers, opt.ndt, opt.dropout, #loader.idx2word, 
      opt.word_vec_size, #loader.idx2char, opt.char_vec_size, opt.feature_maps, 
      opt.kernels, loader.max_word_l, opt.use_words, opt.use_chars, opt.batch_norm,opt.highway_layers, opt.hsm, opt.deepOut, opt.deepOutDropout)
  elseif opt.model == 'lstm' then
    protos.rnn = LSTMTDNNdt.lstmtdnndt(opt.rnn_size, opt.num_layers, opt.ndt, opt.dropout, #loader.idx2word, 
      opt.word_vec_size, #loader.idx2char, opt.char_vec_size, opt.feature_maps, 
      opt.kernels, loader.max_word_l, opt.use_words, opt.use_chars, opt.batch_norm,opt.highway_layers, opt.hsm, opt.deepOut, opt.deepOutdim, opt.deepOutDropout, opt.deepOutNonLin, opt.up2down, opt.seq3seq)
  end
 -- print(protos.rnn)

  -- training criterion (negative log likelihood)
  if opt.hsm > 0 then
    protos.criterion = nn.HLogSoftMax(mapping, opt.rnn_size)
  else
    protos.criterion = nn.ClassNLLCriterion() --ClassNLLCriterion
  end
end

-- the initial state of the cell/hidden states
init_state = {}
for L=1,opt.num_layers do
  local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
  if opt.gpuid >=0 then h_init = h_init:cuda() end
  if opt.model == 'lstm' then
    table.insert(init_state, h_init:clone())
    table.insert(init_state, h_init:clone())
  elseif opt.model == 'rnn' then
    table.insert(init_state, h_init:clone())
  end
end
local init_state_global = clone_list(init_state)
local init_state_global2 = clone_list(init_state)
local init_state_global3 = clone_list(init_state)

-- ship the model to the GPU if desired
if opt.gpuid >= 0 then
  for k,v in pairs(protos) do v:cuda() end
end
--print(protos.rnn)
-- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(protos.rnn)
-- hsm has its own params
if opt.hsm > 0 then
  hsm_params, hsm_grad_params = protos.criterion:getParameters()
  hsm_params:uniform(-opt.param_init, opt.param_init)
  print('number of parameters in the model: ' .. params:nElement() + hsm_params:nElement())
else
  print('number of parameters in the model: ' .. params:nElement())
end

-- initialization
if not retrain then
  if opt.use_param_init > 0 then
    params:uniform(-opt.param_init, opt.param_init) -- small numbers uniform if starting from scratch
  end
end


-- get layers which will be referenced layer (during SGD or introspection)
function get_layer(layer)
  local tn = torch.typename(layer)
  if layer.name ~= nil then
    if layer.name == 'word_vecs' then
      word_vecs = layer
    elseif layer.name == 'char_vecs' then
      char_vecs = layer
    elseif layer.name == 'cnn' then
      cnn = layer
    end
  end
end 
protos.rnn:apply(get_layer)

-- make a bunch of clones after flattening, as that reallocates memory
-- not really sure how this part works
clones = {}
for name,proto in pairs(protos) do
  print('cloning ' .. name)
  clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
end

file = io.open (opt.logFile,'a')
for i,module in ipairs(protos.rnn:listModules()) do
   file:write(tostring(module) .. '\n')
end
file:close()
      
-- for easy switch between using words/chars (or both)
function get_input(x, x_char, t, prev_states)
  local u = {}
  if opt.use_chars == 1 then table.insert(u, x_char[{{},t}]) end
  if opt.use_words == 1 then table.insert(u, x[{{},t}]) end
  for i = 1, #prev_states do table.insert(u, prev_states[i]) end
  return u
end

-- for easy switch between using words/chars (or both)
function get_input2(input2, t, prev_states)
  local u = {}
  table.insert(u, input2[t])
  for i = 1, #prev_states do table.insert(u, prev_states[i]) end
  return u
end

if opt.seq3seq == 1 then -- seq2seq
    LSTMTDNNdtSeq = require 'model.LSTMTDNNdtSeq'
    opt.output_size = #loader.idx2word --opt.rnn_size
    opt.input_size = opt.rnn_size
    protos2 = {}
    protos2.rnn = LSTMTDNNdtSeq.lstmtdnndtseq(opt.rnn_size, opt.num_layers, opt.ndt2, opt.dropout, opt.input_size, opt.output_size, 
      opt.batch_norm, opt.highway_layers, opt.hsm, opt.deepOut, opt.deepOutdim, opt.deepOutDropout, opt.deepOutNonLin, opt.up2down)
      
    -- the initial state of the cell/hidden states
    init_state = {}
    for L=1,opt.num_layers do
      local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
      if opt.gpuid >=0 then h_init = h_init:cuda() end
      if opt.model == 'lstm' then
	table.insert(init_state, h_init:clone()) -- for c
	table.insert(init_state, h_init:clone()) -- for h
      elseif opt.model == 'rnn' then
	table.insert(init_state, h_init:clone())
      end
    end
    local init_state_global = clone_list(init_state)

    -- ship the model to the GPU if desired
    if opt.gpuid >= 0 then
      for k,v in pairs(protos2) do v:cuda() end
    end
    
    -- put the above things into one flattened parameters tensor
    params2, grad_params2 = model_utils.combine_all_parameters(protos2.rnn)

    if not retrain then
      if opt.use_param_init > 0 then
	params2:uniform(-opt.param_init, opt.param_init) -- small numbers uniform if starting from scratch
      end
    end

    -- make a bunch of clones after flattening, as that reallocates memory
    -- not really sure how this part works
    clones2 = {}
    for name,proto in pairs(protos2) do
      print('cloning ' .. name)
      clones2[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters2)
    end
end

if opt.seq3seq == 1 then -- decoder
  LSTMTDNNdtDec = require 'model.LSTMTDNNdtDec'
  opt.input_size = #loader.idx2word --opt.rnn_size
  protos3 = {}
  protos3.rnn = LSTMTDNNdtDec.lstmtdnndtdec(opt.rnn_size, opt.num_layers, opt.ndt, opt.dropout, opt.input_size, #loader.idx2word,  
    opt.batch_norm, opt.highway_layers, opt.hsm, opt.deepOut, opt.deepOutdim, opt.deepOutDropout, opt.deepOutNonLin, opt.up2down)
    
    -- training criterion (negative log likelihood)
    if opt.hsm > 0 then
      protos3.criterion = nn.HLogSoftMax(mapping, opt.rnn_size)
    else
      protos3.criterion = nn.ClassNLLCriterion() --ClassNLLCriterion
    end
      
    -- the initial state of the cell/hidden states
    init_state = {}
    for L=1,opt.num_layers do
      local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
      if opt.gpuid >=0 then h_init = h_init:cuda() end
      if opt.model == 'lstm' then
	table.insert(init_state, h_init:clone()) -- for c
	table.insert(init_state, h_init:clone()) -- for h
      elseif opt.model == 'rnn' then
	table.insert(init_state, h_init:clone())
      end
    end
    local init_state_global = clone_list(init_state)

    -- ship the model to the GPU if desired
    if opt.gpuid >= 0 then
      for k,v in pairs(protos3) do v:cuda() end
    end
    
    -- put the above things into one flattened parameters tensor
    params3, grad_params3 = model_utils.combine_all_parameters(protos3.rnn)
    
    if opt.hsm > 0 then
      hsm_params, hsm_grad_params = protos3.criterion:getParameters()
      hsm_params:uniform(-opt.param_init, opt.param_init)
      print('number of parameters in the model: ' .. params:nElement() + hsm_params:nElement())
    else
      print('number of parameters in the model: ' .. params:nElement())
    end

    if not retrain then
      if opt.use_param_init > 0 then
	params3:uniform(-opt.param_init, opt.param_init) -- small numbers uniform if starting from scratch
      end
    end

    -- make a bunch of clones after flattening, as that reallocates memory
    -- not really sure how this part works
    clones3 = {}
    for name,proto in pairs(protos3) do
      print('cloning ' .. name)
      clones3[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters3)
    end
end

function generate_text(phrase_initial, num_of_words_to_be_generated, genTextNoise)
local cumMeanGeomPplx = 1
local phrase_generated = phrase_initial
  print('generating text from: ' .. phrase_initial)
--  opt.batch_size = 1; -------------------------------- may cause problems; then restore it back after generating text if you wish
  local rnn_state = {[0] = init_state}
  for i = 1,num_of_words_to_be_generated do -- iterate over batches in the split
    -- fetch a batch
      local x = {}
      local x_char = {} -- x: 20x35, y: 20x35, x_char: 20x35x36
      for q = 1,opt.seq_length do x_char[q] = {} end
      for rword in phrase_initial:gmatch'([^%s]+)' do
	x[#x+1] = loader.word2idx[rword]
	for char in rword:gmatch'.' do
	  x_char[#x][#x_char[#x]+1] = loader.char2idx[char]
	end
	for i = 1, opt.padding do
        table.insert(x_char[#x], 1, 1) -- 1 is always char idx for zero pad 
	end
	while #x_char[#x] < opt.max_word_l do
        table.insert(x_char[#x], 1)
	end	
      end
	-- x_char[#x] = torch.LongTensor(x_char[#x]):sub(1, opt.max_word_l)
      local x_temp = {}
      local x_char_temp = {}
--      print('x')
--     print({x})
--      print('i = ' .. i)
 --     print('x[1]')
--      print({x[1]})
      for temp = 1,opt.batch_size do
      x_temp[temp] = x;
      x_char_temp[temp] = x_char;
      end
--      print('x_temp')
--      print(x_temp)
----      print('{x_temp}')
--      print(phrase_initial)
 ---     print({x_temp})
   --   print(x_char)
      x = torch.Tensor(x_temp) 
      x_char = torch.Tensor(x_char_temp) -- if error here then very probably some words were not in vocab; try print(x_char)
					 -- num of array with irregularity = 1 + num of word which isn't in vocab
      --[[phrase_initial = '. . . . . . . . . i go to see if you want to see me , i want you to see me , we can have a good day if you'; x = {}; x_char = {}; for q=1,35 do x_char[q]={} end; for rword in phrase_initial:gmatch'([^%s]+)' do x[#x+1] = 17; for char in rword:gmatch'.' do x_char[#x][#x_char[#x]+1] = 7; end; for i = 1,0 do table.insert(x_char[#x], 1, 1); end; while #x_char[#x] < 36 do table.insert(x_char[#x], 1); end; end; x = torch.Tensor(x); x_char = torch.Tensor(x_char); 
      {x_char} --]]
      if opt.gpuid >= 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        x = x:float():cuda()
        x_char = x_char:float():cuda()
      end
      -- forward pass
      local rnn_state = {[0] = init_state_global}
      local predictions = {}           -- softmax outputs
 --     local loss = 0
      local phrase_guessed = ''
      local phrase_guessed_add = ''
      local phrase_target = ''
      local phrase_guess_priority = ''
      for t=1,opt.seq_length do
        clones.rnn[t]:evaluate() -- for dropout proper functioning
--	print({x})
--	print({x_char})
--	print(t)
--	print({rnn_state[t-1]})
--	print({get_input(x, x_char, t, rnn_state[t-1])})
--	print({clones.rnn[t]})
        local lst = clones.rnn[t]:forward(get_input(x, x_char, t, rnn_state[t-1]))
        rnn_state[t] = {}
        for i=1,#init_state do 
          table.insert(rnn_state[t], lst[i])
        end
        predictions[t] = lst[#lst]
	-- print({predictions[t]})
	if opt.showSentences > 0 then
	  local tt = #predictions
	--  local targets = y[{{}, tt}]
	  local pred_max, index_pred_max = torch.max(predictions[tt],2)
	  local pred_sorted, index_pred_sorted = torch.sort(predictions[tt],2,true) --descending sort
	  if (t == 10) then collectgarbage() end
	  local smth857, guess_priority = torch.sort(index_pred_sorted,2,false) --ascending sort
	  phrase_guessed_add = loader.idx2word[index_pred_max[1][1]]
--	  print('pga = ')
--	  print(phrase_guessed_add)
	  local pga = phrase_guessed_add
	  local numPred = 1
--	  while (pga=='.')or(pga==',')or(pga=='!')or(pga=='?')or(pga=='и')or(pga==opt.unk) do
--	  pga = loader.idx2word[index_pred_sorted[1][numPred]]
--	  numPred = numPred + 1
--	  end
	  phrase_guessed_add = pga
	--  if guess_priority[1][targets[1]] < 1 + opt.forgive then phrase_guessed_add = loader.idx2word[targets[1]] end
	  if genTextNoise > 0 then 
	    phrase_guessed_add = loader.idx2word[index_pred_sorted[1][1 + torch.floor(10 - torch.log(torch.uniform(-1+torch.pow(genTextNoise,10)))/torch.log(genTextNoise))]]  
	    --genTextNoise = 2; a = {0,0,0,0,0,0,0,0,0,0}; for i=1,1000 do rn = 1 + torch.floor(10 - torch.log(torch.uniform(-1+torch.pow(genTextNoise,10)))/torch.log(genTextNoise)); a[rn] = a[rn] + 1; end; print(a) -- for statistics of noise
	  end 
	  if opt.showUnk == 0 then
	    local num = 1
	    while phrase_guessed_add == opt.unk do
	      phrase_guessed_add = loader.idx2word[index_pred_sorted[1][num]]
	      num = num + 1
	    end
	  end
	  phrase_guessed = phrase_guessed .. ' ' .. phrase_guessed_add
	--  phrase_target = phrase_target .. ' ' .. loader.idx2word[targets[1]]
	--  phrase_guess_priority = phrase_guess_priority .. ' ' .. guess_priority[1][targets[1]]
	--  cumMeanGeomPplx = cumMeanGeomPplx * 0.99 + math.log(guess_priority[1][targets[1]])
	end
      --  loss = loss + clones.criterion[t]:forward(predictions[t], y[{{}, t}])
      end
      -- carry over lstm state
      rnn_state[0] = rnn_state[#rnn_state]
 --     print('phrase_guessed_add = ')
--      print(phrase_guessed_add)
      phrase_initial = phrase_initial .. ' ' .. phrase_guessed_add
      phrase_generated = phrase_generated .. ' ' .. phrase_guessed_add
      phrase_initial = string.sub(phrase_initial, 1+string.find(phrase_initial,' ',1),-1) --deleting first word
      
      if (i%20 == 0) or (i == num_of_words_to_be_generated) then
      print(' ')
      print(phrase_generated)
  --    print(' ')
  --    print(phrase_target)
  --    print(phrase_guess_priority)
      -- print(math.exp(cumMeanGeomPplx/100)) -- 100 = 1/(1-0.99)
      end
  end
end

-------------------------------------------------------
-- evaluate the loss over an entire split
local cumMeanGeomPplx = 1
function eval_split(split_idx, max_batches)
  print('evaluating loss over split index ' .. split_idx)
  local n = loader.split_sizes[split_idx]
  if opt.hsm > 0 then
    protos.criterion:change_bias()
  end

  if max_batches ~= nil then n = math.min(max_batches, n) end

  loader:reset_batch_pointer(split_idx) -- move batch iteration pointer for this split to front
  local loss = 0
  local rnn_state = {[0] = init_state}    
  if split_idx<=2 then -- batch eval        
    for i = 1,n do -- iterate over batches in the split
      -- fetch a batch
      local x, y, x_char = loader:next_batch(split_idx)
      if opt.use_chars == 0 then x_char = torch.Tensor{1,2} end
      if opt.gpuid >= 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        x = x:float():cuda()
        y = y:float():cuda()
        x_char = x_char:float():cuda()
      end
      -- forward pass
      local rnn_state = {[0] = init_state_global}
      local predictions = {}           -- softmax outputs
      local phrase_guessed = ''
      local phrase_guessed_add = ''
      local pg9 = {}
      local pt9 = {}
      local pgp9 = {}
      local phrase_target = ''
      local phrase_guess_priority = ''
      for t=1,opt.seq_length do
        clones.rnn[t]:evaluate() -- for dropout proper functioning
        local lst = clones.rnn[t]:forward(get_input(x, x_char, t, rnn_state[t-1]))
        rnn_state[t] = {}
        for i=1,#init_state do 
          table.insert(rnn_state[t], lst[i])
        end
        predictions[t] = lst[#lst]
	-- print({predictions[t]})
	if opt.showSentences > 0 then
	  local tt = #predictions
	  local targets = y[{{}, tt}]
	  local pred_max, index_pred_max = torch.max(predictions[tt],2)
	  local pred_sorted, index_pred_sorted = torch.sort(predictions[tt],2,true) --descending sort
	  local smth857, guess_priority = torch.sort(index_pred_sorted,2,false) --ascending sort
	  phrase_guessed_add = loader.idx2word[index_pred_max[1][1]]
	  local pga = phrase_guessed_add
	  local numPred = 1
--	  while (pga=='.')or(pga==',')or(pga=='!')or(pga=='?')or(pga=='и')or(pga==opt.unk) do
--	  pga = loader.idx2word[index_pred_sorted[1][numPred]]
--	  numPred = numPred + 1
--	  end
	  phrase_guessed_add = pga
	  if guess_priority[1][targets[1]] < 1 + opt.forgive then phrase_guessed_add = loader.idx2word[targets[1]] end
	  if opt.showUnk == 0 then
	    if phrase_guessed_add == opt.unk then phrase_guessed_add = loader.idx2word[index_pred_sorted[1][2]] end --------------------- unk
	  end
	  phrase_guessed = phrase_guessed .. ' ' .. phrase_guessed_add
	  phrase_target = phrase_target .. ' ' .. loader.idx2word[targets[1]]
	  phrase_guess_priority = phrase_guess_priority .. ' ' .. guess_priority[1][targets[1]]
	  cumMeanGeomPplx = cumMeanGeomPplx * 0.99 + math.log(guess_priority[1][targets[1]])
	  pg9[t] = phrase_guessed_add; pt9[t] = loader.idx2word[targets[1]]; pgp9[t] = tostring(guess_priority[1][targets[1]]);
	end
        loss = loss + clones.criterion[t]:forward(predictions[t], y[{{}, t}])
      end
      -- carry over lstm state
      rnn_state[0] = rnn_state[#rnn_state]
      if opt.showSentences > 0 then
	phrase_guessed = ''
	phrase_target = ''
	phrase_guess_priority = '';
	for k = 1,opt.seq_length do
	  local str_utf_pg9, len_not_utf_pg9 = string.gsub(pg9[k],"[^\128-\255]","")
	  local str_utf_pt9, len_not_utf_pt9 = string.gsub(pt9[k],"[^\128-\255]","")
	  local pg9_visible_len = #str_utf_pg9/2 + len_not_utf_pg9
	  local pt9_visible_len = #str_utf_pt9/2 + len_not_utf_pt9
	  local max_len = 2 + math.max(pg9_visible_len, pt9_visible_len, #pgp9[k])
	  phrase_guessed = phrase_guessed .. pg9[k] .. string.rep(' ', max_len - pg9_visible_len)
	  phrase_target = phrase_target .. pt9[k] .. string.rep(' ', max_len - pt9_visible_len)
	  phrase_guess_priority = phrase_guess_priority .. pgp9[k] .. string.rep(' ', max_len - #pgp9[k])
	end
      print(phrase_guessed)
      print(phrase_target)
      print(phrase_guess_priority)
      print(math.exp(cumMeanGeomPplx/100)) -- 100 = 1/(1-0.99)
      print(' ')
      end
    end
    loss = loss / opt.seq_length / n
    

  else -- full eval on test set
    local x, y, x_char = loader:next_batch(split_idx)
    if opt.gpuid >= 0 then -- ship the input arrays to GPU
      -- have to convert to float because integers can't be cuda()'d
      x = x:float():cuda()
      y = y:float():cuda()
      x_char = x_char:float():cuda()
    end
    protos.rnn:evaluate() -- just need one clone
    for t = 1, x:size(2) do
      local lst = protos.rnn:forward(get_input(x, x_char, t, rnn_state[0]))
      rnn_state[0] = {}
      for i=1,#init_state do table.insert(rnn_state[0], lst[i]) end
      predictions = lst[#lst] 
      local tok_perp
      tok_perp = protos.criterion:forward(predictions, y[{{},t}])
      loss = loss + tok_perp
    end
    loss = loss / x:size(2)
    
  end    
  local perp = torch.exp(loss)    
  return perp
end

-- do fwd/bwd and return loss, grad_params
-- local init_state_global = clone_list(init_state)
-- local cumMeanGeomPplx = 1
function feval(x, num_iteration)
  if x ~= params then
    params:copy(x)
  end
  grad_params:zero()
  if opt.hsm > 0 then
    hsm_grad_params:zero()
  end
  ------------------ get minibatch -------------------
  local x, y, x_char = loader:next_batch(1) --from train
  if opt.use_chars == 0 then x_char = torch.Tensor{1,2} end -- x_char may eat 2gb of CPU RAM
  if opt.gpuid >= 0 then -- ship the input arrays to GPU
    -- have to convert to float because integers can't be cuda()'d
    x = x:float():cuda()
    y = y:float():cuda()
    x_char = x_char:float():cuda()
  end
  ------------------- forward pass -------------------
  local rnn_state = {[0] = init_state_global}
  local rnn_state2 = {[0] = init_state_global2} --here, for if inside "if seq3seq" then it annihilates after
  local rnn_state3 = {[0] = init_state_global3}
  local predictions = {}           -- softmax outputs
  local loss = 0
  local phrase_guessed = ''
  local phrase_guessed_add = ''
  local phrase_target = ''
  local phrase_guess_priority = ''
  if num_iteration % opt.print_every == 1 then 
    file = io.open (opt.logFile,'a'); 
    file:write('num_iteration = ' .. num_iteration .. ', mean(abs(c1)), max(abs(c1)), mean(c),    same for h1,c2,h2 \n')
    file:close()
  end
  for t=1,opt.seq_length do
    clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag) --for dropout to function in dropout regime (not *(1-p))
    local lst = clones.rnn[t]:forward(get_input(x, x_char, t, rnn_state[t-1]))
    rnn_state[t] = {}
    for i=1,#init_state do -- #init_state = 4 for num_layers = 5,  so #rnn_state[t] = 4, and also we have predictions[t]
      table.insert(rnn_state[t], lst[i]) 
    end -- extract the state, without output
	-- print(#lst)
    predictions[t] = lst[#lst] -- last element is the prediction
    if num_iteration % opt.print_every == 1 then
      file = io.open (opt.logFile,'a')
      for i=1,#lst do -- #init_state = 4 for num_layers = 5,  so #rnn_state[t] = 4, and also we have predictions[t]
	file:write(string.format('%6.4f', torch.mean(torch.abs(   lst[i]   )) ) .. ' ') -- c1,h1,c2,h2 (for 1st timestep, then for 2nd, then for 3rd...)
	file:write(string.format('%6.4f',  torch.max(torch.abs(   lst[i]   )) ) .. ' ') -- c1,h1,c2,h2 (for 1st timestep, then for 2nd, then for 3rd...)
	file:write(string.format('%6.4f', torch.mean(             lst[i]    )  ) .. '   ') -- c1,h1,c2,h2 (for 1st timestep, then for 2nd, then for 3rd...)
      end
      file:write('\n\nmean(abs(layer)), max(layer), min(layer), mean(layer)\n')
      for i=1,#clones.rnn[t].modules do
	file:write(string.format('%6.4f',   torch.mean(torch.abs(   clones.rnn[t].modules[i].output ))) .. ' ' )
	file:write(string.format('%6.4f',   torch.max(   clones.rnn[t].modules[i].output )) .. ' ' )
	file:write(string.format('%6.4f',   torch.min(   clones.rnn[t].modules[i].output )) .. ' ' )
	file:write(string.format('%6.4f',   torch.mean(   clones.rnn[t].modules[i].output )) .. '   ' )
	file:write(tostring(protos.rnn:listModules()[i+1]) .. '\n')
      end
      file:write('\n')
      file:close()
   --[[
  print(clones.rnn[t].modules[38].output)
  print({clones.rnn[t].modules[39].bias})
  print({clones.rnn[t].modules[39].weight})
  print(clones.rnn[t].modules[39].output)
  print(clones.rnn[t].modules)
  print(clones.rnn[t].modules[39].addBuffer)
	--]]
    end
    --print({protos.rnn})
    
    if opt.showSentences > 0 then
      local t = #predictions
      local targets = y[{{}, t}]
      local pred_max, index_pred_max = torch.max(predictions[t],2)
      local pred_sorted, index_pred_sorted = torch.sort(predictions[t],2,true) --descending sort
      local smth857, guess_priority = torch.sort(index_pred_sorted,2,false) --ascending sort
      phrase_guessed_add = loader.idx2word[index_pred_max[1][1]]
      if phrase_guessed_add == opt.unk then phrase_guessed_add = loader.idx2word[index_pred_sorted[1][2]] end
      phrase_guessed = phrase_guessed .. ' ' .. phrase_guessed_add
      phrase_target = phrase_target .. ' ' .. loader.idx2word[targets[1]]
      phrase_guess_priority = phrase_guess_priority .. ' ' .. guess_priority[1][targets[1]]
      cumMeanGeomPplx = cumMeanGeomPplx * 0.99 + math.log(guess_priority[1][targets[1]])
    end
    -- print('--forward pass 3--') --cuda runtime error (77) : an illegal memory access
    if opt.seq3seq == 0 then
      loss = loss + clones.criterion[t]:forward(predictions[t], y[{{}, t}])  
    end
    -- print('--forward pass 4--') --cuda runtime error (77) : an illegal memory access
  end
  if opt.seq3seq == 0 then
    loss = loss / opt.seq_length
  end
  
  if opt.showSentences > 0 then
    print(phrase_guessed)
    print(' ')
    print(phrase_target)
    print(phrase_guess_priority)
    print(math.exp(cumMeanGeomPplx/100)) -- 100 = 1/(1-0.99)
  end
  
  if num_iteration % opt.print_every == 1 then
    file = io.open (opt.logFile,'a')
    file:write('mean(abs(weights)), max(abs(weights)), mean(weights)   for each of layers with weights:\n')
    file:write('    nn.Linear gives two lines: (1) for weights (2) for biases\n')
    file:write('    nn.BatchNormalisation gives two lines the same way\n')
    file:write('    nn.LookupTable has no bias and gives only one line\n')
    local currMod = 1 -- not 0 because 1 is nn.gModule
    local nm = 0
    for i = 1, #({ ({ protos.rnn:listModules()[1]:parameters() }) [1] })[1] do
      while nm < i do
	currMod = currMod + 1
	if ({ protos.rnn:listModules()[currMod]:parameters() }) [1] ~= nil then
	  nm = nm + #({ protos.rnn:listModules()[currMod]:parameters() }) [1]
	  if nm >= i then file:write(tostring(protos.rnn:listModules()[currMod]) .. '\n') end
	end
      end
      file:write(string.format('%8.6f', torch.mean(torch.abs(   ({ protos.rnn:listModules()[1]:parameters() }) [1][i] ))   ) .. ' ')
      file:write(string.format('%8.6f',  torch.max(torch.abs(   ({ protos.rnn:listModules()[1]:parameters() }) [1][i] ))   ) .. ' ')
      file:write(string.format('%8.6f', torch.mean(             ({ protos.rnn:listModules()[1]:parameters() }) [1][i]  )   ) .. ' ')
      file:write('\n')
    end
    file:write('\n\n')
    file:close()
    
  --  for i,module in ipairs(protos.rnn:listModules()) do
  --    print(module)
  --    print({module:parameters()})
  --  end
  end
  -- print( ({ ({ protos.rnn:listModules()[1]:parameters() }) [1] })[1][2][70][35] )
  -- protos.rnn:listModules()[1] — это всегда nn.gModule
  -- protos.rnn:listModules()[1]:parameters() содержит таблицу всех весов нейросети и градиентов этих весов
  -- ({ protos.rnn:listModules()[1]:parameters() }) [1] содержит таблицу весов нейросети, но в виде двумерной таблицы. 
  -- А вот  ({ ({ protos.rnn:listModules()[1]:parameters() }) [1] }) [1]
  -- является уже понятной таблицей, в которой k-ый элемент является тензором весов k-го элемента нейросети
  
  local if_sent_ends = {}
  if opt.seq3seq == 1 then
    

    ------------------- forward LSTM seq2seq pass -------------------
   -- print('--forward LSTM seq2seq pass--')
  --local sent_lens = {}
  local input2 = {}
  local sent_ends = {} -- like {7,17,32}
  -- local if_sent_ends = {} -- like {0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,0,0,0}
  for i = 1,opt.seq_length do
    if x[1][i] == loader.word2idx["."] or x[1][i] == loader.word2idx[","] then -- support for now only batch_size = 1, so x[1] refers to 1st batch if we have 											   -- batch_size > 1
      table.insert(input2, predictions[i]); -- rnn_state[i][#rnn_state[i]-2])
      sent_ends[#sent_ends+1] = i
      if_sent_ends[i] = #sent_ends
    else
      if_sent_ends[i] = 0
    end    
    --sent_lens[#sent_lens] = sent_lens[#sent_lens] + 1
  end
  grad_params2:zero()
  if opt.hsm > 0 then
    hsm_grad_params2:zero()
  end
  -- local rnn_state2 = {[0] = init_state_global2}
  local predictions2 = {}           -- to decoder LSTM
  local loss2 = 0
  for t=1,#input2 do
    clones2.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag) --for dropout to function in dropout regime (not *(1-p))
    local lst = clones2.rnn[t]:forward(get_input2(input2, t, rnn_state2[t-1]))
    rnn_state2[t] = {}
    for i=1,#init_state do -- #init_state = 4 for num_layers = 5,  so #rnn_state[t] = 4, and also we have predictions[t]
      table.insert(rnn_state2[t], lst[i]) 
    end -- extract the state, without output
	-- print(#lst)
    predictions2[t] = lst[#lst] -- last element is the prediction 
  end
    ------------------- forward LSTM-Decoder pass -------------------
  --print('--forward LSTM-Decoder pass--')
  -- input3 = predictions2 
  local input3 = {}
  for i = 1,opt.seq_length do
    if if_sent_ends[i] > 0 then --x[1][i] == loader.word2idx["."] then --support for now only batch_size = 1,so x[1] refers to 1st batch if we have batch_size > 1
      table.insert(input3, predictions2[if_sent_ends[i]]) -- rnn_state[i][#rnn_state[i]-2])
    else
      table.insert(input3, x[{{},t}]) --predictions2[1]:mul(0))
    end    
  end
  grad_params3:zero()
  if opt.hsm > 0 then
    hsm_grad_params3:zero()
  end
  -- local rnn_state3 = {[0] = init_state_global3}
  local predictions3 = {}           -- to decoder LSTM
  local loss3 = 0
  for t=1,opt.seq_length do
    clones3.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag) --for dropout to function in dropout regime (not *(1-p))
      --print(input3)
      --print(rnn_state3[t-1])
     -- print(get_input2(input3, t, rnn_state3[t-1]))
      local lst = clones3.rnn[t]:forward(get_input2(input3, t, rnn_state3[t-1])) -- get_input2 fits for get_input3 also
    rnn_state3[t] = {}
    for i=1,#init_state do -- #init_state = 4 for num_layers = 5,  so #rnn_state[t] = 4, and also we have predictions[t]
      table.insert(rnn_state3[t], lst[i]) 
    end -- extract the state, without output
	-- print(#lst)
    predictions3[t] = lst[#lst] -- last element is the prediction
    loss = loss + clones.criterion[t]:forward(predictions3[t], y[{{}, t}])  
  end
  if num_iteration % opt.print_every == 0 then
    print(predictions3[1][1][1]..' '..predictions3[1][1][2]..' '..predictions3[1][1][3]..' '..predictions3[1][1][4]..' '..predictions3[1][1][5])
  end
  loss = loss / opt.seq_length
  
  
  ------------------ backward pass through Decoder -------------------
  --print('--backward pass through Decoder --')
  -- initialize gradient at time t to be zeros (there's no influence from future)
  local drnn_state3 = {[opt.seq_length] = clone_list(init_state, true)} -- true also zeros the clones
  local temp = {}
  for t=opt.seq_length,1,-1 do
    -- backprop through loss, a0nd softmax/linear
    local doutput_t = clones3.criterion[t]:backward(predictions3[t], y[{{}, t}])
    table.insert(drnn_state3[t], doutput_t)
    table.insert(rnn_state3[t-1], drnn_state3[t])
    local dlst = clones3.rnn[t]:backward(get_input2(input3, t, rnn_state3[t-1]), drnn_state3[t])
    drnn_state3[t-1] = {}
    local tmp = 1 -- opt.use_words + opt.use_chars -- not the safest way but quick
    for k,v in pairs(dlst) do
      if k > tmp then -- k == 1 is gradient on x, which we dont need
        -- note we do k-1 because first item is dembeddings, and then follow the 
        -- derivatives of the state, starting at index 2. I know...
        drnn_state3[t-1][k-tmp] = v
      end
    end	
    if num_iteration % opt.print_every == 1 then
      file = io.open (opt.logFile,'a')
      file:write('\n\ndecoder: mean(abs(gradInput)), max(gradInput), min(gradInput), mean(gradInput)\n')
      for i=1,#clones3.rnn[t].modules do
	if #{clones3.rnn[t].modules[i].gradInput} and tostring(#clones3.rnn[t].modules[i].gradInput) ~= '[torch.LongStorage of size 0]\n'
						 and type(clones3.rnn[t].modules[i].gradInput) ~= 'table'  then
	  file:write(string.format('%9.8f',   torch.mean(torch.abs(   clones3.rnn[t].modules[i].gradInput ))) .. ' ' )
	  file:write(string.format('%9.8f',   torch.max(   clones3.rnn[t].modules[i].gradInput )) .. ' ' )
	  file:write(string.format('%9.8f',   torch.min(   clones3.rnn[t].modules[i].gradInput )) .. ' ' )
	  file:write(string.format('%9.8f',   torch.mean(   clones3.rnn[t].modules[i].gradInput )) .. '   ' )
	  file:write(tostring(protos3.rnn:listModules()[i+1]) .. '\n')
	else
	  file:write('                               ' .. tostring(protos2.rnn:listModules()[i+1]) .. '\n')
	end
      end
      file:write('\n')
      file:close()
    end
  end
  ------------------ backward pass through seq2seq -------------------
  --print('--backward pass through seq2seq --')
  -- initialize gradient at time t to be zeros (there's no influence from future)
  local drnn_state2 = {[#input2] = clone_list(init_state, true)} -- true also zeros the clones
  local temp = {}
  
  for t = #input2,1,-1 do
    -- backprop through loss, and softmax/linear
    local doutput_t = clones3.rnn[sent_ends[t]].modules[opt.mbp].gradInput  ----------------- really modules[1]? check --clones3, that's right. Not clones2
    table.insert(drnn_state2[t], doutput_t)
    table.insert(rnn_state2[t-1], drnn_state2[t])
    local dlst = clones2.rnn[t]:backward(get_input2(input2, t, rnn_state2[t-1]), drnn_state2[t])
    drnn_state2[t-1] = {}
    local tmp = 1 -- opt.use_words + opt.use_chars -- not the safest way but quick
    for k,v in pairs(dlst) do
      if k > tmp then -- k == 1 is gradient on x, which we dont need
        -- note we do k-1 because first item is dembeddings, and then follow the 
        -- derivatives of the state, starting at index 2. I know...
        drnn_state2[t-1][k-tmp] = v
      end
    end	
    if num_iteration % opt.print_every == 1 then
      file = io.open (opt.logFile,'a')
      file:write('\n\nseq2seq: mean(abs(gradInput)), max(gradInput), min(gradInput), mean(gradInput)\n')
      for i=1,#clones2.rnn[t].modules do
	if #{clones2.rnn[t].modules[i].gradInput} and tostring(#clones2.rnn[t].modules[i].gradInput) ~= '[torch.LongStorage of size 0]\n'
						 and type(clones2.rnn[t].modules[i].gradInput) ~= 'table'  then
	  file:write(string.format('%9.8f',   torch.mean(torch.abs(   clones2.rnn[t].modules[i].gradInput ))) .. ' ' )
	  file:write(string.format('%9.8f',   torch.max(   clones2.rnn[t].modules[i].gradInput )) .. ' ' )
	  file:write(string.format('%9.8f',   torch.min(   clones2.rnn[t].modules[i].gradInput )) .. ' ' )
	  file:write(string.format('%9.8f',   torch.mean(   clones2.rnn[t].modules[i].gradInput )) .. '   ' )
	  file:write(tostring(protos2.rnn:listModules()[i+1]) .. '\n')
	else
	  file:write('                               ' .. tostring(protos2.rnn:listModules()[i+1]) .. '\n')
	end
      end
      file:write('\n')
      file:close()
    end
  end
  
---------------==========================\\\\\\\\\\\\\\\\\\\\\\
  end ----------- if seq3seq == 1
  
  
  ------------------ backward pass -------------------
  --print('--backward pass--')
  -- initialize gradient at time t to be zeros (there's no influence from future)
  local drnn_state = {[opt.seq_length] = clone_list(init_state, true)} -- true also zeros the clones
  local temp = {}
  
  for t=opt.seq_length,1,-1 do
    -- backprop through loss, and softmax/linear
    local doutput_t = {}
    if opt.seq3seq == 1 then
      if if_sent_ends[t] > 0 then
	doutput_t = clones2.rnn[if_sent_ends[t]].modules[opt.mbp].gradInput  ----------------- really modules[1]? check
      else
	doutput_t = clones2.rnn[1].modules[opt.mbp].gradInput : mul(0)  ----------------- really modules[1]? check
      end
    else
      doutput_t = clones.criterion[t]:backward(predictions[t], y[{{}, t}])
    end
   
    table.insert(drnn_state[t], doutput_t)
    table.insert(rnn_state[t-1], drnn_state[t])
    local dlst = clones.rnn[t]:backward(get_input(x, x_char, t, rnn_state[t-1]), drnn_state[t])
    drnn_state[t-1] = {}
    local tmp = opt.use_words + opt.use_chars -- not the safest way but quick
    for k,v in pairs(dlst) do
      if k > tmp then -- k == 1 is gradient on x, which we dont need
        -- note we do k-1 because first item is dembeddings, and then follow the 
        -- derivatives of the state, starting at index 2. I know...
        drnn_state[t-1][k-tmp] = v
      end
    end	
    if num_iteration % opt.print_every == 1 then
      file = io.open (opt.logFile,'a')
      file:write('\n\nmean(abs(gradInput)), max(gradInput), min(gradInput), mean(gradInput)\n')
      for i=1,#clones.rnn[t].modules do
	if #{clones.rnn[t].modules[i].gradInput} and tostring(#clones.rnn[t].modules[i].gradInput) ~= '[torch.LongStorage of size 0]\n'
						 and type(clones.rnn[t].modules[i].gradInput) ~= 'table'  then
	  file:write(string.format('%9.8f',   torch.mean(torch.abs(   clones.rnn[t].modules[i].gradInput ))) .. ' ' )
	  file:write(string.format('%9.8f',   torch.max(   clones.rnn[t].modules[i].gradInput )) .. ' ' )
	  file:write(string.format('%9.8f',   torch.min(   clones.rnn[t].modules[i].gradInput )) .. ' ' )
	  file:write(string.format('%9.8f',   torch.mean(   clones.rnn[t].modules[i].gradInput )) .. '   ' )
	  file:write(tostring(protos.rnn:listModules()[i+1]) .. '\n')
	else
	  file:write('                               ' .. tostring(protos.rnn:listModules()[i+1]) .. '\n')
	end
      end
      file:write('\n')
      file:close()
    end
  end
  
  ------------------------ misc ----------------------
  -- transfer final state to initial state (BPTT)
  init_state_global = rnn_state[#rnn_state] -- NOTE: I don't think this needs to be a clone, right?
  init_state_global2 = rnn_state2[#rnn_state2] -- NOTE: I don't think this needs to be a clone, right?
  init_state_global3 = rnn_state3[#rnn_state3] -- NOTE: I don't think this needs to be a clone, right?

  -- renormalize gradients
  --print(protos.rnn:listModules()[40].gradBias)
  local grad_norm, shrink_factor
  if opt.hsm==0 then
    grad_norm = grad_params:norm()
  else
    grad_norm = torch.sqrt(grad_params:norm()^2 + hsm_grad_params:norm()^2)
  end
  if grad_norm > opt.max_grad_norm then
    -- print(grad_norm)
    shrink_factor = opt.max_grad_norm / grad_norm
    grad_params:mul(shrink_factor)
    if opt.hsm > 0 then
      hsm_grad_params:mul(shrink_factor)
    end
  end    
  if grad_norm < opt.min_grad_norm then
    -- print(grad_norm)
    shrink_factor = opt.min_grad_norm / grad_norm
    grad_params:mul(shrink_factor)
    if opt.hsm > 0 then
      hsm_grad_params:mul(shrink_factor)
    end
  end
  
  local grad_norm2, shrink_factor2
    grad_norm2 = grad_params2:norm()
  if grad_norm2 > opt.max_grad_norm then
    shrink_factor2 = opt.max_grad_norm / grad_norm2
    grad_params2:mul(shrink_factor2)
  end  
  if grad_norm2 < opt.min_grad_norm then
    shrink_factor2 = opt.min_grad_norm / grad_norm2
    grad_params2:mul(shrink_factor2)
  end
    
  
  local grad_norm3, shrink_factor3
    grad_norm3 = grad_params3:norm()
  if grad_norm3 > opt.max_grad_norm then
    shrink_factor3 = opt.max_grad_norm / grad_norm3
    grad_params3:mul(shrink_factor3)
  end   
  if grad_norm3 < opt.max_grad_norm then
    shrink_factor3 = opt.min_grad_norm / grad_norm3
    grad_params3:mul(shrink_factor3)
  end
  
  --print(protos.rnn:listModules()[40].gradBias)
--  print(torch.max(grad_params))
  --print(protos.rnn:listModules()[40].bias)
  params:add(grad_params:mul(-lr)) -- update params
  if opt.seq3seq > 0 then
    params2:add(grad_params2:mul(-lr))
    params3:add(grad_params3:mul(-lr))
  end
  --print(protos.rnn:listModules()[40].bias)
  --protos.rnn:listModules()[40].weight : add(protos.rnn:listModules()[40].gradWeight:mul(-lr)) -- returns back (and yes, -lr it should be for that, again -lr)
  --print(protos.rnn:listModules()[40].bias)
  --protos.rnn:listModules()[40].bias : add(protos.rnn:listModules()[40].gradBias:mul(-lr))
  if opt.deepOut == 1 then
    --[[
    protos.rnn:listModules()[40].weight = torch.eye(200):float():cuda()
    protos.rnn:listModules()[40].bias : mul(0)
    --print(protos.rnn.modules)
    local eye_my = torch.eye(200):float():cuda()
    protos.rnn.modules[39].weight : mul(0): add(eye_my) -- may be this isn't needed; but it's experimented to be insufficient w/o clones.rnn...... also
    clones.rnn[35].modules[39].weight : mul(0) : add(eye_my)
    protos.rnn.modules[39].bias : mul(0) -- may be this isn't needed; but it's experimented to be insufficient w/o clones.rnn...... also
    clones.rnn[35].modules[39].bias : mul(0)
    --]]
    clones.rnn[35].modules[39].weight : add(clones.rnn[35].modules[39].gradWeight:mul(-0.9*lr)) --yes, again -lr -- to revert weight changes
    clones.rnn[35].modules[39].bias : add(clones.rnn[35].modules[39].gradBias:mul(-0.9*lr))
 ---   local mmm = torch.Tensor(20,200):float():cuda()
    -- print({clones.rnn[35].modules[39].weight})
 ---   mmm:addmm(clones.rnn[35].modules[38].output, clones.rnn[35].modules[39].weight)
    --mmm:addr(clones.rnn[35].modules[39].addBuffer, clones.rnn[35].modules[39].bias)
 ---   print('--------------mmm ----------------')
 ---   print(mmm)
  end
  
  params:mul(opt.weight_decay)
  if opt.seq3seq > 0 then
    params2:mul(opt.weight_decay)
    params3:mul(opt.weight_decay)
  end
--  params[params:ge(1.0)] = 1.0
--  params[params:le(-1.0)] = -1.0
  if opt.hsm > 0 then
    hsm_params:add(hsm_grad_params:mul(-lr))
  end
  return torch.exp(loss)
end

if opt.genText > 0 then
  generate_text(opt.phrase_initial,1000, opt.genTextNoise)
end

if opt.validationFirst > 0 then
  local val_loss = eval_split(2) -- 2 = validation
  print('validation loss = ' .. val_loss)
end
  
-- start optimization here
train_losses = {}
val_losses = {}
lr = opt.learning_rate -- starting learning rate which will be decayed
local iterations = opt.max_epochs * loader.split_sizes[1]
if char_vecs ~= nil then char_vecs.weight[1]:zero() end -- zero-padding vector is always zero
for i = 1, iterations do
  local epoch = i / loader.split_sizes[1]

  local timer = torch.Timer()
  local time = timer:time().real

  train_loss = feval(params, i) -- fwd/backprop and update params
  if char_vecs ~= nil then -- zero-padding vector is always zero
    char_vecs.weight[1]:zero() 
    char_vecs.gradWeight[1]:zero()
  end 
  train_losses[i] = train_loss

  -- every now and then or on last iteration
  if i % loader.split_sizes[1] == 0 then
    -- evaluate loss on validation data
    local val_loss = eval_split(2) -- 2 = validation
    val_losses[#val_losses+1] = val_loss
    local savefile = string.format('%s/lm_%s_epoch%.2f_%.2f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_loss)
    local checkpoint = {}
    checkpoint.protos = protos
    checkpoint.opt = opt
    checkpoint.train_losses = train_losses
    checkpoint.val_loss = val_loss
    checkpoint.val_losses = val_losses
    checkpoint.i = i
    checkpoint.epoch = epoch
    checkpoint.vocab = {loader.idx2word, loader.word2idx, loader.idx2char, loader.char2idx}
    checkpoint.lr = lr
    print('saving checkpoint to ' .. savefile)
    if epoch == opt.max_epochs or epoch % opt.save_every == 0 then
      torch.save(savefile, checkpoint)
    end
  end

  -- decay learning rate after epoch
  if i % loader.split_sizes[1] == 0 and #val_losses > 2 then
    if val_losses[#val_losses-1] - val_losses[#val_losses] < (opt.decay_when * lr) then
      lr = lr * opt.learning_rate_decay
      if lr < opt.lr_min then lr = opt.lr_min end
    end
  end    

  if i % opt.print_every == 0 then
    local timeprint = string.format(" ")
    if opt.time ~= 0 then
      timeprint = string.format(", batch time: %6.4f", timer:time().real - time)
    end
    print(string.format("%d/%d (epoch %.2f), train_loss = %6.4f, lr = %9.7f", i, iterations, epoch, train_loss, lr) .. timeprint)
    lr = lr * opt.lr_decay_continuous;
  end   
  if i % 10 == 0 then collectgarbage() end

end

--evaluate on full test set. this just uses the model from the last epoch
--rather than best-performing model. it is also incredibly inefficient
--because of batch size issues. for faster evaluation, use evaluate.lua, i.e.
--th evaluate.lua -model m
--where m is the path to the best-performing model

if opt.ignoreTestSet == 0 then
  test_perp = eval_split(3)
  print('Perplexity on test set: ' .. test_perp)
end

