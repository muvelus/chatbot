require 'neuralconvo'
require 'xlua'

--Load dataset
print("-- Loading dataset")
dataset = neuralconvo.DataSet(neuralconvo.CornelMovieDialogs("data/cornell_movie_dialogs"), 
   {
   loadFirst = options.dataset,
   minWordFreq = options.minWordFreq
   })

--Build Model
model = neuralconvo.Seq2Seq(dataset.wordsCount, options.hiddenSize)
model.goToken = dataset.goToken
model.eosToken = dataset.eosToken

--Training parameters
model.criterion = nn.SequenceCriterion(nn.ClassNLLCriterion())
model.learningRate = options.learningRate
model.momentum = options.momentum
local decayFactor = (options.minLR - options.learningRate / options.saturateEpoch)
local minMeanError = nil

-- Enable CUDA
if options.cuda then
   require 'cutorch'
   require 'cunn'
   model:cuda()
elseif options.opencl then
   require 'cltorch'
   require 'clnm'
   model:cl()

-- Train the model using backpropagation

