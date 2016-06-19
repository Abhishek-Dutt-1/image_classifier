require 'torch'
require 'nn'
require 'image'
require 'my_util'

classes = {
'one', 'two', 'three', 'four', 'five', 'six'
}

--[[
images_path = {
{ path: './data/Class1_def/', class: 'one'}
'./data/Class2_def/',
'./data/Class3_def/',
'./data/Class4_def/',
'./data/Class5_def/',
'./data/Class6_def/'
}
]]--

input_data = {
  { path = './data/Class1_def/', ext = 'png', class = 1 },
  { path = './data/Class2_def/', ext = 'png', class = 2 },
  { path = './data/Class3_def/', ext = 'png', class = 3 },
  { path = './data/Class4_def/', ext = 'png', class = 4 },
  { path = './data/Class5_def/', ext = 'png', class = 5 },
  { path = './data/Class6_def/', ext = 'png', class = 6 }
}

trainset_1 = { data = {}, label = {} }
--[[
Read images
]]--
--first = true
for idx, data in ipairs(input_data) do

  files = {}
  files = loadImages(data.path, data.ext)

  if #files ~= 0 then
    --[[
    if first then
      trainset_1.data = files
      first = false
    else
      trainset_1.data = torch.cat( trainset_1.data, files, 1 )
    end
    ]]--
    for k, file in ipairs(files) do
      table.insert( trainset_1.data, file )
      table.insert( trainset_1.label, data.class )
    end
  else
    error('No files found')
  end

end

trainset = prepData( trainset_1 )

--[[
Add :size() and index [i] operator
]]--
setmetatable(trainset, {
  __index = function(t, i) 
    return {t.data[i], t.label[i]} 
  end
});

trainset.data = trainset.data:double() -- convert the data from a ByteTensor to a DoubleTensor.

function trainset:size() 
    return self.data:size(1) 
end

--[[
net = nn.Sequential()
net:add( nn.SpatialConvolution(1, 6, 5, 5)  ) -- 3 input image channels, 6 output channels, 5x5 convolution kernel
net:add( nn.ReLU()                          ) -- non-linearity 
net:add( nn.SpatialMaxPooling(2, 2, 2, 2)   ) -- A max-pooling operation that looks at 2x2 windows and finds the max.
net:add( nn.SpatialConvolution(6, 16, 5, 5) )
net:add( nn.ReLU()                          ) -- non-linearity
net:add( nn.SpatialMaxPooling(2, 2, 2, 2)   )
net:add( nn.View(16*5*5)                    ) -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
net:add( nn.Linear(16*5*5, 120)             ) -- fully connected layer (matrix multiplication between input and weights)
net:add( nn.ReLU()                          ) -- non-linearity 
net:add( nn.Linear(120, 84)                 )
net:add( nn.ReLU()                          ) -- non-linearity 
net:add( nn.Linear(84, 6)                   ) -- 10 is the number of outputs of the network (in this case, 10 digits)
net:add( nn.LogSoftMax()                    ) -- converts the output to a log-probability. Useful for classification problems
--]]--

---[[
net = nn.Sequential()
net:add( nn.SpatialConvolution(1, 6, 5, 5)  )
net:add( nn.ReLU()                          ) 
net:add( nn.SpatialMaxPooling(10, 10, 10, 10) )
net:add( nn.SpatialConvolution(6, 16, 5, 5) )
net:add( nn.ReLU()                          )
net:add( nn.SpatialMaxPooling(9, 9, 9, 9)   )
net:add( nn.View(16*5*5)                    )
net:add( nn.Linear(16*5*5, 120)             )
net:add( nn.ReLU()                          ) 
net:add( nn.Linear(120, 84)                 )
net:add( nn.ReLU()                          ) 
net:add( nn.Linear(84, 6)                   )
net:add( nn.LogSoftMax()                    )
--]]

criterion = nn.ClassNLLCriterion()

trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 5 -- just do 5 epochs of training.




--------------------------
-- Test Data

testset = trainset
num_testset = #testset.label
correct = 0
for i=1,num_testset do
    local groundtruth = testset.label[i]
    local prediction = net:forward(testset.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        correct = correct + 1
    end
end

print(correct, 100*correct/num_testset .. ' % ')

class_performance = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}

for i=1,num_testset do
    local groundtruth = testset.label[i]
    local prediction = net:forward(testset.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        class_performance[groundtruth] = class_performance[groundtruth] + 1
    end
end

for i=1,#classes do
    print(classes[i], 100*class_performance[i]/1000 .. ' %')
end




