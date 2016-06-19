require 'torch'
require 'image'

-- https://github.com/torch/demos/blob/master/load-data/load-images.lua

-- 2. Load all files in directory

files = {}
train_ext = "png"
train_path = "./data/Class1_def/"

for file in paths.files(train_path) do
	if file:find(train_ext .. '$') then
	  table.insert(files, paths.concat(train_path, file))
  end
end

if #files == 0 then
  error('Given folder: ' .. train_path .. ' does not contain any files of type: ' .. train_ext)
end

-- 3. Sort file names

table.sort(files, function (a,b) return a < b end)

print('Found files:')
print(files)

-- 4. Finally we load images
-- Go over the file list:
images = {}
for i,file in ipairs(files) do
   -- load each image
   table.insert(images, image.load(file))
end

print('Loaded images:')
print(images)

-- Display a of few them
for i = 1,math.min(#files,10) do
   image.display{image=images[i], legend=files[i]}
end

