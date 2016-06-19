
--[[
Returns all images paths with extension image_ext in folder image_path 
]]--
function readImagePaths ( image_path, image_ext)
  
  files = {}
  
  for file in paths.files( image_path ) do
	  if file:find( image_ext .. '$' ) then
	    table.insert( files, paths.concat( image_path, file ) )
    end
  end

  if #files == 0 then
    error( 'Given folder: ' .. image_path .. ' does not contain any files of type: ' .. image_ext )
  end
  
  return files
  
end

--[[
  Loads and returns all imges
]]--
function loadImages ( image_path, image_ext )

-- 2. Load all files in directory
  files = {}
  files = readImagePaths( image_path, image_ext )
  
-- 3. Sort file names
  table.sort( files, function( a, b ) return a < b end )

-- 4. Load images

  max_images = 500
  images = {}
  for i, file in ipairs( files ) do
    if i > max_images then break end
    table.insert(images, image.load(file))
  end

--[[
  first = true
  images = {}
  for i, file in ipairs( files ) do    
    if first then
      images = image.load(file):resize(1 , 1, 512, 512)
      first = false
    else
      images = torch.cat( images, image.load(file):resize(1 , 1, 512, 512), 1 )
    end
    print(images:size())
    -- table.insert(images, image.load(file))
  end
]]--

  -- Display a of few them
  --[[
  for i = 1,math.min(#files, 3) do
     image.display{image=images[i], legend=files[i]}
  end
  --]]
  return images

end


--[[
- Converts a table of 3D tensor to 4D tensor
- Normalaizes data to N(0, 1)
]]--
function prepData ( input_data )

-- 4D tensor
  preped_data = { data = {}, label = {} }
  preped_data.data = torch.Tensor(#input_data.data, 1, 512, 512)

  for k, v in ipairs(input_data.data) do
    preped_data.data[k] = v
    preped_data.label[k] = input_data.label[k]
  end

-- Normalization
  num_color_channels = 1
  mean = {} -- store the mean, to normalize the test set in the future
  stdv = {} -- store the standard-deviation for the future
  for i=1, num_color_channels do -- over each image channel
    mean[i] = preped_data.data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    print('Channel ' .. i .. ', Mean: ' .. mean[i])
    preped_data.data[{ {}, {i}, {}, {} }]:add(-mean[i]) -- mean subtraction

    stdv[i] = preped_data.data[{ {}, {i}, {}, {}  }]:std() -- std estimation
    print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
    preped_data.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
  end

  return preped_data
end
















