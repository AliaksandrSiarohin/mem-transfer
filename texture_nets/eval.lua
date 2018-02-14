require 'image'
require 'torch'
require 'caffe'
require 'nn'

local cmd = torch.CmdLine()

cmd:option('-input_file', '../../lamem/splits/test_1.txt', 'File with image names and scores')
cmd:option('-image_dir', '../lamem/images/', 'Directory with images')
cmd:option('-output_file', 'predictions.txt', 'Output file with predicted scores')
cmd:option('-weights', '../lamem/internal_predictor.caffemodel', 'Weights of the network .caffemodel')
cmd:option('-network', '../lamem/deploy.prototxt',  'Network description')
cmd:option('-backend', 'cuda' , 'Use cuda or cpu')
cmd:option('-image_size', 256, 'Size of the image from which we will sample crops')
cmd:option('-crop_count', 10, 'Number of random crops per image')
cmd:option('-display_score', 100, 'Display score per this number of image')
cmd:option('-seed_name', 'seed.jpg', 'Seed that generate this image')

local params = cmd:parse(arg)

local function read_image_names(filename)
	local df = { }
	for l in io.lines(filename) do
		local n, m = unpack(l:split(" "))
		table.insert(df, {image_name = n, mem_score = tonumber(m)})
	end
	return df
end

local function random_image_crop(img, max_size)
	w = img:size(2)
	h = img:size(3)
	--[[if h < w then
        	sc = image.scale(img, max_size, w*max_size / h)
    	else
        	sc = image.scale(img, h*max_size/w, max_size)
	end]]--
	sc = image.scale(img, max_size, max_size)
	cx = torch.random(1, sc:size(2) - 227 - 1)
	cy = torch.random(1, sc:size(3) - 227 - 1)
	croped = sc [{{}, {cx, cx + 227 - 1}, {cy, cy + 227 - 1}}]
	return croped
end

function normalize_image(img)
	local mv = torch.FloatTensor{104.0069879317889, 116.66876761696767, 122.6789143406786}
	img = img:index(1, torch.LongTensor{3,2,1})
	img[1]:csub(mv[1])
	img[2]:csub(mv[2])
	img[3]:csub(mv[3])

	return img
end

function print_stats(true_vals, predicted_vals, count)
	true_scores = true_vals[{{1, count}}]
	predictions = predicted_vals[{{1, count}}]
	local abs_err = torch.abs(true_scores - predictions)
	local abs_err_sqr = torch.pow(abs_err, 2)
	print ("Squared value error: " .. tostring(abs_err_sqr:mean()))
	print ("Absolute value error: " .. tostring(abs_err:mean()))
	print ("Variance of absolute error: " .. tostring(abs_err:var()))

	local corelation = (true_scores - torch.mean(true_scores)) * (predictions - torch.mean(predictions))

	local c = (corelation / count) / math.sqrt(torch.var(predictions) * torch.var(true_scores))

	print ("Correlation coeficient: " .. tostring(c)) 
end




local images = read_image_names(params.input_file)
local net = caffe.Net(params.network, params.weights, 'test')

local input = torch.FloatTensor(10,3,227,227)

if params.backend == 'cuda' then
	net:setModeGPU()
end

local image_count = #images
local true_scores = torch.FloatTensor(image_count)
local predictions = torch.FloatTensor(image_count)

for i=1,image_count do
	img = image.load(params.image_dir .. images[i]['image_name'], 3) * 255
	img = normalize_image(img)
	for crop=1,params.crop_count do
		input[crop] = random_image_crop(img, params.image_size)
	end

	output = net:forward(input)
	score = output:mean()
	images[i]['predicted_score'] = score
	true_scores[i] = images[i]['mem_score']
	predictions[i] = score
	if i % params.display_score == 0 then
		print_stats(true_scores, predictions, i)
	end
	
	print (tostring(i) .. "/" .. tostring(image_count), images[i]['image_name'], images[i]['mem_score'], score)
end

function file_exists(file)
	local f = io.open(file, 'r')
	if  f == nil then
		return false
	else
		return true
	end
end

if file_exists(params.output_file) then
	f = io.open(params.output_file, "a")
	io.output(f)
else
	f = io.open(params.output_file, 'w')
	io.output(f)
	io.write('in_img,style_img,out_img,in_img_mem,out_img_mem\n')
end

for i=1,image_count do
	io.write('../lamem/images/' .. images[i]['image_name'] .. "," .. params.seed_name .. "," .. params.image_dir .. images[i]['image_name'] .. ","  ..
		 tostring(images[i]['mem_score']) .. "," .. tostring(images[i]['predicted_score'] .. "\n"))
end
io.close(f)
