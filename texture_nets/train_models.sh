dataset=$1
style_weight=$2
texture_folder=datasets/"$dataset"_seeds
textures=$(ls $texture_folder)
models_folder=data/checkpoints/$style_weight/$dataset
results_folder=datasets/result/$style_weight/$dataset
input_folder=datasets/"$dataset"_test
mem_score_file=$1_$2_memorability_internal.csv
rm log.out.txt, log.err.txt
mkdir -p $results_folder
mkdir -p $models_folder
progress=1
echo "Training..."
for file in $textures
do
	echo $progress/$(echo $textures|wc -w|cut -f1)
	progress=$(($progress + 1))
	mkdir -p $models_folder/$file
	th train.lua -data datasets/lamem -style_image $texture_folder/$file \
	 -style_size 256 -image_size 256 -model johnson -batch_size 4 -learning_rate 1e-2 -style_weight $style_weight \
	 -style_layers relu1_2,relu2_2,relu3_2,relu4_2 -content_layers relu4_2 \
	 -backend nn -checkpoints_path $models_folder/$file -num_iterations 10000 >> log.out.txt 2>> log.err.txt
	mkdir -p $results_folder/$file
	

	th test_dir.lua -image_size 256 -input_path $input_folder -model_t7 $models_folder/$file/model_10000.t7 -save_path $results_folder/$file >> log.out.txt 2>> log.err.txt


	th eval.lua -input_file ../lamem/lamem/splits/test_1.txt -seed_name $file -image_dir $results_folder/$file/ -output_file $mem_score_file >> log.out.txt 2>> log.err.txt

done
python memorability_file_format.py $mem_score_file "$dataset"_external_sw"$style_weight".txt
