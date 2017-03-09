
echo "Current style loss:0.5"
./train_models.sh abstract_art 0.5
for style_loss in `seq 1 2`
do	
	echo "Current style loss:"$style_loss
	./train_models.sh abstract_art $style_loss
done

echo "Current style loss:0.5"
./train_models.sh cveta_art 0.5
for style_loss in `seq 1 3`
do	
	echo "Current style loss:"$style_loss
	./train_models.sh cveta_art $style_loss
done
