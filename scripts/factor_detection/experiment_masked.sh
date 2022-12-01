while read d; do
	julia --project factor_detection_masked.jl sgvaegan100 wildlife_MNIST $d 
done < wmnist_models.txt
