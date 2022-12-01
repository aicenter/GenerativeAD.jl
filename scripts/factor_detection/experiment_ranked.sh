while read d; do
	julia --project factor_detection_ranked.jl sgvaegan100 wildlife_MNIST $d 
done < wmnist_models.txt
