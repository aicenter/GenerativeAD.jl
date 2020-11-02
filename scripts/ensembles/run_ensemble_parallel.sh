# this will run either images or tabular datasets
MODEL=$1 		# which model to run
DATASET_TYPE=$2 # images | tabular
DATASET_FILE=$3	# file with dataset list

LOG_DIR="${HOME}/logs/${MODEL}_ensemble"

if [ ! -d "$LOG_DIR" ]; then
	mkdir $LOG_DIR
fi

while read d; do
    sbatch \
    --output="${LOG_DIR}/${d}-%J.out" \
     ./run_ensemble.sh $MODEL $d $DATASET_TYPE
done < ${DATASET_FILE}
