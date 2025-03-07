# PARAMS
MODELS="gd_b gd_t"
LRS="1e-3 1e-4 1e-5 1e-6"
SEEDS=`seq 0 4`
HLS=`seq 0 2`
EXS=`seq 1 10`
INS=`seq 0 9`
TO=120
# Running the experiments
cd ../src
for model in $MODELS; do
	for lr in $LRS; do
		for seed in $SEEDS; do
			for hls in $HLS; do
				for exs in $EXS; do
					for ins in $INS; do
						python3 run.py --model=\"$model\" --lr=$lr --seed=$seed --hls=$hls --exs=$exs --ins=$ins --to=$TO
					done
				done
			done
		done
	done
done