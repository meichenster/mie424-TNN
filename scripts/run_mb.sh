# PARAMS
MODELS="mip cp hw ha"
OBJECTIVES="min-w max-m"
HLS=`seq 0 2`
EXS=`seq 1 10`
INS=`seq 0 9`
TO=120
# Running the experiments
cd ../src
for model in $MODELS; do
	for obj in $OBJECTIVES; do
		for hls in $HLS; do
			for exs in $EXS; do
				for ins in $INS; do
					python3 run.py --model=\"$model\" --obj=\"$obj\" --hls=$hls --exs=$exs --ins=$ins --to=$TO
				done
			done
		done
	done
done