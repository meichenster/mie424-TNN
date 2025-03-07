# PARAMS
MODELS="mip cp hw ha"
OBJECTIVES="min-w max-m"
HLS=`seq 0 2`
EXS=`seq 1 3`
INS=`seq 0 0`
TO=5
# Running the experiments
cd ../src
for model in $MODELS; do
	for obj in $OBJECTIVES; do
		for hls in $HLS; do
			for exs in $EXS; do
				for ins in $INS; do
					python3 run.py --model=$model --obj=$obj --hls=$hls --exs=$exs --ins=$ins --to=$TO
				done
			done
		done
	done
done