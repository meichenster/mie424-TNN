# PARAMS
TO=5
# Running the experiments
cd ../src
python3 print_results.py --model="mip" --obj="min-w" --to=$TO
python3 print_results.py --model="mip" --obj="max-m" --to=$TO
python3 print_results.py --model="cp" --obj="min-w" --to=$TO
python3 print_results.py --model="cp" --obj="max-m" --to=$TO
python3 print_results.py --model="hw" --obj="min-w" --to=$TO
python3 print_results.py --model="hw" --obj="max-m" --to=$TO
python3 print_results.py --model="ha" --obj="min-w" --to=$TO
python3 print_results.py --model="ha" --obj="max-m" --to=$TO
python3 print_results.py --model="gd_b" --to=$TO
python3 print_results.py --model="gd_t" --to=$TO