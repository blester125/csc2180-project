mkdir -p results

TRIALS=5
SAMPLES=(5 10 50 100 200 500 1000 2000 5000)

for trial in `seq $TRIALS`; do
    for sample in ${SAMPLES[@]}; do
        python one_vs_one_mnist.py --sample ${sample} | tee "results/result-${sample}-${trial}.txt"
    done
    python one_vs_one_mnist.py | tee "results/result-full-${trial}"
done
