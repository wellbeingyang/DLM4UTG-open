set -ex

MODEL=DiffuCoder-7B-cpGRPO
NAME=diffucoder
for LANG in python cpp java; do
    if [[ "$LANG" == "python" ]]; then
        length=128
    else
        length=192
    fi

    exec >> "outputs/output_${NAME}_${LANG}_accelerate.log"
    ns=(1 2 3 4 5 6 7)
    for i in "${ns[@]}"; do
        echo "********************"
        echo "n = ${i} with acceleration"
        python testeval.py --n ${i} --model_path /ckpts/${MODEL} -l ${LANG} -t 1.5 --max_length ${length} --accelerate
    done

    exec >> "outputs/output_${NAME}_${LANG}_baseline.log"
    ns=(1 2 3 4 5 6 7)
    for i in "${ns[@]}"; do
        echo "********************"
        echo "n = ${i} without acceleration"
        python testeval.py --n ${i} --model_path /ckpts/${MODEL} -l ${LANG} -t 1.5 --max_length ${length}
    done
done

MODEL=Dream-v0-Instruct-7B
NAME=dream
for LANG in python cpp java; do
    if [[ "$LANG" == "python" ]]; then
        length=128
    else
        length=192
    fi

    exec >> "outputs/output_${NAME}_${LANG}_accelerate.log"
    ns=(1 2 3 4 5 6 7)
    for i in "${ns[@]}"; do
        echo "********************"
        echo "n = ${i} with acceleration"
        python testeval.py --n ${i} --model_path /ckpts/${MODEL} -l ${LANG} -t 1.0 --max_length ${length} --accelerate
    done

    exec >> "outputs/output_${NAME}_${LANG}_baseline.log"
    ns=(1 2 3 4 5 6 7)
    for i in "${ns[@]}"; do
        echo "********************"
        echo "n = ${i} without acceleration"
        python testeval.py --n ${i} --model_path /ckpts/${MODEL} -l ${LANG} -t 1.0 --max_length ${length}
    done
done

