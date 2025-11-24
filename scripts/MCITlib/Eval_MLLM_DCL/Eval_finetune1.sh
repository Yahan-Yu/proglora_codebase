# #!/bin/bash

TASK_ID=$1
HARD_PATH=/home/yahan/code/MCITlib

if [ "$TASK_ID" == "1" ]; then
    pip install -e .
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_rs.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/RS.json $HARD_PATH/configs/train_configs/proglora/LLaVA/MLLM-DCL/eval/task1.json 0
elif [ "$TASK_ID" == "2" ]; then
    pip install -e .
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_med.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/Med.json $HARD_PATH/configs/train_configs/proglora/LLaVA/MLLM-DCL/eval/task2.json 1
    pip install -e .
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_rs.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/RS.json $HARD_PATH/configs/train_configs/proglora/LLaVA/MLLM-DCL/eval/task2.json 1
elif [ "$TASK_ID" == "3" ]; then
    pip install -e .
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_med.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/Med.json $HARD_PATH/configs/train_configs/proglora/LLaVA/MLLM-DCL/eval/task3.json 2
    pip install -e .
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_rs.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/RS.json $HARD_PATH/configs/train_configs/proglora/LLaVA/MLLM-DCL/eval/task3.json 2
    pip install -e .
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_ad.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/AD.json $HARD_PATH/configs/train_configs/proglora/LLaVA/MLLM-DCL/eval/task3.json 2
elif [ "$TASK_ID" == "4" ]; then
    pip install -e .
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_ad.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/AD.json $HARD_PATH/configs/train_configs/proglora/LLaVA/MLLM-DCL/eval/task4.json 3
    pip install -e .
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_rs.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/RS.json $HARD_PATH/configs/train_configs/proglora/LLaVA/MLLM-DCL/eval/task4.json 3
    pip install -e .
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_med.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/Med.json $HARD_PATH/configs/train_configs/proglora/LLaVA/MLLM-DCL/eval/task4.json 3
    pip install -e .
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_sci.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/Sci.json $HARD_PATH/configs/train_configs/proglora/LLaVA/MLLM-DCL/eval/task4.json 3
else
    pip install -e .
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_ad.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/AD.json $HARD_PATH/configs/train_configs/proglora/LLaVA/MLLM-DCL/eval/task5.json 4
    pip install -e .
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_rs.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/RS.json $HARD_PATH/configs/train_configs/proglora/LLaVA/MLLM-DCL/eval/task5.json 4
    pip install -e .
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_med.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/Med.json $HARD_PATH/configs/train_configs/proglora/LLaVA/MLLM-DCL/eval/task5.json 4
    pip install -e .
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_sci.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/Sci.json $HARD_PATH/configs/train_configs/proglora/LLaVA/MLLM-DCL/eval/task5.json 4
    pip install -e .
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_fin.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/Fin.json $HARD_PATH/configs/train_configs/proglora/LLaVA/MLLM-DCL/eval/task5.json 4
fi