# #!/bin/bash

TASK_ID=$1
HARD_PATH=/home/yahan/code/MCITlib

if [ "$TASK_ID" == "1" ]; then
    pip install -e .
    bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/ImageNet-R.json $HARD_PATH/configs/train_configs/proglora/LLaVA/UCIT/eval/task1.json 0
elif [ "$TASK_ID" == "2" ]; then
    pip install -e .
    bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/ImageNet-R.json $HARD_PATH/configs/train_configs/proglora/LLaVA/UCIT/eval/task2.json 1
    pip install -e .
    bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/ArxivQA.json $HARD_PATH/configs/train_configs/proglora/LLaVA/UCIT/eval/task2.json 1
elif [ "$TASK_ID" == "3" ]; then
    pip install -e .
    bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/ImageNet-R.json $HARD_PATH/configs/train_configs/proglora/LLaVA/UCIT/eval/task3.json 2
    pip install -e .
    bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/ArxivQA.json $HARD_PATH/configs/train_configs/proglora/LLaVA/UCIT/eval/task3.json 2
    pip install -e .
    bash scripts/MCITlib/Eval_UCIT/eval_vizwiz.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/VizWiz.json $HARD_PATH/configs/train_configs/proglora/LLaVA/UCIT/eval/task3.json 2
elif [ "$TASK_ID" == "4" ]; then
    pip install -e .
    bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/ImageNet-R.json $HARD_PATH/configs/train_configs/proglora/LLaVA/UCIT/eval/task4.json 3
    pip install -e .
    bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/ArxivQA.json $HARD_PATH/configs/train_configs/proglora/LLaVA/UCIT/eval/task4.json 3
    pip install -e .
    bash scripts/MCITlib/Eval_UCIT/eval_vizwiz.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/VizWiz.json $HARD_PATH/configs/train_configs/proglora/LLaVA/UCIT/eval/task4.json 3
    pip install -e .
    bash scripts/MCITlib/Eval_UCIT/eval_iconqa.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/IconQA.json $HARD_PATH/configs/train_configs/proglora/LLaVA/UCIT/eval/task4.json 3
elif [ "$TASK_ID" == "5" ]; then
    pip install -e .
    bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/ArxivQA.json $HARD_PATH/configs/train_configs/proglora/LLaVA/UCIT/eval/task5.json 4
    pip install -e .
    bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/ImageNet-R.json $HARD_PATH/configs/train_configs/proglora/LLaVA/UCIT/eval/task5.json 4
    pip install -e .
    bash scripts/MCITlib/Eval_UCIT/eval_vizwiz.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/VizWiz.json $HARD_PATH/configs/train_configs/proglora/LLaVA/UCIT/eval/task5.json 4
    pip install -e .
    bash scripts/MCITlib/Eval_UCIT/eval_iconqa.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/IconQA.json $HARD_PATH/configs/train_configs/proglora/LLaVA/UCIT/eval/task5.json 4
    pip install -e .
    bash scripts/MCITlib/Eval_UCIT/eval_clevr.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/CLEVR-Math.json $HARD_PATH/configs/train_configs/proglora/LLaVA/UCIT/eval/task5.json 4
else
    pip install -e .
    bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/ImageNet-R.json $HARD_PATH/configs/train_configs/proglora/LLaVA/UCIT/eval/task6.json 5
    pip install -e .
    bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/ArxivQA.json $HARD_PATH/configs/train_configs/proglora/LLaVA/UCIT/eval/task6.json 5
    pip install -e .
    bash scripts/MCITlib/Eval_UCIT/eval_vizwiz.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/VizWiz.json $HARD_PATH/configs/train_configs/proglora/LLaVA/UCIT/eval/task6.json 5
    pip install -e .
    bash scripts/MCITlib/Eval_UCIT/eval_iconqa.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/IconQA.json $HARD_PATH/configs/train_configs/proglora/LLaVA/UCIT/eval/task6.json 5
    pip install -e .
    bash scripts/MCITlib/Eval_UCIT/eval_clevr.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/CLEVR-Math.json $HARD_PATH/configs/train_configs/proglora/LLaVA/UCIT/eval/task6.json 5
    pip install -e .
    bash scripts/MCITlib/Eval_UCIT/eval_flickr30k.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/Flickr30k.json $HARD_PATH/configs/train_configs/proglora/LLaVA/UCIT/eval/task6.json 5
fi