# python finetune.py \
#     --dataset_name [xsum/persona_chat/...] \
#     --n_epochs [1 by default] \
#     --dialog_context [fact/history/fact_history/history_fact]

TOKENIZERS_PARALLELISM=false python finetune.py \
    --dataset_name persona_chat \
    --n_epochs 1 \
    --dialog_context fact

TOKENIZERS_PARALLELISM=false python finetune.py \
    --dataset_name persona_chat_fact \
    --n_epochs 1 \
    --dialog_context fact

TOKENIZERS_PARALLELISM=false python finetune.py \
    --dataset_name persona_chat \
    --n_epochs 1 \
    --dialog_context fact_history

TOKENIZERS_PARALLELISM=false python finetune.py \
    --dataset_name persona_chat_fact \
    --n_epochs 1 \
    --dialog_context fact_history

TOKENIZERS_PARALLELISM=false python finetune.py \
    --dataset_name topical_chat \
    --n_epochs 1 \
    --dialog_context fact

TOKENIZERS_PARALLELISM=false python finetune.py \
    --dataset_name topical_chat_fact \
    --n_epochs 1 \
    --dialog_context fact

TOKENIZERS_PARALLELISM=false python finetune.py \
    --dataset_name topical_chat \
    --n_epochs 1 \
    --dialog_context fact_history

TOKENIZERS_PARALLELISM=false python finetune.py \
    --dataset_name topical_chat_fact \
    --n_epochs 1 \
    --dialog_context fact_history
