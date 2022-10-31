# convert our alignment datasets for training our models for the regression (R) approach
#python data_utils/convert_constructed_data_to_bleurt_format.py --data_path constructed_data/xsum/example.json --dataset_name xsum --aggr_type mean --train_pct 0.9
#python data_utils/convert_constructed_data_to_bleurt_format.py --data_path constructed_data/cnndm/example.json --dataset_name cnndm --aggr_type mean --train_pct 0.9
#python data_utils/convert_constructed_data_to_bleurt_format.py --data_path constructed_data/cnndm_ref/examples.json --dataset_name cnndm_ref --aggr_type mean --reverse_cand_ref --train_pct 0.9
#python data_utils/convert_constructed_data_to_bleurt_format.py --data_path constructed_data/yelp/example.json --dataset_name cnndm --aggr_type mean --train_pct 0.9
python data_utils/convert_constructed_data_to_bleurt_format.py --data_path constructed_data/persona_chat/example.json --dataset_name persona_chat --aggr_type sum --remove_stopwords --dialog_context fact_history --train_pct 0.9
python data_utils/convert_constructed_data_to_bleurt_format.py --data_path constructed_data/persona_chat_fact/examples.json --dataset_name persona_chat_fact --aggr_type sum --remove_stopwords --dialog_context fact --train_pct 0.9
python data_utils/convert_constructed_data_to_bleurt_format.py --data_path constructed_data/topical_chat/example.json --dataset_name topical_chat --aggr_type sum --remove_stopwords --dialog_context fact_history --train_pct 0.9
python data_utils/convert_constructed_data_to_bleurt_format.py --data_path constructed_data/topical_chat_fact/examples.json --dataset_name topical_chat_fact --aggr_type sum --remove_stopwords --dialog_context fact --train_pct 0.9

# trains our regression (R) model with the BLEURT codebase (https://github.com/google-research/bleurt)
STEPS=15000
BERT_CKPT=/content/ctc-gen-eval/train/ckpt_new

DATASET=persona_chat
CONTEXT=fact_history
MY_BLEURT_CKPT=/content/drive/MyDrive/bleurt_ckpt/${DATASET}
TRAIN_PATH=/content/ctc-gen-eval/train/bleurt_data/${DATASET}/${CONTEXT}_sum_remove_stopwords_train.jsonl
DEV_PATH=/content/ctc-gen-eval/train/bleurt_data/${DATASET}/${CONTEXT}_sum_remove_stopwords_dev.jsonl
python -m bleurt.finetune \
  -init_checkpoint=${BERT_CKPT}/roberta_base.ckpt \
  -bert_config_file=${BERT_CKPT}/bert_config.json \
  -vocab_file=${BERT_CKPT}/vocab.txt \
  -model_dir=${MY_BLEURT_CKPT} \
  -train_set=${TRAIN_PATH} \
  -dev_set=${DEV_PATH} \
  -num_train_steps=${STEPS} \
  -max_seq_length=512 \
  -batch_size=6 \
  -do_lower_case=true \
  -export_metric=correlation \
  -keep_checkpoint_max=15


DATASET=persona_chat_fact
CONTEXT=fact
MY_BLEURT_CKPT=/content/drive/MyDrive/bleurt_ckpt/${DATASET}
TRAIN_PATH=/content/ctc-gen-eval/train/bleurt_data/${DATASET}/${CONTEXT}_sum_remove_stopwords_train.jsonl
DEV_PATH=/content/ctc-gen-eval/train/bleurt_data/${DATASET}/${CONTEXT}_sum_remove_stopwords_dev.jsonl
python -m bleurt.finetune \
  -init_checkpoint=${BERT_CKPT}/roberta_base.ckpt \
  -bert_config_file=${BERT_CKPT}/bert_config.json \
  -vocab_file=${BERT_CKPT}/vocab.txt \
  -model_dir=${MY_BLEURT_CKPT} \
  -train_set=${TRAIN_PATH} \
  -dev_set=${DEV_PATH} \
  -num_train_steps=${STEPS} \
  -max_seq_length=512 \
  -batch_size=6 \
  -do_lower_case=true \
  -export_metric=correlation \
  -keep_checkpoint_max=15

DATASET=topical_chat
CONTEXT=fact_history
MY_BLEURT_CKPT=/content/drive/MyDrive/bleurt_ckpt/${DATASET}
TRAIN_PATH=/content/ctc-gen-eval/train/bleurt_data/${DATASET}/${CONTEXT}_sum_remove_stopwords_train.jsonl
DEV_PATH=/content/ctc-gen-eval/train/bleurt_data/${DATASET}/${CONTEXT}_sum_remove_stopwords_dev.jsonl
python -m bleurt.finetune \
  -init_checkpoint=${BERT_CKPT}/roberta_base.ckpt \
  -bert_config_file=${BERT_CKPT}/bert_config.json \
  -vocab_file=${BERT_CKPT}/vocab.txt \
  -model_dir=${MY_BLEURT_CKPT} \
  -train_set=${TRAIN_PATH} \
  -dev_set=${DEV_PATH} \
  -num_train_steps=${STEPS} \
  -max_seq_length=512 \
  -batch_size=6 \
  -do_lower_case=true \
  -export_metric=correlation \
  -keep_checkpoint_max=15


DATASET=topical_chat_fact
CONTEXT=fact
MY_BLEURT_CKPT=/content/drive/MyDrive/bleurt_ckpt/${DATASET}
TRAIN_PATH=/content/ctc-gen-eval/train/bleurt_data/${DATASET}/${CONTEXT}_sum_remove_stopwords_train.jsonl
DEV_PATH=/content/ctc-gen-eval/train/bleurt_data/${DATASET}/${CONTEXT}_sum_remove_stopwords_dev.jsonl
python -m bleurt.finetune \
  -init_checkpoint=${BERT_CKPT}/roberta_base.ckpt \
  -bert_config_file=${BERT_CKPT}/bert_config.json \
  -vocab_file=${BERT_CKPT}/vocab.txt \
  -model_dir=${MY_BLEURT_CKPT} \
  -train_set=${TRAIN_PATH} \
  -dev_set=${DEV_PATH} \
  -num_train_steps=${STEPS} \
  -max_seq_length=512 \
  -batch_size=6 \
  -do_lower_case=true \
  -export_metric=correlation \
  -keep_checkpoint_max=15

