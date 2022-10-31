echo "Copy the constructed data folder from the main repo in the train folder"
echo "Unzip to form constructed data"

conda activate ctc_v1

pip3 install -r requirements.txt
python3 download_dependencies.py

# Unzip basic data
rm -rf data/topical_chat
unzip data/topical_chat.zip
rm -rf data/yelp
unzip data/yelp.zip

# For spacy in hallucination
python3 -m spacy download en_core_web_sm

# The bleurt codebase
cd models
rm -rf bleurt
git clone https://github.com/Abhitipu/bleurt
cd bleurt
pip3 install .
cd ..

# Modified bert-score
rm -rf bert_score
git clone https://github.com/Abhitipu/bert_score
cd bert_score
pip3 install .
cd ../../

# For downloading DMI ckpts
CKPT_LINK="https://iitkgpacin-my.sharepoint.com/:u:/g/personal/bishal_santra_iitkgp_ac_in/ERb1URQCFtlMmlRnPTsBFvEBkF_8uUezO007YLaSOW3jTg?e=Ts151M&download=1" \
MODEL_NAME_PATH="DMI-Base_Rob-10_Sep/model_best_auc.pth" \
bash preamble.sh

# For the disc model to use dmi
cd models
cp -r /content/ctc-gen-eval/train/models/bert_score/bert_score/my_utils .
cp /content/ctc-gen-eval/train/models/bert_score/bert_score/core.py .
cd ..
