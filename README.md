# LMSYS

Link to Kaggle competition here: https://www.kaggle.com/competitions/lmsys-chatbot-arena/leaderboard

Brief writeup here: https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/528288

We placed 19th place of 1849 teams with our Gemma 9B - Finetuned Gemma 9B using QLora with custom header and TTA

Changes to reach gold: Using the same pipeline we could have gotten gold with some simple changes changing truncation to left instead of right, and replacing TTA with a second model in which we flip the responses in the training data. So instead of training on prompt+response_a+response_b of the first model we use prompt+response_b+response_a

Files:
https://www.kaggle.com/models/emiz6413/gemma-2/Transformers/gemma-2-9b-it-4bit/1
https://www.kaggle.com/datasets/emiz6413/lmsys-wheel-files
https://www.kaggle.com/datasets/roschildrui/v38-cp2422

Hyperparameters: 
!python scripts/train_v4_freeze0_newheaderbf16.py --ver 39 --max_len 3072 --lr 1e-4 --freeze_layers 0 \
--batch_size 2 --grad_acc_steps 16 --warmup_steps 20 --save_steps 100 --eval_batch_size 4 \
--extra_data "/root/autodl-tmp/lmsys/data/lmsys-33k-deduplicated.csv" --lora_r 64 \
--n_splits 5 --lora_alpha 64
