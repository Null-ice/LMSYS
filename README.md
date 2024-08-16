# LMSYS

Link to Kaggle competition here: https://www.kaggle.com/competitions/lmsys-chatbot-arena/leaderboard

Brief writeup here: https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/528288

We placed 19th place of 1849 teams with our Gemma 9B - Finetuned Gemma 9B using QLora with custom header and TTA

Changes to reach gold: Using the same pipeline we could have gotten gold with some simple changes changing truncation to left instead of right, and replacing TTA with a second model in which we flip the responses in the training data. So instead of training on prompt+response_a+response_b of the first model we use prompt+response_b+response_a
