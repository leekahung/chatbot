# Chatbot
 In the ever expanding space of virtual assistants and automated customer support, chatbots has garnered  
enormous attention over the past decade from industry giants like Apple or Amazon to up-and-coming local  
business that wishes to streamline customer services online. Many chatbots today utilizes a "retrieval-  
based" approach, where the program would use machine learning or deep learning models to identify the  
most appropriate response for a given reply. Although typically pre-defined, retrieval-based chatbots can  
become quite sohpisticated given a sufficiently large dataset. (See [Kuki AI](https://www.kuki.ai/))  

However, this kind of chatbot couldn't generate new responses on its own. Therefore, I plan on developing  
a chatbot utilizing a "generative-based" approach instead and one that could carry out casual conversations  
on its own without retrieval.

# 1. Data
 The main goal of this capstone project is to produce a workable chatbot with the ability to "speak" with  
its users. The dataset used for this Jupyter notebook is collected from the following website:  
 * [Cornell Movie Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html.)

The dataset is published in 2011 by Cristian Danescu-Niculescu-Mizil and Lillian Lee of Cornell University  
for the Proceedings of the Workshop on Cognitive Modeling and Computational Linguistics at the Association  
of Computational Linguistics (ACL).

Citation: Cristian Danescu-Niculescu-Mizil and Lillian Lee. 2011. Chameleons in imagined conversations: A  
new approach to understanding coordination of linguistic style in dialogs. In Proceedings of the Workshop  
on Cognitive Modeling and Computational Linguistics, ACL 2011, pp.76-87.

# 2. Model
 The Model that I'll be using for my generative chatbot would be the Seq2Seq model. Seq2Seq is learning  
model that converts sequences from one domain (e.g. an utterance) to sequences in another domain (e.g.  
a reply or response). This model can be used when you need to generate new text, which is something that  
we want for our chatbot.

A common approach to Seq2Seq is to create a Recurrent Neural Network (RNN) with Long Short-Term  
Memory (or LSTM). LSTMs are a special kind of RNN that could encode input sequences and decode output  
sequences.

![Image of LSTM Encoder and Decoder](https://miro.medium.com/max/804/1*1P-cOZ5rqBLZdfQ8p-IFpA.jpeg)

A newer approach utilizes Transformers as opposed to RNNs. Transformers are specifically designed to handle  
sequential input data, such as natural language. However, unlike RNNs, transformers does not process the  
data in sequential order, allowing for parallization, which reduces training times. Transformers are the  
model of choice for NLP problems, most famous for its development of BERT (Bidirectional Encoder Representations  
from Transformers) and GPT (Generative Pre-trained Transformer).

![Image of BERT and GPT-2](https://lh4.googleusercontent.com/PtV2hzpn2OqLONB3JxvbzLjJz6GURCVM7xqUJ9I4hgCZUof5ci11FUthQVo9bzbpJU3aivGYQ9jQ3Wj2KF4vQt9pVQzbVtpO058KaSc_39ztS7y0QSnPPwSYPessubzsRNTHGeJq)

While transformers are the way to go, I start with LSTMs, applying what I know now at this moment into a  
working model. I do plan on implementing transformers in the future.

# 3. Data Cleaning
 * [Data Cleaning Notebook](https://github.com/leekahung/chatbot/blob/main/notebooks/movie_dialogues_cleaning.ipynb)

In order for me to implement the Seq2Seq model above using LSTMs, I would first have to clean and process the  
data from the Cornell Movie Dialogs Corpus. The desirable format for the Seq2Seq is to separate dialogues into  
comments (input sequences) and replies (output sequences). Here, as you would see in the notebook, we utilized  
the list of dialogue orderings from one of our dataset files "movie_conversations.txt" and call the specific  
dialogue line from "movie_lines.txt" to the ordering. To do this, we first must generate pairs of dialogue  
orderings from the original ordering of dialogues. A customized function was used such that a list of orderings  
would be separated into sequential pairs until the end of the list is reached.

 * Example of function: List of Movie Ordering [Line1, Line2, Line3, Line4] -> List of Dialogue Pairs [[Line1,Line2],[Line2,Line3],[Line3,Line4]] 

The first element of the pair would be the comment and the second element of the pair would be the reply.  

After unnesting and placing the dialogue pairs into a fresh pandas' DataFrame, the text was further cleaned by  
removing text modifiers, strings like "\<b>" or "\<\/b>", that came along with the corpus. Special characters  
like non-ASCII characters were replaced with dashes, and a few special cases were replaced (i.e. strange  
formatting like <\u<, misspellings like mrcroscope, or censorings like G-d). Before we strip the rest of the  
text of puncuations, word contractions like "I'm" or "they'll" were also expanded into their full form ("I am"  
and "they will"). Texts with more than roughly 3 times the standard deviation of words (50 words) were removed  
from our dataset to prevent rambling as our end goal are conversations.

# 4. Exploratory Data Analysis (EDA)
 * [EDA Notebook](https://github.com/leekahung/chatbot/blob/main/notebooks/movie_dialogue_pairs_eda.ipynb)

For our EDA, we've utilized the NLTK library for our natural language processing. Two sets of data were made,  
one with stop-words and one without stop-words. The distinction was made with the intent that our chatbot  
doesn't sound to robot-like. Any rows with empty comments of replies post-processing were dropped as they'll  
have little value for training our chatbot.

# 5. Data Preprocessing and Modeling
 * [Data Preprocessing Notebook](https://github.com/leekahung/chatbot/blob/main/notebooks/movie_dialogues_preprocessing.ipynb)
 * [Data Modeling Notebook](https://github.com/leekahung/chatbot/blob/main/notebooks/movie_dialogues_modeling.ipynb)

The original intent for this Capstone was to use the full corpus for training, validating, and testing our  
chatbot. However, it soon became apparent, that it would not be possible. While we have a decently sized  
dataset at this point (around 200k), training neural networks can be quite computing intensive. The training  
model for this Capstone alone took over 2 days and that with only 6000 pairs. Thus, we made due with what  
we have in terms of computing power.

For the dataset, it was split up using Sci-kit Learn's train_test_split (test_size=0.25 and random_state=7)  
before we selected the first 6000 pairs from X_train and y_train. A 10% validation split was also used for  
the model training (train on 5400 samples, validate on 600 samples) for 200 epochs. Early Stopping was also  
used (monitor='acc', patience=5) in case the training accuracy stopped improving.

However, as we see below, the accuracy aren't exactly the greatest (16.32% training, 1.60 validation). The  
model itself appears to be overfitting.

![Image of Model Accuracy and Loss](https://github.com/leekahung/chatbot/blob/main/images/model_acc_and_loss.png)

Nevertheless, the Seq2Seq model with LSTM was able to generate human readable text, albeit somewhat strange  
reponses from the chatbot.

![Image of Chatbot talking](https://github.com/leekahung/chatbot/blob/main/images/chatbot.png)

# 6. Conclusion and Future Improvements

Despite the fact that it has only been trained on 5400 data points (600 validation), the chatbot shows signs  
that it could potentially carry out a coversation (see above). With additional training, the model may be able  
to smooth out any weird kinks. Additionally, other neural network architectures, like transformers, could also  
be tested using this dataset.

# Original Roadmap
[Proposed roadmap](https://docs.google.com/document/d/1Opvs5nyCXC_f0TujGuuRu_xIB5joNCbSddMkL91Rgfk/edit)

# Credits
Special thanks to my mentor Ricardo Alanís for his continued support and mentorship. The advice given had  
always been helpful during my journey into Data Science and AI.
