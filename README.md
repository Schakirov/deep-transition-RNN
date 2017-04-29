Code is based on https://github.com/yoonkim/lstm-char-cnn

I added options:
1) possibility to use the idea of deep-transition RNN/LSTM in several possible variants. This idea is taken from "How to Construct Deep Recurrent Neural Networks" by Razvan Pascanu, Caglar Gulcehre, Kyunghyun Cho, Yoshua Bengio. 
2) possibility to use hierarchical RNN/LSTM
3) some convenient functional for preprocessing of dataset texts
4) my own ideas of visualization of progress (f.e. writing the number of guesses which RNN takes to guess next word)

For this, I changed mainly file "main.lua" and created some new files in /util (those with "dt" in their name).
