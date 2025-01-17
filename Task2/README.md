This is the Bonus Task Submission Hangman game .

Here is the summary of the model used

CynapticsHangman Model Summary

----------------------------------------

Model Architecture: HangmanGuessModel

Device: cpu

Number of Lives: 6

Dictionary Size: 22730

Guessed Letters: 


Total Parameters: 175,386

Trainable Parameters: 175,386


Model Structure:

HangmanGuessModel(

  (word_embedding): Embedding(28, 128)
  
  (guessed_fc): Linear(in_features=26, out_features=128, bias=True)
  
  (lstm): LSTM(128, 128, batch_first=True)
  
  (fc_combine): Linear(in_features=256, out_features=128, bias=True)
  
  (output_fc): Linear(in_features=128, out_features=26, bias=True)
  
)
