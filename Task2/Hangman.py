import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import string

class HangmanGuessModel(nn.Module):
    def __init__(self, input_size_word=30, input_size_letters=26, hidden_size=128, output_size=26):
        super(HangmanGuessModel, self).__init__()
        self.word_fc = nn.Linear(input_size_word, hidden_size)
        self.guessed_fc = nn.Linear(input_size_letters, hidden_size)
        self.output_fc = nn.Linear(hidden_size, output_size)

    def create_guess_tensors(self, guessed_letters):
        tensor = torch.zeros((1, 26))
        for letter in guessed_letters:
            index = ord(letter) - ord('a')
            if 0 <= index < 26:
                tensor[0, index] = 1
        return tensor

    def preprocess_word(self, word):
        tensor = torch.zeros((1, 30))
        clean_word = word.replace(" ", "")
        for pos, letter in enumerate(clean_word):
            if letter == '_':
                tensor[0, pos] = 27
            else:
                tensor[0, pos] = ord(letter) - ord('a') + 1
        return tensor

    def forward(self, masked_word, guessed_letters):
        guessed_tensor = self.create_guess_tensors(guessed_letters)
        word_tensor = self.preprocess_word(masked_word)
        word_features = F.relu(self.word_fc(word_tensor))
        guessed_features = F.relu(self.guessed_fc(guessed_tensor))
        combined_features = word_features + guessed_features
        return self.output_fc(combined_features)


class CynapticsHangman:
    def __init__(self):
        self.guessed_letters = []
        self.lives_remaining = 6
        self.train_file = "train.txt"
        self.valid_file = "valid.txt"
        self.train_dict = self.build_dictionary(self.train_file)
        self.valid_dict = self.build_dictionary(self.valid_file)
        self.model = HangmanGuessModel().to(self.get_device())
        self.device = self.get_device()

    def get_device(self):
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def build_dictionary(self, file_path):
        with open(file_path, "r") as file:
            return file.read().splitlines()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        print(f"Model loaded from {path}")

    def guess(self, masked_word, lives_left):
        probabilities = self.model(masked_word, self.guessed_letters)
        mask = torch.ones_like(probabilities)
        for letter in self.guessed_letters:
            letter_index = ord(letter) - ord('a')
            mask[0, letter_index] = 0
        masked_probabilities = probabilities * mask
        predicted_index = torch.argmax(masked_probabilities).item()
        return chr(predicted_index + ord('a'))

    def return_status(self, word, masked_word, guessed_letter):
        if guessed_letter in word:
            masked_word = ''.join([c if c in self.guessed_letters + [guessed_letter] else '_' for c in word])
            if '_' in masked_word:
                return "ongoing", "Correct guess", masked_word
            return "success", "Word guessed", masked_word
        self.lives_remaining -= 1
        if self.lives_remaining == 0:
            return "failed", "No lives left", masked_word
        return "ongoing", "Wrong guess", masked_word

    def start_game(self, game_id, verbose=True):
        self.guessed_letters = [' ']
        word = random.choice(self.valid_dict)
        masked_word = '_' * len(word)
        if verbose:
            print(f"Game {game_id} started. Word: {' '.join(masked_word)}")
        while self.lives_remaining > 0:
            guess = self.guess(masked_word, self.lives_remaining)
            self.guessed_letters.append(guess)
            status, message, masked_word = self.return_status(word, masked_word, guess)
            if verbose:
                print(f"Guess: {guess}, {message}. Masked: {masked_word}")
            if status in ["success", "failed"]:
                return status == "success"

    def train(self, episodes=10):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        for episode in range(episodes):
            word = random.choice(self.train_dict)
            masked_word = '_' * len(word)
            self.guessed_letters = []
            self.lives_remaining = 6
            for _ in range(len(word) + 6):
                guess = self.guess(masked_word, self.lives_remaining)
                self.guessed_letters.append(guess)
                probabilities = self.model(masked_word, self.guessed_letters)
                target = torch.zeros((1, 26)).to(self.device)
                for letter in word:
                    target[0, ord(letter) - ord('a')] = 1
                loss = criterion(probabilities, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.save_model("hangman_model.pth")


# Example Usage
hangman = CynapticsHangman()
hangman.train(episodes=100)
hangman.load_model("hangman_model.pth")
win_count = 0
for i in range(10):
    if hangman.start_game(i, verbose=True):
        win_count += 1
print(f"Success Rate: {win_count / 10:.2f}")
