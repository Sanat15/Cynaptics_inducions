import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import string
import matplotlib.pyplot as plt

class HangmanGuessModel(nn.Module):
    def __init__(self, input_size_word=30, input_size_letters=26, hidden_size=128, output_size=26):
        super(HangmanGuessModel, self).__init__()

        # Embedding for words (letters + special tokens)
        self.word_embedding = nn.Embedding(28, hidden_size)  # 26 letters + '_' + space
        self.guessed_fc = nn.Linear(input_size_letters, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)  # LSTM for sequential data
        self.fc_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.output_fc = nn.Linear(hidden_size, output_size)

    def create_guess_tensors(self, guessed_letters):
        tensor = torch.zeros((1, 26))
        for letter in guessed_letters:
            index = ord(letter) - ord('a')
            if 0 <= index < 26:
                tensor[0, index] = 1
        return tensor

    def preprocess_word(self, word):
        indices = []
        for letter in word.replace(" ", ""):
            if letter == '_':
                indices.append(27)
            else:
                indices.append(ord(letter) - ord('a') + 1)
        return torch.tensor(indices, dtype=torch.long).unsqueeze(0)  # Add batch dimension

    def forward(self, masked_word, guessed_letters):
        word_indices = self.preprocess_word(masked_word)
        guessed_tensor = self.create_guess_tensors(guessed_letters)

        # Pass word through embedding and LSTM
        word_features = self.word_embedding(word_indices)
        lstm_out, _ = self.lstm(word_features)
        word_features = lstm_out[:, -1, :]  # Take the last LSTM output

        # Process guessed letters
        guessed_features = F.relu(self.guessed_fc(guessed_tensor))

        # Combine and pass through the final layers
        combined_features = torch.cat((word_features, guessed_features), dim=1)
        combined_features = F.relu(self.fc_combine(combined_features))
        return self.output_fc(combined_features)


class CynapticsHangman:
    def __init__(self):
        self.guessed_letters = []
        self.lives_remaining = 6
        self.train_file = "train.txt"
        self.valid_file = "valid.txt"
        self.train_dict = self.build_dictionary(self.train_file)
        self.valid_dict = self.build_dictionary(self.valid_file)
        self.device = self.get_device()
        self.model = HangmanGuessModel().to(self.device)

    def get_device(self):
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def build_dictionary(self, file_path):
        with open(file_path, "r") as file:
            lines = file.read().splitlines()
            if not lines:
                raise ValueError(f"The dictionary file at {file_path} is empty.")
            return lines

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        print(f"Model loaded from {path}")

    def generate_masked_variants(self, word):
        variants = []
        for _ in range(5):
            mask = ''.join('_' if random.random() < 0.5 else letter for letter in word)
            variants.append(mask)
        return variants

    def guess(self, masked_word, lives_left):
        probabilities = self.model(masked_word, self.guessed_letters)
        
        # Create a mask to exclude already guessed letters
        mask = torch.ones_like(probabilities)
        for letter in self.guessed_letters:
            index = ord(letter) - ord('a')
            if 0 <= index < 26:
                mask[0, index] = 0

        masked_probabilities = probabilities * mask

        # If no valid guess is available, randomly choose an unguessed letter
        if torch.max(masked_probabilities) == 0:
            remaining_letters = [
                chr(i + ord('a'))
                for i in range(26)
                if chr(i + ord('a')) not in self.guessed_letters
            ]
            guessed_letter = random.choice(remaining_letters)
        else:
            predicted_index = torch.argmax(masked_probabilities).item()
            guessed_letter = chr(predicted_index + ord('a'))

        return guessed_letter

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
        self.lives_remaining = 6

        if verbose:
            print(f"Game {game_id} started. Word: {' '.join(masked_word)}")

        while self.lives_remaining > 0:
            guess = self.guess(masked_word, self.lives_remaining)
            self.guessed_letters.append(guess)

            status, message, masked_word = self.return_status(word, masked_word, guess)

            if verbose:
                print(f"Guess: {guess}, {message}. Masked: {masked_word}")

            if status == "success":
                if verbose:
                    print(f"Game {game_id} won! Word: {word}")
                return True

            if status == "failed":
                if verbose:
                    print(f"Game {game_id} lost! Word: {word}")
                return False

        return False

    def train(self, episodes=100):
        print("Starting training...")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

        training_loss = []
        for episode in range(episodes):
            try:
                word = random.choice(self.train_dict)
                masked_variants = self.generate_masked_variants(word)

                for masked_word in masked_variants:
                    self.guessed_letters = []
                    self.lives_remaining = 6
                    max_tries = len(word) + 6

                    while '_' in masked_word and self.lives_remaining > 0 and max_tries > 0:
                        max_tries -= 1
                        guess = self.guess(masked_word, self.lives_remaining)
                        self.guessed_letters.append(guess)

                        probabilities = self.model(masked_word, self.guessed_letters)
                        target = torch.zeros((1, 26)).to(self.device)
                        for letter in word:
                            if letter not in self.guessed_letters:
                                target[0, ord(letter) - ord('a')] = 1

                        loss = criterion(probabilities, target)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    training_loss.append(loss.item())

                scheduler.step()

                if (episode + 1) % 10 == 0:
                    avg_loss = sum(training_loss[-10:]) / 10
                    print(f"Episode {episode + 1}/{episodes}: Avg Loss: {avg_loss:.4f}")

            except Exception as e:
                print(f"Error in episode {episode + 1}: {e}")

        self.save_model("improved_hangman_model.pth")
        print("Training complete.")


# Example Usage
hangman = CynapticsHangman()
hangman.train(episodes=100)
hangman.load_model("improved_hangman_model.pth")

win_count = 0
total_games = 10
for i in range(total_games):
    if hangman.start_game(i, verbose=True):
        win_count += 1

print(f"Success Rate: {win_count / total_games:.2f}")
