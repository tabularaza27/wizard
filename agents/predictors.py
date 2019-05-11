import os
from typing import List

import numpy as np
import tensorflow.keras as K

from agents.featurizers import Featurizer
from game_engine.card import Card

class Predictor:
    """Predictor object, predicts the number of tricks achieved in a round.

    Attributes:
        x_dim (int): The input shape of the NN. Equal to
            - 4 * 13 = 52 for numbered color cards
            - 2 for wizards & jesters
            - 5 for trump colors (4 colors + no trump)
        max_num_tricks (int): Determines the output shape of the NN and
            therefore restricts the possible number of tricks
            which can be predicted
        y_dim (int): The output shape of the NN.
            Equal to max_num_tricks + 1 (as 0 tricks can be predicted)
        x_batch (np.array[train_batch_size][x_dim]): Because we don't want to
            train the NN after each round, we store the data
            in these batch arrays and train it after train_batch_size rounds
        y_batch (np.array[train_batch_size][y_dim]):
            Same as x_batch for the labels
        batch_position (int): Determines our current position
            in x_batch, y_batch. Resets to 0 after train_batch_size rounds.
        model_path (str): The path to the file where the parameters etc.
            of the NN are stored
        model (keras.models.Model): The NN
    """

    x_dim = 59

    def __init__(self, model_path='prediction_model', max_num_tricks=15,
            train_batch_size=500):
        self.max_num_tricks = max_num_tricks
        self.y_dim = self.max_num_tricks + 1
        self._build_prediction_to_expected_num_points_matrix()

        self.x_batch = np.zeros((train_batch_size, Predictor.x_dim))
        self.y_batch = np.zeros((train_batch_size, self.y_dim))
        self.batch_position = 0
        self.train_batch_size = train_batch_size

        self.model_path = model_path + str(max_num_tricks) + '.h5'
        if os.path.isfile(self.model_path):
            self.model = K.models.load_model(self.model_path)
        else:
            self._build_new_model()

    def _build_prediction_to_expected_num_points_matrix(self):
        # We can describe the calculation from the output of the NN
        # (array of probabilities) to the array of expected points as a
        # matrix-vector-multiplication where the matrix describes for each
        # possible prediction we could make and each game outcome the
        # points we would get in this case. This function computes this matrix
        # once and stores it in self.prediction_to_points

        self.prediction_to_points = np.zeros((self.y_dim, self.y_dim))
        for actual_num_tricks in range(self.y_dim):
            for predicted_num_tricks in range(self.y_dim):
                difference = np.abs(predicted_num_tricks - actual_num_tricks)
                if difference == 0:
                    num_points = 20 + predicted_num_tricks * 10
                else:
                    num_points = -10 * difference
                self.prediction_to_points[actual_num_tricks] \
                    [predicted_num_tricks] = num_points

    def _build_new_model(self):
        self.model = K.Sequential([
            K.layers.Dense(32, input_dim=Predictor.x_dim, activation='relu'),
            K.layers.Dense(self.y_dim, activation='softmax')
        ])

        self.model.compile(optimizer=K.optimizers.Adam(),
            loss='categorical_crossentropy', metrics=['accuracy'])

    def save_model(self):
        self.model.save(self.model_path)

    def make_prediction(self, initial_cards: List[Card],
            trump_color_card: List[Card]) -> int:
        """Predict the number of tricks based on initial cards + trump color.

        Args:
            initial_cards: The current hand of the agent
            trump_color_card: A card which has the trump color

        Returns: The predicted number of tricks based on
            whichever has the highest expected reward
        """

        self.x = np.array(Featurizer.cards_to_arr(initial_cards) +
            Featurizer.color_to_bin_arr(trump_color_card))
        self.x_batch[self.batch_position] = self.x

        probability_distribution = self.model.predict(
            self.x.reshape(1, Predictor.x_dim)).T
        expected_num_points = self.prediction_to_points \
            @ probability_distribution
        return np.argmax(expected_num_points)

    def add_game_result(self, num_tricks_achieved: int):
        """Adds the corresponding label to the cards & trump color
        passed to make_prediction before.

        Also trains the NN if train_batch_size rounds have passed
        since the last training.

        Args:
            num_tricks_achieved: The number of tricks achieved
                after the round which corresponds to the one
                passed to make_prediction before. Used as a label.
        """

        self.y = K.utils.to_categorical(num_tricks_achieved,
            num_classes=self.y_dim)
        self.y_batch[self.batch_position] = self.y
        self.batch_position += 1

        if self.batch_position == self.train_batch_size - 1:
            self.model.fit(self.x_batch, self.y_batch)
            self.batch_position = 0
