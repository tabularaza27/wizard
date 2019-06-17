import os
from typing import List, Tuple

import numpy as np
import tensorflow.keras as K

from agents.featurizers import Featurizer
from game_engine.card import Card

logdir = '/logs'

class Predictor:
    """Predictor object, predicts the number of tricks achieved in a round.

    Attributes:
        x_dim (int): The input shape of the NN. Equal to
            - 4 * 13 = 52 for numbered color cards
            - 2 for wizards & jesters
            - 5 for trump colors (4 colors + no trump)
            - 1 for the prediction
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
        train_step (int): How many samples should be recorded before a training step is executed.
        verbose (bool): Determines if information about the prediction performance should be printed
        keep_models_fixed: If set to true, the NN is not trained
    """

    def __init__(self, model_path='prediction_model', max_num_tricks=15,
                 train_batch_size=1000, train_step=300,
                 verbose=True, keep_models_fixed=False):
        self.max_num_tricks = max_num_tricks

        self.y_dim = self.max_num_tricks + 1
        self.x_dim = 59 + max_num_tricks + 1

        self._build_prediction_to_expected_num_points_matrix()

        self.train_step = train_step
        self.buffer_filled = False

        self.x_batch = np.zeros((train_batch_size, self.x_dim))
        self.y_batch = np.zeros((train_batch_size, self.y_dim))
        self.batch_position = 0
        self.train_batch_size = train_batch_size
        self.verbose = verbose
        self.keep_models_fixed = keep_models_fixed

        # keep track of current loss and acc of predictor for tensorboard plotting
        self.current_loss = None
        self.current_acc = None

        self.model_path = model_path + str(max_num_tricks) + '.h5'
        if os.path.isfile(self.model_path):
            self.model = K.models.load_model(self.model_path)
        else:
            self._build_new_model()

        # stores the predictions made by the predictor (statistics)
        # 0 stores all the predictions, the other keys correspond to the number of cards
        self.predictions = {i: [] for i in range(0, 16)}

        # stores the absolute difference to the predictions (statistics)
        self.prediction_differences = []

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
            K.layers.Dense(128, input_dim=self.x_dim, activation='relu'),
            K.layers.BatchNormalization(),
            K.layers.Dense(64, activation='relu'),
            K.layers.BatchNormalization(),
            K.layers.Dense(32, activation='relu'),
            K.layers.Dense(self.y_dim, activation='softmax')
        ])

        self.model.compile(optimizer=K.optimizers.Adam(),
                           loss='categorical_crossentropy', metrics=['accuracy'])

    def save_model(self):
        self.model.save(self.model_path)

    def make_prediction(self, initial_cards: List[Card],
                        trump_color_card: Card) -> Tuple[np.ndarray, int]:
        """Predict the number of tricks based on initial cards + trump color.

        Args:
            initial_cards: The current hand of the agent
            trump_color_card: A card which has the trump color

        Returns: A tuple consisting of
            - The input used for the NN. Should be passed to
              add_game_result once the result is available
            - The predicted number of tricks based on
              whichever has the highest expected reward
        """

        x = np.concatenate((Featurizer.cards_to_arr(initial_cards),
                     Featurizer.color_to_bin_arr(trump_color_card)))

        X = np.tile(x, (self.y_dim, 1))

        trick_values = K.utils.to_categorical(np.arange(self.y_dim), num_classes=self.y_dim)

        X = np.hstack([X, trick_values])

        probability_distributions = self.model.predict(X)

        # dot product between same rows of both matrices
        expected_value = (self.prediction_to_points * probability_distributions).sum(axis=1)

        prediction = int(np.argmax(expected_value))
        self.predictions[0].append(prediction)
        self.predictions[len(initial_cards)].append(prediction)

        prediction_encoded = K.utils.to_categorical(prediction, num_classes=self.y_dim)
        x = np.append(x, prediction_encoded)

        return x, prediction

    def add_game_result(self, x: np.ndarray, num_tricks_achieved: int):
        """Adds the corresponding label to the cards & trump color in x.

        Also trains the NN if train_batch_size rounds have passed
        since the last training.

        Args:
            x: The result from make_prediction which has been called
                when the game started.
            num_tricks_achieved: The number of tricks achieved
                after the round which corresponds to the one
                passed to make_prediction before. Used as a label.
        """
        y = K.utils.to_categorical(num_tricks_achieved, num_classes=self.y_dim)

        prediction_encoded = x[-self.y_dim:]
        prediction = np.argmax(prediction_encoded)
        self.prediction_differences.append(abs(prediction - num_tricks_achieved))

        self.x_batch[self.batch_position] = x
        self.y_batch[self.batch_position] = y
        self.batch_position += 1

        # Train when train_step samples were reached
        if self.buffer_filled and self.batch_position % self.train_step == 0:
            if not self.keep_models_fixed:
                history = self.model.fit(self.x_batch, self.y_batch)
                # update predictors values of loss and acc --> used for tensorforce reporting
                self.current_acc = history.history['accuracy'][0]
                self.current_loss = history.history['loss'][0]
            else:
                self.current_loss, self.current_acc = \
                    self.model.evaluate(self.x_batch, self.y_batch)

        if self.batch_position == self.train_batch_size - 1:
            self.buffer_filled = True
            self.batch_position = 0

            if self.verbose:
                print("Mean Prediction: ", np.mean(self.predictions[0]))
                print("Std Prediction: ", np.std(self.predictions[0]))
                print("Abs Prediction difference: ", np.mean(self.prediction_differences))
            # self.predictions = {i: [] for i in range(0, 16)}
            # self.prediction_differences = []
