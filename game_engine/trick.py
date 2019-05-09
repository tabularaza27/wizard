from game_engine.card import Card


class Trick(object):
    def __init__(self, trump_card, players, first_player, played_cards_in_game):
        self.trump_card = trump_card
        self.players = players
        self.first_player = first_player
        self.first_card = None
        self.played_cards_in_game = played_cards_in_game

    def play(self):
        winner = None
        num_players = len(self.players)
        trick_cards = []
        for i in range(num_players):
            player_index = (self.first_player+i) % num_players
            # Start with the first player and ascend, then reset at 0.
            player = self.players[player_index]
            played_card = player.play_card(self.trump_card, self.first_card,
                                           trick_cards, self.players,
                                           self.played_cards_in_game)
            trick_cards.append(played_card)
            if self.first_card is None and played_card.value != 0:
                self.first_card = played_card
            if winner is None or Trick.is_new_winner(played_card, winner[0],
                                               self.trump_card,
                                               self.first_card):
                winner = (played_card, player_index)
            """print("First card: {}\nTrump card: {}\nWinning: {}".format(self.first_card,
                                                                       self.trump_card,
                                                                       winner))"""
        return winner[1], trick_cards

    @staticmethod
    def is_new_winner(new_card, old_card, trump, first_card):
        """
        Returns True if the new_card wins, taking into account trump
        colors, first_card color and order.

        :param new_card: Card played LATER.
        :param old_card: Current winning Card.
        :param trump: Trump card.
        :param first_card: First card played. May be None.
        :return: The winning card.
        """
        # If a Z was played first, it wins.
        if old_card.value == 14:
            return False
        # If not and the new card is a Z, the new card wins.
        if new_card.value == 14:
            return True
        # First N wins, so if the second card is N, it always wins.
        if new_card.value == 0:
            return False
        # Second N wins only if new_card is NOT N.
        elif old_card.value == 0:
            return True
        # If they are both colored cards, the trump color wins.
        if old_card.color == trump.color:
            if new_card.color != trump.color:
                return False
            else:  # If both are trump color, the higher value wins.
                return old_card.value < new_card.value
        else:
            # old_card is not trump color, then if new_card is, new_card wins
            if new_card.color == trump.color:
                return True
            else:
                # Neither are trump color, so check for first color.
                if old_card.color == first_card.color:
                    if new_card.color != first_card.color:
                        # old card is first_card color but new card is not, old wins.
                        return False
                    else:
                        # Both are first_card color, bigger value wins.
                        return old_card.value < new_card.value
