from django.shortcuts import render, redirect
import sys
sys.path.append('../')
from game_engine.deck import Deck
from game_engine.card import Card
from django.views import View


class WizardGameManager(View):
    var = "Hello"

    def get(self,  request):
        print(self.var)
        print("Hello World")
        return render(request, 'start.html')

    def play_game(self, request):
        return redirect('play_round', round=1)


def play_round(request, round):
    deck = Deck()
    request.session['left_agent_cards'] = convert_pack_to_values(deck.draw(round))
    request.session['top_agent_cards'] = convert_pack_to_values(deck.draw(round))
    request.session['right_agent_cards'] = convert_pack_to_values(deck.draw(round))
    request.session['player_cards'] = convert_pack_to_values(deck.draw(round))
    trump = deck.draw(1)
    request.session['trump'] = convert_card_to_value(trump[0].color, trump[0].value)
    return redirect('ask_prediction', round=round)


def ask_prediction(request, round):
    left_agent_cards = convert_pack_to_cards(request.session['left_agent_cards'])
    top_agent_cards = convert_pack_to_cards(request.session['top_agent_cards'])
    right_agent_cards = convert_pack_to_cards(request.session['right_agent_cards'])
    player_cards = convert_pack_to_cards(request.session['player_cards'])
    trump_card = convert_value_to_card(request.session['trump'])
    return render(request, 'game.html', {'left_agent_cards': left_agent_cards, 'top_agent_cards': top_agent_cards,
                                         'right_agent_cards': right_agent_cards, 'player_cards': player_cards,
                                         'prediction_phase': True, 'round': round, 'width': round*66,
                                         'height': round*50, 'trump_card': trump_card,
                                         'prediction_range': range(0, round+1)})


def get_prediction(request, round, prediction):
    pass


def play_trick(request, round, trick):
    pass


def convert_pack_to_values(cards):
    result = []
    for card in cards:
        result.append(convert_card_to_value(card.color, card.value))
    return result


def convert_pack_to_cards(values):
    result = []
    for value in values:
        result.append(convert_value_to_card(value))
    return result


def convert_card_to_value(color, value):
    if color == "Red":
        return value
    elif color == "Green":
        return 1 * 13 + value
    elif color == "Blue":
        return 2 * 13 + value
    elif color == "Yellow":
        return 3 * 13 + value
    elif color == "White":
        if value == 0:
            return 53
        else:
            return 54
    else:
        return ValueError


def convert_value_to_card(value):
    if value == 54:
        return Card(color="White", value=14)
    elif value == 53:
        return Card(color="White", value=0)
    else:
        if value > 39:
            return Card(color="Yellow", value=(value - (13*3)))
        elif value > 26:
            return Card(color="Blue", value=(value - (13 * 2)))
        elif value > 13:
            return Card(color="Green", value=(value - (13 * 1)))
        else:
            return Card(color="Red", value=value)
