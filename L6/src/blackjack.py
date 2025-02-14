import random

class Card:
    """
    Represents a single playing card in Blackjack.
    Face values: 2..10, 10->(J, Q, K), 1->(A)
    """
    def __init__(self, value):
        # value in [1..10], where 1 can stand for an Ace, and 10 can stand for 10/J/Q/K
        self.value = value

    def __str__(self):
        return f"Card({self.value})"


class Hand:
    """
    Represents a player's or dealer's hand.
    """
    def __init__(self):
        self.cards = []

    def add_card(self, card):
        self.cards.append(card)

    def get_values(self):
        """
        Return possible totals for the hand.
        Typically a hand has two possible totals if it includes an Ace.
        """
        total = 0
        ace_count = 0
        for card in self.cards:
            if card.value == 1:  # Ace
                ace_count += 1
                total += 1
            else:
                total += card.value

        # We can treat Aces as 1 or 11. 
        # We'll try to use as many aces as 11 as possible without busting.
        totals = [total]
        for _ in range(ace_count):
            new_totals = []
            for t in totals:
                # Convert one of the Aces from 1 to 11 (+10 difference)
                if t + 10 <= 21:
                    new_totals.append(t + 10)
            totals.extend(new_totals)

        return sorted(set(totals))  # unique possible totals, sorted
    
    def get_best_value(self):
        """
        Return the highest valid total <= 21, or the lowest total if all are bust.
        """
        values = self.get_values()
        valid_values = [v for v in values if v <= 21]
        if not valid_values:
            return min(values)  # bust
        return max(valid_values)

    def is_bust(self):
        return self.get_best_value() > 21

    def __str__(self):
        return f"Hand({[str(card) for card in self.cards]})"
    
    def __repr__(self):
        return self.__str__()
    
    
class BlackjackEnv:
    """
    A simplified Blackjack environment for DP.
    We track only (player_sum, dealer_upcard, usable_ace) as states.
    """
    def __init__(self, seed:int=None):
        """
        Initialize the Blackjack environment

        Args:
            seed (int, optional): Random seed for shuffling the deck. Defaults to None.
        """

        self.deck = self._generate_deck()
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.deck)

        self.player_hand = Hand()
        self.dealer_hand = Hand()

        # Terminal flag to indicate if game is finished
        self.done = False
        self.reward = 0

    def _generate_deck(self):
        """
        Generate a deck of 52 cards (no jokers)

        Returns:
            List[Card]: A list representing the deck of cards.
        """
        # 4 suits * 13 ranks, but face cards (J, Q, K) are all 10
        # Ace = 1
        # Cards 2..10 as normal
        deck = []
        for _ in range(4):  # suits
            for value in range(1, 14):  # 1..13
                if value > 10:
                    card_value = 10
                else:
                    card_value = value
                deck.append(Card(card_value))
        return deck

    def reset(self):
        if len(self.deck) < 15:  # reshuffle if deck is low
            self.deck = self._generate_deck()
            random.shuffle(self.deck)

        self.player_hand = Hand()
        self.dealer_hand = Hand()
        self.done = False
        self.reward = 0

        # Initial deal
        self.player_hand.add_card(self.deck.pop())
        self.player_hand.add_card(self.deck.pop())
        self.dealer_hand.add_card(self.deck.pop())
        self.dealer_hand.add_card(self.deck.pop())

        return self._get_obs()

    def _get_obs(self):
        """
        Return the observation (player_sum, dealer_upcard, usable_ace).
        dealer_upcard = dealer_hand.cards[0] in our simplified scenario
        """
        player_sum = self.player_hand.get_best_value()
        dealer_upcard_value = self.dealer_hand.cards[0].value
        # Convert dealer_upcard_value to 10 if it's >10 in real deck logic,
        # but we already do so in deck generation.
        
        # Check if we have a 'usable' ace
        # i.e., does any combination treat an Ace as 11 without busting?
        values = self.player_hand.get_values()
        usable_ace = (len(values) > 1 and max(values) <= 21)

        return (player_sum, dealer_upcard_value, usable_ace)

    def step(self, action: int):
        """
        Take a step in the environment by applying the given action.
        
        action: 0->HIT, 1->STICK
        Returns: (next_state, reward, done)
        """
        if self.done:
            return self._get_obs(), self.reward, self.done

        if action == 0:  # HIT
            self.player_hand.add_card(self.deck.pop())
            if self.player_hand.is_bust():
                self.done = True
                self.reward = -1  # player loses
            return self._get_obs(), self.reward, self.done

        else:  # STICK
            # Dealer's turn
            while self.dealer_hand.get_best_value() < 17:
                self.dealer_hand.add_card(self.deck.pop())

            self.done = True
            # Compare totals
            if self.dealer_hand.is_bust():
                self.reward = +1
            else:
                player_total = self.player_hand.get_best_value()
                dealer_total = self.dealer_hand.get_best_value()
                if player_total > dealer_total:
                    self.reward = +1
                elif player_total < dealer_total:
                    self.reward = -1
                else:
                    self.reward = 0  # push

            return self._get_obs(), self.reward, self.done