from tkinter import *
from tkinter import ttk
import random
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils
from self_play import play_random_game, PNNAgent, Agent, PLAYERS
from PIL import Image, ImageTk

fmap = {'C': 'clubs', 'S': 'spades', 'H': 'hearts', 'D': 'diamonds'}
vmap = {'J': 'jack', 'Q': 'queen', 'K': 'king', 'A': 'ace'}
FONT = ('Comic Sans MS', 16)

def ftrans(card):
    c, s = card[:-1], card[-1]
    c = vmap.get(c, c)
    s = fmap[s]
    cardf = f'{c}_of_{s}'
    return cardf

def resize_cards(card):
    card = os.path.join(os.getcwd(), 'gui', 'img', card)
    our_card_img = Image.open(card)
    #our_card_resize_image = our_card_img.resize((150, 218))
    our_card_resize_image = our_card_img.resize((90, 88+44))
    global our_card_image
    our_card_image = ImageTk.PhotoImage(our_card_resize_image)
    return our_card_image


class BridgeGUI(Agent):

    def __init__(self):
        self.root = Tk()
        self.root.title('RL Trained Bridge Agent (CS224R)')
        self.root.geometry('1600x1600')
        self.root.configure(background='green')


        self.my_frame = Frame(self.root, bg='green')
        self.my_frame.pack(pady=20)

        # Create a Button to get the input data
        self.bid_entered = StringVar()
        s = ttk.Style()
        s.configure('my.TButton', font=FONT)
        #self.bid_button = ttk.Button(self.root, text="Enter Bid", style='my.TButton', command=self.get_data)
        self.bid_button = ttk.Button(self.my_frame, text="Enter Bid", style='my.TButton', command=self.get_data)
        self.bid_button.grid(row=4, column=1, padx=20, ipadx=20)

        # Create an Entry Widget
        self.entry = Entry(self.my_frame, width=42)
        self.entry.grid(row=5, column=1, padx=0, ipadx=0)

        self.start_button = ttk.Button(self.my_frame, text="Start Game", style='my.TButton', command=self.start_game)
        self.start_button.grid(row=0, column=2, padx=0, ipadx=0)


        self.bidding_history_frame = LabelFrame(self.my_frame, text='bidding history', font=FONT, bd=0)
        self.bidding_history_frame.grid(row=1, column=2, rowspan=10, padx=20, ipadx=20)
        self.bidding_history_label = Label(self.bidding_history_frame, text='')
        self.bidding_history_label.pack(pady=20)

        self.frames = {}
        self.labels = {}
        #for r, c, side in zip([0, 2, 1, 1], [1, 1, 2, 0], ['N', 'S', 'E', 'W']):
        for r, c, side in zip([0, 1, 2, 3], [1, 1, 1, 1], ['N', 'E', 'S', 'W']):
            self.frames[side] = LabelFrame(self.my_frame, text=side, font=FONT, bd=0)
            self.frames[side].grid(row=r, column=c, padx=2, pady=2, ipadx=2)
            for i in range(13):
                self.labels[side, i] = Label(self.frames[side], text='')
                self.labels[side, i].grid(row=0, column=i)
                #self.labels[side, i].pack(padx=.1)

    def get_data(self):
        input = self.entry.get()
        self.bid_entered.set('bid_entered')
        return input

    def start_game(self):
        agent1 = PNNAgent(stochastic=False)
        play_random_game(agent1, self, verbose=True)
        pass

    def transmit_game(self, game, agent1_side):
        global images
        images = {}
        for side, cards in game['hands'].items():
            cardss = cards.split(',')
            for i, card in enumerate(cardss):
                cardf = ftrans(card)
                images[side, i] = resize_cards(f'{cardf}.png')
                self.labels[side, i].config(image=images[side, i])

    def bid(self, game):
        decl_idx = PLAYERS.index(game['dealer'])
        n_prev_bids = len(game['bids'])
        current_player = PLAYERS[(decl_idx + n_prev_bids) % len(PLAYERS)]
        bid_history = ''
        for i in range(n_prev_bids):
            bid_history += f"Player {PLAYERS[(decl_idx + i) % len(PLAYERS)]} bid {game['bids'][i]}\n"
        bid_history += f"Awaiting your bid as {current_player} ..."
        self.bidding_history_label.config(text=bid_history, font=FONT)
        self.root.title(f'RL Trained Bridge Agent (CS224R): ENTER YOUR BID AS {current_player}')
        self.bid_entered = StringVar()
        self.bid_button.wait_variable(self.bid_entered)
        bid = self.get_data()
        return None, bid

    def run(self):
        self.root.mainloop()

if __name__ == '__main__':

    bridge = BridgeGUI()
    bridge.run()
