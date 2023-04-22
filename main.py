from data.parser import parse_linfile

if __name__ == '__main__':
    example = parse_linfile('data/records/example.lin')
    print(example.dealer)
    print(example.players)
    print(example.hands)
    print(example.vuln)
    print(example.bids)
    print(example.made)
    print(example.play)
    print(example.claimed)
    print(example.contract)
    print(example.declarer)
    print(example.doubled)