from game.Connect4 import Connect4

def main():
    game = Connect4()
    
    game.train(10, False)
    
    #game.run()

if __name__ == "__main__":
    main()