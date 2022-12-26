import logging
from alphazero.suragnair.envs.othello.othello_game import OthelloGame
from alphazero.suragnair.envs.othello.pytorch.nnet import NNetWrapper
from alphazero.suragnair.lib.coach import Coach
from alphazero.suragnair.lib.utils import DotDict

Game = OthelloGame
nn = NNetWrapper

log = logging.getLogger(__name__)

args = DotDict(
    num_iters=1000,
    num_eps=10,  # Number of complete self-play games to simulate during a new iteration.
    temp_threshold=15,
    update_threshold=0.6,  # During arena playoff, new nnet will be accepted if threshold or more of games are won.
    max_len_of_queue=200_000,  # Number of game examples to train the neural networks.
    num_mcts_sims=25,  # Number of games moves for MCTS to simulate.
    arena_compare=40,  # Number of games to play during arena play to determine if new net will be accepted.
    cpuct=1,
    checkpoint="./temp/",
    load_model=False,
    load_folder_file=("/dev/models/8x100x50", "best.pth.tar"),
    num_iters_for_train_examples_history=20,  # 20 iterations. 1 iteration = 100 episodes <= 200k
)


def main():
    log.info("Loading %s...", Game.__name__)
    game = Game(6)

    log.info("Loading %s...", nn.__name__)
    nnet = nn(game)

    if args.load_model:
        log.info(
            'Loading checkpoint "%s/%s"...',
            args.load_folder_file[0],
            args.load_folder_file[1],
        )
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning("Not loading a checkpoint!")

    log.info("Loading the Coach...")
    coach = Coach(game, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        coach.load_train_examples()

    log.info("Starting the learning process 🎉")
    coach.learn()


if __name__ == "__main__":
    main()
