import logging

from src import Gui


def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(funcName)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    Gui().start()


if __name__ == "__main__":
    main()
