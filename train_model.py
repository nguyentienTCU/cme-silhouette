"""Backward-compatible training command; imports have no execution side effects."""
from scripts.train import main

if __name__ == '__main__':
    main()
