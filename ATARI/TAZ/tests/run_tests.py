import os

RESET_COLOR = '\u001b[0m'
YELLOW = '\u001b[33m'

TESTDIR = os.path.dirname(__file__)

if __name__ == '__main__':
    print()
    print(f'{YELLOW}Running all unit tests:{RESET_COLOR}')
    os.system(f'python -m unittest discover {TESTDIR}') # runs all of the unit tests
    print()
    print(f'{YELLOW}All unittests complete!{RESET_COLOR}')
    print(f'Histogram tests will show plots in "error_plots" directory when an error occurs.')
    print()