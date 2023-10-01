import numpy as np


def main():
    x = np.stack([np.ones(5), np.arange(5)])
    e_x = np.exp(x)
    sm = np.sum(e_x, axis=1, keepdims=True)
    print(e_x/sm)


if __name__ == '__main__':
    main()
