import numpy as np
def main():
    print(np.ones(5).reshape(-1, 1) * np.array([1, 2, 3]).reshape(1, -1))


if __name__ == '__main__':
    main()