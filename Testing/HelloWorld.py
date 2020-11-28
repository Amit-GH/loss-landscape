import numpy as np

if __name__ == '__main__':
    print("This is inside a python file.")

    d = np.array([[1, 2, 5], [2, 3, 4]])
    print(d)
    for ele in d:
        ele += 1
    print(d)

    f = 123.3456
    print("integer {}".format(int(f)))
