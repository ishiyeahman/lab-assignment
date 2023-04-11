import numpy as np

def calc():
    
    a = np.array([10, 20, 30, 40])
    b = np.array([1, 2, 3, 4])
    
    # addition
    print(a + b)
    
    # subtraction
    print(a - b)
    
    # multiplication
    print(a * b)

    # division
    print(a / b)

    # dot product
    print(a.dot(b))

if __name__ == '__main__':
    calc()