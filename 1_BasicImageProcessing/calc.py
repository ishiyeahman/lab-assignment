import numpy as np

def calc():
    
    a = np.array([10, 20, 30, 40])
    b = np.array([1, 2, 3, 4])
    
    # addition
    print("addition:", a + b)
    
    # subtraction
    print("subtraction:", a - b)
    
    # multiplication
    print("multiplication:", a * b)
    
    # division
    print("division:", a / b)
    
    # dot product
    print("dot product:", a.dot(b))
    
if __name__ == '__main__':
    calc()