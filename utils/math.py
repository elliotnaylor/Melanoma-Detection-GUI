import math

def euclidean2d(v1, v2):

    dis = math.pow(v1, 2) + math.pow(v2, 2)

    return dis

#JND = square_root((L_b1 - L_b2)^2 + (A_b1 - A_b2)^2 + (B_b1 - B_b2)^2)
def euclidean3d(v1, v2):
    x = math.pow(int(v1[0]) - int(v2[0]), 2)
    y = math.pow(int(v1[1]) - int(v2[1]), 2)
    z = math.pow(int(v1[2]) - int(v2[2]), 2)
    
    dis = math.sqrt(x + y + z)
    
    return dis

def distance(x1, x2, y1, y2):
    dis = math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1, y2), 2))
    return dis

def make_even(n):
    return n-n % 2
