import numpy as np

def vector_scramble(data_vector, mu, x0, size,m, n):

    matrix = data_vector.reshape(size,size)

    row_seq = logistic_map(x0, mu, size, m)
    row_perm = generate_permutation(row_seq)

    col_seq = logistic_map(row_seq[-1], mu, size, n)
    col_perm = generate_permutation(col_seq)
 
    scrambled = matrix[row_perm, :]
    scrambled = scrambled[:, col_perm]

    return scrambled.flatten()

def vector_descramble(scrambled_vector, mu, x0,size, m, n):

    matrix = scrambled_vector.reshape(size, size)
    
    row_seq = logistic_map(x0, mu, size, m)
    row_perm = generate_permutation(row_seq)
    col_seq = logistic_map(row_seq[-1], mu, size, n)
    col_perm = generate_permutation(col_seq)

    inv_row = np.argsort(row_perm)
    inv_col = np.argsort(col_perm)

    descrambled = matrix[:, inv_col]
    descrambled = descrambled[inv_row, :]
    
    return descrambled.flatten()

def logistic_map(initial, mu, length, discard=0):
    x = initial
    for _ in range(discard):
        x = mu * x * (1 - x)
    seq = []
    for _ in range(length):
        x = mu * x * (1 - x)
        seq.append(x)
    return np.array(seq)

def generate_permutation(seq):
    return np.argsort(seq)

def encryted(data_vector, mu, x0, m, n):
    data_vector = data_vector.flatten() 
    z1=data_vector[:16384]
    z2=data_vector[16384:32768]
    z3=data_vector[32768:]
    size=128
    m1=vector_scramble(z1, mu, x0,size, m, n)
    m2 = vector_scramble(z2, mu,x0,size, m, n)
    m3= vector_scramble(z3, mu, x0,size, m, n)
    tempz = np.concatenate((m1, m2, m3))
    return tempz
def decryted(data_vector, mu, x0, m, n):
    data_vector = data_vector.flatten() 
    z1=data_vector[:16384]
    z2=data_vector[16384:32768]
    z3=data_vector[32768:]
    size=128
    m1=vector_descramble(z1, mu, x0, size,m, n)
    m2 = vector_descramble(z2, mu, x0,size, m, n)
    m3= vector_descramble(z3, mu, x0,size, m, n)
    tempz = np.concatenate((m1, m2, m3))
    return tempz

if __name__ == "__main__":

    original_vector = np.arange(4096).astype(np.uint8)
    print("Original vector:", original_vector)

    mu = 3.9999  
    x0 = 0.3333   
    m = 100     
    n = 200
    
    encrypted = vector_scramble(original_vector, mu, x0, m, n)
    print("Encrypted vector:", encrypted)

    decrypted = vector_descramble(encrypted, mu, 0.444, m, n)
    print("Decrypted vector:", decrypted)
    
    print("Decryption correct?", np.all(original_vector == decrypted))