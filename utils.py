import math

# Max Iterations
int_max_iter = 15
int_const = 0.1

# pnew_trans = trans_en_fil_matrix
# pold_trans = trans_en_fil_matrix_prev
def is_converged(pnew_trans, pold_trans, pint_iter):
    # converging factor
    float_conv_factor = 0.00000001

    if pint_iter > int_max_iter:
        return True

    # loop for checking if the length of the dictionaries is converging to the converging factor
    for int_index in range(len(pnew_trans)):
        for int_index2 in range(len(pnew_trans[0])):
            if math.fabs(pnew_trans[int_index][int_index2] - pold_trans[int_index][int_index2]) > float_conv_factor:
                return False
    return True
