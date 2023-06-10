import pandas as pd
import os

import torch
from sympy import *


# Define signal matrix u which is ([-1,L], [-1,L])


def generate_initial_matrix(coeff_rows_amount, coeff_vec_size):
    return {
        'A_1': torch.zeros(coeff_rows_amount, coeff_vec_size),
        'A_2': torch.zeros(coeff_rows_amount, coeff_vec_size),
        'A_3': torch.zeros(coeff_rows_amount, coeff_vec_size),
        'A_4': torch.zeros(coeff_rows_amount, coeff_vec_size),
        'B': torch.zeros(coeff_rows_amount, 2),
    }


class CoeffCalculator:

    def __init__(self, L):
        """
        The form is either X or H currently, where X means:
        A_3 = A_2 , A_4 = A_1
        and H means:
        A_3 = A_1 , A_4 = A_2
        """
        self.L = L
        self.u = []

        self.u_0 = Symbol('u_0_0')
        self.u_0 = 1

        self.A_1 = Symbol('A_1')
        self.A_2 = Symbol('A_2')
        self.A_3 = Symbol('A_3')
        self.A_4 = Symbol('A_4')
        self.B_1 = Symbol('B_1')
        self.B_2 = Symbol('B_2')

        self.C_1 = Symbol('C_1')
        self.C_2 = Symbol('C_2')

        # Define that where i=-1 or j=-1 x[i,j] = 0
        self.x_h_symbols = [[0 for i in range(self.L)] for j in range(self.L)]
        self.x_v_symbols = [[0 for i in range(self.L)] for j in range(self.L)]

        self.coeff_rows_amount = 2 * (L ** 3)
        self.coeff_vec_size = 2 * L
        self.matrices = \
            {'horizontal': generate_initial_matrix(self.coeff_rows_amount, self.coeff_vec_size),
             'vertical': generate_initial_matrix(self.coeff_rows_amount, self.coeff_vec_size)
             }

        # Print files in current directory
        import os
        self.cache_location = f'./coeffs_cache/matrices_{L}.pt'
        # self.cache_location = f'../../coeffs_cache/matrices_{L}.pt'

    def calc_coeffs_lazy(self, force=False):
        checked_cache_location = self.cache_location.format(self.L)
        if os.path.exists(checked_cache_location) and not force:
            self.matrices = torch.load(checked_cache_location)
        else:
            self.initialize_matrices()
            self.set_final_coeffs_matrix()
            torch.save(self.matrices, checked_cache_location)

    def initialize_matrices(self):
        for i in range(self.L):
            if i % 5 == 0:
                print(f'Initializing row {i} / {self.L}')
            for j in range(self.L):
                self.x_h_symbols[i][j] = self.calc_x_h(i, j)
                self.x_v_symbols[i][j] = self.calc_x_v(i, j)

    # Define expression with A_1 .. A_4, B_1,B_2 as variables
    def calc_x_h(self, i, j):
        last_j = j - 1
        relevant_x_h = self.x_h_symbols[i][last_j] if i >= 0 and last_j >= 0 else 0
        relevant_x_v = self.x_v_symbols[i][last_j] if i >= 0 and last_j >= 0 else 0
        current_u = self.u_0 if i == 0 and j == 0 else 0
        if relevant_x_v == 0 or relevant_x_h == 0:
            expr = self.A_1 * relevant_x_h + self.A_2 * relevant_x_v + self.B_1 * current_u
        else:
            expr = (self.A_1 * relevant_x_h + self.A_2 * relevant_x_v) / 2 + self.B_1 * current_u
        return expand(expr)

    def calc_x_v(self, i, j):
        last_i = i - 1
        relevant_x_h = self.x_h_symbols[last_i][j] if i >= 0 and last_i >= 0 else 0
        relevant_x_v = self.x_v_symbols[last_i][j] if i >= 0 and last_i >= 0 else 0
        current_u = self.u_0 if i == 0 and j == 0 else 0
        if relevant_x_v == 0 or relevant_x_h == 0:
            expr = self.A_3 * relevant_x_h + self.A_4 * relevant_x_v + self.B_2 * current_u
        else:
            expr = (self.A_3 * relevant_x_h + self.A_4 * relevant_x_v) / 2 + self.B_2 * current_u
        return expand(expr)

    def convert_to_vector_for_symbol(self, coeff_dict, symbol):
        vec = torch.zeros(self.coeff_vec_size, self.coeff_vec_size)
        i = 0
        for key, val in coeff_dict.items():
            found = False
            for arg in key.args:
                if len(arg.args) == 0:
                    s = str(arg)
                    pow = 1
                else:
                    s = str(arg.args[0])
                    pow = int(arg.args[1])
                if s == symbol:
                    vec[i, pow] = 1
                    found = True
            if not found:
                vec[i, 0] = 1
            i += 1
        return vec

    def convert_to_vector_for_B_12(self, coeffs_dict):
        vec = torch.zeros(self.coeff_vec_size, 2)
        i = 0
        for key, val in coeffs_dict.items():
            if str(key) == 'B_1':
                vec[i, 0] = float(val)
            elif str(key) == 'B_2':
                vec[i, 1] = float(val)
            else:
                for arg in key.args:
                    if str(arg) == 'B_1':
                        vec[i, 0] = float(val)
                    elif str(arg) == 'B_2':
                        vec[i, 1] = float(val)
            i += 1
        return vec

    def set_final_coeffs_matrix(self):
        amount_for_cell = 2 * self.L
        for direction in ['horizontal', 'vertical']:
            x = self.x_h_symbols if direction == 'horizontal' else self.x_v_symbols
            for i in range(self.coeff_rows_amount // amount_for_cell):
                if i % 5 == 0:
                    print(f'Direction: {direction}, Calculating row {i} / {self.coeff_rows_amount // amount_for_cell}')
                ver = i // self.L
                hor = i % self.L
                for symbol, matrix in self.matrices[direction].items():
                    if symbol == 'B':
                        matrix[i * amount_for_cell: (i + 1) * amount_for_cell] = self.convert_to_vector_for_B_12(
                            x[ver][hor].as_coefficients_dict())
                    else:
                        outcome = self.convert_to_vector_for_symbol(
                            x[ver][hor].as_coefficients_dict(), symbol)
                        matrix[i * amount_for_cell: (i + 1) * amount_for_cell] = outcome

    def calc_y(self, i, j):
        if i == 0 and j == 0:
            return self.C_1 * self.x_h_symbols[i][j] + self.C_2 * self.x_v_symbols[i][j]
        elif i == 0 or j == 0:
            return 2 * (self.C_1 * self.x_h_symbols[i][j] + self.C_2 * self.x_v_symbols[i][j])
        return self.C_1 * self.x_h_symbols[i][j] + self.C_2 * self.x_v_symbols[i][j]

    def compute_symbolic_kernel(self):
        lst = []
        for i in range(self.L):
            lst.append([])
            for j in range(self.L):
                lst[i].append(self.calc_y(i, j))
        return lst

    def compute_sympy_kernel(self, A_vals, B_1, B_2, C_1, C_2):
        df = pd.DataFrame(self.compute_symbolic_kernel())
        values = {self.A_1: A_vals['A_1'],
                  self.A_2: A_vals['A_2'],
                  self.A_3: A_vals['A_3'],
                  self.A_4: A_vals['A_4'], }
        values.update({self.C_1: C_1, self.C_2: C_2,
                       self.B_1: B_1, self.B_2: B_2})
        kernel_outcome = df.applymap(lambda x: x.subs(values))
        return kernel_outcome


if __name__ == '__main__':
    L = 4
    calc = CoeffCalculator(L)
    calc.initialize_matrices()
    calc.set_final_coeffs_matrix()
    fake_A = {
        'A_1': 1,
        'A_2': 1,
        'A_3': 1,
        'A_4': 1,
    }
    fake_B_1 = 1
    fake_B_2 = 1
    fake_C_1 = 1
    fake_C_2 = 1
    kernel = calc.compute_sympy_kernel(fake_A, fake_B_1, fake_B_2, fake_C_1, fake_C_2)
    print(kernel)
