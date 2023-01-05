// Copyright 2022 Yunin Daniil
#include "../../../modules/task_2/yunin_d_gauss_ribbon_horizontal_scheme/gauss_ribbon_horizontal_scheme.h"
#include <mpi.h>
#include <vector>
#include <string>
#include <iostream>
#include <random>
#include <algorithm>

using std::vector;
using std::mt19937;
using std::random_device;

// Метод Гаусса используется для решения систем алгебраических уравнений
// Матрицы являются плотными - большинство элементов отличны от нуля
// Подразумевается, что решаем квадратные матрицы

// Help functions start
void UpdateRandNumbers(mt19937 *gen) {
    random_device rd;
    (*gen).seed(rd());
}

vector<double> CreateMatrix(int size) {
    // чтобы сразу было всё хорошо (невырожденная матрица типа)
    /*
    1 2 3 4
    2 1 2 3
    3 2 1 2
    4 3 2 1
    */
    vector<double> matrix(size * size);
    int number = 1;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i*size+j] = number;
            (j < i) ? number-- : number++;
        }
        number = i+2;
    }
    return matrix;
}

vector<double> UnionMatrVect(const vector<double> &matr, const vector<double> &right_vec, int size_matr) {
    vector<double> result(size_matr * (size_matr + 1));
    int j = 0, k = 0, iter = 0;
    for (int i = 0; i < size_matr * (size_matr + 1); i++) {
        iter++;
        if (iter == size_matr + 1) {
            result[i] = right_vec[j];
            j++;
            iter = 0;
        } else {
            result[i] = matr[k];
            k++;
        }
    }
    return result;
}

vector<double> CreateVector(int size) {
    vector<double> vec(size);
    for (int i = 0; i < size; i++) {
        vec[i] = i*i*i;
    }
    return vec;
}

void CreateMatrixRandom(vector<double> &matrix, int size, mt19937 *gen) {
    // дело случая
    for (int i = 0; i < size * size; i++) {
        matrix[i] = (*gen)() % size; 
    }   
}

void CreateVectorRandom(vector<double> &vec, int size, mt19937 *gen) {
    // дело случая
    for (int i = 0; i < size; i++) {    
        vec[i] = (*gen)() % size + 1;
    }
}

// add this function
void InitHelpingVector(vector<int> &vec, int size) {
    for (int i = 0; i < size; i++) {
        vec[i] = -1; // указывает, что данная строка не была ещё выбрана в качестве ведущей
    }   
}

// add this function in header file
void SubtractCurrentRowMatrix(vector<double> &matr, int size, int num_row, int num_column, double main_elem, double coef) {
    for (int j = num_column; j < size; j++) {
        // std::cout << std::endl << "Вычитаю элемент - (" << num_row << ',' << j << ')' << std::endl; 
        matr[num_row*size+j] = matr[num_row*size+j] - coef*matr[num_column*size+j]; //можно использовать num_column, так как в данном случае = номеру главной строки
    }
}

// add this function in header file
void SubtractCurrentRowVector(vector<double> &vec, int size, int num_iter, double main_elem, double coef) {
    vec[num_iter] = vec[num_iter] - coef*main_elem;
}

// add this function in header file
double CalculateCoef(double main_elem, double reduced_elem) {
    double coef = reduced_elem/main_elem;
    return coef;
}


vector<double> GaussConsequent(int matrix_size) {
    vector<double> matrix(matrix_size * matrix_size); // матрица коэффициентов (одномерный массив, где элементы хранятся построчно) индекс массива - i*size-matrix+j
    vector<double> right_vector(matrix_size); // правая части (после знака =) СЛАУ
    vector<double> results(matrix_size); // вектор с ответами (значения вычесленных x)
    // helping vectors
    vector<int> sequence_numbers_rows(matrix_size); // номер строки, которая была выбрана на i-ом шаге
    vector<int> sequence_numbers_iterations(matrix_size);// номер итерации на котором i-ая строка была выбрана
    // helping vectors
    const int size_matrix = matrix_size; // на всякий случай сохраню
    mt19937 gen;
    UpdateRandNumbers(&gen);
    // инициализирую матрицу и вектор значениями
    matrix = CreateMatrix(size_matrix);
    right_vector = CreateVector(size_matrix);
    InitHelpingVector(sequence_numbers_iterations, size_matrix);
    for (int i = 0; i < size_matrix; i++) {
        double current_matrix_elem = matrix[i*size_matrix+i];
        double current_vector_elem = right_vector[i];
        double coef = 0;
        for (int j = i+1; j < size_matrix; j++) {
            coef = CalculateCoef(current_matrix_elem, matrix[j*size_matrix+i]);
            SubtractCurrentRowMatrix(matrix, size_matrix, j, i, current_matrix_elem, coef);
            SubtractCurrentRowVector(right_vector, size_matrix, j, current_vector_elem, coef);        
        }
    }
    // обратный ход
    for (int i = size_matrix-1; i >= 0; i--) {
        // пересчитываем матрицу
        for (int j = size_matrix-1; j > i; j--) {
            // std::cout << "элементы строки матрицы " << matrix[i*size_matrix+j] << std::endl;
            right_vector[i] = right_vector[i] - results[j]*matrix[i*size_matrix+j];
            matrix[i*size_matrix+j] = 0;
        }
        results[i] = right_vector[i] / matrix[i*size_matrix+i];
        // std::cout << "элемент матрицы " << matrix[i*size_matrix+i] << std::endl;
    }
    return results; // True code
} 


// Сonsequent Gauss End

void PrintMatrixVector(vector<double> &matr, int size_matr) {
    std::cout << "Матрица и вектор\n";
    int i = 0;
    for (int k = 0; k < size_matr * (size_matr + 1); k++) {
        i++;
        if (i == size_matr + 1) {
            std::cout << "| " << matr[k] << std::endl;
            i = 0;
        } else {
            std::cout << matr[k] << ' ';
        }
    }
}

void PrintMatrix(vector<double> matr, int size_matr) {
    std::cout << "Матрица\n";
    for (int k = 0; k < size_matr; k++) {
        for (int l = 0; l < size_matr; l++) {
            std::cout << matr[k*size_matr+l] << ' ';
        }
        std::cout << std::endl;
    }
}

void PrintVector(vector<double> &vec, int vec_size) {
    std::cout << "Правая часть матрицы\n";
    for (int i = 0; i < vec_size; i++) {
        std::cout << vec[i] << ' ';
    }
    std::cout << std::endl;
}

// Parallel Gauss Start

// not correct variant of algorithm
vector<double> GaussParallel(vector<double> &matr, vector<double> &right_part, int size_matr) {
    int proc_rank, proc_size;
    vector<double> global_result;
    
    MPI_Comm_size(MPI_COMM_WORLD, &proc_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    vector<int> proc_row_ind(proc_size), proc_row_num(proc_size);
    int num_rows = size_matr / proc_size;
    int remaining_rows = size_matr % proc_size;

    // чтобы распределить строки между процессами равномерно,
    // положим в те процессы, чей ранк меньше остатка на одну строку больше
    int proc_num_rows;
    if (proc_rank < remaining_rows) {
        proc_num_rows = num_rows + 1;
    }
    else {
        proc_num_rows = num_rows;
    }
    // отлаживаем число строк в процессе
    // std::cout << "Число строк " << proc_rank << proc_num_rows << std::endl;
    
    if (proc_rank == 0) {
    //     matr.resize(size_matr*size_matr);
    //     right_part.resize(size_matr);
    //     global_result.resize(size_matr);
    //     CreateMatrix(matr, size_matr);
    //     CreateVector(right_part, size_matr);
        PrintMatrix(matr, size_matr); // для отладки
        PrintVector(right_part, size_matr); // для отладки
    }
    // MPI_Barrier(MPI_COMM_WORLD);   
    

    vector<double> local_matr(proc_num_rows * size_matr);
    vector<double> local_right_vector(proc_num_rows);
    for (int i = 0; i < proc_size; i++) {
        if (proc_rank == i) {
            std::cout << "номер процесса = " << proc_rank << std::endl;
            std::cout << "размер данной матрицы = " << local_matr.size() << std::endl;  
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    vector<int> num_send_elems(proc_size), ind_send_elems(proc_size);

    ind_send_elems[0] = 0;
    for (int i = 0; i < proc_size; i++) {
        if (i < (size_matr % proc_size)) {
            num_send_elems[i] = proc_num_rows * size_matr; 
        }
        else {
            num_send_elems[i] = num_rows * size_matr;
        }
        if (i != 0) {
            std::cout << "номер процесса = " << proc_rank << 
                " индекс предыдущего " << ind_send_elems[i-1] << " количество элементов " << num_send_elems[i-1] << std::endl;
            ind_send_elems[i] = (ind_send_elems[i-1] + num_send_elems[i-1]);
        }
    }
    MPI_Scatterv(matr.data(), num_send_elems.data(), ind_send_elems.data(),
        MPI_DOUBLE, &local_matr[0], proc_num_rows * size_matr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    proc_row_ind[0] = 0; // номер первой строки в нулевом процессе
    proc_row_num[0] = proc_num_rows; // сколько строк на нулевом процессе
    for (int i = 0; i < proc_size; i++) {
        if (i < (size_matr % proc_size)) {
            proc_row_num[i] = proc_num_rows; 
        }
        else {
            proc_row_num[i] = num_rows;
        }
        if (i != 0) {
            std::cout << "номер процесса = " << proc_rank << 
                " индекс предыдущего " << proc_row_ind[i-1] << " количество элементов " << proc_row_num[i-1] << std::endl;
            proc_row_ind[i] = (proc_row_ind[i-1] + proc_row_num[i-1]);
        }
    }
    
    MPI_Scatterv(right_part.data(), proc_row_num.data(), proc_row_ind.data(),
        MPI_DOUBLE, &local_right_vector[0], proc_num_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i=0; i<proc_size; i++) {
        if (proc_rank == i) {
            printf("\nProcRank = %d \n", proc_rank);
            printf(" Matrix Stripe:\n");
            for (int k = 0; k < proc_num_rows; k++) {
                for (int l = 0; l < size_matr; l++) {
                    std::cout << local_matr[k*size_matr+l] << ' ';
                }
                std::cout << std::endl;
            }
            printf(" Vector: \n");
            for (int j = 0; j < proc_num_rows; j++) {
                std::cout << local_right_vector[j] << ' ';
            }
            std::cout << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // прямой ход
    // пока не понятно
   vector<int> num_row_main_global(size_matr),  num_row_iter_loc(proc_num_rows);
   for (int i = 0; i < proc_num_rows; i++) {
        num_row_iter_loc[i] = -2;
   }
   vector<double> global_main_row(size_matr + 1); // массив - буфер для передач строки другим процессам
   double min_value; // обратить внимание на эту строчку, может получиться ошибка
   int main_pos; // позиция выбранной строки
   struct { int min_value; int proc_rank; } main_proc, current_main;
   for (int i = 0; i < size_matr; i++) {
        for (int j = 0; j < proc_num_rows; j++) {
            if ((num_row_iter_loc[j] == -2) && (min_value > fabs(local_matr[j*size_matr+i]))) {
                min_value = fabs(local_matr[j*size_matr+i]);
                main_pos = j;
            }
        }
        main_proc.min_value = min_value;
        main_proc.proc_rank = proc_rank;
        MPI_Allreduce(&main_proc, &current_main, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
        if ( proc_rank == current_main.proc_rank ) {
            num_row_iter_loc[main_pos] = i;
            num_row_main_global[i] = proc_row_ind[proc_rank] + main_pos;
        }
        MPI_Bcast(&num_row_main_global[i], 1, MPI_INT, current_main.proc_rank, MPI_COMM_WORLD);
        if ( proc_rank == current_main.proc_rank ){
            for (int j=0; j < size_matr; j++) {
                global_main_row[j] = local_matr[main_pos*size_matr + j];
            }
            global_main_row[size_matr] = local_right_vector[main_pos];
        }
        MPI_Bcast(&global_main_row[0], size_matr+1, MPI_DOUBLE, current_main.proc_rank, MPI_COMM_WORLD);
        double coef;
        for (int j=0; j<proc_num_rows; j++) {
            if (num_row_iter_loc[j] == -2) {
                coef = local_matr[j*size_matr+i] / global_main_row[i];
                for (int k=i; k<size_matr; k++) {
                    local_matr[j*size_matr + k] -= coef* global_main_row[k];
                }
                local_right_vector[j] -= coef * global_main_row[size_matr];
            }
        }
    }
    return global_result;
}

vector<double> GaussParallels(vector<double> &matr, int size_matr) {
    int proc_size, proc_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &proc_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    int num_rows = size_matr / proc_size;
    int remaining_rows = size_matr % proc_size;
    int local_num_rows = num_rows;
    if (proc_rank < remaining_rows)
        local_num_rows++;
    vector<double> local_matr;
    vector<int> num_send_counts(proc_size);
    vector<int> num_send_ind(proc_size);
    // create local vector
    if (local_num_rows == 0) {
        local_matr.resize(1);
    } else {
        local_matr.resize(local_num_rows * (size_matr + 1));
    }
    num_send_ind[0] = 0;
    for (int i = 0; i < proc_size; i++) {
        if (i < (size_matr % proc_size))
            num_send_counts[i] = (num_rows + 1) * (size_matr + 1);
        else
            num_send_counts[i] = num_rows * (size_matr + 1);
        if (i > 0)
            num_send_ind[i] = (num_send_ind[i - 1] + num_send_counts[i - 1]);
    }
    MPI_Scatterv(matr.data(), num_send_counts.data(), num_send_ind.data(),
        MPI_DOUBLE, &local_matr[0], local_num_rows * (size_matr + 1), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    vector<double> main_row((size_matr + 1));
    for (int i = 0; i < num_send_ind[proc_rank] / (size_matr + 1); i++) {
        int root_proc = 0;
        int sum = 0;
        for (int j = 0; j < proc_size; j++, root_proc++) {
            sum += num_send_counts[j] / (size_matr + 1);
            if (i < sum) {
                root_proc = j; break;
            }
        }
        MPI_Bcast(&main_row[0], (size_matr + 1), MPI_DOUBLE, root_proc, MPI_COMM_WORLD);

        for (int j = 0; j < num_send_counts[proc_rank] / (size_matr + 1); j++) {
            double coef = main_row[i] / local_matr[j * (size_matr + 1) + i];
            for (int tmp = i; tmp < (size_matr + 1); tmp++)
                local_matr[j * (size_matr + 1) + tmp] = coef * local_matr[j * (size_matr + 1) + tmp]
                - main_row[tmp];
        }
    }

    for (int i = 0; i < num_send_counts[proc_rank] / (size_matr + 1); i++) {
        for (int j = 0; j < (size_matr + 1); j++)
            main_row[j] = local_matr[i * (size_matr + 1) + j];
        MPI_Bcast(&main_row[0], (size_matr + 1), MPI_DOUBLE, proc_rank, MPI_COMM_WORLD);
        for (int j = i + 1; j < num_send_counts[proc_rank] / (size_matr + 1); j++) {
            double coef = main_row[num_send_ind[proc_rank] / (size_matr + 1) + i] /
                local_matr[j * (size_matr + 1) + i + num_send_ind[proc_rank] / (size_matr + 1)];
            for (int tmp = i + num_send_ind[proc_rank] / (size_matr + 1); tmp < (size_matr + 1); tmp++)
                local_matr[j * (size_matr + 1) + tmp] = coef * local_matr[j * (size_matr + 1) + tmp]
                - main_row[tmp];
        }
    }
    int root_proc = 0;
    for (int i = num_send_ind[proc_rank] / (size_matr + 1) + num_send_counts[proc_rank] / (size_matr + 1);
        i < size_matr; i++) {
        int sum = 0;
        for (int j = 0; j < proc_size; j++, root_proc++) {
            sum += num_send_counts[j] / (size_matr + 1);
            if (i < sum) {
                root_proc = j; 
                break;
            }
        }
        MPI_Bcast(&main_row[0], (size_matr + 1), MPI_DOUBLE, root_proc, MPI_COMM_WORLD);
    }
    vector<double> res_matr(0);
    if (proc_rank == 0) {
        res_matr.resize(size_matr * (size_matr + 1));
    }
    MPI_Gatherv(local_matr.data(), local_num_rows * (size_matr + 1), MPI_DOUBLE, res_matr.data(),
        num_send_counts.data(), num_send_ind.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    vector<double> result_vector_x(0);
    if (proc_rank == 0) {
        result_vector_x.resize(size_matr);
        for (int i = size_matr - 1; i >= 0; --i) {
            double b = res_matr[i * (size_matr + 1) + (size_matr + 1) - 1];
            for (int j = size_matr - 1; j >= i + 1; --j)
                b -= res_matr[i * (size_matr + 1) + j] * result_vector_x[j];
            result_vector_x[i] = b / res_matr[i * (size_matr + 1) + i];
        }
    }
    return result_vector_x;
}
// Parallel Gauss End