// Copyright 2023 Bataev Ivan
#include <mpi.h>
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>
#include "../../../modules/task_2/bataev_i_gauss_horizon/gauss_horizon.h"

std::vector<double> getRandomVector(int size, int left, int right) {
    std::random_device rd;
    std::mt19937 mersenne(rd());
    std::uniform_int_distribution<> distr(left, right);

    std::vector<double> v(size);
    for (auto& elem : v) { elem = distr(mersenne); }
    return v;
}

void printVector(const std::vector<double>& v, const std::string& prefix) {
    std::cout << prefix << "[";
    for (int i = 0; i < v.size() - 1; i++) { std::cout << v[i] << ", "; }
    std::cout << v[v.size() - 1] << "]\n";
}

bool isAlmostEqual(double a, double b) { return fabs(a - b) <= 0.0000001; }

void printFullMatr(const std::vector<double>& M, const int n, const std::string& prefix) {
    std::cout << prefix;
    std::cout.precision(4);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; ++j) {
            std::cout.width(8);
            if (isAlmostEqual(M[j + i * (n+1)], 0))
                std::cout << "0" << " ";
            else
                std::cout << M[j + i * (n+1)] << " ";
        }
        std::cout << "| " << std::setw(8) << M[n + i * (n+1)] << "\n";
    }
    std::cout << "\n";
}

std::vector<double> gaussMethSequential(std::vector<double> M, const int n) {
    // printFullMatr(M, n);

    // The first stage
    for (int j = 0; j < n - 1; ++j) {
        // find the row where the elem in the leading col is the largest modulo and below the current leading row
        int maxRow = j;
        for (int i = j + 1; i < n; ++i)
            if (fabs(M[j + i * (n+1)]) > fabs(M[j + maxRow * (n+1)]))  // take the first matching row
                maxRow = i;
        // swap with it
        if (maxRow != j) {
            for (int k = 0; k < n + 1; ++k)
                std::swap(M[k + j * (n+1)], M[k + maxRow * (n+1)]);
            // printFullMatr(M, n);
        }

        if (isAlmostEqual(M[j + j * (n+1)], 0))
            continue;  // if leading element == 0, then go to the next row to avoid division by zero

        // divide the leading row by the leading element
        for (int i = j + 1; i < n; ++i) {
            double alfa = M[j + i*(n+1)] / M[j + j*(n+1)];
            for (int k = j; k < n + 1; k++)
                M[k + i*(n+1)] -= alfa * M[k + j*(n+1)];
        }
        // printFullMatr(M, n);
    }

    // The second stage
    std::vector<double> x(n);
    for (int i = n - 1; i >= 0; i--) {
        double sum = 0;
        for (int j = i + 1; j < n; j++)
            sum += M[j + i*(n+1)] * x[j];

        if (isAlmostEqual(M[i + i * (n+1)], 0))  // if we have risen to a zero leading element
            if (isAlmostEqual(M[n + i * (n+1)] - sum, 0))  // if there is an expression 0x == 0
                x[i] = 1;  // then the variable can take any real value, let it be 1
            else  // if there is an expression 0x == (b[i] - sum), where (b[i] - sum) != 0
                return std::vector<double>();  // then there is no solution, return a null vector
        else
            x[i] = (M[n + i * (n+1)] - sum) / M[i + i * (n+1)];
    }
    return x;
}

std::vector<double> gaussMethParallel(std::vector<double> M, const int n) {
    int commSize, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // send strings to processes using alternating (cyclic) horizontal splitting
    if (rank == 0) {
        for (int _rank = 1; _rank < commSize; _rank++) {
            std::vector<double> _lM;
            for (int i = _rank; i < n; i += commSize) {
                for (int j = 0; j < n + 1; ++j)
                    _lM.push_back(M[j + i * (n+1)]);
            }
            MPI_Send(_lM.data(), _lM.size(), MPI_DOUBLE, _rank, 0, MPI_COMM_WORLD);
        }
    }
    int lMSize = (n + 1) * ((n % commSize > rank) ? n / commSize + 1 : n / commSize);
    std::vector<double> lM(lMSize);
    if (rank == 0) {
        for (int i = 0; i*commSize < n; i++)
            for (int j = 0; j < n + 1; ++j)
                lM[j + i * (n+1)] = M[j + i*commSize*(n+1)];
    } else {
        MPI_Status status;
        MPI_Recv(lM.data(), lM.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
    }

    // if (rank == 0) { printFullMatr(M, n); }
    // printVector(lM, std::to_string(rank) + ":lM=");

    // The first stage
    for (int j = 0; j < n - 1; ++j) {
        // find the local row where the elem in the leading col is the largest modulo and below the current leading row
        int lMaxRow = j;
        double lMax = (j % commSize == rank) ? fabs(lM[j + j / commSize * (n+1)]) : 0;
        for (int i = j + 1; i < n; ++i) {
            if (i % commSize == rank) {
                if (fabs(lM[j + i / commSize * (n+1)]) > lMax) {
                    lMaxRow = i;
                    lMax = fabs(lM[j + i / commSize * (n+1)]);
                }
            }
        }
        // find the global row
        int gMaxRow = 0;
        double gMax = 0;
        MPI_Allreduce(&lMax, &gMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (lMax != gMax) { lMaxRow = n; }
        MPI_Allreduce(&lMaxRow, &gMaxRow, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);  // take the first matching row

        // swap with it
        if (j % commSize == rank || gMaxRow % commSize == rank) {  // only for these two or this one rank
            if (gMaxRow != j) {  // if current row isn't leading
                if (gMaxRow % commSize == j % commSize) {  // if current leading row and true leading row are at the same rank
                    for (int k = 0; k < n + 1; ++k)
                        std::swap(lM[k + j / commSize * (n+1)], lM[k + gMaxRow / commSize * (n+1)]);
                } else {  // if current leading row and true leading row are at different ranks
                    std::vector<double> tmpRow(n+1);
                    int li = 0, dest = 0;
                    if (j % commSize == rank) {
                        li = j / commSize;
                        dest = gMaxRow % commSize;
                    } else {
                        li = gMaxRow / commSize;
                        dest = j % commSize;
                    }
                    MPI_Status status;
                    MPI_Sendrecv(lM.data() + li * (n+1), n+1, MPI_DOUBLE, dest, rank * j,
                        tmpRow.data(), n+1, MPI_DOUBLE, dest, dest * j, MPI_COMM_WORLD, &status);

                    for (int k = 0; k < n+1; ++k)
                        lM[k + li * (n+1)] = tmpRow[k];
                }
            }
        }
        // printVector(lM, "j=" + std::to_string(j) + "|" + std::to_string(rank) + ":lM=");

        // send leading row to other ranks
        std::vector<double> leadRow(n+1);
        if (j % commSize == rank)
            for (int k = 0; k < n+1; ++k)
                leadRow[k] = lM[k + j / commSize * (n+1)];
        MPI_Bcast(leadRow.data(), n+1, MPI_DOUBLE, j % commSize, MPI_COMM_WORLD);

        if (isAlmostEqual(leadRow[j], 0))
            continue;  // if leading element == 0, then go to the next row to avoid division by zero

        // divide the leading row by the leading element
        for (int i = j + 1; i < n; ++i) {
            if (i % commSize == rank) {
                double alfa = lM[j + i / commSize * (n+1)] / leadRow[j];
                for (int k = j; k < n + 1; k++)
                    lM[k + i / commSize * (n+1)] -= alfa * leadRow[k];
            }
        }
        // printVector(lM, "j=" + std::to_string(j) + "|" + std::to_string(rank) + ":lM=");
    }

    // The second stage
    std::vector<double> gx(n);
    for (int i = n - 1; i >= 0; i--) {
        int noSol = 0;
        double leadx = 0;

        if (i % commSize == rank) {
            if (isAlmostEqual(lM[i + i / commSize * (n+1)], 0))  // if we have risen to a zero leading element
                if (isAlmostEqual(lM[n + i / commSize * (n+1)], 0))  // if there is an expression 0x == 0
                    leadx = 1;  // then the variable can take any real value, let it be 1
                else  // if there is an expression 0x == (b[i] - sum), where (b[i] - sum) != 0
                    noSol = 1;  // then there is no solution, return a null vector
            else
                leadx = lM[n + i / commSize * (n+1)] / lM[i + i / commSize * (n+1)];
        }

        MPI_Bcast(&noSol, 1, MPI_INT, i% commSize, MPI_COMM_WORLD);
        if (noSol)
            return std::vector<double>();
        MPI_Bcast(&leadx, 1, MPI_DOUBLE, i % commSize, MPI_COMM_WORLD);

        if (rank == 0)
            gx[i] = leadx;

        // decrease all b[k] that are above the current row
        for (int k = 0; k < lMSize / (n + 1); k++)
            if (k * commSize + rank < i)
                lM[n + k * (n + 1)] -= lM[i + k * (n + 1)] * leadx;
            else
                break;
    }
    return gx;
}