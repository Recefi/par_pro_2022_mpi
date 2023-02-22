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

void printFullMatr(const std::vector<double>& A, const std::vector<double>& b, const std::string& prefix) {
    int n = b.size();
    std::cout << prefix;
    std::cout.precision(4);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; ++j) {
            std::cout.width(8);
            if (isAlmostEqual(A[j + i * n], 0))
                std::cout << "0" << " ";
            else
                std::cout << A[j + i * n] << " ";
        }
        std::cout << "| " << std::setw(8) << b[i] << "\n";
    }
    std::cout << "\n";
}

std::vector<double> gaussMethSequential(std::vector<double> A, std::vector<double> b, const int n) {
    // printFullMatr(A, b);

    // The first stage
    for (int j = 0; j < n - 1; ++j) {
        // find the row where the element in the leading column is the largest modulo and which is below the current leading row
        int maxRow = j;
        for (int i = j + 1; i < n; ++i)
            if (fabs(A[j + i * n]) > fabs(A[j + maxRow * n]))  // take the first matching row
                maxRow = i;
        // swap with it
        if (maxRow != j) {
            for (int k = 0; k < n; ++k)
                std::swap(A[k + j * n], A[k + maxRow * n]);
            std::swap(b[j], b[maxRow]);
            // printFullMatr(A, b);
        }

        if (isAlmostEqual(A[j + j * n], 0))
            continue;  // if leading element = 0, then go to the next row to avoid division by zero

        for (int i = j + 1; i < n; ++i) {
            double alfa = A[j + i*n] / A[j + j*n];
            for (int k = j; k < n; k++)
                A[k + i*n] -= alfa * A[k + j*n];
            b[i] -= alfa * b[j];
        }
        // printFullMatr(A, b);
    }

    // The second stage
    std::vector<double> x(n);
    for (int i = n - 1; i >= 0; i--) {
        double sum = 0;
        for (int j = i + 1; j < n; j++)
            sum += A[j + i*n] * x[j];

        if (isAlmostEqual(A[i + i * n], 0)) {  // if we have risen to a zero leading element
            if (isAlmostEqual(b[i] - sum, 0)) { // if there is an expression 0x == 0
                x[i] = 1;  // then the variable can take any real value, let it be 1
            } else {  // if there is an expression 0x == (b[i] - sum), where (b[i] - sum) != 0
                // std::cout << "There is no solution\n\n";
                return std::vector<double>();  // then there is no solution, return a null vector
            }
        }
        else
            x[i] = (b[i] - sum) / A[i + i * n];
    }
    return x;
}

std::vector<double> gaussMethParallel(std::vector<double> A, std::vector<double> b, const int n) {
    int commSize, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int lbSize = (n % commSize - 1 >= rank) ? n / commSize + 1 : n / commSize;
    int lASize = n * lbSize;

    /* working
    if (rank == 0) {
        for (int i = 1; i < n; i++) {
            if (i % commSize == 0)
                continue;
            MPI_Send(A.data() + i*n, n, MPI_DOUBLE, i % commSize, i*n, MPI_COMM_WORLD);
            MPI_Send(b.data() + i, 1, MPI_DOUBLE, i % commSize, i, MPI_COMM_WORLD);
        }
    }
    std::vector<double> lA(lASize), lb(lbSize);
    for (int i = rank; i < n; i += commSize) {
        int k = (i - rank) / commSize;
        if (rank == 0) {
            for (int j = 0; j < n; ++j)
                lA[j + k*n] = A[j + i*n];
            lb[k] = b[i];
        } else {
            MPI_Status status;
            MPI_Recv(lA.data() + k*n, n, MPI_DOUBLE, 0, i*n, MPI_COMM_WORLD, &status);
            MPI_Recv(lb.data() + k, 1, MPI_DOUBLE, 0, i, MPI_COMM_WORLD, &status);
        }
    }
    */

    /* partially working
    std::vector<double> lA(lASize), lb(lbSize);
    std::vector<int> lASizes(n, n), lbSizes(n, 1), lAShifts(n), lbShifts(n);
    for (int k = 0; k < n / commSize; ++k) {
        MPI_Scatter(A.data() + k * n, n, MPI_DOUBLE, lA.data() + k*n, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(b.data() + k, 1, MPI_DOUBLE, lb.data() + k, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    */

    /* partially working
    //std::vector<int> lASizes(n), lbSizes(n), lAShifts(n), lbShifts(n);
    //for (int _rank = 0; _rank < commSize; _rank++) {
    //    for (int i = _rank; i < n; i += commSize) {
    //        lASizes[i] = n;
    //        lbSizes[i] = 1;
    //        lAShifts[i] = i * n;
    //        lbShifts[i] = i;
    //    }
    //}

    std::vector<int> lASizes(commSize), lbSizes(commSize), lAShifts(commSize), lbShifts(commSize);
    int shiftA = 0, shiftb = 0;
    for (int _rank = 0; _rank < commSize; _rank++) {
        lbSizes[_rank] = (n % commSize - 1 >= _rank) ? n / commSize + 1 : n / commSize;
        lASizes[_rank] = n * lbSizes[_rank];
        lAShifts[_rank] = shiftA;
        lbShifts[_rank] = shiftb;
        shiftA += lASizes[_rank];
        shiftb += lbSizes[_rank];
    }
    std::vector<double> lA(lASize), lb(lbSize);
    MPI_Scatterv(A.data(), lASizes.data(), lAShifts.data(), MPI_DOUBLE, lA.data(), lASize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(b.data(), lbSizes.data(), lbShifts.data(), MPI_DOUBLE, lb.data(), lbSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    */

    if (rank == 0) {
        for (int _rank = 1; _rank < commSize; _rank++) {
            std::vector<double> _lA, _lb;
            for (int i = _rank; i < n; i += commSize) {
                for (int j = 0; j < n; ++j)
                    _lA.push_back(A[j + i * n]);
                _lb.push_back(b[i]);
            }
            MPI_Send(_lA.data(), _lA.size(), MPI_DOUBLE, _rank, 0, MPI_COMM_WORLD);
            MPI_Send(_lb.data(), _lb.size(), MPI_DOUBLE, _rank, 1, MPI_COMM_WORLD);
        }
    }
    std::vector<double> lA(lASize), lb(lbSize);
    if (rank == 0) {
        for (int i = 0; i*commSize < n; i++) {
            for (int j = 0; j < n; ++j)
                lA[j + i * n] = A[j + i*commSize*n];
            lb[i] = b[i*commSize];
        }
    } else {
        MPI_Status status;
        MPI_Recv(lA.data(), lA.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(lb.data(), lb.size(), MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
    }

    // printVector(lA, std::to_string(rank) + ":A=");
    // printVector(lb, std::to_string(rank) + ":b=");
    // if (rank == 0) { printFullMatr(A, b); }

    // The first stage
    for (int j = 0; j < n - 1; ++j) {
        // find the local row where the element in the leading column is the largest modulo and which is below the current leading row
        int lMaxRow = j;
        double lMax = (j % commSize == rank) ? fabs(lA[j + (j - rank) / commSize * n]) : 0;
        for (int i = j + 1; i < n; ++i) {
            if (i % commSize == rank) {
                if (j % commSize != rank || fabs(lA[j + (i - rank) / commSize * n]) > lMax) {
                    lMaxRow = i;
                    lMax = fabs(lA[j + (i - rank) / commSize * n]);
                }
            }
        }
        // find the global row
        int gMaxRow = j;
        double gMax = 0;
        MPI_Allreduce(&lMax, &gMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        /*if (j == 1) {
            std::cout << rank << ": lMax = " << lMax << "\n";
            std::cout << "gMax = " << gMax << "\n";
        }*/
        if (lMax != gMax)
            lMaxRow = n;
        MPI_Allreduce(&lMaxRow, &gMaxRow, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);  // take the first matching row

        /*if (j == 1) {
            std::cout << rank << ": _lMax = " << lMax << "\n";
            std::cout << "_gMax = " << gMax << "\n";
            std::cout << rank << ": lMaxRow = " << lMaxRow << "\n";
            std::cout << "gMaxRow = " << gMaxRow << "\n";
        }*/
        // swap with it
        if (j % commSize == rank || gMaxRow % commSize == rank) {  // only for these two or this one rank
            if (gMaxRow != j) {  // if current row isn't leading
                if (gMaxRow % commSize == j % commSize) {  // if current leading row and true leading row are at the same rank
                    for (int k = 0; k < n; ++k)
                        std::swap(lA[k + (j - rank) / commSize * n], lA[k + (gMaxRow - rank) / commSize * n]);
                    std::swap(lb[(j - rank) / commSize], lb[(gMaxRow - rank) / commSize]);
                } else {  // if current leading row and true leading row are at different ranks
                    std::vector<double> tmpA(n);
                    double tmpb = 0;
                    int li = 0, dest = 0;
                    if (j % commSize == rank) {
                        li = (j - rank) / commSize;
                        dest = gMaxRow % commSize;
                    }
                    if (gMaxRow % commSize == rank) {
                        li = (gMaxRow - rank) / commSize;
                        dest = j % commSize;
                    }
                    MPI_Status status;
                    /*MPI_Send(lA.data() + li * n, n, MPI_DOUBLE, dest, rank * j, MPI_COMM_WORLD);
                    MPI_Send(lb.data() + li, 1, MPI_DOUBLE, dest, (rank + commSize) * j, MPI_COMM_WORLD);
                    MPI_Recv(tmpA.data(), n, MPI_DOUBLE, dest, dest * j, MPI_COMM_WORLD, &status);
                    MPI_Recv(&tmpb, 1, MPI_DOUBLE, dest, (dest + commSize) * j, MPI_COMM_WORLD, &status);*/
                    MPI_Sendrecv(lA.data() + li * n, n, MPI_DOUBLE, dest, rank * j,
                        tmpA.data(), n, MPI_DOUBLE, dest, dest * j, MPI_COMM_WORLD, &status);
                    MPI_Sendrecv(lb.data() + li, 1, MPI_DOUBLE, dest, (rank + commSize) * j,
                        &tmpb, 1, MPI_DOUBLE, dest, (dest + commSize) * j, MPI_COMM_WORLD, &status);

                    for (int k = 0; k < n; ++k)
                        lA[k + li * n] = tmpA[k];
                    lb[li] = tmpb;
                }
            }
        }
          // printVector(lA, "j=" + std::to_string(j) + "|" + std::to_string(rank) + ":A=");
          // printVector(lb, "j=" + std::to_string(j) + "|" + std::to_string(rank) + ":b=");

        // send leading row to other ranks
        std::vector<double> leadA(n);
        double leadb = 0;
        /*if (j % commSize == rank) {
            int li = (j - rank) / commSize;
            MPI_Send(lA.data() + li * n, n, MPI_DOUBLE, dest, rank * j, MPI_COMM_WORLD);
            MPI_Send(lb.data() + li, 1, MPI_DOUBLE, dest, (rank + commSize) * j, MPI_COMM_WORLD);
        } else {
            MPI_Status status;
            MPI_Recv(tmpA.data(), n, MPI_DOUBLE, dest, dest * j, MPI_COMM_WORLD, &status);
            MPI_Recv(&tmpb, 1, MPI_DOUBLE, dest, (dest + commSize) * j, MPI_COMM_WORLD, &status);
        }*/
        if (j % commSize == rank) {
            for (int k = 0; k < n; ++k)
                leadA[k] = lA[k + (j - rank) / commSize * n];
            leadb = lb[(j - rank) / commSize];
        }
        MPI_Bcast(leadA.data(), n, MPI_DOUBLE, j % commSize, MPI_COMM_WORLD);
        MPI_Bcast(&leadb, 1, MPI_DOUBLE, j % commSize, MPI_COMM_WORLD);

        if (isAlmostEqual(leadA[j], 0))
            continue;  // if leading element = 0, then go to the next row to avoid division by zero

        // printVector(leadA, "j=" + std::to_string(j) + "|" + std::to_string(rank) + ":A=");
        // std::cout << "j=" << j << "|" << rank << ":b=" << leadb << "\n";

        for (int i = j + 1; i < n; ++i) {
            if (i % commSize == rank) {
                double alfa = lA[j + (i - rank) / commSize * n] / leadA[j];
                for (int k = j; k < n; k++)
                    lA[k + (i - rank) / commSize * n] -= alfa * leadA[k];
                lb[(i - rank) / commSize] -= alfa * leadb;
            }
        }
          // printVector(lA, "j=" + std::to_string(j) + "|" + std::to_string(rank) + ":A=");
          // printVector(lb, "j=" + std::to_string(j) + "|" + std::to_string(rank) + ":b=");
    }

    // The second stage
    std::vector<double> lx(lbSize);
    for (int i = n - 1; i >= 0; i--) {
        double leadx = 0;
        int err = 0;

        if (i % commSize == rank) {
            if (isAlmostEqual(lA[i + (i - rank) / commSize * n], 0))  // if we have risen to a zero leading element
                if (isAlmostEqual(lb[(i - rank) / commSize], 0))  // if there is an expression 0x == 0
                    lx[(i - rank) / commSize] = 1;  // then the variable can take any real value, let it be 1
                else  // if there is an expression 0x == (b[i] - sum), where (b[i] - sum) != 0
                    err = 1;  // then there is no solution, return a null vector
            else
                lx[(i - rank) / commSize] = lb[(i - rank) / commSize] / lA[i + (i - rank) / commSize * n];
            leadx = lx[(i - rank) / commSize];
            // std::cout << "i=" << i << "|" << rank << ": leadx = " << leadx << "\n";
        }
        MPI_Bcast(&err, 1, MPI_INT, i% commSize, MPI_COMM_WORLD);
        if (err) {
            // std::cout << "There is no solution\n\n";
            return std::vector<double>();
        }
        MPI_Bcast(&leadx, 1, MPI_DOUBLE, i % commSize, MPI_COMM_WORLD);

        // printVector(lb, "b=");
        for (int k = lbSize - 1; k >= 0; --k)
            if (k * commSize + rank < i)
                lb[k] -= lA[i + k * n] * leadx;
    }
    // printVector(lx, "rank:" + std::to_string(rank) + " lx=");

    // gathering result
    if (commSize >= 2) {
        std::vector<double> gx(n);
        if (rank >= 1) {
            MPI_Send(lx.data(), lx.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        } else {
            for (int i = 0; i < n; i += commSize)
                gx[i] = lx[i / commSize];
            MPI_Status status;
            for (int _rank = 1; _rank < commSize; _rank++) {
                int _lxSize = (n % commSize - 1 >= _rank) ? n / commSize + 1 : n / commSize;
                // std::vector<double> _lx(_lxSize);
                MPI_Recv(lx.data(), _lxSize, MPI_DOUBLE, _rank, 0, MPI_COMM_WORLD, &status);
                for (int i = _rank; i < n; i += commSize)
                    gx[i] = lx[(i - _rank) / commSize];
            }
        }
        return gx;
    }
    else
       return lx;
}