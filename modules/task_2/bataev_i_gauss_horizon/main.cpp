// Copyright 2023 Bataev Ivan
#include <gtest/gtest.h>
#include <vector>
#include "../../../modules/task_2/bataev_i_gauss_horizon/gauss_horizon.h"
#include <gtest-mpi-listener.hpp>

TEST(Vect_Alt_Signs_MPI, Test_Par_Count1) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const int n = 5;
    std::vector<double> M, x1, x2;

    if (rank == 0) {
        double matr[n][n+1] = {
            {0, 0, 12, -18, 5, -9},
            {-2, 4, 3, 5, 0, -7},
            {-1, 2, 3, 0, 1, -4},
            {-4, 8, 12, -6, 14, -1},
            {4, -8, -6, -10, 0, 14}
        };
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n + 1; ++j)
                M.push_back(matr[i][j]);

        printVector(M, "M = ");
    }

    x1 = gaussMethParallel(M, n);
    if (rank == 0) {
        if (x1.size()) { printVector(x1, "x = "); }
        else { std::cout << "There is no solution\n"; }

        x2 = gaussMethSequential(M, n);
        ASSERT_EQ(x1.size(), x2.size());
        for (int i = 0; i < x1.size(); ++i)
            ASSERT_NEAR(x1[i], x2[i], 0.0000001);  // to avoid rounding errors after floating point operations
    }
}

TEST(Vect_Alt_Signs_MPI, Test_Par_Count2) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const int n = 4;
    std::vector<double> M, x1, x2;

    if (rank == 0) {
        double matr[n][n+1] = {
            {4, -3, 2, -1, 8},
            {3, -2, 1, -3, 7},
            {-4, 3, -2, 1, -8},
            {5, -3, 1, -8, 1}
        };
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n+1; ++j)
                M.push_back(matr[i][j]);

        printVector(M, "M = ");
    }

    x1 = gaussMethParallel(M, n);
    if (rank == 0) {
        if (x1.size()) { printVector(x1, "x = "); }
        else { std::cout << "There is no solution\n"; }

        x2 = gaussMethSequential(M, n);
        ASSERT_EQ(x1.size(), x2.size());
        for (int i = 0; i < x1.size(); ++i)
            ASSERT_NEAR(x1[i], x2[i], 0.0000001);
    }
}

TEST(Vect_Alt_Signs_MPI, Test_Par_Count3) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const int n = 5, left = -50, right = 50;
    std::vector<double> M, x1, x2;

    if (rank == 0) {
        M = getRandomVector(n * (n + 1), left, right);
        printVector(M, "M = ");
    }

    x1 = gaussMethParallel(M, n);
    if (rank == 0) {
        if (x1.size()) { printVector(x1, "x = "); }
        else { std::cout << "There is no solution\n"; }

        x2 = gaussMethSequential(M, n);
        ASSERT_EQ(x1.size(), x2.size());
        for (int i = 0; i < x1.size(); ++i)
            ASSERT_NEAR(x1[i], x2[i], 0.0000001);
    }
}

TEST(Vect_Alt_Signs_MPI, Test_Par_Count4) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const int n = 30, left = -50, right = 50;
    std::vector<double> M, x1, x2;

    if (rank == 0)
        M = getRandomVector(n * (n + 1), left, right);

    x1 = gaussMethParallel(M, n);
    if (rank == 0) {
        x2 = gaussMethSequential(M, n);
        ASSERT_EQ(x1.size(), x2.size());
        for (int i = 0; i < x1.size(); ++i)
            ASSERT_NEAR(x1[i], x2[i], 0.0000001);
    }
}

TEST(Vect_Alt_Signs_MPI, Test_Par_Count5) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const int n = 100, left = -100, right = 100;
    std::vector<double> M, x1, x2;

    if (rank == 0)
        M = getRandomVector(n * (n + 1), left, right);

    x1 = gaussMethParallel(M, n);
    if (rank == 0) {
        x2 = gaussMethSequential(M, n);
        ASSERT_EQ(x1.size(), x2.size());
        for(int i = 0; i < x1.size(); ++i)
            ASSERT_NEAR(x1[i], x2[i], 0.0000001);  // to avoid rounding errors after floating point operations
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);

    ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
    ::testing::TestEventListeners& listeners =
        ::testing::UnitTest::GetInstance()->listeners();

    listeners.Release(listeners.default_result_printer());
    listeners.Release(listeners.default_xml_generator());

    listeners.Append(new GTestMPIListener::MPIMinimalistPrinter);
    return RUN_ALL_TESTS();
}
