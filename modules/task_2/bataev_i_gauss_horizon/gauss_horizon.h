// Copyright 2023 Bataev Ivan
#ifndef MODULES_TASK_2_BATAEV_I_GAUSS_HORIZON_GAUSS_HORIZON_H_
#define MODULES_TASK_2_BATAEV_I_GAUSS_HORIZON_GAUSS_HORIZON_H_

#include <vector>
#include <string>

std::vector<double> getRandomVector(int size, int left, int right);
void printVector(const std::vector<double>& v, const std::string& prefix = "");
void printFullMatr(const std::vector<double>& M, const int n, const std::string& prefix = "");
std::vector<double> gaussMethSequential(std::vector<double> M, const int n);
std::vector<double> gaussMethParallel(std::vector<double> M, const int n);

#endif  // MODULES_TASK_2_BATAEV_I_GAUSS_HORIZON_GAUSS_HORIZON_H_
