#include "kalman_filter.h"
#include "tools.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

KalmanFilter::KalmanFilter() {
  Tools tools;
  I_ = MatrixXd::Identity(4, 4);
}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  Ht_ = H_in.transpose();
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
    * predict the state
  */
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
    * update the state by using Kalman Filter equations
  */
  VectorXd y = z - H_ * x_;
//  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht_ + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K =  P_ * Ht_ * Si;

  //new state
  x_ = x_ + (K * y);
  P_ = (I_ - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
    * update the state by using Extended Kalman Filter equations
  */

  VectorXd hx = tools.ConvertCartesianToPolar(x_);
  MatrixXd Hj = tools.CalculateJacobian(x_);

  VectorXd y = z - hx;
  // normalize the angle
  y(1) = tools.normalizeAngle(y(1));

  MatrixXd Hjt = Hj.transpose();
  MatrixXd S = Hj * P_ * Hjt + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K =  P_ * Hjt * Si;

  //new state
  x_ = x_ + (K * y);
  P_ = (I_ - K * Hj) * P_;
}
