#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

// predict function the same to linear KF and non-linear EKF
void KalmanFilter::Predict() {
    // u = 0, no external motion
    x_ = F_ * x_;
    MatrixXd Ft = F_.transpose();
    P_ = F_ * P_ * Ft + Q_;
}

// linear KF update function
void KalmanFilter::Update(const VectorXd &z) {
    VectorXd y = z - H_ * x_;
    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd Si = S.inverse();
    MatrixXd K =  P_ * Ht * Si;

    // new state
    x_ = x_ + (K * y);
    int x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;
}

// non-linear EKF update function
void KalmanFilter::UpdateEKF(const VectorXd &z) {
    // convert state x' to polar coordinates
    float px = x_(0);
    float py = x_(1);
    float vx = x_(3);
    float vy = x_(4);
    double rho = pow((pow(px, 2) + pow(py, 2)), 0.5);
    double theta = atan(py / px);
    double rho_dot = (px * vx + py * vy) / rho;
    VectorXd hx_(3);
    hx_ << rho, theta, rho_dot;
    VectorXd y = z - hx_;

    // check if y(1) element phi is in range ( - pi, pi)
    bool in_range = false;
    while (in_range == false){
        if (y(1) > 3.14159){
            y(1) = y(1) - 6.28318;
        } else if (y(1) < -3.14159){
            y(1) = y(1) + 6.28318;
        } else{
            in_range = true;
        }
    }

    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd Si = S.inverse();
    MatrixXd PHt = P_ * Ht;
    MatrixXd K = PHt * Si;

    // new estimate
    x_ = x_ + (K * y);
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;
}
