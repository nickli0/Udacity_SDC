#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>
#include <math.h>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);


  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
        0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;

  // H matrix - laser
  H_laser_ << 1,0,0,0,
              0,1,0,0;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
    TODO:
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */

    // first measurement
    // initialize state vector ekf_.x_
    cout << "EKF: Initializing Kalman Filter" << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // state polar range
      float rho = measurement_pack.raw_measurements_[0];
      // state polar bearing
      float phi = measurement_pack.raw_measurements_[1];
      // state polar rho velocity/rate
      float rho_dot = measurement_pack.raw_measurements_[2];
      // convert state polar to cartesian coordinates
      float px = rho * cos(phi);
      float py = rho * sin(phi);
      float vx = rho_dot * cos(phi);
      float vy = rho_dot * sin(phi);
      // initialize radar state vector
      ekf_.x_ << px, py, vx, vy;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // initialize laser state vector, first measurement velocities set to zeroes
      ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0.0, 0.0;
    }

    // special initialization case by dividing zero
    if (fabs(ekf_.x_(0)) < 0.0001 ){
      ekf_.x_(0) = 0.0001;
      cout << "Error: Initial px in x state vector is too small" << endl;
    }
    if (fabs(ekf_.x_(1)) < 0.0001 ){
      ekf_.x_(1) = 0.0001;
      cout << "Error: Initial py in x state vector is too small" << endl;
    }

    // initialize state covariance matrix ekf_.P_
    ekf_.P_ = MatrixXd(4,4);
    ekf_.P_ << 1,0,0,0,
            0,1,0,0,
            0,0,1000,0,
            0,0,0,1000;
    // print initial state vector x
    cout << "EKF Initial State Vector x is " << ekf_.x_ << endl;
    // save initial timestamp dt
    previous_timestamp_ = measurement_pack.timestamp_;
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }


  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
   TODO:
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */

  // calculate timestep in second
  float dt = (measurement_pack.timestamp_ - previous_timestamp_);
  dt = dt / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;
  // update state transition matrix F
  ekf_.F_ = MatrixXd(4,4);
  ekf_.F_ << 1, 0, dt, 0,
          0, 1, 0, dt,
          0, 0, 1, 0,
          0, 0, 0, 1;
  // set process noise covariance Q matrix
  float noise_ax = 9.0;
  float noise_ay = 9.0;
  float dt_2 = pow(dt, 2);
  float dt_3 = pow(dt, 3);
  float dt_4 = pow(dt, 4);
  ekf_.Q_ = MatrixXd(4,4);
  ekf_.Q_ << noise_ax * (dt_4 / 4), 0, noise_ax * (dt_3 / 2), 0,
          0, noise_ay * (dt_4 / 4), 0, noise_ay * (dt_3 / 2),
          noise_ax * (dt_3 / 2), 0, noise_ax * dt_2, 0,
          0, noise_ay * (dt_3 / 2), 0, noise_ay * dt_2;
  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/
  /**
   TODO:
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    Tools tools;
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // Laser updates
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
