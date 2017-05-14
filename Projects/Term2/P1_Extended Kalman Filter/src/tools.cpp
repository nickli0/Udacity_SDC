#include <iostream>
#include "tools.h"

using namespace std;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;


Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
    // initiate variable rmse
    VectorXd rmse(4);
    rmse << 0,0,0,0;
    // check input estimations validity
    if(estimations.size() == 0){
        cout << "Error: Empty Input" << endl;
        return rmse;
    }
    // check the size of two vector inputs sizes are the same
    if(estimations.size() != ground_truth.size()){
        cout << "Error: estimation and ground_truth are NOT same size." << endl;
        return rmse;
    }
    // calculate squared residuals
    for(unsigned int i=0; i < estimations.size(); ++i){
        VectorXd residual = estimations[i] - ground_truth[i];
        // coefficient-wise multiplication
        residual = pow(residual.array(), 2);
        rmse += residual;
    }

    // calculate mean
    rmse = rmse / estimations.size();
    rmse = rmse.array().sqrt();       // rmse = pow(rmse.array(), 0.5)
    return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
    // initiate variables
    float px = x_state(0);
    float py = x_state(1);
    float vx = x_state(2);
    float vy = x_state(3);
    MatrixXd Hj(3,4);

    float c1 = px*px + py*py;
    float c2 = sqrt(c1);
    float c3 = (c1*c2);

    //check special case of division by zero
    if(fabs(c1) < 0.0001){
        cout << "Error: Jacobian Matrix Calculation Division by Zero" <<endl;
        return Hj;
    }

    //compute Jacobian matrix
    Hj << (px/c2), (py/c2), 0, 0,
            -(py/c1), (px/c1), 0, 0,
            py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

    return Hj;
}
