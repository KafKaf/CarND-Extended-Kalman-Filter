#ifndef TOOLS_H_
#define TOOLS_H_
#include <vector>
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

class Tools {
public:
  /**
  * Constructor.
  */
  Tools();

  /**
  * Destructor.
  */
  virtual ~Tools();

  /**
  * A helper method to calculate RMSE.
  */
  VectorXd CalculateRMSE(const vector<VectorXd> &estimations, const vector<VectorXd> &ground_truth);

  /**
  * A helper method to calculate Jacobians.
  */
  MatrixXd CalculateJacobian(const VectorXd& x_state);

  /**
  * A helper method to convert cartesian to polar coordinates.
  */
  VectorXd ConvertCartesianToPolar(const VectorXd& x_state);

  /**
  * A helper method to convert polar to cartesian coordinates.
  */
  VectorXd ConvertPolarToCartesian(const VectorXd& x_state);

  /**
  * A helper method to normalize radians to be between -pi and pi
  */
  float normalizeAngle(float angleInRadians);

};

#endif /* TOOLS_H_ */
