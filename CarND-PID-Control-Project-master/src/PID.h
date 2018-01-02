#ifndef PID_H
#define PID_H

class PID {
public:
  /*
  * Errors
  */
  double p_error;
  double i_error;
  double d_error;

  /*
  * Coefficients
  */ 
  double Kp;
  double Ki;
  double Kd;

  double dpp[3];
  /*
  * Constructor
  */
  PID();

  /*
  * Destructor.
  */
  virtual ~PID();

  /*
  * Initialize PID.
  */
  void Init(double _Kp, double _Ki, double _Kd);

  /*
  * Update the PID error variables given cross track error.
  */
  void UpdateError(double cte);

  /*
  * Calculate the total PID error.
  */
  double TotalError();

	//
	double precte;
	double intcte;
	double diff_cte;

	double total_err;
	double t1;
	double t2;
	double t3;
	//void SetGain(double Kp, double Ki, double Kd);
	void UpdateGain(double _Kp, double _Ki, double _Kd);
	double Twiddle(double cte);

    int keyframeidex;
	int n;
};

#endif /* PID_H */
