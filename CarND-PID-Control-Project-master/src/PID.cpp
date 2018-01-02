#include "PID.h"
#include <iostream>
#include <math.h>

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double _Kp, double _Ki, double _Kd) {
	Kp=_Kp;
	Ki=_Ki;
	Kd=_Kd;
	precte=0.0;
	intcte=0.0;
    dpp[0]=1;
    dpp[1]=1;
    dpp[2]=1;
	total_err=0;
	keyframeidex=0;
	n=10;
}

void PID::UpdateError(double cte) {
	diff_cte=cte-precte;
	precte=cte;
	intcte+=cte;
	p_error=-Kp*cte;
	d_error=-Kd*diff_cte;
	i_error=-Ki*intcte;
}

double PID::TotalError() {
	double terror=p_error+d_error+i_error;
	return terror;
}

/*void PID::SetGain(double Kp, double Ki, double Kd)
{
	Kp=Kp;
	Ki=Ki;
	Kd=Kd;
}
*/
void PID::UpdateGain(double _Kp, double _Ki, double _Kd)
{
	Kp=_Kp;
	Ki=_Ki;
	Kd=_Kd;
}
double PID::Twiddle(double cte)
{
	double p[3]={Kp,Ki,Kd};
	double dp[3]={dpp[0],dpp[1],dpp[2]};
	diff_cte=cte-precte;
	precte=cte;
	intcte+=cte;
	int iter=0;
	double p_error1=-p[0]*cte;
	double d_error1=-p[2]*diff_cte;
	double i_error1=-p[1]*intcte;	
	total_err+=pow(cte,2);//p_error1+d_error1+i_error1;
	keyframeidex++;
	if(keyframeidex==100)
	{
	while((dp[0]+dp[1]+dp[2])>0.00000000000000000000001)
	{
		iter++;	
		for(int i=0;i<3;i++)
		{
		        p[i]+=dp[i];
				//printf("k:%lf %lf %lf\n",p[0],p[1],p[2]);
		        p_error1=-p[0]*cte;
				d_error1=-p[2]*diff_cte;
				i_error1=-p[1]*intcte;	
				double e=total_err;//p_error1+d_error1+i_error1;
				//std::cout<<total_err<<"  "<<e<<std::endl;
		        //printf("%f\n",e);
				/*if(e>1)
				{
					e=1;
				}
				else if(e<-1)
				{
					e=-1;
				}
*/
		        if(e<total_err)
		        {
		            total_err=e;
		            dp[i]*=1.1;
		        }
		        else
		        {
					p[i]-=(2*dp[i]);	
					//e=total_err;//p_error1+i_error1+d_error1;
					p_error1=-p[0]*cte;
					d_error1=-p[2]*diff_cte;
					i_error1=-p[1]*intcte;	
				    e=p_error1+d_error1+i_error1;
					/*if(e>1)
					{
						e=1;
					}
					else if(e<-1)
					{
						e=-1;
					}
*/
		            if(e<total_err)
				    {
				        total_err=e;
				        dp[i]*=1.1;
				    }
		            else
		            {
		                p[i]+=dp[i];
		                dp[i]*=0.9;
						//printf("k:%lf %lf %lf\n",p[0],p[1],p[2]);
		            }
		        }
				//printf("k:%lf %lf %lf\n",p[0],p[1],p[2]);
		}
	}
		printf("k:%lf %lf %lf\n",p[0],p[1],p[2]);
	}
	std::cout<<"iter:"<<iter<<std::endl;
    UpdateGain(p[0], p[1], p[2]);
	//dpp[0]=dp[0];
	//dpp[1]=dp[1];
	//dpp[2]=dp[2];

	p_error1=-p[0]*cte;
	d_error1=-p[2]*diff_cte;
	i_error1=-p[1]*intcte;

	if(p_error1+d_error1+i_error1>1)
	{
		return 1;
	}
	else if(p_error1+d_error1+i_error1<-1)
	{
		return -1;
	}
	return p_error1+d_error1+i_error1;
	
}
