
/* saved from url=(0050)http://www.ailab.si/dorian/MLControl/BikeCCode.htm */
#include <math.h>
#include <stdio.h> 
#include "parametre.h"


float calc_dist_to_goal(float xf, float xb, float yf, float yb);
float calc_angle_to_goal(float xf, float xb, float yf, float yb);

#define R1        -1.0
#define R2         0.0
#define R3        +0.01
#define R_FAKTOR ((float) 0.0001) 
#define NO_STATES2  20
#define sqr(x)              ((x)*(x))

#define dt             0.01   
#define v            (10.0/3.6)  /* 10 km/t in m/s */
#define g   9.82
#define dCM 0.3
#define c   0.66
#define h   0.94
#define Mc  15.0
#define Md  1.7
#define Mp  60.0
#define M   (Mc + Mp)
#define R   0.34          /* tyre radius */
#define sigma_dot  ( ((float) v) /R)
#define I_bike ((13.0/3)*Mc*h*h + Mp*(h+dCM)*(h+dCM))
#define I_dc             (Md*R*R)  
#define I_dv     ((3.0/2)*Md*R*R)  
#define I_dl     ((1.0/2)*Md*R*R)  
#define l              1.11     /* distance between the point where
   the front and back tyre touch the ground */
#define pi           3.1415927

/* position of goal */
const float x_goal=1000, y_goal=0, radius_goal=10;

static unsigned long i;

void bike(long* return_state, long* return_state2, int action, 
  float *reinforcement, int to_do)
{
  static float omega, omega_dot, omega_d_dot,
    theta, theta_dot, theta_d_dot, 
    xf, yf, xb, yb;         /* tyre position */
  static double rCM, rf, rb;
  static float T, d, phi,
    psi,            /* bike's angle to the y-axis */
    psi_goal;       /* Angle to the goal */
  float temp;

  T = 2*((action / 3)-1);
  d = 0.02*((action % 3)-1);
  d = d + 0.04*(0.5-random); /* Max noise is 2 cm */
  
  switch (to_do) {
  case initialize : { 

    omega = omega_dot = omega_d_dot = 0;
    theta =  theta_dot = theta_d_dot = 0;
    xb = 0; yb = 0;
    xf = 0; yf = l;
    psi =  atan((xb-xf)/(yf-yb));
    psi_goal = calc_angle_to_goal(xf, xb, yf, yb);
  
    break;
  }

  case execute_action : {

    if (theta == 0) {
      rCM = rf = rb = 9999999; /* just a large number */
    } else {
      rCM = sqrt(pow(l-c,2) + l*l/(pow(tan(theta),2)));
      rf = l / fabs(sin(theta));
      rb = l / fabs(tan(theta));
    } /* rCM, rf and rb are always positiv */

    /* Main physics eq. in the bicycle model coming here: */
    phi = omega + atan(d/h);
    omega_d_dot = ( h*M*g*sin(phi) 
                  - cos(phi)*(I_dc*sigma_dot*theta_dot + sign(theta)*v*v*
                              (Md*R*(1.0/rf + 1.0/rb) +  M*h/rCM))
                  ) / I_bike;
    theta_d_dot =  (T - I_dv*omega_dot*sigma_dot) /  I_dl;

    /*--- Eulers method ---*/
    omega_dot += omega_d_dot * dt;
    omega += omega_dot * dt;
    theta_dot += theta_d_dot * dt;
    theta += theta_dot * dt;

    if (fabs(theta) > 1.3963) { /* handlebars cannot turn more than 
       80 degrees */
      theta = sign(theta) * 1.3963;
    }

    /* New position of front tyre */
    temp = v*dt/(2*rf);                             
    if (temp > 1) temp = sign(psi + theta) * pi/2;
    else temp = sign(psi + theta) * asin(temp); 
    xf += v * dt * (-sin(psi + theta + temp));
    yf += v * dt * cos(psi + theta + temp);
      
     /* New position of back tyre */
    temp = v*dt/(2*rb);               
    if (temp > 1) temp = sign(psi) * pi/2;
    else temp = sign(psi) * asin(temp); 
    xb += v * dt * (-sin(psi + temp));
    yb += v * dt * (cos(psi + temp));

    /* Round off errors accumulate so the length of the bike changes over many
    iterations. The following take care of that: */
    temp = sqrt((xf-xb)*(xf-xb)+(yf-yb)*(yf-yb));
    if (fabs(temp - l) > 0.01) {
      xb += (xb-xf)*(l-temp)/temp;
      yb += (yb-yf)*(l-temp)/temp;
    }

    temp = yf - yb;
    if ((xf == xb) && (temp < 0)) psi = pi;
    else {
      if (temp > 0) psi = atan((xb-xf)/temp);
      else psi = sign(xb-xf)*(pi/2) - atan(temp/(xb-xf));
    }

    psi_goal = calc_angle_to_goal(xf, xb, yf, yb);

    break;
  }

  default: ;
  }
   
  /*-- Calculation of the reinforcement  signal --*/
  if (fabs(omega) &gt; (pi/15)) { /* the bike has fallen over */
    *reinforcement = R1;
    /* a good place to print some info to a file or the screen */  
  } else { 
    temp = calc_dist_to_goal(xf, xb, yf, yb);

    if (temp &lt; 1e-3) *reinforcement = R3;
    else *reinforcement = (0.95 - sqr(psi_goal)) * R_FACTOR; 
  }


  /* There are two sorts of state information. The first (*return_state) is
     about the state of the bike, while the second (*return_state2) deals with the
     position relative to the goal */
  *return_state = get_box(theta, theta_dot, omega, omega_dot, omega_d_dot,
    psi_goal);

  i = 0 ; *return_state2 = -1; 
  while (*return_state2 &lt; 0) {
    temp = -2 + ((float) (4*(i)))/NO_STATES2;
    if (psi_goal &lt; temp) *return_state2 = i;
    i++;
  }
  
}

int get_box(float theta, float theta_dot, float omega, 
float omega_dot, float omega_d_dot, float psi_goal)
{
  int box;
 
  if (theta &lt; -1)                           box = 0; 
  else if (theta &lt; -0.2)                    box = 1; 
  else if (theta &lt; 0)                       box = 2; 
  else if (theta &lt; 0.2)                     box = 3; 
  else if (theta &lt; 1)                       box = 4; 
  else                                      box = 5; 
  /* The last restriction is taken care off in the physics part */
  
  if (theta_dot &lt; -2)                       ;
  else if (theta_dot &lt; 0)                   box += 6; 
  else if (theta_dot &lt; 2)                   box += 12;
  else                                      box += 18;

  if (omega &lt; -0.15)                        ;
  else if (omega &lt; -0.06)                   box += 24; 
  else if (omega &lt; 0)                       box += 48; 
  else if (omega &lt; 0.06)                    box += 72; 
  else if (omega &lt; 0.15)                    box += 96; 
  else                                      box += 120; 

  if (omega_dot &lt; -0.45)                    ;
  else if (omega_dot &lt; -0.24)               box += 144;
  else if (omega_dot &lt; 0)                   box += 288;
  else if (omega_dot &lt; 0.24)                box += 432;
  else if (omega_dot &lt; 0.45)                box += 576;
  else                                      box += 720;

  if (omega_d_dot &lt; -1.8)                   ;
  else if (omega_d_dot &lt; 0)                 box += 864; 
  else if (omega_d_dot &lt; 1.8)               box += 1728;
  else                                      box += 2592;

  return(box);
}


float calc_dist_to_goal(float xf, float xb, float yf, float yb)
{
  float temp;

  temp = sqrt(max(0, (x_goal-xf)*(x_goal-xf) + (y_goal-yf)*(y_goal-yf) 
             - radius_goal*radius_goal)); 
  return(temp);
}


float calc_angle_to_goal(float xf, float xb, float yf, float yb)
{
  float temp, scalar, tvaer;

  temp = (xf-xb)*(x_goal-xf) + (yf-yb)*(y_goal-yf); 
  scalar =  temp / (l * sqrt(sqr(x_goal-xf)+sqr(y_goal-yf)));
  tvaer = (-yf+yb)*(x_goal-xf) + (xf-xb)*(y_goal-yf); 

  if (tvaer &lt;= 0) temp = scalar - 1;
  else temp = fabs(scalar - 1);

  /* These angles are neither in degrees nor radians, but something
     strange invented in order to save CPU-time. The measure is arranged the
     same way as radians, but with a slightly different negative factor. 

     Say, the goal is to the east.
     If the agent rides to the east then  temp = 0
     - " -          - " -   north              = -1
     - " -                  west               = -2 or 2
     - " -                  south              =  1 */ 

  return(temp);
}
