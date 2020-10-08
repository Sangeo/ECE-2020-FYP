/*
 * Filter Coefficients (C Source) generated by the Filter Design and Analysis Tool
 * Generated by MATLAB(R) 9.6 and Signal Processing Toolbox 8.2.
 * Generated on: 03-Oct-2020 16:20:22
 */

/*
 * Discrete-Time IIR Filter (real)
 * -------------------------------
 * Filter Structure    : Direct-Form II, Second-Order Sections
 * Number of Sections  : 14
 * Stable              : Yes
 * Linear Phase        : No
 */

/* General type conversion for MATLAB generated C-code  */
#include "tmwtypes.h"
/* 
 * Expected path to tmwtypes.h 
 * C:\Program Files\MATLAB\R2019a\extern\include\tmwtypes.h 
 */
#define MWSPT_NSEC 29
const int NL[MWSPT_NSEC] = { 1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1 };
const real64_T NUM[MWSPT_NSEC][3] = {
  {
     0.8971272345387,                 0,                 0 
  },
  {
                   1,   -1.976200588789,                 1 
  },
  {
     0.8971272345387,                 0,                 0 
  },
  {
                   1,   -1.663548103609,                 1 
  },
  {
     0.8716956935806,                 0,                 0 
  },
  {
                   1,    -1.97762458728,                 1 
  },
  {
     0.8716956935806,                 0,                 0 
  },
  {
                   1,   -1.643925229086,                 1 
  },
  {
     0.8324454793136,                 0,                 0 
  },
  {
                   1,   -1.980488218947,                 1 
  },
  {
     0.8324454793136,                 0,                 0 
  },
  {
                   1,   -1.596671248617,                 1 
  },
  {
     0.7703078213498,                 0,                 0 
  },
  {
                   1,   -1.984752131367,                 1 
  },
  {
     0.7703078213498,                 0,                 0 
  },
  {
                   1,    -1.49756794473,                 1 
  },
  {
     0.6707780982686,                 0,                 0 
  },
  {
                   1,   -1.990124359858,                 1 
  },
  {
     0.6707780982686,                 0,                 0 
  },
  {
                   1,    -1.27306604699,                 1 
  },
  {
     0.5194795641971,                 0,                 0 
  },
  {
                   1,   -1.995666686329,                 1 
  },
  {
     0.5194795641971,                 0,                 0 
  },
  {
                   1,  -0.6545263925076,                 1 
  },
  {
      0.349390182384,                 0,                 0 
  },
  {
                   1,   -1.999458329619,                 1 
  },
  {
      0.349390182384,                 0,                 0 
  },
  {
                   1,    1.209280309009,                 1 
  },
  {
                   1,                 0,                 0 
  }
};
const int DL[MWSPT_NSEC] = { 1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1 };
const real64_T DEN[MWSPT_NSEC][3] = {
  {
                   1,                 0,                 0 
  },
  {
                   1,   -1.674969781853,   0.9720768960541 
  },
  {
                   1,                 0,                 0 
  },
  {
                   1,   -1.964590760094,   0.9912832898973 
  },
  {
                   1,                 0,                 0 
  },
  {
                   1,   -1.614069266615,    0.912730439423 
  },
  {
                   1,                 0,                 0 
  },
  {
                   1,   -1.947688412957,   0.9731390566991 
  },
  {
                   1,                 0,                 0 
  },
  {
                   1,   -1.529224698662,   0.8383328403577 
  },
  {
                   1,                 0,                 0 
  },
  {
                   1,   -1.929158218751,   0.9524032459198 
  },
  {
                   1,                 0,                 0 
  },
  {
                   1,   -1.406162764473,   0.7334719345343 
  },
  {
                   1,                 0,                 0 
  },
  {
                   1,   -1.906600888143,   0.9268136539047 
  },
  {
                   1,                 0,                 0 
  },
  {
                   1,   -1.877010109317,   0.8937107949808 
  },
  {
                   1,                 0,                 0 
  },
  {
                   1,   -1.231442537394,   0.5804253224478 
  },
  {
                   1,                 0,                 0 
  },
  {
                   1,   -1.837562927723,   0.8510211321487 
  },
  {
                   1,                 0,                 0 
  },
  {
                   1,   -1.017341322014,   0.3782568300838 
  },
  {
                   1,                 0,                 0 
  },
  {
                   1,   -1.794382477042,   0.8060762326902 
  },
  {
                   1,                 0,                 0 
  },
  {
                   1,  -0.8534251758965,   0.2011588383441 
  },
  {
                   1,                 0,                 0 
  }
};
