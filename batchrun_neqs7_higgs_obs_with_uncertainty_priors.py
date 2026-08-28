# Core imports man
import sys
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import time
import pygsl.odeiv as odeiv


# Numerical libraries
from scipy.integrate import solve_ivp, odeint, cumulative_trapezoid
from scipy.interpolate import (
    UnivariateSpline, splrep, splev, CubicSpline,
    interp1d, PchipInterpolator, InterpolatedUnivariateSpline
)

import numdifftools as nd

# Random number generator (GSL)
import pygsl.rng
rng = np.random.default_rng()


# custom InflationModels code to path the one below is for wkb approximation method
sys.path.append(
    '/Users/epmeador/Desktop/research/rwarthur/inflation_gravitywaves/inflation_code/InflationModels'
)

# from IPython.display import clear_output

#the path below assumes the original numerics from Deyan's code
# sys.path.append('/Users/epmeador/Desktop/InflationModels-master')

# Enable spectra
# SPECTRUM = True
SPECTRUM = True


# Local modules from InflationModels
from MacroDefinitions import *
from calcpath import *
from int_de import *
if SPECTRUM:
#     from spectrum_noS0_fixedNstar import * #this is mode modified one that I have been working with for wkb approx
#     from spectrum_OG import * #this is the original spectrum from full numerical code
#     from spectrum_pyoscode_tensor_clean import * #this is the spectrum from pyoscode
#     from spectrum_pyoscode_scalar import * #this is the spectrum from pyoscode with scalar
#     from spectrum_pyoscode_optimized_wkb import *
#     from spectrum_pyoscode_test_N import *
    from spectrum_OG_nanoscale_nodiagnostics import * #this is equivalent to the original spectrum from full numerical code w/o diagnostics



# # ========================
# # GLOBAL SETTINGS
# # ========================
NEQS = 7
BATCH_ID = 9   # will need to change this as I go: 1, 2, 3, 4, 5

SAVEPATHS = True

# The sampled slow-roll parameters remain defined at N=60.
# After calcpath constructs the usual N=0 -> 60 trajectory,
# extend that same trajectory backward to N=70. If I had diff trun uncertainty, I could
# alter extension.
N_ANCHOR = 60.0
N_EXTEND = 70.0
EXTENSION_KMAX = 20000

SAVEPATHS = True

## For Higgs N=60:
## ns = 0.969485
## r = 2.737e-03
#NMAX = 0.97
#NMIN = 0.96
##0.001 gives 3sigma measurement of amp
#R_TARGET = 2.737e-03 #0.00335
##Try doubling error bar to 0.002
#R_DELTA = 0.002
#RMIN = R_TARGET - R_DELTA
#RMAX = R_TARGET + R_DELTA
##ALPHA_TARGET =
#ALPHA_MIN = -0.005
#ALPHA_MAX = 0.005


# For Higgs N=55:
#r_first_order = 0.00365885
#r_second_order = 0.0036694
#ns_first_order = 0.964163
#ns_second_order = 0.964613
#alphas_first_order = 0
#alphas_second_order = -0.000630587

### ORIGINALLY INCLUDED CMB CUTS AT THIS STAGE
### EXPECTATIONS FOR SCALAR INDEX
#NMAX = 0.97
#NMIN = 0.96
#
### TENSOR TO SCALAR RATIO
##0.001 gives 3sigma measurement of amp
#R_TARGET = 0.003664125
##Try doubling error bar to 0.002
#R_DELTA = 0.002
#
#RMIN = R_TARGET - R_DELTA
#RMAX = R_TARGET + R_DELTA
#
### THE RUNNING
#ALPHA_TARGET = -0.000630587
#ALPHA_DELTA = 0.002
#
#ALPHA_MIN = ALPHA_TARGET - ALPHA_DELTA
#ALPHA_MAX = ALPHA_TARGET + ALPHA_DELTA


## SUPER BROAD CUTS TO OBSERVABLES
NMAX = 0.99
NMIN = 0.95
RMIN = 0
RMAX = 0.038
#ALPHA_MIN = -0.02
#ALPHA_MAX = 0.02


### Test of Interest To Us
### For Higgs
EPSILON_BASE = 0.000209237
SIGMA_BASE = -0.0342419
LAM2_BASE = 0.000278972
LAM3_BASE = -4.60971e-6
LAM4_BASE = 6.87065e-08
LAM5_BASE = -8.92461e-9


#Kinney Range:
LAM5_MIN = -5e-5
LAM5_MAX = 5e-5

LAM4_MIN = -5e-4
LAM4_MAX = 5e-4

LAM3_MIN = -5e-3
LAM3_MAX = 5e-3

LAM2_MIN = -5e-2
LAM2_MAX = 5e-2

#ORIGINAL RUN INCLUDED FIRST ORDER APPROXIMATION OF EPS AND SIGMA
#SIGMA_MIN = NMIN - 1
#SIGMA_MAX = NMAX - 1
#
#EPSILON_MIN = RMIN/16
#EPSILON_MAX = RMAX/16

#WE NOW WANT TO RUN THIS OVER THE AMOUNT THAT EPS AND SIG VARY FROM N=60 DOWN TO N=45 WITHIN 2SIGMA

#EPSILON
EXPECTED_FIDUCIAL_EPSILON_AT_N60 = 2.1528510000e-04
EXPECTED_DELTA_EPSILON = 2.8480436771e-04
DESIRED_EPSILON_WIDTH_FROM_N60_TO_N45 = 2 * EXPECTED_DELTA_EPSILON
EPSILON_MIN = 0
EPSILON_MAX = EXPECTED_FIDUCIAL_EPSILON_AT_N60 + EXPECTED_DELTA_EPSILON

#SIGMA
EXPECTED_FIDUCIAL_SIGMA_AT_N60 = -3.4745380000e-02
EXPECTED_DELTA_SIGMA = 1.8406802056e-02
DESIRED_SIGMA_WIDTH_FROM_N60_TO_N45 = 2 * EXPECTED_DELTA_SIGMA
SIGMA_MIN = EXPECTED_FIDUCIAL_SIGMA_AT_N60 - EXPECTED_DELTA_SIGMA
SIGMA_MAX = EXPECTED_FIDUCIAL_SIGMA_AT_N60 + EXPECTED_DELTA_SIGMA


NUM_LAM_GRID = 100

## Here we are writing out our output files we can name to analyze data
# BASE_OUTDIR = "/Users/epmeador/Desktop/research/rwarthur/inflation_gravitywaves/inflation_code/Slow-Roll Parameters Tests/higgs_potential_tests/neqs6"
BASE_PATH_ROOT = "/Users/epmeador/Desktop/research/rwarthur/inflation_gravitywaves/inflation_code/Slow-Roll Parameters Tests/higgs_potential_tests"
#BASE_OUTDIR = f"{BASE_PATH_ROOT}/neqs{NEQS}"

## For a batch run we save differently,
BASE_OUTDIR = f"{BASE_PATH_ROOT}/neqs{NEQS}_higgs_batch{BATCH_ID:03d}"

# RNG initialiation process
#This can be used to generate a random generator that works well with simulations

# my_random = pygsl.rng.ranlxd2() #gsl random number generator
# my_random.set(0) #seeds with 0, fully reproducible
# np.random.seed(None) #seeds numpy separately, need to set to None instead of 0 I think

#This was with the batch runs
#np.random.seed(12345 + BATCH_ID)

## Modifying so I dont have to print each time
VERBOSE = False              # master debug switch
PRINT_FIRST_TRIAL = True     # print detailed info for first trial
PRINT_EVERY = 1000           # progress update frequency
PRINT_REJECTIONS = False     # rejection spam


#Here we define a class for several variables, this will initialize several variables
#The size we are setting for Y and initY is the same as NEQs which may be number of equations and is set in flow.py
#We set a state, the initial state, an empty string in the class, number of points, and e-folds

class Calc:
    def __init__(self):
        self.Y = np.zeros(NEQS, dtype=float, order='C')
        self.initY = np.zeros(NEQS, dtype=float, order='C')
        self.ret = ""
        self.npoints = 0
        self.Nefolds = 0.0

#Below we are initializing our starting slow roll parameter values
#We should be able to choose what ell we are extending to.

#NEQS=7
def pick_init_vals(eps, sig, lam2, lam3, lam4):

    init_vals = np.zeros(NEQS, dtype=float, order='C')

    init_vals[0] = 5.5/np.sqrt(8*np.pi)
    init_vals[1] = 1.0
    init_vals[2] = eps
    init_vals[3] = sig
    init_vals[4] = lam2
    init_vals[5] = lam3
    init_vals[6] = lam4
#    init_vals[7] = lam5

    init_Nefolds = 60
    return init_vals, init_Nefolds


# ## For Quad Inflation Deformation
# def pick_init_vals(lam3, lam4, lam5):

#     init_vals = np.zeros(NEQS, dtype=float, order='C')

#     init_vals[0] = 19.5
#     init_vals[1] = 1.0
#     init_vals[2] = 0.00213166
#     init_vals[3] = -0.0356948
#     init_vals[4] = -0.000228347
#     init_vals[5] = lam3
#     init_vals[6] = lam4
#     init_vals[7] = lam5

#     init_Nefolds = 60
#     return init_vals, init_Nefolds



# #SAVING PATHS
# #This next part of the code will print either asymptote or nontrivial
# #And decides when to save the code
# #This is based on what the model itself is doing

#This function below will calculation the spectrum computed from y
#As long as its in a particular range
#This is where I would limit my range of spectrum models of interest
#Otherwise it returns false

#if cut includes ns and r
#def we_should_calc_spec(y):
#    ns = specindex(y)
#    r = tsratio(y)
#    return (NMIN < ns < NMAX) and (RMIN < r < RMAX)

#we shoudl cal spec is defined but not actually used, the cuts are applied much later
## if we include ns, r, and alpha
#def we_should_calc_spec(y):
#    ns = specindex(y)
#    r = tsratio(y)
#    alpha_s = dspecindex(y)
#
#    return (
#        (NMIN < ns < NMAX)
#        and (RMIN < r < RMAX)
#        and (ALPHA_MIN < alpha_s < ALPHA_MAX)
#    )

#Specifically, we will save a model with some interesting dynamics
#And this means either asymptote or nontrivial - for us nontrivial
#Also only if the path has not been saved yet, and only for
#Every nth successful model
#This governs path saving in the non-spectral case


#we should save path is defined but not actually used, the cuts are applied much later
#def we_should_save_path(retval, save, pointcount, printevery):
#    return (retval == "nontrivial") and (not save) and (pointcount % printevery == 0)

#Overall, this writes model trajectory to a file
#the SRP at each integration step, the number of N remaining, gives a reconstructed V, and e_H
    # Open output file

def save_path(y, N, kount, fname):
    with open(fname, "w") as outfile:
    # Output intermediate data from the integration
        for i in range(kount):
            for j in range(NEQS):
            #get y variable as a function of i in kount
            #should be like SRP as a function of time
            #j loops over NEQS which are used
                outfile.write("%le " % y[j, i])
            outfile.write("%lf " % N[i])
            #Will also write out N at that time step
#             V = 3 * y[1, i]**2 * (1. - y[2, i]/3.) #this one is wrong, i think this is what he had
            V = (3./(8.*np.pi)) * y[1, i] * y[1, i] * (1.-y[2, i]/3.) #what the original code had, need 8pi
            outfile.write("%le %le\n" % (V, (V*y[2, i])/(3. - y[2, i]))) #I think this is KE


def extend_path_to_larger_N(
    y_at_anchor,
    N_anchor=60.0,
    N_target=70.0,
    kmax_ext=20000,
):
    """
    Starting from a state defined at N_anchor (normally N=60),
    integrate the SAME truncated flow hierarchy backward in time
    toward larger N, ending at N_target (normally N=70).

    This does NOT change the physical meaning of the sampled
    slow-roll parameters: they remain defined at N=60.

    Returns
    -------
    path_ext : ndarray, shape (NEQS, n_ext)
        Flow trajectory from ~N_anchor to N_target.

    N_ext : ndarray, shape (n_ext,)
        Corresponding N values.

    y_Ntarget : ndarray, shape (NEQS,)
        State reached at N_target.
    """
#Built like calcpath
    y_ext = y_at_anchor.copy()

#allocate buffers for integration like before
    yp_ext = np.zeros(
        (NEQS, kmax_ext),
        dtype=float,
        order='C'
    )

    xp_ext = np.zeros(
        kmax_ext,
        dtype=float,
        order='C'
    )

    kount_ext = None

#integrate y from N_anchor to N_target given these parameters
    z_ext, kount_ext = int_de(
        y_ext,
        N_anchor,
        N_target,
        kount_ext,
        kmax_ext,
        yp_ext,
        xp_ext,
        NEQS,
        derivs
    )

    if z_ext:
        raise RuntimeError(
            f"Extension integration failed: "
            f"N={N_anchor} -> N={N_target}"
        )

    if kount_ext is None or kount_ext < 2:
        raise RuntimeError(
            f"Extension integration produced too few points: "
            f"kount_ext={kount_ext}"
        )

#save path and e-folds
    path_ext = yp_ext[:, :kount_ext].copy()
    N_ext = xp_ext[:kount_ext].copy()

    return path_ext, N_ext, y_ext


#This function right now is not necessary for us to use since we will account for uncertainty in a diff way
#def get_state_at_N(y_end, N_start, N_target, derivs):
#    """
#    Given the state y_end at N_start (e.g. N=60, from calcpath),
#    integrate to N_target (e.g. N=55) and return the state there.
#    y_end of full state vector at starting N
#    N_start is where that full state lives i.e. 60
#    N_target is the Npivot in this case 55
#    derivs is the Hubble flow DE
#    """
#    s = odeiv.step_rk4(NEQS, derivs) #will create numerical integration methods
#    c = odeiv.control_y_new(s, 1e-10, 1e-10) #num tolerances
#    e = odeiv.evolve(s, c, NEQS) #evolution of object, e advnaces state in N
#
#    #For the code I have yend is phi, H, eps, sig, 2lam, 3lam, 4lam, 5lam all at N=60
#    ydoub = y_end.copy() #copy of starting state at nstart
#    N = N_start
#    h = 0.01 #initial step
#    if N_target < N_start: #Need to integrate to smaller in so travel down
#        h = -h
#
#    while abs(N - N_target) > 1e-8: #keep integrating while N start is greater than Ntarget
#        N, h, ydoub = e.apply(N, N_target, h, ydoub) #returns an updated N
#
#    return ydoub #returns evolved state at desired N
#    
#    

def run_neqs7_lam4_models_batch(clean_output=True):
    import shutil

    TARGET_ACCEPTED = 50
    MAX_TRIALS = 20000000

    summary_records = []

    if clean_output and os.path.exists(BASE_OUTDIR):
        print(f"Removing old output directory:\n{BASE_OUTDIR}")
        shutil.rmtree(BASE_OUTDIR)

    os.makedirs(BASE_OUTDIR, exist_ok=True)

    accepted_count = 0
    trial_count = 0

    rejected_asymptote = 0
    rejected_bad_r = 0
    rejected_bad_ns = 0
    rejected_bad_alpha_s = 0
    rejected_other = 0
    spectrum_error_count = 0
    duplicate_dir_count = 0

    while accepted_count < TARGET_ACCEPTED and trial_count < MAX_TRIALS:
        trial_count += 1
        
        
        if trial_count % 2500 == 0:
#             clear_output(wait=True)

            print(f"[progress] trial={trial_count}")
            print(f"accepted={accepted_count}/{TARGET_ACCEPTED}")

## BELOW I AM ADDING ALL TO BE RANDOM. ABOVE IT WAS JUST THE LAST 3.
#        if trial_count == 1 and BATCH_ID == 1:
#            eps = EPSILON_BASE
#            sig = SIGMA_BASE
#            lam2 = LAM2_BASE
#            lam3 = LAM3_BASE
#            lam4 = LAM4_BASE
#            lam5 = LAM5_BASE
#        else:
#            eps = rng.uniform(EPSILON_MIN, EPSILON_MAX)
#            sig = rng.uniform(SIGMA_MIN, SIGMA_MAX)
#            lam2 = rng.uniform(LAM2_MIN, LAM2_MAX)
#            lam3 = rng.uniform(LAM3_MIN, LAM3_MAX)
#            lam4 = rng.uniform(LAM4_MIN, LAM4_MAX)
#            lam5 = rng.uniform(LAM5_MIN, LAM5_MAX)
         
# Now if i just wanna test out lamX variation
        if trial_count == 1 and BATCH_ID == 1:
            eps = EPSILON_BASE
            sig = SIGMA_BASE
            lam2 = LAM2_BASE
            lam3 = LAM3_BASE
            lam4 = LAM4_BASE
#            lam5 = LAM5_BASE
        else:
            eps = rng.uniform(EPSILON_MIN, EPSILON_MAX)
            sig = rng.uniform(SIGMA_MIN, SIGMA_MAX)
            lam2 = rng.uniform(LAM2_MIN,LAM2_MAX)
            lam3 = rng.uniform(LAM3_MIN, LAM3_MAX)
            lam4 = rng.uniform(LAM4_MIN, LAM4_MAX)
#            lam5 = rng.uniform(LAM5_MIN, LAM5_MAX)
           


#For reproducible batch runs:
#            eps = np.random.uniform(EPSILON_MIN, EPSILON_MAX)
#            sig = np.random.uniform(SIGMA_MIN, SIGMA_MAX)
#            lam2 = np.random.uniform(LAM2_MIN, LAM2_MAX)
#            lam3 = np.random.uniform(LAM3_MIN, LAM3_MAX)
#            lam4 = np.random.uniform(LAM4_MIN, LAM4_MAX)
#            lam5 = np.random.uniform(LAM5_MIN, LAM5_MAX)

        # Progress printout
        if trial_count % PRINT_EVERY == 0:
            print(
                f"[progress] trial={trial_count} "
                f"accepted={accepted_count}/{TARGET_ACCEPTED}"
            )

        # Detailed printout only for first trial or verbose mode
        if VERBOSE or (PRINT_FIRST_TRIAL and trial_count == 1):

            print("\n" + "=" * 70)
            print(f"Trial {trial_count}")
            print(f"Accepted {accepted_count}/{TARGET_ACCEPTED}")
            
            print(f"ε = {eps:.10e}")
            print(f"σ = {sig:.10e}")
            print(f"λ2 = {lam2:.10e}")
            print(f"λ3 = {lam3:.10e}")
            print(f"λ4 = {lam4:.10e}")
#            print(f"λ5 = {lam5:.10e}")

        calc = Calc()

#         yinit, calc.Nefolds = pick_init_vals(lam5)
        yinit, calc.Nefolds = pick_init_vals(eps, sig, lam2, lam3, lam4)
        y = yinit.copy()

        path = np.array([[]])
        N = np.array([])

        t0 = time.perf_counter()
        calc.ret = calcpath(calc.Nefolds, y, path, N, calc)
        t1 = time.perf_counter()

        if VERBOSE or (PRINT_FIRST_TRIAL and trial_count == 1):
            print(f"calcpath runtime: {t1 - t0:.4f} s")
            print(f"calc.ret = {calc.ret}")


        if calc.ret == "asymptote":
            rejected_asymptote += 1
            if PRINT_REJECTIONS:
                print("REJECTED: asymptote")
            continue

        if calc.ret != "nontrivial":
            rejected_other += 1
            if PRINT_REJECTIONS:
                print(f"REJECTED: {calc.ret}")
            continue

        ## This was an original look into the parameters
        ##These cuts correspond to N=60, but that happens at a k scale of 3.2e-4
        r = tsratio(y)
        ns = specindex(y)
        alpha_s = dspecindex(y)

        # This is what I had before, where it found the model at a specific N state and saved those models only
#        Within our chosen r, ns and alphas range
        #If I want to to find models whose observables correspond with k=0.05 hMpc^-1
#        N_CMB = 60.0
#
#        y_cmb = get_state_at_N(y, calc.Nefolds, N_CMB, derivs)
#        #y_cmb[2] is eps at N=55, y_cmb[3] is sig at N=55)
#
#        r = tsratio(y_cmb)
#        ns = specindex(y_cmb)
#        alpha_s = dspecindex(y_cmb)


        if VERBOSE or (PRINT_FIRST_TRIAL and trial_count == 1):
            
            print("Candidate observables:")
            print(f"  r       = {r:.10e}")
            print(f"  ns      = {ns:.10f}")
#            print(f"  alpha_s = {alpha_s:.10e}")
            
    
        if not (NMIN < ns < NMAX):
            rejected_bad_ns += 1
            if PRINT_REJECTIONS:
                print(f"REJECTED: ns={ns:.10f} outside ({NMIN}, {NMAX})")
            continue
            
        if not (RMIN < r < RMAX):
            rejected_bad_r += 1
            if PRINT_REJECTIONS:
                print(f"REJECTED: r={r:.10e} outside ({RMIN:.3e}, {RMAX:.3e})")
            continue

#        if not (ALPHA_MIN < alpha_s < ALPHA_MAX):
#            rejected_bad_alpha_s += 1
#            if PRINT_REJECTIONS:
#                print(f"REJECTED: alpha_s={alpha_s:.10e} outside ({ALPHA_MIN:.3e}, {ALPHA_MAX:.3e})")
#            continue


##Okay here we will extend the trajectory such that, we begin with trajectory from N=60 to N=0
## and only bother to extend models in coarse CMB space

        # ========================================================
        # EXTEND ACCEPTED N=0 -> 60 TRAJECTORY BACK TO N=70
        #
        # IMPORTANT:
        # y is still the state at N=60 returned by calcpath.
        # We copy it so the spectrum calculation below continues
        # to see exactly the same N=60 state as before.
        # ========================================================

        y_N60 = y.copy()

        try:
            path_60_to_70, N_60_to_70, y_N70 = extend_path_to_larger_N(
                y_at_anchor=y_N60,
                N_anchor=N_ANCHOR,
                N_target=N_EXTEND,
                kmax_ext=EXTENSION_KMAX,
            )

        except RuntimeError as exc:
            rejected_other += 1

            print(
                "REJECTED: could not construct "
                f"N={N_ANCHOR:g} -> N={N_EXTEND:g} extension"
            )
            print(exc)

            continue

        # First-run diagnostics
        if VERBOSE or (PRINT_FIRST_TRIAL and trial_count == 1):

            print("\n" + "-" * 70)
            print("BACKWARD PATH EXTENSION CHECK")

            print(
                f"Original path N range: "
                f"{N[0]:.10f} -> {N[-1]:.10f}"
            )

            print(
                f"Extension N range: "
                f"{N_60_to_70[0]:.10f} -> "
                f"{N_60_to_70[-1]:.10f}"
            )

#            print("\nN=60 state:")
#            print(f"  epsilon = {y_N60[2]:.10e}")
#            print(f"  sigma   = {y_N60[3]:.10e}")
#            print(f"  lambda2 = {y_N60[4]:.10e}")
#            print(f"  lambda3 = {y_N60[5]:.10e}")
#
#            print("\nN=70 state:")
#            print(f"  epsilon = {y_N70[2]:.10e}")
#            print(f"  sigma   = {y_N70[3]:.10e}")
#            print(f"  lambda2 = {y_N70[4]:.10e}")
#            print(f"  lambda3 = {y_N70[5]:.10e}")

            print("-" * 70)
            
            
        accepted_count += 1
        ## To clarify we have three separate objects now
        
        #y_N60          # physical anchor / sampled model
        #y_N70          # extrapolated earlier state
        #path_60_to_70  # extra truncation-systematics trajectory


        print(
            f"\n[ACCEPTED {accepted_count}/{TARGET_ACCEPTED}] "
            f"trial={trial_count} | "
            f"r={r:.3e} |"
            f"ns={ns:.6f} | "
#            f"alpha_s={alpha_s:.3e} |"
            f"ε = {eps:.10e} |"
            f"σ = {sig:.10e} |"
            f"λ2 = {lam2:.10e} |"
            f"λ3 = {lam3:.10e} |"
            f"λ4 = {lam4:.10e} |"
#            f"λ5 = {lam5:.10e} |"
        )

        if VERBOSE:
            print(f"ε = {eps:.10e}")
            print(f"σ = {sig:.10e}")
            print(f"λ2 = {lam2:.10e}")
            print(f"λ3 = {lam3:.10e}")
            print(f"λ4 = {lam4:.10e}")
#            print(f"λ5 = {lam5:.10e}")
            
    

    #Save name here
#        OUTDIR = f"{BASE_OUTDIR}/lam5_{lam5:.10e}"
#         OUTDIR = f"{BASE_OUTDIR}/lam4_{lam4:.10e}_lam5_{lam5:.10e}" #I could change OUTDIR for clarity sake
#    
#        if os.path.exists(OUTDIR):
#            duplicate_dir_count += 1
#            OUTDIR = f"{BASE_OUTDIR}/lam5_{lam5:.10e}_trial_{trial_count:06d}"

        # Change this if running NEQs = 7 or 8
#        lam4 = 0
        lam5 = 0
        
        model_name = (
            f"lam2_{lam2:.10e}_"
            f"lam3_{lam3:.10e}_"
            f"lam4_{lam4:.10e}_"
            f"lam5_{lam5:.10e}"
        )

        OUTDIR = f"{BASE_OUTDIR}/{model_name}"

        if os.path.exists(OUTDIR):
            duplicate_dir_count += 1
            OUTDIR = f"{BASE_OUTDIR}/{model_name}_trial_{trial_count:06d}"
            
        os.makedirs(OUTDIR, exist_ok=False)

        OUTFILE1_NAME = f"{OUTDIR}/test_nr_neqs{NEQS}.dat"
        OUTFILE2_NAME = f"{OUTDIR}/test_esigma_neqs{NEQS}.dat"

        with open(OUTFILE1_NAME, "w") as outfile1:
#            outfile1.write(f"{r:.10f} {ns:.10f}\n")
            outfile1.write(f"{r:.10f} {ns:.10f} {alpha_s:.10f}\n")

        with open(OUTFILE2_NAME, "w") as outfile2:
            for i in range(NEQS):
                outfile2.write("%le " % y[i])
            outfile2.write("%f\n" % calc.Nefolds)

        if SPECTRUM:
            u_s = np.empty((2, knos))
            u_t = np.empty((2, knos))
            y_final = np.empty(NEQS + 1)

            if calc.npoints <= 3:
                print("WARNING: not enough path points for spectrum. Skipping spectrum.")
            else:
                y_final[:NEQS] = path[:NEQS, 3]
                y_final[NEQS] = N[3]

                if VERBOSE:
                    print("Evaluating spectrum for accepted model...")

                t0 = time.perf_counter()
                spectrum_status = spectrum(
                    y_final,
                    y,
                    u_s,
                    u_t,
                    calc.Nefolds,
                    derivs1,
                    scalarsys,
                    tensorsys
                )
                t1 = time.perf_counter()

                if VERBOSE:
                    print(f"spectrum runtime: {t1 - t0:.4f} s")
    
                if spectrum_status:
                    spectrum_error_count += 1
                    print("WARNING: spectrum returned an error/status flag.")

                np.savetxt(f"{OUTDIR}/spec_s_neqs{NEQS}.dat", u_s[:, :knos].T)
                np.savetxt(f"{OUTDIR}/spec_t_neqs{NEQS}.dat", u_t[:, :knos].T)

        if SPECTRUM:
            if VERBOSE:
                print(f"Before path normalization: y[1] = {y[1]:.6e}")
    
##We may need to adjust the saved path to include that 70 now. Original is below:
#            for j in range(calc.npoints):
#                path[0, j] = path[0, j] - path[0, calc.npoints - 1]
#                path[1, j] = path[1, j] * y[1]
#
#            if VERBOSE:
#                print(f"After path normalization: max(path[1,:]) = {np.max(path[1,:]):.6e}")
#
##        path_name = f"{OUTDIR}/path_neqs{NEQS}_lam5_{lam5:.10e}.dat"
#        path_name = f"{OUTDIR}/path_neqs{NEQS}_{model_name}.dat"
#        save_path(path, N, calc.npoints, path_name)



##If we want the whole N=70 down to N=0 path
        # ========================================================
        # BUILD COMPLETE N=0 -> 70 PATH
        # ========================================================

        # Work on copies. Keep the original calcpath arrays untouched
        # until the spectrum calculation above has completely finished.
        path_main = path[:, :calc.npoints].copy()
        N_main = N[:calc.npoints].copy()

        path_ext = path_60_to_70.copy()
        N_ext = N_60_to_70.copy()

        # --------------------------------------------------------
        # Remove any duplicated N=60 endpoint.
        #
        # Do this generically rather than assuming int_de includes
        # or excludes the exact endpoint.
        # --------------------------------------------------------


        N_combined = np.concatenate([N_main,N_ext[1:]])

        path_combined = np.concatenate([path_main,path_ext[:, 1:]], axis=1)


        tol = 1e-8

        print("N_main end =", N_main[-1])
        print("N_ext start =", N_ext[0])
        print("N_ext end =", N_ext[-1])

        if not np.all(np.diff(N_main) > 0):
            raise RuntimeError("N_main is not monotonically increasing.")

        if not np.all(np.diff(N_ext) > 0):
            raise RuntimeError("N_ext is not monotonically increasing.")

        if abs(N_ext[0] - N_main[-1]) < tol:
            # Same N=60 anchor appears in both pieces
            N_combined = np.concatenate([
                N_main,
                N_ext[1:]
            ])

            path_combined = np.concatenate([
                path_main,
                path_ext[:, 1:]
            ], axis=1)

        else:
            # No duplicate endpoint -- preserve both
            N_combined = np.concatenate([
                N_main,
                N_ext
            ])

            path_combined = np.concatenate([
                path_main,
                path_ext
            ], axis=1)
                
            
        
        # ========================================================
        # APPLY THE SAME NORMALIZATION TO THE ENTIRE 0 -> 70 PATH
        # ========================================================

        if SPECTRUM:

            if VERBOSE:
                print(
                    f"Before extended-path normalization: "
                    f"y[1] = {y[1]:.6e}"
                )

            # IMPORTANT:
            # Use ONE phi reference for both pieces of the path.
            #
            # The ordinary path contains the end-of-inflation point,
            # so use its phi there as the zero point.
            phi_end_reference = path_main[0, -1]

            path_combined[0, :] -= phi_end_reference

            # spectrum() has already replaced y[1] with the H
            # normalization factor, just as in your original runner.
            path_combined[1, :] *= y[1]

            if VERBOSE:
                print(
                    "After extended-path normalization: "
                    f"max(H) = "
                    f"{np.max(path_combined[1, :]):.6e}"
                )

        # ========================================================
        # SANITY CHECK COMPLETE RANGE
        # ========================================================

        if VERBOSE or (PRINT_FIRST_TRIAL and accepted_count == 1):

            print("\nCOMPLETE SAVED PATH CHECK")
            print(
                f"  N min = {N_combined.min():.10f}"
            )
            print(
                f"  N max = {N_combined.max():.10f}"
            )
            print(
                f"  number of points = {len(N_combined)}"
            )

            print(
                f"  closest point to N=60: "
                f"{N_combined[np.argmin(np.abs(N_combined - 60.0))]:.10f}"
            )

            print(
                f"  closest point to N=70: "
                f"{N_combined[np.argmin(np.abs(N_combined - 70.0))]:.10f}"
            )

        # ========================================================
        # SAVE FULL N=0 -> 70 PATH
        # ========================================================

        path_name = (
            f"{OUTDIR}/"
            f"path_neqs{NEQS}_N0to70_paramsAtN60_{model_name}.dat"
        )

        save_path(
            path_combined,
            N_combined,
            len(N_combined),
            path_name
        )
        
        
## To about here
        
#         print("\nDEBUG original calcpath end:")
#         print("  original_end_index      =", getattr(calc, "original_end_index", None))
#         print("  original_N_end          =", getattr(calc, "original_N_end", None))
#         print("  original_N_before_end   =", getattr(calc, "original_N_before_end", None))
#         print("  original_N_after_end    =", getattr(calc, "original_N_after_end", None))
#         print("  original_eps_end        =", getattr(calc, "original_eps_end", None))
#         print("  spectrum_N_start N[3]   =", N[3])
#         print("  path_N_end N[-1]        =", N[calc.npoints-1])

        
    #With native N path
        summary_records.append({
            "accepted_index": accepted_count,
            "trial_index": trial_count,

            "eps": eps,
            "sig": sig,
            "lam2": lam2,
            "lam3": lam3,
            "lam4": lam4,
#            "lam5": lam5,

            "r": r,
            "n_s": ns,
            "alpha_s": alpha_s,
            "Nefolds": calc.Nefolds,

            "original_end_index": getattr(calc, "original_end_index", np.nan),
            "original_N_end": getattr(calc, "original_N_end", np.nan),
            "original_N_before_end": getattr(calc, "original_N_before_end", np.nan),
            "original_N_after_end": getattr(calc, "original_N_after_end", np.nan),
            "original_eps_end": getattr(calc, "original_eps_end", np.nan),
            
            
            # Sampled / anchored state at N=60
            "eps_N60": y_N60[2],
            "sig_N60": y_N60[3],
            "lam2_N60": y_N60[4],
            "lam3_N60": y_N60[5],

            # Same truncated hierarchy evolved backward to N=70
            "eps_N70": y_N70[2],
            "sig_N70": y_N70[3],
            "lam2_N70": y_N70[4],
            "lam3_N70": y_N70[5],

            "extended_N_min": np.min(N_combined),
            "extended_N_max": np.max(N_combined),

            "spectrum_N_start": N[3],
#            "path_N_end": N[calc.npoints - 1],
            "original_path_N_end": N[calc.npoints - 1],
            "extended_path_N_max": np.max(N_combined),

            "calc_ret": calc.ret,
            "outdir": OUTDIR
        })
        

        summary_df = pd.DataFrame(summary_records)
        summary_file = f"{BASE_OUTDIR}/neqs{NEQS}_summary.csv"
        summary_df.to_csv(summary_file, index=False)


    print("\n" + "=" * 70)
    print("DONE")
    print(f"Accepted viable nontrivial models: {accepted_count}")
    print(f"Total trials: {trial_count}")
    print(f"Rejected asymptotes: {rejected_asymptote}")
    print(f"Rejected bad r: {rejected_bad_r}")
    print(f"Rejected bad ns: {rejected_bad_ns}")
    print(f"Rejected bad alpha_s: {rejected_bad_alpha_s}")
    print(f"Rejected other: {rejected_other}")
    print(f"Spectrum error count: {spectrum_error_count}")
    print(f"Duplicate directory count: {duplicate_dir_count}")
    print(f"Summary written to:\n{summary_file}")

    if accepted_count < TARGET_ACCEPTED:
        print(
            f"WARNING: only found {accepted_count}/{TARGET_ACCEPTED} accepted models "
            f"before hitting MAX_TRIALS={MAX_TRIALS}."
        )
        
        
# %time run_neqs8_lam3_lam4_lam5_models()
if __name__ == "__main__":
    t0 = time.perf_counter()
    run_neqs7_lam4_models_batch()
    t1 = time.perf_counter()
    print(f"Total runtime: {(t1 - t0)/60:.2f} min")
    
    
## For higgs:
## [ACCEPTED 1/1] trial=1| ns=0.969485 | r=2.737e-03 |λ3 = -4.6097100000e-06 |λ4 = 6.8706500000e-08 |λ5 = -8.9246100000e-09 |

##For non-higgs:
##[ACCEPTED 1/1] trial=657 | ns=0.960824 | r=3.296e-02 |λ3 = -8.6231791933e-04 |λ4 = -3.4851399219e-04 |λ5 = -3.3792701497e-05 |


