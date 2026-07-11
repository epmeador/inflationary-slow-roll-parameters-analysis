# Core imports man
import sys
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import time


# Numerical libraries
from scipy.integrate import solve_ivp, odeint, cumulative_trapezoid
from scipy.interpolate import (
    UnivariateSpline, splrep, splev, CubicSpline,
    interp1d, PchipInterpolator, InterpolatedUnivariateSpline
)

import numdifftools as nd

# Random number generator (GSL)
import pygsl.rng

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
NEQS = 8
BATCH_ID = 5   # will need to change this as I go: 1, 2, 3, 4, 5

# SPECTRUM = True
# SPECTRUM = False


SAVEPATHS = True


### For Generic:
#
NMAX = 0.98
NMIN = 0.96

RMIN = 0
RMAX = 0.04

#ALPHA_TARGET =
ALPHA_MIN = -0.005
ALPHA_MAX = 0.005


## For Higgs:
## ns = 0.969485
## r = 2.737e-03
#
#NMAX = 0.97
#NMIN = 0.96
#
#R_TARGET = 2.737e-03 #0.00335
#R_DELTA = 0.001
#
#RMIN = R_TARGET - R_DELTA
#RMAX = R_TARGET + R_DELTA
#
##ALPHA_TARGET =
#ALPHA_MIN = -0.005
#ALPHA_MAX = 0.005

# #For higgs beforehand:
# #RMAX = 3e-3
# #RMIN = 2e-3

# ## For non_Higgs:
# # ns = 0.9608243182
# # r = 3.2958886892e-02

# NMAX = 0.97
# NMIN = 0.96

# R_TARGET = 3.2958886892e-02
# R_DELTA = 0.001

# RMIN = R_TARGET - R_DELTA
# RMAX = R_TARGET + R_DELTA

#For non-higgs beforehand:
#RMAX = 4e-2
#RMIN = 3e-2


#Test of Interest To Us

# ##For Higgs
EPSILON_BASE = 0.000209237
SIGMA_BASE = -0.0342419
LAM2_BASE = 0.000278972
LAM3_BASE = -4.60971e-6
LAM4_BASE = 6.87065e-08
LAM5_BASE = -8.92461e-9


##For non-Higgs
#Exact quad deformed
# LAM3_BASE = 0.000190155
# LAM4_BASE = 0.0337111
# LAM5_BASE = -0.000378739

# # #Better true trajectory
# LAM3_BASE = -8.6231791933e-04
# LAM4_BASE = -3.4851399219e-04
# LAM5_BASE = -3.3792701497e-05


#Kinney Range:
#For neqs8 or lam5 variation I want to randomly pick in this lam5 range
LAM5_MIN = -5e-5
LAM5_MAX = 5e-5

LAM4_MIN = -5e-4
LAM4_MAX = 5e-4

LAM3_MIN = -5e-3
LAM3_MAX = 5e-3

#LAM2_MIN = -5e-2
#LAM2_MAX = 5e-2
#
#SIGMA_MIN = -5e-1
#SIGMA_MAX = 5e-1
#
#EPSILON_MIN = 0.0
#EPSILON_MAX = 0.005


##cmb priors: conservative
LAM2_MIN = -5e-3
LAM2_MAX = 5e-3
#
#LAM2_MIN = -5e-2
#LAM2_MAX = 5e-2

SIGMA_MIN = NMIN - 1
SIGMA_MAX = NMAX - 1

EPSILON_MIN = RMIN/16
EPSILON_MAX = RMAX/16


##cmb priors: less conservative
#LAM2_MIN = -5e-3
#LAM2_MAX = 5e-3
#
#SIGMA_MIN = (NMIN - 1)-0.01
#SIGMA_MAX = (NMAX - 1)+0.01
#
#EPSILON_MIN = 0.5*(RMIN/16)
#EPSILON_MAX = 1.5*(RMAX/16)


NUM_LAM_GRID = 100

# lam5_set = np.linspace(LAM5_MIN, LAM5_MAX, num=NUM_LAM5_GRID)
lam5_set = np.random.uniform(LAM5_MIN, LAM5_MAX, size=NUM_LAM_GRID) #Can cycle through randomly
lam5_set = np.sort(np.append(lam5_set, [LAM5_BASE]))
# # lam5_set = np.unique(lam5_set)


lam4_set = np.random.uniform(LAM4_MIN, LAM4_MAX, size=NUM_LAM_GRID) #Can cycle through randomly
lam4_set = np.sort(np.append(lam4_set, [LAM4_BASE]))


lam3_set = np.random.uniform(LAM3_MIN, LAM3_MAX, size=NUM_LAM_GRID) #Can cycle through randomly
lam3_set = np.sort(np.append(lam3_set, [LAM3_BASE]))


lam2_set = np.random.uniform(LAM2_MIN, LAM2_MAX, size=NUM_LAM_GRID) #Can cycle through randomly
lam2_set = np.sort(np.append(lam2_set, [LAM2_BASE]))


sigma_set = np.random.uniform(SIGMA_MIN, SIGMA_MAX, size=NUM_LAM_GRID) #Can cycle through randomly
sigma_set = np.sort(np.append(sigma_set, [SIGMA_BASE]))


epsilon_set = np.random.uniform(EPSILON_MIN, EPSILON_MAX, size=NUM_LAM_GRID) #Can cycle through randomly
epsilon_set = np.sort(np.append(epsilon_set, [EPSILON_BASE]))


#Short Range for Testing Convergence Order
#Set below is good for testing smallest number needs to be to match
# # lam5_set = [-8.92461e-09, 0, -8.92461e-08, -8.92461e-07, -8.92461e-06, -8.92461e-05, -8.92461e-10, -8.92461e-11, -8.92461e-12, -8.92461e-13, -8.92461e-14, -8.92461e-15, -8.92461e-16, -8.92461e-17, -8.92461e-18, -8.92461e-19, -8.92461e-20, -8.92461e-23, -8.92461e-26, -8.92461e-29, -8.92461e-32]


# derivs1 = derivs ##added this for now
# print(f"Total models: {len(lam5_set)}")  # Should be 250 (or between 1-250)
# print(f"Base λ5={LAM5_BASE:.6e} included: {LAM5_BASE in lam5_set}")

#This sets how many nontrivial models you are running (interesting models you want)
NUMPOINTS = 1        # number of nontrivial models per λ5

#This will give us the min and max number of e-folds we are looking for
NUMEFOLDSMAX = 65.0
NUMEFOLDSMIN = 57.0

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

np.random.seed(12345 + BATCH_ID)

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

def pick_init_vals(eps, sig, lam2, lam3, lam4, lam5):

    init_vals = np.zeros(NEQS, dtype=float, order='C')

    init_vals[0] = 5.5/np.sqrt(8*np.pi)
    init_vals[1] = 1.0
    init_vals[2] = eps
    init_vals[3] = sig
    init_vals[4] = lam2
    init_vals[5] = lam3
    init_vals[6] = lam4
    init_vals[7] = lam5

    init_Nefolds = 60
    return init_vals, init_Nefolds


## For Higgs
#def pick_init_vals(lam3, lam4, lam5):
#
#    init_vals = np.zeros(NEQS, dtype=float, order='C')
#
#    init_vals[0] = 5.5/np.sqrt(8*np.pi)
#    init_vals[1] = 1.0
#    init_vals[2] = 0.000209237
#    init_vals[3] = -0.0342419
#    init_vals[4] = 0.000278972
#    init_vals[5] = lam3
#    init_vals[6] = lam4
#    init_vals[7] = lam5
#
#    init_Nefolds = 60
#    return init_vals, init_Nefolds

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


# if we include ns, r, and alpha
def we_should_calc_spec(y):
    ns = specindex(y)
    r = tsratio(y)
    alpha_s = dspecindex(y)

    return (
        (NMIN < ns < NMAX)
        and (RMIN < r < RMAX)
        and (ALPHA_MIN < alpha_s < ALPHA_MAX)
    )

#Specifically, we will save a model with some interesting dynamics
#And this means either asymptote or nontrivial - for us nontrivial
#Also only if the path has not been saved yet, and only for
#Every nth successful model
#This governs path saving in the non-spectral case

def we_should_save_path(retval, save, pointcount, printevery):
    return (retval == "nontrivial") and (not save) and (pointcount % printevery == 0)

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


def run_neqs8_lam3_lam4_lam5_models_batch(clean_output=True):
    import shutil

    TARGET_ACCEPTED = 40
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
            
#        if trial_count == 1:
#            lam3 = LAM3_BASE
#            lam4 = LAM4_BASE
#            lam5 = LAM5_BASE
#        else:
#            lam3 = np.random.uniform(LAM3_MIN, LAM3_MAX)
#            lam4 = np.random.uniform(LAM4_MIN, LAM4_MAX)
#            lam5 = np.random.uniform(LAM5_MIN, LAM5_MAX)
#            
#
## BELOW I AM ADDING ALL TO BE RANDOM. ABOVE IT WAS JUST THE LAST 3.
        if trial_count == 1 and BATCH_ID == 1:
            eps = EPSILON_BASE
            sig = SIGMA_BASE
            lam2 = LAM2_BASE
            lam3 = LAM3_BASE
            lam4 = LAM4_BASE
            lam5 = LAM5_BASE
        else:
            eps = np.random.uniform(EPSILON_MIN, EPSILON_MAX)
            sig = np.random.uniform(SIGMA_MIN, SIGMA_MAX)
            lam2 = np.random.uniform(LAM2_MIN, LAM2_MAX)
            lam3 = np.random.uniform(LAM3_MIN, LAM3_MAX)
            lam4 = np.random.uniform(LAM4_MIN, LAM4_MAX)
            lam5 = np.random.uniform(LAM5_MIN, LAM5_MAX)

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
            print(f"λ5 = {lam5:.10e}")

        calc = Calc()

#         yinit, calc.Nefolds = pick_init_vals(lam5)
        yinit, calc.Nefolds = pick_init_vals(eps, sig, lam2, lam3, lam4, lam5)
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

        r = tsratio(y)
        ns = specindex(y)
        alpha_s = dspecindex(y)

        
        if VERBOSE or (PRINT_FIRST_TRIAL and trial_count == 1):
            
            print("Candidate observables:")
            print(f"  r       = {r:.10e}")
            print(f"  ns      = {ns:.10f}")
            print(f"  alpha_s = {alpha_s:.10e}")
            
    
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


        if not (ALPHA_MIN < alpha_s < ALPHA_MAX):
            rejected_bad_alpha_s += 1
            if PRINT_REJECTIONS:
                print(f"REJECTED: alpha_s={alpha_s:.10e} outside ({ALPHA_MIN:.3e}, {ALPHA_MAX:.3e})")
            continue
            
            
        accepted_count += 1
        
        print(
            f"\n[ACCEPTED {accepted_count}/{TARGET_ACCEPTED}] "
            f"trial={trial_count} | "
            f"r={r:.3e} |"
            f"ns={ns:.6f} | "
            f"alpha_s={alpha_s:.3e} |"
            f"ε = {eps:.10e} |"
            f"σ = {sig:.10e} |"
            f"λ2 = {lam2:.10e} |"
            f"λ3 = {lam3:.10e} |"
            f"λ4 = {lam4:.10e} |"
            f"λ5 = {lam5:.10e} |"
        )

        if VERBOSE:
            print(f"ε = {eps:.10e}")
            print(f"σ = {sig:.10e}")
            print(f"λ2 = {lam2:.10e}")
            print(f"λ3 = {lam3:.10e}")
            print(f"λ4 = {lam4:.10e}")
            print(f"λ5 = {lam5:.10e}")
            
        ### When it was just last random3
#        print(
#            f"\n[ACCEPTED {accepted_count}/{TARGET_ACCEPTED}] "
#            f"trial={trial_count} | "
#            f"r={r:.3e} |"
#            f"ns={ns:.6f} | "
#            f"alpha_s={alpha_s:.3e} |"
#            f"λ3 = {lam3:.10e} |"
#            f"λ4 = {lam4:.10e} |"
#            f"λ5 = {lam5:.10e} |"
#        )
#
#        if VERBOSE:
#            print(f"λ3 = {lam3:.10e}")
#            print(f"λ4 = {lam4:.10e}")
#            print(f"λ5 = {lam5:.10e}")
#

    
        OUTDIR = f"{BASE_OUTDIR}/lam5_{lam5:.10e}"
#         OUTDIR = f"{BASE_OUTDIR}/lam4_{lam4:.10e}_lam5_{lam5:.10e}" #I could change OUTDIR for clarity sake

        if os.path.exists(OUTDIR):
            duplicate_dir_count += 1
            OUTDIR = f"{BASE_OUTDIR}/lam5_{lam5:.10e}_trial_{trial_count:06d}"

        os.makedirs(OUTDIR, exist_ok=False)

        OUTFILE1_NAME = f"{OUTDIR}/test_nr_neqs{NEQS}.dat"
        OUTFILE2_NAME = f"{OUTDIR}/test_esigma_neqs{NEQS}.dat"

        with open(OUTFILE1_NAME, "w") as outfile1:
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
    

            for j in range(calc.npoints):
                path[0, j] = path[0, j] - path[0, calc.npoints - 1]
                path[1, j] = path[1, j] * y[1]

            if VERBOSE:
                print(f"After path normalization: max(path[1,:]) = {np.max(path[1,:]):.6e}")

        path_name = f"{OUTDIR}/path_neqs{NEQS}_lam5_{lam5:.10e}.dat"
        save_path(path, N, calc.npoints, path_name)
        
#         print("\nDEBUG original calcpath end:")
#         print("  original_end_index      =", getattr(calc, "original_end_index", None))
#         print("  original_N_end          =", getattr(calc, "original_N_end", None))
#         print("  original_N_before_end   =", getattr(calc, "original_N_before_end", None))
#         print("  original_N_after_end    =", getattr(calc, "original_N_after_end", None))
#         print("  original_eps_end        =", getattr(calc, "original_eps_end", None))
#         print("  spectrum_N_start N[3]   =", N[3])
#         print("  path_N_end N[-1]        =", N[calc.npoints-1])

        #Without native N path
#         summary_records.append({
#             "accepted_index": accepted_count,
#             "trial_index": trial_count,
#             "lam5": lam5,
#             "r": r,
#             "n_s": ns,
#             "alpha_s": alpha_s,
#             "Nefolds": calc.Nefolds,
#             "calc_ret": calc.ret,
#             "outdir": OUTDIR
#         })
        
    #With native N path
        summary_records.append({
            "accepted_index": accepted_count,
            "trial_index": trial_count,

            "eps": eps,
            "sig": sig,
            "lam2": lam2,
            "lam3": lam3,
            "lam4": lam4,
            "lam5": lam5,

            "r": r,
            "n_s": ns,
            "alpha_s": alpha_s,
            "Nefolds": calc.Nefolds,

            "original_end_index": getattr(calc, "original_end_index", np.nan),
            "original_N_end": getattr(calc, "original_N_end", np.nan),
            "original_N_before_end": getattr(calc, "original_N_before_end", np.nan),
            "original_N_after_end": getattr(calc, "original_N_after_end", np.nan),
            "original_eps_end": getattr(calc, "original_eps_end", np.nan),

            "spectrum_N_start": N[3],
            "path_N_end": N[calc.npoints - 1],

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
    run_neqs8_lam3_lam4_lam5_models_batch()
    t1 = time.perf_counter()
    print(f"Total runtime: {(t1 - t0)/60:.2f} min")
    
    

## For higgs:
## [ACCEPTED 1/1] trial=1| ns=0.969485 | r=2.737e-03 |λ3 = -4.6097100000e-06 |λ4 = 6.8706500000e-08 |λ5 = -8.9246100000e-09 |

##For non-higgs:
##[ACCEPTED 1/1] trial=657 | ns=0.960824 | r=3.296e-02 |λ3 = -8.6231791933e-04 |λ4 = -3.4851399219e-04 |λ5 = -3.3792701497e-05 |


