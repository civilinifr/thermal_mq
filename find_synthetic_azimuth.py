"""
Locates thermal moonquakes using stochastic gradient descent for a synthetic event.

"""

# Import packages
import glob
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import os
import math
from joblib import Parallel, delayed


def preprocess(data, geo_locs, avg_velocity=45.0):
    """
    Preprocesses the input panda array of data. Produces:
    - Distances between each geophone
    - Computes measure of average velocity to use for later processing
    - Creates a new dataframe using relative times for each station

    :param data: [pd df] Pandas dataframe of the arrival times for each geophone
    :param geo_locs: [list] Geophone relative locations in m
    :param avg_velocity: [float] Average velocity for the regolith.
    :return:
    """
    # For right now we are not going to compute a velocity
    # We use the group velocity measurement from Larose et al. [2005] (45 m/s)
    rel_time = data['new_relative_time'].values
    new_rel_time = data['new_relative_time'].values - np.min(rel_time)

    # We need to fix the algorithm to be more accurate, as it's really just a matter of seconds.
    # For the time being, we will use a relative time used in Nick Schmerr's code
    new_rel_time = [0.1, 1.9, 0.8, 1.1]

    return new_rel_time, avg_velocity


def tt_misfit(ts, xs, ys, avg_velocity, geo_locs, new_rel_time):
    """
    Computes the total travel time misfit across all geophone stations

    :param ts: [float] Source time (param)
    :param xs: [float] Source x location (param)
    :param ys: [float] Source y location (param)
    :param avg_velocity: [float] Average velocity (m/s)
    :param geo_locs: [list] Geophone relative locations in m
    :param new_rel_time: [vector] Relative arrival time (ambiguous)
    :return:
    """
    # Initialize the vector
    sta_misfit = np.zeros(len(new_rel_time))

    for sta_ind in np.arange(len(new_rel_time)):

        # Setup the variables that we're going to need for easier viewing
        xi = geo_locs[sta_ind][0]
        yi = geo_locs[sta_ind][1]
        v = avg_velocity
        ti = new_rel_time[sta_ind]

        # Compute the misfit
        sta_misfit[sta_ind] = (ts + (np.sqrt(((xs-xi)**2)+((ys-yi)**2))/v) - ti)**2
    misfit = np.sum(sta_misfit)

    return misfit


def tt_misfit_geo3(ts, xs, ys, avg_velocity, geo_locs, t3):
    """
    Computes the total travel time misfit across only geophone 3

    :param ts: [float] Source time (param)
    :param xs: [float] Source x location (param)
    :param ys: [float] Source y location (param)
    :param avg_velocity: [float] Average velocity (m/s)
    :param geo_locs: [list] Geophone relative locations in m
    :param t3: [float] Relative arrival time for geophone 3
    :return:
    """
    # Compute the misfit
    xi = geo_locs[2][0]
    yi = geo_locs[2][1]
    ti = t3
    misfit = (ts + (np.sqrt(((xs-xi)**2)+((ys-yi)**2))/avg_velocity) - ti)**2

    return misfit


def tt_misfit_polar(ts, rs, avg_velocity, geo_locs, new_rel_time):
    """
    Computes the total travel time misfit relative to station Geophone 3 using polar coordinates

    :param ts: [float] Source time (param)
    :param rs: [float] Source distance location (in meters)
    :param avg_velocity: [float] Average velocity (m/s)
    :param geo_locs: [list] Geophone relative locations in m
    :param new_rel_time: [vector] Relative arrival time (ambiguous)
    :return:
    """
    # Isolate the geophone 3 time
    t3 = new_rel_time[2]

    # Compute the misfit
    misfit = (ts + (rs/avg_velocity)-t3)**2

    return misfit


def comp_dt_grad(ts, xs, ys, vs, geo_locs, new_rel_time):
    """
    Computes the values of the analytical gradient for dt

    :param ts: [float] Source time (param)
    :param xs: [float] Source x location (param)
    :param ys: [float] Source y location (param)
    :param avg_velocity: [float] Average velocity (m/s)
    :param geo_locs: [list] Geophone relative locations in m
    :param new_rel_time: [vector] Relative arrival time (ambiguous)
    :return:
    """

    # Initialize the vector
    sta_dt_grad = np.zeros(len(new_rel_time))

    for sta_ind in np.arange(len(new_rel_time)):
        # Setup the variables that we're going to need for easier viewing
        xi = geo_locs[sta_ind][0]
        yi = geo_locs[sta_ind][1]
        v = vs
        ti = new_rel_time[sta_ind]

        # Compute the misfit'
        sta_dt_grad[sta_ind] = (2*(ts + (np.sqrt(((xs - xi) ** 2) + ((ys - yi) ** 2)) / v) - ti))

    dt_grad = np.sum(sta_dt_grad)

    return dt_grad


def comp_dx_grad(ts, xs, ys, vs, geo_locs, new_rel_time):
    """
    Computes the values of the analytical gradient for dx

    :param ts: [float] Source time (param)
    :param xs: [float] Source x location (param)
    :param ys: [float] Source y location (param)
    :param vs: [float] Average velocity (m/s)
    :param geo_locs: [list] Geophone relative locations in m
    :param new_rel_time: [vector] Relative arrival time (ambiguous)
    :return:
    """

    # Initialize the vector
    sta_dx_grad = np.zeros(len(new_rel_time))

    for sta_ind in np.arange(len(new_rel_time)):
        # Setup the variables that we're going to need for easier viewing
        xi = geo_locs[sta_ind][0]
        yi = geo_locs[sta_ind][1]
        v = vs
        ti = new_rel_time[sta_ind]

        # Compute the misfit
        sta_dx_grad[sta_ind] = (2*(xs-xi)*(ts + (np.sqrt(((xs - xi) ** 2) + ((ys - yi) ** 2)) / v) - ti))/(v*(np.sqrt(((xs - xi) ** 2) + ((ys - yi) ** 2))))

    dx_grad = np.sum(sta_dx_grad)

    return dx_grad


def comp_dy_grad(ts, xs, ys, vs, geo_locs, new_rel_time):
    """
    Computes the values of the analytical gradient for dy

    :param ts: [float] Source time (param)
    :param xs: [float] Source x location (param)
    :param ys: [float] Source y location (param)
    :param vs: [float] Average velocity (m/s)
    :param geo_locs: [list] Geophone relative locations in m
    :param new_rel_time: [vector] Relative arrival time (ambiguous)
    :return:
    """

    # Initialize the vector
    sta_dy_grad = np.zeros(len(new_rel_time))

    for sta_ind in np.arange(len(new_rel_time)):
        # Setup the variables that we're going to need for easier viewing
        xi = geo_locs[sta_ind][0]
        yi = geo_locs[sta_ind][1]
        v = vs
        ti = new_rel_time[sta_ind]

        # Compute the misfit
        sta_dy_grad[sta_ind] = (2 * (ys - yi) * (ts + (np.sqrt(((xs - xi) ** 2) + ((ys - yi) ** 2)) / v) - ti)) / (
                    v * (np.sqrt(((xs - xi) ** 2) + ((ys - yi) ** 2))))

    dy_grad = np.sum(sta_dy_grad)

    return dy_grad


def comp_vs_grad(ts, xs, ys, vs, geo_locs, new_rel_time):
    """
    Computes the values of the analytical gradient for vs

    :param ts: [float] Source time (param)
    :param xs: [float] Source x location (param)
    :param ys: [float] Source y location (param)
    :param vs: [float] Average velocity (m/s)
    :param geo_locs: [list] Geophone relative locations in m
    :param new_rel_time: [vector] Relative arrival time (ambiguous)
    :return:
    """

    # Initialize the vector
    sta_vs_grad = np.zeros(len(new_rel_time))

    for sta_ind in np.arange(len(new_rel_time)):
        # Setup the variables that we're going to need for easier viewing
        xi = geo_locs[sta_ind][0]
        yi = geo_locs[sta_ind][1]
        v = vs
        ti = new_rel_time[sta_ind]

        # Compute the misfit
        sta_vs_grad[sta_ind] = (-2 * (ts + (np.sqrt(((xs - xi) ** 2) + ((ys - yi) ** 2)) / v) - ti))*\
                               ((np.sqrt(((xs - xi) ** 2) + ((ys - yi) ** 2)) / v)/(v**2))

    vs_grad = np.sum(sta_vs_grad)

    return vs_grad

def compute_dt2_grad_polar(ts, rs, avg_velocity, new_rel_time):

    """
    Computes the gradient of ts using polar coordinates.
    Note: we are only caring about geophone 3 in this instance.

    :param ts: [float] Source time
    :param rs: [float] Distance to source
    :param avg_velocity: [float] Average the velocity
    :param new_rel_time: [vector] Arrival times at all geophones
    :return:
    """
    # Isolate the geophone 3 time
    t3 = new_rel_time[2]

    # Solve for the gradient according to ts from the analytical expression
    dt2_grad = 2*(ts + (rs/avg_velocity) - t3)

    return dt2_grad


def compute_dr_grad_polar(ts, rs, avg_velocity, new_rel_time):

    """
    Computes the gradient of rs using polar coordinates.
    Note: we are only caring about geophone 3 in this instance.

    :param ts: [float] Source time
    :param rs: [float] Distance to source
    :param avg_velocity: [float] Average the velocity
    :param new_rel_time: [vector] Arrival times at all geophones
    :return:
    """
    # Isolate the geophone 3 time
    t3 = new_rel_time[2]

    # Solve for the gradient according to ts from the analytical expression
    dr_grad = (2*(avg_velocity*(ts - t3) + rs))/(avg_velocity**2)

    return dr_grad


def model(t_vector, x_vector, y_vector, new_rel_time, geo_locs, evid, ts_syn, xs_syn, ys_syn, v_syn,
          input_parameters, xs, ys, vs, output_directory):
    """
    Stochastic gradient descent
    :param t_vector: [vector] Time parameters space (prior to first arrival)
    :param x_vector: [vector] Distance x (from Geo3) parameter space in meters
    :param y_vector: [vector] Distance y (from Geo3) parameter space in meters
    :param new_rel_time: [vector] Relative arrival time (ambiguous)
    :param avg_velocity: [float] Average expected wave velocity
    :return:
    """
    # Set the number of iterations and the learning rate
    # True number of iterations is one less than displayed
    num_iterations = 500001
    time_step = float(input_parameters.split('_')[1])
    space_step = float(input_parameters.split('_')[0])
    lr_t = time_step
    lr_x = space_step
    lr_y = lr_x
    lr_v = 50

    # Set a misfit improvement cutoff value. If the mean improvement of the past number of iterations is below this,
    # we are probably ok with stopping.
    iteration_num_cutoff = 10000
    iteration_value_cutoff = 0.1

    # We will want ot save the misfit, but the plot step of 100 is too large. 1000 is fine.
    # misfit_save_bounds = np.arange(0, num_iterations, 1000)

    # Initialize the parameters randomly
    # ts = np.random.choice(t_vector)
    ts = 0

    misfit_vector = []
    iteration_vector = []
    ts_vector = []
    xs_vector = []
    ys_vector = []
    vs_vector = []
    theta_vector = []

    # For easier plotting, pull out the x and y values of the geophone locations
    geo_loc_x = []
    geo_loc_y = []
    for geo_loc in geo_locs:
        geo_loc_x.append(geo_loc[0])
        geo_loc_y.append(geo_loc[1])

    # Take the final x and y location from the first step and calculate a theta between that and geo3
    x3 = geo_loc_x[2]
    y3 = geo_loc_y[2]

    # f = open(f"{output_directory}{evid}_{np.round(ts)}_{np.round(xs)}_{np.round(ys)}.txt", "a")
    for iteration in np.arange(num_iterations):

        # Do a forward propagation of the traces
        misfit = tt_misfit(ts, xs, ys, vs, geo_locs, new_rel_time)

        # Compute theta
        atan_val = math.atan((xs - x3) / (ys - y3))
        atan_val_deg = math.degrees(atan_val)
        if xs > 0 and ys > 0:
            theta_deg = atan_val_deg
        if xs > 0 and ys < 0:
            theta_deg = atan_val_deg + 180
        if xs < 0 and ys < 0:
            theta_deg = atan_val_deg + 180
        if xs < 0 and ys > 0:
            theta_deg = atan_val_deg + 360
        theta_vector.append(theta_deg)

        if iteration > iteration_num_cutoff + 1:
            if abs(theta_vector[iteration - iteration_num_cutoff] - theta_deg) < iteration_value_cutoff:
                break

        iteration_vector.append(iteration)
        misfit_vector.append(misfit)
        ts_vector.append(ts)
        xs_vector.append(xs)
        ys_vector.append(ys)
        vs_vector.append(vs)

        # Compute the gradient using the analytical derivative
        # To make things clear, we will create a new function for each parameter
        dt = comp_dt_grad(ts, xs, ys, vs, geo_locs, new_rel_time)
        dx = comp_dx_grad(ts, xs, ys, vs, geo_locs, new_rel_time)
        dy = comp_dy_grad(ts, xs, ys, vs, geo_locs, new_rel_time)
        dv = comp_vs_grad(ts, xs, ys, vs, geo_locs, new_rel_time)

        # Update parameters using the learning rate
        ts = ts - (lr_t * dt)
        xs = xs - (lr_x * dx)
        ys = ys - (lr_y * dy)
        vs = vs - (lr_v * dv)

        # if iteration % 10000 == 0:
        #     print(f'Misfit1 for iteration {iteration}: {misfit}')
        # f.write(f'{iteration} {misfit}\n')

    # f.close()

    # Some of the misfit vectors are too large to plot all the results. Instead, we will set up plot bounds.
    # We shall plot 1000 points of the vector


    plot_bounds = np.arange(0, len(iteration_vector), int(len(iteration_vector) / 1000))
    plot_bounds = np.append(plot_bounds, np.arange(500))
    plot_bounds = sorted(np.unique(plot_bounds))

    # Check velocity measurements
    # figvel = plt.figure(figsize=(10, 10))
    # ax0 = plt.subplot(3, 1, 1)
    # ax0.scatter(np.array(iteration_vector)[plot_bounds], np.array(misfit_vector)[plot_bounds], c='black')
    # ax0.set_title('Misfit', fontweight='bold')
    # ax0.set_yscale('log')
    # ax1 = plt.subplot(3, 1, 2)
    # ax1.scatter(np.array(iteration_vector)[plot_bounds], np.array(vs_vector)[plot_bounds], c='blue')
    # ax1.axhline(y=v_syn, linestyle='dashed', color='black')
    # ax2 = plt.subplot(3, 1, 3)
    # ax2.scatter(np.array(iteration_vector)[plot_bounds], np.array(ts_vector)[plot_bounds], c='red')
    # ax2.axhline(y=ts[-1], linestyle='dashed', color='black')

    # Create variables for the first source values
    xs_start = xs_vector[0]
    ys_start = ys_vector[0]
    ts_start = ts_vector[0]

    xs1 = xs_vector[-1]
    ys1 = ys_vector[-1]

    atan_val = math.atan((xs1 - x3) / (ys1 - y3))
    atan_val_deg = math.degrees(atan_val)
    if xs1 > 0 and ys1 > 0:
        theta_deg = atan_val_deg
    if xs1 > 0 and ys1 < 0:
        theta_deg = atan_val_deg + 180
    if xs1 < 0 and ys1 < 0:
        theta_deg = atan_val_deg + 180
    if xs1 < 0 and ys1 > 0:
        theta_deg = atan_val_deg + 360

    # Find the theta of the source
    atan_val_syn = math.atan((xs_syn - x3) / (ys_syn - y3))
    atan_val_deg_syn = math.degrees(atan_val_syn)
    if xs_syn > 0 and ys_syn > 0:
        theta_deg_syn = atan_val_deg_syn
    if xs_syn > 0 and ys_syn < 0:
        theta_deg_syn = atan_val_deg_syn + 180
    if xs_syn < 0 and ys_syn < 0:
        theta_deg_syn = atan_val_deg_syn + 180
    if xs_syn < 0 and ys_syn > 0:
        theta_deg_syn = atan_val_deg_syn + 360

    # Calculate the degrees difference between the actual and calculated difference
    theta_diff = np.abs(theta_deg - theta_deg_syn)

    # Do a plot of the misfit after our analysis
    fig1 = plt.figure(figsize=(14, 6), num=2, clear=True)

    # Plot the varying location in space of the location
    ax0 = plt.subplot2grid((2, 4), (0, 0), colspan=2, rowspan=2)
    ax0.scatter(geo_loc_x, geo_loc_y, marker='^')
    ax0.set_xlim((np.min(x_vector), np.max(x_vector)))
    ax0.set_ylim((np.min(y_vector), np.max(y_vector)))
    ax0.scatter(np.array(xs_vector)[plot_bounds], np.array(ys_vector)[plot_bounds],
                c=np.array(np.arange(num_iterations))[plot_bounds], marker='o',
                s=35, cmap=cm.coolwarm)
    ax0.scatter(xs_start, ys_start, marker='o', s=35, facecolors=None, edgecolors='black')
    # plt.scatter(xs_vector[plot_bounds], ys_vector[plot_bounds], c='b', marker='o', s=35)
    # plt.scatter(xs2_vector[plot_bounds], ys2_vector[plot_bounds], c='r', marker='o', s=35)
    ax0.plot([x3, xs_syn], [y3, ys_syn], c='k')
    ax0.plot([x3, xs_vector[-1]], [y3, ys_vector[-1]], c='k')
    ax0.scatter(xs_syn, ys_syn, marker='*', c='k', s=40)
    ax0.set_xlabel('X Distance', fontweight='bold')
    ax0.set_ylabel('Y Distance', fontweight='bold')

    ax1 = plt.subplot2grid((2, 4), (0, 2), colspan=1, rowspan=1)
    ax1.scatter(np.array(iteration_vector)[plot_bounds], np.array(misfit_vector)[plot_bounds],
                c=np.arange(num_iterations)[plot_bounds], marker='o', s=15, cmap=cm.coolwarm)
    ax1.set_xlabel('Iteration', fontweight='bold')
    ax1.set_ylabel('Misfit', fontweight='bold')
    ax1.set_yscale('log')

    ax2 = plt.subplot2grid((2, 4), (1, 2), colspan=1, rowspan=1)
    ax2.scatter(np.array(iteration_vector)[plot_bounds], np.array(theta_vector)[plot_bounds],
                c=np.arange(num_iterations)[plot_bounds], marker='o', s=15, cmap=cm.coolwarm)
    ax2.axhline(y=theta_deg, color='black', linestyle='dashed')
    ax2.set_xlabel('Iteration', fontweight='bold')
    ax2.set_ylabel('Theta', fontweight='bold')

    ax3 = plt.subplot2grid((2, 4), (0, 3), colspan=1, rowspan=1)
    ax3.scatter(np.array(iteration_vector)[plot_bounds], np.array(vs_vector)[plot_bounds],
                c=np.arange(num_iterations)[plot_bounds], marker='o', s=15, cmap=cm.coolwarm)
    ax3.axhline(y=v_syn, color='black', linestyle='dashed')
    ax3.set_xlabel('Iteration', fontweight='bold')
    ax3.set_ylabel('Velocity (m/s)', fontweight='bold')

    ax4 = plt.subplot2grid((2, 4), (1, 3), colspan=1, rowspan=1)
    ax4.scatter(np.array(iteration_vector)[plot_bounds], np.array(ts_vector)[plot_bounds],
                c=np.arange(num_iterations)[plot_bounds], marker='o', s=15, cmap=cm.coolwarm)
    ax4.axhline(y=ts_syn, color='black', linestyle='dashed')
    ax4.set_xlabel('Iteration', fontweight='bold')
    ax4.set_ylabel('Origin Time (s)', fontweight='bold')

    fig1.tight_layout()
    fig1.subplots_adjust(top=0.9)
    fig1.suptitle(f'Params: Source: ({xs_syn}, {ys_syn}, {str(np.round(ts_syn, decimals=1))}), Start: ({xs_start}, {ys_start}) '
                  f': Theta difference = {np.round(theta_diff, 1)} degrees')
    fig1.savefig(f'{output_directory}{input_parameters}/source_{xs_syn}_{ys_syn}_start_{xs_start}_{ys_start}.png')
    # fig1.savefig(f'{output_directory}{input_parameters}/source_{xs_syn}_{ys_syn}_start_{xs_start}_{ys_start}.eps')



    return theta_diff, xs_start, ys_start, ts_start, xs_vector[-1], ys_vector[-1], ts_vector[-1]


def randomize_source(geo_locs, v_syn, xs_syn, ys_syn, ts_syn):
    """
    Returns a randomized source

    :return:
    """

    # Source time
    # ts_syn = -12

    # Set random seed

    # Source location
    # xs_syn = -600
    # ys_syn = -1300

    # Propagate the source to the array to find the arrival time
    new_rel_time = []
    for geo_ind in np.arange(len(geo_locs)):

        # Find the distance between the source and the array
        stax = geo_locs[geo_ind][0]
        stay = geo_locs[geo_ind][1]
        dist = np.sqrt((xs_syn-stax)**2 + (ys_syn-stay)**2)
        new_rel_time.append((dist/v_syn) + ts_syn)

    return new_rel_time


def synthetic_wrapper(input_parameters, xs_syn, ys_syn, ts_syn, xs, ys, v_syn, vs, output_directory):
    """
    Wrapper code for the location algorithm using synthetic data. Requires no input.
    :return:
    """
    # Set average velocity for this area
    # avg_velocity = 34.0

    # Set locations of each geophone relative to geophone 3, the center of the array (in meters)
    geo1 = np.multiply((0.0455, 0.0341), 1000)
    geo2 = np.multiply((-0.0534, 0.0192), 1000)
    geo3 = (0, 0)
    geo4 = np.multiply((0.0119, -0.0557), 1000)
    geo_locs = [geo1, geo2, geo3, geo4]

    # Read in the data into a dataframe
    new_rel_time_perfect = randomize_source(geo_locs, v_syn, xs_syn, ys_syn, ts_syn)

    new_rel_time = []
    for perfect_val in new_rel_time_perfect:
        new_rel_time.append(np.round(perfect_val, decimals=1))

    # Setup the parameter space
    t_vector = np.arange(-10, -0.1, 0.1)
    x_vector = np.arange(-2000, 2001)
    y_vector = np.arange(-2000, 2001)

    # Solve for the location
    evid = f'Synthetic ({xs_syn}, {ys_syn}), Ts = {ts_syn}'
    theta_diff, xs_start, ys_start, ts_start, xs_fin, ys_fin, ts_fin = \
        model(t_vector, x_vector, y_vector, new_rel_time, geo_locs, evid, ts_syn, xs_syn, ys_syn, v_syn,
              input_parameters, xs, ys, vs, output_directory)

    return theta_diff, xs_syn, ys_syn, ts_syn, xs_start, ys_start, ts_start, xs_fin, ys_fin, ts_fin


def test_params(input_parameters, seed_number, num_examples, output_directory):
    """
    Obtains the accuracy for a set of parameters across a number of examples.

    :param input_parameters: [str] Representation of the space and time gradient steps separated by an underscore
    :param seed_number: [int] Random seed. Set to 'no' if you don't want to enable this.
    :param num_examples: [int] Number of iterations to run.
    :param output_directory: [str] Directory where the CSV files created by the param steps are located.
    :return:
    """

    time_step = float(input_parameters.split('_')[1])
    space_step = float(input_parameters.split('_')[0])

    if len(glob.glob(f'{output_directory}{input_parameters}/*.csv')) > 0:
        print(f'{input_parameters} already teststed! Skipping...')
        return

    theta_diff_vector = []
    syn_source_vector = []
    rand_start_vector = []
    sgd_end_vector = []

    # Setup the random start
    if not seed_number == 'no':
        np.random.seed(seed=seed_number)
    xs_syn_vector = np.random.choice(np.arange(-1500, 1500, 1), num_examples)
    ys_syn_vector = np.random.choice(np.arange(-1500, 1500, 1), num_examples)
    ts_syn_vector = np.random.choice(np.arange(-50, 50, 0.1), num_examples)
    v_syn_vector = np.random.choice(np.arange(10, 200, 0.1), num_examples)

    xs_vector = np.random.choice(np.arange(-1500, 1500), num_examples)
    ys_vector = np.random.choice(np.arange(-1500, 1500), num_examples)
    vs_vector = np.random.choice(np.arange(10, 200, 0.1), num_examples)


    for example_ind in np.arange(num_examples):
        theta_diff_out, xs_syn_out, ys_syn_out, ts_syn_out, xs_start_out, ys_start_out, ts_start_out, \
        xs_fin_out, ys_fin_out, ts_fin_out = synthetic_wrapper(input_parameters, xs_syn_vector[example_ind],
                                                               ys_syn_vector[example_ind], ts_syn_vector[example_ind],
                                                               xs_vector[example_ind], ys_vector[example_ind],
                                                               v_syn_vector[example_ind], vs_vector[example_ind],
                                                               output_directory)

        theta_diff_vector.append(np.round(theta_diff_out, 3))
        syn_source_vector.append((xs_syn_out, ys_syn_out, np.round(ts_syn_out, 1)))
        rand_start_vector.append((xs_start_out, ys_start_out, ts_start_out))
        sgd_end_vector.append((np.round(xs_fin_out), np.round(ys_fin_out), np.round(ts_fin_out, 1)))


    theta_average = np.mean(theta_diff_vector)
    combined_data = list(zip(syn_source_vector, rand_start_vector, sgd_end_vector, theta_diff_vector))
    df = pd.DataFrame(combined_data, columns=['syn_source', 'rand_start_vector', 'sgd_end_vector', 'theta_diff'])
    df.to_csv(f'{output_directory}{input_parameters}/theta_num{num_examples}_avgtheta_{np.round(theta_average, 1)}.csv',
              index=False)
    print(f'Finished saving parameters for {input_parameters}!')

    return


def assess_accuracy(output_directory, final_results_directory):
    """
    Assesses the accuracy within all the parameters

    :param output_directory: [str] Directory where the CSV files created by the param steps are located.
    :param final_results_directory: [str] Final output of the comparisons between all of the gradient steps.
    :return:
    """
    # Get a list of all the parameters
    param_list = sorted(glob.glob(f'{output_directory}*'))
    params = []
    for param in param_list:
        params.append(os.path.basename(param))

    # Cycle through each param and get the accuracy
    mean_difference, stdev_difference = [], []
    for param in params:
        result_file = glob.glob(f'{output_directory}{param}/*.csv')[0]
        df = pd.read_csv(result_file)
        theta_differences = df['theta_diff'].values
        mean_difference.append(np.mean(theta_differences))
        stdev_difference.append(np.std(theta_differences))

        fig, axs = plt.subplots(1, 1, figsize=(16, 8), num=20, clear=True)
        axs.errorbar(np.arange(len(theta_differences)), theta_differences, ecolor='black', marker='o', mfc='blue')
        axs.set_xticks(np.arange(len(theta_differences)))
        axs.set_ylabel('Theta Difference', fontweight='bold')
        axs.set_yscale('log')
        axs.set_xlabel('Synthetic Test Number', fontweight='bold')
        axs.set_title(f'Synthetic results for run {param}: mean={np.mean(theta_differences)}, stdev={np.std(theta_differences)}', fontweight='bold')
        fig.tight_layout()
        fig.savefig(f'{final_results_directory}{param}_accuracy.png')
        # fig.savefig(f'{final_results_directory}{param}_accuracy.eps')

    # Find the set of parameters with lowest error rate
    least_min_index = np.where(mean_difference == np.min(mean_difference))[0][0]
    best_run = params[least_min_index]

    # Plot the result
    fig20, axs = plt.subplots(1, 1, figsize=(16, 8), num=20, clear=True)
    axs.errorbar(params, mean_difference, yerr=stdev_difference, ecolor='black', marker='o', mfc='blue')
    axs.set_ylabel('Mean', fontweight='bold')
    axs.set_yscale('log')
    axs.set_xlabel('Runs', fontweight='bold')
    fig20.autofmt_xdate()
    axs.set_title(f'Mean and standard deviation for all runs. Best: {best_run}', fontweight='bold')
    fig20.tight_layout()

    fig20.savefig(f'{final_results_directory}allrun_accuracy.png')
    # fig20.savefig(f'{final_results_directory}allrun_accuracy.eps')
    print('Finished comparing across all parameters.')

    return



# Main
def main(outdir='C:/Users/fcivi/Dropbox/NASA_codes/publication/', num_cores=12, seed_number=1, num_examples=20,
         space_step_windows=[50, 75, 100, 125, 150, 175, 200], time_step_windows=[0.05, 0.1, 0.15, 0.2, 0.25]):
    """
    Main wrapper function for the code.
    If a number of cores greater than 1 is used, the code will run in parallel using the joblib library.

    :param outdir: [str] Sets the output directory
    :param num_cores: [int] Sets the number of CPU cores to use in the processing
    :param seed_number: [int] Random seed. Set to 'no' if you don't want to enable this.
    :param num_examples: [int] Number of iterations to run.
    :param space_step_windows: [list] Vector of x and y gradient steps (m)
    :param time_step_windows: [list] Vector of time gradient steps (s)
    :return:

    """
    output_directory = f'{outdir}synthetic_verification/'
    final_results_directory = f'{outdir}synthetic_results/'
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    if not os.path.exists(final_results_directory):
        os.mkdir(final_results_directory)

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    if not os.path.exists(final_results_directory):
        os.mkdir(final_results_directory)

    # Find all the parameter combinations
    param_combinations = []
    for space_step_win in space_step_windows:
        for time_step_win in time_step_windows:
            if not os.path.exists(f'{output_directory}{space_step_win}_{time_step_win}'):
                os.mkdir(f'{output_directory}{space_step_win}_{time_step_win}')
            param_combinations.append(f'{space_step_win}_{time_step_win}')

    # Computes the test
    if num_cores == 1:
        for param_combination in param_combinations:
                test_params(param_combination, seed_number, num_examples, output_directory)
    else:
        Parallel(n_jobs=num_cores)(delayed(test_params)(param_combination, seed_number, num_examples, output_directory)
                                   for param_combination in param_combinations)

    # Assess the accuracy for all the tests done
    assess_accuracy(output_directory, final_results_directory)

main()




