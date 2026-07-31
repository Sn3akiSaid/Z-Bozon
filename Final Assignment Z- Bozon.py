# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
from scipy.constants import hbar, elementary_charge

GAMMA_EE = 83.91 / 1000  # GeV
MASS_GUESS = 90  # GeV/c^2
GAMMA_Z_GUESS = 3  # GeV


def reading_data(boson_file_number):

    boson_files = [("z_boson_data_1.csv"), ("z_boson_data_2.csv")]
    boson_file = np.genfromtxt(
        boson_files[boson_file_number], delimiter=",", skip_header=1
    )

    boson_file_modified = boson_file[boson_file[:, 2] != 0]

    nan_row_indices = []
    for row_index, row in enumerate(boson_file_modified):
        if (
            np.isnan(row[0]) is True
            or np.isnan(row[1]) is True
            or np.isnan(row[2]) is True
        ):
            nan_row_indices.append(row_index)
    boson_file_modified = np.delete(boson_file_modified, nan_row_indices, axis=0)
    for i in [0, 1, 2]:
        boson_file_modified = boson_file_modified[boson_file_modified[:, i] >= 0]

    cross_section = boson_file_modified[:, 1]
    standard_deviation = np.std(cross_section)
    median = np.median(cross_section)

    anomalies_row_indices = []

    for row_index, row in enumerate(boson_file_modified):
        if np.abs(row[1] - median) > 3 * standard_deviation:
            anomalies_row_indices.append(row_index)
    boson_file_modified = np.delete(boson_file_modified, anomalies_row_indices, axis=0)
    return boson_file_modified


def data_combination(data_set_1, data_set_2):
    finalised_data_set = np.array([])
    finalised_data_set = np.vstack((data_set_1, data_set_2))
    return finalised_data_set


def outlier_removal(datafile, mass, gamma_z):
    """


    Parameters
    ----------
    datafile : TYPE
        DESCRIPTION.
    mass : TYPE
        DESCRIPTION.
    gamma_z : TYPE
        DESCRIPTION.

    Returns
    -------
    datafile_modified : TYPE
        DESCRIPTION.

    """

    outlier_row_indices = []
    for row_index, _ in enumerate(datafile):
        energy = datafile[row_index, 0]
        sigma = datafile[row_index, 2]
        cross_section = datafile[row_index, 1]
        predicted_cross_section = cross_section_function(energy, mass, gamma_z)

        # if cross_section > predicted_cross_section + 3 * sigma:
        #     outlier_row_indices.append(row_index)
        if (
            predicted_cross_section + 3 * sigma
            < cross_section
            < predicted_cross_section - 3 * sigma
        ):
            outlier_row_indices.append(row_index)
    datafile_modified = np.delete(datafile, outlier_row_indices, axis=0)
    return datafile_modified


def cross_section_function(energy, mass, gamma_z):
    return 0.3894e6 * (
        (12 * np.pi / mass**2)
        * (energy * GAMMA_EE) ** 2
        / ((energy**2 - mass**2) ** 2 + (mass * gamma_z) ** 2)
    )  # nanobarns


def lifetime(gamma):
    return hbar / (gamma * elementary_charge * 1e9)


def chi_squared(mass_width, energy, observed, uncertainty):
    mass = mass_width[0]
    width = mass_width[1]

    return np.sum(
        ((cross_section_function(energy, mass, width) - observed) / uncertainty) ** 2
    )


def reduced_chi_squared(mass_width, energy, observed, uncertainty):
    degrees_of_freedom = len(observed) - 2

    return chi_squared(mass_width, energy, observed, uncertainty) / degrees_of_freedom


def array_gen(mass, width):
    x_array = np.linspace(mass - 0.0007 * mass, mass + 0.0007 * mass, 150)
    y_array = np.linspace(width - 0.1, width + 0.1, 150)
    return x_array, y_array


def mesh_arrays(x_array, y_array, energy, observed, uncertainty):
    """Returns two meshed arrays of size len(x_array)
    by len(y_array)
    x_array array[floats]
    y_array array[floats]
    """
    x_array_mesh = np.empty((0, len(x_array)))

    for _ in y_array:
        x_array_mesh = np.vstack((x_array_mesh, x_array))

    y_array_mesh = np.empty((0, len(y_array)))

    for dummy_element in x_array:
        y_array_mesh = np.vstack((y_array_mesh, y_array))

    y_array_mesh = np.transpose(y_array_mesh)

    chi_squared_array = np.empty((150, 150))

    for i, x_value in enumerate(x_array):
        for j, y_value in enumerate(y_array):
            chi_squared_value = chi_squared(
                (x_value, y_value), energy, observed, uncertainty
            )
            chi_squared_array[i, j] = chi_squared_value

    return x_array_mesh, y_array_mesh, chi_squared_array


def minimisation(mass_start, gamma_start, x_data, y_data, uncert):
    guesses = [mass_start, gamma_start]
    minimised_results = fmin(
        chi_squared, guesses, args=(x_data, y_data, uncert), full_output=True, disp=0
    )
    [mass_min, gamma_min] = minimised_results[0]
    cross_section_min = minimised_results[1]
    return mass_min, gamma_min, cross_section_min


def contour_plot(energy, observed, uncertainty, chi_square):
    mass_array, gamma_array = array_gen(mass_min3, gamma_min3)

    mass_mesh, gamma_mesh, chi_squared_mesh = mesh_arrays(
        mass_array, gamma_array, energy, observed, uncertainty
    )
    ax = plt.figure().add_subplot(111)
    ax.set_title(r"$\chi^2$ contours against $m_Z$ and $\Gamma_Z$.", fontsize=14)
    ax.set_xlabel("$m_Z$", fontsize=14)
    ax.set_ylabel(r"$\Gamma_Z$", fontsize=14)
    ax.scatter(mass_min3, gamma_min3)
    plot = ax.contour(
        mass_mesh,
        gamma_mesh,
        chi_squared_mesh,
        levels=[chi_square + 1, chi_square + 2.3, chi_square + 5.99, chi_square + 9.21],
    )
    labels = [
        "Minimum",
        r"$\chi^2_{{\mathrm{{min.}}}}+1.00$",
        r"$\chi^2_{{\mathrm{{min.}}}}+2.30$",
        r"$\chi^2_{{\mathrm{{min.}}}}+5.99$",
        r"$\chi^2_{{\mathrm{{min.}}}}+9.21$",
    ]

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

    for index, label in enumerate(labels):
        ax.collections[index].set_label(label)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=14)
    return plot


def cross_section_plot(Energy_Range, mass_min3, gamma_min3):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # plot of data
    ax.set_xlabel("Energy", fontsize=14)
    ax.set_ylabel(r"$\sigma$", fontsize=14)
    ax.tick_params(axis="x", width=1, labelsize=10)
    ax.scatter(Energy2, Cross_Section2, marker="x", linewidth=1, label="Data points")
    ax.errorbar(
        Energy2,
        Cross_Section2,
        Cross_Section_errors2,
        ls="none",
        ecolor="green",
        linewidth=1.3,
        label=r"Uncertainty on $\sigma$",
    )
    ax.set_title(r"Cross Section($\sigma$) against Energy of $Z_0$ boson")

    ax.plot(
        Energy_Range,
        cross_section_function(Energy_Range, mass_min3, gamma_min3),
        color="r",
        label="Fit of data",
    )
    ax.legend(loc="upper left")


def uncertanties(contour):
    get_paths = contour.collections[0].get_paths()[0]
    vertices = get_paths.vertices
    mass_vertex = vertices[:, 0]
    gamma_vertex = vertices[:, 1]
    mass_uncertainty = np.max(mass_vertex) - np.min(mass_vertex)
    gamma_uncertainty = np.max(gamma_vertex) - np.min(gamma_vertex)
    return mass_uncertainty, gamma_uncertainty


final_data = data_combination(reading_data(0), reading_data(1))

Energy1 = final_data[:, 0]
Cross_Section1 = final_data[:, 1]
Cross_Section_errors1 = final_data[:, 2]
mass_min2, gamma_min2, cross_section_min2 = minimisation(
    MASS_GUESS, GAMMA_Z_GUESS, Energy1, Cross_Section1, Cross_Section_errors1
)

final_data2 = outlier_removal(final_data, mass_min2, gamma_min2)
Energy2 = final_data2[:, 0]
Cross_Section2 = final_data2[:, 1]
Cross_Section_errors2 = final_data2[:, 2]
mass_min3, gamma_min3, cross_section_min3 = minimisation(
    MASS_GUESS, GAMMA_Z_GUESS, Energy2, Cross_Section2, Cross_Section_errors2
)

# contour plot from function
#
_contour = contour_plot(
    Energy2, Cross_Section2, Cross_Section_errors2, cross_section_min3
)

# Uncertainties and Call plot
mass_uncertainty, gamma_uncertainty = uncertanties(_contour)


# fitting plot
Energy_Minimum = np.min(Energy1)
Energy_Maximum = np.max(Energy1)
Energy_Range = np.linspace(Energy_Minimum, Energy_Maximum, 1000)
cross_section_plot(Energy_Range, mass_min3, gamma_min3)


plt.show()


print(
    f"The Gamma value is {0:.4g} +/- {3:.4f} and the mass is {1:.4g} +/- {4}."
    " The lifetime of the particle is {2:.3E}s.".format(
        gamma_min3, mass_min3, lifetime(gamma_min3), mass_uncertainty, gamma_uncertainty
    )
)
print(
    f"The reduced chi^2 value is {0:.3f}".format(
        reduced_chi_squared(
            (mass_min3, gamma_min3), Energy2, Cross_Section2, Cross_Section_errors2
        )
    )
)
