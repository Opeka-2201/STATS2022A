import numpy as np
import pandas as pd
import os

from scipy.stats import beta, chi2, bootstrap, t
import scipy.optimize as sco

import matplotlib.pyplot as plt

import scipy.special as sc

from typing import List

FIXED_COUNTRIES = ["USA", "Belgium", "China", "Togo"]


#############################
# Functions from helpers.py #
#############################

def population(data_csv: pd.DataFrame, ids: List[int]) -> pd.DataFrame:
    """
    Extract a population for the original dataset.

    Parameters
    ----------
    data_csv: pandas.DataFrame
        Dataset obtained with pandas.read_csv
    ids: List[int]
        List of ULiege ids for each group member (e.g. s167432 and s189134 -> [20167432,20189134])

    Returns
    -------
    DataFrame containing your population
    """
    pop = data_csv.drop(FIXED_COUNTRIES).sample(146, random_state=sum(ids))
    for c in FIXED_COUNTRIES:
        pop.loc[c] = data_csv.loc[c]
    return pop


def beta_log_likelihood(theta, *x):
    """
    Function equal to -log L(\theta;x) to be fed to scipy.optimize.minimize

    Parameters
    ----------
    theta: theta[0] is alpha and theta[1] is beta
    x: x[0] is the data
    """
    a = theta[0]
    b = theta[1]
    n = len(x[0])

    # Log-likelihood
    obj = (a - 1) * np.log(x[0]).sum() + (b - 1) * np.log(1 - x[0]).sum() - n * np.log(sc.beta(a, b))
    # We want to maximize
    sense = -1

    return sense * obj


def scientific_delta(pop: pd.DataFrame) -> float:
    """

    Parameters
    ----------
    pop: pandas.DataFrame
        Dataframe containing a column 'PIB_habitant' and 'CO2_habitant'

    Returns
    -------
    Delta value computed by scientists
    """
    median_gdp = pop["PIB_habitant"].median()
    pop["Rich"] = pop.apply(lambda x: x["PIB_habitant"] >= median_gdp, axis=1)
    means = pop.groupby("Rich")['CO2_habitant'].mean()
    return means[True] - means[False]


###########################
# Complementary functions #
###########################


def compute_pivot(df: pd.DataFrame, pop_size: int, alpha: float):
    """
    Computes the pivot method for interval estimation

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe containing our population (Country, Top10, PIB/habitant, CO2/habitant)
    pop_size: size of sample of population to compute the pivot for
    alpha: confidence percentage

    Returns
    -------
    The interval computed using the pivot method
    """
    degrees_of_freedom = pop_size * 2
    mu = df.sample(n=pop_size).mean()
    chi = [1/chi2.ppf(1-alpha / 2, degrees_of_freedom), 1/chi2.ppf(alpha / 2, degrees_of_freedom)]
    interval = [1 / (2 * pop_size * mu * chi[1]), 1 / (2 * pop_size * mu * chi[0])]
    return interval


def compute_bootstrap(df: pd.DataFrame, pop_size: int, bootstrap_size: int, alpha: float):
    """
    Computes the boostrap method for interval estimation

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe containing our population (Country, Top10, PIB/habitant, CO2/habitant)
    pop_size: size of sample of population to compute the pivot for
    bootstrap_size : number of resamples to compute bootstrap
    alpha: confidence percentage

    Returns
    -------
    The interval computed using the pivot method
    """
    boot = bootstrap((df.sample(n=pop_size).to_numpy(),), np.std, confidence_level=1-alpha, n_resamples=bootstrap_size)
    interval = [1/boot.confidence_interval[1], 1/boot.confidence_interval[0]]
    return interval


def compute_hypothesis(df: pd.DataFrame, n_samples: int, n_runs: int, alpha: float, delta: float):
    """
    Computes the hypothesis testing using student-t

    Parameters
    ----------
    df: pandas.Dataframe
        Dataframe containing our population (Country, Top10, PIB/habitant, CO2/habitant)
    n_samples: size of population to test the hypothesis H_1
    n_runs: number of tests to obtain the percentage
    alpha: confidence percentage
    delta: the H_0 hypothesis

    Returns
    -------
    A proportion (%) of tests passed
    """
    positive_tests = 0
    for i in range(n_runs):
        pop_run = df.sample(n=n_samples)
        median_PIB = df.loc[:, "PIB_habitant"].median()

        CO2_run = pop_run.loc[:, "CO2_habitant"]
        PIB_run = pop_run.loc[:, "PIB_habitant"]

        richs_CO2 = CO2_run[PIB_run > median_PIB]
        poors_CO2 = CO2_run[PIB_run <= median_PIB]

        student_t_dof = len(richs_CO2) + len(poors_CO2) - 2

        s_pooled = np.sqrt(1 / student_t_dof * ((len(richs_CO2) - 1) * richs_CO2.var() + (len(poors_CO2) - 1) *
                                                poors_CO2.var()))

        computed_delta = richs_CO2.mean() - poors_CO2.mean()

        test = computed_delta > delta + t.ppf(1 - alpha, student_t_dof) * s_pooled * np.sqrt(1 / len(richs_CO2) +
                                                                                             1 / len(poors_CO2))
        if test:
            positive_tests += 1

    return positive_tests / n_runs


############################
# Q1 : Analyse descriptive #
############################


def Q1(data_csv: pd.DataFrame, save: bool):
    id1 = 20191230
    id2 = 20190931

    print("\n ### Q1 ###\n")
    pop = population(data_csv, [id1, id2])
    print(pop[146:][:])
    print("")

    mean = np.mean(data_csv, axis=0)
    print("Moyenne : ")
    print(mean)
    print("")

    print("Standard deviation : ")
    std = np.std(data_csv, axis=0)
    print(std)
    print("")

    median = np.median(data_csv, axis=0)
    print("Median :", median)
    print("")

    quantile1 = np.quantile(data_csv, 1 / 4, axis=0)
    print("Quart1: ", quantile1)
    quantile2 = np.quantile(data_csv, 3 / 4, axis=0)
    print("Quart2 :", quantile2)

    bminTOP10 = quantile1[0] - (1.5 * (quantile2[0] - quantile1[0]))
    bmaxTOP10 = quantile2[0] + (1.5 * (quantile2[0] - quantile1[0]))
    print("min TOP10 : ", bminTOP10)
    print("max TOP10 : ", bmaxTOP10)

    bminCO2 = quantile1[1] - (1.5 * (quantile2[1] - quantile1[1]))
    bmaxCO2 = quantile2[1] + (1.5 * (quantile2[1] - quantile1[1]))
    print("min CO2 : ", bminCO2)
    print("max CO2 : ", bmaxCO2)

    bminPIB = quantile1[2] - (1.5 * (quantile2[2] - quantile1[2]))
    bmaxPIB = quantile2[2] + (1.5 * (quantile2[2] - quantile1[2]))
    print("min PIB : ", bminPIB)
    print("max PIB : ", bmaxPIB)

    corr = pd.read_csv('data.csv').corr()
    print(corr)

    ay1 = plt.subplot(131)
    ay1.axes.get_xaxis().set_visible(False)
    plt.boxplot(data_csv.iloc[:, 0])
    plt.title("10% richest \nPIB proportion")

    ay2 = plt.subplot(132)
    ay2.axes.get_xaxis().set_visible(False)
    plt.boxplot(data_csv.iloc[:, 1])
    plt.title("CO2 / Habitant (in T)")

    ay3 = plt.subplot(133)
    ay3.axes.get_xaxis().set_visible(False)
    plt.boxplot(data_csv.iloc[:, 2])
    plt.title("PIB / Habitant")
    plt.tick_params('y', labelleft=False, labelright=True,
                    right=True, left=False, bottom=False)
    if save:
        plt.savefig("figs/boxplot.svg")
    plt.show()

    plt.figure()
    plt.hist(data_csv.iloc[:, 0])
    plt.title("Histogram of 10% richest PIB proportion")
    plt.ylabel("# of state")
    plt.xlabel("percent held")
    if save:
        plt.savefig("figs/Hist_TOP10.svg")
    plt.show()

    plt.figure()
    plt.hist(data_csv.iloc[:, 1])
    plt.title("Histogram of CO2/Habitant")
    plt.ylabel("# of state")
    plt.xlabel("percent held")
    if save:
        plt.savefig("figs/Hist_CO2.svg")
    plt.show()

    plt.figure()
    plt.hist(data_csv.iloc[:, 2])
    plt.title("Histogram of PIB/Habitant")
    plt.ylabel("# of state")
    plt.xlabel("percent held")
    if save:
        plt.savefig("figs/Hist_PIB.svg")
    plt.show()

    plt.figure()
    plt.plot(np.sort(data_csv.iloc[:, 0]), np.linspace(
        0, 1, len(data_csv.iloc[:, 0]), endpoint=False))
    plt.title("ECDF 10% richest PIB proportion")
    if save:
        plt.savefig("figs/ECDF_TOP10.svg")
    plt.show()

    plt.figure()
    plt.plot(np.sort(data_csv.iloc[:, 1]), np.linspace(
        0, 1, len(data_csv.iloc[:, 1]), endpoint=False))
    plt.title("figs/ECDF of CO2/Habitant")
    if save:
        plt.savefig("figs/ECDF_CO2.svg")
    plt.show()

    plt.figure()
    plt.plot(np.sort(data_csv.iloc[:, 2]), np.linspace(
        0, 1, len(data_csv.iloc[:, 2]), endpoint=False))
    plt.title("ECDF of PIB/Habitant")
    if save:
        plt.savefig("figs/ECDF_PIB.svg")
    plt.show()

    plt.subplot(231)
    plt.scatter(data_csv.iloc[:, 0], data_csv.iloc[:, 1], s=1)
    plt.title("TOP 10")
    plt.subplot(232)
    plt.text(0.45, 0.45, s="CO2", fontsize="x-large")
    plt.axis("off")
    plt.subplot(234)
    plt.scatter(data_csv.iloc[:, 0], data_csv.iloc[:, 2], s=1)
    plt.subplot(235)
    plt.scatter(data_csv.iloc[:, 1], data_csv.iloc[:, 2], s=1)
    plt.tick_params('y', labelleft=False)
    plt.subplot(236)
    plt.text(0, 0.45, s="PIB", fontsize="x-large")
    plt.axis("off")
    plt.suptitle("Matrix plot                              ")
    if save:
        plt.savefig("figs/matrix_plot.svg")
    plt.show()


##############################
# Q2 : Estimation ponctuelle #
##############################


def Q2(populationTest: pd.DataFrame, n_samples: int, save: bool, data_csv):
    print("\n ### Q2 ###\n")
    print("### Specific population\n")
    testTop10 = populationTest[["Top10"]].sample(n_samples)

    var = testTop10.var()[0]
    mean = testTop10.mean()[0]
    aMom = mean * ((((1 - mean) * mean) / var) - 1)
    bMom = ((((1 - mean) * mean) / var) - 1) * (1 - mean)
    print("alpha MOM:", aMom)
    print("beta MOM:", bMom)

    (aMle, bMle) = Mle(testTop10)
    print("alpha MLE:", aMle)
    print("beta MLE: ", bMle)

    testTop10 = populationTest[["Top10"]]
    plt.figure()

    plt.hist(testTop10, density=True)
    x = np.arange(0.01, 1, 0.01)
    y = (beta.pdf(x, aMle, bMle))
    plt.plot(x, y, label='MLE')
    y = beta.pdf(x, aMom, bMom)
    plt.plot(x, y, label='MOM')
    plt.title("Real data compared to Beta distribution \nwith estimated parameters")
    plt.legend()
    if save:
        plt.savefig("figs/MLE_MOM_Beta.svg")
    plt.show()

    print("\n### Sampled population ###\n")
    test = [50, 20, 40, 60, 80, 100]
    test_evo = [20, 40, 60, 80, 100]

    aMom_v = np.zeros([1, 5])
    bMom_v = np.zeros([1, 5])
    biasAMom_v = np.zeros([1, 5])
    biasBMom_v = np.zeros([1, 5])
    varAMom_v = np.zeros([1, 5])
    varBMom_v = np.zeros([1, 5])
    MSEAMom_v = np.zeros([1, 5])
    MSEBMom_v = np.zeros([1, 5])

    aMle_v = np.zeros([1, 5])
    bMle_v = np.zeros([1, 5])
    biasAMle_v = np.zeros([1, 5])
    biasBMle_v = np.zeros([1, 5])
    varAMle_v = np.zeros([1, 5])
    varBMle_v = np.zeros([1, 5])
    MSEAMle_v = np.zeros([1, 5])
    MSEBMle_v = np.zeros([1, 5])

    for j in test:
        echantillon1 = np.zeros((500, 2))
        echantillon2 = np.zeros((500, 2))
        for i in range(500):
            pop = np.random.choice(data_csv.iloc[:, 0], j)
            echantillon1[i] = Mom(pop)
            echantillon2[i] = Mle(pop)

        if j != 50:
            aMom_v[0][int(j / 20) - 1] = echantillon1[:, 0].mean()
            bMom_v[0][int(j / 20) - 1] = echantillon1[:, 1].mean()
            biasAMom_v[0][int(j / 20) - 1] = echantillon1[:, 0].mean() - 13.35
            biasBMom_v[0][int(j / 20) - 1] = echantillon1[:, 1].mean() - 16.31
            varAMom_v[0][int(j / 20) - 1] = np.var(echantillon1[:, 0])
            varBMom_v[0][int(j / 20) - 1] = np.var(echantillon1[:, 1])
            MSEAMom_v[0][int(j / 20) - 1] = np.square(np.subtract(echantillon1[:, 0], 13.35)).mean()
            MSEBMom_v[0][int(j / 20) - 1] = np.square(np.subtract(echantillon1[:, 1], 16.31)).mean()

            aMle_v[0][int(j / 20) - 1] = echantillon2[:, 0].mean()
            bMle_v[0][int(j / 20) - 1] = echantillon2[:, 1].mean()
            biasAMle_v[0][int(j / 20) - 1] = echantillon2[:, 0].mean() - 13.35
            biasBMle_v[0][int(j / 20) - 1] = echantillon2[:, 1].mean() - 16.31
            varAMle_v[0][int(j / 20) - 1] = np.var(echantillon2[:, 0])
            varBMle_v[0][int(j / 20) - 1] = np.var(echantillon2[:, 1])
            MSEAMle_v[0][int(j / 20) - 1] = np.square(np.subtract(echantillon2[:, 0], 13.35)).mean()
            MSEBMle_v[0][int(j / 20) - 1] = np.square(np.subtract(echantillon2[:, 1], 16.31)).mean()

        print(j, ": aMom: ", echantillon1[:, 0].mean())
        print(j, ": bMom: ", echantillon1[:, 1].mean())
        print(j, ": Biais aMom: ", echantillon1[:, 0].mean() - 13.35)
        print(j, ": Biais bMom: ", echantillon1[:, 1].mean() - 16.31)
        print(j, ": Variance aMom: ", np.var(echantillon1[:, 0]))
        print(j, ": Variance bMom: ", np.var(echantillon1[:, 1]))
        print(j, ": Quad Error aMom: ", np.square(np.subtract(echantillon1[:, 0], 13.35)).mean())
        print(j, ": Quad Error bMom: ", np.square(np.subtract(echantillon1[:, 1], 16.31)).mean())

        print(j, ": aMle: ", echantillon2[:, 0].mean())
        print(j, ": bMle: ", echantillon2[:, 1].mean())
        print(j, ": Biais aMle: ", echantillon2[:, 0].mean() - 13.35)
        print(j, ": Biais bMle: ", echantillon2[:, 1].mean() - 16.31)
        print(j, ": Variance aMle: ", np.var(echantillon2[:, 0]))
        print(j, ": Variance bMle: ", np.var(echantillon2[:, 1]))
        print(j, ": Quad Error aMle: ", np.square(
            np.subtract(echantillon2[:, 0], 13.35)).mean())
        print(j, ": Quad Error bMle: ", np.square(
            np.subtract(echantillon2[:, 1], 16.31)).mean())
        if j == 50:
            print("\n --- Bonus ---")

    plt.figure()
    plt.title("Evolution of parameters wrt sample size")
    plt.plot(test_evo, aMom_v[0], label="a_MOM")
    plt.plot(test_evo, bMom_v[0], label="b_MOM")
    plt.plot(test_evo, aMle_v[0], label="a_MLE")
    plt.plot(test_evo, bMle_v[0], label="b_MLE")
    plt.legend()
    plt.xlabel("Sample size")
    plt.ylabel("Parameters")
    if save:
        plt.savefig("figs/evo.svg")
    plt.show()

    plt.figure()
    plt.title("Evolution of bias wrt sample size")
    plt.plot(test_evo, biasAMom_v[0], label="bias_a_MOM")
    plt.plot(test_evo, biasBMom_v[0], label="bias_b_MOM")
    plt.plot(test_evo, biasAMle_v[0], label="bias_a_MLE")
    plt.plot(test_evo, biasBMle_v[0], label="bias_b_MLE")
    plt.legend()
    plt.xlabel("Sample size")
    plt.ylabel("Bias")
    if save:
        plt.savefig("figs/evobias.svg")
    plt.show()

    plt.figure()
    plt.title("Evolution of variance wrt sample size")
    plt.plot(test_evo, varAMom_v[0], label="var_a_MOM")
    plt.plot(test_evo, varBMom_v[0], label="var_b_MOM")
    plt.plot(test_evo, varAMle_v[0], label="var_a_MLE")
    plt.plot(test_evo, varBMle_v[0], label="var_b_MLE")
    plt.legend()
    plt.xlabel("Sample size")
    plt.ylabel("Variance")
    if save:
        plt.savefig("figs/evovar.svg")
    plt.show()

    plt.figure()
    plt.title("Evolution of MSE wrt sample size")
    plt.plot(test_evo, MSEAMom_v[0], label="MSE_a_MOM")
    plt.plot(test_evo, MSEBMom_v[0], label="MSE_b_MOM")
    plt.plot(test_evo, MSEAMle_v[0], label="MSE_a_MLE")
    plt.plot(test_evo, MSEBMle_v[0], label="MSE_b_MLE")
    plt.legend()
    plt.xlabel("Sample size")
    plt.ylabel("MSE")
    if save:
        plt.savefig("figs/evomse.svg")
    plt.show()


def Mle(pop: pd.DataFrame):
    x = sco.minimize(beta_log_likelihood, np.array([1, 1]), pop)
    aMle = x.x[0]
    bMle = x.x[1]
    return aMle, bMle


def Mom(pop: pd.DataFrame):
    var = pop.var()
    mean = pop.mean()
    aMom = mean * ((((1 - mean) * mean) / var) - 1)
    bMom = ((((1 - mean) * mean) / var) - 1) * (1 - mean)
    return aMom, bMom


##################################
# Q3 : Estimation par intervalle #
##################################

def Q3(pop: pd.DataFrame, pop_size: int, n_samples: int, bootstrap_size: int, save: bool):
    print("\n ### Q3 ###\n")
    alpha = 0.05
    lambda_computed = 5.247 * 10 ** (-5)
    PIB = pop.loc[:, 'PIB_habitant']
    int_pivot = compute_pivot(PIB, pop_size, alpha)
    int_bootstrap = compute_bootstrap(PIB, pop_size, bootstrap_size, alpha)

    print("pivot : ", int_pivot)
    print("bootstrap : ", int_bootstrap)

    sample_range = range(5, pop_size + 1, 5)
    pvt_size_evo = np.zeros(len(sample_range))
    pvt_continuous = np.zeros(len(sample_range))
    boot_size_evo = np.zeros(len(sample_range))
    boot_continuous = np.zeros(len(sample_range))

    pvt_size = np.zeros(n_samples)
    boot_size = np.zeros(n_samples)

    for sample in sample_range:
        pvt_tmp = 0
        boot_tmp = 0
        for i in range(0, n_samples):
            int_pivot = compute_pivot(PIB, sample, alpha)
            int_bootstrap = compute_bootstrap(PIB, sample, bootstrap_size, alpha)
            pvt_size[i] = int_pivot[1] - int_pivot[0]
            boot_size = int_bootstrap[1] - int_bootstrap[0]

            if int_pivot[0] < lambda_computed < int_pivot[1]:
                pvt_tmp += 1

            if int_bootstrap[0] < lambda_computed < int_bootstrap[1]:
                boot_tmp += 1

        pvt_size_evo[sample_range.index(sample)] = pvt_size.mean()
        pvt_continuous[sample_range.index(sample)] = pvt_tmp / n_samples
        boot_size_evo[sample_range.index(sample)] = boot_size.mean()
        boot_continuous[sample_range.index(sample)] = boot_tmp / n_samples

    plt.figure()
    plt.plot(sample_range, pvt_size_evo, label="Pivot")
    plt.plot(sample_range, boot_size_evo, label="Bootstrap")
    plt.legend()

    if save:
        plt.savefig("figs/intervalle_echantillon.svg")

    plt.show()

    plt.figure()
    plt.plot(sample_range, pvt_continuous, label="Pivot")
    plt.plot(sample_range, boot_continuous, label="Bootstrap")
    plt.legend()

    if save:
        plt.savefig("figs/prop_lambda.svg")

    plt.show()

#########################
# Q4 : Test d'hypothèse #
#########################


def Q4(pop: pd.DataFrame, alpha: float, n_runs: int):
    print("\n ### Q4 ###\n")
    PIB = pop.loc[:, 'PIB_habitant']
    CO2 = pop.loc[:, 'CO2_habitant']

    median_PIB = PIB.median()
    delta_sc = scientific_delta(pop)
    poors_CO2 = CO2[PIB <= median_PIB]
    richs_CO2 = CO2[PIB > median_PIB]

    delta_real = richs_CO2.mean() - poors_CO2.mean()
    error = np.abs(delta_real - delta_sc)
    print(error)
    print('prop sample = 75 : ', compute_hypothesis(pop, 75, n_runs, alpha, delta_real))
    print('prop sample = 25 : ', compute_hypothesis(pop, 25, n_runs, alpha, delta_real))


###############
# Main script #
###############
os.system('mkdir figs')

data = pd.read_csv("data.csv", index_col=0)
studentsID = [20191230, 20190931]
popTest = population(data, studentsID)

samples = 50

boot_sample_size = 100
N = 500

samples_hypo = 10000
confidence = 0.05

Q1(data, False)
Q2(data, samples, False, data)
Q3(popTest, samples, N, boot_sample_size, False)
Q4(popTest, confidence, samples_hypo)
