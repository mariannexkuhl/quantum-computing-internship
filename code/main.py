# packages  

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from functools import partial

# scikit
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

# qiskit
from qiskit import QuantumCircuit, transpile
import qiskit.quantum_info as qi
from qiskit.circuit.library import n_local, TwoLocal, RXGate, RZZGate, CXGate

from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from qiskit.visualization import plot_histogram, plot_state_city
from qiskit.providers.aer import AerSimulator

from qiskit_finance.applications import PortfolioOptimization
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver, QAOA
from qiskit_optimization.applications import max_cut
from qiskit_optimization.converters import QuadraticProgramToQubo

# azure
import azure.quantum
from azure.quantum.qiskit import AzureQuantumProvider
from azure.quantum import Workspace

# Azure Quantum set-up
workspace = Workspace(
    #replace with your IBM Quantum workspace details
)
AZURE_CONFIG = {"preferred_backend": "rigetti.sim.qvm", "location": "westus"}

provider = AzureQuantumProvider(workspace=workspace)
print("This workspace's targets:")
for backend in provider.backends():
    print("- " + backend.name())
backend_name = AZURE_CONFIG["preferred_backend"]
backend = provider.get_backend(backend_name)

# ====================================================================================== classes, functions and all the main code below


class HelperFunctions:
    """Class for miscellaneous functions."""

    def __init__(
        self, selected_tickers, expected_returns, covariance_matrix, latest_prices, table
    ):
        """Initialises the instance of helper functions

        Args:
            budget (integer): number of assets user wishes to invest in
            risk (float): risk tolerance of the user expressed as a number between 0 and 1
            expected_returns (float): expected returns of asset determiend from historical financial data
            tickers (array): ticker name of asset
            covariance_matrix (array): matrix encoding variance and covariance of all assets in market
        """
        self.selected_tickers = selected_tickers
        self.expected_returns = expected_returns
        self.covariance_matrix = covariance_matrix
        self.latest_prices = latest_prices
        self.table = table

    def evaluate_portfolio(self):
        """Evaluates how good the portfolio is, calculates outputs (total cost, expected returns, risk) and outputs in a user-friendly way.

        Args:
            selected_tickers (array): array indicating which assets are included (indicated by a 1) or not included (indicated by a 0) in the portfolio
            expected_returns (float): expected returns of asset determiend from historical financial data
            covariance_matrix (array): matrix encoding variance and covariance of all assets in market
            latest_prices (array): array of the latest prices of each asset as indicated on Yahoo Finance database.

        Returns:
            summary (dictionary): user friendly formatted summary of optimal portfolio
        """
        weights = np.ones(len(selected_tickers)) / len(
            selected_tickers
        )  # equal weighting for all assets for simplicity
        mu = self.expected_returns[
            selected_tickers
        ]  # extract expected returns for selected tickers
        sigma = covariance_matrix.loc[
            selected_tickers, selected_tickers
        ]  # extract covariance matrix for selected tickers

        expected_return = np.dot(weights, mu)  # total expected return of the portfolio
        risk = np.dot(
            weights, np.dot(sigma, weights)
        )  # total risk of the portfolio using risk = wᵗ Σ w
        total_cost = sum(latest_prices[ticker] for ticker in selected_tickers)
        # total cost of the portfolio based on latest prices, could be different from budget

        ticker_to_name = dict(zip(self.table["Symbol"], self.table["Security"]))

        summary = {
            "Selected Assets": selected_tickers,
            # "Expected Annual Return": expected_return,
            # "Risk (Volatility)": risk,
            "Total Cost": total_cost,
        }
        summary["Selected Assets"] = [
            ticker_to_name[t] for t in summary["Selected Assets"]
        ]
        summary["Expected Annual Return ± Risk (Volatility)"] = (
            f"{expected_return * 100:.2f} ± {risk * 100:.2f} %"
        )
        summary["Total Cost"] = f"{total_cost * 1:.2f} USD"
        return summary


class ClassicalMethod:
    """Class for classical portfolio opimization implementation"""

    def __init__(self, budget, risk, expected_returns, tickers, covariance_matrix):
        """Initialises the instance of classical implementation

        Args:
            budget (integer): number of assets user wishes to invest in
            risk (float): risk tolerance of the user expressed as a number between 0 and 1
            expected_returns (float): expected returns of asset determiend from historical financial data
            tickers (array): ticker name of asset
            covariance_matrix (array): matrix encoding variance and covariance of all assets in market
        """
        self.budget = budget
        self.risk = risk
        self.expected_returns = expected_returns
        self.tickers = tickers
        self.covariance_matrix = covariance_matrix

    def library_qubo_conversion(self, mu, sigma, risk_factor, budget):
        """Converts QUBO to Quadratic Problem (QP) form using Python libraries

        Args:
            mu (array): expected return of a specific asset (i.e. mu_i corresponds to expected returns of asset i)
            sigma (array): covariance matrix
            risk_factor (float): risk tolerance of the user expressed as a number between 0 and 1
            budget (integer): number of assets user wishes to invest in

        Returns:
            qp (array): quadratic problem
        """
        portfolio = PortfolioOptimization(  # create PortfolioOptimization instance
            expected_returns=mu,
            covariances=sigma,
            risk_factor=risk_factor,
            budget=budget,
        )
        qp = portfolio.to_quadratic_program()
        return qp

    def manual_qubo_conversion(self):
        """Converts QUBO to Quadratic Problem (QP) form using manual implementation

        Returns:
            Q_dict (dictionary): dictionary form of QUBO problem
        """
        returns = np.array(
            list(self.expected_returns.values)
        )  # convert expected returns to numpy array
        mu = returns / np.max(
            returns
        )  # normalise expected returns and name them mu as in the formula
        tickers = list(
            self.expected_returns.keys()
        )  # extract tickers from expected returns
        Sigma = self.covariance_matrix / np.max(
            self.covariance_matrix.values
        )  # normalise covariance matrix and name it Sigma as in the formula
        n_assets = len(self.tickers)

        Q = np.zeros(
            (n_assets, n_assets)
        )  # create an empty matrix with dimensions of number of assets

        for i in range(n_assets):  # iterate through each asset
            for j in range(n_assets):
                Q[i][j] += (1 - self.risk) * Sigma.loc[
                    tickers[i], tickers[j]
                ]  # take risk factor into account with the covariance
            Q[i][i] -= (
                self.risk * mu[i]
            )  # diagonal elements are adjusted by risk factor and expected returns (one stock's risk rather than two stocks covarying)

        penalty = 10  # ensure that total value of portfolio is equal to budget
        prices = np.array(
            [self.expected_returns[t] for t in tickers]
        )  # convert expected returns to prices for each stock

        for i in range(n_assets):  # iterate through each asset
            for j in range(n_assets):
                Q[i][j] += (
                    penalty * prices[i] * prices[j]
                )  # update matrix with penalty quadratic term
            Q[i][i] += (
                -2 * penalty * self.budget * prices[i]
            )  # diagonal terms have factor of -2 because of expanding the quadratic term
        Q_dict = {}  # turn matrix into dictionary for easier access
        for i in range(n_assets):
            for j in range(i, n_assets):
                Q_dict[(i, j)] = Q[i][j]
        return Q_dict

    # option 1: using library optimizer
    def classical_library_optimise_portfolio(self, qp):
        """Carries out potfolio optimization using Python libraries

        Args:
            qp (array): quadratic problem

        Returns:
            result (array, float): optimiser result as an array of assets (x) and the expected returns value (fval)
        """
        optimizer = MinimumEigenOptimizer(NumPyMinimumEigensolver())
        result = optimizer.solve(qp)
        # print("Optimal portfolio solution:")
        # print("Asset selection vector (x):", result.x)
        # print("Optimal value (objective function):", result.fval)
        return result

    # option 2: using own optimizer implementation
    def classical_manual_optimise_portfolio(self, Q_dict, tickers, n_assets):
        """Carries out potfolio optimization using manual implementation

        Args:
            Q_dict (dictionary): dictionary form of QUBO problem
            tickers (array): ticker name of asset
            n_assets (integer): total number of assets in the market

        Returns:
            selected_tickers (array): array indicating which assets are included (indicated by a 1) or not included (indicated by a 0) in the portfolio
        """
        # 2^n possible portfolios so very inefficient
        best_obj = float(
            "inf"
        )  # initializing the best objective value to +infinity since we are looking for the lowest value
        best_sol = None  # to store the best solution found
        # loop to brute force through all combinations of assets
        for x in product(
            [0, 1], repeat=n_assets
        ):  # cartesian product of 0 and 1 for n_assets to check all combinations
            if sum(x) != self.budget:
                continue
            obj = 0  # initialise obj value
            for (
                i,
                j,
            ), q_val in Q_dict.items():  # fetch penalty adjusted, covariance values
                obj += (
                    q_val * x[i] * x[j]
                )  # update obj using penalty adjusted covariance values for each pair of assets
            if (
                obj < best_obj
            ):  # if the objective value is lower than the best found so far so we update the best value
                best_obj = obj
                best_sol = x  # best solution is the array of indices indicating which assets are selected
            selected_tickers = [
                tickers[i] for i, bit in enumerate(best_sol) if bit == 1
            ]  # returns tickers rather than indices so user can understand the output
        return selected_tickers


class QuantumMethod:
    """Class for quantum portfolio opimization implementation"""
    def __init__(self, budget=None, risk=None, expected_returns=None, tickers=None, covariance_matrix=None):
        self.budget = budget
        self.risk = risk
        self.expected_returns = expected_returns
        self.tickers = tickers
        self.covariance_matrix = covariance_matrix

    def quantum_library_optimise_portfolio(self, qp, sampler):
        """Carries out potfolio optimization using Python and Qiskit libraries

        Args:
            qp (array): quadratic problem

        Returns:
            selected_tickers (array): array indicating which assets are included (indicated by a 1) or not included (indicated by a 0) in the portfolio
        """
        # sampler = Sampler(backend=backend)
        #sampler = backend.get_sampler()
        cobyla = COBYLA()  # initialise classical optimizer that optimises paramters
        cobyla.set_options(
            maxiter=250
        )  # ensuring we constrain the iterations so that it doesn't take up too much computing resources
        qaoa_mes = QAOA(
            sampler=sampler,
            optimizer=cobyla,
            reps=1,  # change to Sampler() if using simulator
        )  # creates QAOA based Minimum EigenSolver (the actual solver)
        qaoa = MinimumEigenOptimizer(
            qaoa_mes
        )  # MinimumEigenOptimizer translates the QP into an Ising Hamiltonian that QAOA can actually handle
        result = qaoa.solve(qp)  # solve
        return result

    # old function for quantum manual (WORKS FOR SIMULATOR ONLY)
    def quantum_manual_optimise_portfolio(self, Q_dict, tickers, n_assets):
        """Carries out potfolio optimization using manual implementation. This function is useable only on simulators.

        Args:
            Q_dict (dictionary): dictionary form of QUBO problem
            tickers (array): ticker name of asset
            n_assets (integer): total number of assets in the market

        Returns:
            selected_tickers (array): array indicating which assets are included (indicated by a 1) or not included (indicated by a 0) in the portfolio
        """
        # initialise classical optimizer
        cobyla = COBYLA()
        cobyla.set_options(maxiter=250, tol=1)

        # initialise quantum circuit
        qc = QuantumCircuit(
            n_assets
        )  # create the circuit with n_assets number of qubits

        # define the variable parameters
        gamma = np.pi / 4
        beta = np.pi / 4

        for i in range(
            n_assets
        ):  # place each qubit in superposition using H gate (0 = not selected, 1 = selected)
            qc.h(i)

        for (i, j), coeff in Q_dict.items():
            if i == j:  # diagonal terms
                qc.rz(2 * gamma * coeff, i)
            else:  # off-digonal terms
                #qc.append(RZZGate(2 * gamma * coeff), [i, j])
                qc.cx(i, j)
                qc.rz(2 * gamma * coeff, j)
                qc.cx(i, j)

        for i in range(n_assets):
            qc.rx(2 * beta, i)

        qc.measure_all()
        print(qc)
        print("Number of qubits:", qc.num_qubits)
        print("Depth:", qc.depth())
        job = backend.run(qc)
        print("Job ID:", job.job_id())
        try:
            job = backend.run(qc)
            print("Job ID:", job.job_id())
            result = job.result(timeout=120)
            print("Job status:", getattr(result, "status", "No status attribute"))
            counts = result.get_counts()
            print("Counts:", counts)
        except Exception as e:
            print("Exception during job execution:", e)
            return [], {}
        if not counts:
            print("No counts returned from simulator.")
            return [], {}
       # job_result = job.result()
        # print("Job final status:", job_result.status)
        # print(job_result.to_dict())

        #counts = job_result.get_counts()
        most_probable = max(counts, key=counts.get)
        selected = [int(bit) for bit in most_probable[::-1]]
        selected_tickers = [tickers[i] for i, bit in enumerate(selected) if bit == 1]

        return selected_tickers

    # on real QC functions:

    # new functions for quantum optimization
    def qaoa_expectation(self, params, Q_dict, n_assets, backend):
        """Calculates QAOA function expectation value

        Args:
            params (string): gamma and beta parameters of QAOA
            Q_dict (dictionary): dictionary form of QUBO problem
            n_assets (integer): total number of assets in the market
            backend (string): backend being used to run the circuit (simulator or real quantum computer)

        Returns:
            expectation (float): total expectation value of the portfolio
        """
        gamma, beta = params  # defining gamma and beta as parameters
        qc = QuantumCircuit(n_assets)  # run the circuit as before
        qc.h(range(n_assets))
        for (i, j), coeff in Q_dict.items():
            if i == j:
                qc.rz(2 * gamma * coeff, i)
            else:
                #qc.append(RZZGate(2 * gamma * coeff), [i, j])
                qc.cx(i, j)
                qc.rz(2 * gamma * coeff, j)
                qc.cx(i, j)
        for i in range(n_assets):
            qc.rx(2 * beta, i)
        qc.measure_all()
        job = backend.run(qc)
        try:
            result = job.result(timeout=120)
            if result.status != "COMPLETED":
                print("Job failed")
                return np.inf  # infinity = very large number to make sure we have high cost returned
            counts = result.get_counts()
        except Exception:
            print("Error during job execution")
            return np.inf
        expectation = 0
        for (
            bitstring,
            count,
        ) in (
            counts.items()
        ):  # find the expectation value (energy) of this particular portfolio
            z = [int(b) for b in bitstring[::-1]]
            cost = sum(
                Q_dict.get((i, j), 0) * z[i] * z[j]
                for i in range(n_assets)
                for j in range(n_assets)
            )
            expectation += cost * (
                count / sum(counts.values())
            )  # cost times the probability of a specific bitstring = expectation value
        return -expectation

    def run_cobyla(self, Q_dict, n_assets, backend):
        """Runs COBYLA optimizer to update gamma and beta parameters

        Args:
            Q_dict (dictionary): dictionary form of QUBO problem
            n_assets (integer): total number of assets in the market
            backend (string): backend being used to run the circuit (simulator or real quantum computer)

        Returns:
            gamma_opt (float): optimal gamma value for the circuit
            beta_opt (float): optimal beta value for the circuit
        """
        cobyla = COBYLA()
        cobyla.set_options(maxiter=250, tol=1e-3)
        expectation_fn = partial(
            self.qaoa_expectation, Q_dict=Q_dict, n_assets=n_assets, backend=backend
        )
        initial_params = [np.pi / 4, np.pi / 4]
        opt_result = cobyla.minimize(expectation_fn, initial_params)

        x_opt = opt_result.x

        gamma_opt, beta_opt = float(x_opt[0]), float(x_opt[1])
        return gamma_opt, beta_opt

    def get_selected_tickers(self, gamma, beta, Q_dict, n_assets, backend, tickers):
        """Runs optimizer circuit with optimised gamma and beta parameters.

        Args:
            gamma (float): Gamma parameter of QAOA
            beta (float): Beta parameter of QAOA
            Q_dict (dictionary): dictionary form of QUBO problem
            n_assets (integer): total number of assets in the market
            backend (string): backend being used to run the circuit (simulator or real quantum computer)
            tickers (array): ticker name of asset

        Returns:
           selected_tickers (array): array indicating which assets are included (indicated by a 1) or not included (indicated by a 0) in the portfolio
           counts (array): number of times a portfolio was determined to be the most optimal
        """
        qc = QuantumCircuit(n_assets)
        qc.h(range(n_assets))

        for (i, j), coeff in Q_dict.items():
            if i == j:
                qc.rz(2 * gamma * coeff, i)
            else:
                #qc.append(RZZGate(2 * gamma * coeff), [i, j])
                qc.cx(i, j)
                qc.rz(2 * gamma * coeff, j)
                qc.cx(i, j)

        for i in range(n_assets):
            qc.rx(2 * beta, i)

        qc.measure_all()
        job = backend.run(qc)
        result = job.result()

        if result.status != "COMPLETED":
            print(f"Job failed or incomplete. Status: {result.status}")
            return [], {}

        counts = result.get_counts()
        most_probable = max(counts, key=counts.get)
        z = [int(b) for b in most_probable[::-1]]
        selected_tickers = [tickers[i] for i, bit in enumerate(z) if bit == 1]
        return selected_tickers, counts


# problem: yfinance does not list a public API with all stocks,you must indicate a specific ticker. S&P 500 have been used as a proxy for all stocks.
class DataProcessing:
    def __init__(self, data):
        self.data = data

    def import_data(self, data):
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        table = pd.read_html(url)[0]
        tickers = table["Symbol"].tolist()
        tickers = [ticker.replace(".", "-") for ticker in tickers]
        tickers = tickers[:100]
        # tickers = np.random.choice(tickers, size=100, replace=False) #select 100 random tickers

        data = yf.download(
            tickers, period="1y", interval="1d", group_by="ticker", progress=False
        )
        latest_prices = {
            ticker: data[ticker]["Close"].dropna().iloc[-1] for ticker in tickers
        }

        close_prices = pd.concat([data[ticker]["Close"] for ticker in tickers], axis=1)
        close_prices.columns = tickers
        returns = close_prices.pct_change().dropna()

        mean_daily_return = returns.mean()
        expected_annual_return = mean_daily_return * 252  # 252 trading days in a year

        # covariance matrix is not calculated manually, using pandas library to calculate it directly from the returns dataframe.
        covariance_matrix = returns.cov() * 252  # annualized covariance matrix
        return tickers, expected_annual_return, covariance_matrix, table

    def sklearns_preprocessing(data):
        """K-Means clustering pre-processing to include more data in the whole optimization process. Available Quantum Computers have a limited number of qubits and the code seems to only run with about 10 assets. Rather than considering a very small market of 10 total assets, K-Means clustering allows for a much larger dataset to be considered and reduced to 10 representative assets.

        Args:
            data (array): all the relevant data relating to the assets

        Returns:
            expected_annual_returns (array): expected annual returns for the 10 selected representative assets
            covariance_matrix (array): covariance and variance values for the 10 selected representative assets
            latest_prices (array): most recent prices of stock for the 10 selected representative assets
        """
        tickers = data.columns.get_level_values(1).unique()
        close_prices = data.xs('Close', axis=1, level=1)
        close_prices.columns = data.columns.get_level_values(0).unique()
        returns = close_prices.pct_change().dropna()
        mean_daily_return = returns.mean()
        expected_annual_return = mean_daily_return * 252
        scaler = StandardScaler()
        volatility = returns.std() * np.sqrt(252)
        features = pd.DataFrame(
            {
                "Expected Return": expected_annual_return,
                "Volatility": volatility,
            }
        )
        scaled_features = scaler.fit_transform(features)
        kmeans = KMeans(n_clusters=10)
        kmeans.fit(scaled_features)
        features["Cluster"] = kmeans.labels_

        plt.figure
        plt.scatter(
            features["Expected Return"],
            features["Volatility"],
            c=features["Cluster"],
            cmap="tab10",
            s=100,
            edgecolor="k",
        )
        for ticker in features.index:
            x = features.loc[ticker, "Expected Return"]
            y = features.loc[ticker, "Volatility"]
            plt.text(x, y, ticker, fontsize=6, ha="right", va="bottom")
        plt.xlabel("Expected Annual Return")
        plt.ylabel("Annual Volatility")
        plt.title("KMeans Clustering of Assets")
        plt.show()

        representative_tickers = []

        for cluster_label in np.unique(
            kmeans.labels_
        ):  # iterate over each cluster 0 to 9.
            cluster_data = features[
                features["Cluster"] == cluster_label
            ]  # fetch data points in that specific cluster
            cluster_features = cluster_data[
                ["Expected Return", "Volatility"]
            ].values  # fetch the values for each feature
            center = kmeans.cluster_centers_[cluster_label].reshape(
                1, -1
            )  # calculate the centre of the cluster
            distances = pairwise_distances(
                cluster_features, center
            )  # compute distances of each point to the centre
            min_index = np.argmin(distances)  # find the smallest distance in distances
            representative_ticker = cluster_data.index[
                min_index
            ]  # ticker with smallest distance to centre is taken as the reprensative
            representative_tickers.append(representative_ticker)

        print(representative_tickers)
        tickers = representative_tickers

        # need to recalcualte everything to pass the right stuff to next code
        tickers = representative_tickers
        close_prices = close_prices[tickers]
        returns = close_prices.pct_change().dropna()
        mean_daily_return = returns.mean()
        expected_annual_return = mean_daily_return * 252
        covariance_matrix = returns.cov() * 252
        latest_prices = close_prices.iloc[-1].to_dict()
        return expected_annual_return, covariance_matrix, latest_prices


class UserPreferences:
    def __init__(self, num_assets):
        self.num_assets = num_assets

    def get_user_preferences(num_assets):
        library_or_manual_input = input("Library or Manual Implementation?")

        if library_or_manual_input == "Library" or library_or_manual_input == "library":
            library_or_manual = True
        elif library_or_manual_input == "Manual" or library_or_manual_input == "manual":
            library_or_manual = False
        else:
            raise ValueError("Invalid input: please enter library or manual.")

        classical_or_quantum_input = input("Classical or Quantum Implementation?")

        if (
            classical_or_quantum_input == "Classical"
            or classical_or_quantum_input == "classical"
        ):
            classical_or_quantum = True
            sim_or_qpu = None
        elif (
            classical_or_quantum_input == "Quantum"
            or classical_or_quantum_input == "quantum"
        ):
            classical_or_quantum = False
            sim_or_qpu_input = input("Simulator or Quantum Computer?")

            if sim_or_qpu_input == "Simulator" or sim_or_qpu_input == "simulator" or sim_or_qpu_input == "sim":
                sim_or_qpu = True 
            elif sim_or_qpu_input == "Quantum Computer" or sim_or_qpu_input == "quantum computer" or sim_or_qpu_input == "qc":
                sim_or_qpu = False
            else:
                raise ValueError("Invalid input: please enter simulator or quantum computer")
        else:
            raise ValueError("Invalid input: please enter classical or quantum.")
        

        budget = int(input("Enter number of assets to invest in: "))
        if budget <= 0 or budget > num_assets:
            raise ValueError("Budget must be a positive integer.")

        risk_factor = float(input("Enter risk factor (0-1): "))
        if risk_factor < 0 or risk_factor > 1:
            raise ValueError("Risk factor must be between 0 and 1.")
        return library_or_manual, classical_or_quantum, sim_or_qpu, budget, risk_factor


# =============================================================================== calling and running the code
if __name__ == "__main__":

    # Data import and preprocessing
    # Instantiate DataProcessing and import data
    processor = DataProcessing(None)
    tickers, expected_annual_return, covariance_matrix, table = processor.import_data(None)

    # Downloaded data for all tickers
    data = yf.download(
        tickers, period="1y", interval="1d", group_by="ticker", progress=False
    )
    # Preprocess with sklearns_preprocessing to reduce to 10 representative assets
    expected_annual_return, covariance_matrix, latest_prices = DataProcessing.sklearns_preprocessing(data)

    # Get user preferences
    tickers = list(expected_annual_return.index)
    num_assets = len(tickers)
    library_or_manual, classical_or_quantum, sim_or_qpu, budget, risk_factor = (
        UserPreferences.get_user_preferences(num_assets)
    )

    # Instantiate optimizer classes
    classical_optimizer = ClassicalMethod(
        budget=budget,
        risk=risk_factor,
        expected_returns=expected_annual_return,
        tickers=tickers,
        covariance_matrix=covariance_matrix,
    )
    quantum_optimizer = QuantumMethod()

    # Select backend for quantum runs
    if library_or_manual and classical_or_quantum:
        qp = classical_optimizer.library_qubo_conversion(
            mu=expected_annual_return.values,
            sigma=covariance_matrix.values,
            risk_factor=risk_factor,
            budget=budget,
        )
        result = classical_optimizer.classical_library_optimise_portfolio(qp=qp)
        selected = np.array(result.x) == 1
        selected_tickers = np.array(tickers)[selected]

    elif library_or_manual and not classical_or_quantum:
        qp = classical_optimizer.library_qubo_conversion(
            mu=expected_annual_return.values,
            sigma=covariance_matrix.values,
            risk_factor=risk_factor,
            budget=budget,
        )
        if sim_or_qpu:  # Simulator
            sim_backend = AerSimulator()
            sampler = Sampler()
        else:  # Quantum Computer
            sampler = Sampler(backend)
        qaoa_mes = QAOA(sampler=sampler, optimizer=COBYLA(), reps=1)
        qaoa = MinimumEigenOptimizer(qaoa_mes)
        result = quantum_optimizer.quantum_library_optimise_portfolio(qp=qp, sampler=Sampler())
        selected = np.array(result.x) == 1
        selected_tickers = np.array(tickers)[selected]

    elif not library_or_manual and classical_or_quantum:
        Q = classical_optimizer.manual_qubo_conversion()
        selected_tickers = classical_optimizer.classical_manual_optimise_portfolio(
            Q_dict=Q, tickers=tickers, n_assets=len(tickers)
        )

    elif not library_or_manual and not classical_or_quantum:
        Q = classical_optimizer.manual_qubo_conversion()
        gamma_opt, beta_opt = quantum_optimizer.run_cobyla(
            Q_dict=Q, n_assets=len(tickers), backend=backend
        )
        selected_tickers, counts = quantum_optimizer.get_selected_tickers(
            gamma=gamma_opt,
            beta=beta_opt,
            Q_dict=Q,
            n_assets=len(tickers),
            backend=backend,
            tickers=tickers,
        )

    helper = HelperFunctions(
        selected_tickers= selected_tickers,
        expected_returns=expected_annual_return,
        covariance_matrix=covariance_matrix,
        latest_prices=latest_prices,
        table=table,
    )


    summary = helper.evaluate_portfolio()



    # maybe add this to the evaluate portfolio method? so that its within the summary thing
    print("\n -----------------Portfolio Summary:-----------------\n")
    if library_or_manual == True:
        print("Implementation: Library")
    else:
        print("Implementation: Manual")
    if classical_or_quantum == True:
        print("Implementation: Classical")
    else:
        print("Implementation: Quantum")
    if sim_or_qpu == True:
        print("Implementation: Simulator")
    else:
        print("Implementation: Quantum Computer")
    print(f"Budget:{budget} asset(s)")
    print(f"Risk:{risk_factor}")
    for key, value in summary.items():
        print(f"{key}: {value}")

def prepared_data():
    import yfinance as yf
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)[0]
    tickers = table["Symbol"].tolist()
    tickers = [ticker.replace(".", "-") for ticker in tickers[:10]]

    data = yf.download(
        tickers, period="1y", interval="1d", group_by="ticker", progress=False
    )
    latest_prices = {
        ticker: data[ticker]["Close"].dropna().iloc[-1] for ticker in tickers
    }
    close_prices = pd.concat([data[ticker]["Close"] for ticker in tickers], axis=1)
    close_prices.columns = tickers
    returns = close_prices.pct_change().dropna()
    expected_annual_return = returns.mean() * 252
    covariance_matrix = returns.cov() * 252

    scaler = StandardScaler()
    volatility = returns.std() * np.sqrt(252)
    features = pd.DataFrame(
        {
            "Expected Return": expected_annual_return,
            "Volatility": volatility,
        }
    )
    scaled_features = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=10).fit(scaled_features)
    features["Cluster"] = kmeans.labels_

    representative_tickers = []
    for cluster_label in np.unique(kmeans.labels_):
        cluster_data = features[features["Cluster"] == cluster_label]
        center = kmeans.cluster_centers_[cluster_label].reshape(1, -1)
        distances = np.linalg.norm(
            cluster_data[["Expected Return", "Volatility"]].values - center, axis=1
        )
        representative_ticker = cluster_data.index[np.argmin(distances)]
        representative_tickers.append(representative_ticker)

    tickers = representative_tickers
    close_prices = pd.concat([data[ticker]["Close"] for ticker in tickers], axis=1)
    close_prices.columns = tickers
    returns = close_prices.pct_change().dropna()
    expected_annual_return = returns.mean() * 252
    covariance_matrix = returns.cov() * 252
    latest_prices = {
        ticker: data[ticker]["Close"].dropna().iloc[-1] for ticker in tickers
    }

    return {
        "tickers": tickers,
        "expected_returns": expected_annual_return,
        "covariance_matrix": covariance_matrix,
        "latest_prices": latest_prices,
        "budget": 2,
        "risk": 0.5,
    }

