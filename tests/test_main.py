from qiskit import Aer 
from qiskit.utils import QuantumInstance

import sys
import os
import importlib.util

import time
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
spec = importlib.util.spec_from_file_location("main", os.path.join("project_code", "main.py"))
main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main)

ClassicalMethod = main.ClassicalMethod
QuantumMethod = main.QuantumMethod
prepared_data = main.prepared_data



def benchmark_classical_manual(
    expected_returns, covariance_matrix, max_assets=15, budget=3, risk_factor=0.5
):
    """
    Benchmarks classical manual optimization for different asset counts.

    Parameters:
    - expected_returns: pd.Series of expected returns indexed by ticker
    - covariance_matrix: pd.DataFrame covariance matrix of assets
    - max_assets: max number of assets to test
    - budget: number of assets to invest in
    - risk_factor: risk parameter (0-1)

    Returns:
    - asset_counts: list of asset counts benchmarked
    - times: list of execution times corresponding to asset counts
    """
    times = []
    asset_counts = list(range(4, max_assets + 1))

    for num_assets in asset_counts:
        subset_tickers = list(expected_returns.index[:num_assets])
        subset_returns = expected_returns[subset_tickers]
        subset_cov = covariance_matrix.loc[subset_tickers, subset_tickers]

        classical_optimizer = ClassicalMethod(
            budget=min(budget, num_assets),
            risk=risk_factor,
            expected_returns=subset_returns,
            tickers=subset_tickers,
            covariance_matrix=subset_cov,
        )

        Q_dict = classical_optimizer.manual_qubo_conversion()

        try:
            start = time.time()
            _ = classical_optimizer.classical_manual_optimise_portfolio(
                Q_dict=Q_dict,
                tickers=subset_tickers,
                n_assets=len(subset_tickers),  
            )
            end = time.time()
            times.append(end - start)
            print(f"Assets: {num_assets}, Time: {end - start:.4f}s")
        except Exception as e:
            print(f"Error with {num_assets} assets: {e}")
            times.append(None)

    return asset_counts, times

def benchmark_classical_library(expected_returns, covariance_matrix, max_assets=10, budget=3, risk_factor=0.5):
    times = []
    asset_counts = list(range(4, max_assets + 1))

    for num_assets in asset_counts:
        subset_tickers = list(expected_returns.index[:num_assets])
        subset_returns = expected_returns[subset_tickers]
        subset_cov = covariance_matrix.loc[subset_tickers, subset_tickers]

        classical_optimizer = ClassicalMethod(
            budget=min(budget, num_assets),
            risk=risk_factor,
            expected_returns=subset_returns,
            tickers=subset_tickers,
            covariance_matrix=subset_cov,
        )
        Q_dict = classical_optimizer.manual_qubo_conversion()

        
        qp = classical_optimizer.library_qubo_conversion(
    mu=subset_returns.values,              
    sigma=subset_cov.values,               
    risk_factor=risk_factor,
    budget=min(budget, num_assets)
)

        try:
            start = time.time()
            result = classical_optimizer.classical_library_optimise_portfolio(qp)
            end = time.time()
            times.append(end - start)
            print(f"[Classical Library] Assets: {num_assets}, Time: {end - start:.4f}s")
        except Exception as e:
            print(f"[Classical Library] Error with {num_assets} assets: {e}")
            times.append(None)

    return asset_counts, times


def benchmark_quantum_library(expected_returns, covariance_matrix, max_assets=10, budget=3, risk_factor=0.5):
    times = []
    asset_counts = list(range(4, max_assets + 1))
    backend = Aer.get_backend('aer_simulator')
    quantum_instance = QuantumInstance(backend)

    for num_assets in asset_counts:
        subset_tickers = list(expected_returns.index[:num_assets])
        subset_returns = expected_returns[subset_tickers]
        subset_cov = covariance_matrix.loc[subset_tickers, subset_tickers]

        quantum_optimizer = QuantumMethod()
        classical_optimizer = ClassicalMethod(
            budget=min(budget, num_assets),
            risk=risk_factor,
            expected_returns=subset_returns,
            tickers=subset_tickers,
            covariance_matrix=subset_cov,
        )
        Q_dict = classical_optimizer.manual_qubo_conversion()

        qp = classical_optimizer.library_qubo_conversion(mu=subset_returns.values,  sigma=subset_cov.values,risk_factor=risk_factor,budget=min(budget, num_assets))  

        try:
            print(f"Running quantum_library_optimise_portfolio for {num_assets} assets...")
            start = time.time()
            result = quantum_optimizer.quantum_library_optimise_portfolio(
                qp=qp,
                sampler=quantum_instance,
            )
            end = time.time()
            print(f"Result status: {result.status}, Optimal value: {result.fval}")
            times.append(end - start)
            print(f"[Quantum Library] Assets: {num_assets}, Time: {end - start:.4f}s")
        except Exception as e:
            print(f"[Quantum Library] Error with {num_assets} assets: {e}")
            times.append(None)


    return asset_counts, times



def benchmark_quantum_manual(
    expected_returns, covariance_matrix, max_assets=10, budget=3, risk_factor=0.5
):
    """
    Benchmarks quantum manual optimization (e.g. using QAOA + COBYLA) for different asset counts.

    Returns:
    - asset_counts: list of asset counts tested
    - times: list of execution times
    """
    from project_code.main import QuantumMethod 
    times = []
    asset_counts = list(range(4, max_assets + 1))
    backend = Aer.get_backend('aer_simulator')

    for num_assets in asset_counts:
        subset_tickers = list(expected_returns.index[:num_assets])
        subset_returns = expected_returns[subset_tickers]
        subset_cov = covariance_matrix.loc[subset_tickers, subset_tickers]

        quantum_optimizer = QuantumMethod()

        classical_optimizer = ClassicalMethod(
            budget=min(budget, num_assets),
            risk=risk_factor,
            expected_returns=subset_returns,
            tickers=subset_tickers,
            covariance_matrix=subset_cov,
        )

        Q_dict = classical_optimizer.manual_qubo_conversion()

        try:
            start = time.time()

            gamma_opt, beta_opt = quantum_optimizer.run_cobyla(
                Q_dict=Q_dict,
                n_assets=num_assets,
                backend=backend,   
            )
            _ = quantum_optimizer.get_selected_tickers(
                gamma=gamma_opt,
                beta=beta_opt,
                Q_dict=Q_dict,
                n_assets=num_assets,
                backend=backend,
                tickers=subset_tickers,
            )
            end = time.time()
            times.append(end - start)
            print(f"[Quantum Manual] Assets: {num_assets}, Time: {end - start:.4f}s")
        except Exception as e:
            print(f"[Quantum Manual] Error with {num_assets} assets: {e}")
            times.append(None)

    return asset_counts, times



def run_benchmark():
    data = prepared_data()

    asset_counts_classical, times_classical = benchmark_classical_manual(
        data["expected_returns"],
        data["covariance_matrix"],
        max_assets=10,
        budget=data["budget"],
        risk_factor=data["risk"],
    )

    asset_counts_quantum, times_quantum = benchmark_quantum_manual(
        data["expected_returns"],
        data["covariance_matrix"],
        max_assets=10,
        budget=data["budget"],
        risk_factor=data["risk"],
    )

    asset_counts_classical_lib, times_classical_lib = benchmark_classical_library(
        data["expected_returns"], data["covariance_matrix"], max_assets=10, budget=data["budget"], risk_factor=data["risk"]
    )
    asset_counts_quantum_lib, times_quantum_lib = benchmark_quantum_library(
        data["expected_returns"], data["covariance_matrix"], max_assets=10, budget=data["budget"], risk_factor=data["risk"]
    )
    print("Classical Manual:", times_classical)
    print("Classical Library:", times_classical_lib)
    print("Quantum Manual:", times_quantum)
    print("Quantum Library:", times_quantum_lib)


    #uncomment plots according to your preferences of which graphs you want to see
    
    plt.figure(figsize=(10, 6))
    #plt.plot(asset_counts_classical, times_classical, marker="o", label="Classical (Manual)")
    #plt.plot(asset_counts_quantum, times_quantum, marker="x", label="Quantum (Manual)")
    #plt.plot(asset_counts_classical_lib, times_classical_lib, marker='o', label="Classical Library")
    plt.plot(asset_counts_quantum_lib, times_quantum_lib, marker='x', label="Quantum Library")
    plt.xlabel("Number of Assets")
    plt.ylabel("Execution Time (s)")
    plt.title("Benchmark: Quantum and Classical with Manual and Library Implementations)")
    plt.grid(True)
    plt.legend()
    plt.show()


def test_run_benchmark():
    print(">>> Starting run_benchmark()")
    run_benchmark()


if __name__ == "__main__":
    test_run_benchmark()