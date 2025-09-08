# Classical-Quantum Hybrid Portfolio Optimization using the Quantum Approximate Optimization Algorithm (QAOA)

This project was completed during June-August of 2025 as part of my internship with PALO IT. Many thanks to PALO IT for this opportunity and their support, especially my supervisor @mparramont.

This project leverages both classical and quantum computing to solve a simple portfolio optimization problem. The problem involves selecting a subset of assets from a market to form a portfolio. The problem is formulated as Quadratic Unconstrained Binary Optimization problem, with assets either selected or unselected, with equal weighting for all selected assets in the final portfolio.

## Dependencies

This project requires the following Python, Qiskit and Azure libraries:

- `numpy`, `matplotlib`, `pandas` - for numerical calculations and data handling
- `yfinance` - to fetch historical financial data
- `scikit-learn` - for classical data pre-processing using machine learning
- `azure-quantum` - for running on real quantum hardware
- `qiskit` and its submodules, `qiskit-finance`, `qiskit-optimization`, `qiskit-algorithms` - for portfolio optimization

For more details regarding dependencies, please consult requirements.txt file.

To install these dependencies, please follow the steps below.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/GLOBAL-PALO-IT/quantum-internship/tree/main/docs
    ```
2. Navigate to the project directory:
    ```sh
    cd quantum-internship
    ```
3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

To run the simulation, use the following command:
```sh
python main2.py
```
## Example Output

### Quantum Simulator Sample Run

```
 -----------------Portfolio Summary:-----------------

Implementation: Library
Implementation: Quantum
Implementation: Simulator
Budget:2 asset(s)
Risk:0.3
Selected Assets: ['Capital One', 'Axon Enterprise']
Total Cost: 975.60 USD
Expected Annual Return ± Risk (Volatility): 70.03 ± 15.30 %
```
## Technical Details

This portfolio optimization is carried out by encoding the problem as a QUBO and applying QAOA.

### QAOA

The Quantum Appriximate Optimization Algorithm, or QAOA works in the following steps:

1. Initialise the superposition of all the possible states. This is carried out by beginning with all the qubits in the $|0 \rangle $ state, then applying the Hadamard gate to each qubit. This creates a uniform superposition of all possible combinations.

2. Next, a cost Hamiltonian and a mixer Hamiltonian are applied alternately $n$ times. 

    2.1 The cost Hamiltonian for this problem takes the form shown below:

    $C(\textbf{x}) = -\Sigma_i \mu_i x_i + \rho \Sigma_{i,j} Q_{ij} x_i x_j + P(\Sigma_i x_i - B)^2, $

    where $\textbf{x}$ is a vector, consisting of $a$ entries, where $a$ is the total number of assets in the market. Each entry takes on a value of either 1 or 0 depending on whether that particular is selected or not, respectively. 

    QAOA seeks to minimise this function $C$, or in other words, find which $\textbf{x}$ provides the smallest possible value.

    The first term $-\Sigma_i \mu_i x_i$ is the total expected returns, where $\mu_i$ is the expected returns of asset $i$ and $x_i$ is the binary indicator taking on a value of either 0 or 1. 

    The second term $\rho \Sigma_{i,j} Q_{ij} x_i x_j$ is the risk factor, where $\rho$ is a risk factor chosen by a user taking on a value anywhere between 0 and 1. $Q$ is the covariance matrix for the assets in the market, with $Q_{ij}$ being the entry in that matrix corresponding to the covariance between assets $i$ and $j$.

    The third term encodes an additional constraint not typically included in most QUBO problems. This constraint ensures that the number of assets that user desires to invest in is respected. This number of assets is denoted $B$. $P$ is an arbitrary large number to ensure that the penalty term is large enough to successfully enforce this constraint.

    2.2 The mixer Hamiltonian's purpose is to randomly switch certain qubits' states to ensure that the entire solution space is reasonanbly explored. The mixer Hamiltonian for this problem is shown below:

    $H_M = \Sigma_{i,j} (X_iX_j + Y_iY_j),$

    where $X$ and $Y$ are the X and Y quantum gates respectively.

3. The final state is measured, collapsing the quantum superposition.

4. The outputted solution is evaluated and the parameters within the cost Hamiltonian are tuned.

5. Steps 1-4 are iterated either until convergence is achieved or until the maximum number of iterations has been reached.

### K-Means Clustering Data Pre-Processing

The number of possible portfolios is $2^N$, where $N$ is the total number of assets in the market. Only about $2^10$ combinations can be checked on a classical computer in a reasonable time. In addition, in the quantum computing section of this code, each asset maps directly onto one qubit. As of writing, the available quantum computing hardware at Azure is about 25-100 qubits, creating another constraint on the size of the market that can be investigated using this code.

As a result of these constraints, the author esteems that this code would be best used as an optimizer within a particular niche or industry rather than a whole market portfolio optimizer.

As a temporary solution, K-Means clustering was implemented as a way of pre-processing the data. The assets are clustered into 10 clusters and a represntative asset is selected from each group. The representative is chosen as the one closest to the centre of the cluster. These representative assets are then passed to the rest of the code. An example output is displayed below.

<div align="center">
<img width="635" height="476" alt="Screenshot 2025-08-19 at 15 32 55" src="https://github.com/user-attachments/assets/ae2d8b87-0ed4-4596-bfaa-fe251274c4e6" />
</div>

### Assumptions & Limitations

- Expected returns and risk are evaluated based on historical data only
- Only 10 assets are considered after data pre-processing for adequate classical-quantum comparisons
- Only selection of assets is optimised, not weighting

## Troubleshooting

### Financial Data Import Error
```
ValueError: Input X contains NaN.
KMeans does not accept missing values encoded as NaN natively.
...

```
This error may occur if the connection to the Yahoo Financial Data provider failed. Simply re-run the script until the code runs.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.
