# Architecture

The architecture for portfolio optimization using QAOA is indicated here. Use case, class and sequence diagrams are included.

## Use Case Diagram

<p align="center">
<img width="593" height="350" alt="UseCaseDiagram" src="https://github.com/user-attachments/assets/8974c557-184b-483c-ade5-2690bef92a45" />
</p>

This diagram gives an overview of which actors interact with this system and how they do so. Firstly, a user wishing to carry out portfolio optimization using this program must specify their preferences. These include whether to use a library or manual implementation, a classical or quantum implementation, the number of assets they wish to invest in and the level of risk they would like to take on. A financial data provider is required to determine expected returns and covariances of the assets, which is currently set as Yahoo Finance. A quantum backend, being either a simulator or real quantum computer is required when running the quantum version of the program, where the quantum circuits are run.

## Class Diagram

<p align="center">
<img width="3840" height="732" alt="ClassDiagram" src="https://github.com/user-attachments/assets/b8f55819-2a20-4bd0-b854-6891ae8920cc" />
</p>

The class diagram displays the different classes of functions used in this program. The two main classes are ClassicalMethod and QuantumMethod, which include functions to run each method respectively. The ClassicalMethod class includes a manual and library version of a conversion of the optimziation problem to a QUBO format, necessary for subsequent optimization. It also incldues a manual and library implementation of the optimization. The QuantumMethod class includes a library implementation function named quantum_library_optimise_portfolio. The manual implementation was carried out in two different ways, one being the quantum_manual_optimise_portfolio, which is suited to simulators but not real quantum computers. The second manual implementation is made up of the remaining three functions (qaoa_expectation, run_cobyla, get_selected_tickers) which is adapted to being run on real quantum computers, as well as simulators.

The HelperFunctions class is a catchall class for miscellaneous functions used by both methods. It includes the evaluate_portfolio function which returns a user-friendly output of the final portfolio optimization, including an array of specific selected assets, cost of the initial investment, expected returns after one year and risk level of the investment.

The DataProcessing class includes the import_data function which fetches financial data from the financial data provider. It also includes the function sklearns_preprocessing which classically pre-processes the data using K-Means clustering. This was deemed necessary to be able to include more data in the overall optimization as it it possible to pass only about 10 assets to the quantum part of the code, mainly due to current hardware limitations.

The UserPreferences class takes in all the user preferences necessary to run the optimization and return a personalised portfolio to the user.

## Sequence Diagram

<p align="center">
<img width="3268" height="3840" alt="SequenceDiagram" src="https://github.com/user-attachments/assets/a05eefc3-3401-4935-9687-741d023c8a31" />
</p>

The overall flow of the program begins with data being imported from the financial provider when the code is run. Next, user preferences are collected and stored. Depending on the user's preferences, either the classical or quantum implementation is run using either a library or manual implementation. For Library & Classical, Library & Quantum and Manual & Classical; the process is almost identical with the data being converted into a suitable form for the optimizer, the optimization being carried out and the result being returned as a user-friendly summary. For the Manual & Quantum, the problem is converted to a QUBO form, then the COBYLA optimizer optimizes the parameters gamma and beta used in the quantum circuit, then the optimizer is run with these updated parameters and the result is returned to the user. 
