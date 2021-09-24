"""
This script shows how to run a pre-existing Q# operation in a .qs file assumed to be generated by this package,
either targeting the local QDK simulator (option 2.a) below, or as a job submitted through Azure Quantum (option 2.b).

Use the local QDK to test that your code is working as expected, and get noiseless reference results
(warning: QDK is slow and shot-based. BackendBuddy can help with obtaining reference faster on other backends)
Submit a job to Azure Quantum to run on QPUs.

Qubits are numbered left-to-right in the results, in both cases (e.g q0q1q2...)

In the case of Azure Quantum, the following tutorials and documentations may be helpful to you:
https://1qbit-intra.atlassian.net/wiki/spaces/QSD/pages/1319600305/Submit+monitor+Hardware+Experiments+on+Azure+Quantum
https://github.com/MicrosoftDocs/quantum-docs-private/wiki/

You can elegantly generate your Q# circuits using qsdk.backendbuddy and submit them right away in one single script.
"""


# 1. Set parameters
# -----------------

# Are you running your Q# code of the local QDK, or targeting hardware available through azure quantum?
target_azure_quantum = False
# Set number of shots (needed by both QDK simulator or Azure Quantum)
n_shots = 10**4

# Local QDK run only: set number of qubits (check your .qs file)
n_qubits = 2

# Azure quantum only: operation name, job description, resource id of quantum workspace and compute target
# You may have to modify your operation name directly in the code below too
qsharp_operation_name = 'MyQsharpOperation'
job_name = f'{qsharp_operation_name}_{n_shots}'
resource_id = "/subscriptions/ba4c1f83-dd03-4899-a70b-9fb2fa8be7094/.../YourWorkspace"
target = "ionq.simulator"


# 2. Run on desired target
# ------------------------

# Local run using the QDK simulator: good for testing or noiseless reference results
if not target_azure_quantum:
    import qsharp
    qsharp.reload()
    from MyNamespace import EstimateFrequencies

    frequencies_list = EstimateFrequencies.simulate(nQubits=n_qubits, nShots=n_shots)
    frequencies = {bin(i).split('b')[-1]: frequencies_list[i] for i, freq in enumerate(frequencies_list)}
    frequencies = {("0"*(n_qubits-len(k))+k)[::-1]: v for k, v in frequencies.items() if v > 1e-5}
    print(f"Local QDK results:\n{frequencies}\n")

# Azure Quantum job submission
else:
    import qsharp
    import qsharp.azure
    from MyNamespace import MyQsharpOperation

    qsharp.azure.connect(resourceId=resource_id)
    qsharp.azure.target(target)

    # If your Q# operation takes no parameter (results need to be retrieved through Azure portal, under the relevant Quantum Workspace).
    job_id = qsharp.azure.submit(MyQsharpOperation, shots=n_shots, jobName=job_name)
    # If your Q# operation takes parameters (currently not available in qsdk.backendbuddy, but you can write your own Q#):
    # job_id = qsharp.azure.submit(MyQsharpOperation, param1=value1, param2=value2, ..., shots=n_shots, jobName=job_name')
