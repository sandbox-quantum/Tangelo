# Copyright 2021 Good Chemistry Company.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Python wrappers facilitating quantum experiment submission, monitoring and
post-processing, through QEMIST Cloud.

Users are expected to set the environment variables QEMIST_AUTH_TOKEN and
QEMIST_PROJECT_ID with values retrieved from their QEMIST Cloud dashboard.
"""

try:
    import qemist_client as qclient
except ModuleNotFoundError:
    print("qemist_client python package not found (optional dependency for hardware experiment submission)")


def job_submit(circuit, n_shots, backend):
    """Job submission to run a circuit on quantum hardware.

    Args:
        circuit: a quantum circuit in the abstract format.
        n_shots (int): the number of shots.
        backend (str): the identifier string for the desired backend.

    Returns:
        int: A problem handle / job ID that can be used to retrieve the result
            or cancel the problem.
    """

    # Serialize circuit data
    circuit_data = circuit.serialize()

    # Build option dictionary
    job_options = {'shots': n_shots, 'backend': backend}

    # Submit the problem
    qemist_cloud_job_id = qclient.util.solve_quantum_circuits_async(serialized_fragment=circuit_data,
                                                                    serialized_solver=job_options)[0]
    return qemist_cloud_job_id


def job_status(qemist_cloud_job_id):
    """Returns the current status of the problem, as a string. Possible values:
    ready, in_progress, complete, cancelled.

    Args:
        qemist_cloud_job_id (int): problem handle / job identifier.

    Returns:
        str: current status of the problem, as a string.
    """
    res = qclient.util.get_problem_status(qemist_cloud_job_id)

    return res


def job_cancel(qemist_cloud_job_id):
    """Cancels the job matching the input job id, if done in time before it
    starts.

    Args:
        qemist_cloud_job_id (int): problem handle / job identifier.

    Returns:
        dict: cancelled problems / subproblems.
    """
    res = qclient.util.cancel_problems(qemist_cloud_job_id)
    # TODO: If res is coming out as an error code, we should raise an error

    return res


def job_result(qemist_cloud_job_id):
    """Blocks until the job results are available. Returns a tuple containing
    the histogram of frequencies, and also the more in-depth raw data from the
    cloud services provider as a nested dictionary

    Args:
        qemist_cloud_job_id (int): problem handle / job identifier.

    Returns:
        dict: Histogram of measurement frequencies.
        dict: The cloud provider raw data.
    """

    try:
        qclient.util.monitor_problem_status(problem_handle=qemist_cloud_job_id, verbose=False)

    except KeyboardInterrupt:
        print(f"\nYour problem is still running with id {qemist_cloud_job_id}.\n")
        command = input("Type 'cancel' and return to cancel your problem."
                        "Type anything else to disconnect but keep the problem running.\n")
        if command.lower() == "cancel":
            ret = job_cancel(qemist_cloud_job_id)
            print("Problem cancelled.", ret)
        else:
            print(f"Reconnect and block until the problem is complete with "
                  f"qemist_client.util.monitor_problem_status({qemist_cloud_job_id}).\n\n")
        raise

    except Exception:
        print(f"\n\nYour problem is still running with handle {qemist_cloud_job_id}.\n"
              f"Cancel the problem with qemist_client.util.cancel_problems({qemist_cloud_job_id}).\n"
              f"Reconnect and block until the problem is complete with qemist_client.util.monitor_problem_status({qemist_cloud_job_id}).\n\n")
        raise

    # Once a result is available, retrieve it.
    # If the util module is not found earlier, an error has been raised.
    output = qclient.util.get_quantum_results(problem_handle=qemist_cloud_job_id)[qemist_cloud_job_id]

    # Amazon Braket: parsing of output
    freqs = output['result']['results']['measurement_probabilities']
    raw_data = output

    return freqs, raw_data


def job_estimate(circuit, n_shots, backend=None):
    """Returns an estimate of the cost of running an experiment, for a specified backend
    or all backends available. Some service providers care about the
    complexity / structure of the input quantum circuit, some do not.

    The backend identifier strings that a user can provide as argument can be obtained
    by calling this function without specifying a backend. They appear as keys in
    the returned dictionary. These strings may change with time, as we adjust to the
    growing cloud quantum offer (services and devices).

    Args:
        circuit (Circuit): the abstract circuit to be run on the target device.
        n_shots (int): number of shots in the expriment.
        backend (str): the identifier string for the desired backend.

    Returns:
        dict: Returns dict of prices in USD. If backend is not None, dictionary
        contains the cost for running the desired job. If backend is None,
        returns dictionary of prices for all supported backends.
    """

    # Serialize circuit data
    circuit_data = circuit.serialize()

    # Build option dictionary
    job_options = {'shots': n_shots}
    if backend:
        job_options['backend'] = backend

    price_estimate = qclient.util.check_qpu_cost(circuit_data, job_options)

    return price_estimate
