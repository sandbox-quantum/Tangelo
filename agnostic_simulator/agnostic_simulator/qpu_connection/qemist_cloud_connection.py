"""
    Python wrappers facilitating quantum experiment submission, monitoring and post-processing, through QEMIST Cloud.

    Users are expected to set the environment variables QEMIST_AUTH_TOKEN and QEMIST_PROJECT_ID with values
    retrieved from their QEMIST Cloud dashboard.
"""

try:
    from qemist_client import util
except ModuleNotFoundError:
    print("qemist_client python package not found (optional dependency for hardware experiment submission)")


def job_submit(circuit, n_shots, backend):
    """ Job submission to run a circuit on quantum hardware.

    Args:
        circuit: a quantum circuit in the abstract format
        n_shots (int): the number of shots
        backend (str): the identifier string for the desired backend

    Returns:
        job_id (int): A problem handle / job ID that can be used to retrieve the result or cancel the problem.
    """

    # Serialize circuit data
    circuit_data = circuit.serialize()

    # Build option dictionary
    job_options = {'shots': n_shots, 'backend': backend}

    # Submit the problem
    qemist_cloud_job_id = util.solve_quantum_circuits_async(serialized_fragment=circuit_data,
                                                            serialized_solver=job_options)[0]

    return qemist_cloud_job_id


def job_status(qemist_cloud_job_id):
    """ Returns the current status of the problem, as a string.
     Possible values: ready, in_progress, complete, cancelled.

        Args:
            qemist_cloud_job_id (int): problem handle / job identifier

        Returns:
            status (str): current status of the problem, as a string
    """
    return util.get_problem_status(qemist_cloud_job_id)


def job_cancel(qemist_cloud_job_id):
    """ Cancels the job matching the input job id, if done in time before it starts.
    Returns a list of cancelled problems and number of subproblems, if any.

        Args:
            qemist_cloud_job_id (int): problem handle / job identifier

        Returns:
            res (dict): cancelled problems / subproblems
    """

    res = util.cancel_problems(qemist_cloud_job_id)
    # TODO: If res is coming out as an error code, qSDK should raise an error

    return res


def job_result(qemist_cloud_job_id):
    """ Blocks until the job results are available.
    Returns a tuple containing the histogram of frequencies, and also the more in-depth raw data from
    the cloud services provider as a nested dictionary

        Args:
            qemist_cloud_job_id (int): problem handle / job identifier

        Returns:
            freqs (dict): histogram of measurement frequencies
            raw_data (dict): cloud provider raw data coming out as as nested dictionary
    """

    try:
        util.monitor_problem_status(problem_handle=qemist_cloud_job_id, verbose=False)

    except KeyboardInterrupt:
        print(f"\nYour problem is still running with id {qemist_cloud_job_id}.\n")
        command = input("Type 'cancel' and return to cancel your problem."
                        "Type anything else to disconnect but keep the problem running.\n")
        if command.lower() == "cancel":
            ret = job_cancel()
            print("Problem cancelled.", ret)
        else:
            print(f"Reconnect and block until the problem is complete with "
                  f"qemist_client.util.monitor_problem_status({qemist_cloud_job_id}).\n\n")

    except:
        print(f"\n\nYour problem is still running with handle {qemist_cloud_job_id}.\n"
              f"Cancel the problem with qemist_client.util.cancel_problems({qemist_cloud_job_id}).\n"
              f"Reconnect and block until the problem is complete with qemist_client.util.monitor_problem_status({qemist_cloud_job_id}).\n\n")
        raise

    # Once a result is available, retrieve it
    output = util.get_quantum_results(problem_handle=qemist_cloud_job_id)[qemist_cloud_job_id]

    # Amazon Braket: parsing of output
    freqs = output['result']['results']['measurement_probabilities']
    raw_data = output

    return freqs, raw_data


def job_estimate(circuit, n_shots):
    """
        Returns an estimate of the cost of running an experiment. Some service providers care about
        the complexity / structure of the input quantum circuit, some do not.

        Some backends may charge per minute (such as simulators), which is difficult to estimate
        and may be misleading. They are currently not included.

        Braket prices: https://aws.amazon.com/braket/pricing/
        Azure Quantum prices: TBD

    Args:
        circuit (Circuit): the abstract circuit to be run on the target device.
        n_shots (int): number of shots in the expriment.

    Returns:
        A dictionary of floating-point values (prices) in USD.

    """

    # Compute prices for each available backend (see provider formulas)
    price_estimate = dict()
    price_estimate['braket_ionq'] = 0.3 + 0.01 * n_shots
    price_estimate['braket_rigetti'] = 0.3 + 0.00035 * n_shots

    # Round up to a cent for readability
    price_estimate = {k: round(v, 2) for k, v in price_estimate.items()}

    return price_estimate
