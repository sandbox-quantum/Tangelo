"""
    Python wrappers around Honeywell REST API, to facilitate job submission, result retrieval and post-processing

    Using Honeywell services require logins to their portal: https://um.qapi.honeywell.com/index.html
    Users are expected to set the environment variables HONEYWELL_EMAIL, HONEYWELL_PASSWORD with their credentials

    The portal above provides access to a dashboard, which is better suited for job monitoring experiments.
"""

from qemist_client import util


def job_submit(circuit, n_shots, backend):
    """ Runs the circuit on quantum hardware.

    Args:
        circuit: a quantum circuit in the abstract format
        n_shots (int): the number of shots
        backend (str): the identifier string for the desired backend

    Returns:
        int: A problem handle that can be used to retrieve the result or cancel the problem.
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
    """ Returns the current status of the problem """
    return util.get_problem_status(qemist_cloud_job_id)


def job_cancel(qemist_cloud_job_id):
    """ Cancels the most recently dispatched problem and its subproblems.

    Only queued problems can be cancelled, running ones cannot.  If problem
    has no subproblems and is already running, it will continue running to
    completion. A problem that has subproblems will terminate all queued
    subproblems, then will terminate when all running subproblems complete.

    Returns:
        list: Cancelled problems and number of subproblems.

    Raises:
        NoSimulationRun: If `simulate` has not been called.
    """

    res = util.cancel_problems(qemist_cloud_job_id)
    # If res is coming out as an error code, raise error

    return res


def job_result(qemist_cloud_job_id):
    """ Blocks until the result of the simulation is available.

    Returns:
        dict: A dictionary containing the results of the
              latest simulation run by the solver.

    Raises:
        NoSimulationRun: If `simulate` has not been called.
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
    result = util.get_quantum_results(problem_handle=qemist_cloud_job_id)[qemist_cloud_job_id]

    return result
