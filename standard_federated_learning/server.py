import socket
import time

import numpy as np
import tensorflow as tf
import datetime

from data_reader.data_reader import get_data
from data_reader.data_reader import get_data
from models.get_model import get_model
from statistic.collect_stat import CollectStatistics
from util.utils import send_msg, recv_msg, get_indices_each_node_case

# Configurations are in a separate config.py file
from .config import *

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/standard_fl/' + current_time + '/train'
test_log_dir = 'logs/standard_fl/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

if __name__ == '__main__':

    model = get_model(model_name)
    if hasattr(model, 'create_graph'):
        model.create_graph(learning_rate=step_size)

    if time_gen is not None:
        use_fixed_averaging_slots = True
    else:
        use_fixed_averaging_slots = False

    if batch_size < total_data:   # Read all data once when using stochastic gradient descent
        train_image, train_label, test_image, test_label, train_label_orig = get_data(dataset, total_data, dataset_file_path)

        # This function takes a long time to complete,
        # putting it outside of the sim loop because there is no randomness in the current way of computing the indices
        indices_each_node_case = get_indices_each_node_case(n_nodes, MAX_CASE, train_label_orig)

    listening_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listening_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listening_sock.bind((SERVER_ADDR, SERVER_PORT))
    client_sock_all=[]

    # Establish connections to each client, up to n_nodes clients
    while len(client_sock_all) < n_nodes:
        listening_sock.listen(5)
        print("Waiting for incoming connections...")
        (client_sock, (ip, port)) = listening_sock.accept()
        print('Got connection from ', (ip,port))
        print(client_sock)

        client_sock_all.append(client_sock)

    if single_run:
        stat = CollectStatistics(results_file_name=single_run_results_file_path, is_single_run=True)
    else:
        stat = CollectStatistics(results_file_name=multi_run_results_file_path, is_single_run=False)

    for sim in sim_runs:

        if batch_size >= total_data:  # Read data again for different sim. round
            train_image, train_label, test_image, test_label, train_label_orig = get_data(dataset, total_data, dataset_file_path, sim_round=sim)

            # This function takes a long time to complete,
            indices_each_node_case = get_indices_each_node_case(n_nodes, MAX_CASE, train_label_orig)

        for case in case_range:

            for tau_setup in tau_setup_all:

                stat.init_stat_new_global_round()

                dim_w = model.get_weight_dimension(train_image, train_label)
                w_global_init = model.get_init_weight(dim_w, rand_seed=sim)
                w_global = w_global_init

                w_global_min_loss = None
                loss_min = np.inf
                prev_loss_is_min = False

                for n in range(0, n_nodes):
                    indices_this_node = indices_each_node_case[case][n]
                    msg = ['MSG_INIT_SERVER_TO_CLIENT', model_name, dataset, step_size,
                        batch_size, total_data, indices_this_node, read_all_data_for_stochastic,
                        use_min_loss, sim]
                    send_msg(client_sock_all[n], msg)

                print('All clients connected')

                # Wait until all clients complete data preparation and sends a message back to the server
                for n in range(0, n_nodes):
                    recv_msg(client_sock_all[n], 'MSG_DATA_PREP_FINISHED_CLIENT_TO_SERVER')

                print('Start learning')

                time_global_aggregation_all = None

                total_time = 0      # Actual total time, where use_fixed_averaging_slots has no effect
                total_time_recomputed = 0  # Recomputed total time using estimated time for each local and global update,
                                            # using predefined values when use_fixed_averaging_slots = true
                it_each_local = None
                it_each_global = None

                is_last_round = False
                is_eval_only = False

                # Loop for multiple rounds of local iterations + global aggregation
                for epoch in range(epochs):

                    print('---------------------------------------------------------------------------')

                    time_total_all_start = time.time()

                    for n in range(0, n_nodes):
                        msg = ['MSG_WEIGHT_TAU_SERVER_TO_CLIENT', w_global, tau, is_last_round, prev_loss_is_min]
                        send_msg(client_sock_all[n], msg)

                    w_global_prev = w_global

                    print('Waiting for local iteration at client')

                    w_global = np.zeros(dim_w)
                    loss_last_global = 0.0
                    loss_w_prev_min_loss = 0.0
                    received_loss_local_w_prev_min_loss = False
                    data_size_total = 0
                    time_all_local_all = 0
                    data_size_local_all = []

                    for n in range(0, n_nodes):
                        msg = recv_msg(client_sock_all[n], 'MSG_WEIGHT_TIME_SIZE_CLIENT_TO_SERVER')
                        # ['MSG_WEIGHT_TIME_SIZE_CLIENT_TO_SERVER', w, time_all_local, tau_actual, data_size_local,
                        # loss_last_global, loss_w_prev_min_loss]
                        w_local = msg[1]
                        time_all_local = msg[2]
                        data_size_local = msg[3]
                        loss_local_last_global = msg[4]
                        loss_local_w_prev_min_loss = msg[5]

                        w_global += w_local * data_size_local
                        data_size_local_all.append(data_size_local)
                        data_size_total += data_size_local
                        time_all_local_all = max(time_all_local_all, time_all_local)   #Use max. time to take into account the slowest node

                        if use_min_loss:
                            loss_last_global += loss_local_last_global * data_size_local
                            if loss_local_w_prev_min_loss is not None:
                                loss_w_prev_min_loss += loss_local_w_prev_min_loss * data_size_local
                                received_loss_local_w_prev_min_loss = True

                    w_global /= data_size_total

                    if True in np.isnan(w_global):
                        print('*** w_global is NaN, using previous value')
                        w_global = w_global_prev   # If current w_global contains NaN value, use previous w_global
                        use_w_global_prev_due_to_nan = True
                    else:
                        use_w_global_prev_due_to_nan = False

                    if use_min_loss:
                        loss_last_global /= data_size_total

                        if received_loss_local_w_prev_min_loss:
                            loss_w_prev_min_loss /= data_size_total
                            loss_min = loss_w_prev_min_loss

                        if loss_last_global < loss_min:
                            loss_min = loss_last_global
                            w_global_min_loss = w_global_prev
                            prev_loss_is_min = True
                        else:
                            prev_loss_is_min = False

                        print("Loss of previous global value: " + str(loss_last_global))
                        print("Minimum loss: " + str(loss_min))

                    # Calculate time
                    time_total_all_end = time.time()
                    time_total_all = time_total_all_end - time_total_all_start
                    time_global_aggregation_all = max(0.0, time_total_all - time_all_local_all)

                    print('Time for one local iteration:', time_all_local_all / tau)
                    print('Time for global averaging:', time_global_aggregation_all)

                    if use_fixed_averaging_slots:
                        if isinstance(time_gen, (list,)):
                            t_g = time_gen[case]
                        else:
                            t_g = time_gen
                        it_each_local = max(0.00000001, np.sum(t_g.get_local(tau)) / tau)
                        it_each_global = t_g.get_global(1)[0]
                    else:
                        it_each_local = max(0.00000001, time_all_local_all / tau)
                        it_each_global = time_global_aggregation_all

                    #Compute number of iterations is current slot
                    total_time_recomputed += it_each_local * tau + it_each_global

                    #Compute time in current slot
                    total_time += time_total_all

                    # stat.collect_stat_end_local_round_standard_fl(case, it_each_local, it_each_global, model,
                                                    #   train_image, train_label, test_image, test_label, w_global,
                                                    #   total_time_recomputed)

                    with test_summary_writer.as_default():
                        global_loss = model.loss(test_image, test_label, w_global)
                        global_accuracy = model.accuracy(test_image, test_label, w_global)
                        tf.summary.scalar('global_test_loss', global_loss, step=epoch)
                        tf.summary.scalar('global_test_accuracy', global_accuracy, step=epoch)

                    # with train_summary_writer.as_default():
                        # global_loss = model.loss(train_image, train_label, w_global)
                        # global_accuracy = model.accuracy(train_image, train_image, w_global)
                        # tf.summary.scalar('global_train_loss', global_loss, step=epoch)
                        # tf.summary.scalar('global_train_accuracy', global_accuracy, step=epoch)

                    if use_min_loss:
                        tmp_time_for_executing_remaining = total_time_recomputed + it_each_local * (tau + 1) + it_each_global * 2
                    else:
                        tmp_time_for_executing_remaining = total_time_recomputed + it_each_local * tau + it_each_global

                    if is_last_round:
                        break

                    if is_eval_only:
                        is_last_round = True


                if use_min_loss:
                    w_eval = w_global_min_loss
                else:
                    w_eval = w_global

                global_loss, global_accuracy = stat.collect_stat_end_global_round(sim, case, tau_setup, total_time, model, train_image, train_label,
                                                test_image, test_label, w_eval, total_time_recomputed)

                
