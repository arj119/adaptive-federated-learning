import socket
import time
import struct

from data_reader.data_reader import get_data, get_data_train_samples
from models.get_model import get_model
from util.sampling import MinibatchSampling
from util.utils import send_msg, recv_msg

# Configurations are in a separate config.py file
from config import SERVER_ADDR, SERVER_PORT, dataset_file_path

sock = socket.socket()
sock.connect((SERVER_ADDR, SERVER_PORT))

print('---------------------------------------------------------------------------')

batch_size_prev = None
total_data_prev = None
sim_prev = None

if __name__ == '__main__':
    try:
        while True:
            msg = recv_msg(sock, 'MSG_INIT_SERVER_TO_CLIENT')
            # ['MSG_INIT_SERVER_TO_CLIENT', model_name, dataset, num_iterations_with_same_minibatch_for_tau_equals_one, step_size, batch_size,
            # total_data, use_control_alg, indices_this_node, read_all_data_for_stochastic, use_min_loss, sim]

            model_name = msg[1]
            dataset = msg[2]
            step_size = msg[3]
            batch_size = msg[4]
            total_data = msg[5]
            indices_this_node = msg[6]
            read_all_data_for_stochastic = msg[7]
            use_min_loss = msg[8]
            sim = msg[9]

            model = get_model(model_name)
            model2 = get_model(model_name)   # Used for computing loss_w_prev_min_loss for stochastic gradient descent,
                                            # so that the state of model can be still used by control algorithm later.

            if hasattr(model, 'create_graph'):
                model.create_graph(learning_rate=step_size)
            if hasattr(model2, 'create_graph'):
                model2.create_graph(learning_rate=step_size)

            # Assume the dataset does not change
            if read_all_data_for_stochastic or batch_size >= total_data:
                if batch_size_prev != batch_size or total_data_prev != total_data or (batch_size >= total_data and sim_prev != sim):
                    print('Reading all data samples used in training...')
                    train_image, train_label, _, _, _ = get_data(dataset, total_data, dataset_file_path, sim_round=sim)

            batch_size_prev = batch_size
            total_data_prev = total_data
            sim_prev = sim

            if batch_size >= total_data:
                sampler = None
                train_indices = indices_this_node
            else:
                sampler = MinibatchSampling(indices_this_node, batch_size, sim)
                train_indices = None  # To be defined later
            last_batch_read_count = None

            data_size_local = len(indices_this_node)

            w_prev_min_loss = None
            w_last_global = None
            total_iterations = 0

            msg = ['MSG_DATA_PREP_FINISHED_CLIENT_TO_SERVER']
            send_msg(sock, msg)

            while True:
                print('---------------------------------------------------------------------------')

                msg = recv_msg(sock, 'MSG_WEIGHT_TAU_SERVER_TO_CLIENT')
                # ['MSG_WEIGHT_TAU_SERVER_TO_CLIENT', w_global, tau, is_last_round, prev_loss_is_min]
                w = msg[1]
                tau = msg[2]
                is_last_round = msg[3]
                prev_loss_is_min = msg[4]

                if prev_loss_is_min or ((w_prev_min_loss is None) and (w_last_global is not None)):
                    w_prev_min_loss = w_last_global

                time_local_start = time.time()  #Only count this part as time for local iteration because the remaining part does not increase with tau

                # Perform local iteration
                grad = None
                loss_last_global = None   # Only the loss at starting time is from global model parameter
                loss_w_prev_min_loss = None

                for i in range(0, tau):

                    # When batch size is smaller than total data, read the data here; else read data during client init above
                    if batch_size < total_data:
                        # When using the control algorithm, we want to make sure that the batch in the last local iteration
                        # in the previous round and the first iteration in the current round is the same,
                        # because the local and global parameters are used to
                        # estimate parameters used for the adaptive tau control algorithm.
                        # Therefore, we only change the data in minibatch when (i != 0) or (sample_indices is None).
                        # The last condition with tau <= 1 is to make sure that the batch will change when tau = 1,
                        # this may add noise in the parameter estimation for the control algorithm,
                        # and the amount of noise would be related to NUM_ITERATIONS_WITH_SAME_MINIBATCH.

                        if (i != 0) or (train_indices is None) \
                                or (tau <= 1):

                            sample_indices = sampler.get_next_batch()

                            if read_all_data_for_stochastic:
                                train_indices = sample_indices
                            else:
                                train_image, train_label = get_data_train_samples(dataset, sample_indices, dataset_file_path)
                                train_indices = range(0, min(batch_size, len(train_label)))

                            last_batch_read_count = 0

                        last_batch_read_count += 1

                    grad = model.gradient(train_image, train_label, w, train_indices)

                    if i == 0:
                        try:
                            # Note: This has to follow the gradient computation line above
                            loss_last_global = model.loss_from_prev_gradient_computation()
                            print('*** Loss computed from previous gradient computation')
                        except:
                            # Will get an exception if the model does not support computing loss
                            # from previous gradient computation
                            loss_last_global = model.loss(train_image, train_label, w, train_indices)
                            print('*** Loss computed from data')

                        w_last_global = w

                        if use_min_loss:
                            if (batch_size < total_data) and (w_prev_min_loss is not None):
                                # Compute loss on w_prev_min_loss so that the batch remains the same
                                loss_w_prev_min_loss = model2.loss(train_image, train_label, w_prev_min_loss, train_indices)

                    w = w - step_size * grad

                    total_iterations += 1

                # Local operation finished, global aggregation starts
                time_local_end = time.time()
                time_all_local = time_local_end - time_local_start
                print('time_all_local =', time_all_local)

                msg = ['MSG_WEIGHT_TIME_SIZE_CLIENT_TO_SERVER', w, time_all_local, data_size_local, loss_last_global, loss_w_prev_min_loss]
                send_msg(sock, msg)

                if is_last_round:
                    break

    except (struct.error, socket.error):
        print('Server has stopped')
        pass
