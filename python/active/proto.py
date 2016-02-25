
from google.protobuf import text_format
import caffe
from caffe.proto.caffe_pb2 import SolverParameter, NetParameter, BlobShape


def load_proto(path, proto_type):
    proto = proto_type()
    data = open(path).read()
    text_format.Merge(data, proto)
    return proto


def save_proto(path, proto):
    data = text_format.MessageToString(proto)
    with open(path, 'w') as f:
        f.write(data)


def extract_param_from_proto(input_path, proto_type, param_name):

    proto = load_proto(input_path, proto_type)
    if hasattr(param_name, '__iter__') and not isinstance(param_name, str):
        return [getattr(proto, param) for param in param_name]
    return getattr(proto, param_name)


def get_net_from_solver(solver_path):
    return str(extract_param_from_proto(solver_path, SolverParameter, 'net'))


def get_db_from_net(net_path, phase=caffe.TRAIN):
    proto = load_proto(net_path, NetParameter)
    for layer in proto.layer:
        if layer.type == 'Data' and layer.include[0].phase == phase:
            return layer.data_param.source


def write_db_to_net(net_proto, db, phase=caffe.TRAIN):
    for layer in net_proto.layer:
        if layer.type == 'Data' and layer.include[0].phase == phase:
            layer.data_param.source = db
            break


def prepare_solver(solver_path, out_path, prepared_net):
    print 'Preparing solver.prototxt...'
    proto = load_proto(solver_path, SolverParameter)

    proto.snapshot = 0
    proto.snapshot_after_train = True
    proto.max_iter = 2000
    proto.net = prepared_net
    proto.test_initialization = False

    save_proto(out_path, proto)
    return proto.snapshot_prefix, proto.max_iter


def prepare_net(net_path, out_path, new_train_db):
    print 'Preparing net.prototxt'
    proto = load_proto(net_path, NetParameter)

    write_db_to_net(proto, new_train_db, caffe.TRAIN)
    # write_db_to_net(proto, new_train_db, caffe.TEST)

    # # remove test layers
    # for i in xrange(len(proto.layer)-1, -1, -1):
    #     if len(proto.layer[i].include) and proto.layer[i].include[0].phase == caffe.TEST:
    #         del proto.layer[i]

    save_proto(out_path, proto)


def prepare_deploy_net(net_path, out_path, batch_size, input_size):
    print 'Preparing deploy net'
    proto = load_proto(net_path, NetParameter)

    tops = set()
    for layer in proto.layer:
        if layer.type == 'Data':
            tops.update(layer.top)

    # find layers using labels
    layers_to_remove = []
    for i in xrange(len(proto.layer)):
        for b in proto.layer[i].bottom:
            if b == 'label':
                layers_to_remove.append(i)
                break


    input_shape = BlobShape()
    input_shape.dim.extend([batch_size])
    input_shape.dim.extend(input_size)

    proto.input.extend([proto.layer[0].top[0]])
    proto.input_shape.extend([input_shape])

    # remove layers using labels
    for i in reversed(layers_to_remove):
        del proto.layer[i]

    # remove data layers
    del proto.layer[1]
    del proto.layer[0]

    save_proto(out_path, proto)


def get_batch_mean_from_net(net_path):
    proto = load_proto(net_path, NetParameter)
    batch_size = proto.layer[0].data_param.batch_size
    mean_file = proto.layer[0].transform_param.mean_file

    return batch_size, mean_file


def increase_max_iters(solver_path, how_many):
    proto = load_proto(solver_path, SolverParameter)

    proto.max_iter += how_many
    save_proto(solver_path, proto)


if __name__ == '__main__':
    import sys
    print 'net: {0}'.format(extract_param_from_proto(sys.argv[1], SolverParameter, 'net'))