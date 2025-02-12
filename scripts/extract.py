import onnx
from onnx.utils import extract_model
import onnxruntime as rt


def extract_and_run(new_name, output_names):
    input_path = '/home/kcvlex/my-onnx/models/validated/yolov4/yolov4.onnx'
    input_tensor_path = '/home/kcvlex/my-onnx/models/validated/yolov4/test_data_set_0/input_0.pb'
    output_path = new_name + '.onnx'
    new_inputs = ['input_1:0']
    new_outputs = output_names
    extract_model(input_path, output_path, new_inputs, new_outputs)
    input_tensor = onnx.load_tensor(input_tensor_path)
    input_tensor = onnx.numpy_helper.to_array(input_tensor)
    sess = rt.InferenceSession(output_path)
    result = sess.run([], {'input_1:0': input_tensor})
    for i, tensor in enumerate(result):
        onnx.save_tensor(onnx.numpy_helper.from_array(tensor), new_name + '_' + str(i) + '.pb')
    return result


result = extract_and_run(
    'yolov4_until_tf_op_layer_leakyrelu_2_leakyrelu_2', 
    ['StatefulPartitionedCall/model/tf_op_layer_LeakyRelu_2/LeakyRelu_2:0']
)
