#include <vector>

#include <algorithm>
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/uncertainty_crossentropy_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
UncertaintyCrossentropyLossLayer<Dtype>::UncertaintyCrossentropyLossLayer(const LayerParameter& param)
        : LossLayer<Dtype>(param), diff_() {

    const UncertaintyLossParameter& uncertainty_param = param.uncertainty_loss_param();
    this->uncertainty_weight_ = uncertainty_param.uncertainty_weight();
}

template <typename Dtype>
void UncertaintyCrossentropyLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->num(), bottom[1]->count())
      << "Inputs must have the same dimension.";
  CHECK_EQ(bottom[0]->num(), bottom[2]->count())
      << "There must be one uncertainty score per data sample";

  diff_.ReshapeLike(*bottom[0]);
}


// computes MSE uncertainty loss. There's one output per sample (e.g. 30 outputs
// when the minibatch size=30)
// assumption: bottom[0] - prediction, bottom[1] - groundtruth, bottom[2] - uncertainty measure
template <typename Dtype>
void UncertaintyCrossentropyLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* prediction_data =  bottom[0]->cpu_data();
  const Dtype* groundtruth_data = bottom[1]->cpu_data();
  int num = bottom[2]->count(); // count of uncertainty outputs (or samples)
  int count = bottom[0]->count() / num; // count of groundtruth/predictions per sample

  Dtype num_correct = 1;
  Dtype num_false = 0;
  for(int i = 0; i < num; ++i, prediction_data += count) {

    int prediction = std::distance(prediction_data, std::max_element(prediction_data, prediction_data +  count));
    int label = groundtruth_data[i];

    num_correct += prediction == label;

    // correct == 1
    diff_.mutable_cpu_diff()[i] = prediction == label;
  }

  num_false = 2 + num - num_correct;

  Dtype uncertainty_loss = 0;

  prediction_data =  bottom[0]->cpu_data();
  //iterate over samples
  for(int i = 0; i < num; ++i) {

    Dtype correct = diff_.mutable_cpu_diff()[i];
    Dtype uncertainty = bottom[2]->cpu_data()[i];

    uncertainty += (- (uncertainty > (1 - 1e-4)) + (uncertainty < 1e-4)) * (1e-4);


    uncertainty_loss -= correct ? log(1 - uncertainty) : log(uncertainty);

    // Save the derivative for backprop
    diff_.mutable_cpu_diff()[i] = - correct / uncertainty + (1 - correct) / (1 -
                                                                             uncertainty);
//    diff_.mutable_cpu_diff()[i] = - correct / uncertainty / num_correct + (1 - correct) / (1 - uncertainty) / num_false;

//    LOG(ERROR) << diff_.mutable_cpu_diff()[i];
  }

  // average over samples, weight and divide by 2
  top[0]->mutable_cpu_data()[0] = uncertainty_loss * uncertainty_weight_ / num;
}

template <typename Dtype>
void UncertaintyCrossentropyLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  if (propagate_down[2]) {
      caffe_cpu_axpby(
          bottom[2]->count(),
          uncertainty_weight_ / bottom[2]->count() * top[0]->cpu_diff()[0],
          diff_.cpu_diff(),
          Dtype(0),
          bottom[2]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
STUB_GPU(UncertaintyCrossentropyLossLayer);
#endif

INSTANTIATE_CLASS(UncertaintyCrossentropyLossLayer);
REGISTER_LAYER_CLASS(UncertaintyCrossentropyLoss);

}  // namespace caffe
