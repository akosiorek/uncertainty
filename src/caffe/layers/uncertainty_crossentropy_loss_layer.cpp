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

template<typename T>
void write_to_file(const T* data, size_t size, const std::string& filename) {

  std::ofstream file(filename, std::ofstream::app);
  for(int i = 0; i < size; ++i) {
    file << data[i] << std::endl;
  }
  file.close();
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

  Dtype num_correct = 0;
  for(int i = 0; i < num; ++i, prediction_data += count) {

    auto prediction = std::distance(prediction_data, std::max_element(prediction_data, prediction_data +  count));
    auto label = groundtruth_data[i];

    num_correct += prediction == label;

    // correct == 1
    diff_.mutable_cpu_diff()[i] = prediction == label;
  }

//  Dtype weight_correct = 1;
//  Dtype weight_false = 1;

  Dtype weight_correct = 2 * (num - num_correct) / num;
  Dtype weight_false = 2 - weight_correct;

  //  hack for testing
  if(this->phase_ == TEST) {
    write_to_file(diff_.mutable_cpu_diff(), num, "label.txt");
    write_to_file(bottom[2]->cpu_data(), num, "uncert.txt");
  }

  Dtype uncertainty_loss = 0;
  //iterate over samples
  for(int i = 0; i < num; ++i) {

    Dtype correct = diff_.mutable_cpu_diff()[i];
    Dtype uncertainty = bottom[2]->cpu_data()[i];
    uncertainty = correct * (1 - uncertainty) + (1 - correct) * uncertainty;

    Dtype eps = 1e-6;
    uncertainty += (- (uncertainty > (1 - eps)) + (uncertainty < eps)) * (eps);

    uncertainty_loss -= log(uncertainty);// * (correct * weight_correct + (1 - correct) * weight_false);

    // Save the derivative for backprop
//    diff_.mutable_cpu_diff()[i] = (1 / uncertainty - (1 - correct) * 2 / uncertainty);
    diff_.mutable_cpu_diff()[i] = (1 / uncertainty * weight_correct - (1 - correct) / uncertainty * (weight_correct + weight_false));
  }

  // average over samples and weigh
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

//  LOG(ERROR) << caffe_cpu_dot<Dtype>(bottom[2]->count(), bottom[2]->cpu_data(), bottom[2]->cpu_data());
}

#ifdef CPU_ONLY
STUB_GPU(UncertaintyCrossentropyLossLayer);
#endif

INSTANTIATE_CLASS(UncertaintyCrossentropyLossLayer);
REGISTER_LAYER_CLASS(UncertaintyCrossentropyLoss);

}  // namespace caffe
