#include <vector>

#include <algorithm>
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/uncertainty_loss_layer.hpp"

namespace caffe
{
template <typename Dtype>
UncertaintyLossLayer<Dtype>::UncertaintyLossLayer(const LayerParameter& param)
        : LossLayer<Dtype>(param), diff_() {

    const UncertaintyLossParameter& uncertainty_param = param.uncertainty_loss_param();
    this->uncertainty_weight_ = uncertainty_param.uncertainty_weight();
}

template <typename Dtype>
void UncertaintyLossLayer<Dtype>::Reshape(
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
void UncertaintyLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* prediction_data =  bottom[0]->cpu_data();
  const Dtype* groundtruth_data = bottom[1]->cpu_data();
  int num = bottom[2]->count(); // count of uncertainty outputs (or samples)
  int count = bottom[0]->count() / num; // count of groundtruth/predictions per sample

  Dtype num_correct = 0;
  for(int i = 0; i < num; ++i, prediction_data += count) {

    int prediction = std::distance(prediction_data, std::max_element(prediction_data, prediction_data +  count));
    int label = groundtruth_data[i];

    num_correct += prediction == label;

    // correct == 1
    diff_.mutable_cpu_diff()[i] = prediction == label;
  }

    Dtype weight_correct = Dtype(1) / num_correct;
    Dtype weight_false = Dtype(1) / (num - num_correct);

//    Dtype total_weight = weight_correct + weight_false;
//    weight_correct /= total_weight;
//    weight_false /= total_weight;

  Dtype normalizer = Dtype(num * num) / 2 / num_correct / (num - num_correct);
  weight_correct *= normalizer;
  weight_false *= normalizer;

//  LOG(ERROR) << "weight correct = " << weight_correct << "\tweight false = " << weight_false;

  Dtype uncertainty_loss = 0;

  std::ofstream uncert_file("uncert.txt", std::ofstream::app);
  std::ofstream label_file("label.txt", std::ofstream::app);

  //iterate over samples
  for(int i = 0; i < num; ++i) {

    Dtype correct = diff_.mutable_cpu_diff()[i];
    Dtype partial_uncertainty_loss = (bottom[2]->cpu_data()[i] + correct - 1);
    Dtype weight = diff_.mutable_cpu_diff()[i] ? weight_correct : weight_false;

    // Save the derivative for backprop
    diff_.mutable_cpu_diff()[i] = weight * partial_uncertainty_loss;
//    diff_.mutable_cpu_diff()[i] = partial_uncertainty_loss;
    uncertainty_loss += diff_.mutable_cpu_diff()[i] * partial_uncertainty_loss;

    if(this->phase_ == TEST) {
      label_file << correct << std::endl;
      uncert_file << bottom[2]->cpu_data()[i] << std::endl;
    }
  }

  uncert_file.close();
  label_file.close();


  // average over samples, weight and divide by 2
  top[0]->mutable_cpu_data()[0] = uncertainty_loss * uncertainty_weight_ / Dtype(2) / num;

}

template <typename Dtype>
void UncertaintyLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
STUB_GPU(UncertaintyLossLayer);
#endif

INSTANTIATE_CLASS(UncertaintyLossLayer);
REGISTER_LAYER_CLASS(UncertaintyLoss);

}  // namespace caffe
