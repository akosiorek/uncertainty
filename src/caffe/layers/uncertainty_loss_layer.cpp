#include <vector>

#include <algorithm>
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

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
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  CHECK_EQ(bottom[0]->num(), bottom[2]->count())
      << "There must be one uncertainty score per data sample";

  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void UncertaintyLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / bottom[0]->num() / Dtype(2);

  top[0]->mutable_cpu_data()[0] = loss + computeUncertaintyLoss_cpu(bottom);
}

template <typename Dtype>
void UncertaintyLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }

  if (propagate_down[2]) {
      caffe_cpu_axpby(
          bottom[2]->count(),
          uncertainty_weight_ / bottom[2]->count() * top[0]->cpu_diff()[0],
          diff_.cpu_diff(),
          Dtype(0),
          bottom[2]->mutable_cpu_diff());
  }
}

template <typename Dtype>
Dtype UncertaintyLossLayer<Dtype>::computeUncertaintyLoss_cpu(const vector<Blob<Dtype>*>& bottom) {

    const Dtype* prediction_data =  bottom[0]->cpu_data();
    const Dtype* groundtruth_data = bottom[1]->cpu_data();
    int num = bottom[2]->count(); // count of uncertainty outputs (or samples)
    int count = bottom[0]->count() / num; // count of groundtruth/predictions per sample

    Dtype uncertainty_loss = 0;
    for(int i = 0; i < num; ++i) {
        int prediction = std::distance(prediction_data, std::max(prediction_data, prediction_data + count));
        int label = std::distance(groundtruth_data, std::max(groundtruth_data, groundtruth_data + count));

        // expected_uncertainty: = 1 if prediction != label
        //                       = 0 if prediction == label
        int expected_uncertainty = (prediction != label);
        Dtype partial_uncertainty_loss = bottom[2]->cpu_data()[i] - expected_uncertainty;

        // store for backprop
        diff_.mutable_cpu_diff()[i] = partial_uncertainty_loss;
        uncertainty_loss += partial_uncertainty_loss;

        prediction_data += count;
        groundtruth_data += count;
    }

    return uncertainty_loss * uncertainty_loss * uncertainty_weight_ / Dtype(2) / num;
}

#ifdef CPU_ONLY
STUB_GPU(UncertaintyLossLayer);
#endif

INSTANTIATE_CLASS(UncertaintyLossLayer);
REGISTER_LAYER_CLASS(UncertaintyLoss);

}  // namespace caffe
