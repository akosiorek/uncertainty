#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/uncertainty_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void UncertaintyLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  top[0]->mutable_cpu_data()[0] = computeUncertaintyLoss_cpu(bottom);
}

template <typename Dtype>
void UncertaintyLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  if (propagate_down[2]) {
    caffe_gpu_axpby(
      bottom[2]->count(),
      uncertainty_weight_ / bottom[2]->count() * top[0]->gpu_diff()[0],
      diff_.gpu_diff(),
      Dtype(0),
      bottom[2]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(UncertaintyLossLayer);

}  // namespace caffe
