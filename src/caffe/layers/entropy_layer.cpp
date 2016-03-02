#include <vector>

#include "caffe/layers/entropy_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EntropyLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {}

template <typename Dtype>
void EntropyLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  CHECK_EQ(bottom.size(), 1);
  CHECK_EQ(top.size(), 1);

  top[0]->Reshape(bottom[0]->num(), 1, 1, 1);
}

template <typename Dtype>
void EntropyLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  const int channels = bottom[0]->channels();
  const int num = bottom[0]->num();

  const Dtype* input_data = bottom[0]->cpu_data();
  Dtype* entropy = top[0]->mutable_cpu_data();
  Dtype* grad = bottom[0]->mutable_cpu_diff();

  for (int i = 0; i < num; ++i) {
    Dtype sum = caffe_cpu_asum<Dtype>(channels, input_data + i * channels);

    entropy[i] = 0;
    Dtype grad_sum = 0;
    for(int j = 0; j < channels; ++j) {
      Dtype val = input_data[i * channels + j] / sum;
      grad[i * channels + j] = log(val) + 1;
      grad_sum -= (log(val) + 1) * val;


      entropy[i] -= val * log(val);
    }

    for(int j = 0; j < channels; ++j) {
      grad[i * channels + j] += grad_sum;
      grad[i * channels + j] /= sum;
    }

    entropy[i] /=  log(channels);
  }
}

template <typename Dtype>
void EntropyLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  const int channels = bottom[0]->channels();
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();

  Dtype* grad = bottom[0]->mutable_cpu_diff();
  const Dtype* top_grad = top[0]->cpu_diff();

  for (int i = 0; i < num; ++i) {
    Dtype* current = grad + i * channels;
    caffe_cpu_scale<Dtype>(channels, top_grad[i], current, current);
  }

  caffe_cpu_scale<Dtype>(count, -1/log(channels), grad, grad);
}

#ifdef CPU_ONLY
STUB_GPU(EntropyLayer);
#endif

INSTANTIATE_CLASS(EntropyLayer);
REGISTER_LAYER_CLASS(Entropy);

}  // namespace caffe
