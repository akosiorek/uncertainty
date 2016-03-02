//#include <vector>
//
//#include "caffe/layers/entropy_layer.hpp"
//#include "caffe/util/math_functions.hpp"
//
//namespace caffe {
//
//template <typename Dtype>
//void EntropyLayer<Dtype>::Backward_gpu(
//    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
//    const vector<Blob<Dtype>*>& bottom) {
//
//  const int channels = bottom[0]->channels();
//  const int count = bottom[0]->count();
//  const int num = bottom[0]->num();
//
//  LOG(ERROR) << 1;
//
//  Dtype* grad = bottom[0]->mutable_gpu_diff();
//  Dtype* top_grad = top[0]->mutable_gpu_diff();
//LOG(ERROR) << "top grad size = " << top[0]->count();
//
//LOG(ERROR) << 2;
//
//
//  for (int i = 0; i < num; ++i) {
//    LOG(ERROR) << 2 << " " << i;
//    Dtype* current = grad + i * channels;
//    caffe_gpu_scal<Dtype>(channels, top[0]->mutable_gpu_diff()[i], &bottom[0]->mutable_gpu_diff()[i*channels]);
//  }
//LOG(ERROR) << 3;
//  caffe_gpu_scal<Dtype>(count, -1/log(channels), grad);
//LOG(ERROR) << 4;
//}
//
//INSTANTIATE_LAYER_GPU_BACKWARD(EntropyLayer);
//
//}  // namespace caffe
