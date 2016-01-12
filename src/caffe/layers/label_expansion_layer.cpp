#include <vector>

#include "caffe/layers/label_expansion_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LabelExpansionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Layer<Dtype>::LayerSetUp(bottom, top);
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
    "allow in-place computation.";

  CHECK_EQ(bottom[0]->num(), bottom[0]->count()) << " Layer expects "
    "one label per sampe.";

  auto param = this->layer_param_.label_expansion_param();
  this->num_classes = param.num_classes();
}

template <typename Dtype>
void LabelExpansionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
             const vector<Blob<Dtype>*>& top) {

    top[0]->Reshape(bottom[0]->num(), this->num_classes, 1, 1);
}

template <typename Dtype>
void LabelExpansionLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  const Dtype* in = bottom[0]->cpu_data();
  Dtype* out = top[0]->mutable_cpu_data();
  caffe_set<Dtype>(top[0]->count(), 0, out);
  for(int i = 0; i < bottom[0]->num(); ++i) {
    out[static_cast<int>(i * this->num_classes + in[i])] = 1;
  }

}

//template <typename Dtype>
//void LabelExpansionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
//    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
//  const int count = top[0]->count();
//  const Dtype* top_diff = top[0]->cpu_diff();
//  if (propagate_down[0]) {
//    const Dtype* bottom_data = bottom[0]->cpu_data();
//    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
//    caffe_cpu_sign(count, bottom_data, bottom_diff);
//    caffe_mul(count, bottom_diff, top_diff, bottom_diff);
//  }
//}

#ifdef CPU_ONLY
STUB_GPU(LabelExpansionLayer);
#endif

INSTANTIATE_CLASS(LabelExpansionLayer);
REGISTER_LAYER_CLASS(LabelExpansion);

}  // namespace caffe
