#ifndef CAFFE_CUDNN_RELU_LAYER_HPP_
#define CAFFE_CUDNN_RELU_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class UncertaintyLossLayer : public LossLayer<Dtype> {
public:
    explicit UncertaintyLossLayer(const LayerParameter& param);


    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "UncertaintyLoss"; }

    virtual inline bool AllowForceBackward(const int bottom_index) const {
      return true;
    }

    virtual inline int ExactNumBottomBlobs() const { return 3; }

protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
//    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
//    const vector<Blob<Dtype>*>& top);

    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
//    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
//    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    Blob<Dtype> diff_;
    Dtype uncertainty_weight_;
};

}  // namespace caffe

#endif  // CAFFE_CUDNN_RELU_LAYER_HPP_
