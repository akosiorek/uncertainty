#include <vector>

#include <algorithm>
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
UncertaintyLossCrossEntropyLayer<Dtype>::UncertaintyLossCrossEntropyLayer(const LayerParameter& param)
        : LossLayer<Dtype>(param), diff_() {

    const UncertaintyLossParameter& uncertainty_param = param.uncertainty_loss_param();
    this->uncertainty_weight_ = uncertainty_param.uncertainty_weight();
}

template <typename Dtype>
void UncertaintyLossCrossEntropyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->num(), bottom[1]->count())
      << "Inputs must have the same dimension.";
  CHECK_EQ(bottom[0]->num(), bottom[2]->count())
      << "There must be one uncertainty score per data sample";

  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void UncertaintyLossCrossEntropyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  // computes uncertainty loss
  top[0]->mutable_cpu_data()[0] = computeUncertaintyLoss_cpu(bottom);
}

template <typename Dtype>
void UncertaintyLossCrossEntropyLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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

template <typename Dtype>
Dtype UncertaintyLossCrossEntropyLayer<Dtype>::computeUncertaintyLoss_cpu(const vector<Blob<Dtype>*>& bottom) {

  const Dtype* prediction_data =  bottom[0]->cpu_data();
  const Dtype* groundtruth_data = bottom[1]->cpu_data();
  int num = bottom[2]->count(); // count of uncertainty outputs (or samples)
  int count = bottom[0]->count() / num; // count of groundtruth/predictions per sample

	int num_correct = 0;
	int num_false = 0;
  Dtype sum_correct = 0;
  Dtype sum_false = 0;
  for(int i = 0; i < num; ++i, prediction_data += count) {

    int prediction = std::distance(prediction_data, std::max_element(prediction_data, prediction_data +  count));
    int label = groundtruth_data[i];

    num_correct += prediction == label;

    // correct == 1
    diff_.mutable_cpu_diff()[i] = prediction == label;
  }

  num_false = num - num_correct;

  Dtype uncertainty_loss = 0;

    //iterater over samples
    for(int i = 0; i < num; ++i) {

      Dtype correct = diff_.mutable_cpu_diff()[i];
      Dtype u_raw =   bottom[2]->cpu_data()[i];
      
//      printf( "u_raw:                  %f\n", u_raw );
//      printf( "correct:                %f\n", correct );
      
      
      Dtype partial_uncertainty_loss;
      
      // When abs(u_raw ) >> 400 exp(u_raw) can get infinite.
      // Therefore use an approximation for that case.
      if( abs(u_raw) > 400 ){
           partial_uncertainty_loss = correct * -u_raw + (1.0-correct) * u_raw;
      }
      else{
           partial_uncertainty_loss = correct * log(1.0 + exp(-u_raw)) + (1.0-correct) * log(1.0 + exp(u_raw));
      }
      Dtype weight = diff_.mutable_cpu_diff()[i] ? 1.0 / Dtype(num_correct) : 1.0 / Dtype(num_false);

//      printf( "weight:                 %f\n", weight );
//      printf( "partial_loss:           %f\n", partial_uncertainty_loss );
//      printf( "log(1.0 + exp(-u_raw)): %f\n", log(1.0 + exp(-u_raw)) );
//      printf( "log(1.0 + exp(u_raw)):  %f\n", log(1.0 + exp(u_raw)) );
//      printf( "\n---\n\n" );

      // Save the derivative for backprop
      diff_.mutable_cpu_diff()[i] = 2.0 * weight * ( 1.0/(1.0+exp(-u_raw)) - correct ) ;
      uncertainty_loss += partial_uncertainty_loss;

      sum_correct += correct * bottom[2]->cpu_data()[i];
      sum_false += !correct * bottom[2]->cpu_data()[i];
    }

    LOG_IF(ERROR, this->phase_ == TEST) << "correct:\tnum: " << num_correct << "\tsum: " << sum_correct << "\tmean: " << sum_correct / num_correct;
    LOG_IF(ERROR, this->phase_ == TEST) << "false:\tnum: " << num_false << "\tsum: " << sum_false << "\tmean: " << sum_false / num_false;

    // average over samples, weight and divide by 2
    return uncertainty_loss * uncertainty_weight_ / Dtype(2) / num;
}



#ifdef CPU_ONLY
STUB_GPU(UncertaintyLossCrossEntropyLayer);
#endif

INSTANTIATE_CLASS(UncertaintyLossCrossEntropyLayer);
REGISTER_LAYER_CLASS(UncertaintyLossCrossEntropy);

}  // namespace caffe
