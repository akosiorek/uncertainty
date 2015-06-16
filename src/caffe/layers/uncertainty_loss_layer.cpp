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
  CHECK_EQ(bottom[0]->num(), bottom[1]->count())
      << "Inputs must have the same dimension.";
  CHECK_EQ(bottom[0]->num(), bottom[2]->count())
      << "There must be one uncertainty score per data sample";

  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void UncertaintyLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  // computes uncertainty loss
  top[0]->mutable_cpu_data()[0] = computeUncertaintyLoss_cpu(bottom);
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

// computes MSE uncertainty loss. There's one output per sample (e.g. 30 outputs
// when the minibatch size=30)
// assumption: bottom[0] - prediction, bottom[1] - groundtruth, bottom[2] - uncertainty measure
template <typename Dtype>
Dtype UncertaintyLossLayer<Dtype>::computeUncertaintyLoss_cpu(const vector<Blob<Dtype>*>& bottom) {

    const Dtype* prediction_data =  bottom[0]->cpu_data();
    const Dtype* groundtruth_data = bottom[1]->cpu_data();
    int num = bottom[2]->count(); // count of uncertainty outputs (or samples)
    int count = bottom[0]->count() / num; // count of groundtruth/predictions per sample

	const Dtype* tmp = prediction_data;	
	int num_correct = 0;
	int num_misclassified = 0;
    for(int i = 0; i < num; ++i) 
    {           
        int prediction = std::distance(prediction_data, std::max_element(prediction_data, prediction_data +  count));
        int label = groundtruth_data[i];
        num_misclassified += (prediction != label);

        prediction_data += count;
    }          
		
	num_correct = num - num_misclassified; 
	prediction_data = tmp;	

    Dtype uncertainty_loss = 0;
    //iterater over samples
    for(int i = 0; i < num; ++i) {

        int prediction = std::distance(prediction_data, std::max_element(prediction_data, prediction_data + count));
        int label = groundtruth_data[i];

//        LOG(ERROR) << "prediction: " << prediction << " label: " << label;
	
        // expected_uncertainty: = 1 if prediction != label
        //                       = 0 if prediction == label
        int expected_uncertainty = (prediction != label);

		Dtype partial_uncertainty_loss = bottom[2]->cpu_data()[i] - expected_uncertainty;
		
		// Save the derivative for backprop
        diff_.mutable_cpu_diff()[i] = partial_uncertainty_loss;


		if(expected_uncertainty == 1)
		{
	  		// store the difference for backprop
          	diff_.mutable_cpu_diff()[i] *= (1/Dtype(num_misclassified));
			
          	//computes as a sum of squared differences
          	uncertainty_loss += (1/Dtype(num_misclassified)) * partial_uncertainty_loss * partial_uncertainty_loss;
		}
       	
		else
		{
           // store the difference for backprop
           diff_.mutable_cpu_diff()[i] *= (1/Dtype(num_correct));
			
           //computes as a sum of squared differences
           uncertainty_loss += (1/Dtype(num_correct)) * partial_uncertainty_loss * partial_uncertainty_loss;
		} 


        prediction_data += count;
    }

    // average over samples, weight and divide by 2
    return uncertainty_loss * uncertainty_weight_ / Dtype(2) / num;
}



#ifdef CPU_ONLY
STUB_GPU(UncertaintyLossLayer);
#endif

INSTANTIATE_CLASS(UncertaintyLossLayer);
REGISTER_LAYER_CLASS(UncertaintyLoss);

}  // namespace caffe
