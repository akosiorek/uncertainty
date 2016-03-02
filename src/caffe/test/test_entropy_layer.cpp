#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/entropy_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class EntropyLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
    EntropyLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_top_entropy_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_mean(1);
    filler_param.set_std(0.1);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_top_vec_.push_back(blob_top_entropy_);
  }
  virtual ~EntropyLayerTest() {
    delete blob_bottom_data_;
    delete blob_top_entropy_;
  }

  void TestForward() {
    // Get the loss without a specified objective weight -- should be
    // equivalent to explicitly specifiying a weight of 1.
    LayerParameter layer_param;
    EntropyLayer<Dtype> layer_weight_1(layer_param);
    layer_weight_1.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

    EXPECT_EQ(blob_bottom_data_->num(), blob_top_entropy_->num());
    EXPECT_EQ(blob_top_entropy_->channels(), 1);
    EXPECT_EQ(blob_top_entropy_->width(), 1);
    EXPECT_EQ(blob_top_entropy_->height(), 1);


    layer_weight_1.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    for(int i = 0; i < blob_top_entropy_->num(); ++i) {
      EXPECT_GE(blob_top_entropy_->cpu_data()[i], 0);
      EXPECT_LE(blob_top_entropy_->cpu_data()[i], 1);
    }
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_top_entropy_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(EntropyLayerTest, TestDtypesAndDevices);
//TYPED_TEST_CASE(EntropyLayerTest, ::testing::Types<CPUDevice<float>>);

TYPED_TEST(EntropyLayerTest, TestForward) {
  this->TestForward();
}

TYPED_TEST(EntropyLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  EntropyLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
