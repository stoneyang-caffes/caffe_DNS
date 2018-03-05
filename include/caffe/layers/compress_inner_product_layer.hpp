#ifndef CAFFE_COMPRESS_INNER_PRODUCT_LAYER_HPP_
#define CAFFE_COMPRESS_INNER_PRODUCT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
/**
 * @brief The compressed InnerProduct layer, also known as a compressed 
 *  "fully-connected" layer
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class CInnerProductLayer : public Layer<Dtype> {
 public:
  explicit CInnerProductLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CInnerProduct"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int M_;
  int K_;
  int N_;
  bool bias_term_;
  Blob<Dtype> bias_multiplier_;

 private:
  Blob<Dtype> weight_tmp_;
  Blob<Dtype> bias_tmp_;   
  Blob<Dtype> rand_weight_m_;
  Blob<Dtype> rand_bias_m_;    
  Dtype gamma,power; 
  Dtype crate;  
  Dtype mu,std;  
  int iter_stop_;
};
}  // namespace caffe

#endif  // CAFFE_COMPRESS_INNER_PRODUCT_LAYER_HPP_