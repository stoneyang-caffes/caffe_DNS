#include <vector>
#include <cmath>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/layers/compress_conv_layer.hpp"

using namespace std;

namespace caffe {

template <typename Dtype>
void CConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BaseConvolutionLayer <Dtype>::LayerSetUp(bottom, top);

  CConvolutionParameter cconv_param = this->layer_param_.cconvolution_param();
  this->pruning_type = cconv_param.pruning_type();
  LOG(INFO) << "Pruning method: " << this->pruning_type;
  if (this->blobs_.size() == 2 && (this->bias_term_)) {
    this->blobs_.resize(4);
    // Intialize and fill the weightmask & biasmask
    this->blobs_[2].reset(new Blob<Dtype>(this->blobs_[0]->shape()));
    shared_ptr<Filler<Dtype> > weight_mask_filler(GetFiller<Dtype>(
        cconv_param.weight_mask_filler()));
    weight_mask_filler->Fill(this->blobs_[2].get());
    this->blobs_[3].reset(new Blob<Dtype>(this->blobs_[1]->shape()));
    shared_ptr<Filler<Dtype> > bias_mask_filler(GetFiller<Dtype>(
        cconv_param.bias_mask_filler()));
    bias_mask_filler->Fill(this->blobs_[3].get());
  } else if (this->blobs_.size() == 1 && (!this->bias_term_)) {
    this->blobs_.resize(2);
    // Intialize and fill the weightmask
    this->blobs_[1].reset(new Blob<Dtype>(this->blobs_[0]->shape()));
    shared_ptr<Filler<Dtype> > bias_mask_filler(GetFiller<Dtype>(
        cconv_param.bias_mask_filler()));
    bias_mask_filler->Fill(this->blobs_[1].get());
  }
  // Intializing the tmp tensor
  this->weight_tmp_.Reshape(this->blobs_[0]->shape());
  this->bias_tmp_.Reshape(this->blobs_[1]->shape());
  if (this->pruning_type == "dns") {
    /************ For dynamic network surgery ***************/
    // Intialize the hyper-parameters
    this->std = 0;
    this->mu = 0;   
    this->gamma = cconv_param.gamma(); 
    this->power = cconv_param.power();
    this->crate = cconv_param.c_rate();  
    this->iter_stop_ = cconv_param.iter_stop();
    LOG(INFO) << "gamma: " << gamma;
    LOG(INFO) << "power: " << power;
    LOG(INFO) << "crate: " << crate;
    LOG(INFO) << "iter stop: " << iter_stop_;
    /********************************************************/
  } else if (this->pruning_type == "han") {
    /*************** For Han Song's method ******************/
    this->sparsity_ratio = cconv_param.sparsity_ratio();
    LOG(INFO) << "sparsity_ratio: " << this->sparsity_ratio;
    const Dtype* weight = this->blobs_[0]->mutable_cpu_data();
    int count = this->blobs_[0]->count();
    // int count = this->blobs_[0]->count() + this->blobs_[1]->count();
    vector<Dtype> sorted_weights(count);
    for (unsigned int k = 0; k < count; ++k) {
      sorted_weights[k] = fabs(weight[k]);
	}
    sort(sorted_weights.begin(), sorted_weights.end());
    int pruning_index = int(count*sparsity_ratio);
    this->pruning_threshold = sorted_weights[pruning_index-1];
    LOG(INFO) << "sparsity ratio: " << this->sparsity_ratio;
    LOG(INFO) << "pruning threshold: " << this->pruning_threshold;
    /********************************************************/
  } else {
    LOG(FATAL) << "Unknown pruning type! " << this->pruning_type;
  }
}

template <typename Dtype>
void CConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void CConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {    
  const Dtype* weight = this->blobs_[0]->mutable_cpu_data();
  Dtype* weightMask = this->blobs_[2]->mutable_cpu_data();
  Dtype* weightTmp = this->weight_tmp_.mutable_cpu_data();
  const Dtype* bias = NULL;
  Dtype* biasMask = NULL;
  Dtype* biasTmp = NULL;
  if (this->bias_term_) {
    bias = this->blobs_[1]->mutable_cpu_data();
    biasMask = this->blobs_[3]->mutable_cpu_data();
    biasTmp = this->bias_tmp_.mutable_cpu_data();
  }

  if (this->phase_ == TRAIN && this->pruning_type == "dns") {
    /************ For dynamic network surgery ***************/
    // Calculate the mean and standard deviation of learnable parameters 
    if (this->std == 0 && this->iter_ == 0) {
      unsigned int ncount = 0;
      for (unsigned int k = 0; k < this->blobs_[0]->count(); ++k) {
        this->mu  += fabs(weightMask[k]*weight[k]);       
        this->std += weightMask[k]*weight[k]*weight[k];
        if (weightMask[k]*weight[k] != 0) {
          ncount++;
        }
      }
      if (this->bias_term_) {
        for (unsigned int k = 0; k < this->blobs_[1]->count(); ++k) {
          this->mu  += fabs(biasMask[k]*bias[k]);
          this->std += biasMask[k]*bias[k]*bias[k];
          if (biasMask[k]*bias[k] != 0) {
            ncount++;
          }
      	}       
      }
      this->mu /= ncount;
      this->std -= ncount*mu*mu;
      this->std /= ncount;
      this->std = sqrt(std);
      LOG(INFO) << "Mean value: " << mu;
      LOG(INFO) << "Std. variance: " << std;
      LOG(INFO) << "Non-zero count: " << ncount;
    }
		
    // Calculate the weight mask and bias mask with probability
    Dtype r = static_cast<Dtype>(rand())/static_cast<Dtype>(RAND_MAX);
    if (pow(1+(this->gamma)*(this->iter_),-(this->power)) > r && (this->iter_) < (this->iter_stop_)) {
      for (unsigned int k = 0; k < this->blobs_[0]->count(); ++k) {
        if (weightMask[k] == 1 && fabs(weight[k]) <= 0.9*std::max(mu+crate*std,Dtype(0))) {
          weightMask[k] = 0;
        } else if (weightMask[k] == 0 && fabs(weight[k]) > 1.1*std::max(mu+crate*std,Dtype(0))) {
          weightMask[k] = 1;
		}
      }
      if (this->bias_term_) {
        for (unsigned int k = 0; k < this->blobs_[1]->count(); ++k) {
          if (biasMask[k] == 1 && fabs(bias[k]) <= 0.9*std::max(mu+crate*std,Dtype(0))) {
            biasMask[k] = 0;
          } else if (biasMask[k] == 0 && fabs(bias[k]) > 1.1*std::max(mu+crate*std,Dtype(0))) {
            biasMask[k] = 1;
          }
        }
      }
    }
    /********************************************************/
  } else if (this->phase_ == TRAIN && this->pruning_type == "han") {
    /*************** For Han Song's method ******************/
    if (this->iter_ == 0) {
      for (unsigned int k = 0; k < this->blobs_[0]->count(); ++k) {
        if (fabs(weight[k]) >= this->pruning_threshold) {
          weightMask[k] = 1;
	    } else if (fabs(weight[k]) < this->pruning_threshold) {
          weightMask[k] = 0;
	    }
	  }
      // (FanYang) bias masks generation, we temporarilly do not support this
      // if (this->bias_term_) {
        // for (unsigned int k = 0; k < this->blobs_[1]->count(); ++k) {
          // if (fabs(bias[k]) >= pruning_threshold) {
            // biasMask[k] = 1;
	      // } else if (fabs(bias[k]) < pruning_threshold) {
            // biasMask[k] = 0;
	      // }
	    // }
	  // }
      // (FanYang) bias masks generation, we temporarilly do not support this
    /********************************************************/
    }
  }

  // Demonstrate the sparsity of compressed convolutional layer
  /********************************************************/
  if (this->iter_ % 1000 == 0 && this->iter_ != 0) {
    unsigned int ncount = 0;
    for (unsigned int k = 0; k < this->blobs_[0]->count(); ++k) {
      if (weightMask[k]*weight[k] != 0) {
        ncount++;
      }
    }
    if (this->bias_term_) {
      for (unsigned int k = 0; k < this->blobs_[1]->count(); ++k) {
        if (biasMask[k]*bias[k] != 0) {
          ncount++;
        }
      }
    }
    LOG(INFO) << "Current non-zero count: " << ncount;
    LOG(INFO) << "Current sparsity ratio: " << 1. - ncount / float(this->blobs_[0]->count());
  }
  /********************************************************/
    
  // Calculate the current (masked) weight and bias
  for (unsigned int k = 0; k < this->blobs_[0]->count(); ++k) {
    weightTmp[k] = weight[k]*weightMask[k];
  }
  if (this->bias_term_) {
    for (unsigned int k = 0; k < this->blobs_[1]->count(); ++k) {
      biasTmp[k] = bias[k]*biasMask[k];
    }
  }
  
  // Forward calculation with (masked) weight and bias 
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + bottom[i]->offset(n), weightTmp,
          top_data + top[i]->offset(n));
      if (this->bias_term_) {
        this->forward_cpu_bias(top_data + top[i]->offset(n), biasTmp);
      }
    }
  }
}

template <typename Dtype>
void CConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weightTmp = this->weight_tmp_.cpu_data();  
  // const Dtype* weightMask = this->blobs_[2]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();    
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      // const Dtype* biasMask = this->blobs_[3]->cpu_data();
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();			
      // for (unsigned int k = 0; k < this->blobs_[1]->count(); ++k) {
        // bias_diff[k] = bias_diff[k]*biasMask[k];
      // }
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + top[i]->offset(n));
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->cpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();	
      // for (unsigned int k = 0; k < this->blobs_[0]->count(); ++k) {
        // weight_diff[k] = weight_diff[k]*weightMask[k];
      // }
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + bottom[i]->offset(n),
              top_diff + top[i]->offset(n), weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + top[i]->offset(n), weightTmp,
              bottom_diff + bottom[i]->offset(n));
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(CConvolutionLayer);
#endif

INSTANTIATE_CLASS(CConvolutionLayer);
REGISTER_LAYER_CLASS(CConvolution);

}  // namespace caffe
