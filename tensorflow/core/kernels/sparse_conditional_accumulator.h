/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_KERNELS_SPARSE_CONDITIONAL_ACCUMULATOR_H_
#define TENSORFLOW_CORE_KERNELS_SPARSE_CONDITIONAL_ACCUMULATOR_H_

#include "tensorflow/core/kernels/typed_conditional_accumulator_base.h"
#include "tensorflow/core/platform/logging.h"
#include <map>

namespace tensorflow {

/**
 * An aggregation object for adding sparse gradients, represented as a tuple of
 * indices, values, and a (possibly empty) shape.
 *
 * The two main methods of this class are TryApplyGrad and TryTakeGrad.
 *
 * TryApplyGrad tries add a gradient to the accumulator. The attempt is
 * successful if local_step >= global_step, i.e., if the gradient is not stale,
 * having been computed using up-to-date information. Otherwise, the gradient is
 * silently dropped.
 *
 * TryTakeGrad logs an attempt to read the average gradient. The attempt is
 * blocked until the number of gradients accumulated (via TryApplyGrad) is equal
 * or exceeds the number requested by TryTakeGrad.
 * Once this condition is satisfied, the following actions are taken:
 * (1) the value of the average gradient is returned
 * (2) the count of accumulated gradients is reset to 0
 * (3) the internal global_step value (current_global_step_) is incremented by 1
 *
 * SparseConditionalAccumulator is the datatype-dependent templated sub-class of
 * ConditionalAccumulatorBase. It implements the virtual arithmetic methods that
 * are used by for aggregating, averaging, allocating, returning indexed slices.
 */
template <typename Device, typename T>
class SparseConditionalAccumulator
    : public TypedConditionalAccumulatorBase<
          std::tuple<const Tensor*, const Tensor*, const Tensor*>> {
 public:
  SparseConditionalAccumulator(const DataType& dtype,
                               const PartialTensorShape& shape,
                               const string& name)
      : TypedConditionalAccumulatorBase<
            std::tuple<const Tensor*, const Tensor*, const Tensor*>>(
            dtype, shape, name) {
    LOG(INFO) << "Constructor";
    accum_idx_val_val_persistent_map_ = new std::map<int64, std::pair<Tensor*, PersistentTensor*>>();
    count_element_map_ = new std::map<int64, int>();
    LOG(INFO) << "Constructor";
  }

  ~SparseConditionalAccumulator() override {
    if (accum_idx_val_val_persistent_map_ != nullptr) {
        for (std::map<int64, std::pair<Tensor*, PersistentTensor*>>::iterator it= (*accum_idx_val_val_persistent_map_).begin(); it!=(*accum_idx_val_val_persistent_map_).end(); ++it) {
            if ( it->second.second != nullptr) delete it->second.second;
        }
        delete accum_idx_val_val_persistent_map_;
    }
    if (count_element_map_ != nullptr) delete count_element_map_;
  };

 protected:
  std::map<int64, std::pair<Tensor*, PersistentTensor*>>* accum_idx_val_val_persistent_map_ = nullptr;
  std::map<int64, int>* count_element_map_ = nullptr;

  typedef Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>,
                           Eigen::Unaligned>
      SliceT;
  typedef Eigen::TensorMap<Eigen::Tensor<const T, 1, Eigen::RowMajor>,
                           Eigen::Unaligned>
      SliceConstT;

  Status ValidateShape(
      std::tuple<const Tensor*, const Tensor*, const Tensor*>* tensor,
      bool has_known_shape) EXCLUSIVE_LOCKS_REQUIRED(this->mu_) {
    const Tensor* tensor_idx = std::get<0>(*tensor);
    const Tensor* tensor_val = std::get<1>(*tensor);
    const Tensor* tensor_shape = std::get<2>(*tensor);
    int64 grad_val_dims = tensor_val->dims();
    int64 grad_dims = grad_val_dims;

    // Compare with provided shape
    if (has_known_shape) {
      if (shape_.dims() > tensor_shape->NumElements()) {
        return errors::InvalidArgument(
            "Shape mismatch: expected shape rank at least ", shape_.dims(),
            ", got ", tensor_shape->NumElements());
      }
      const auto tensor_shape_flat = tensor_shape->flat<int64>();
      for (int64 i = 0; i < shape_.dims(); i++) {
        if (shape_.dim_size(i) != -1 &&
            shape_.dim_size(i) != tensor_shape_flat(i)) {
          return errors::InvalidArgument("Shape mismatch: expected shape dim ",
                                         i, " to be ", shape_.dim_size(i),
                                         ", got ", tensor_shape_flat(i));
        }
      }
    }
    // Check that indices are within limits
    if (shape_.dims() > 0 && shape_.dim_size(0) != -1 &&
        tensor_idx->dims() > 0) {
      for (int64 i = 0; i < tensor_idx->dim_size(0); i++) {
        if (tensor_idx->vec<int64>()(i) >= shape_.dim_size(0)) {
          return errors::InvalidArgument(
              "Shape mismatch: index of slice ", i, " exceeded limits of shape",
              "; index is ", tensor_idx->vec<int64>()(i), " exceeded ",
              shape_.dim_size(0));
        }
      }
    }

    // Check values compatibility with accumulated gradient if available
    if (counter_ > 0) {
      Tensor* accum_map_val_first_ = (*accum_idx_val_val_persistent_map_).begin()->second.first;
      // Note that the new tensor's shape is also a matrix which has one column
      int64 accum_val_map_dims = accum_map_val_first_->dims();
      if (accum_val_map_dims != grad_val_dims) {
        return errors::InvalidArgument("Shape mismatch: expected values rank ",
                                       accum_val_map_dims, ", got ", grad_val_dims, "**CAUSED IN MAP IMPLEMENTATION**");
      }
      for (int64 i = 1; i < accum_val_map_dims; i++) {
        if (accum_map_val_first_->dim_size(i) != tensor_val->dim_size(i)) {
          return errors::InvalidArgument("Shape mismatch: expected values dim ",
                                         i, " to be ", accum_map_val_first_->dim_size(i),
                                         ", got ", tensor_val->dim_size(i), "**CAUSED IN MAP IMPLEMENTATION**");
        }
      }
    } else {
      // If there are no accumulated gradients, check against shape_
      if (shape_.dims() > grad_dims) {
        return errors::InvalidArgument(
            "Shape mismatch: expected values rank at least ", shape_.dims(),
            ", got ", grad_dims);
      }
      // Check that values have correct dimensions
      for (int64 i = 1; i < shape_.dims(); i++) {
        if (shape_.dim_size(i) != -1 &&
            shape_.dim_size(i) != tensor_val->dim_size(i)) {
          return errors::InvalidArgument("Shape mismatch: expected values dim ",
                                         i, " to be ", shape_.dim_size(i),
                                         ", got ", tensor_val->dim_size(i));
        }
      }
    }

    return Status::OK();
  }

  void AllocateAndAssignToAccumGradFunction(
      OpKernelContext* ctx,
      std::tuple<const Tensor*, const Tensor*, const Tensor*>* grad) override {
    const Tensor* grad_idx = std::get<0>(*grad);
    const Tensor* grad_val = std::get<1>(*grad);

    const int64 nnz = grad_idx->dim_size(0);

    LOG(INFO) << "Allocate";

    // Assign indices and values to accum_idx_val_val_persistent_map_
    if (accum_idx_val_val_persistent_map_ != nullptr) {
        for (std::map<int64, std::pair<Tensor*, PersistentTensor*>>::iterator it= (*accum_idx_val_val_persistent_map_).begin(); it!=(*accum_idx_val_val_persistent_map_).end(); ++it) {
            if ( it->second.second != nullptr) delete it->second.second;
            LOG(INFO) << "Allocate : inside iterator";
        }
        delete accum_idx_val_val_persistent_map_;
    }
    accum_idx_val_val_persistent_map_ = new std::map<int64, std::pair<Tensor*, PersistentTensor*>>();

    TensorShape tensor_shape = grad_val->shape();
    tensor_shape.set_dim(0, 1);

    auto grad_flat = grad_val->flat_outer_dims<T>();

    const int num_col = grad_flat.dimension(1);

    LOG(INFO) << "Allocate - num_col : " << num_col;

    Eigen::array<long, 2> extent = {1, num_col};
    Eigen::array<long, 2> extent2 = {1, num_col<3 ? num_col : 3};
 
    // Assign count_element_map_
    if (count_element_map_ != nullptr) {
        delete count_element_map_;
    }

    LOG(INFO) << "Allocate nnz : " << nnz;

    count_element_map_ = new std::map<int64, int>();
    
    for (int64 i = 0; i < nnz; i++) {
        Tensor* temp_accum_val_ = nullptr;
        PersistentTensor* temp_accum_val_persistent_ = new PersistentTensor();
        // TODO(b/32704451): Don't just ignore the ::tensorflow::Status object!
        ctx->allocate_persistent(dtype_, tensor_shape, temp_accum_val_persistent_,
                             &temp_accum_val_)
            .IgnoreError();
        
        Eigen::array<long, 2> offset = {i, 0};

        Eigen::array<long, 2> offset2 = {0, 0};

        temp_accum_val_->flat_outer_dims<T>().device(ctx->template eigen_device<Device>()) =
            grad_flat.slice(offset, extent).reshape(Eigen::array<long, 2>({1, num_col}));

        LOG(INFO) << "Allocate : grad_idx->vec<int64>()(i) : " << grad_idx->vec<int64>()(i);

        LOG(INFO) << "Allocate : temp_accum_val_ : " << temp_accum_val_->flat_outer_dims<T>().slice(offset2, extent2);

        (*accum_idx_val_val_persistent_map_)[grad_idx->vec<int64>()(i)] = std::make_pair(temp_accum_val_, temp_accum_val_persistent_); 
        (*count_element_map_)[grad_idx->vec<int64>()(i)] = 1; 
    }

    LOG(INFO) << "Allocate";
    // Do not need shape; Assume that the op has checked that the shapes match,
    // so grad's shape == shape_
  }

  void AddToAccumGradFunction(
      OpKernelContext* ctx,
      std::tuple<const Tensor*, const Tensor*, const Tensor*>* grad) override {
    // Modeled after third_party/tensorflow/core/kernels/sparse_add_op

    const Tensor* grad_idx = std::get<0>(*grad);
    const Tensor* grad_val = std::get<1>(*grad);

    const int64 grad_nnz = grad_idx->dim_size(0);

    auto grad_flat = grad_val->flat_outer_dims<T>();
    const int64 num_col = grad_flat.dimension(1);
    Eigen::DSizes<Eigen::DenseIndex, 1> slice_shape(num_col);

    LOG(INFO) << "Add";

    LOG(INFO) << "Add : num_col : " << num_col;

    int64 j = 0;
    {
        std::map<int64, std::pair<Tensor*, PersistentTensor*>>::iterator it = (*accum_idx_val_val_persistent_map_).begin();
        
        TensorShape tensor_shape = grad_val->shape();
        tensor_shape.set_dim(0, 1);

        Eigen::array<long, 2> extent = {1, num_col};
        Eigen::array<long, 2> extent2 = {1, num_col<3 ? num_col : 3};

        while (j < grad_nnz) {
            int64 b = grad_idx->vec<int64>()(j); 
            it = accum_idx_val_val_persistent_map_->lower_bound(b);

        LOG(INFO) << "Add : grad_idx->vec<int64>()(j) : " << grad_idx->vec<int64>()(j);
            // Element is a sum of accumulated value and new gradient;
            // compute sum here
            if (it != accum_idx_val_val_persistent_map_->end() && it->first == b) {
                const T* grad_slice_ptr = &grad_flat(j, 0);
                SliceConstT grad_slice(grad_slice_ptr, slice_shape);
                T* accum_slice_ptr = &(it->second.first->flat_outer_dims<T>())(0, 0);
                SliceT accum_slice(accum_slice_ptr, slice_shape);
                accum_slice = grad_slice + accum_slice;

        LOG(INFO) << "Add : (*count_element_map_)[it->first]" << (*count_element_map_)[it->first];
                (*count_element_map_)[it->first] += 1;

                Eigen::array<long, 2> offset2 = {0, 0};

        LOG(INFO) << "Add : it->second.first->flat_outer_dims<T>() : " << it->second.first->flat_outer_dims<T>().slice(offset2, extent2);

        LOG(INFO) << "Add : (*count_element_map_)[it->first]" << (*count_element_map_)[it->first];


            }
            else {
            // Element comes from new gradient; make a copy of indices and values
                Tensor* temp_accum_val_ = nullptr;
                PersistentTensor* temp_accum_val_persistent_ = new PersistentTensor();
                ctx->allocate_persistent(dtype_, tensor_shape, temp_accum_val_persistent_,
                                     &temp_accum_val_)
                    .IgnoreError();
    
                Eigen::array<long, 2> offset = {j, 0};
                Eigen::array<long, 2> offset2 = {0, 0};

                temp_accum_val_->flat<T>().device(ctx->template eigen_device<Device>()) =
                    grad_flat.slice(offset, extent).reshape(Eigen::array<long, 2>({1, num_col}));
    
                (*accum_idx_val_val_persistent_map_).insert(it, std::make_pair(grad_idx->vec<int64>()(j), std::make_pair(temp_accum_val_, temp_accum_val_persistent_))); 
                (*count_element_map_)[it->first] = 1;

        LOG(INFO) << "Add : (*accum_idx_val_val_persistent_map_)[grad_idx->vec<int64>()(j)].first->flat_outer_dims<T>() : " << (*accum_idx_val_val_persistent_map_)[grad_idx->vec<int64>()(j)].first->flat_outer_dims<T>().slice(offset2, extent2);

            }
            j++; 
        }
    }

    LOG(INFO) << "Add";
    // No need to copy shape, since shape remains the same after sum.
  }

  void DivideAccumGradByCounter(OpKernelContext* ctx, int average_option) override
      EXCLUSIVE_LOCKS_REQUIRED(this->mu_) {
    const int64 nnz = accum_idx_val_val_persistent_map_->size();

    Tensor* accum_val_tensor_ = nullptr;
    Tensor* accum_map_val_first_ = (*accum_idx_val_val_persistent_map_).begin()->second.first;
    TensorShape accum_val_shape = accum_map_val_first_->shape();
    accum_val_shape.set_dim(0, nnz);
    PersistentTensor* tensor_accum_val_persistent = new PersistentTensor();
    OP_REQUIRES_OK(
        ctx, ctx->allocate_persistent(dtype_, accum_val_shape, tensor_accum_val_persistent,
                                      &accum_val_tensor_));

    auto accum_val_flat = accum_val_tensor_->flat_outer_dims<T>();
    const int64 num_col = (accum_map_val_first_->flat_outer_dims<T>()).dimension(1);
    LOG(INFO) << "Divide";

    {
        Eigen::DSizes<Eigen::DenseIndex, 1> slice_shape(num_col);
        std::map<int64, std::pair<Tensor*, PersistentTensor*>>::iterator it;
        int64 i;
        for (it = (*accum_idx_val_val_persistent_map_).begin(), i = 0; it!=(*accum_idx_val_val_persistent_map_).end(); ++it, ++i) {
            T* accum_val_slice_ptr = &accum_val_flat(i, 0);
            SliceT accum_val_slice(accum_val_slice_ptr, slice_shape);

            T* accum_slice_ptr = &(it->second.first->flat_outer_dims<T>())(0, 0);
            SliceT accum_slice(accum_slice_ptr, slice_shape);
            accum_val_slice.device(ctx->template eigen_device<Device>()) = accum_slice;
        }
    }

    std::vector<int>* count_element_ = new std::vector<int>();    
    count_element_->reserve(nnz);

    {
        std::map<int64, int>::iterator it;
        for (it = (*count_element_map_).begin(); it!=(*count_element_map_).end(); ++it) {
            count_element_->push_back(it->second);
        }
    }

    auto accum_flat = accum_val_tensor_->flat_outer_dims<T>();
    std::vector<T> count_typet;
    std::transform(count_element_->begin(), count_element_->end(),
                   std::back_inserter(count_typet),
                   TypeConverter<T, int>::ConvertUToT);

    switch(average_option) {
      case 1:
        // Option 1: divide all by counter
    
        std::transform(
          &accum_flat(0,0), &accum_flat(nnz,0), &accum_flat(0,0),
          std::bind2nd(std::divides<T>(),
                       TypeConverter<T, int>::ConvertUToT(this->counter_)));
      break;
      case 2:    
        // Option 2: average element-wise
        Eigen::DSizes<Eigen::DenseIndex, 1> slice_shape(accum_flat.dimension(1));
        for (int64 i = 0; i < nnz; i++) {
        T* accum_slice_ptr = &accum_flat(i, 0);
        SliceT accum_slice(accum_slice_ptr, slice_shape);
        accum_slice.device(ctx->template eigen_device<Device>()) =
            accum_slice / count_typet[i];
        }
      break;
    }

    //accum_val_tensor_ -> map
    //below assignment might not be needed
    accum_val_flat = accum_val_tensor_->flat_outer_dims<T>();
    {
        Eigen::DSizes<Eigen::DenseIndex, 1> slice_shape(num_col);
        std::map<int64, std::pair<Tensor*, PersistentTensor*>>::iterator it;
        int64 i;
        for (it = (*accum_idx_val_val_persistent_map_).begin(), i = 0; it!=(*accum_idx_val_val_persistent_map_).end() ; ++it, ++i) {
            T* accum_slice_ptr = &(it->second.first->flat_outer_dims<T>())(0, 0);
            SliceT accum_slice(accum_slice_ptr, slice_shape);
            T* accum_val_slice_ptr = &accum_val_flat(i, 0);
            SliceT accum_val_slice(accum_val_slice_ptr, slice_shape);

            accum_slice.device(ctx->template eigen_device<Device>()) = accum_val_slice;
        }
    }

    if (count_element_ != nullptr) delete count_element_;
    LOG(INFO) << "Divide";
  }

  bool SetOutput(OpKernelContext* ctx) override {
    bool is_successful = true;
    LOG(INFO) << "SetOutput";
    if (is_successful) is_successful = ReturnIdxTensor(ctx);
    LOG(INFO) << "SetOutput1";
    if (is_successful) is_successful = ReturnValTensor(ctx);
    LOG(INFO) << "SetOutput2";
    if (is_successful) is_successful = ReturnShapeTensor(ctx);
    LOG(INFO) << "SetOutput3";
    return is_successful;
  }

  bool GetAndValidateTensorInputForApplyGrad(
      OpKernelContext* ctx,
      std::tuple<const Tensor*, const Tensor*, const Tensor*>** tensor) override
      EXCLUSIVE_LOCKS_REQUIRED(this->mu_) {
    // TODO(xinghao, jmchen): The roundabout way of getting attr from
    // OpKernelContext (instead of OpKernelConstruction) is a hack, and should
    // be fixed if it affects efficiency.
    bool has_known_shape = false;
    OP_REQUIRES_OK_BOOLEAN(
        ctx, GetNodeAttr(ctx->op_kernel().def(), "has_known_shape",
                         &has_known_shape));

    // Get input gradient tensors
    const Tensor* grad_idx_tensor;
    OP_REQUIRES_OK_BOOLEAN(ctx,
                           ctx->input("gradient_indices", &grad_idx_tensor));
    const Tensor* grad_val_tensor;
    OP_REQUIRES_OK_BOOLEAN(ctx,
                           ctx->input("gradient_values", &grad_val_tensor));
    const Tensor* grad_shape_tensor = nullptr;
    if (has_known_shape) {
      OP_REQUIRES_OK_BOOLEAN(ctx,
                             ctx->input("gradient_shape", &grad_shape_tensor));
    }

    // Checks
    OP_REQUIRES_BOOLEAN(
        ctx, TensorShapeUtils::IsVector(grad_idx_tensor->shape()),
        errors::InvalidArgument(
            "Input indices should be vector but received shape: ",
            grad_idx_tensor->shape().DebugString()));
    const int64 nnz = grad_idx_tensor->dim_size(0);
    OP_REQUIRES_BOOLEAN(
        ctx, grad_val_tensor->dims() > 0,
        errors::InvalidArgument("Values cannot be 0-dimensional."));
    OP_REQUIRES_BOOLEAN(ctx, grad_val_tensor->dim_size(0) == nnz,
                        errors::InvalidArgument("Expected ", nnz,
                                                " non-empty input values, got ",
                                                grad_val_tensor->dim_size(0)));

    *tensor = new std::tuple<const Tensor*, const Tensor*, const Tensor*>(
        grad_idx_tensor, grad_val_tensor, grad_shape_tensor);

    OP_REQUIRES_OK_BOOLEAN(ctx, this->ValidateShape(*tensor, has_known_shape));

    return true;
  }

  void CleanUpGradTensor(std::tuple<const Tensor*, const Tensor*,
                                    const Tensor*>* tensor) override {
    if (tensor != nullptr) delete tensor;
  }

 private:
  inline bool ReturnIdxTensor(OpKernelContext* ctx) {
    Tensor* idx_tensor;
    const int64 nnz = accum_idx_val_val_persistent_map_->size();
    OP_REQUIRES_OK_BOOLEAN(ctx, ctx->allocate_output(0, {nnz}, &idx_tensor));
    // If allocate_output fails, OP_REQUIRES_OK_BOOLEAN will short-circuit
    // the remaining code and just return false
    auto idx_tensor_vec = idx_tensor->vec<int64>();
    {
        std::map<int64, std::pair<Tensor*, PersistentTensor*>>::iterator it;
        int64 i;
        for (it = (*accum_idx_val_val_persistent_map_).begin(), i = 0 ; it!=(*accum_idx_val_val_persistent_map_).end() ; ++it, ++i) {
            idx_tensor_vec(i) = it->first;
        }
    }

    LOG(INFO) << "ReturnIdx - idx : " << idx_tensor->flat<int64>();

    return true;
  }

  inline bool ReturnValTensor(OpKernelContext* ctx) {
    Tensor* accum_val_tensor = nullptr;
    const int64 nnz = accum_idx_val_val_persistent_map_->size();
    LOG(INFO) << "ReturnVal - nnz : " << nnz;
    Tensor* accum_map_val_first_ = (*accum_idx_val_val_persistent_map_).begin()->second.first;
    TensorShape accum_val_shape = accum_map_val_first_->shape();
    accum_val_shape.set_dim(0, nnz);
    OP_REQUIRES_OK_BOOLEAN(ctx, ctx->allocate_output(1, accum_val_shape, &accum_val_tensor));
    auto accum_val_flat = accum_val_tensor->flat_outer_dims<T>();
    const int64 num_col = (accum_map_val_first_->flat_outer_dims<T>()).dimension(1);

    LOG(INFO) << "ReturnVal - num_col : " << num_col;

    Eigen::DSizes<Eigen::DenseIndex, 1> slice_shape(num_col);
    {
        std::map<int64, std::pair<Tensor*, PersistentTensor*>>::iterator it;
        int64 i;
        for (it = (*accum_idx_val_val_persistent_map_).begin(), i = 0; it!=(*accum_idx_val_val_persistent_map_).end(); ++it, ++i) {
            T* accum_val_slice_ptr = &accum_val_flat(i, 0);
            SliceT accum_val_slice(accum_val_slice_ptr, slice_shape);

            T* accum_slice_ptr = &(it->second.first->flat_outer_dims<T>())(0, 0);
            SliceT accum_slice(accum_slice_ptr, slice_shape);
            accum_val_slice = accum_slice;
        }
    }

    LOG(INFO) << "ReturnVal - val : " << accum_val_tensor->flat_outer_dims<T>();

    return true;
  }

  inline bool ReturnShapeTensor(OpKernelContext* ctx) {
    Tensor* accum_map_val_first_ = (*accum_idx_val_val_persistent_map_).begin()->second.first;
    int64 accum_val_map_dims = accum_map_val_first_->dims();
    Tensor* shape_tensor;
    OP_REQUIRES_OK_BOOLEAN(
        ctx, ctx->allocate_output(2, {accum_val_map_dims}, &shape_tensor));
    // If allocate_output fails, OP_REQUIRES_OK_BOOLEAN will short-circuit
    // the remaining code and just return false

    LOG(INFO) << "ReturnShape";

    // First dim of shape is defined by shape_, others by accum_val_->shape
    shape_tensor->flat<int64>()(0) =
        (shape_.dims() > 0) ? shape_.dim_size(0) : -1;
    for (int64 i = 1; i < accum_val_map_dims; i++) {
      shape_tensor->flat<int64>()(i) = accum_map_val_first_->dim_size(i);
    }

    LOG(INFO) << "ReturnShape - shape : " << shape_tensor->flat<int64>();

    return true;
  }

  TF_DISALLOW_COPY_AND_ASSIGN(SparseConditionalAccumulator);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SPARSE_CONDITIONAL_ACCUMULATOR_H_
