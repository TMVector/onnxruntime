// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/math/det.h"
#include "core/util/math_cpuonly.h"

using namespace onnxruntime::common;

namespace onnxruntime {

ONNX_CPU_OPERATOR_KERNEL(
    Det,
    11,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Det<float>);

template <>
Status Det::Compute<float>(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  ORT_ENFORCE(X != nullptr);

  Tensor* Y = ctx->Output(0, X->Shape());

  EigenVectorMap<T>(Y->template MutableData<T>(), Y->Shape().Size()) =
      ConstEigenVectorMap<T>(X->template Data<T>(), X->Shape().Size())
          .cwiseMax(min_val)
          .cwiseMin(max_val);

  return Status::OK();
}
}  // namespace onnxruntime
