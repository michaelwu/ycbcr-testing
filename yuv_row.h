// Copyright (c) 2010 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// yuv_row internal functions to handle YUV conversion and scaling to RGB.
// These functions are used from both yuv_convert.cc and yuv_scale.cc.

// TODO(fbarchard): Write function that can handle rotation and scaling.

#ifndef MEDIA_BASE_YUV_ROW_H_
#define MEDIA_BASE_YUV_ROW_H_

extern "C" {

#define SIMD_ALIGNED(var) var __attribute__((aligned(16)))
extern SIMD_ALIGNED(int16_t kCoefficientsRgbY[768][4]);

}  // extern "C"

#endif  // MEDIA_BASE_YUV_ROW_H_
