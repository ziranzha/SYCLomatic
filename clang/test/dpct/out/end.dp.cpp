
// ===-------- text_experimental_obj_surf.cu ------- *- CUDA -* ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>

#define PRINT_PASS 1

using namespace std;

int passed = 0;
int failed = 0;

void checkResult(string name, bool IsPassed) {
  cout << name;
  if (IsPassed) {
    cout << " ---- passed" << endl;
    passed++;
  } else {
    cout << " ---- failed" << endl;
    failed++;
  }
}

template <typename T, typename EleT>
void kernel1(EleT *output, sycl::ext::oneapi::experimental::sampled_image_handle surf, int w) {
  for (int i = 0; i < w; ++i) {
    auto ret = syclcompat::experimental::sample_image<T>(surf, float(i * sizeof(T)));
    output[i] = ret;
  }
}

template <typename T, typename EleT>
void kernel1(EleT *output, sycl::ext::oneapi::experimental::sampled_image_handle surf, int w, int h) {
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      auto ret = syclcompat::experimental::sample_image<T>(surf, sycl::float2(j * sizeof(T), i));
      output[w * i + j] = ret;
    }
  }
}

template <typename T, typename EleT>
void kernel1(EleT *output, sycl::ext::oneapi::experimental::sampled_image_handle surf, int w, int h,
             int d) {
  for (int i = 0; i < d; ++i) {
    for (int j = 0; j < h; ++j) {
      for (int k = 0; k < w; ++k) {
        auto ret = syclcompat::experimental::sample_image<T>(surf, sycl::float3(k * sizeof(T), j, i));
        output[w * h * i + w * j + k] = ret;
      }
    }
  }
}

template <typename T, typename EleT>
void kernel2(EleT *output, sycl::ext::oneapi::experimental::sampled_image_handle surf, int w) {
  for (int i = 0; i < w; ++i) {
    auto ret = syclcompat::experimental::sample_image<T>(surf, float(i * sizeof(T)));
    output[2 * i] = ret.x();
    output[2 * i + 1] = ret.y();
  }
}

template <typename T, typename EleT>
void kernel2(EleT *output, sycl::ext::oneapi::experimental::sampled_image_handle surf, int w, int h) {
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      auto ret = syclcompat::experimental::sample_image<T>(surf, sycl::float2(j * sizeof(T), i));
      output[2 * (w * i + j)] = ret.x();
      output[2 * (w * i + j) + 1] = ret.y();
    }
  }
}

template <typename T, typename EleT>
void kernel2(EleT *output, sycl::ext::oneapi::experimental::sampled_image_handle surf, int w, int h,
             int d) {
  for (int i = 0; i < d; ++i) {
    for (int j = 0; j < h; ++j) {
      for (int k = 0; k < w; ++k) {
        auto ret = syclcompat::experimental::sample_image<T>(surf, sycl::float3(k * sizeof(T), j, i));
        output[2 * (w * h * i + w * j + k)] = ret.x();
        output[2 * (w * h * i + w * j + k) + 1] = ret.y();
      }
    }
  }
}

template <typename T, typename EleT>
void kernel4(EleT *output, sycl::ext::oneapi::experimental::sampled_image_handle surf, int w) {
  for (int i = 0; i < w; ++i) {
    auto ret = syclcompat::experimental::sample_image<T>(surf, float(i * sizeof(T)));
    output[4 * i] = ret.x();
    output[4 * i + 1] = ret.y();
    output[4 * i + 2] = ret.z();
    output[4 * i + 3] = ret.w();
  }
}

template <typename T, typename EleT>
void kernel4(EleT *output, sycl::ext::oneapi::experimental::sampled_image_handle surf, int w, int h) {
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      auto ret = syclcompat::experimental::sample_image<T>(surf, sycl::float2(j * sizeof(T), i));
      output[4 * (w * i + j)] = ret.x();
      output[4 * (w * i + j) + 1] = ret.y();
      output[4 * (w * i + j) + 2] = ret.z();
      output[4 * (w * i + j) + 3] = ret.w();
    }
  }
}

template <typename T, typename EleT>
void kernel4(EleT *output, sycl::ext::oneapi::experimental::sampled_image_handle surf, int w, int h,
             int d) {
  for (int i = 0; i < d; ++i) {
    for (int j = 0; j < h; ++j) {
      for (int k = 0; k < w; ++k) {
        auto ret = syclcompat::experimental::sample_image<T>(surf, sycl::float3(k * sizeof(T), j, i));
        output[4 * (w * h * i + w * j + k)] = ret.x();
        output[4 * (w * h * i + w * j + k) + 1] = ret.y();
        output[4 * (w * h * i + w * j + k) + 2] = ret.z();
        output[4 * (w * h * i + w * j + k) + 3] = ret.w();
      }
    }
  }
}

template <typename T, typename ArrT>
dpct::experimental::image_mem_wrapper *getInput(ArrT &expect, size_t w, const dpct::image_channel &desc) {
  dpct::experimental::image_mem_wrapper *input;
  input = new dpct::experimental::image_mem_wrapper(desc, w, 0);
  dpct::experimental::dpct_memcpy(input, 0, 0, expect, sizeof(T) * w, sizeof(T) * w,
                                  1);
  return input;
}

template <typename T, typename ArrT>
dpct::experimental::image_mem_wrapper *getInput(ArrT &expect, size_t w, size_t h,
                                                const dpct::image_channel &desc) {
  dpct::experimental::image_mem_wrapper *input;
  input = new dpct::experimental::image_mem_wrapper(desc, w, h);
  dpct::experimental::dpct_memcpy(input, 0, 0, expect, sizeof(T) * w, sizeof(T) * w, h);
  return input;
}

template <typename T, typename ArrT>
dpct::experimental::image_mem_wrapper *getInput(ArrT &expect, size_t w, size_t h, size_t d,
                                                const dpct::image_channel &desc) {
  dpct::experimental::image_mem_wrapper *input;
  input = new dpct::experimental::image_mem_wrapper(desc, {w, h, d}, sycl::ext::oneapi::experimental::image_type::standard);
  dpct::memcpy_parameter p = {};
  p.from.pitched = dpct::pitched_data(expect, w * sizeof(T), w, h);
  p.to.image_bindless = input;
  p.size = sycl::range<3>(w, h, d);
  p.direction = dpct::host_to_device;
  dpct::dpct_memcpy(p);
  return input;
}

sycl::ext::oneapi::experimental::sampled_image_handle getSurf(dpct::experimental::image_mem_wrapper_ptr input) {
  dpct::image_data resDesc;
  memset(&resDesc, 0, sizeof(resDesc));

  resDesc.set_data(input);

  sycl::ext::oneapi::experimental::sampled_image_handle surf;
  surf = dpct::experimental::create_bindless_image(resDesc);

  return surf;
}

int main() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  bool pass = true;

  { // 1 element uint 1D image.
    const int w = 8;
    uint32_t expect[w] = {
        {1},
        {2},
        {3},
        {4},
        {5},
        {6},
        {7},
        {8},
    };
    auto *input = getInput<uint32_t>(expect, w, dpct::image_channel::create<uint32_t>());
    unsigned int *output;
    output = (unsigned int *)sycl::malloc_shared(sizeof(expect), q_ct1);
    auto surf = getSurf(input);
    q_ct1.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
        [=](sycl::nd_item<3> item_ct1) {
          kernel1<uint32_t>(output, surf, w);
        });
    dev_ct1.queues_wait_and_throw();
    dpct::experimental::destroy_bindless_image(surf, q_ct1);
    delete input;
    for (int i = 0; i < w; ++i) {
      if (output[i] != expect[i]) {
        pass = false;
        break;
      }
    }
    checkResult("uint1-1D", pass);
    if (PRINT_PASS || !pass) {
      for (int i = 0; i < w; ++i)
        cout << "{" << output[i] << "}, ";
      cout << endl;
    }
    dpct::dpct_free(output, q_ct1);
    pass = true;
  }

  { // 1 element int 2D image.
    const int h = 3;
    const int w = 2;
    int32_t expect[h * w] = {
        {1}, {2}, // 1
        {3},
        {4}, // 2
        {5},
        {6}, // 3
    };
    auto *input = getInput<int32_t>(expect, w, h, dpct::image_channel::create<int32_t>());
    int *output;
    output = (int *)sycl::malloc_shared(sizeof(expect), q_ct1);
    auto surf = getSurf(input);
    q_ct1.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
        [=](sycl::nd_item<3> item_ct1) {
          kernel1<int32_t>(output, surf, w, h);
        });
    dev_ct1.queues_wait_and_throw();
    dpct::experimental::destroy_bindless_image(surf, q_ct1);
    delete input;
    for (int i = 0; i < w * h; ++i) {
      if (output[i] != expect[i]) {
        pass = false;
        break;
      }
    }
    checkResult("int1-2D", pass);
    if (PRINT_PASS || !pass)
      for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j)
          cout << "{" << (int)output[w * i + j] << "}, ";
        cout << endl;
      }
    dpct::dpct_free(output, q_ct1);
    pass = true;
  }

  { // 1 element short 3D image.
    const int d = 2;
    const int h = 3;
    const int w = 2;
    int16_t expect[d * h * w] = {
        {1}, {2}, // 1.1
        {3},
        {4}, // 1.2
        {5},
        {6}, // 1.3

        {7},
        {8}, // 2.1
        {9},
        {10}, // 2.2
        {11},
        {12}, // 2.3
    };
    auto *input =
        getInput<int16_t>(expect, w, h, d, dpct::image_channel::create<int16_t>());
    short *output;
    output = (short *)sycl::malloc_shared(sizeof(expect), q_ct1);
    auto surf = getSurf(input);
    q_ct1.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
        [=](sycl::nd_item<3> item_ct1) {
          kernel1<int16_t>(output, surf, w, h, d);
        });
    dev_ct1.queues_wait_and_throw();
    dpct::experimental::destroy_bindless_image(surf, q_ct1);
    delete input;
    for (int i = 0; i < w * h * d; ++i) {
      if (output[i] != expect[i]) {
        pass = false;
        break;
      }
    }
    checkResult("short1-3D", pass);
    if (PRINT_PASS || !pass)
      for (int i = 0; i < d; ++i) {
        for (int j = 0; j < h; ++j) {
          for (int k = 0; k < w; ++k)
            cout << "{" << output[w * h * i + j * w + k] << "}, ";
          cout << endl;
        }
        cout << endl;
      }
    dpct::dpct_free(output, q_ct1);
    pass = true;
  }

  { // 2 element char 1D image.
    const int w = 4;
    sycl::char2 expect[w] = {
        {1, 2},
        {3, 4},
        {5, 6},
        {7, 8},
    };
    auto *input = getInput<sycl::char2>(expect, w, dpct::image_channel::create<sycl::char2>());
    char *output;
    output = (char *)sycl::malloc_shared(sizeof(expect), q_ct1);
    auto surf = getSurf(input);
    q_ct1.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
        [=](sycl::nd_item<3> item_ct1) {
          kernel2<sycl::char2>(output, surf, w);
        });
    dev_ct1.queues_wait_and_throw();
    dpct::experimental::destroy_bindless_image(surf, q_ct1);
    delete input;
    for (int i = 0; i < w; ++i) {
      if (output[2 * i] != expect[i].x() || output[2 * i + 1] != expect[i].y()) {
        pass = false;
        break;
      }
    }
    checkResult("char2-1D", pass);
    if (PRINT_PASS || !pass)
      for (int i = 0; i < w; ++i)
        cout << "{" << (int)output[2 * i] << ", " << (int)output[2 * i + 1]
             << "}, ";
    cout << endl;
    dpct::dpct_free(output, q_ct1);
    pass = true;
  }

  { // 2 element uchar 2D image.
    const int h = 3;
    const int w = 2;
    sycl::uchar2 expect[h * w] = {
        {1, 2}, {3, 4}, // 1
        {5, 6},
        {7, 8}, // 2
        {9, 10},
        {11, 12}, // 3
    };
    auto *input =
        getInput<sycl::uchar2>(expect, w, h, dpct::image_channel::create<sycl::uchar2>());
    unsigned char *output;
    output = (unsigned char *)sycl::malloc_shared(sizeof(expect), q_ct1);
    auto surf = getSurf(input);
    q_ct1.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
        [=](sycl::nd_item<3> item_ct1) {
          kernel2<sycl::uchar2>(output, surf, w, h);
        });
    dev_ct1.queues_wait_and_throw();
    dpct::experimental::destroy_bindless_image(surf, q_ct1);
    delete input;
    for (int i = 0; i < w * h; ++i) {
      if (output[2 * i] != expect[i].x() || output[2 * i + 1] != expect[i].y()) {
        pass = false;
        break;
      }
    }
    checkResult("uchar2-2D", pass);
    if (PRINT_PASS || !pass)
      for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j)
          cout << "{" << (int)output[2 * (w * i + j)] << ", "
               << (int)output[2 * (w * i + j) + 1] << "}, ";
        cout << endl;
      }
    dpct::dpct_free(output, q_ct1);
    pass = true;
  }

  { // 2 element ushort 3D image.
    const int d = 2;
    const int h = 3;
    const int w = 2;
    sycl::ushort2 expect[d * h * w] = {
        {1, 2}, {3, 4}, // 1.1
        {5, 6},
        {7, 8}, // 1.2
        {9, 10},
        {11, 12}, // 1.3

        {13, 14},
        {15, 16}, // 2.1
        {17, 18},
        {19, 20}, // 2.2
        {21, 22},
        {23, 24}, // 2.3
    };
    auto *input =
        getInput<sycl::ushort2>(expect, w, h, d, dpct::image_channel::create<sycl::ushort2>());
    unsigned short *output;
    output = (unsigned short *)sycl::malloc_shared(sizeof(expect), q_ct1);
    auto surf = getSurf(input);
    q_ct1.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
        [=](sycl::nd_item<3> item_ct1) {
          kernel2<sycl::ushort2>(output, surf, w, h, d);
        });
    dev_ct1.queues_wait_and_throw();
    dpct::experimental::destroy_bindless_image(surf, q_ct1);
    delete input;
    for (int i = 0; i < w * h * d; ++i) {
      if (output[2 * i] != expect[i].x() || output[2 * i + 1] != expect[i].y()) {
        pass = false;
        break;
      }
    }
    checkResult("ushort2-3D", pass);
    if (PRINT_PASS || !pass)
      for (int i = 0; i < d; ++i) {
        for (int j = 0; j < h; ++j) {
          for (int k = 0; k < w; ++k)
            cout << "{" << output[2 * (w * h * i + j * w + k)] << ", "
                 << output[2 * (w * h * i + j * w + k) + 1] << "}, ";
          cout << endl;
        }
        cout << endl;
      }
    dpct::dpct_free(output, q_ct1);
    pass = true;
  }

  { // 4 element float 1D image.
    const int w = 4;
    sycl::float4 expect[w] = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 16},
    };
    auto *input = getInput<sycl::float4>(expect, w, dpct::image_channel::create<sycl::float4>());
    float *output;
    output = (float *)sycl::malloc_shared(sizeof(expect), q_ct1);
    auto surf = getSurf(input);
    q_ct1.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
        [=](sycl::nd_item<3> item_ct1) {
          kernel4<sycl::float4>(output, surf, w);
        });
    dev_ct1.queues_wait_and_throw();
    dpct::experimental::destroy_bindless_image(surf, q_ct1);
    delete input;
    float precision = 0.001;
    for (int i = 0; i < w; ++i) {
      if ((output[4 * i] < expect[i].x() - precision ||
           output[4 * i] > expect[i].x() + precision) ||
          (output[4 * i + 1] < expect[i].y() - precision ||
           output[4 * i + 1] > expect[i].y() + precision) ||
          (output[4 * i + 2] < expect[i].z() - precision ||
           output[4 * i + 2] > expect[i].z() + precision) ||
          (output[4 * i + 3] < expect[i].w() - precision ||
           output[4 * i + 3] > expect[i].w() + precision)) {
        pass = false;
        break;
      }
    }
    checkResult("float4-1D", pass);
    if (PRINT_PASS || !pass)
      for (int i = 0; i < w; ++i)
        cout << "{" << output[4 * i] << ", " << output[4 * i + 1] << ", "
             << output[4 * i + 2] << ", " << output[4 * i + 3] << "}, ";
    cout << endl;
    dpct::dpct_free(output, q_ct1);
    pass = true;
  }

  { // 4 element int 2D image.
    const int h = 3;
    const int w = 2;
    sycl::int4 expect[h * w] = {
        {1, 2, 3, 4}, {5, 6, 7, 8}, // 1
        {9, 10, 11, 12},
        {13, 14, 15, 16}, // 2
        {17, 18, 19, 20},
        {21, 22, 23, 24}, // 3
    };
    auto *input = getInput<sycl::int4>(expect, w, h, dpct::image_channel::create<sycl::int4>());
    int *output;
    output = (int *)sycl::malloc_shared(sizeof(expect), q_ct1);
    auto surf = getSurf(input);
    q_ct1.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
        [=](sycl::nd_item<3> item_ct1) {
          kernel4<sycl::int4>(output, surf, w, h);
        });
    dev_ct1.queues_wait_and_throw();
    dpct::experimental::destroy_bindless_image(surf, q_ct1);
    delete input;
    for (int i = 0; i < w * h; ++i) {
      if (output[4 * i] != expect[i].x() || output[4 * i + 1] != expect[i].y() ||
          output[4 * i + 2] != expect[i].z() ||
          output[4 * i + 3] != expect[i].w()) {
        pass = false;
        break;
      }
    }
    checkResult("int4-2D", pass);
    if (PRINT_PASS || !pass)
      for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j)
          cout << "{" << output[4 * (w * i + j)] << ", "
               << output[4 * (w * i + j) + 1] << ", "
               << output[4 * (w * i + j) + 2] << ", "
               << output[4 * (w * i + j) + 3] << "}, ";
        cout << endl;
      }
    dpct::dpct_free(output, q_ct1);
    pass = true;
  }

  { // 4 element uint 3D image.
    const int d = 2;
    const int h = 3;
    const int w = 2;
    sycl::uint4 expect[d * h * w] = {
        {1, 2, 3, 4}, {5, 6, 7, 8}, // 1.1
        {9, 10, 11, 12},
        {13, 14, 15, 16}, // 1.2
        {17, 18, 19, 20},
        {21, 22, 23, 24}, // 1.3

        {25, 26, 27, 28},
        {29, 30, 31, 32}, // 2.1
        {33, 34, 35, 36},
        {37, 38, 39, 40}, // 2.2
        {41, 42, 43, 44},
        {45, 46, 47, 48}, // 2.3
    };
    auto *input =
        getInput<sycl::uint4>(expect, w, h, d, dpct::image_channel::create<sycl::uint4>());
    unsigned int *output;
    output = (unsigned int *)sycl::malloc_shared(sizeof(expect), q_ct1);
    auto surf = getSurf(input);
    q_ct1.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
        [=](sycl::nd_item<3> item_ct1) {
          kernel4<sycl::uint4>(output, surf, w, h, d);
        });
    dev_ct1.queues_wait_and_throw();
    dpct::experimental::destroy_bindless_image(surf, q_ct1);
    delete input;
    for (int i = 0; i < w * h * d; ++i) {
      if (output[4 * i] != expect[i].x() || output[4 * i + 1] != expect[i].y() ||
          output[4 * i + 2] != expect[i].z() ||
          output[4 * i + 3] != expect[i].w()) {
        pass = false;
        break;
      }
    }
    checkResult("uint4-3D", pass);
    if (PRINT_PASS || !pass)
      for (int i = 0; i < d; ++i) {
        for (int j = 0; j < h; ++j) {
          for (int k = 0; k < w; ++k)
            cout << "{" << output[4 * (w * h * i + j * w + k)] << ", "
                 << output[4 * (w * h * i + j * w + k) + 1] << ", "
                 << output[4 * (w * h * i + j * w + k) + 2] << ", "
                 << output[4 * (w * h * i + j * w + k) + 3] << "}, ";
          cout << endl;
        }
        cout << endl;
      }
    dpct::dpct_free(output, q_ct1);
    pass = true;
  }

  cout << "passed " << passed << "/" << passed + failed << " cases!" << endl;
  if (failed) {
    cout << "failed!" << endl;
  }
  return failed;
}

