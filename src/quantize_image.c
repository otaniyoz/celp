#include "image.h"
#include <assert.h>

// TODO: implement k-means color quantization
// https://en.wikipedia.org/wiki/K-means_clustering
// Limits color palette to a specified number of colors using k-means
// image im: input image
// int k: number of colors or clusters
// returns: image q with colors limited to a specified number

// Applies Otsu's thresholding to an image. It automatically calculates optimal
// threshold and separates pixels into two classes based on that threshold.
// Based on  https://en.wikipedia.org/wiki/Otsu%27s_method
// image im: single-channel input image with pixels in 0...1
// int n: number of bins in histogram
void apply_otsu_threshold(image im, int n) {
  assert(im.c == 1);

  int i, t, tb = 0;
  float sb = 0.0;

  // Image pixels are in range 0...1
  // To calculate histogram, scale them to 0...255
  scale_image(im, 0, 255.0);
  int *h = get_histogram_image(im, n);

  for (t = 0; t < n; t++) {
    float w0 = 0.0;
    float w1 = 0.0;
    float s0 = 0.0;
    float s1 = 0.0;

    for (i = 0; i < t; i++) {
      w0 += (float)h[i];
      s0 += (float)i * h[i];
    }
    for (i = t; i < n; i++) {
      w1 += (float)h[i];
      s1 += (float)i * h[i];
    }

    float s = w0 * (s0 / w0 - s0 - s1) * (s0 / w0 - s0 - s1) +
              w1 * (s1 / w1 - s0 - s1) * (s1 / w1 - s0 - s1);

    if (s > sb) {
      sb = s;
      tb = t;
    }
  }

  threshold_image(im, tb);
}

// TODO: implement Atkinson dithering
// https://en.wikipedia.org/wiki/Atkinson_dithering

// TODO: implement ordered dithering
// https://en.wikipedia.org/wiki/Ordered_dithering

// Applies in-place Floyd-Steinberg dithering to an image. Based on
// https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering
// image im: input image 
// image q: input image with colors limited to a specified number
void floyd_steinberg_dither(image im, image q) {
  assert(im.c == q.c && im.h == q.h && im.w == q.w);
  int k, j, i;
  for (j = 0; j < im.h; j++) {
    for (i = 0; i < im.w; i++) {
      for (k = 0; k < im.c; k++) {
        float old = get_pixel(im, k, j, i);
        float new = get_pixel(q, k, j, i);
        float error = old - new;
        set_pixel(im, k, j, i, new);
        set_pixel(im, k, j, i + 1,
                  get_pixel(im, k, j, i + 1) + 7.0 / 16.0 * error);
        set_pixel(im, k, j + 1, i - 1,
                  get_pixel(im, k, j + 1, i - 1) + 3.0 / 16.0 * error);
        set_pixel(im, k, j + 1, i,
                  get_pixel(im, k, j + 1, i) + 5.0 / 16.0 * error);
        set_pixel(im, k, j + 1, i + 1,
                  get_pixel(im, k, j + 1, i + 1) + 1.0 / 16.0 * error);
      }
    }
  }
}