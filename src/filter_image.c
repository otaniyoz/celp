#include "image.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Normalizes image in-place to sum to 1 in-place
// image im: input image
void l1_normalize(image im) {
  int i;
  float s = 0.0;
  for (i = 0; i < im.c * im.h * im.w; i++) {
    s += im.data[i];
  }
  for (i = 0; i < im.c * im.h * im.w; i++) {
    im.data[i] /= s;
  }
}

image make_box_filter(int w) {
  int i;
  image box = make_image(1, w, w);
  for (i = 0; i < w * w; i++) {
    box.data[i] = 1.0;
  }
  l1_normalize(box);
  return box;
}

// Performs weighted cross-correlation between two images
// image im: input image
// image filter: convolution filter or kernel
// int preserve: whether to preserve input image dimensions or not
// returns: image convolved
image convolve_image(image im, image filter, int preserve) {
  assert(filter.c == 1 || filter.c == im.c);

  float q;
  int k, j, i, y, x, fy, fx;
  image convolved = make_image(im.c, im.h, im.w);
  for (k = 0; k < im.c; k++) {
    for (j = 0; j < im.h; j++) {
      for (i = 0; i < im.w; i++) {
        q = 0.0;
        for (y = 0; y < filter.h; y++) {
          fy = j - filter.h / 2 + y;
          for (x = 0; x < filter.w; x++) {
            fx = i - filter.w / 2 + x;
            q += get_pixel(im, k, fy, fx) * get_pixel(filter, k, y, x);
          }
        }
        set_pixel(convolved, k, j, i, q);
      }
    }
  }
  if (!preserve) {
    image new_convolved = make_image(1, im.h, im.w);
    for (i = 0; i < im.w; i++) {
      for (j = 0; j < im.h; j++) {
        q = 0.0;
        for (k = 0; k < im.c; k++)
          q += get_pixel(convolved, k, j, i);
        set_pixel(new_convolved, 0, j, i, q);
      }
    }
    return new_convolved;
  }
  return convolved;
}

image make_highpass_filter() {
  image highpass = make_image(1, 3, 3);
  set_pixel(highpass, 0, 0, 0, 0.0);
  set_pixel(highpass, 0, 0, 1, -1.0);
  set_pixel(highpass, 0, 0, 2, 0.0);
  set_pixel(highpass, 0, 1, 0, -1.0);
  set_pixel(highpass, 0, 1, 1, 4.0);
  set_pixel(highpass, 0, 1, 2, -1.0);
  set_pixel(highpass, 0, 2, 0, 0.0);
  set_pixel(highpass, 0, 2, 1, -1.0);
  set_pixel(highpass, 0, 2, 2, 0.0);
  return highpass;
}

image make_sharpen_filter() {
  image sharpen = make_highpass_filter();
  set_pixel(sharpen, 0, 1, 1, 5.0);
  return sharpen;
}

image make_emboss_filter() {
  image emboss = make_highpass_filter();
  set_pixel(emboss, 0, 0, 0, -2.0);
  set_pixel(emboss, 0, 0, 1, -1.0);
  set_pixel(emboss, 0, 1, 0, -1.0);
  set_pixel(emboss, 0, 1, 1, 1.0);
  set_pixel(emboss, 0, 1, 2, 1.0);
  set_pixel(emboss, 0, 2, 1, 1.0);
  set_pixel(emboss, 0, 2, 2, 2.0);
  return emboss;
}

image make_gaussian_filter(float sigma) {
  float sq = sigma * sigma;
  float d1 = TWOPI * sq;
  float d2 = 2.0 * sq;
  int w = ((int)sigma * 6) | 1;
  if (w % 2 == 0)
    w++;

  int i, j;
  image g = make_image(1, w, w);
  for (j = 0; j < w; j++) {
    for (i = 0; i < w; i++) {
      g.data[j * w + i] =
          exp(-((i - w / 2) * (i - w / 2) + (j - w / 2) * (j - w / 2)) / d2) /
          d1;
    }
  }
  l1_normalize(g);
  return g;
}

image make_gx_filter() {
  image gx = make_image(1, 3, 3);
  set_pixel(gx, 0, 0, 0, -1.0);
  set_pixel(gx, 0, 0, 1, 0.0);
  set_pixel(gx, 0, 0, 2, 1.0);
  set_pixel(gx, 0, 1, 0, -2.0);
  set_pixel(gx, 0, 1, 1, 0.0);
  set_pixel(gx, 0, 1, 2, 2.0);
  set_pixel(gx, 0, 2, 0, -1.0);
  set_pixel(gx, 0, 2, 1, 0.0);
  set_pixel(gx, 0, 2, 2, 1.0);
  return gx;
}

image make_gy_filter() {
  image gy = make_image(1, 3, 3);
  set_pixel(gy, 0, 0, 0, -1.0);
  set_pixel(gy, 0, 0, 1, -2.0);
  set_pixel(gy, 0, 0, 2, -1.0);
  set_pixel(gy, 0, 1, 0, 0.0);
  set_pixel(gy, 0, 1, 1, 0.0);
  set_pixel(gy, 0, 1, 2, 0.0);
  set_pixel(gy, 0, 2, 0, 1.0);
  set_pixel(gy, 0, 2, 1, 2.0);
  set_pixel(gy, 0, 2, 2, 1.0);
  return gy;
}

void feature_normalize(image im) {
  float max = im.data[0];
  float min = im.data[0];
  float p;
  int i;
  for (i = 0; i < im.c * im.h * im.w; i++) {
    p = im.data[i];
    min = (p < min) ? p : min;
    max = (p > max) ? p : max;
  }
  for (i = 0; i < im.c * im.h * im.w; i++) {
    im.data[i] = (max - min == 0.0) ? 0.0 : (im.data[i] - min) / (max - min);
  }
}

// Applies Sobel operators to an image and calculates magnitude and angle
// image im: input image
// returns: array of two images: [magnitudes, angles]
image *sobel_image(image im) {
  int i;
  image gx = make_gx_filter();
  image gy = make_gy_filter();
  image sx = convolve_image(im, gx, 0);
  image sy = convolve_image(im, gy, 0);
  image *sobel = calloc(2, sizeof(image));
  sobel[0] = make_image(1, im.h, im.w);
  sobel[1] = make_image(1, im.h, im.w);
  for (i = 0; i < im.h * im.w; i++) {
    sobel[0].data[i] = sqrtf(sx.data[i] * sx.data[i] + sy.data[i] * sy.data[i]);
    sobel[1].data[i] = atan2f(sy.data[i], sx.data[i]);
  }
  free_image(gx);
  free_image(gy);
  free_image(sx);
  free_image(sy);
  return sobel;
}

image colorize_sobel(image im) {
  int i, j;

  im = convolve_image(im, make_emboss_filter(), 1);
  im = smooth_image(im, 4);

  image *sobel = sobel_image(im);
  feature_normalize(sobel[0]);
  feature_normalize(sobel[1]);
  image color = make_image(3, im.h, im.w);
  for (i = 0; i < im.w; i++) {
    for (j = 0; j < im.h; j++) {
      set_pixel(color, 0, j, i, get_pixel(sobel[1], 0, j, i));
      set_pixel(color, 1, j, i, get_pixel(sobel[0], 0, j, i));
      set_pixel(color, 2, j, i, get_pixel(sobel[0], 0, j, i));
    }
  }
  hsv_to_rgb(color);

  free_image(sobel[0]);
  free_image(sobel[1]);
  free(sobel);

  return color;
}

// Helper function for the line integral convolution
void static update(float vx, float vy, int *x, int *y, float *fx, float *fy,
                   int w, int h) {
  float tx, ty;

  if (vx >= 0) {
    tx = (1.0 - *fx) / vx;
  } else {
    tx = -*fx / vx;
  }

  if (vy >= 0) {
    ty = (1.0 - *fy) / vy;
  } else {
    ty = -*fy / vy;
  }

  if (tx < ty) {
    if (vx >= 0) {
      *x++;
      *fx = 0;
    } else {
      *x--;
      *fx = 1;
    }
    *fy += tx * vy;
  } else {
    if (vy >= 0) {
      *y++;
      *fy = 0;
    } else {
      *y--;
      *fy = 1;
    }
    *fx += ty * vx;
  }
}

// Calculates line integral convolution.
// Taken from https://github.com/scipy/scipy-cookbook
image convolve_line_integral(image vectors, image im, image filter) {
  int r = filter.w;
  int h = vectors.h;
  int w = vectors.w;
  image res = make_image(1, h, w);

  float q, fx, fy;
  int i, j, x, y, k;
  for (j = 0; j < h; j++) {
    for (i = 0; i < w; i++) {
      q = 0;
      x = i;
      y = j;
      fx = 0.5;
      fy = 0.5;
      k = r / 2;
      q += filter.data[k] * get_pixel(im, 0, j, i);
      while (k < (r - 1)) {
        k++;
        update(get_pixel(vectors, 0, j, i), get_pixel(vectors, 1, j, i), &x, &y,
               &fx, &fy, w, h);
        q += filter.data[k] * get_pixel(im, 0, j, i);
      }

      x = i;
      y = j;
      fx = 0.5;
      fy = 0.5;
      while (k > 0) {
        k--;
        update(-get_pixel(vectors, 0, j, i), -get_pixel(vectors, 1, j, i), &x,
               &y, &fx, &fy, w, h);
        q += filter.data[k] * get_pixel(im, 0, j, i);
      }
      set_pixel(res, 0, j, i, q);
    }
  }
  return res;
}

// Compares two floats. Helper function for qsort
int static float_compare(const void *a, const void *b) {
  float x = *(const float *)a;
  float y = *(const float *)b;
  if (x < y)
    return -1;
  else if (x > y)
    return 1;
  else
    return 0;
}

// Applies in-place pixel sort to an image. Based on Kim's Processing script:
// https://github.com/kimasendorf/ASDFPixelSort
// image im: input image
// float low: lower bound for pixels to be sorted
// float high: upper bpund for pixels to be
// sorted int axis: 0 -> horizontal sort, 1 -> vertical sort, otherwise -> no
// sort int mode: 0 -> gray, 1 -> hue, otherwise -> red channel
void sort_pixels(image im, float low, float high, int axis, int mode) {
  assert(im.c == 3);
  image values = copy_image(im);
  if (mode == 0) {
    values = rgb_to_grayscale(im);
  } else if (mode == 1) {
    rgb_to_hsv(values);
  }

  int i, j, k;
  if (axis == 0) {
    for (i = 0; i < im.h; i++) {
      int j0 = 0;
      int jn = 0;
      while (jn < (im.w - 1)) {
        while (get_pixel(values, 0, i, j0) <= low && j0 < (im.w - 1)) {
          j0++;
        }
        while (get_pixel(values, 0, i, jn) > high && jn < (im.w - 1)) {
          jn++;
        }
        int n = jn - j0 + 1;
        n = (n >= 0) ? n : 0; // just to make sure size is valid
        float *span = calloc(n, sizeof(float));
        for (k = 0; k < im.c; k++) {
          for (j = 0; j < n; j++) {
            span[j] = get_pixel(im, k, i, j0 + j);
          }
          qsort(span, n, sizeof(float), float_compare);
          for (j = 0; j < n; j++) {
            set_pixel(im, k, i, j0 + j, span[j]);
          }
        }
        free(span);
        jn++;
        j0 = jn;
      }
    }
  } else if (axis == 1) {
    for (i = 0; i < im.w; i++) {
      int j0 = 0;
      int jn = 0;
      while (jn < (im.h - 1)) {
        while (get_pixel(values, 0, j0, i) <= low && j0 < (im.h - 1)) {
          j0++;
        }
        while (get_pixel(values, 0, jn, i) > high && jn < (im.h - 1)) {
          jn++;
        }
        int n = jn - j0 + 1;
        n = (n >= 0) ? n : 0;
        float *span = calloc(n, sizeof(float));
        for (k = 0; k < im.c; k++) {
          for (j = 0; j < n; j++) {
            span[j] = get_pixel(im, k, j0 + j, i);
          }
          qsort(span, n, sizeof(float), float_compare);
          for (j = 0; j < n; j++) {
            set_pixel(im, k, j0 + j, i, span[j]);
          }
        }
        free(span);
        jn++;
        j0 = jn;
      }
    }
  }
  free_image(values);
}

// TODO: implement deep image prior
// https://github.com/DmitryUlyanov/deep-image-prior