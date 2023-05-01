#include "image.h"
#include "matrix.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Frees an array of descriptors.
// descriptor *d: the array.
// int n: number of elements in array.
void free_descriptors(descriptor *d, int n) {
  int i;
  for (i = 0; i < n; ++i) {
    free(d[i].data);
  }
  free(d);
}

// Create a feature descriptor for an index in an image.
// image im: source image.
// int i: index in image for the pixel we want to describe.
// returns: descriptor for that index.
descriptor describe_index(image im, int i) {
  int w = 5;
  descriptor d;
  d.p.x = i % im.w;
  d.p.y = i / im.w;
  d.data = calloc(w * w * im.c, sizeof(float));
  d.n = w * w * im.c;
  int c, dx, dy;
  int count = 0;
  // If you want you can experiment with other descriptors
  // This subtracts the central value from neighbors
  // to compensate some for exposure/lighting changes.
  for (c = 0; c < im.c; ++c) {
    float cval = im.data[c * im.w * im.h + i];
    for (dx = -w / 2; dx < (w + 1) / 2; ++dx) {
      for (dy = -w / 2; dy < (w + 1) / 2; ++dy) {
        float val = get_pixel(im, c, i / im.w + dy, i % im.w + dx);
        d.data[count++] = cval - val;
      }
    }
  }
  return d;
}

// Marks the spot of a point in an image.
// image im: image to mark.
// ponit p: spot to mark in the image.
void mark_spot(image im, point p) {
  int x = p.x;
  int y = p.y;
  int i;
  for (i = -9; i < 10; ++i) {
    set_pixel(im, 0, y, x + i, 1.00);
    set_pixel(im, 0, y + i, x, 1.00);
    set_pixel(im, 1, y, x + i, 0.47);
    set_pixel(im, 1, y + i, x, 0.47);
    set_pixel(im, 2, y, x + i, 0.78);
    set_pixel(im, 2, y + i, x, 0.78);
  }
}

// Marks corners denoted by an array of descriptors.
// image im: image to mark.
// descriptor *d: corners in the image.
// int n: number of descriptors to mark.
void mark_corners(image im, descriptor *d, int n) {
  int i;
  for (i = 0; i < n; ++i) {
    mark_spot(im, d[i].p);
  }
}

// Creates a 1d Gaussian filter.
// float sigma: standard deviation of Gaussian.
// returns: single row image of the filter.
image make_1d_gaussian(float sigma) {
  int i, x, w = ((int)(6 * sigma)) | 1;
  if (w % 2 == 0)
    w++;

  float d1 = 2.0 * sigma * sigma;
  float d2 = sqrtf(TWOPI) * sigma;
  image f = make_image(1, 1, w);
  for (i = 0; i < w; i++) {
    x = i - w / 2;
    set_pixel(f, 0, 0, i, expf(-(x * x) / d1) / d2);
  }
  l1_normalize(f);
  return f;
}

// Smooths an image with two-pass gaussian blur
// image im: image to smooth.
// float sigma: std dev. for Gaussian.
// returns: smoothed image.
image smooth_image(image im, float sigma) {
  int i;
  image gx = make_1d_gaussian(sigma);
  image gy = make_image(1, gx.w, 1);
  for (i = 0; i < gy.h; i++) {
    set_pixel(gy, 0, i, 0, get_pixel(gx, 0, 0, i));
  }
  image sx = convolve_image(im, gx, 1);
  image sy = convolve_image(sx, gy, 1);
  free_image(gx);
  free_image(gy);
  free_image(sx);
  return sy;
}

// Calculate the structure matrix of an image.
// image im: the input image.
// float sigma: std dev. to use for weighted sum.
// returns: structure matrix. 1st channel is Ix^2, 2nd channel is Iy^2,
//          third channel is IxIy.
image structure_matrix(image im, float sigma) {
  int i, j;
  float x, y;
  image fx = make_gx_filter();
  image fy = make_gy_filter();
  image ix = convolve_image(im, fx, 0);
  image iy = convolve_image(im, fy, 0);
  image S = make_image(3, im.h, im.w);
  for (i = 0; i < im.w; i++) {
    for (j = 0; j < im.h; j++) {
      x = get_pixel(ix, 0, j, i);
      y = get_pixel(iy, 0, j, i);
      set_pixel(S, 0, j, i, x * x);
      set_pixel(S, 1, j, i, y * y);
      set_pixel(S, 2, j, i, x * y);
    }
  }
  S = smooth_image(S, sigma);
  free_image(fx);
  free_image(fy);
  free_image(ix);
  free_image(iy);
  return S;
}

// Estimate the cornerness of each pixel given a structure matrix S.
// image S: structure matrix for an image.
// returns: a response map of cornerness calculations.
image cornerness_response(image S) {
  int i, j;
  float alpha = 0.06;
  float trace, det, xx, yy, xy;
  image R = make_image(1, S.h, S.w);
  for (i = 0; i < S.w; i++) {
    for (j = 0; j < S.h; j++) {
      xx = get_pixel(S, 0, j, i);
      yy = get_pixel(S, 1, j, i);
      xy = get_pixel(S, 2, j, i);

      trace = xx + yy;
      det = xx * yy - xy * xy;
      set_pixel(R, 0, j, i, det - alpha * trace * trace);
    }
  }
  return R;
}

// Perform non-max supression on an image of feature responses.
// image im: 1-channel image of feature responses.
// int w: distance to look for larger responses.
// returns: image with only local-maxima responses within w pixels.
image nms_image(image im, int w) {
  assert(im.c == 1);
  float px;
  int i, j, y, x;
  image r = copy_image(im);
  for (i = 0; i < im.w; i++) {
    for (j = 0; j < im.h; j++) {
      px = get_pixel(im, 0, j, i);
      for (y = j - w; y <= j + w; y++) {
        for (x = i - w; x <= i + w; x++) {
          if (get_pixel(im, 0, y, x) > px)
            set_pixel(r, 0, j, i, -999999);
        }
      }
    }
  }
  return r;
}

// Perform harris corner detection and extract features from the corners.
// image im: input image.
// float sigma: std. dev for harris.
// float thresh: threshold for cornerness.
// int nms: distance to look for local-maxes in response map.
// int *n: pointer to number of corners detected, should fill in.
// returns: array of descriptors of the corners in the image.
descriptor *harris_corner_detector(image im, float sigma, float thresh, int nms,
                                   int *n) {
  // Calculate structure matrix
  image S = structure_matrix(im, sigma);
  // Estimate cornerness
  image R = cornerness_response(S);
  // Run NMS on the responses
  image Rnms = nms_image(R, nms);
  // Count number of responses over threshold
  int i, j;
  int count = 0;
  int capacity = 64;
  int *corners = calloc(capacity, sizeof(int));
  for (j = 0; j < Rnms.h; j++) {
    for (i = 0; i < Rnms.w; i++) {
      if (get_pixel(Rnms, 0, j, i) >= thresh) {
        count++;
        if (count > capacity) {
          capacity = 2 * capacity;
          int *new_corners = calloc(capacity, sizeof(int));
          memcpy(new_corners, corners, (capacity / 2) * sizeof(int));
          free(corners);
          corners = new_corners;
        }
        corners[count - 1] = j * Rnms.w + i;
      }
    }
  }
  *n = count;
  descriptor *d = calloc(count, sizeof(descriptor));
  for (i = 0; i < count; i++)
    d[i] = describe_index(im, corners[i]);
  free(corners);
  free_image(S);
  free_image(R);
  free_image(Rnms);
  return d;
}

// Find and draw corners on an image.
// image im: input image.
// float sigma: std. dev for harris.
// float thresh: threshold for cornerness.
// int nms: distance to look for local-maxes in response map.
void detect_and_draw_corners(image im, float sigma, float thresh, int nms) {
  int n = 0;
  descriptor *d = harris_corner_detector(im, sigma, thresh, nms, &n);
  mark_corners(im, d, n);
}
