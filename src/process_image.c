#include "image.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

float three_way_max(float a, float b, float c) {
  return (a > b) ? ((a > c) ? a : c) : ((b > c) ? b : c);
}

float three_way_min(float a, float b, float c) {
  return (a < b) ? ((a < c) ? a : c) : ((b < c) ? b : c);
}

image float_to_image(float *data, int c, int h, int w) {
  image out = {0};
  out.data = data;
  out.w = w;
  out.h = h;
  out.c = c;
  return out;
}

// returns the pixel at (c,h,w) in the input image im
// and checks image bounds with clamp padding strategy:
//   if asking for a pixel at column -3, use column 0
//   if asking for a pixel at column 300 for a 256x256 image,
//     use column 255
float get_pixel(image im, int c, int h, int w) {
  c = (c < 0) ? 0 : ((c >= im.c) ? (im.c - 1) : c);
  h = (h < 0) ? 0 : ((h >= im.h) ? (im.h - 1) : h);
  w = (w < 0) ? 0 : ((w >= im.w) ? (im.w - 1) : w);
  return im.data[c * im.h * im.w + h * im.w + w];
}

void set_pixel(image im, int c, int h, int w, float v) {
  if ((c < im.c && c >= 0) && (h < im.h && h >= 0) && (w < im.w && w >= 0)) {
    im.data[c * im.h * im.w + h * im.w + w] = v;
  }
}

image get_channel(image im, int c) {
  c = (c < 0) ? 0 : ((c >= im.c) ? (im.c - 1) : c);
  image channel = make_image(1, im.h, im.w);
  for (int i = 0; i < im.h * im.w; i++) {
    channel.data[i] = im.data[c * im.h * im.w + i];
  }
  return channel;
}

image copy_image(image im) {
  image copy = make_image(im.c, im.h, im.w);
  memcpy(copy.data, im.data, im.c * im.h * im.w * sizeof(float));
  return copy;
}

// calculates luminance
image rgb_to_grayscale(image im) {
  assert(im.c == 3);
  image gray = make_image(1, im.h, im.w);
  for (int i = 0; i < im.h * im.w; i++) {
    gray.data[i] = 0.299 * im.data[i] + 0.587 * im.data[im.h * im.w + i] +
                   0.114 * im.data[2 * im.h * im.w + i];
  }
  return gray;
}

image grayscale_to_rgb(image im, float r, float g, float b) {
  image rgb = make_image(3, im.h, im.w);
  for (int i = 0; i < im.h * im.w; i++) {
    float v = im.data[i];
    rgb.data[i] = r * v;
    rgb.data[im.h * im.w + i] = g * v;
    rgb.data[2 * im.h * im.w + i] = b * v;
  }
  return rgb;
}

void threshold_image(image im, float thresh) {
  for (int i = 0; i < im.c * im.h * im.w; i++) {
    im.data[i] = (im.data[i] <= thresh) ? 0.0 : 1.0;
  }
}

void shift_image(image im, int c, float v) {
  c = (c < 0) ? 0 : ((c >= im.c) ? (im.c - 1) : c);
  for (int i = 0; i < im.h * im.w; i++) {
    im.data[c * im.h * im.w + i] += v;
  }
}

// clamps image values to [0.0, 1.0]
void clamp_image(image im) {
  int i;
  for (i = 0; i < im.c * im.h * im.w; i++) {
    float v = im.data[i];
    v = (v > 1.0) ? 1.0 : ((v < 0.0) ? 0.0 : v);
    im.data[i] = v;
  }
}

void correct_gamma(image im, float gamma, float A) {
  int i;
  for (i = 0; i < im.c * im.h * im.w; i++) {
    im.data[i] = A * pow(im.data[i], gamma);
  }
}

int same_image(image a, image b, float eps) {
  assert(a.c == b.c && a.h == b.h && a.w == b.w);
  for (int i = 0; i < a.c * a.h * a.w; i++) {
    float thresh = (fabs(a.data[i]) + fabs(b.data[i])) * eps / 2.0;
    eps = (thresh > eps) ? thresh : eps;
    if (!((a.data[i] - eps) < b.data[i] && b.data[i] < (a.data[i] + eps))) {
      return 0;
    }
  }
  return 1;
}

image add_image(image a, image b) {
  assert(a.c == b.c && a.h == b.h && a.w == b.w);
  int i;
  image c = make_image(a.c, a.h, a.w);
  for (i = 0; i < a.c * a.h * a.w; i++) {
    c.data[i] = a.data[i] + b.data[i];
  }
  return c;
}

image sub_image(image a, image b) {
  assert(a.c == b.c && a.h == b.h && a.w == b.w);
  int i;
  image c = make_image(a.c, a.h, a.w);
  for (i = 0; i < a.c * a.h * a.w; i++) {
    c.data[i] = a.data[i] - b.data[i];
  }
  return c;
}

// Lerps between two images
// image a: first image
// image b: second image
// float k: weight of the interpolation
// returns: image c containing interpolated pixels
image blend_image(image a, image b, float k) {
  assert(a.c == b.c && a.h == b.h && a.w == b.w);
  int i;
  image c = make_image(a.c, a.h, a.w);
  for (i = 0; i < a.c * a.h * a.w; i++) {
    c.data[i] = (1.0 - k) * a.data[i] + k * b.data[i];
  }
  return c;
}

void invert_image(image im) {
  int i;
  for (i = 0; i < im.c * im.h * im.w; i++) {
    im.data[i] = 1 - im.data[i];
  }
}

void rgb_to_hsv(image im) {
  int i;
  float R, G, B, V, C, S, H;
  for (i = 0; i < im.h * im.w; i++) {
    R = im.data[i];
    G = im.data[im.h * im.w + i];
    B = im.data[2 * im.h * im.w + i];

    V = three_way_max(R, G, B);
    C = V - three_way_min(R, G, B);
    S = (V == 0.0) ? 0.0 : (C / V);

    H = 0.0;
    if (V == R) {
      H = (G - B) / C;
    } else if (V == G) {
      H = (B - R) / C + 2.0;
    } else if (V == B) {
      H = (R - G) / C + 4.0;
    }
    H = (C == 0.0) ? 0.0 : ((H < 0.0) ? (H / 6.0 + 1.0) : (H / 6.0));

    im.data[i] = H;
    im.data[im.h * im.w + i] = S;
    im.data[2 * im.h * im.w + i] = V;
  }
}

void scale_image(image im, int c, float v) {
  int i;
  c = (c < 0) ? 0 : ((c >= im.c) ? (im.c - 1) : c);
  for (i = 0; i < im.h * im.w; i++) {
    im.data[c * im.h * im.w + i] *= v;
  }
}

void hsv_to_rgb(image im) {
  int i;
  float H, V, C, m, X;
  for (i = 0; i < im.h * im.w; i++) {
    H = 6.0 * im.data[i];
    V = im.data[2 * im.h * im.w + i];
    C = V * im.data[im.h * im.w + i];
    m = V - C;
    X = C * (1.0 - fabs(fmod(H, 2.0) - 1));
    if ((H >= 0.0) && (H < 1.0)) {
      im.data[i] = C + m;
      im.data[im.h * im.w + i] = X + m;
      im.data[2 * im.h * im.w + i] = m;
    } else if ((H >= 1.0) && (H < 2.0)) {
      im.data[i] = X + m;
      im.data[im.h * im.w + i] = C + m;
      im.data[2 * im.h * im.w + i] = m;
    } else if ((H >= 2.0) && (H < 3.0)) {
      im.data[i] = m;
      im.data[im.h * im.w + i] = C + m;
      im.data[2 * im.h * im.w + i] = X + m;
    } else if ((H >= 3.0) && (H < 4.0)) {
      im.data[i] = m;
      im.data[im.h * im.w + i] = X + m;
      im.data[2 * im.h * im.w + i] = C + m;
    } else if ((H >= 4.0) && (H < 5.0)) {
      im.data[i] = X + m;
      im.data[im.h * im.w + i] = m;
      im.data[2 * im.h * im.w + i] = C + m;
    } else if ((H >= 5.0) && (H < 6.0)) {
      im.data[i] = C + m;
      im.data[im.h * im.w + i] = m;
      im.data[2 * im.h * im.w + i] = X + m;
    } else {
      im.data[i] = V;
      im.data[im.h * im.w + i] = V;
      im.data[2 * im.h * im.w + i] = V;
    }
  }
}

void rgb_to_xyz(image im) {
  int i;
  for (i = 0; i < im.h * im.w; i++) {
    float R = im.data[i];
    float G = im.data[im.h * im.w + i];
    float B = im.data[2 * im.h * im.w + i];
    im.data[i] = 0.490 * R + 0.310 * G + 0.200 * B;
    im.data[im.h * im.w + i] = 0.177 * R + 0.812 * G + 0.011 * B;
    im.data[2 * im.h * im.w + i] = 0.010 * G + 0.990 * B;
  }
}

void xyz_to_rgb(image im) {
  int i;
  for (i = 0; i < im.h * im.w; i++) {
    float X = im.data[i];
    float Y = im.data[im.h * im.w + i];
    float Z = im.data[2 * im.h * im.w + i];
    im.data[i] = 2.365 * X - 0.897 * Y - 0.468 * Z;
    im.data[im.h * im.w + i] = -0.515 * X + 1.426 * Y + 0.089 * Z;
    im.data[2 * im.h * im.w + i] = 0.005 * X + 0.014 * Y + 1.009 * Z;
  }
}

// Converts CIEXYZ to CIELAB
// https://en.wikipedia.org/wiki/CIELAB_color_space
void xyz_to_lab(image im) {
  int i;
  float d = 6.0 / 29.0;
  for (i = 0; i < im.h * im.w; i++) {
    // Using Standard Illuminant D65
    float tx = im.data[i] / 95.0489;
    float ty = im.data[im.h * im.w + i] / 100.0;
    float tz = im.data[2 * im.h * im.w + i] / 108.8840;

    float fx = (tx > d * d * d) ? powf(tx, 1.0 / 3.0)
                                : tx / (3.0 * d * d) + 4.0 / 29.0;
    float fy = (ty > d * d * d) ? powf(ty, 1.0 / 3.0)
                                : ty / (3.0 * d * d) + 4.0 / 29.0;
    float fz = (tz > d * d * d) ? powf(tz, 1.0 / 3.0)
                                : tz / (3.0 * d * d) + 4.0 / 29.0;

    im.data[i] = 116 * fy - 16;
    im.data[im.h * im.w + i] = 500 * (fx - fy);
    im.data[2 * im.h * im.w + i] = 200 * (fy - fz);
  }
}

void lab_to_xyz(image im) {
  int i;
  float d = 6.0 / 29.0;
  for (i = 0; i < im.h * im.w; i++) {
    float L = im.data[i];
    float a = im.data[im.h * im.w + i];
    float b = im.data[2 * im.h * im.w + i];

    float tx = (L + 16.0) / 116.0 + a / 500.0;
    float ty = (L + 16.0) / 116.0;
    float tz = (L + 16.0) / 116.0 - b / 200.0;

    float fx = (tx > d) ? (tx * tx * tx) : 3.0 * d * d * (tx - 4.0 / 29.0);
    float fy = (ty > d) ? (ty * ty * ty) : 3.0 * d * d * (ty - 4.0 / 29.0);
    float fz = (tz > d) ? (tz * tz * tz) : 3.0 * d * d * (tz - 4.0 / 29.0);

    im.data[i] = 95.0489 * fx;
    im.data[im.h * im.w + i] = 100.0 * fy;
    im.data[2 * im.h * im.w + i] = 108.8840 * fz;
  }
}

// Creates an n-element histogram of a grayscale image intensities
// image im: single channel input image
// int n: number of bins in historgram
// returns: array h of frequencies
int *get_histogram_image(image im, int n) {
  assert(im.c == 1);

  int i;
  int *h = calloc(n, sizeof(n));
  memset(h, 0, n * sizeof(int));
  float min = 999999;
  float max = -999999;

  for (i = 0; i < im.h * im.w; i++) {
    float v = im.data[i];
    min = (v < min) ? v : min;
    max = (v > max) ? v : max;
  }

  float bw = (max - min) / n;
  for (i = 0; i < im.h * im.w; i++) {
    // To map to a bin: normalize to 0...1, multiply by n
    // This is a linear mapping to n-bins, I guess
    // Probably not a very good one but idk any better rn
    h[(int)floor((im.data[i] - min) / bw)] += 1;
  }

  return h;
}