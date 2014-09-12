#include <algorithm>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include <emmintrin.h>

#include <png.h>

#include "yuv_row_table.cpp"

// TODO:
// Flexible downsample configuration
// UV line deinterlacing support
// Unaligned output widths
// Unaligned output addresses
// Scaling?

// Try to keep the struct here within a 64 byte cacheline
struct SSE2YUVConsts {
	const uint8_t byteconsts[16];
	const int16_t colorcoef[8];
	const int16_t wordconsts[8];
};

/* This allows the specification of consts used for YUV conversion.
 *
 * low:    Specifies the lowest valid luminance value.
 *         Values below this are saturated to zero.
 *         0 <= (int)low <= 255
 * high:   Specifies the difference between the highest valid
 *         luminance value and 255.
 *         Values above this are saturated to the highest valid value.
 *         0 <= (int)high <= 255
 * rvcoef: Red V constant.
 *         r = scaledY + rvcoef * scaledV
 *         0.0 <= rvcoef < 2.0
 * gucoef: Green U constant.
 *         g = scaledY + gucoef * scaledU + gvcoef * scaledV
 *         -2.0 < gucoef <= 0
 * gvcoef: Green V constant.
 *         g = scaledY + gucoef * scaledU + gvcoef * scaledV
 *         -2.0 < gvcoef <= 0
 * bucoef: Blue U constant.
 *         b = scaledY + bucoef * scaledU
 *         1.0 <= bucoef < 3.0
 *
 * Specifying coefficients outside the range will cause overflow
 * and/or rounding in the wrong direction.
 */

#define YUVCONSTS(name, low, high, rvcoef, gucoef, gvcoef, bucoef) \
	static const SSE2YUVConsts name __attribute__ ((aligned (64))) = { \
		{ low + high, low + high, low + high, low + high, \
		  high, high, high, high, \
		  0, 0, 0, 0, \
		  0, 0xFF, 0, 0xFF }, \
		{ static_cast<int16_t>((rvcoef * (1 << 7)) + 0.5f), \
                  static_cast<int16_t>((rvcoef * (1 << 7)) + 0.5f), \
		  static_cast<int16_t>((gucoef * (1 << 7)) - 0.5f), \
		  static_cast<int16_t>((gucoef * (1 << 7)) - 0.5f), \
		  static_cast<int16_t>((gvcoef * (1 << 7)) - 0.5f), \
		  static_cast<int16_t>((gvcoef * (1 << 7)) - 0.5f), \
		  static_cast<int16_t>(((bucoef - 1.0f) * (1 << 7)) + 0.5f), \
		  static_cast<int16_t>(((bucoef - 1.0f) * (1 << 7)) + 0.5f) }, \
		{ (0xFF << 7) / (0xFF - (low + high)), \
		  (0xFF << 7) / (0xFF - (low + high)), \
		  1 << 7, 1 << 7, \
		  1 << 6, 1 << 6, \
		  0, 0 },\
	}

YUVCONSTS(sBT601Consts, 16, 20, \
	  1.5960267857142858, \
	 -0.3917622900949137, \
	 -0.8129676472377708, \
	  2.017232142857143);

static inline void
convert_yuv_word_lines(__m128i yline,
		       __m128i uline,
		       __m128i vline,
		       const SSE2YUVConsts &consts,
		       char **outrgba)
{
	__m128i round = _mm_shuffle_epi32(*(const __m128i *)consts.wordconsts, 0xAA);

	// Convert blue
	__m128i bucoef = _mm_shuffle_epi32(*(const __m128i *)consts.colorcoef, 0xFF);
	__m128i buline = _mm_mullo_epi16(uline, bucoef);
	__m128i blueline = _mm_adds_epi16(yline, buline);
	buline = _mm_slli_epi16(uline, 7);
	blueline = _mm_adds_epi16(blueline, round);
	blueline = _mm_adds_epi16(blueline, buline);
	blueline = _mm_max_epi16(blueline, round);
	blueline = _mm_slli_epi16(blueline, 1);
	blueline = _mm_srli_epi16(blueline, 8);

	// Convert green
	__m128i gucoef = _mm_shuffle_epi32(*(const __m128i *)consts.colorcoef, 0x55);
	__m128i gvcoef = _mm_shuffle_epi32(*(const __m128i *)consts.colorcoef, 0xAA);
	__m128i guline = _mm_mullo_epi16(uline, gucoef);
	__m128i gvline = _mm_mullo_epi16(vline, gvcoef);
	__m128i guvline = _mm_adds_epi16(guline, gvline);

	__m128i greenline = _mm_adds_epi16(guvline, yline);
	greenline = _mm_adds_epi16(round, greenline);
	greenline = _mm_max_epi16(greenline, round);
	greenline = _mm_srli_epi16(greenline, 7);
	greenline = _mm_slli_epi16(greenline, 8);

	__m128i bluegreenline = _mm_or_si128(blueline, greenline);

	// Convert red
	__m128i rvcoef = _mm_shuffle_epi32(*(const __m128i *)consts.colorcoef, 0x00);
	__m128i rvline = _mm_mullo_epi16(vline, rvcoef);
	__m128i redline = _mm_adds_epi16(yline, rvline);
	redline = _mm_adds_epi16(round, redline);
	redline = _mm_max_epi16(redline, round);
	redline = _mm_srli_epi16(redline, 7);

	__m128i alphaline = _mm_shuffle_epi32(*(const __m128i *)consts.byteconsts, 0xFF);
	__m128i redalphaline = _mm_or_si128(redline, alphaline);
	__m128i rgbalinelow = _mm_unpacklo_epi16(bluegreenline, redalphaline);
	__m128i rgbalinehigh = _mm_unpackhi_epi16(bluegreenline, redalphaline);

	_mm_stream_si128(*(__m128i**)outrgba, rgbalinelow);
	_mm_stream_si128(*(__m128i**)outrgba + 1, rgbalinehigh);
	*outrgba += sizeof(__m128i) * 2;
}

static inline void
convert_yuv_byte_lines(__m128i full_yline,
		       __m128i full_uline,
		       __m128i full_vline,
		       const SSE2YUVConsts &consts,
		       char **outrgba)
{
	__m128i zero = _mm_setzero_si128();
	__m128i yunscaledline = _mm_unpacklo_epi8(full_yline, zero);
	__m128i uunscaledline = _mm_unpacklo_epi8(full_uline, zero);
	__m128i vunscaledline = _mm_unpacklo_epi8(full_vline, zero);

	__m128i yscaler = _mm_shuffle_epi32(*(const __m128i *)consts.wordconsts, 0x00);
	__m128i uvnormalizer = _mm_shuffle_epi32(*(const __m128i *)consts.wordconsts, 0x55);
	// Scale to 255 and convert to 9.7 fixed point
	__m128i yline = _mm_mullo_epi16(yunscaledline, yscaler);
	// Scale to -128 to 127
	__m128i uline = _mm_sub_epi16(uunscaledline, uvnormalizer);
	__m128i vline = _mm_sub_epi16(vunscaledline, uvnormalizer);

	convert_yuv_word_lines(yline, uline, vline, consts, outrgba);

	yunscaledline = _mm_unpackhi_epi8(full_yline, zero);
	uunscaledline = _mm_unpackhi_epi8(full_uline, zero);
	vunscaledline = _mm_unpackhi_epi8(full_vline, zero);

	// Scale to 255 and convert to 9.7 fixed point
	yline = _mm_mullo_epi16(yunscaledline, yscaler);
	// Scale to -128 to 127
	uline = _mm_sub_epi16(uunscaledline, uvnormalizer);
	vline = _mm_sub_epi16(vunscaledline, uvnormalizer);

	convert_yuv_word_lines(yline, uline, vline, consts, outrgba);
}

void
convert_to_rgba_sse2(
	const void *yplane,
	const void *uplane,
	const void *vplane,
	uint32_t ystride, uint32_t uvstride,
	uint32_t width, uint32_t height,
	const SSE2YUVConsts consts,
	char *outrgba)
{
	__m128i whiteclip = _mm_shuffle_epi32(*(const __m128i *)consts.byteconsts, 0x55);
	__m128i blackclip = _mm_shuffle_epi32(*(const __m128i *)consts.byteconsts, 0x00);

	__m128i *yrow = (__m128i *)yplane;
	__m128i *urow = (__m128i *)uplane;
	__m128i *vrow = (__m128i *)vplane;
	char *outrow = outrgba;

	for (uint32_t j = 0; j < height; j += 2) {
		uint32_t i = 0;
		__m128i *y = yrow;
		__m128i *y_odd = (__m128i *)(((char *)yrow) + ystride);
		__m128i *u = urow;
		__m128i *v = vrow;
		char *out = outrow;
		char *out_odd = outrow + width * 4;
		while (i < width) {
			/* Use saturation arithmetic to clamp y values */
			__m128i full_yline = _mm_adds_epu8(*y++, whiteclip);
			full_yline = _mm_subs_epu8(full_yline, blackclip);

			/* Convert high half */
			__m128i full_uline = _mm_unpacklo_epi8(*u, *u);
			__m128i full_vline = _mm_unpacklo_epi8(*v, *v);

			convert_yuv_byte_lines(full_yline,
					       full_uline,
					       full_vline,
					       consts,
					       &out);

			__m128i full_yline_odd = _mm_adds_epu8(*y_odd++, whiteclip);
			full_yline_odd = _mm_subs_epu8(full_yline_odd, blackclip);

			convert_yuv_byte_lines(full_yline_odd,
					       full_uline,
					       full_vline,
					       consts,
					       &out_odd);
			i += sizeof(__m128i);
			if (i >= width)
				break;

			/* Convert lower half */
			full_yline = _mm_adds_epu8(*y++, whiteclip);
			full_yline = _mm_subs_epu8(full_yline, blackclip);
			full_uline = _mm_unpackhi_epi8(*u, *u);
			full_vline = _mm_unpackhi_epi8(*v, *v);

			convert_yuv_byte_lines(full_yline,
					       full_uline,
					       full_vline,
					       consts,
					       &out);

			full_yline_odd = _mm_adds_epu8(*y_odd++, whiteclip);
			full_yline_odd = _mm_subs_epu8(full_yline_odd, blackclip);

			convert_yuv_byte_lines(full_yline_odd,
					       full_uline,
					       full_vline,
					       consts,
					       &out_odd);
			i += sizeof(__m128i);
			u++;
			v++;
		}
		yrow = (__m128i *)(((char *)yrow) + (ystride * 2));
		urow = (__m128i *)(((char *)urow) + uvstride);
		vrow = (__m128i *)(((char *)vrow) + uvstride);
		outrow += width * 4 * 2;
	}
}

void FastConvertYUVToRGB32Row(const uint8_t* y_buf,  // rdi
                              const uint8_t* u_buf,  // rsi
                              const uint8_t* v_buf,  // rdx
                              uint8_t* rgb_buf,      // rcx
                              int width) {         // r8
  asm(
  "jmp    1f\n"
"0:"
  "movzb  (%1),%%r10\n"
  "add    $0x1,%1\n"
  "movzb  (%2),%%r11\n"
  "add    $0x1,%2\n"
  "movq   2048(%5,%%r10,8),%%xmm0\n"
  "movzb  (%0),%%r10\n"
  "movq   4096(%5,%%r11,8),%%xmm1\n"
  "movzb  0x1(%0),%%r11\n"
  "paddsw %%xmm1,%%xmm0\n"
  "movq   (%5,%%r10,8),%%xmm2\n"
  "add    $0x2,%0\n"
  "movq   (%5,%%r11,8),%%xmm3\n"
  "paddsw %%xmm0,%%xmm2\n"
  "paddsw %%xmm0,%%xmm3\n"
  "shufps $0x44,%%xmm3,%%xmm2\n"
  "psraw  $0x6,%%xmm2\n"
  "packuswb %%xmm2,%%xmm2\n"
  "movq   %%xmm2,0x0(%3)\n"
  "add    $0x8,%3\n"
"1:"
  "sub    $0x2,%4\n"
  "jns    0b\n"

"2:"
  "add    $0x1,%4\n"
  "js     3f\n"

  "movzb  (%1),%%r10\n"
  "movq   2048(%5,%%r10,8),%%xmm0\n"
  "movzb  (%2),%%r10\n"
  "movq   4096(%5,%%r10,8),%%xmm1\n"
  "paddsw %%xmm1,%%xmm0\n"
  "movzb  (%0),%%r10\n"
  "movq   (%5,%%r10,8),%%xmm1\n"
  "paddsw %%xmm0,%%xmm1\n"
  "psraw  $0x6,%%xmm1\n"
  "packuswb %%xmm1,%%xmm1\n"
  "movd   %%xmm1,0x0(%3)\n"
"3:"
  :
  : "r"(y_buf),  // %0
    "r"(u_buf),  // %1
    "r"(v_buf),  // %2
    "r"(rgb_buf),  // %3
    "r"(width),  // %4
    "r" (kCoefficientsRgbY)  // %5
  : "memory", "r10", "r11", "xmm0", "xmm1", "xmm2", "xmm3"
);
}

void convert_to_rgba_sse2_lookuptable(
	const void *yplane,
	const void *uplane,
	const void *vplane,
	uint32_t ystride, uint32_t uvstride,
	uint32_t width, uint32_t height,
	char *outrgba)
{
	uint8_t *yrow = (uint8_t*)yplane;
	uint8_t *urow = (uint8_t*)uplane;
	uint8_t *vrow = (uint8_t*)vplane;
	for (uint32_t row = 0; row < height; row++) {
		FastConvertYUVToRGB32Row(yrow, urow, vrow, (uint8_t*)outrgba, width);
		outrgba += width * 4;
		yrow += ystride;
		if (row % 2) {
			urow += uvstride;
			vrow += uvstride;
		}
	}
}

void convert_to_rgba_reference(
	const uint8_t *yplane,
	const uint8_t *uplane,
	const uint8_t *vplane,
	uint32_t ystride, uint32_t uvstride,
	uint32_t width, uint32_t height,
	char *outrgba)
{
	uint32_t *refrgba = (uint32_t*) outrgba;
	uint32_t i, j;
	for (j = 0; j < 4096; j++) {
		for (i = 0; i < 4096; i++) {
			double ybase = yplane[(j * ystride) + i];
			ybase -= 16;
			ybase = std::max(ybase, 0.0);
			ybase *= 255;
			ybase /= 219;
			ybase = std::min(ybase, 255.0);

			double ubase = uplane[((j / 2) * uvstride) + (i / 2)];
			ubase -= 128;

			double vbase = vplane[((j / 2) * uvstride) + (i / 2)];
			vbase -= 128;

			double r = ybase + 1.5960267857142858 * vbase;
			double g = ybase - 0.3917622900949137 * ubase
					 - 0.8129676472377708 * vbase;
			double b = ybase + 2.017232142857143 * ubase;

			r = std::min(r, 255.0);
			g = std::min(g, 255.0);
			b = std::min(b, 255.0);

			r = std::max(r, 0.0);
			g = std::max(g, 0.0);
			b = std::max(b, 0.0);

			refrgba[(j * width) + i] =
				0xFF << 24 |
				lround(r) << 16 |
				lround(g) << 8  |
				lround(b);
		}
	}

}

static void write_to_png(void *buf, uint32_t stride, uint32_t width, uint32_t height, const char* name)
{
	FILE *f = fopen(name, "w");
	if (!f) {
		printf("Could not open %s for writing.\n", name);
		return;
	}

	png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING,
						      NULL, NULL, NULL);
	if (!png_ptr) {
		printf("Could not create png struct.\n");
		fclose(f);
		return;
	}

	png_infop info_ptr = png_create_info_struct(png_ptr);
	if (!info_ptr) {
		printf("Could not create info struct.\n");
		png_destroy_write_struct(&png_ptr, NULL);
		fclose(f);
		return;
	}

	if (setjmp(png_jmpbuf(png_ptr))) {
		printf("setjmp failed.\n");
		png_destroy_write_struct(&png_ptr, &info_ptr);
		fclose(f);
		return;
	}

	png_init_io(png_ptr, f);

	png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB_ALPHA, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

	png_bytepp rows = (png_bytepp) malloc(sizeof(void *) * height);
	for (uint32_t i = 0; i < height; i++) {
		rows[i] = (png_byte *)buf + (i * stride);
	}

	png_set_rows(png_ptr, info_ptr, rows);

	png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_BGR, NULL);

	free(rows);
	png_destroy_write_struct(&png_ptr, &info_ptr);
	fclose(f);
}

static void
error_check(
	uint32_t *refrgba, uint32_t *fastrgba,
	uint32_t width, uint32_t height)
{
	uint32_t bucketsR[256];
	uint32_t bucketsG[256];
	uint32_t bucketsB[256];
	memset(bucketsR, 0, sizeof(bucketsR));
	memset(bucketsG, 0, sizeof(bucketsG));
	memset(bucketsB, 0, sizeof(bucketsB));

	for (uint32_t i = 0; i < width * height; i++) {
		int32_t row = (i / width) % 16;
		if (row == 0 || row == 15)
			continue;

		int32_t refR = (refrgba[i] >> 16) & 0xFF;
		int32_t fastR = (fastrgba[i] >> 16) & 0xFF;
		int32_t refG = (refrgba[i] >> 8) & 0xFF;
		int32_t fastG = (fastrgba[i] >> 8) & 0xFF;
		int32_t refB = refrgba[i] & 0xFF;
		int32_t fastB = fastrgba[i] & 0xFF;

		bucketsR[std::min(abs(fastR - refR), 255)]++;
		bucketsG[std::min(abs(fastG - refG), 255)]++;
		bucketsB[std::min(abs(fastB - refB), 255)]++;
	}

	printf("Error distribution R:\n");
	for (uint32_t i = 0; i < 256; i++) {
		if (!bucketsR[i])
			continue;
		printf("%3d: %d\n", i, bucketsR[i]);
	}
	printf("Error distribution G:\n");
	for (uint32_t i = 0; i < 256; i++) {
		if (!bucketsG[i])
			continue;
		printf("%3d: %d\n", i, bucketsG[i]);
	}
	printf("Error distribution B:\n");
	for (uint32_t i = 0; i < 256; i++) {
		if (!bucketsB[i])
			continue;
		printf("%3d: %d\n", i, bucketsB[i]);
	}
}

int main()
{
	uint8_t *yplane = (uint8_t *)aligned_alloc(16, 4096 * 4096);
	uint8_t *uplane = (uint8_t *)aligned_alloc(16, 2048 * 2048);
	uint8_t *vplane = (uint8_t *)aligned_alloc(16, 2048 * 2048);
	uint8_t *ycur, *ucur, *vcur;

	uint32_t i, j;
	ycur = yplane;
	ucur = uplane;
	vcur = vplane;

	for (j = 0; j < 4096; j++) {
		uint8_t basey = (j % 16) * 16;
		for (i = 0; i < 4096; i++) {
			*ycur++ = basey + (i % 16);
		}
	}
	for (j = 0; j < 2048; j++) {
		uint8_t baseu = ((j / 8) % 16) * 16;
		uint8_t basev = (j / 128) * 16;
		for (i = 0; i < 2048; i++) {
			*ucur++ = baseu + ((i / 8) % 16);
			*vcur++ = basev + (i / 128);
		}
	}

	uint32_t *refrgba = (uint32_t *)aligned_alloc(16, 4096 * 4096 * 4);
	uint32_t *fastrgba = (uint32_t *)aligned_alloc(16, 4096 * 4096 * 4);

#define BENCHMARK 0

#if !BENCHMARK
	convert_to_rgba_reference(yplane, uplane, vplane,
				  4096, 2048,
				  4096, 4096,
				  (char*)refrgba);


	convert_to_rgba_sse2(yplane, uplane, vplane,
			     4096, 2048,
			     4096, 4096,
			     sBT601Consts,
			     (char*)fastrgba);
#endif

#if BENCHMARK
	for (int n = 0; n < 100; n++) {
#if 1
	convert_to_rgba_sse2(yplane, uplane, vplane,
			     4096, 2048,
			     4096, 4096,
			     sBT601Consts,
			     (char*)fastrgba);
#else
	convert_to_rgba_sse2_lookuptable(yplane, uplane, vplane,
			     4096, 2048,
			     4096, 4096,
			     (char*)fastrgba);
#endif
	}
#endif

#if !BENCHMARK
	write_to_png(refrgba, 4096 * 4, 4096, 4096, "refrgba.png");
	write_to_png(fastrgba, 4096 * 4, 4096, 4096, "fastrgba.png");
	error_check(refrgba, fastrgba, 4096, 4096);
//	write_to_png(lookuprgba, 4096 * 4, 4096, 4096, "lookuprgba.png");
#endif

	return 0;
}
