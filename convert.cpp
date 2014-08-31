#include <algorithm>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include <emmintrin.h>

#include <png.h>

static const uint8_t SSE2masks[16] =
	{ 36, 36, 36, 36,
	  20, 20, 20, 20,
	  0, 0, 0, 0,
	  0, 0xFF, 0, 0xFF };

static const int16_t SSE2colorcoef[8] =
	{  204,  204,
	  -50,  -50,
	  -104, -104,
	   258,  258 };

static const uint8_t SSE2const[16] =
	{ 149, 0, 149, 0,
	  128, 0, 128, 0,
	  0, 0, 0, 0,
	  0, 0, 0, 0 };

static inline void
convert_yuv_word_lines(__m128i yline,
		       __m128i uline,
		       __m128i vline,
		       char **outrgba)
{
	__m128i zero = _mm_setzero_si128();

	// Convert red
	__m128i rvcoef = _mm_shuffle_epi32(*(__m128i *)&SSE2colorcoef, 0x00);
	__m128i rvline = _mm_mullo_epi16(vline, rvcoef);

	__m128i redline = _mm_adds_epi16(rvline, yline);
	redline = _mm_max_epi16(redline, zero);
	redline = _mm_slli_epi16(redline, 1);
	redline = _mm_srli_epi16(redline, 8);

	// Convert green
	__m128i gucoef = _mm_shuffle_epi32(*(__m128i *)&SSE2colorcoef, 0x55);
	__m128i gvcoef = _mm_shuffle_epi32(*(__m128i *)&SSE2colorcoef, 0xAA);
	__m128i guline = _mm_mullo_epi16(uline, gucoef);
	__m128i gvline = _mm_mullo_epi16(vline, gvcoef);
	__m128i guvline = _mm_adds_epi16(guline, gvline);

	__m128i greenline = _mm_adds_epi16(guvline, yline);
	greenline = _mm_max_epi16(greenline, zero);
	greenline = _mm_slli_epi16(greenline, 1);
	greenline = _mm_srli_epi16(greenline, 8);

	__m128i shiftedgreenline = _mm_slli_epi16(greenline, 8);

	__m128i redgreenline = _mm_or_si128(redline, shiftedgreenline);

	// Convert blue
	__m128i bucoef = _mm_shuffle_epi32(*(__m128i *)&SSE2colorcoef, 0xFF);
	__m128i buline = _mm_mullo_epi16(uline, bucoef);
	__m128i blueline = _mm_adds_epi16(buline, yline);
	blueline = _mm_max_epi16(blueline, zero);
	blueline = _mm_srli_epi16(blueline, 7);

	__m128i alphaline = _mm_shuffle_epi32(*(__m128i *)&SSE2masks, 0xFF);
	__m128i bluealphaline = _mm_or_si128(blueline, alphaline);
	__m128i rgbalinelow = _mm_unpacklo_epi16(redgreenline, bluealphaline);
	__m128i rgbalinehigh = _mm_unpackhi_epi16(redgreenline, bluealphaline);

	_mm_stream_si128(*(__m128i**)outrgba, rgbalinelow);
	_mm_stream_si128(*(__m128i**)outrgba + 1, rgbalinehigh);
	*outrgba += sizeof(__m128i) * 2;
}

static inline void
convert_yuv_byte_lines(__m128i full_yline,
		       __m128i full_uline,
		       __m128i full_vline,
		       char **outrgba)
{
	__m128i zero = _mm_setzero_si128();
	__m128i yunscaledline = _mm_unpacklo_epi8(full_yline, zero);
	__m128i uunscaledline = _mm_unpacklo_epi8(full_uline, zero);
	__m128i vunscaledline = _mm_unpacklo_epi8(full_vline, zero);

	// Scale to 255 and convert to 9.7 fixed point
	__m128i yscaler = _mm_shuffle_epi32(*(__m128i *)&SSE2const, 0x00);
	__m128i uvnormalizer = _mm_shuffle_epi32(*(__m128i *)&SSE2const, 0x55);
	__m128i yline = _mm_mullo_epi16(yunscaledline, yscaler);
	// Scale to -128 to 127
	__m128i uline = _mm_sub_epi16(uunscaledline, uvnormalizer);
	__m128i vline = _mm_sub_epi16(vunscaledline, uvnormalizer);

	convert_yuv_word_lines(yline, uline, vline, outrgba);

	yunscaledline = _mm_unpackhi_epi8(full_yline, zero);
	uunscaledline = _mm_unpackhi_epi8(full_uline, zero);
	vunscaledline = _mm_unpackhi_epi8(full_vline, zero);

	// Scale to 255 and convert to 9.7 fixed point
	yline = _mm_mullo_epi16(yunscaledline, yscaler);
	// Scale to -128 to 127
	uline = _mm_sub_epi16(uunscaledline, uvnormalizer);
	vline = _mm_sub_epi16(vunscaledline, uvnormalizer);

	convert_yuv_word_lines(yline, uline, vline, outrgba);
}

void
convert_to_rgba_sse2(
	const void *yplane,
	const void *uplane,
	const void *vplane,
	uint32_t ystride, uint32_t uvstride,
	uint32_t width, uint32_t height,
	char *outrgba)
{
	__m128i whiteclip = _mm_shuffle_epi32(*(__m128i *)&SSE2masks, 0x55);
	__m128i blackclip = _mm_shuffle_epi32(*(__m128i *)&SSE2masks, 0x00);

	__m128i *yrow = (__m128i *)yplane;
	__m128i *urow = (__m128i *)uplane;
	__m128i *vrow = (__m128i *)vplane;
	char *outrow = outrgba;

	for (uint32_t j = 0; j < height; j++) {
		uint32_t i = 0;
		__m128i *y = yrow;
		__m128i *u = urow;
		__m128i *v = vrow;
		char *out = outrow;
		while (i < width) {
			/* Use saturation arithmetic to clamp y values */
			__m128i full_yline = _mm_adds_epu8(*y++, whiteclip);
			full_yline = _mm_subs_epu8(full_yline, blackclip);

			/* Convert high half */
			__m128i full_uline = _mm_unpackhi_epi8(*u, *u);
			__m128i full_vline = _mm_unpackhi_epi8(*v, *v);

			convert_yuv_byte_lines(full_yline,
					       full_uline,
					       full_vline,
					       &out);

			i += sizeof(__m128i);
			if (i >= width)
				break;

			/* Convert lower half */
			full_yline = _mm_adds_epu8(*y++, whiteclip);
			full_yline = _mm_subs_epu8(full_yline, blackclip);
			full_uline = _mm_unpacklo_epi8(*u, *u);
			full_vline = _mm_unpacklo_epi8(*v, *v);

			convert_yuv_byte_lines(full_yline,
					       full_uline,
					       full_vline,
					       &out);

			i += sizeof(__m128i);
			u += 1;
			v += 1;
		}
		yrow = (__m128i *)(((char *)yrow) + ystride);
		urow = (__m128i *)((char *)uplane + uvstride * (j / 2));
		vrow = (__m128i *)((char *)vplane + uvstride * (j / 2));
		outrow += width * 4;
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

	png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);

	free(rows);
	png_destroy_write_struct(&png_ptr, &info_ptr);
	fclose(f);
}

int main()
{
	uint8_t *yplane = (uint8_t *)malloc(4096 * 4096);
	uint8_t *uplane = (uint8_t *)malloc(2048 * 2048);
	uint8_t *vplane = (uint8_t *)malloc(2048 * 2048);
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

	uint32_t *refrgba = (uint32_t *)malloc(4096 * 4096 * 4);
	for (j = 0; j < 4096; j++) {
		for (i = 0; i < 4096; i++) {
			double ybase = yplane[(j * 4096) + i];
			ybase -= 16;
			ybase = std::max(ybase, 0.0);
			ybase *= 255;
			ybase /= 219;
			ybase = std::min(ybase, 255.0);

			double ubase = uplane[((j / 2) * 2048) + (i / 2)];
			ubase -= 128;

			double vbase = vplane[((j / 2) * 2048) + (i / 2)];
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

			refrgba[(j * 4096) + i] =
				0xFF << 24 |
				lround(b) << 16 |
				lround(g) << 8  |
				lround(r);
		}
	}

	uint32_t *fastrgba = (uint32_t *)malloc(4096 * 4096 * 4);
	convert_to_rgba_sse2(yplane, uplane, vplane,
			     4096, 2048,
			     4096, 4096,
			     (char*)fastrgba);

	write_to_png(refrgba, 4096 * 4, 4096, 4096, "refrgba.png");
	write_to_png(fastrgba, 4096 * 4, 4096, 4096, "fastrgba.png");

	return 0;
}
