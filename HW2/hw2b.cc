//v7: omp schedule => 73.57
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include <atomic>
#include <immintrin.h>

#define CACHE_LINE 64

int iters;
double left, right, lower, upper;
int width, height;
int* image;
double x_interv, y_interv;
int row_seg;
int rank, size;
double* y0_values;

void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int tmp = height - 1 - y;
            int p = buffer[((tmp%size)*row_seg+tmp/size)*width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

void process_row(int row, int* image) {
    double y0 = y0_values[row];
    const __m512d v_four = _mm512_set1_pd(4.0);
    const __m512d v_two = _mm512_set1_pd(2.0);
    const __m512d v_y0 = _mm512_set1_pd(y0);
    const __m512d v_x_interval = _mm512_set1_pd(x_interv);
    const __m512d v_left = _mm512_set1_pd(left);
    const __m512i v_one = _mm512_set1_epi64(1);
    const __m512i base_indices = _mm512_set_epi64(7,6,5,4,3,2,1,0);
    
    for(int col = 0; col < width-31; col += 32) {
        // Calculate x0 values for all four sets
        __m512i indices1 = _mm512_add_epi64(base_indices, _mm512_set1_epi64(col));
        __m512i indices2 = _mm512_add_epi64(base_indices, _mm512_set1_epi64(col + 8));
        __m512i indices3 = _mm512_add_epi64(base_indices, _mm512_set1_epi64(col + 16));
        __m512i indices4 = _mm512_add_epi64(base_indices, _mm512_set1_epi64(col + 24));

        __m512d v_x0_1 = _mm512_fmadd_pd(_mm512_cvtepi64_pd(indices1), v_x_interval, v_left);
        __m512d v_x0_2 = _mm512_fmadd_pd(_mm512_cvtepi64_pd(indices2), v_x_interval, v_left);
        __m512d v_x0_3 = _mm512_fmadd_pd(_mm512_cvtepi64_pd(indices3), v_x_interval, v_left);
        __m512d v_x0_4 = _mm512_fmadd_pd(_mm512_cvtepi64_pd(indices4), v_x_interval, v_left);
        
        // Initialize coordinates
        __m512d v_x1 = _mm512_setzero_pd();
        __m512d v_y1 = _mm512_setzero_pd();
        __m512d v_x2 = _mm512_setzero_pd();
        __m512d v_y2 = _mm512_setzero_pd();
        __m512d v_x3 = _mm512_setzero_pd();
        __m512d v_y3 = _mm512_setzero_pd();
        __m512d v_x4 = _mm512_setzero_pd();
        __m512d v_y4 = _mm512_setzero_pd();
        
        // Initialize counters and masks
        __m512i v_repeats1 = _mm512_setzero_si512();
        __m512i v_repeats2 = _mm512_setzero_si512();
        __m512i v_repeats3 = _mm512_setzero_si512();
        __m512i v_repeats4 = _mm512_setzero_si512();
        __mmask8 mask1 = 0xFF, mask2 = 0xFF, mask3 = 0xFF, mask4 = 0xFF;
        
        for(int iter = 0; iter < iters && (mask1 || mask2 || mask3 || mask4); iter++) {
            // Process first set
            if(mask1) {
                const __m512d v_x2_1 = _mm512_mul_pd(v_x1, v_x1);
                const __m512d v_y2_1 = _mm512_mul_pd(v_y1, v_y1);
                const __m512d v_length_squared1 = _mm512_add_pd(v_x2_1, v_y2_1);
                const __mmask8 new_mask1 = _mm512_cmp_pd_mask(v_length_squared1, v_four, _CMP_LT_OS);
                const __mmask8 active_mask1 = mask1 & new_mask1;
                
                if(active_mask1) {
                    v_repeats1 = _mm512_mask_add_epi64(v_repeats1, active_mask1, v_repeats1, v_one);
                    const __m512d v_xy1 = _mm512_mul_pd(v_x1, v_y1);
                    const __m512d v_new_y1 = _mm512_fmadd_pd(v_two, v_xy1, v_y0);
                    const __m512d v_new_x1 = _mm512_add_pd(
                        _mm512_sub_pd(v_x2_1, v_y2_1),
                        v_x0_1
                    );
                    v_x1 = _mm512_mask_mov_pd(v_x1, active_mask1, v_new_x1);
                    v_y1 = _mm512_mask_mov_pd(v_y1, active_mask1, v_new_y1);
                }
                mask1 = new_mask1;
            }
            
            // Process second set
            if(mask2) {
                const __m512d v_x2_2 = _mm512_mul_pd(v_x2, v_x2);
                const __m512d v_y2_2 = _mm512_mul_pd(v_y2, v_y2);
                const __m512d v_length_squared2 = _mm512_add_pd(v_x2_2, v_y2_2);
                const __mmask8 new_mask2 = _mm512_cmp_pd_mask(v_length_squared2, v_four, _CMP_LT_OS);
                const __mmask8 active_mask2 = mask2 & new_mask2;
                
                if(active_mask2) {
                    v_repeats2 = _mm512_mask_add_epi64(v_repeats2, active_mask2, v_repeats2, v_one);
                    const __m512d v_xy2 = _mm512_mul_pd(v_x2, v_y2);
                    const __m512d v_new_y2 = _mm512_fmadd_pd(v_two, v_xy2, v_y0);
                    const __m512d v_new_x2 = _mm512_add_pd(
                        _mm512_sub_pd(v_x2_2, v_y2_2),
                        v_x0_2
                    );
                    v_x2 = _mm512_mask_mov_pd(v_x2, active_mask2, v_new_x2);
                    v_y2 = _mm512_mask_mov_pd(v_y2, active_mask2, v_new_y2);
                }
                mask2 = new_mask2;
            }
            
            // Process third set
            if(mask3) {
                const __m512d v_x2_3 = _mm512_mul_pd(v_x3, v_x3);
                const __m512d v_y2_3 = _mm512_mul_pd(v_y3, v_y3);
                const __m512d v_length_squared3 = _mm512_add_pd(v_x2_3, v_y2_3);
                const __mmask8 new_mask3 = _mm512_cmp_pd_mask(v_length_squared3, v_four, _CMP_LT_OS);
                const __mmask8 active_mask3 = mask3 & new_mask3;
                
                if(active_mask3) {
                    v_repeats3 = _mm512_mask_add_epi64(v_repeats3, active_mask3, v_repeats3, v_one);
                    const __m512d v_xy3 = _mm512_mul_pd(v_x3, v_y3);
                    const __m512d v_new_y3 = _mm512_fmadd_pd(v_two, v_xy3, v_y0);
                    const __m512d v_new_x3 = _mm512_add_pd(
                        _mm512_sub_pd(v_x2_3, v_y2_3),
                        v_x0_3
                    );
                    v_x3 = _mm512_mask_mov_pd(v_x3, active_mask3, v_new_x3);
                    v_y3 = _mm512_mask_mov_pd(v_y3, active_mask3, v_new_y3);
                }
                mask3 = new_mask3;
            }
            
            // Process fourth set
            if(mask4) {
                const __m512d v_x2_4 = _mm512_mul_pd(v_x4, v_x4);
                const __m512d v_y2_4 = _mm512_mul_pd(v_y4, v_y4);
                const __m512d v_length_squared4 = _mm512_add_pd(v_x2_4, v_y2_4);
                const __mmask8 new_mask4 = _mm512_cmp_pd_mask(v_length_squared4, v_four, _CMP_LT_OS);
                const __mmask8 active_mask4 = mask4 & new_mask4;
                
                if(active_mask4) {
                    v_repeats4 = _mm512_mask_add_epi64(v_repeats4, active_mask4, v_repeats4, v_one);
                    const __m512d v_xy4 = _mm512_mul_pd(v_x4, v_y4);
                    const __m512d v_new_y4 = _mm512_fmadd_pd(v_two, v_xy4, v_y0);
                    const __m512d v_new_x4 = _mm512_add_pd(
                        _mm512_sub_pd(v_x2_4, v_y2_4),
                        v_x0_4
                    );
                    v_x4 = _mm512_mask_mov_pd(v_x4, active_mask4, v_new_x4);
                    v_y4 = _mm512_mask_mov_pd(v_y4, active_mask4, v_new_y4);
                }
                mask4 = new_mask4;
            }
        }
        
        // Store results
        const __m256i v_repeats1_32 = _mm512_cvtepi64_epi32(v_repeats1);
        const __m256i v_repeats2_32 = _mm512_cvtepi64_epi32(v_repeats2);
        const __m256i v_repeats3_32 = _mm512_cvtepi64_epi32(v_repeats3);
        const __m256i v_repeats4_32 = _mm512_cvtepi64_epi32(v_repeats4);
        
        _mm256_storeu_si256((__m256i*)&image[(row/size) * width + col], v_repeats1_32);
        _mm256_storeu_si256((__m256i*)&image[(row/size) * width + col + 8], v_repeats2_32);
        _mm256_storeu_si256((__m256i*)&image[(row/size) * width + col + 16], v_repeats3_32);
        _mm256_storeu_si256((__m256i*)&image[(row/size) * width + col + 24], v_repeats4_32);
    }

    // Handle remaining pixels
    for(int i = width - (width % 32); i < width; i++) {
        double x0 = i * x_interv + left;
        double x = 0;
        double y = 0;
        int repeats = 0;
        
        while(repeats < iters && (x*x + y*y) < 4.0) {
            double temp = x*x - y*y + x0;
            y = 2*x*y + y0;
            x = temp;
            ++repeats;
        }
        
        image[(row/size) * width + i] = repeats;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv); 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //get the rank (id) of the current process
    MPI_Comm_size(MPI_COMM_WORLD, &size); //get the number of the process

    /* detect how many CPUs are available */
    cpu_set_t cpu_set; //an set of available cpu, each bit represent the status: available or not
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set); //which cpu cores the process is allowed to execute on 
    int ncpus = CPU_COUNT(&cpu_set);
    if(rank == 0) printf("%d cpus available\n", ncpus);

    assert(argc == 9);
    const char* filename = argv[1]; //the path to the output file
    iters = strtol(argv[2], 0, 10); //number of iterations
    left = strtod(argv[3], 0); //inclusive bound of the real axis
    right = strtod(argv[4], 0); //non-inclusive bound of the real axis
    lower = strtod(argv[5], 0); //inclusive bound of the imaginary axis
    upper = strtod(argv[6], 0); //non-inclusive bound of the imaginary axis
    width = strtol(argv[7], 0, 10); //number of points in the x-axis for output
    height = strtol(argv[8], 0, 10); //number of points in the y-axis for output
    
    y_interv = (upper - lower) / height; //pt interval on y-axis
    x_interv = (right - left) / width; //pt interval on x-axis
    row_seg = (height + size - 1) / size; //dirstribute rows to each MPI process

    //aligned allocation
    image = (int*)aligned_alloc(CACHE_LINE, 
        ((width * row_seg * sizeof(int) + CACHE_LINE - 1) / CACHE_LINE) * CACHE_LINE);
    y0_values = (double*)aligned_alloc(CACHE_LINE, 
        ((height * sizeof(double) + CACHE_LINE - 1) / CACHE_LINE) * CACHE_LINE);
    assert(image && y0_values);

    for(int row = 0; row < height; row++) {
        y0_values[row] = row * ((upper - lower) / height) + lower;
    }

    int* all_image = NULL;
    if(rank == 0) {
        all_image = (int*)aligned_alloc(CACHE_LINE, 
            ((size * width * row_seg * sizeof(int) + CACHE_LINE - 1) / CACHE_LINE) * CACHE_LINE);
        assert(all_image);
    }
    
    #pragma omp parallel num_threads(ncpus)
    {
        #pragma omp for schedule(dynamic)
        for(int row = rank; row < height; row += size) {
            process_row(row, image);
        }
    }

    MPI_Gather(image, width * row_seg, MPI_INT, 
               all_image, width * row_seg, MPI_INT, 
               0, MPI_COMM_WORLD);
    
    if(rank == 0) {
        write_png(filename, iters, width, height, all_image);
        free(all_image);
    }

    MPI_Finalize();
    free(image);
    free(y0_values);
    
    return 0;
}