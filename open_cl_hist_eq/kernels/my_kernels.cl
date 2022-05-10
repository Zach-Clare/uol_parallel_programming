kernel void hist(global const uchar* A, global int* H, int nr_bins, int min_value, int max_value) { 
	int id = get_global_id(0);

	//assumes that H has been initialised to 0
	int bin_index = A[id];//take value as a bin index

	//if value A is out of bounds, we just ignore it.
	//if (bin_index < min_value || bin_index > max_value) {
	//	return;
	//}

	//the bin index should start from the minimum value
	//bin_index = bin_index - min_value;

	//ensure value from vector doesn't exceed maximum bin size
	//bin_index = min(bin_index, nr_bins);

	atomic_inc(&H[bin_index]);//serial operation, not very efficient!
}

//a double-buffered version of the Hillis-Steele inclusive scan
//requires two additional input arguments which correspond to two local buffers
kernel void hist_cumulative(__global const int* A, global int* B, local int* scratch_1, local int* scratch_2) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	local int *scratch_3;//used for buffer swap

	//cache all N values from global memory to local memory
	scratch_1[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (lid >= i)
			scratch_2[lid] = scratch_1[lid] + scratch_1[lid - i];
		else
			scratch_2[lid] = scratch_1[lid];

		barrier(CLK_LOCAL_MEM_FENCE);

		//buffer swap
		scratch_3 = scratch_2;
		scratch_2 = scratch_1;
		scratch_1 = scratch_3;
	}

	//copy the cache to output array
	B[id] = scratch_1[lid];
}

kernel void divide_array(global const int* H, global float* N, float B) {
	int id = get_global_id(0);
	float hist_value = (float)H[id];

	N[id] = hist_value / B; // divide by B

	printf(id);
}

//a simple OpenCL kernel which copies all pixels from A to B
kernel void identity(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);
	B[id] = A[id];
}

kernel void filter_r(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);
	int image_size = get_global_size(0) / 3; //each image consists of 3 colour channels
	int colour_channel = id / image_size; // 0 - red, 1 - green, 2 - blue
	if (colour_channel != 0) {
		id = 0;
	}

	//this is just a copy operation, modify to filter out the individual colour channels
	B[id] = A[id];
}

kernel void invert(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);
	int pixel_colour = 255 - A[id];


	B[id] = pixel_colour;
}

kernel void rgb2grey(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);
	int image_size = get_global_size(0) / 3;
	if (id > image_size) {
		return;
	}

	// need to look ahead at the different channels
	int red = A[id];
	int green = A[id + image_size];
	int blue = A[id + (image_size * 2)];

	float avg = (red + green + blue) / 3;
	int result = (int)(avg + 0.5);
	
	B[id] = result;
	B[id + image_size] = result;
	B[id + (image_size * 2)] = result;
}

//simple ND identity kernel
kernel void identityND(global const uchar* A, global uchar* B) {
	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels
	int image_size = width*height; //image size in pixels
	int channels = get_global_size(2); //number of colour channels: 3 for RGB

	int x = get_global_id(0); //current x coord.
	int y = get_global_id(1); //current y coord.
	int c = get_global_id(2); //current colour channel

	int id = x + y*width + c*image_size; //global id in 1D space

	B[id] = A[id];
}

//2D averaging filter
kernel void avg_filterND(global const uchar* A, global uchar* B) {
	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels
	int image_size = width*height; //image size in pixels
	int channels = get_global_size(2); //number of colour channels: 3 for RGB

	int x = get_global_id(0); //current x coord.
	int y = get_global_id(1); //current y coord.
	int c = get_global_id(2); //current colour channel

	int id = x + y*width + c*image_size; //global id in 1D space

	uint result = 0;

	//simple boundary handling - just copy the original pixel
	if ((x == 0) || (x == width-1) || (y == 0) || (y == height-1)) {
		result = A[id];	
	} else {
		for (int i = (x-1); i <= (x+1); i++)
		for (int j = (y-1); j <= (y+1); j++) 
			result += A[i + j*width + c*image_size];

		result /= 9;
	}

	B[id] = (uchar)result;
}

//2D 3x3 convolution kernel
kernel void convolutionND(global const uchar* A, global uchar* B, constant float* mask) {
	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels
	int image_size = width*height; //image size in pixels
	int channels = get_global_size(2); //number of colour channels: 3 for RGB

	int x = get_global_id(0); //current x coord.
	int y = get_global_id(1); //current y coord.
	int c = get_global_id(2); //current colour channel

	int id = x + y*width + c*image_size; //global id in 1D space

	float result = 0;

	//simple boundary handling - just copy the original pixel
	if ((x == 0) || (x == width-1) || (y == 0) || (y == height-1)) {
		result = A[id];	
	} else {
		for (int i = (x-1); i <= (x+1); i++)
		for (int j = (y-1); j <= (y+1); j++) 
			result += A[i + j*width + c*image_size]*mask[i-(x-1) + j-(y-1)];
	}

	B[id] = (uchar)result;
}