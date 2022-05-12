kernel void hist(global const uchar* A, global int* H) { 
	int id = get_global_id(0);

	int bin_index = A[id]; // take value and use as a bin index
	atomic_inc(&H[bin_index]); //serial operation, not very efficient!
}

// Scan Add algorithm - a double-buffered version of the Hillis-Steele inclusive scan
kernel void hist_cumulative(__global const int* A, global int* B, local int* scratch_1, local int* scratch_2) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	local int *scratch_3; // used for buffer swap

	// cache all N values from global memory to local memory
	scratch_1[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE); // wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (lid >= i)
			scratch_2[lid] = scratch_1[lid] + scratch_1[lid - i];
		else
			scratch_2[lid] = scratch_1[lid];

		barrier(CLK_LOCAL_MEM_FENCE);

		// buffer swap
		scratch_3 = scratch_2;
		scratch_2 = scratch_1;
		scratch_1 = scratch_3;
	}

	// copy the cache to output array
	B[id] = scratch_1[lid];
}

kernel void normalise_array(global const int* H, global float* N, float B) {
	int id = get_global_id(0);
	float hist_value = (float)H[id];

	N[id] = hist_value / B; // divide by B
}

kernel void lut(global const float* N, global int* L) {
	int id = get_global_id(0);
	float norm = N[id]; // get normalised value

	L[id] = round(norm * 255); // multiply by (desired range - 1 = 255)
}

kernel void back_prop(global const uchar* I, global uchar* O, global int* L) {
	int id = get_global_id(0);
	int pix = I[id]; // get old pixel value

	O[id] = L[pix];
}