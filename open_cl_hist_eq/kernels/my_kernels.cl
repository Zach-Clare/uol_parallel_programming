kernel void hist(global const uchar* A, global int* H, int bin_size, int bit_depth) { 
	int id = get_global_id(0);

	int pix_value = A[id]; // take pixel value

	float norm = (float)pix_value / (float)bit_depth; // normalise value to between 0 and 1
	int bin_index = round(norm * bin_size); // scale value to fit within bounds of H

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

	L[id] = norm * 255; // multiply by (desired range - 1 = 255)
}

kernel void back_proj(global const uchar* I, global uchar* O, global int* L, int bin_size, int bit_depth) {
	int id = get_global_id(0);
	int pix = I[id]; // get old pixel value

	// normalise and scale to create a lut index
	float norm = (float)pix / (float)bit_depth; // normalise value to between 0 and 1
	int lut_index = round(norm * bin_size); // spread out value to fit within bounds of H

	O[id] = L[lut_index]; // use as index for look up table
}


/////// Blelloch attempt

// Blelloch basic exclusive scan
kernel void scan_bl(global int* A) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	int t;

	// up-sweep (first stage)
	for (int stride = 1; stride < N; stride *= 2) {
		if (((id + 1) % (stride*2)) == 0)
			A[id] += A[id - stride];

		barrier(CLK_GLOBAL_MEM_FENCE); // sync the step with other instances
	}

	//down-sweep (second stage)
	if (id == 0)
		A[N-1] = 0; // exclusive scan

	barrier(CLK_GLOBAL_MEM_FENCE); //sync the step

	for (int stride = N/2; stride > 0; stride /= 2) {
		if (((id + 1) % (stride*2)) == 0) {
			t = A[id];
			A[id] += A[id - stride]; // reduce 
			A[id - stride] = t;		 // move
		}

		barrier(CLK_GLOBAL_MEM_FENCE); // sync the step with other instances
	}
}

// Calculates the block sums
kernel void block_sum(global const int* A, global int* B, int local_size) {
	int id = get_global_id(0);
	B[id] = A[(id + 1) * local_size - 1]; // Nth block * final element ID = ID of final element of current block
}

// Simple exclusive serial scan based on atomic operations - sufficient for small number of elements
kernel void scan_add_atomic(global int* A, global int* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	for (int i = id + 1; i < N && id < N; i++)
		atomic_add(&B[i], A[id]);
}

// Adjust the values stored in partial scans by adding the block sums to corresponding blocks
kernel void scan_add_adjust(global int* A, global const int* B) {
	int id = get_global_id(0);
	int gid = get_group_id(0);
	A[id] += B[gid];
}