//a simple OpenCL kernel which adds two vectors A and B together into a third vector C
kernel void add(global const int* A, global const int* B, global int* C) {
	int id = get_global_id(0);
	C[id] = A[id] + B[id];
}

//a simple smoothing kernel averaging values in a local window (radius 1)
kernel void avg_filter(global const int* A, global int* B) { // This doesn't work!!
	int id = get_global_id(0);
	int se[5] = { -2, -1, 0, 1, 2 };
	int sum = 0;
	int len = 0;
	for (int i = 0; i < se; i++) { // The idea here is that if a particular ID is invalid, it simply won't be included in the calculation. If it was in image, it might be slightly less blurry? Idk.
		// go through collection structuring element
		int element = se[i];
		int new_element = id + element;
		if (new_element >= 0 || new_element < get_global_size(0)) {
			sum = sum + A[new_element];
			len++;
		}
	}
	B[id] = sum/len;
}

//a simple 2D kernel
kernel void add2D(global const int* A, global const int* B, global int* C) {
	int x = get_global_id(0);
	int y = get_global_id(1);
	int width = get_global_size(0);
	int height = get_global_size(1);
	int id = x + y*width;

	printf("id = %d x = %d y = %d w = %d h = %d\n", id, x, y, width, height);

	C[id]= A[id]+ B[id];
}
