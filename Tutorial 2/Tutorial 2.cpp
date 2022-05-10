#include <iostream>
#include <vector>

#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -f : input image file (default: test.ppm)" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {

	//std::cout << "Hello World!\n";

	int platform_id = 0; // specify OpenCL platform ID
	int device_id = 0; // specify OpenCL device ID
	std::string filename = "test.pgm"; // name of test image

	CImg<unsigned char> image_input(filename.c_str()); // initialise image
	CImgDisplay disp_input = CImgDisplay(image_input, "input"); // display image with title "input"

	cl::Context context = GetContext(platform_id, device_id);

	// && !disp_output.is_closed()
	// && !disp_output.is_keyESC()
	// disp_output.wait(1);

	while (!disp_input.is_closed() && !disp_input.is_keyESC()) { // if user has not closed image
		disp_input.wait(1); // keep the image displayed
	}


	


	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	//int platform_id = 0;
	//int device_id = 0;
	string image_filename = "test.pgm";

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	cimg::exception_mode(0);

	//detect any potential exceptions
	try {
		CImg<unsigned char> image_input(image_filename.c_str());
		CImgDisplay disp_input(image_input,"input");

		//a 3x3 convolution mask implementing an averaging filter
		/*std::vector<float> convolution_mask = { 1.f / 9, 1.f / 9, 1.f / 9,
												1.f / 9, 1.f / 9, 1.f / 9,
												1.f / 9, 1.f / 9, 1.f / 9 };*/

		//Part 3 - host operations
		//3.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context);

		//3.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels/my_kernels.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try { 
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		//Part 4 - device operations
		/////// get histogram

		size_t local_size = 8; // no padding needed, as input vector is multiple of local_size

		std::vector<int> histogram(256); // bin size
		size_t histogram_size = histogram.size() * sizeof(int);

		//device - buffers
		cl::Buffer buffer_dev_image_input(context, CL_MEM_READ_ONLY, image_input.size());
		cl::Buffer buffer_histogram(context, CL_MEM_READ_WRITE, histogram_size);
//		cl::Buffer dev_convolution_mask(context, CL_MEM_READ_ONLY, convolution_mask.size()*sizeof(float));

		//4.1 Copy images to device memory
		queue.enqueueWriteBuffer(buffer_dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);
		queue.enqueueFillBuffer(buffer_histogram, 0, 0, histogram_size);

		//4.2 Setup and execute the kernel (i.e. device code)
		cl::Kernel kernel = cl::Kernel(program, "hist");
		kernel.setArg(0, buffer_dev_image_input);
		kernel.setArg(1, buffer_histogram);
		kernel.setArg(2, 256); // num bins
		kernel.setArg(3, 0); // min_value
		kernel.setArg(4, 256);// max_value

		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange);
		//
		//vector<unsigned char> output_buffer(histogram.size());
		//4.3 Copy the result from device to host
		queue.enqueueReadBuffer(buffer_histogram, CL_TRUE, 0, histogram_size, &histogram[0]);

		//CImg<unsigned char> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		//CImgDisplay disp_output(output_image,"output");

 		/*while (!disp_input.is_closed() && !disp_output.is_closed()
			&& !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
		    disp_input.wait(1);
		    disp_output.wait(1);
	    }*/		

		//cout << histogram;
		/*for (int i = 0; i != histogram_size; i++) {
			cout << dev_image_output[i];
		}*/

		std::cout << "B = " << histogram << std::endl;

		/////// Cumulative histogram

		for (int i = 0; i < histogram.size(); i++) {
			if (i != 0) {
				histogram[i] += histogram[i - 1];
			}
		}

		std::cout << "B = " << histogram << std::endl;

	}
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}

	return 0;
}
