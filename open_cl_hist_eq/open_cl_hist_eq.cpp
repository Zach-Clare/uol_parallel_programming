/*

Thomas Clare - CLA19704945

Functionality:
	- Image enhancement via a variety of kernels.
	- atomic_int() is used to prevent race conditions.
	- Each memory and kernel operation is timed and displayed.
	- Timings are also collated into overall memory transfer time, overall kernel operation time, and total program execution time.
	- The bin size is variable (see bin_size variable).
	- Colour images are supported (tested with test_colour.ppm).

Original developments:
	- normalise_array() kernel
	- lut() kernel
	- back_proj() kernel

External sources:
	- None

Optimisation strategies:
	- The scan add algorithm is used in the hist_cumulative() kernel.
	- The hist_cumulative() kernel uses local memory.
	- Buffers are re-used where possible to reduce memory transfer times.
	- Blelloch steps are attempted but not implemented as part of main program.

	(word count: 112)
*/

#include <iostream>
#include <vector>
#include <numeric>

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

	int platform_id = 0; // specify default OpenCL platform ID
	int device_id = 0; // specify default OpenCL device ID

	// Handle command line options such as device selection, verbosity, etc.
	string image_filename = "test.pgm";

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); } // custom platform id
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); } // custom device id
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; } // list platforms and devices
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; } // custom image
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; } // display help page
	}

	cimg::exception_mode(0); // quiet mode

	// detect any potential exceptions
	try {
		CImg<unsigned char> image_input(image_filename.c_str()); // init image
		CImgDisplay disp_input(image_input,"Raw image"); // display raw image with title
		int bit_depth = 256;

		// Host operations
		cl::Context context = GetContext(platform_id, device_id); // select computing devices to be used with kernels
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl; // display the selected hardware	
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE); // create a queue to which we will push commands for the device

		cl::Program::Sources sources; // load and build device code from file
		AddSources(sources, "kernels/my_kernels.cl"); // file
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

		/////////// Calculate histogram ////////////////////////////////////////////////////////////////////////////////////////////

		int bin_size = 256; // Variable bin size
		std::vector<int> histogram(bin_size); // make bin size of histogram equal to amount of intensities in input image
		size_t histogram_size = histogram.size() * sizeof(int); // get byte length of histogram space

		cl::Buffer buffer_image_input(context, CL_MEM_READ_ONLY, image_input.size()); // prepare input buffer for the kernel
		cl::Buffer buffer_histogram(context, CL_MEM_READ_WRITE, histogram_size); // prepare output buffer for the kernel

		cl::Event event_hist_write; // create event to measure performance
		queue.enqueueWriteBuffer(buffer_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0], NULL, &event_hist_write); // write input image to input buffer

		cl::Kernel kernel = cl::Kernel(program, "hist"); // create hist kernel 
		kernel.setArg(0, buffer_image_input); // set appropriate arguements (arrays start at 0)
		kernel.setArg(1, buffer_histogram);
		kernel.setArg(2, bin_size);
		kernel.setArg(3, bit_depth);

		cl::Event event_hist_kernel; // new event for histogram calculation kernel
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &event_hist_kernel); // being task (with event)
		
		cl::Event event_hist_read; // timing event for data retrieval
		queue.enqueueReadBuffer(buffer_histogram, CL_TRUE, 0, histogram_size, &histogram[0], NULL, &event_hist_read); // copy results from device to host (with timing event)

		std::cout << "Raw histogram = " << histogram << std::endl << std::endl; // display calculated histogram for debug purposes


		/////////// Create cumulative histogram  ///////////////////////////////////////////////////////////////////////////////////

		std::vector<int> cumulative_histogram(bin_size); // needs identical bin size
		size_t cumulative_histogram_size = cumulative_histogram.size() * sizeof(int); // calc total size in bytes

		cl::Buffer buffer_cumulative_histogram(context, CL_MEM_READ_WRITE, cumulative_histogram_size); // create output buffer for cumulative histogram

		kernel = cl::Kernel(program, "hist_cumulative"); // create handle for hist_cumulative kernel
		kernel.setArg(0, buffer_histogram); // re-use previously filled buffer
		kernel.setArg(1, buffer_cumulative_histogram); // set output buffer
		kernel.setArg(2, cl::Local(histogram_size)); // size for scratch 1
		kernel.setArg(3, cl::Local(histogram_size)); // size for scratch 2

		cl::Event event_cumulative_kernel; // event for cumulative kernel
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(cumulative_histogram.size()), cl::NullRange, NULL, &event_cumulative_kernel); // begin cumulative histogram task

		cl::Event event_cumulative_read; // event for reading cumulative histogram
		queue.enqueueReadBuffer(buffer_cumulative_histogram, CL_TRUE, 0, cumulative_histogram_size, &cumulative_histogram[0], NULL, &event_cumulative_read); // read cumulative histogram

		std::cout << "Cumulative histogram = " << cumulative_histogram << std::endl << std::endl; // display for debug purposes


		/////////// Normalise histogram  ///////////////////////////////////////////////////////////////////////////////////////////

		float max = cumulative_histogram.back(); // Get the max value, which will be the final value in the array.

		std::vector<float> norm_histogram(bin_size); // bin size
		size_t norm_histogram_size = norm_histogram.size() * sizeof(float); // size of normalised histogram vector

		cl::Buffer buffer_norm_histogram(context, CL_MEM_READ_WRITE, norm_histogram_size); // new buffer for normalised histogram result

		kernel = cl::Kernel(program, "normalise_array"); // set kernel target to normalise_array
		kernel.setArg(0, buffer_cumulative_histogram); // set args
		kernel.setArg(1, buffer_norm_histogram);
		kernel.setArg(2, max);

		cl::Event event_norm_kernel; // new event for normalisation kernel
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(norm_histogram.size()), cl::NullRange, NULL, &event_norm_kernel); // begin normalisation task
		
		cl::Event event_norm_read; // event for reading normalisation buffer
		queue.enqueueReadBuffer(buffer_norm_histogram, CL_TRUE, 0, norm_histogram_size, &norm_histogram[0], NULL, &event_norm_read); // read normalisation buffer

		std::cout << "Normalised histogram = " << norm_histogram << std::endl << std::endl; // display for debug purposes


		/////////// Create look up table /////////////////////////////////////////////////////////////////////////////////////////////////

		std::vector<int> lut(bin_size); // look up table
		size_t lut_size = lut.size() * sizeof(int); // get lut size in bytes

		cl::Buffer buffer_lut(context, CL_MEM_READ_WRITE, lut_size); // create buffer for lut output

		kernel = cl::Kernel(program, "lut"); // set target to lut kernel
		kernel.setArg(0, buffer_norm_histogram); // set args
		kernel.setArg(1, buffer_lut);

		cl::Event event_lut_kernel; // event for lut kernel
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(lut_size), cl::NullRange, NULL, &event_lut_kernel); // begin lut task

		cl::Event event_lut_read; // event for lut buffer read
		queue.enqueueReadBuffer(buffer_lut, CL_TRUE, 0, lut_size, &lut.data()[0], NULL, &event_lut_read); // read the lut buffer

		std::cout << "Look up table = " << lut << std::endl << std::endl; // display for debug purposes


		/////////// Create enhanced image from LUT ///////////////////////////////////////////////////////////////////////////////////////
		
		std::vector<unsigned char> image_output(image_input.size()); // create space for image output
		size_t image_output_size = image_output.size(); // calc size

		cl::Buffer buffer_output(context, CL_MEM_READ_WRITE, image_output_size); // new buffer for enhanced image output

		kernel = cl::Kernel(program, "back_proj"); // target back_proj kernel
		kernel.setArg(0, buffer_image_input); // set args
		kernel.setArg(1, buffer_output);
		kernel.setArg(2, buffer_lut);
		kernel.setArg(3, bin_size);
		kernel.setArg(4, bit_depth);

		cl::Event event_enhance_kernel; // event for enhancement kernel
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &event_enhance_kernel); // begin enhancement kernel

		cl::Event event_enhance_read; // event for reading results
		queue.enqueueReadBuffer(buffer_output, CL_TRUE, 0, image_output_size, &image_output.data()[0], NULL, &event_enhance_read); // read from buffer


		CImg<unsigned char> output_image(image_output.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum()); // new image from enhanced data
		CImgDisplay disp_output(output_image,"Enahnced image"); // display enhanced image

		while (!disp_input.is_closed() && !disp_output.is_closed()
			&& !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
			disp_input.wait(1);
			disp_output.wait(1);
		}

		/////////// Performance monitoring ///////////////////////////////////////////////////////////////////////////////////////////////

		std::vector<unsigned long long> performance = {
			event_hist_write.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event_hist_write.getProfilingInfo<CL_PROFILING_COMMAND_START>(),
			event_hist_kernel.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event_hist_kernel.getProfilingInfo<CL_PROFILING_COMMAND_START>(),
			event_hist_read.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event_hist_read.getProfilingInfo<CL_PROFILING_COMMAND_START>(),
			event_cumulative_kernel.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event_cumulative_kernel.getProfilingInfo<CL_PROFILING_COMMAND_START>(),
			event_cumulative_read.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event_cumulative_read.getProfilingInfo<CL_PROFILING_COMMAND_START>(),
			event_norm_kernel.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event_norm_kernel.getProfilingInfo<CL_PROFILING_COMMAND_START>(),
			event_norm_read.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event_norm_read.getProfilingInfo<CL_PROFILING_COMMAND_START>(),
			event_lut_kernel.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event_lut_kernel.getProfilingInfo<CL_PROFILING_COMMAND_START>(),
			event_lut_read.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event_lut_read.getProfilingInfo<CL_PROFILING_COMMAND_START>(),
			event_enhance_kernel.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event_enhance_kernel.getProfilingInfo<CL_PROFILING_COMMAND_START>(),
			event_enhance_read.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event_enhance_read.getProfilingInfo<CL_PROFILING_COMMAND_START>()
		};

		std::cout << "Histogram calculation:" << std::endl; // Performance monitoring for creating the histogram
		std::cout << "- buffer write time (ns): " << performance[0] << std::endl;
		std::cout << "- \"hist\" kernel execution time (ns): " << performance[1] << std::endl;
		std::cout << "- buffer read time (ns): " << performance[2] << std::endl;
		std::cout << std::endl; // line break
		std::cout << "Cumulative histogram calculation:" << std::endl; // Performance monitoring for creating the cumulative histogram
		std::cout << "- \"hist_cumulative\" kernel execution time (ns): " << performance[3] << std::endl;
		std::cout << "- buffer read time (ns): " << performance[4] << std::endl;
		std::cout << std::endl; // line break
		std::cout << "Normalised histogram calculation:" << std::endl; // Performance monitoring for creating the normalised histogram
		std::cout << "- \"divide_array\" kernel execution time (ns): " << performance[5] << std::endl;
		std::cout << "- buffer read time (ns): " << performance[6] << std::endl;
		std::cout << std::endl; // line break
		std::cout << "Look up table calculation:" << std::endl; // Performance monitoring for creating the look up table
		std::cout << "- \"lut\" kernel execution time (ns): " << performance[7] << std::endl;
		std::cout << "- buffer read time (ns): " << performance[8] << std::endl;
		std::cout << std::endl; // line break
		std::cout << "Image enhancement:" << std::endl; // Performance monitoring for applying the look up table to the original
		std::cout << "- \"back_prop\" kernel execution time (ns) :" << performance[9] << std::endl;
		std::cout << "- buffer read time (ns): " << performance[10] << std::endl;
		std::cout << std::endl;
		std::cout << "Memory transfer time (ns): " << performance[0] + performance[2] + performance[4] + performance[6] + performance[8] + performance[10] << std::endl;
		std::cout << "Kernel execution time (ns): " << performance[1] + performance[3] + performance[5] + performance[7] + performance[9] << std::endl;
		std::cout << "Total program execution time (ns): " << std::accumulate(performance.begin(), performance.end(), 0) << std::endl;

	}
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}

	return 0;
}
