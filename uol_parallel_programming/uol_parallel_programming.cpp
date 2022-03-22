// uol_parallel_programming.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "Utils.h" // import OpenCL tools
#include "CImg.h" // import CImg tools

using namespace cimg_library;

int main()
{
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

    while (!disp_input.is_closed() && !disp_input.is_keyESC() ) { // if user has not closed image
        disp_input.wait(1); // keep the image displayed
    }

}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
