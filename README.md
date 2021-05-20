## RAY TRACING WITH CUDA ##

In order to properly launch the project:

1. change the version of CUDA used in the file _cudaraytracer.vcxproj_;
2. change 'Code Generation' in '_Properties_' &#8594; '_CUDA C/C++_' &#8594; '_Device_' accordingly to the NVIDIA hardware that will run the code;
3. add 'extended-lambda' in '_Properties_' &#8594; '_CUDA/C++_' &#8594; '_Command Line_'.

### Examples ###

![](https://raw.githubusercontent.com/biancofla/cuda-raytracing/main/renders/gifs/multiple_spheres.m4v?raw=true)

![](https://github.com/biancofla/cuda-raytracing/blob/main/renders/cornell_box.bmp?raw=true)

![](https://github.com/biancofla/cuda-raytracing/blob/main/renders/dragon.bmp?raw=true)
