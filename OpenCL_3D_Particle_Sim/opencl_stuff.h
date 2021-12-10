#pragma once
#include <CL\cl.hpp>
#include <CL\opencl.h>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

// globals

const char * ker = "program";

cl_command_queue queue;
cl_kernel kernel;
std::vector<cl_device_id> deviceIds;
cl_int error;
cl_context context;


// base function for CL

std::string GetPlatformName(cl_platform_id id)
{
	size_t size = 0;
	clGetPlatformInfo(id, CL_PLATFORM_NAME, 0, nullptr, &size);

	std::string result;
	result.resize(size);
	clGetPlatformInfo(id, CL_PLATFORM_NAME, size,
		const_cast<char*> (result.data()), nullptr);

	return result;
}
std::string GetDeviceName(cl_device_id id)
{
	size_t size = 0;
	clGetDeviceInfo(id, CL_DEVICE_NAME, 0, nullptr, &size);

	std::string result;
	result.resize(size);
	clGetDeviceInfo(id, CL_DEVICE_NAME, size,
		const_cast<char*> (result.data()), nullptr);

	return result;
}
inline void CheckError(cl_int error)
{
	if (error != CL_SUCCESS) {
		std::cerr << "[OpenCL] OpenCL call failed with error " << error << std::endl;
		std::system("PAUSE");
		std::exit(1);
	}
}
std::string LoadKernel(const char* name)
{
	std::ifstream in(name);
	std::string result(
		(std::istreambuf_iterator<char>(in)),
		std::istreambuf_iterator<char>());
	return result;
}
cl_program CreateProgram(const std::string& source,
cl_context context)
{
	size_t lengths[1] = { source.size() };
	const char* sources[1] = { source.data() };

	cl_int error = 0;
	cl_program program = clCreateProgramWithSource(context, 1, sources, lengths, &error);
	CheckError(error);

	return program;
}
int openclSetup()
{
	////////////////////////////////////////////////////////////////////////////////////////////////////////
	cl_uint platformIdCount = 0;
	clGetPlatformIDs(0, nullptr, &platformIdCount);

	if (platformIdCount == 0) {
		std::cerr << "[OpenCL] No OpenCL platform found" << std::endl;
		return 1;
	}
	else {
		std::cout << "[OpenCL] Found " << platformIdCount << " platform(s)" << std::endl;
	}

	std::vector<cl_platform_id> platformIds(platformIdCount);
	clGetPlatformIDs(platformIdCount, platformIds.data(), nullptr);

	for (cl_uint i = 0; i < platformIdCount; ++i) {
		std::cout << "[OpenCL] \t (" << (i + 1) << ") : " << GetPlatformName(platformIds[i]) << std::endl;
	}

	// http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clGetDeviceIDs.html
	cl_uint deviceIdCount = 0;
	clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_ALL, 0, nullptr,
		&deviceIdCount);

	if (deviceIdCount == 0) {
		std::cerr << "[OpenCL] No OpenCL devices found" << std::endl;
		return 1;
	}
	else {
		std::cout << "[OpenCL] Found " << deviceIdCount << " device(s)" << std::endl;
	}

	std::vector<cl_device_id> deviceIds(deviceIdCount);
	clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_ALL, deviceIdCount,
		deviceIds.data(), nullptr);

	for (cl_uint i = 0; i < deviceIdCount; ++i) {
		std::cout << "[OpenCL] \t (" << (i + 1) << ") : " << GetDeviceName(deviceIds[i]) << std::endl;
	}

	// http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clCreateContext.html
	const cl_context_properties contextProperties[] =
	{
		CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties> (platformIds[0]),
		0, 0
	};

	cl_int error = CL_SUCCESS;
	context = clCreateContext(contextProperties, deviceIdCount,
		deviceIds.data(), nullptr, nullptr, &error);
	CheckError(error);

	std::cout << "[OpenCL] Context created" << std::endl;
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////

	cl_program program = CreateProgram(LoadKernel("kernel.cl"),
		context);
	CheckError(clBuildProgram(program, deviceIdCount, deviceIds.data(), nullptr, nullptr, nullptr));

	std::cout << "[OpenCL] Program built" << std::endl;

	kernel = clCreateKernel(program, ker, &error);
	CheckError(error);

	std::cout << "[OpenCL] Kernel program found" << std::endl;

	queue = clCreateCommandQueue(
		context,
		deviceIds[0],
		0, &error);
	CheckError(error);

	std::cout << "[OpenCL] Queue created" << std::endl;
}
void openclRelease()
{
	clReleaseCommandQueue(queue);
	clReleaseKernel(kernel);
	clReleaseContext(context);
}

// structs

struct DataSet
{
	cl_float4 * set;
	cl_mem buffer;
	int set_size;

	DataSet()
	{
		set = nullptr;
		set_size = 0;
	}

	DataSet(cl_float4 * set, int set_size)
	{
		this->set = set;
		this->set_size = set_size;
	}

	
	void createBuffer()
	{
		buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			sizeof(cl_float4) * (set_size),
			set, &error);
		CheckError(error);
	}

	void readBuffer()
	{
		CheckError(clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0,
			sizeof(cl_float4) * set_size,
			set,
			0, nullptr, nullptr));
	}

	void releaseBuffer()
	{
		clReleaseMemObject(buffer);
	}

	void freeSet()
	{
		free(set);
	}
	
};