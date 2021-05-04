// Add you host code
#include <CL/cl.h>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <fstream>


unsigned int mem_tam_A;
unsigned int mem_tam_B;
unsigned int mem_tam_C;
float* h_A;
float* h_B;
float* C;

const char* KernelSource = " \n" \
"#define TILE_WIDTH 16 \n" \
"__kernel void MatrixMultSimple( \n" \
" __global float* mC, \n" \
" __global float* mA, \n" \
" __global float* mB, \n" \
" const int Mdim, const int Ndim, const int Pdim) \n" \
"{ \n" \
" __local float matrixAsub [TILE_WIDTH][TILE_WIDTH]; \n" \
" __local float matrixBsub [TILE_WIDTH][TILE_WIDTH]; \n" \
" int k; \n" \
" int i = get_global_id(0); \n" \
" int j = get_global_id(1); \n" \
" int Locali = get_local_id(0); \n" \
" int Localj = get_local_id(1); \n" \
" float sum = 0; \n" \
" for (k = 0; k < Pdim/TILE_WIDTH; k++){ \n" \
"	matrixAsub[Localj][Locali] = mA[j * Pdim + (Locali + k*TILE_WIDTH)];\n" \
"	matrixBsub[Localj][Locali] = mA[(Localj + k*TILE_WIDTH) * Pdim + i];\n" \
"	barrier ( CLK_LOCAL_MEM_FENCE); \n" \
"	for(int l = 0; l < TILE_WIDTH; l++){\n" \
"		sum += matrixAsub[Localj][l] * matrixBsub[l][Locali]; \n" \
"	}\n" \
"	barrier (CLK_LOCAL_MEM_FENCE); \n" \
" } \n" \
"	mC[j * Pdim + i] = sum; \n" \
"} \n" \
"\n";


void randomInit(float* data, int size) {
	for (int i = 0; i < size; ++i)
		data[i] = rand() / (float)RAND_MAX;
}

void MatMultOpencl(cl_uint Mdim, cl_uint Pdim, cl_uint Ndim, int tipoDisp) {
	// Declaración de las variables OpenCL
	cl_uint num_devs_returned = 0;
	cl_uint num_platforms_returned = 0;
	cl_uint start = 0;
	cl_uint end = 0;
	cl_context_properties properties[3];
	cl_device_id device_id;
	cl_int err;
	cl_platform_id platform_id[2];
	cl_context context;
	cl_command_queue command_queue;
	cl_program program;
	cl_kernel kernel;
	cl_mem input_buffer1, input_buffer2, output_buffer;
	cl_event events[2];
	size_t global[2];
	//Obtenemos el número de plataformas disponibles
	err = clGetPlatformIDs(0, NULL, &num_platforms_returned);
	if (err != CL_SUCCESS) {
		printf("Error al buscar plataformas\n");
		exit(1);
	}

	printf("Número de plataformas disponibles: %d\n", num_platforms_returned);
	//Obtenemos las plataformas disponibles
	err = clGetPlatformIDs(num_platforms_returned, platform_id, NULL);
	if (err != CL_SUCCESS) {
		printf("Error al buscar plataformas\n");
		exit(1);
	}


	//obtenemos el id del dispositivo que queremos
//Cogemos el id de la plataforma 1( cambiar de plataforma segun el dispositivo que queramos usar
	clGetDeviceIDs(platform_id[1], CL_DEVICE_TYPE_ALL, 1, &device_id, NULL);

	properties[0] = CL_CONTEXT_PLATFORM;
	properties[1] = (cl_context_properties)platform_id[1];
	properties[2] = 0;
	//creamos el contexto
	context = clCreateContext(properties, 1, &device_id, NULL, NULL, &err);
	if (!context) {
		printf("Error en la creacion del contexto\n");
		exit(1);
	}
	//creamos la cola de comandos
	command_queue = clCreateCommandQueue(context, device_id,
		CL_QUEUE_PROFILING_ENABLE, &err);
	if (!command_queue) {
		printf("Error al crear la cola de comandos\n");
		exit(1);
	}
	//creamos el objeto de programa
	program = clCreateProgramWithSource(context, 1, (const
		char**)&KernelSource, NULL, &err);
	if (!program) {
		printf("Error al crear el programa \n");
		exit(1);
	}
	//creamos el programa
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error número %d", err);
		printf("Error contruyendo el programa\n");
		char buffer[4096];
		size_t length;
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
			sizeof(buffer), buffer, &length);
		printf("%s\n", buffer);
		exit(1);
	}
	//creamos el kernel
	kernel = clCreateKernel(program, "MatrixMultSimple", &err);
	if (!kernel || err != CL_SUCCESS)
	{
		printf("%d", err);
		printf("Error al crear el kernel\n");
		exit(1);
	}
	//creamos los buffers
	cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_WRITE |
		CL_MEM_COPY_HOST_PTR, mem_tam_A, h_A, &err);
	if (err != CL_SUCCESS) {
		printf("Error al reservar espacio para la matriz A\n");
		exit(1);
	}

	cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_WRITE |
		CL_MEM_COPY_HOST_PTR, mem_tam_B, h_B, &err);
	if (err != CL_SUCCESS) {
		printf("Error al reservar espacio para la matriz B\n");
		exit(1);
	}

	cl_mem d_C = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_tam_C, NULL,
		&err);
	if (err != CL_SUCCESS) {
		printf("Error al reservar espacio para la matriz C\n");
		exit(1);
	}
	//Lista de argumentos

	clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&d_C);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&d_A);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&d_B);
	clSetKernelArg(kernel, 3, sizeof(int), &Mdim);
	clSetKernelArg(kernel, 4, sizeof(int), &Ndim);
	clSetKernelArg(kernel, 5, sizeof(int), &Pdim);

	global[0] = (size_t)Ndim;
	global[1] = (size_t)Mdim;

	size_t local[2];
	local[0] = 16;
	local[1] = 16;

	err = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global, local,
		0, NULL, &events[0]);
	if (err != CL_SUCCESS) {
		printf("%d", err);
		printf("Error al poner el kernel en cola.\n");
		exit(1);
	}

	err = clEnqueueReadBuffer(command_queue, d_C, CL_TRUE, 0, mem_tam_C, C, 0,
		NULL, &events[1]);
	if (err != CL_SUCCESS) {
		printf("Error al leer de la Matriz C\n");
		exit(1);
	}
	clReleaseMemObject(d_A);
	clReleaseMemObject(d_B);
	clReleaseMemObject(d_C);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clFlush(command_queue);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
}
int main(int argc, char* argv[]) {

	srand(time(NULL));
	double t;

	int Mdim, Ndim, Pdim;

	Mdim = 2000;
	Ndim = 2000;
	Pdim = 2000;

	// NxP * PxM = NxM
	//Estas variables permiten el producto de matrices no cuadradas

	unsigned int tam_A = Ndim * Pdim;
	mem_tam_A = sizeof(float) * tam_A;
	h_A = (float*)malloc(mem_tam_A);

	unsigned int tam_B = Pdim * Mdim;
	mem_tam_B = sizeof(float) * tam_B;
	h_B = (float*)malloc(mem_tam_B);

	unsigned int tam_C = Ndim * Mdim;
	mem_tam_C = sizeof(float) * tam_C;
	C = (float*)malloc(mem_tam_C);


	randomInit(h_A, tam_A);
	randomInit(h_B, tam_B);


	clock_t t0, t1;
	double segundos = 0;
	t0 = clock();

	MatMultOpencl(Ndim, Pdim, Mdim, CL_DEVICE_TYPE_GPU);

	t1 = clock();
	segundos = (double)(t1 - t0) / CLOCKS_PER_SEC;

	printf("Tiempo empleado: %f \n", segundos);
	printf("\nC[50][50] = %f", C[50 * Ndim + 50]);

	int aux;
	scanf("%d", &aux);

	free(h_A);
	free(h_B);
	free(C);
	return 0;
}