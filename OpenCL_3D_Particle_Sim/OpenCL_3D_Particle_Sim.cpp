// OpenCL_3D_Particle_Sim.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "opencl_stuff.h"
#include <GL\glut.h>
#include <fstream>
#include <sstream>
#include <complex>
#include "CustomTools.h"
#include <stdlib.h>
#include <windows.h>
#include <GL/glut.h>
#include <gdiplus.h>
#include <iostream>
#include <WinUser.h>
#include <cstdlib>
#include <math.h>
#include <cmath>
#include <time.h>
#include <fstream>
#include "stb_image.h"
#include <minmax.h>
#include <string>
#include <iomanip>


using namespace Tools;
using namespace Gdiplus;
using namespace std;

GLuint loadTexturePNG(const char *fname)
{
	int x, y, n;
	unsigned char *data = stbi_load(fname, &x, &y, &n, 0);
	GLuint textureId;
	glGenTextures(1, &textureId);
	glBindTexture(GL_TEXTURE_2D, textureId);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, x, y, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
	return textureId;
}

static const int nthread = 256;
static const int nt = 256;
static const int n = nthread * nt;

GLuint star, dust;

DataSet positions, velocities, forces;

cl_float4 graphics_buffer1[n], graphics_buffer2[n];

cl_float4 well_pos = {0,0,0,0};

float dt = 1, eps = 1e0;

float speed_limit = 1;
float big_g = 1; //6.6e-11;
float box_size = 1e5;
float particle_size = box_size / 1000;
float theta = 0, alpha = 0;
float rot_speed = box_size/720;
float zoom_scale = 0.01;
volatile int gb1_in_use = 0, gb2_in_use = 0;

int capture_rate = 500;

static const int screen_width = 1280;
uint8_t pixel_data[screen_width * screen_width * 3] = { 0 };
float pixel_heat[screen_width * screen_width] = { 0 };
float pixel_heat2[screen_width * screen_width *3] = { 0 };

int textures_on = false;
int point_brightness = 4;
int point_size = 2;

int GetEncoderClsid(const WCHAR* format, CLSID* pClsid)
{
	UINT  num = 0;		  // number of image encoders
	UINT  size = 0;		 // size of the image encoder array in bytes
	ImageCodecInfo* pImageCodecInfo = NULL;
	GetImageEncodersSize(&num, &size);
	if (size == 0)
		return -1;  // Failure
	pImageCodecInfo = (ImageCodecInfo*)(malloc(size));
	if (pImageCodecInfo == NULL)
		return -1;  // Failure
	GetImageEncoders(num, size, pImageCodecInfo);
	for (UINT j = 0; j < num; ++j)
	{
		if (wcscmp(pImageCodecInfo[j].MimeType, format) == 0)
		{
			*pClsid = pImageCodecInfo[j].Clsid;
			free(pImageCodecInfo);
			return j;  // Success
		}
	}
	free(pImageCodecInfo);
	return -1;
}
void createBMP(unsigned char *data, int w, int h, int comp, string filename)
{
	FILE *f;
	int filesize = 54 + comp * w * h;
	unsigned char bmpfileheader[14] = { 'B', 'M', 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0 };
	unsigned char bmpinfoheader[40] = { 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24, 0 };
	unsigned char bmppad[3] = { 0, 0, 0 };

	unsigned char *output_data = new unsigned char[w*h * 3];
	for (int i = 0; i < w*h*comp; i += comp)
	{
		output_data[i] = data[i + 2];
		output_data[i + 1] = data[i + 1];
		output_data[i + 2] = data[i + 0];
	}

	bmpfileheader[2] = (unsigned char)(filesize);
	bmpfileheader[3] = (unsigned char)(filesize >> 8);
	bmpfileheader[4] = (unsigned char)(filesize >> 16);
	bmpfileheader[5] = (unsigned char)(filesize >> 24);

	bmpinfoheader[4] = (unsigned char)(w);
	bmpinfoheader[5] = (unsigned char)(w >> 8);
	bmpinfoheader[6] = (unsigned char)(w >> 16);
	bmpinfoheader[7] = (unsigned char)(w >> 24);
	bmpinfoheader[8] = (unsigned char)(h);
	bmpinfoheader[9] = (unsigned char)(h >> 8);
	bmpinfoheader[10] = (unsigned char)(h >> 16);
	bmpinfoheader[11] = (unsigned char)(h >> 24);

	remove(filename.c_str());

	f = fopen(filename.c_str(), "wb");
	fwrite(bmpfileheader, 1, 14, f);
	fwrite(bmpinfoheader, 1, 40, f);
	for (int i = 0; i<h; i++)
	{
		fwrite(output_data + (w*(h - i - 1) * 3), 3, w, f);
		fwrite(bmppad, 1, (4 - (w * 3) % 4) % 4, f);
	}
	fclose(f);
}
void createJPG(unsigned char *data, int w, int h, int comp, string filename)
{
	FILE *f;
	int filesize = 54 + comp * w * h;
	unsigned char bmpfileheader[14] = { 'B', 'M', 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0 };
	unsigned char bmpinfoheader[40] = { 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24, 0 };
	unsigned char bmppad[3] = { 0, 0, 0 };

	unsigned char *output_data = new unsigned char[w*h * 3];
	for (int i = 0; i < w*h*comp; i += comp)
	{
		output_data[i] = data[i + 2];
		output_data[i + 1] = data[i + 1];
		output_data[i + 2] = data[i + 0];
	}

	bmpfileheader[2] = (unsigned char)(filesize);
	bmpfileheader[3] = (unsigned char)(filesize >> 8);
	bmpfileheader[4] = (unsigned char)(filesize >> 16);
	bmpfileheader[5] = (unsigned char)(filesize >> 24);

	bmpinfoheader[4] = (unsigned char)(w);
	bmpinfoheader[5] = (unsigned char)(w >> 8);
	bmpinfoheader[6] = (unsigned char)(w >> 16);
	bmpinfoheader[7] = (unsigned char)(w >> 24);
	bmpinfoheader[8] = (unsigned char)(h);
	bmpinfoheader[9] = (unsigned char)(h >> 8);
	bmpinfoheader[10] = (unsigned char)(h >> 16);
	bmpinfoheader[11] = (unsigned char)(h >> 24);

	f = fopen("temp.bmp", "wb");
	fwrite(bmpfileheader, 1, 14, f);
	fwrite(bmpinfoheader, 1, 40, f);
	for (int i = 0; i<h; i++)
	{
		fwrite(output_data + (w*(h - i - 1) * 3), 3, w, f);
		fwrite(bmppad, 1, (4 - (w * 3) % 4) % 4, f);
	}
	fclose(f);

	GdiplusStartupInput startupInput;
	ULONG_PTR token;
	GdiplusStartup(&token, &startupInput, NULL);
	Image testImg(L"temp.bmp", false);
	CLSID clsid;
	int ret = -1;
	if (-1 != GetEncoderClsid(L"image/jpeg", &clsid))
	{
		ret = testImg.Save(L"output.jpg", &clsid);
	}
	int result;
	char oldname[] = "output.jpg";
	result = rename(oldname, filename.c_str());
}

inline void cl2translate(cl_float4 * x)
{
	glTranslatef(x->s[0], x->s[1], x->s[2]);
}
inline void cl2v(cl_float4 * x)
{
	glVertex3f(x->s[0], x->s[1], x->s[2]);
}

vector<vec4> points;
inline void draw3D()
{
	start3dDraw();

	/*cl_float4 * set = graphics_buffer1;
	if (gb1_in_use)
		set = graphics_buffer2;
	else
		memcpy(graphics_buffer2, graphics_buffer1, 4 * sizeof(cl_float4));*/

	points.clear();
	
	glColor4f(0, 1, 0, 0.0);
	push();
	glBegin(GL_POINTS);
	for (int i = 0; i < n; ++i)
	{
		//vec4(vec4(forces.set[i].s[0], forces.set[i].s[1], forces.set[i].s[2]).magnitude2() * 10000000 + 0.5f * positions.set[i].s[3]).glColour4();
		//cl2v(&positions.set[i]);	
		points.push_back(w2s(vec4(positions.set[i].s[0], positions.set[i].s[1], positions.set[i].s[2])));
	}
	ge();
	pop();

	COLOUR_RED.glColour4();

	push();
	glTranslatef(-box_size/2,-box_size/2,-box_size/2);
	glScalef(box_size, box_size, box_size);
	lineCube();
	pop();


}
inline void drawDebugInfo()
{
	COLOUR_WHITE.glColour4();
	drawFPS();
}
inline void draw2D()
{
	start2dDraw();
	glPointSize(point_size);
	float s = box_size/worldPosition.magnitude();
	float cutoff = 0.5f;
	glColor4f(0.5, 0.5, 1, 0.01);
	if (textures_on)
		for (int i = 0; i < n; ++i)
		{
		

			//if (positions.set[i].s[3] > cutoff)
			//{
			//	push();
			//	points[i].glTranslate();
			//	glScalef(s/20, s/20, 0);
			//	//vec4(-0.5, -0.5).glTranslate();
			//	//fillRect();		
			//	drawRecTex(-150, -150, 300, 300, dust);
			//	pop();
			//}
			//else
			//{
				push();
				points[i].glTranslate();
				glScalef(s/2, s/2, 0);
				//vec4(-0.5, -0.5).glTranslate();
				//fillRect();		
				drawRecTex(-5.5, -5.5, 11, 11, star);
				pop();
			//}
		}
	
	vec4 force;
	float f;
	push();
	glBegin(GL_POINTS);
	for (int i = 0; i < n; ++i)
	{
		//vec4(vec4(forces.set[i].s[0], forces.set[i].s[1], forces.set[i].s[2]).magnitude2() * 10000000 + 0.5f * positions.set[i].s[3]).glColour4();
		//glPointSize(1000);
		//glColor4f(0.7, 0.7, 1, 0.1);
		//points[i].glVertex2();
		//glPointSize(1);		
		//if (positions.set[i].s[3] <= cutoff)
		{
			force = vec4(forces.set[i].s[0], forces.set[i].s[1], forces.set[i].s[2]);
			f = force.magnitude() * 1;
			glColor4f(f, f, 1, positions.set[i].s[3] / point_brightness);
			points[i].glVertex2();
		}
	}
	ge();
	pop();

	//gui->draw();
	drawDebugInfo();



	end2dDraw();
}
void renderScene()
{
	std::this_thread::sleep_for(std::chrono::milliseconds(10));

	glClearColor(0.1, 0.1, 0.1, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	draw3D();
	draw2D();

	//glFlush();
	glutSwapBuffers();
	glutPostRedisplay();

	incFFC();
}
void reshape3D(int w, int h)
{
	window = vec4(w, h);
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(fov, (float)w / (float)h, 0.001, 1000000000);
}
void customGLInit()
{
	//glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA | GLUT_MULTISAMPLE);
	glutInitWindowPosition(200, 100);
	glutInitWindowSize(window.x, window.y);
	glutCreateWindow(windowName.c_str());
	glutDisplayFunc(renderScene);
	glutReshapeFunc(reshape3D);
	//glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE);
	//glBlendFunc(GL_DST_COLOR, GL_ZERO);
	glEnable(GL_BLEND);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glEnable(GL_SMOOTH);
	glEnable(GL_POINT_SMOOTH);
	glShadeModel(GL_SMOOTH);
	
	worldPosition = vec4(0, box_size*2, -box_size);
}

void random_point(float radius, float *x, float *y, float *z) {
	float v1 = 2.0f * randf() - 1.0f;
	float v2 = sqrt(1.0f - v1*v1);
	complex<float> v3(pow(complex<float>(v1, v2), 1.0f / 3.0f));
	complex<float> v4(pow(complex<float>(v1, -v2), 1.0f / 3.0f));
	complex<float> v5(-(v3 + v4 + complex<float>(0, sqrt(3.0f)) * (v3 - v4)) / 2.0f);
	float height = v5.real();

	float flat_mag = sqrt(randf()) * radius * sqrt(1 - height * height);
	float flat_angle = randf() * 6.283185307179f;

	*x = cos(flat_angle) * flat_mag;
	*y = height * radius;
	*z = sin(flat_angle) * flat_mag;
}
void init_datasets()
{
	cl_float4 * p = new cl_float4[n];

	for (int i = 0; i < n; ++i)
	{
		//random_point(box_size / 2, &p[i].s[0], &p[i].s[1], &p[i].s[2]);
		
		do {
			p[i].s[0] = randf() * box_size - box_size / 2;
			p[i].s[1] = randf() * box_size - box_size / 2;
			p[i].s[2] = randf() * box_size - box_size / 2;
		} while (vec4(p[i].s[0]*1.5, p[i].s[1] * 50, p[i].s[2]).magnitude2() > (box_size / 2)*(box_size / 2));
		p[i].s[3] = randf() * 1;
		
	}
	p[0].s[0] = 0;
	p[0].s[1] = 0;
	p[0].s[2] = 0;
	p[0].s[3] = n/10;
	//for (int i = n/2; i < n; ++i)
	//{
	//	//random_point(box_size / 2, &p[i].s[0], &p[i].s[1], &p[i].s[2]);

	//	do {
	//		p[i].s[0] = randf() * box_size - box_size / 2;
	//		p[i].s[1] = randf() * box_size - box_size / 2;
	//		p[i].s[2] = randf() * box_size - box_size / 2;
	//	} while (vec4(p[i].s[0], p[i].s[1] * 6, p[i].s[2] * 3).magnitude2() >(box_size / 2)*(box_size / 2));
	//	p[i].s[3] = randf() * 1;

	//}

	positions = DataSet(p, n);

	cl_float4 * v = new cl_float4[n];

	for (int i = 0; i < n; ++i)
	{
		vec4 vel = vec4(p[i].s[0],0, p[i].s[2]);
		vec4 pos = vec4(p[i].s[0], p[i].s[1], p[i].s[2]);
		vel.normalize();
		vel.rotateY(pi/2);
		vel *= pos.magnitude()/500;

		//vel *= 25;
		pos.normalize();
		pos *= 25;
		v[i].s[0] = vel.x;
		v[i].s[1] = vel.y;
		v[i].s[2] = vel.z;
		v[i].s[3] = 0;
	}

	v[0].s[0] = 0;
	v[0].s[1] = 0;
	v[0].s[2] = 0;
	v[0].s[3] = 0;

	velocities = DataSet(v, n);

	cl_float4 * f = new cl_float4[n];

	for (int i = 0; i < n; ++i)
	{
		f[i].s[0] = 0;
		f[i].s[1] = 0;
		f[i].s[2] = 0;
		f[i].s[3] = 0;
	}

	forces = DataSet(f, n);

	positions.createBuffer();
	velocities.createBuffer();
	forces.createBuffer();
}
void release_datasets()
{
	positions.releaseBuffer();
	velocities.releaseBuffer();
	forces.releaseBuffer();
}
inline void program1()
{
	clSetKernelArg(kernel, 0, sizeof(float), &dt);
	clSetKernelArg(kernel, 1, sizeof(float), &eps);
	clSetKernelArg(kernel, 2, nthread * sizeof(cl_float4), nullptr);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &positions.buffer);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), &velocities.buffer);
	clSetKernelArg(kernel, 5, sizeof(int), &n);
	clSetKernelArg(kernel, 6, sizeof(cl_float4), &well_pos);
	clSetKernelArg(kernel, 7, sizeof(cl_mem), &forces.buffer);
	clSetKernelArg(kernel, 8, sizeof(float), &speed_limit);
	clSetKernelArg(kernel, 9, sizeof(float), &big_g);

	cl_event eve;
	const size_t globalWorkSize[] = { n, 0, 0 };
	const size_t localWorkSize[] = { nthread, 0, 0 };
	CheckError(clEnqueueNDRangeKernel(queue, kernel, 1,
		nullptr,
		globalWorkSize,
		localWorkSize,
		0, nullptr, &eve));

	CheckError(clFlush(queue));
}

int jpg_num = 0;
void take_snapshot()
{
	for (int i = 0; i < screen_width * screen_width*3; ++i)
		pixel_heat2[i] = 0;

	for (int i = 0; i < n; ++i)
	{
		vec4 s_pos = w2s(vec4(positions.set[i].s[0], positions.set[i].s[1], positions.set[i].s[2]));
		int x = s_pos.x / window.x * screen_width;
		int y = s_pos.y / window.y * screen_width;
		float heat = vec4(forces.set[i].s[0], forces.set[i].s[1], forces.set[i].s[2]).magnitude()* 10000;
		float power = positions.set[i].s[3];
		if (x >= 0 && x < screen_width && y >= 0 && y < screen_width)
			//	pixel_heat[(x + y * screen_width)] += heat * 10000;
		{
			pixel_heat2[(x + y * screen_width) * 3] += (heat * 1);
			//pixel_data[i * 3] %= 256;
			pixel_heat2[(x + y * screen_width) * 3+1] += (heat * 1);
			//pixel_data[i * 3 + 1] %= 256;
			pixel_heat2[(x + y * screen_width) * 3+2] += (1);
			//pixel_data[i * 3 + 2] %= 256;
		}
	}
	
	//for (int i = screen_width; i < screen_width * screen_width - screen_width; ++i)
	//	pixel_heat2[i] += pixel_heat[i] + pixel_heat[i + 1] + pixel_heat[i - 1] + pixel_heat[i + screen_width] + pixel_heat[i - screen_width];

	//for (int i = 0; i < screen_width * screen_width; ++i)
	//	pixel_heat[i] = pixel_heat2[i]/5;

	for (int i = 0; i < screen_width * screen_width*3; ++i)
	{
		pixel_data[i] = (uint8_t)(pixel_heat2[i]);
		//pixel_data[i * 3 + 1] = (uint8_t)(pixel_heat2[i]);
		//pixel_data[i * 3 + 2] = (uint8_t)(pixel_heat2[i]);
	}
	
	
	//createBMP(pixel_data, screen_width, screen_width, 3, "temp.bmp");
	stringstream s;
	//s << "images/img-" << setfill('0') << setw(3) << jpg_num << ".bmp";
	s << "images/img-" << jpg_num << ".jpg"; //setfill('0') << setw(3) << 
	cout << "[Main] image - " << s.str() << endl;
	//createBMP(pixel_data, screen_width, screen_width, 3, s.str());
	createJPG(pixel_data, screen_width, screen_width, 3, s.str());
	jpg_num++;
}

void exit_function()
{
	std::cout << "[Main] Releasing OpenCL data...\n";
	release_datasets();
	openclRelease();

	"ffmpeg -r 60 -f image2 -s 1280x720 -i img-%03d.bmp -vcodec libx264 -crf 25 -pix_fmt yuv420p test.mp4";

	std::cout << "[Main] Press enter to close...";
	//getc(stdin);
	killApp();
}
void keys()
{
	keyboard->run();

	float r = rot_speed;
	/*if (keyboard->keyDown(VK_LSHIFT))
		r *= 10;
	if (keyboard->keyDown(VK_LEFT))
		theta += r;
	if (keyboard->keyDown(VK_RIGHT))
		theta -= r;
	if (keyboard->keyDown(VK_UP))
		alpha -= r;
	if (keyboard->keyDown(VK_DOWN))
		alpha += r;
		*/
	if (keyboard->keyDownEvent(VK_RETURN))
		take_snapshot();

	/*if (keyboard->keyDown('W'))
		worldPosition = vec4(0, 0, -box_size * (zoom_scale += 0.001f));
	if (keyboard->keyDown('S'))
		worldPosition = vec4(0, 0, -box_size * (zoom_scale -= 0.001f));*/

		// keyboard controls
	{
		vec4 ldV(lookDirection.x, 0, lookDirection.z);
		ldV.normalize();

		// wasdqe keys
		float mspd = keyboard->keyDown(VK_SHIFT) ? rot_speed * 10 : rot_speed;
		if (keyboard->keyDown(0x57))
		{
			worldPosition += lookDirection * zoom_scale;
		}
		if (keyboard->keyDown(0x53))
		{
			worldPosition += lookDirection * -zoom_scale;
		}
		if (keyboard->keyDown(0x41))
		{
			worldPosition += ldV.normalY() * -mspd;
		}
		if (keyboard->keyDown(0x44))
		{
			worldPosition += ldV.normalY() * mspd;
		}
		if (keyboard->keyDown(0x51))
		{
			worldPosition += upVec * -mspd;
		}
		if (keyboard->keyDown(0x45))
		{
			worldPosition += upVec * mspd;
		}

		if (keyboard->keyDownEvent('T'))
		{
			textures_on = !textures_on;
		}
		if (keyboard->keyDownEvent('P'))
		{
			point_brightness++;
			point_brightness %= 8;
			if (point_brightness == 0) point_brightness = 1;
		}
		if (keyboard->keyDownEvent('O'))
		{
			point_size++;
			point_size %= 10;
			if (point_size == 0) point_size = 1;
		}

		/*if (keyboard->keyDown(VK_LEFT))
			worldPosition += lookDirection.rotatedY(-pi/2) * mspd;
		if (keyboard->keyDown(VK_RIGHT))
			worldPosition += lookDirection.rotatedY(pi / 2) * mspd;*/
		//if (keyboard->keyDown(VK_UP))
		//	worldPosition += lookDirection.rotatedY(-pi / 2);
		//if (keyboard->keyDown(VK_DOWN))
		//	worldPosition += lookDirection.rotatedY(-pi / 2);
	}
	
	//if (keyboard->keyDownEvent(VK_ESCAPE))
	//	exit_function();
}

int ticker = 0;
void processThread()
{
	while (1)
	{
		startProcessLoop();
		keys();
		program1();

		positions.readBuffer();
		forces.readBuffer();

		//for (int i = 0; i < n; ++i)
		//{
		//	if (vec4(positions.set[i].s[0], positions.set[i].s[1], positions.set[i].s[2]).magnitude2() > box_size * box_size)
		//	{
		//		positions.set[i].s[0] = randf() * box_size - box_size / 2;
		//		positions.set[i].s[1] = randf() * box_size - box_size / 2;
		//		positions.set[i].s[2] = randf() * box_size - box_size / 2;
		//	}
		//}

		//gb1_in_use = 1;
		//memcpy(graphics_buffer1, positions.set, 4 * sizeof(cl_float4));
		//gb1_in_use = 0;

		lookDirection = vec4(worldPosition) * -1;
		//WworldPosition = vec4(cos(theta)*box_size, 0, sin(theta)*box_size);
		//theta += 0.0001;
		

		ticker++;
		if (ticker == capture_rate)
		{
			take_snapshot();
			ticker = 0;
		}

		endProcessLoop();
	}
}
int main(int argc, char* argv[])
{
	srand(time(0));

	std::cout << "[Main] Setting up OpenCL components...\n";
	openclSetup();

	// Prepare some test data

	std::cout << "[Main] Creating datasets...\n";
	init_datasets();
	
	window = vec4(screen_width, screen_width);

	std::cout << "[OpenGL] Setting up OpenGL...\n";
	glutInit(&argc, argv);
	customGLInit();
	win = FindWindowA(NULL, windowName.c_str());
	
	std::thread process(processThread);
	process.detach();

	detachFPSThread();

	//detachEventHandler();

	star = loadTexturePNG("star.png");
	dust = loadTexturePNG("dust.png");

	std::cout << "[OpenGL] Starting OpenGL...\n";
	glutMainLoop();
	
	return 0;
}

