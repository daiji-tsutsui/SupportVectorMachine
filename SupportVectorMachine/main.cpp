//
//  main.cpp
//  SupportVectorMachine
//
//  Created by 筒井 大二 on 2018/06/22.
//  Copyright © 2018年 筒井 大二. All rights reserved.
//

#include <iostream>
#include <OpenGL/OpenGL.h>
#include <GLUT/GLUT.h>
#include <random>
#include "linear.h"

using namespace std;

void drawCircle(double x0, double y0, double r, int div=50);
void drawTriangle(double x0, double y0, double r);
double hinge(double u);
double zero_one(double u);

random_device rnd;     // 非決定的な乱数生成器
mt19937 mt(rnd());  // メルセンヌ・ツイスタの32ビット版、引数は初期シード
uniform_real_distribution<> unif(0.0, 1.0);   // [0.0,1.0]上一様に分布させる
normal_distribution<> gauss(0.0, 1.0);   // 平均0.0、標準偏差1.0で分布させる

double winsize = 1000.0;
int datanum = 500;
mat X(datanum, 3);		//デザイン行列
double m0[2] = {-0.2, -0.3};
double m1[2] = {0.1, 0.2};
vect** m = (vect**)calloc(2, sizeof(vect*));
double s0[4] = {0.03, -0.015,
				-0.015, 0.04};
double s1[4] = {0.05, -0.01,
				-0.01, 0.02};
mat** s = (mat**)calloc(2, sizeof(mat*));
int cnt = 0;
vect A(2);
double b;
double c = 100.0;		//正則化パラメータ
double eps = 1e-4;		//数値微分のための小数
double lRate = 0.001;	//学習定数

/*--For OpenGL-------------------------------------------------------------------------*/
void idle(void){
	glutPostRedisplay();
}
void setup(void) {
	glClearColor(1.0, 1.0, 1.0, 1.0);       //White
}
void resize(int width, int height) {
	glViewport(0, 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0,
				   (double)width/height,
				   0.1,
				   100.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(0.0, 0.0, 2.5,       //Position of Camera
			  0.0, 0.0, 0.0,        //Position of Object
			  0.0, 1.0, 0.0);       //Upward direction of Camera
}


/*--Display func-------------------------------------------------------------------------*/
void display(void){
	
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//	glutWireTeapot(0.5);
	
//	glPointSize(3.0);
	for(int i=0; i<datanum; i++){
		if(X.a[i * 3 + 2] > 0){
			glColor3d(1.0, 0.0, 0.0);	//Red
		}else{
			glColor3d(0.1, 0.6, 0.1);	//Green
		}
//		drawCircle(X.a[i * 3 + 0], X.a[i * 3 + 1], 0.006);
		drawTriangle(X.a[i * 3 + 0], X.a[i * 3 + 1], 0.01);
	}
	
	glColor3d(0.0, 0.0, 0.0);
	glBegin(GL_LINES);
		if(abs((A.v[0]+b)/A.v[1]) > 1.0) glVertex2d(1.0, -(A.v[0]+b)/A.v[1]);
		if(abs((A.v[1]+b)/A.v[0]) > 1.0) glVertex2d(-(A.v[1]+b)/A.v[0], 1.0);
		if(abs((A.v[0]-b)/A.v[1]) > 1.0) glVertex2d(-1.0, (A.v[0]-b)/A.v[1]);
		if(abs((A.v[1]-b)/A.v[0]) > 1.0) glVertex2d((A.v[1]-b)/A.v[0], -1.0);
	glEnd();
	
/*----------------------------------------------------------------------*/
/*                   |A| + c * h(y*(Ax+b))	を最小化する                  */
/*----------------------------------------------------------------------*/
	//数値微分による勾配法
	vect A1(2); A1.v[0] = A.v[0] + eps; A1.v[1] = A.v[1];
	vect A2(2); A2.v[0] = A.v[0]; A2.v[1] = A.v[1] + eps;
	double b3 = b + eps;
	
	double f = A.norm();
	double f1 = A1.norm();
	double f2 = A2.norm();
	double f3 = A.norm();
	vect x(2); double y;
	Function* loss = hinge;
	for(int i=0; i<datanum; i++){
		x.v[0] = X.a[i * 3 + 0];
		x.v[1] = X.a[i * 3 + 1];
		y = X.a[i * 3 + 2];
		f += c * loss(y * (A*x+b)) / datanum;
		f1 += c * loss(y * (A1*x+b)) / datanum;
		f2 += c * loss(y * (A2*x+b)) / datanum;
		f3 += c * loss(y * (A*x+b3)) / datanum;
	}
	
	double df1 = (f1 - f)/eps;
	double df2 = (f2 - f)/eps;
	double df3 = (f3 - f)/eps;
	A.v[0] -= lRate * df1;
	A.v[1] -= lRate * df2;
	b -= lRate * df3;
	
	glFlush();
}


/*--Main func-------------------------------------------------------------------------*/
int main(int argc, char * argv[]) {
	/*--Initialize-------*/
	for(int i=0; i<2; i++) m[i] = new vect(2);
	m[0]->v = m0; m[1]->v = m1;
	for(int i=0; i<2; i++) s[i] = new mat(2,2);
	s[0]->a = s0; s[1]->a = s1;
	
	vect rn(2); vect temp;
	for(int i=0; i<datanum; i++){
		rn.rand();
		if(gauss(mt) < 0.0){
			temp = (*m[0]) + (rn * (*s[0]).sqrt());
			X.a[i * 3 + 0] = temp.v[0];
			X.a[i * 3 + 1] = temp.v[1];
			X.a[i * 3 + 2] = -1.0;
		}else{
			temp = (*m[1]) + (rn * (*s[1]).sqrt());
			X.a[i * 3 + 0] = temp.v[0];
			X.a[i * 3 + 1] = temp.v[1];
			X.a[i * 3 + 2] = 1.0;
		}
	}
//	X.rand(0.0, 0.2);
	A.rand();
	b = gauss(mt)*0.001;
	mat C(4,4); C.rand(); C.print("C");
	
//	getchar();
	
	/*--Main loop-------*/
	glutInit(&argc, argv);
	glutInitWindowSize(winsize, winsize);
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA | GLUT_DEPTH);
	glutCreateWindow("Support Vector Machine");
	glutReshapeFunc(resize);
	glutDisplayFunc(display);
	glutIdleFunc(idle);
	setup();
	glutMainLoop();
	
	return 0;
}


/*--Other func-------------------------------------------------------------------------*/
void drawCircle(double x0, double y0, double r, int div){
	double x, y;
	glBegin(GL_POLYGON);
	for (int i=0; i<div; i++) {
		x = x0 + r * cos(2.0 * M_PI * ((double)i/div));
		y = y0 + r * sin(2.0 * M_PI * ((double)i/div));
		glVertex2d(x, y);
	}
	glEnd();
}
void drawTriangle(double x0, double y0, double r){
	double x, y;
	glBegin(GL_POLYGON);
	for (int i=0; i<4; i++) {
		x = x0 + r * cos(2.0 * M_PI * (0.25 + i/3.0));
		y = y0 + r * sin(2.0 * M_PI * (0.25 + i/3.0));
		glVertex2d(x, y);
	}
	glEnd();
}

double hinge(double u){
	if(u >= 1){
		return 0.0;
	}else{
		return 1.0 - u;
	}
}
double zero_one(double u){
	if(u >= 0){
		return 0.0;
	}else{
		return 1.0;
	}
}

		
		
		


