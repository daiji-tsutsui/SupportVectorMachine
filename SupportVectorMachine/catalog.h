//
//  catalog.h
//  SupportVectorMachine
//
//  Created by 筒井 大二 on 2018/06/27.
//  Copyright © 2018年 筒井 大二. All rights reserved.
//
#include "linear.h"
#ifndef catalog_h
#define catalog_h

void square_root_test(){
	//27.6.2018
	//Test for eigendecomp and sqrt
	mat C(4,4);
	
	C.rand(); C.print("C");
	C = C * C.transpose(); C.print("C C^t");
	
	vect l; mat U;
	C.eigendecomp(&l, &U);
	l.print("lambda");
	U.print("U");
	
	(U*(U.transpose()))->print("U U^t");
	(U * ((*l.diagonal()) * (*U.transpose())))->print("U D U^t");
	
	C = C.sqrt(); C.print("sqrt(C C^t)");
	C = (C * C); C.print("sqrt(C C^t)^2");
}

void destruvtive_op(){
	//27.6.2018
	//To save memory, introduce destructive operator
	
	vect A; A.rand();
	vect d;		//nothing -> memory used: 5.2MB
	for(int i=0; i<50000; i++) d = A + *(A + A);	// -> memory used: 6.7MB
	for(int i=0; i<50000; i++) d = A + (A + A);		// -> memory used: 5.2MB

	getchar();
}

#endif /* catalog_h */
