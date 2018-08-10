
#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <iomanip>
using namespace std;

//24.6.2018
//線形代数をC++でやりたいのならもっとちゃんと便利にする必要がある
//連続関数カルキュラスで戸惑ってちゃ話にならんし
//ランダム行列の初期化も全自動でやりたい
//行分解，列分解とか色々楽にできるようにすべき

//25.6.2018
//何かと日付を残しておくのがいいと気づいた（バージョン管理とかの意味で）．
//いろんな実装をしながらその実装（をエレガントに書くの）に必要な機能を作っていこう


//MathematicalFunctions ----+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
typedef double Function(double);
double Identity(double x) { return x; }
double d_Identity(double x) { return 1.0; }
double sigmoid(double x) {
	if (x > 0.0) {
		return 1.0 / (1.0 + exp(-2.0*x));
	}
	else {
		return exp(2.0*x) / (1.0 + exp(2.0*x));
	}
}                   //sigmoid function
double d_sigmoid(double x) {
	if (x > 0.0) {
		return 2.0 * exp(-2.0*x) / pow(1.0 + exp(-2.0*x), 2.0);
	}
	else {
		return 2.0 * exp(2.0*x) / pow(1.0 + exp(2.0*x), 2.0);
	}
}                //derivative of sigmoidal
double d_tanh(double x) {
	return 1.0 / (cosh(x)*cosh(x));
}                   //derivative of tanh
double PLtanh(double x) {
	return max(-1.0, min(x, 1.0));
}					//piecewise-linear approximating tanh
double d_PLtanh(double x) {
	if (abs(x) < 1.0) return 1.0;
	return 0.0;
}					//derivative of PLtanh
double relu(double x) {
	if (x > 0.0) {
		return x;
	}
	else {
		return 0.0;
	}
}
double d_relu(double x) {
	if (x > 0.0) {
		return 1.0;
	}
	else {
		return 0.0;
	}
}


//Vector and Matrix ----+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
class vect;
class mat;
double *dcalloc(int n) { return (double *)calloc(n, sizeof(double)); }
void QR_decomp(mat& src, mat* Q, mat* R);

class vect {
public:
	double *v;
	int size;

	vect();
	vect(int n);
	vect(double* src, int n);
	vect(vector<double> src);
	~vect();
	void rand(double mean = 0.0, double sigma = 1.0);
	void zeros(int n);
	void ones(int n);
	void scale(double r);
	void print(string str = "", int bl = 3, double err = 1e-10);
	void printc(fstream& file, string chr = ", ");
	void copy(const vect &src);
	void act(Function func);
	double norm();
	mat* diagonal();

	double operator [](int id);
	const double operator [](int id) const;
	double operator *(const vect &src);
	double operator *(vect* src);			//destructive one
	vect* operator *(const double src);
	vect* operator +(const vect &src);
	vect* operator +(vect* src);			//destructive one
	vect* operator -(const vect &src);
	vect* operator -(vect* src);			//destructive one
	vect* HProd(const vect &src);
	vect* HProd(vect* src);					//destructive one
	vect* operator *(const mat &src);
	vect* operator *(mat* src);				//destructive one
	vect& operator =(const vect &src);
	vect* operator =(vect *src);			//destructive one
};

class mat {
public:
	double *a;
	int rsize;
	int csize;

	mat();
	mat(int r, int c);
	mat(double* src, int r, int c);
	mat(vector<vector<double>> src);
	~mat();
	void rand(double mean = 0.0, double sigma = 1.0);
	void zeros(int r, int c);
	void ones(int r, int c);
	void identity(int n);
	void scale(double r);
	void eigendecomp(vect* lambda, mat* U);
	mat* sqrt();
	void print(string str = "", int bl = 3, double err = 1e-10);
	void printc(fstream& file, string chr = ", ");
	void copy(const mat &src);
	mat* transpose();
	void act(Function func);
//	void calc(Function func);
	double v(int i, int j);
	vect* col(int c);
	vect* row(int r);

	double operator ()(int i, int j);
	vect* operator *(const vect &src);
	vect* operator *(vect* src);			//destructive one
	mat* operator +(const mat &src);
	mat* operator +(mat* src);				//destructive one
	mat* operator -(const mat &src);
	mat* operator -(mat* src);				//destructive one
	mat* HProd(const mat &src);
	mat* HProd(mat* src);					//destructive one
	mat* operator *(const mat &src);
	mat* operator *(mat* src);				//destructive one
	mat& operator =(mat &src);
	mat* operator =(mat* src);				//destructive one
};

vect::vect() { size = -1; v = NULL; }
vect::vect(int n) { size = n; v = dcalloc(size); }
vect::vect(double* src, int n) { 
	size = n; v = dcalloc(size); 
	for(int i = 0; i < n; i++){
		v[i] = src[i];
	}
}
vect::vect(vector<double> src) {
	this->size = src.size();
	v = dcalloc(this->size);
	
	for (int i = 0, n = this->size; i < n; i++) {
		v[i] = src[i];
	}
}
vect::~vect() {
	if (v != NULL) free(v);
	v = NULL;
	size = -1;
}
void vect::rand(double mean, double sigma){
	random_device rnd;
	mt19937 mt(rnd());
	uniform_real_distribution<> unif(0.0, 1.0);   // [0.0,1.0]上一様に分布させる
	normal_distribution<> gauss(0.0, 1.0);   // 平均0.0、標準偏差1.0で分布させる
	for(int i=0; i<size; i++) v[i] = mean + sigma * gauss(mt);
}
void vect::zeros(int n) {
	if (n > 0) { size = n; }
	if (v != NULL) free(v);
	v = dcalloc(size);
}
void vect::ones(int n) {
	zeros(n);
	double* ptr = v;
	for (int i = 0, n = size; i < n; i++) {
		ptr[i] = 1.0;
	}
}
void vect::scale(double r) {
	double* ptr = v;
	for (int i = 0, n = size; i < n; i++) {
		ptr[i] = ptr[i] * r;
	}
}
void vect::print(string str, int bl, double err) {
	double* ptr = v;
	if(str == ""){
		cout << "Vector " << size << "-dim" << endl;
	}else{
		cout << str << ": Vector " << size << "-dim" << endl;
	}
	cout << "   ";
	for (int i = 0, n = size; i < n; i++) {
		if(abs(ptr[i]) < err){
			cout << setw(bl+3) << right << showpoint << setprecision(bl) << 0.0 << " ";
		}else{
			cout << setw(bl+3) << right << showpoint << setprecision(bl) << ptr[i] << " ";
		}
	}
	cout << endl << endl;
}
void vect::printc(fstream& file, string chr){
	double* ptr = v;
	for(int i=0; i<size; i++){
		file << setprecision(15) << ptr[i] << chr;
	}
}
void vect::copy(const vect &src) {
	double* this_ptr;
	double* src_ptr = src.v;
	if (src.size <= 0) {
		cout << "ERROR(vect.copy): invalid size of vect" << endl;
	}
	else {
		zeros(src.size);
		this_ptr = this->v;
		for (int i = 0, n = size; i < n; i++) {
			this_ptr[i] = src_ptr[i];
		}
	}
}
void vect::act(Function func) {
	double* ptr = v;
	for (int i = 0, n = size; i < n; i++) {
		ptr[i] = func(ptr[i]);
	}
}
double vect::norm(){
	double r = 0;
	for(int i=0; i<size; i++){
		r += v[i] * v[i];
	}
	return sqrt(r);
}
mat* vect::diagonal(){
	int n = this->size;
	mat* trg = new mat(n, n);
	for(int i=0; i<n; i++){
		trg->a[i * n + i] = v[i];
	}
	return trg;
}

double vect::operator [](int id) {
	if (id >= size || id < 0) {
		cout << "ERROR(vect.[]): invalid index" << endl;
		return 0;
	}
	return v[id];
}
const double vect::operator [](int id) const {
	if (id >= size || id < 0) {
		cout << "ERROR(vect.[]): invalid index" << endl;
		return 0;
	}
	return v[id];
}
double vect::operator *(const vect &src) {
	if (this->size != src.size) {
		cout << "ERROR(vect.*): invalid inner product" << endl;
		return 0.0;
	}
	double sum = 0.0;
	double* this_ptr = this->v;
	double* src_ptr = src.v;
	for (int i = 0, n = size; i < n; i++) {
		sum += this_ptr[i] * src_ptr[i];
	}
	return sum;
}
double vect::operator *(vect* src) {
	if (this->size != src->size) {
		cout << "ERROR(vect.*): invalid inner product" << endl;
		return 0.0;
	}
	double sum = 0.0;
	double* this_ptr = this->v;
	double* src_ptr = src->v;
	for (int i = 0, n = size; i < n; i++) {
		sum += this_ptr[i] * src_ptr[i];
	}
	delete src;
	return sum;
}			//destructive one
vect* vect::operator *(const double src) {
	int n = size;
	vect *trg; trg = new vect(n);
	double* trg_ptr = trg->v;
	for(int i = 0; i < n; i++){
		trg_ptr[i] = src * v[i];
	}
	return trg;
}
vect* vect::operator +(const vect &src) {
	if (this->size != src.size) {
		cout << "ERROR(vect.+): invalid sum" << endl;
		return 0;
	}
	int n = size;
	vect *trg; trg = new vect(n);
	double* this_ptr = this->v;
	double* src_ptr = src.v;
	double* trg_ptr = trg->v;
	for (int i = 0; i < n; i++) {
		trg_ptr[i] = this_ptr[i] + src_ptr[i];
	}
	return trg;
}
vect* vect::operator +(vect* src) {
	if (this->size != src->size) {
		cout << "ERROR(vect.+): invalid sum" << endl;
		return 0;
	}
	int n = size;
	vect *trg; trg = new vect(n);
	double* this_ptr = this->v;
	double* src_ptr = src->v;
	double* trg_ptr = trg->v;
	for (int i = 0; i < n; i++) {
		trg_ptr[i] = this_ptr[i] + src_ptr[i];
	}
	
	delete src;
	return trg;
}			//destructive one
vect* vect::operator -(const vect &src) {
	if (this->size != src.size) {
		cout << "ERROR(vect.-): invalid subtraction" << endl;
		return 0;
	}
	int n = size;
	vect *trg; trg = new vect(n);
	double* this_ptr = this->v;
	double* src_ptr = src.v;
	double* trg_ptr = trg->v;
	for (int i = 0; i < n; i++) {
		trg_ptr[i] = this_ptr[i] - src_ptr[i];
	}
	return trg;
}
vect* vect::operator -(vect* src) {
	if (this->size != src->size) {
		cout << "ERROR(vect.-): invalid subtraction" << endl;
		return 0;
	}
	int n = size;
	vect *trg; trg = new vect(n);
	double* this_ptr = this->v;
	double* src_ptr = src->v;
	double* trg_ptr = trg->v;
	for (int i = 0; i < n; i++) {
		trg_ptr[i] = this_ptr[i] - src_ptr[i];
	}
	
	delete src;
	return trg;
}			//destructive one
vect* vect::HProd(const vect &src) {
	if (this->size != src.size) {
		cout << "ERROR(vect.Hprod): invalid Hadamard product" << endl;
		return 0;
	}
	int n = size;
	vect *trg; trg = new vect(n);
	double* this_ptr = this->v;
	double* src_ptr = src.v;
	double* trg_ptr = trg->v;
	for (int i = 0; i < n; i++) {
		trg_ptr[i] = this_ptr[i] * src_ptr[i];
	}
	return trg;
}
vect* vect::HProd(vect* src) {
	if (this->size != src->size) {
		cout << "ERROR(vect.Hprod): invalid Hadamard product" << endl;
		return 0;
	}
	int n = size;
	vect *trg; trg = new vect(n);
	double* this_ptr = this->v;
	double* src_ptr = src->v;
	double* trg_ptr = trg->v;
	for (int i = 0; i < n; i++) {
		trg_ptr[i] = this_ptr[i] * src_ptr[i];
	}
	
	delete src;
	return trg;
}					//destructive one
vect* vect::operator *(const mat &src) {
	if (this->size != src.rsize) {
		cout << "ERROR(vect.*): invalid product" << endl;
		return 0;
	}
	int len = src.csize; int r = src.rsize;
	vect *trg; trg = new vect(len);
	double sum;
	double* this_ptr = this->v;
	double* src_ptr = src.a;
	double* trg_ptr = trg->v;
	for (int i = 0; i < len; i++) {
		sum = 0.0;
		for (int j = 0; j < r; j++) { 
			sum += this_ptr[j] * src_ptr[j * len + i];
		}
		trg_ptr[i] = sum;
	}
	return trg;
}
vect* vect::operator *(mat* src) {
	if (this->size != src->rsize) {
		cout << "ERROR(vect.*): invalid product" << endl;
		return 0;
	}
	int len = src->csize; int r = src->rsize;
	vect *trg; trg = new vect(len);
	double sum;
	double* this_ptr = this->v;
	double* src_ptr = src->a;
	double* trg_ptr = trg->v;
	for (int i = 0; i < len; i++) {
		sum = 0.0;
		for (int j = 0; j < r; j++) {
			sum += this_ptr[j] * src_ptr[j * len + i];
		}
		trg_ptr[i] = sum;
	}
	
	delete src;
	return trg;
}				//destructive one
vect& vect::operator =(const vect &src) {
	if (src.size < 0) {
		cout << "ERROR(vect.=): invalid size of vect" << endl;
		return *this;
	}else if (src.v == NULL) {
		cout << "ERROR(vect.=): null vect" << endl;
		return *this;
	}
	this->copy(src);
	return *this;
}
vect* vect::operator =(vect* src) {
	if (src->size <= 0) {
		cout << "ERROR(vect.=): invalid size of vect" << endl;
		return this;
	}else if (src->v == NULL) {
		cout << "ERROR(vect.=): null vect" << endl;
		return this;
	}
	this->copy(*src);
	delete src;
	return this;
}			//destructive one


mat::mat() { rsize = -1; csize = -1; a = NULL; };
mat::mat(int r, int c) { rsize = r; csize = c; a = dcalloc(rsize * csize); }
mat::mat(double* src, int r, int c){
	rsize = r; csize = c; a = dcalloc(rsize * csize); 
	double* ptr = a;
	for(int i = 0, n = rsize * csize; i < n; i++){
		ptr[i] = src[i];
	}
}
mat::mat(vector<vector<double>> src) {
	rsize = src.size();
	csize = src[0].size();
	a = dcalloc(rsize * csize);
	double* this_ptr = a;
	for (int i = 0, nr = rsize; i < nr; i++) {
		for (int j = 0, nc = csize; j < nc; j++) {
			this_ptr[i * nc + j] = src[i][j];
		}
	}
}
mat::~mat() {
	if (a != NULL) free(a);
	a = NULL;
	rsize = -1;
	csize = -1;
}
void mat::rand(double mean, double sigma){
	random_device rnd;
	mt19937 mt(rnd());
//	uniform_real_distribution<> unif(0.0, 1.0);   // [0.0,1.0]上一様に分布させる
	normal_distribution<> gauss(0.0, 1.0);   // 平均0.0、標準偏差1.0で分布させる
	for(int i=0; i<rsize; i++){
		for(int j=0; j<csize; j++){
			a[i * csize + j] = mean + sigma * gauss(mt);
		}
	}
}
void mat::zeros(int r, int c) {
	if (r > 0 && c > 0) {
		rsize = r; csize = c;
	}
	if(a != NULL) free(a);
	a = dcalloc(rsize * csize);
}
void mat::ones(int r, int c) {
	zeros(r, c);
	double* ptr = a;
	for (int i = 0, n = rsize*csize; i < n; i++) {
		ptr[i] = 1.0;
	}
}
void mat::identity(int n) {
	zeros(n, n);
	double* ptr = a;
	for (int i = 0; i < n; i++) {
		ptr[i * n + i] = 1.0;
	}
}
void mat::scale(double r) {
	double* ptr = a;
	for (int i = 0, n = rsize*csize; i < n; i++) {
		ptr[i] = ptr[i] * r;
	}
}
void mat::eigendecomp(vect *lambda, mat *U){
	//For given A, give the decomp as "A = U^T diag(l) U".
	if(rsize != csize){
		cout << "ERROR(mat.eigendecomp): is not square mat" << endl;
		return;
	}else if(rsize < 0){
		cout << "ERROR(mat.eigendecomp): invalid size of mat" << endl;
		return;
	}
	
	int n = rsize;
	int cnt = 0;
	int max_itr = 1000;
	double eps = 1e-6;
	double max_entry = 0.0;
	mat A; A.copy(*this);
	mat* Q = new mat();
	mat* R = new mat();
	
	U->identity(n);
	while(cnt < max_itr){
		QR_decomp(A, Q, R);		//A = QR
		A = (*R) * (*Q);		//A <- RQ = Q^T A Q
//		A.print();
		*U = (*U) * (*Q);
		max_entry = 0.0;
		for(int i = 0; i < n; i++){
			for(int j = 0; j < i; j++){
				max_entry = max(max_entry, abs(A(i,j)));
			}
		}
//		cout << max_entry << ", " << cnt << endl;
		if (max_entry < eps) break;
		cnt++;
	}
	
	lambda->zeros(n);
	for(int i = 0; i < n; i++) lambda->v[i] = A(i,i);
	
	delete Q; delete R;
}
mat* mat::sqrt(){
	if(rsize != csize){
		cout << "ERROR(mat.sqrt): is not square mat" << endl;
		return NULL;
	}else if(rsize < 0){
		cout << "ERROR(mat.sqrt): invalid size of mat" << endl;
		return NULL;
	}
	int n = rsize; double* ptr = a;
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			if (ptr[i * n + j] != ptr[j * n + i]){
				cout << "ERROR(mat.sqrt): is not symmetric" << endl;
				return NULL;
			}
		}
	}
	
	vect* lambda = new vect();
	mat* U = new mat();
	eigendecomp(lambda, U);
	double* l_ptr = lambda->v;
	for(int i = 0; i < n; i++){
		if(l_ptr[i] < 0){
			cout << "ERROR(mat.sqrt): is not semi-positive" << endl;
			return NULL;
		}
		l_ptr[i] = ::sqrt(lambda->v[i]);
	}
	mat* D = lambda->diagonal();
	mat* trg = new mat();
	trg = (*U) * ((*D) * U->transpose());
	
	delete lambda; delete U; delete D;
	return trg;
}
void mat::print(string str, int bl, double err) {
	double* ptr = a;
	if(str == ""){
		cout << "Matrix " << rsize << " \u2715 " << csize << endl;
	}else{
		cout << str << ": Matrix " << rsize << " \u2715 " << csize << endl;
	}
	for (int i = 0, nr = rsize; i < nr; i++) {
		cout << "   ";
		for (int j = 0, nc = csize; j < nc; j++) {
			if(abs(ptr[i * csize + j]) < err){
				cout << setw(bl + 4) << right << showpoint << setprecision(bl) << 0.0 << " ";
			}else{
				cout << setw(bl + 4) << right << showpoint << setprecision(bl) << ptr[i * csize + j] << " ";
			}
		}
		cout << endl;
	}
	cout << endl;
}
void mat::printc(fstream& file, string chr){
	double* ptr = a;
	for(int i=0; i<rsize; i++){
		for(int j=0; j<csize; j++){
			file << setprecision(15) << ptr[i * csize + j] << chr;
		}
	}
}
void mat::copy(const mat &src) {
	double* this_ptr;
	double* src_ptr = src.a;
	if (src.rsize <= 0 || src.csize <= 0) {
		cout << "ERROR(mat.copy): invalid size of mat" << endl;
	}
	else {
		zeros(src.rsize, src.csize);
		this_ptr = this->a;
		for (int i = 0, n = rsize*csize; i < n; i++) {
			this_ptr[i] = src_ptr[i];
		}
	}
}
mat* mat::transpose() {
	int r = this->rsize; int c = this->csize;
	mat* trg = new mat(c, r);
	double* this_ptr = this->a;
	double* trg_ptr = trg->a;
	for (int i = 0; i < c; i++) {
		for (int j = 0; j < r; j++) {
			trg_ptr[i * r + j] = this_ptr[j * c + i];
		}
	}
	return trg;
}
void mat::act(Function func) {
	double* ptr = a;
	for (int i = 0, n = rsize*csize; i < n; i++) {
		ptr[i] = func(ptr[i]);
	}
}
double mat::v(int i, int j){
	if(i >= rsize || i < 0){
		cout << "ERROR(mat.v): invalid row index" << endl;
		return 0.0;
	}else if(j >= csize || j < 0){
		cout << "ERROR(mat.v): invalid column index" << endl;
		return 0.0;
	}
	return a[i*csize+j];
}
vect* mat::col(int c) {
	vect *trg; trg = new vect(rsize);
	if (c >= csize || c < 0) {
		cout << "ERROR(mat.col): invalid column index" << endl;
		return trg;
	}
	int n = csize; int m = rsize;
	double* this_ptr = this->a;
	double* trg_ptr = trg->v;
	for (int i = 0; i < m; i++) {
		trg_ptr[i] = this_ptr[i * n + c];
	}
	return trg;
}
vect* mat::row(int r){
	vect *trg; trg = new vect(csize);
	if (r >= rsize || r < 0) {
		cout << "ERROR(mat.row): invalid row index" << endl;
		return trg;
	}
	int n = csize;
	double* this_ptr = this->a;
	double* trg_ptr = trg->v;
	for (int j = 0; j < n; j++) {
		trg_ptr[j] = this_ptr[r * n + j];
	}
	return trg;
}

double mat::operator ()(int i, int j){
	if(i >= rsize || i < 0){
		cout << "ERROR(mat.()): invalid row index" << endl;
		return 0.0;
	}else if(j >= csize || j < 0){
		cout << "ERROR(mat.()): invalid column index" << endl;
		return 0.0;
	}
	return a[i*csize+j];
}
mat* mat::operator *(const mat &src) {
	if (this->csize != src.rsize) {
		cout << "ERROR(mat.*): invalid product" << endl;
		return 0;
	}
	int r = rsize; int c = src.csize;
	mat *trg; trg = new mat(r, c);
	int col = csize;
	double* this_ptr = this->a;
	double* src_ptr = src.a;
	double* trg_ptr = trg->a;
	double sum;
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			sum = 0.0;
			for (int k = 0; k < col; k++) {
				sum += this_ptr[i * col + k] * src_ptr[k * c + j];
			}
			trg_ptr[i * c + j] = sum;
		}
	}
	return trg;
}
mat* mat::operator *(mat* src) {
	if (this->csize != src->rsize) {
		cout << "ERROR(mat.*): invalid product" << endl;
		return 0;
	}
	int r = rsize; int c = src->csize;
	mat *trg; trg = new mat(r, c);
	int col = csize;
	double* this_ptr = this->a;
	double* src_ptr = src->a;
	double* trg_ptr = trg->a;
	double sum;
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			sum = 0.0;
			for (int k = 0; k < col; k++) {
				sum += this_ptr[i * col + k] * src_ptr[k * c + j];
			}
			trg_ptr[i * c + j] = sum;
		}
	}
	delete src;
	return trg;
}				//destructive one
mat* mat::operator +(const mat &src) {
	if (this->rsize != src.rsize || this->csize != src.csize) {
		cout << "ERROR(mat.+): invalid sum" << endl;
		return 0;
	}
	mat *trg; trg = new mat(rsize, csize);
	double* this_ptr = this->a;
	double* src_ptr = src.a;
	double* trg_ptr = trg->a;
	for (int i = 0, n = rsize*csize; i < n; i++) {
		trg_ptr[i] = this_ptr[i] + src_ptr[i];
	}
	return trg;
}
mat* mat::operator +(mat* src) {
	if (this->rsize != src->rsize || this->csize != src->csize) {
		cout << "ERROR(mat.+): invalid sum" << endl;
		return 0;
	}
	mat *trg; trg = new mat(rsize, csize);
	double* this_ptr = this->a;
	double* src_ptr = src->a;
	double* trg_ptr = trg->a;
	for (int i = 0, n = rsize*csize; i < n; i++) {
		trg_ptr[i] = this_ptr[i] + src_ptr[i];
	}
	delete src;
	return trg;
}				//destructive one
mat* mat::operator -(const mat &src) {
	if (this->rsize != src.rsize || this->csize != src.csize) {
		cout << "ERROR(mat.-): invalid subtract" << endl;
		return 0;
	}
	mat *trg; trg = new mat(rsize, csize);
	double* this_ptr = this->a;
	double* src_ptr = src.a;
	double* trg_ptr = trg->a;
	for (int i = 0, n = rsize*csize; i < n; i++) {
		trg_ptr[i] = this_ptr[i] - src_ptr[i];
	}
	return trg;
}
mat* mat::operator -(mat* src) {
	if (this->rsize != src->rsize || this->csize != src->csize) {
		cout << "ERROR(mat.-): invalid subtract" << endl;
		return 0;
	}
	mat *trg; trg = new mat(rsize, csize);
	double* this_ptr = this->a;
	double* src_ptr = src->a;
	double* trg_ptr = trg->a;
	for (int i = 0, n = rsize*csize; i < n; i++) {
		trg_ptr[i] = this_ptr[i] - src_ptr[i];
	}
	delete src;
	return trg;
}				//destructive one
mat* mat::HProd(const mat &src) {
	if (this->rsize != src.rsize || this->csize != src.csize) {
		cout << "ERROR(mat.HProd): invalid Hadamard product" << endl;
		return 0;
	}
	mat *trg; trg = new mat(rsize, csize);
	double* this_ptr = this->a;
	double* src_ptr = src.a;
	double* trg_ptr = trg->a;
	for (int i = 0, n = rsize*csize; i < n; i++) {
		trg_ptr[i] = this_ptr[i] * src_ptr[i];
	}
	return trg;
}
mat* mat::HProd(mat* src) {
	if (this->rsize != src->rsize || this->csize != src->csize) {
		cout << "ERROR(mat.HProd): invalid Hadamard product" << endl;
		return 0;
	}
	mat *trg; trg = new mat(rsize, csize);
	double* this_ptr = this->a;
	double* src_ptr = src->a;
	double* trg_ptr = trg->a;
	for (int i = 0, n = rsize*csize; i < n; i++) {
		trg_ptr[i] = this_ptr[i] * src_ptr[i];
	}
	delete src;
	return trg;
}				//destructive one
vect* mat::operator *(const vect &src) {
	if (this->csize != src.size) {
		cout << "ERROR(mat.*): invalid product" << endl;
		return 0;
	}
	int r = rsize; int c = csize;
	vect *trg; trg = new vect(r);
	double* this_ptr = this->a;
	double* src_ptr = src.v;
	double* trg_ptr = trg->v;
	double sum;
	for (int i = 0; i < r; i++) {
		sum = 0.0;
		for (int j = 0; j < c; j++) {
			sum += this_ptr[i * c + j] * src_ptr[j];
		}
		trg_ptr[i] = sum;
	}
	return trg;
}
vect* mat::operator *(vect* src) {
	if (this->csize != src->size) {
		cout << "ERROR(mat.*): invalid product" << endl;
		return 0;
	}
	int r = rsize; int c = csize;
	vect *trg; trg = new vect(r);
	double* this_ptr = this->a;
	double* src_ptr = src->v;
	double* trg_ptr = trg->v;
	double sum;
	for (int i = 0; i < r; i++) {
		sum = 0.0;
		for (int j = 0; j < c; j++) {
			sum += this_ptr[i * c + j] * src_ptr[j];
		}
		trg_ptr[i] = sum;
	}
	delete src;
	return trg;
}				//destructive one
mat& mat::operator =(mat &src) {
	if (src.rsize <= 0 || src.csize <= 0) {
		cout << "ERROR(mat.=): invalid size of mat" << endl;
		return *this;
	}else if (src.a == NULL) {
		cout << "ERROR(mat.=): null mat" << endl;
		return *this;
	}
	this->copy(src);
	return *this;
}
mat* mat::operator =(mat* src) {
	if (src->rsize <= 0 || src->csize <= 0) {
		cout << "ERROR(mat.=): invalid size of mat" << endl;
		return this;
	}else if (src->a == NULL) {
		cout << "ERROR(mat.=): null mat" << endl;
		return this;
	}
	this->copy(*src);
	delete src;
	return this;
}				//destructive one

void QR_decomp(mat& src, mat* Q, mat* R){
	/*--------------------------------------*/
	/*    Gram-Schmidt orthogonalization    */
	/*--------------------------------------*/
	//For given A, give a decomp as "A = QR" by unitary Q and up-tri R.
	double err = 1e-10;
	int n = src.rsize;
	vect** v; v = (vect**)calloc(n, sizeof(vect*));
	for(int i = 0; i < n; i++){ v[i] = src.col(i); }
	vect** Qv; Qv = (vect**)calloc(n, sizeof(vect*));
	for(int i = 0; i < n; i++) Qv[i] = new vect(n);
	vect** Rv; Rv = (vect**)calloc(n, sizeof(vect*));
	for(int i = 0; i < n; i++) Rv[i] = new vect(n);
	
	//define first vect of basis
	Rv[0]->v[0] = v[0]->norm();
	if(Rv[0]->v[0] < err){
		cout << "ERROR(mat.eigendecomp): unexpected mat" << endl;
		return;
	}
	v[0]->scale(1.0 / Rv[0]->v[0]);
	Qv[0]->copy(*v[0]);
	for(int i = 1; i < n; i++) Rv[0]->v[i] = 0.0;
	
	//sequential calculation
	for(int k = 1; k < n; k++){
		for(int i = 0; i < k; i++){
			Rv[k]->v[i] = (*v[k]) * (*Qv[i]);
			(*v[k]) = (*v[k]) - ((*Qv[i]) * Rv[k]->v[i]);
		}
		for(int i = k; i < n; i++){
			Rv[k]->v[i] = 0.0;
		}
		Rv[k]->v[k] = v[k]->norm();
		if(Rv[k]->v[k] < err){
			cout << "ERROR(mat.eigendecomp): unexpected mat" << endl;
			return;
		}
		v[k]->scale(1.0 / Rv[k]->v[k]);
		Qv[k]->copy(*v[k]);
	}
	
	//output
	Q->zeros(n, n); R->zeros(n, n);
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			Q->a[i * n + j] = Qv[j]->v[i];
			R->a[i * n + j] = Rv[j]->v[i];
		}
	}
	
	//delete and free
	for(int i = 0; i < n; i++){
		delete v[i]; delete Qv[i]; delete Rv[i];
	}
	free(v); free(Qv); free(Rv);
}


