#include "testFitness.h"
#include "cudaErrorChk.h"
#include <cuda_profiler_api.h>
//undefine to remove error checking
//#define CUDA_ERROR_CHECK

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

extern "C" int32_t nCells;
extern "C" int32_t nVars;
//using namespace std;

// from https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#C
#define nSteps 10
#define NENV 4
#define NESPECIES 1000
#define NCELLS 50
#define CONST_WARPS 10 //Numero arbitrario do multiplo das warps 


#define erfA 0.278393      // Abramowitz e Stegun approximation to erf
#define erfB 0.230389      // https://en.wikipedia.org/wiki/Error_function
#define erfC 0.000972
#define erfD 0.078108
#define pi M_PI

typedef struct { 
  float x, y; 
} vec_t, *vec;

typedef struct { 
  int len, alloc; vec v; 
}poly_t, *poly;

// Dynamically resize the length of polygon, by calling realloc function
__device__ void poly_append(poly p, vec v){
	p->v[p->len++] = *v;
}

__device__ float crossMy(vec a, vec b){
	return a->x * b->y - a->y * b->x;
}

__device__ vec vsub(vec a, vec b, vec res){
	res->x = a->x - b->x; 
	res->y = a->y - b->y;
	return res;
}

// tells if vec c lies on the left side of directed edge a->b, 1 if left, -1 if right, 0 if colinear
__device__ int left_of(vec a, vec b, vec c){
	vec_t tmp1, tmp2;
	float x;
	vsub(b, a, &tmp1);
	vsub(c, b, &tmp2);
	x = crossMy(&tmp1, &tmp2);
	return x < 0 ? -1 : x > 0;
}

__device__ int line_sect(vec x0, vec x1, vec y0, vec y1, vec res){
	vec_t dx, dy, d;
	vsub(x1, x0, &dx);
	vsub(y1, y0, &dy);
	vsub(x0, y0, &d);
	/* x0 + a dx = y0 + b dy ->
	   x0 X dx = y0 X dx + b dy X dx ->
	   b = (x0 - y0) X dx / (dy X dx) */
	float dyx = crossMy(&dy, &dx);
	if (!dyx) return 0;
	dyx = crossMy(&d, &dx) / dyx;
	if (dyx <= 0 || dyx >= 1) return 0;

	res->x = y0->x + dyx * dy.x;
	res->y = y0->y + dyx * dy.y;
	return 1;
}

/* this works only if all of the following are true:
 *   1. poly has no colinear edges;
 *   2. poly has no duplicate vertices;
 *   3. poly has at least three vertices;
 *   4. poly is convex (implying 3).
*/

__device__ int poly_winding(poly p){
	return left_of(p->v, p->v + 1, p->v + 2);
}

__device__ void poly_edge_clip(poly sub, vec x0, vec x1, int left, poly res){
	int i, side0, side1;
	vec_t tmp;
	vec v0 = sub->v + sub->len - 1, v1;
	res->len = 0;

	side0 = left_of(x0, x1, v0);
	if (side0 != -left) poly_append(res, v0);

	for (i = 0; i < sub->len; i++) {
		v1 = sub->v + i;
		side1 = left_of(x0, x1, v1);
		if (side0 + side1 == 0 && side0)
			/* last point and current straddle the edge */
			if (line_sect(x0, x1, v0, v1, &tmp))
				poly_append(res, &tmp);
		if (i == sub->len - 1) break;
		if (side1 != -left) poly_append(res, v1);
		v0 = v1;
		side0 = side1;
	}
}

__device__ poly poly_clip(poly sub, poly clip, poly p1, poly p2){
	int i;
	//poly p1 = poly_new(), p2 = poly_new();
	poly tmp;

	p1->len = 0;
	p2->len = 0;

	int dir = poly_winding(clip);
	poly_edge_clip(sub, clip->v + clip->len - 1, clip->v, dir, p2);
	for (i = 0; i < clip->len - 1; i++) {
		tmp = p2; p2 = p1; p1 = tmp;
		if(p1->len == 0) {
			p2->len = 0;
			break;
		}
		poly_edge_clip(p1, clip->v + i, clip->v + i + 1, dir, p2);
	}

	return p2;
}

__device__ float area(poly resPoly){
    float area = 0.0;
    float d = 0.0;
    int j, i;

    if (resPoly->len > 2) {
        j = resPoly->len - 1;
        for (i = 0; (i < resPoly->len); i++) {
            d = (resPoly->v[j].x + resPoly->v[i].x);
            area = area + d * (resPoly->v[j].y - resPoly->v[i].y);
            j = i;
        }
        area = -area * 0.5;
    }
    //Sem print em paralelo 
    //printf("AREA- %.4f \n",area );
    return area;
}

__device__ void NicheCurve(float MinTol, float MidTol, float MaxTol, poly NichePoly, float *MaxFit){
  	// Begin of variable declarations
	  
  	/*
  	float erfA = 0.278393;      // Abramowitz e Stegun approximation to erf
  	float erfB = 0.230389;      // https://en.wikipedia.org/wiki/Error_function
  	float erfC = 0.000972;
  	float erfD = 0.078108;
  	float pi =   3.141592;
  	*/
	
  	float erfX = 0.0f;
  	float erfY = 0.0f;
  	float PhiNum = 0.0f;
  	float PhiDen1 = 0.0f;
  	float PhiDen2 = 0.0f;
	
  	// Read input data
  	float mi =    MidTol;
  	float sigma = (MaxTol - mi) / 2.0;
  	float a =     MinTol;
  	float b =     MaxTol;
	
  	float x = MaxTol;
  	float p;
  	float Tmp = 0.0f;
	
  	float Step = 0.0f;
	
  	int i;
	
   	// Begin of procedure
    	// resPol must be nSteps+3 long
  	Step = ((b-a) / nSteps);
	
  	NichePoly->v[0].x = x;
  	NichePoly->v[0].y = 0.0f;
	
  	*MaxFit = -1.0;
	
  	for(i = 0; i <= nSteps; i++){
    	// https://en.wikipedia.org/wiki/Truncated_normal_distribution
    	Tmp = (x - mi) / sigma;
    	PhiNum = (1.0f/sqrt(2.0f*pi))*exp((-1.0f/2.0f)*(Tmp*Tmp));
	
    	// Error function of (x1)
    	erfX = ((b-mi) / sigma) / sqrt(2.0f);
    	Tmp = fabs(erfX);
    	erfY = 1.0f-(1.0f/powf(1.0f+(erfA*Tmp)+(erfB*(Tmp*Tmp))+(erfC*powf(Tmp,3.0f))+(erfD*powf(Tmp,4.0f)),4.0f));
	        
    	if(erfX < 0.0f){
      		erfY = -1.0f * erfY;
    	}
	
    	PhiDen1 = (1.0f+erfY) / 2.0f;
	
    	// Error function of (x2)
    	erfX = ((a-mi) / sigma) / sqrt(2.0f);
    	Tmp = fabs(erfX);
    	erfY = 1.0f-(1.0f/powf(1.0f+(erfA*Tmp)+(erfB*(Tmp*Tmp))+(erfC*powf(Tmp,3.0f))+(erfD*powf(Tmp,4.0f)),4.0f));
	        
    	if(erfX < 0.0f){
      		erfY = -1.0f * erfY;
    	}
	
    	PhiDen2 = (1.0f+erfY) / 2.0f;
	
    	p = (PhiNum / (sigma * (PhiDen1 - PhiDen2)));
	
    	NichePoly->v[i+1].x = x;
    	NichePoly->v[i+1].y = p;
	
    	x = x - Step;
	
    	if(p > *MaxFit) 
    		*MaxFit = p;
	
  	}
	// ATENÇÃO, PODE DAR ERRO. Duplicando vértices
  	NichePoly->v[nSteps+2].x = NichePoly->v[nSteps+1].x;
  	NichePoly->v[nSteps+2].y = 0.0f;

}

__device__ void EnvSpace(float MinEnv, float MaxEnv, poly EnvPoly, float *MaxFit){
  	EnvPoly->v[0].x = MinEnv;    EnvPoly->v[0].y = 0.0f;
  	EnvPoly->v[1].x = MinEnv;    EnvPoly->v[1].y = *MaxFit;
  	EnvPoly->v[2].x = MaxEnv;    EnvPoly->v[2].y = *MaxFit;
  	EnvPoly->v[3].x = MaxEnv;    EnvPoly->v[3].y = 0.0f;
}

__global__ void CalcFitness(float * SpNiche,float * LocEnv,float * Fitness){
  	// Declare auxiliary private data

  	//uint t = get_global_id(0);
    unsigned int espIndex =(blockDim.x * blockIdx.x) + threadIdx.x;	//indice da especie
  	unsigned int cellIdx;
  	//sai da função se a thread não corresponde a nenhuma espécie (indice da thread maior que quantidade de especies)
    if(espIndex >= NESPECIES) return;
    //unsigned int t =0;

  	poly ClippedTol;
  	float StdAreaNoOverlap, StdSimBetweenCenters;
  	float MidTol;
  	float MinTempTol, MaxTempTol, MinPrecpTol, MaxPrecpTol;
  	float MinEnv, MidEnv, MaxEnv;
  	float LocFitness;
	

  	// Declare private data
  	vec_t NichePtns[nSteps+3];	//pontos do poligono do nicho ( da especie ) ( struct com float x e y)
  	poly_t NichePoly = {nSteps+3, nSteps+3, NichePtns};
	
  	vec_t EnvPtns[NENV];	//pontos do poligono do ambiente ( struct com float x e y)
  	poly_t EnvPoly = {NENV, NENV, EnvPtns};
	
	//p1 e p2 são poligonos auxiliares, um guarda o resultado e o outro guarda uma informação que é usada varias vezes
	//info acima é a mais provavel, mas n é certeza ( de acordo com thiago) - analisar mais tarde
  	vec_t p1Ptns[nSteps+10];
  	poly_t p1Poly = {0, nSteps+10, p1Ptns};
	
  	vec_t p2Ptns[nSteps+3];
  	poly_t p2Poly = {0, nSteps+10, p2Ptns};
	
  	float MaxFit;
	

  	      
	for(cellIdx=0;cellIdx < NCELLS ; cellIdx++){      

    // dados da especie especifica da thread
    //armazenados assim pois são acessados varias vezes porém nunca alterados.
    MinTempTol = SpNiche[espIndex*4 +0];
    MaxTempTol = SpNiche[espIndex*4 + 1];
    MinPrecpTol = SpNiche[espIndex*4 + 2];
    MaxPrecpTol = SpNiche[espIndex*4 + 3];
    


		MinEnv = LocEnv[(cellIdx*4) + 0];
 		MaxEnv = LocEnv[(cellIdx*4) + 1];
		
  	// Does the species tolerate the local environment?
  	if((MinEnv < MinTempTol) || (MaxEnv > MaxTempTol)){
   		LocFitness = 0;
  	}
  	// Yes, it tolerates, lets calculate the fitness
  	else {
   		MidTol = ((MaxTempTol - MinTempTol) / 2.0) + MinTempTol;
   		MidEnv = ((MaxEnv - MinEnv) / 2.0) + MinEnv;
	          
   		if((MaxTempTol - MinTempTol) < 1E-2) {
   			StdAreaNoOverlap = 1; 
   			MidEnv = 0;
  		}
  		else {
  			NicheCurve(MinTempTol, MidTol, MaxTempTol, &NichePoly, &MaxFit);
   			EnvSpace(MinEnv, MaxEnv, &EnvPoly, &MaxFit);     
   			ClippedTol = poly_clip(&NichePoly, &EnvPoly, &p1Poly, &p2Poly); 
   			StdAreaNoOverlap = 1 - area(ClippedTol);
  		}	    
    	StdSimBetweenCenters = 1 - (fabs(MidEnv - MidTol) / ((MaxTempTol - MinTempTol)/2));
    	// Local fitness, given the first environmental variable
    	LocFitness = (StdAreaNoOverlap * StdSimBetweenCenters);
		    
  		// Second environmental variable
   		MinEnv = LocEnv[(cellIdx*4) + 2];
   		MaxEnv = LocEnv[(cellIdx*4) + 3];
	
  		// Does the species tolerates the local environment?
   		if((LocFitness < 1E-4) || (MinEnv < MinPrecpTol) || (MaxEnv > MaxPrecpTol)) {
   			LocFitness = 0;
  		}
  		else {
  			MidTol = ((MaxPrecpTol - MinPrecpTol) / 2.0) + MinPrecpTol;
  			MidEnv = ((MaxEnv - MinEnv) / 2.0) + MinEnv;		                                
      	
        if((MaxPrecpTol - MinPrecpTol) < 1E-2) {
        	StdAreaNoOverlap = 1;
      		MidEnv = 0;
      	}
    		else {
      		NicheCurve(MinPrecpTol, MidTol, MaxPrecpTol, &NichePoly, &MaxFit);
       		EnvSpace(MinEnv, MaxEnv, &EnvPoly, &MaxFit);
      		ClippedTol = poly_clip(&NichePoly, &EnvPoly, &p1Poly, &p2Poly);
       		StdAreaNoOverlap = 1 - area(ClippedTol);
    		}	      
     	
      	StdSimBetweenCenters = 1 - (fabs(MidEnv - MidTol) / ((MaxPrecpTol - MinPrecpTol)/2));
		      
      	// Local fitness, given both environmental variables
      	LocFitness = LocFitness * (StdAreaNoOverlap * StdSimBetweenCenters);
      }
		}
	  
  		// Return fitness value
  		Fitness[ (espIndex*NCELLS) + cellIdx ] = LocFitness;
  		//printf("LocFit-%.8f  CELL- %d\t Especie- %d\n",LocFitness,cellIdx,espIndex );
  }        
}

extern "C" void calc_fitness(float *SpNiche,float *LocEnv,float *Fitness){
	float *d_spNiche,*d_locEnv,*d_fitness;
	int qnt_thr,qnt_blocos;
	 
	printf("\t\t*IN C CODE*\n");

	//cudaProfilerStart();
	CudaSafeCall( cudaMalloc(&d_spNiche,sizeof(float)*nVars*NESPECIES));
	CudaSafeCall( cudaMalloc(&d_locEnv,sizeof(float)*nVars*NCELLS));
	CudaSafeCall( cudaMalloc(&d_fitness,sizeof(float)*NESPECIES*NCELLS));

	printf("valor de \t   nCells = %i\tNESPECIES = %i\tnVars = %i\n",NCELLS,NESPECIES,nVars);
	
	for(int i=0;i<NCELLS;i++){
		LocEnv[i*4+0] = 25.4079513549805;
    	LocEnv[i*4+1] = 27.0897407531738;
    	LocEnv[i*4+2] = 172.994903564453;
    	LocEnv[i*4+3] = 883.234375;
	}
	for(int i=0;i<NESPECIES;i++){
		SpNiche[i*4+0] = 15;
    SpNiche[i*4+1] = 40;
    SpNiche[i*4+2] = 150;
    SpNiche[i*4+3] = 2000;
	}
	
	//cudaDeviceSynchronize();

	CudaSafeCall( cudaMemcpy(d_spNiche, SpNiche, sizeof(float)*nVars*NESPECIES, cudaMemcpyHostToDevice) );
	CudaSafeCall( cudaMemcpy(d_locEnv, LocEnv, sizeof(float)*nVars*NCELLS, cudaMemcpyHostToDevice) );
	//cudaDeviceSynchronize();

  //Selecionando a GPU e pegando suas especificaçoes
  cudaDeviceProp device_prop;
  int device =0;  //Qual GPU será usada
  cudaSetDevice(device);
  cudaGetDeviceProperties(&device_prop,device);


  qnt_thr = device_prop.warpSize * CONST_WARPS;
	qnt_blocos = (NESPECIES+qnt_thr-1)/qnt_thr;
	
  printf("\tCalling device: CalcFitness\n");
	CalcFitness<<<qnt_blocos,qnt_thr>>>(d_spNiche, d_locEnv, d_fitness);
	CudaCheckError();
  //cudaDeviceSynchronize();
	printf("\tLeaving device: CalcFitness\n");

	CudaSafeCall( cudaMemcpy(Fitness,d_fitness,sizeof(float)*NESPECIES*nCells,cudaMemcpyDeviceToHost) );


	cudaFree(d_fitness);
	cudaFree(d_locEnv);
	cudaFree(d_spNiche);
	cudaDeviceSynchronize();

	//cudaProfilerStop();
	printf("\n\t\t*LEAVING C CODE*\n");	
}